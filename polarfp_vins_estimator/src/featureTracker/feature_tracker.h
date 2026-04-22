/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

/**
 * @file feature_tracker.h
 * @brief 特征点跟踪器头文件
 *
 * 该模块负责从图像序列中提取和跟踪视觉特征点。
 * 主要功能包括：
 * - 使用光流法（Lucas-Kanade）进行帧间特征点跟踪
 * - 通过 mask 保证特征点在图像中均匀分布
 * - 正反双向光流一致性检查以剔除误匹配
 * - 计算特征点在归一化平面上的速度
 * - 支持双目相机的特征点跟踪
 * - 利用预测位置加速光流匹配
 */

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"
#include "PolarChannel.h"
#include "feature_tracker_detector.h"
#include "feature_tracker_matcher.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

/** @brief 根据status状态压缩Point2f向量，仅保留有效点 */
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
/** @brief 根据status状态压缩int向量，仅保留有效元素 */
void reduceVector(vector<int> &v, vector<uchar> status);

/**
 * @brief 单个偏振通道的跟踪状态
 *
 * 复用 VINS-Fusion FeatureTracker 的数据组织方式(平行 vector),
 * 每个通道独立维护自己的跟踪状态。
 */
struct ChannelState {
    std::string name;                           ///< 通道名称 "s0"/"dop"/...
    int row = 0, col = 0;                       ///< 通道图像尺寸
    cv::Mat prev_img, cur_img;                  ///< 前后帧图像
    vector<cv::Point2f> prev_pts, cur_pts;      ///< 前后帧特征点坐标
    vector<cv::Point2f> prev_un_pts, cur_un_pts; ///< undistorted 特征点坐标
    vector<cv::Point2f> n_pts;                  ///< 新检测特征点
    vector<int> local_ids;                      ///< 【局部 ID】仅在当前通道内唯一，由 next_local_id 分配
    vector<int> track_cnt;                      ///< 跟踪帧数
    vector<cv::Point2f> pts_velocity;           ///< 像素速度
    map<int, cv::Point2f> cur_un_pts_map;       ///< local_id -> 归一化坐标
    map<int, cv::Point2f> prev_un_pts_map;      ///< 上一帧 local_id -> 归一化坐标
    map<int, cv::Point2f> prevLeftPtsMap;       ///< 上一帧 local_id -> 像素坐标
    double cur_time = 0;                        ///< 当前帧时间戳
    double prev_time = 0;                       ///< 上一帧时间戳
    cv::Mat mask;                               ///< 空间分布掩码

    // BRIEF/ORB 描述子(上一帧), 扁平存储: [feat0_desc, feat1_desc, ...]
    // 每个描述子 brief_bytes 字节。LK 模式下为空。
    std::vector<uchar> prev_brief_desc;
    int brief_bytes = 32;

    int next_local_id = 0;                      ///< 本通道局部 ID 计数器，新特征从此取值

    ChannelState() = default;
    explicit ChannelState(const string& n) : name(n) {}
};

/**
 * @brief 全局特征池：将多通道局部特征通过空间位置去重，映射为统一全局 ID
 *
 * 核心设计：
 * - 局部 ID（Local ID）：每个偏振通道独立计数，仅在通道内唯一。
 * - 全局 ID（Global ID）：由本池统一分配，后端只看到全局 ID。
 * - 同一帧中，一个全局 ID 最多向后端发送一个观测（按通道优先级选）。
 */
struct GlobalFeaturePool {
    struct GlobalFeature {
        int global_id;
        std::map<std::string, int> local_ids;          ///< channel name -> local_id
        std::map<std::string, cv::Point2f> pixel_pts;  ///< channel name -> pixel coordinate
    };

    int next_global_id = 0;

    std::map<int, std::map<std::string, int>> prev_bindings;    ///< global_id -> {ch_name -> local_id}
    std::map<int, std::map<std::string, cv::Point2f>> prev_pts; ///< global_id -> {ch_name -> pixel_pt}

    std::map<int, GlobalFeature> cur_globals;                   ///< current frame active global features
    std::map<std::pair<int, int>, std::vector<int>> grid;       ///< spatial hash (cell_x, cell_y) -> [global_id]

    int grid_size_ = 5;   ///< hash grid cell size in pixels, injected from YAML polar_hash_grid_size

    void beginFrame();
    void propagateTracked(const std::vector<ChannelState>& channels);
    void registerUnboundFeatures(std::vector<ChannelState>& channels, int min_dist);
    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>
    buildFeatureFrame(const std::vector<ChannelState>& channels) const;
    std::map<int, std::vector<std::pair<std::string, int>>> getGlobalToLocalMap() const;
    void endFrame();

private:
    std::pair<int, int> getGridCell(const cv::Point2f& pt) const;
    void insertToGrid(int global_id, const cv::Point2f& pt);
    std::vector<int> queryNearby(const cv::Point2f& pt) const;
};

/**
 * @class FeatureTracker
 * @brief 特征点跟踪器类，负责多通道偏振图像的特征提取、跟踪和管理
 *
 * 每个偏振通道独立运行检测器+匹配器管线，结果经GlobalFeaturePool
 * 合并为统一全局ID后输出给后端估计器。
 */
class FeatureTracker
{
public:
    /** @brief 构造函数 */
    FeatureTracker();

    /**
     * @brief 核心跟踪函数：对输入图像进行特征点跟踪
     * @param _cur_time 当前图像时间戳
     * @param _img 当前帧图像（偏振模式下为原始偏振图像）
     * @param _img1 当前帧右目图像（双目模式，未使用）
     * @return map<全局ID, vector<观测>>，观测包含相机ID和7维向量(归一化xyz, 像素uv, 速度vxvy)
     */
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());

    /** @brief 读取相机内参文件，支持单目（1个文件）和双目（2个文件） */
    void readIntrinsicParameter(const vector<string> &calib_file);

    /** @brief 将像素坐标通过相机模型投影到归一化平面（z=1） */
    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);

    /** @brief 移除后端判定为外点的特征（通过全局ID映射到各通道局部ID后剔除） */
    void removeOutliers(set<int> &removePtsIds);

    /** @brief 返回跟踪可视化图像，供ROS话题发布或调试查看 */
    cv::Mat getTrackImage();

    /**
     * @brief 设置启用的偏振通道
     * @param channel_names 通道名称列表，如 {"s0", "dop", "aopsin", "aopcos"}
     */
    void setPolarChannels(const vector<string>& channel_names);

    /** @brief 设置偏振通道滤波配置（滤波器类型及对应参数） */
    void setPolarFilterConfig(const PolarFilterConfig& cfg);

    /** @brief 根据全局配置创建并初始化检测器和匹配器 */
    void initDetectorAndMatcher();

    // ---- 成员变量 ----
    int row, col;                         ///< 图像行数和列数
    cv::Mat imTrack;                      ///< 用于可视化展示的跟踪结果图像
    cv::Mat mask;                         ///< 均匀化掩膜，确保特征点分布均匀
    cv::Mat fisheye_mask;                 ///< 鱼眼相机掩膜
    cv::Mat prev_img, cur_img;            ///< 上一帧和当前帧图像
    vector<cv::Point2f> n_pts;            ///< 新检测到的特征点
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts; ///< 上一帧、当前帧左目、当前帧右目特征点
    vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts; ///< 对应帧的归一化平面坐标
    vector<cv::Point2f> pts_velocity, right_pts_velocity; ///< 特征点像素速度
    vector<int> ids, ids_right;           ///< 特征点唯一ID（左目和右目）
    vector<int> track_cnt;                ///< 每个特征点被跟踪的帧数
    map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map; ///< 当前帧和上一帧 ID->归一化坐标 映射
    map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map; ///< 右目对应映射
    map<int, cv::Point2f> prevLeftPtsMap; ///< 上一帧左目特征点 ID->像素坐标 映射
    vector<camodocal::CameraPtr> m_camera;///< 相机模型指针列表
    double cur_time;                      ///< 当前帧时间戳
    double prev_time;                     ///< 上一帧时间戳
    bool stereo_cam;                      ///< 是否为双目相机
    int n_id;                             ///< 下一个新特征点的ID计数器

    // ---- 偏振模式特有成员 ----
    vector<ChannelState> channels;        ///< 启用的通道状态列表
    PolarFilterConfig polar_filter_cfg;   ///< 偏振通道滤波配置
    GlobalFeaturePool global_pool_;       ///< 全局特征池（V2.5 新增）

    void setPolarHashGridSize(int size) { global_pool_.grid_size_ = size; }

private:
    /** @brief 从PolarChannelResult中提取指定通道的8bit图像 */
    static cv::Mat getChannelImage(const PolarChannelResult& result, const string& channel);

    /** @brief 对单通道特征点施加mask，按跟踪帧数降序选取并保持空间均匀分布 */
    void setMaskForChannel(ChannelState& ch);

    /** @brief 边界检查：判断点是否在通道图像内部（留1像素边距） */
    static bool inBorderImpl(const ChannelState& ch, const cv::Point2f& pt);

    /** @brief 多通道跟踪可视化：单通道直接绘制，多通道按2x2网格拼接 */
    void drawTrackPolar();

    std::shared_ptr<FeatureDetector> detector_;
    std::shared_ptr<FeatureMatcher> matcher_;
};
