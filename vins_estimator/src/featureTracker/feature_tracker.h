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

/** @brief 判断特征点是否在图像边界内 */
bool inBorder(const cv::Point2f &pt);
/** @brief 根据 status 状态压缩 Point2f 向量，仅保留有效点 */
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
/** @brief 根据 status 状态压缩 int 向量，仅保留有效元素 */
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
    vector<int> ids;                            ///< 特征 ID(全局共享 n_id 分配)
    vector<int> track_cnt;                      ///< 跟踪帧数
    vector<cv::Point2f> pts_velocity;           ///< 像素速度
    map<int, cv::Point2f> cur_un_pts_map;       ///< ID → 归一化坐标
    map<int, cv::Point2f> prev_un_pts_map;      ///< 上一帧 ID → 归一化坐标
    map<int, cv::Point2f> prevLeftPtsMap;       ///< 上一帧 ID → 像素坐标
    double cur_time = 0;                        ///< 当前帧时间戳
    double prev_time = 0;                       ///< 上一帧时间戳
    cv::Mat mask;                               ///< 空间分布掩码

    // BRIEF/ORB 描述子(上一帧), 扁平存储: [feat0_desc, feat1_desc, ...]
    // 每个描述子 brief_bytes 字节。LK 模式下为空。
    std::vector<uchar> prev_brief_desc;
    int brief_bytes = 32;

    ChannelState() = default;
    explicit ChannelState(const string& n) : name(n) {}
};

/**
 * @class FeatureTracker
 * @brief 特征点跟踪器类
 *
 * 核心类，负责单目/双目相机的特征点提取、跟踪和管理。
 * 使用 Lucas-Kanade 光流法跟踪特征点，并通过 mask 机制
 * 确保特征点在图像中均匀分布，避免特征点过度聚集。
 */
class FeatureTracker
{
public:
    /** @brief 构造函数，初始化成员变量 */
    FeatureTracker();

    /**
     * @brief 对输入图像进行特征点跟踪
     * @param _cur_time 当前图像时间戳
     * @param _img 当前帧左目图像
     * @param _img1 当前帧右目图像（双目模式）
     * @return map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
     *         key: 特征点ID, value: 观测列表，每项包含相机ID和
     *         7维向量 (归一化坐标x,y,z, 像素坐标u,v, 速度vx,vy)
     */
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());

    /**
     * @brief 设置 mask 以筛选特征点
     *
     * 优先保留跟踪时间长的特征点，使用 mask 确保新特征点之间
     * 保持最小距离（MIN_DIST），使特征分布均匀。
     */
    void setMask();

    /** @brief 读取相机内参文件，支持多相机（双目） */
    void readIntrinsicParameter(const vector<string> &calib_file);

    /**
     * @brief 显示去畸变后的图像（调试用）
     * @param name 窗口名称
     */
    void showUndistortion(const string &name);

    /**
     * @brief 使用基础矩阵 F 的 RANSAC 方法剔除外点
     *
     * 将特征点反投影到归一化平面再进行 F 矩阵估计，
     * 比直接在像素坐标上计算更加准确。
     */
    void rejectWithF();

    /** @brief 对当前特征点进行去畸变（已废弃，使用 undistortedPts 代替） */
    void undistortedPoints();

    /**
     * @brief 将像素坐标通过相机模型转换到归一化平面
     * @param pts 待转换的像素坐标点集
     * @param cam 相机模型指针
     * @return 归一化平面坐标点集 (z=1 平面)
     */
    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);

    /**
     * @brief 计算特征点的像素速度
     * @param ids 特征点ID列表
     * @param pts 当前帧特征点像素坐标
     * @param cur_id_pts 输出：当前帧ID到坐标的映射
     * @param prev_id_pts 上一帧ID到坐标的映射
     * @return 每个特征点的像素速度 (vx, vy)
     */
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);

    /**
     * @brief 在两张图像上绘制特征点（调试用）
     * @param img1 图像1
     * @param img2 图像2
     * @param pts1 图像1上的特征点
     * @param pts2 图像2上的特征点
     */
    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2,
                      vector<cv::Point2f> pts1, vector<cv::Point2f> pts2);

    /**
     * @brief 绘制特征点跟踪结果
     *
     * 在图像上绘制特征点（颜色表示跟踪时长）、箭头（表示运动方向），
     * 双目模式下同时绘制右目特征点。
     *
     * @param imLeft 左目图像
     * @param imRight 右目图像
     * @param curLeftIds 当前左目特征点ID列表
     * @param curLeftPts 当前左目特征点坐标
     * @param curRightPts 当前右目特征点坐标
     * @param prevLeftPtsMap 上一帧左目特征点坐标映射（ID -> 坐标）
     */
    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                                   vector<int> &curLeftIds,
                                   vector<cv::Point2f> &curLeftPts,
                                   vector<cv::Point2f> &curRightPts,
                                   map<int, cv::Point2f> &prevLeftPtsMap);

    /**
     * @brief 设置特征点预测位置
     *
     * 利用后端优化或IMU预积分预测的3D位置，
     * 投影到图像平面作为光流初始值，加速匹配并提高成功率。
     *
     * @param predictPts 预测的3D点映射（ID -> 3D坐标）
     */
    void setPrediction(map<int, Eigen::Vector3d> &predictPts);

    /** @brief 计算两点间欧氏距离 */
    double distance(cv::Point2f &pt1, cv::Point2f &pt2);

    /**
     * @brief 移除被标记为外点的特征点
     * @param removePtsIds 需要移除的特征点ID集合
     */
    void removeOutliers(set<int> &removePtsIds);

    /** @brief 返回跟踪可视化图像 */
    cv::Mat getTrackImage();

    /** @brief 判断特征点是否在图像边界内 */
    bool inBorder(const cv::Point2f &pt);

    /**
     * @brief 设置启用的偏振通道
     * @param channel_names 通道名称列表,如 {"s0", "dop"}
     *
     * 在 estimator::setParameter() 中调用。
     * 调用后 trackImage 将输入视为原始偏振图像,先解码为多通道再分别处理。
     */
    void setPolarChannels(const vector<string>& channel_names);

    /** @brief 判断是否处于偏振模式 */
    bool isPolarMode() const { return polar_mode; }

    /**
     * @brief 设置偏振通道滤波配置
     * @param cfg 滤波参数（开关 + 双边滤波参数）
     */
    void setPolarFilterConfig(const PolarFilterConfig& cfg);

    /** @brief 根据配置初始化检测器和匹配器 */
    void initDetectorAndMatcher();

    // ---- 成员变量 ----
    int row, col;                         ///< 图像行数和列数
    cv::Mat imTrack;                      ///< 用于可视化展示的跟踪结果图像
    cv::Mat mask;                         ///< 均匀化掩膜，确保特征点分布均匀
    cv::Mat fisheye_mask;                 ///< 鱼眼相机掩膜
    cv::Mat prev_img, cur_img;            ///< 上一帧和当前帧图像
    vector<cv::Point2f> n_pts;            ///< 新检测到的特征点
    vector<cv::Point2f> predict_pts;      ///< 预测的特征点位置（用于加速光流）
    vector<cv::Point2f> predict_pts_debug;///< 预测点调试用
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
    bool hasPrediction;                   ///< 是否有预测位置（用于加速光流匹配）

    // ---- 偏振模式特有成员 ----
    vector<ChannelState> channels;        ///< 启用的通道状态列表
    bool polar_mode = false;              ///< 是否处于偏振模式
    PolarFilterConfig polar_filter_cfg;   ///< 偏振通道滤波配置

private:
    /** @brief 从 PolarChannelResult 提取指定通道的 8bit 图像 */
    static cv::Mat getChannelImage(const PolarChannelResult& result, const string& channel);

    /** @brief setMask 的单通道版本 */
    void setMaskForChannel(ChannelState& ch);

    /** @brief 边界检查的通道版本 */
    static bool inBorderImpl(const ChannelState& ch, const cv::Point2f& pt);

    /** @brief 多通道可视化 */
    void drawTrackPolar();

    std::shared_ptr<FeatureDetector> detector_;
    std::shared_ptr<FeatureMatcher> matcher_;
};
