#ifndef POLARFP_TRACKER_H
#define POLARFP_TRACKER_H

/**
 * @file polarfp_tracker.h
 * @brief 偏振多通道特征跟踪器 —— VINS-Fusion 前端 FeatureTracker 的替代品。
 *
 * 核心流程：
 *   1. 将偏振原始图像分解为 4 个通道（S0/DoP/sin(AoP)/cos(AoP)）
 *   2. 在每个通道上独立检测 FAST/GFTT 特征点
 *   3. LK 光流预测 + KD-tree 最近邻匹配
 *   4. 多通道联合 RANSAC 剔除异常值
 *   5. 基于评分的特征管理（保留高质量特征点）
 *
 * 与 VINS-Fusion 的 FeatureTracker 接口完全兼容：
 *   - trackImage() 返回 map<int, vector<pair<int, Matrix<double,7,1>>>>
 *   - readIntrinsicParameter() 读取相机标定文件
 */

#include <vector>
#include <map>
#include <set>
#include <string>
#include <memory>
#include <boost/shared_ptr.hpp>
#include <random>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using namespace std;
namespace camodocal { class Camera; using CameraPtr = boost::shared_ptr<Camera>; }

/**
 * @brief 偏振特征跟踪器的配置参数
 *
 * 包含特征检测方法、通道选择、LK 光流参数、NMS/RANSAC 阈值等。
 * 各通道使用不同的 FAST 阈值以适配其特性：
 *   - S0 阈值低（5），因为强度图特征丰富
 *   - DoP 阈值中（15），偏振度特征适中
 *   - AoP 阈值高（80），只保留强偏振角特征
 */
struct PolarConfig {
    /** 特征检测方法："fast" 或 "gftt" */
    std::string FP_METHOD = "fast";

    /** 所有可用通道列表 */
    std::vector<std::string> ALL_CHANNELS = {"s0", "dop", "aopsin", "aopcos"};

    /** 实际用于特征提取的通道（可从中 subset） */
    std::vector<std::string> FP_CHANNELS = {"s0", "dop"}; // , "dop",  "aopsin", "aopcos"

    /** 各通道可视化颜色（BGR 格式） */
    std::map<std::string, cv::Scalar> CHANNEL_COLORS = {
        {"s0", cv::Scalar(0, 255, 255)},    // yellow
        {"dop", cv::Scalar(0, 255, 0)},     // green
        {"aopsin", cv::Scalar(0, 0, 255)},  // red
        {"aopcos", cv::Scalar(255, 0, 0)}   // blue
    };

    /** 目标特征点总数（所有通道合计） */
    int KP_NUM_TARGET = 1000;

    /** FAST 特征检测器 —— 各通道使用不同阈值 */
    cv::Ptr<cv::FastFeatureDetector> FAST_S0_DT = cv::FastFeatureDetector::create(5);
    cv::Ptr<cv::FastFeatureDetector> FAST_DOP_DT = cv::FastFeatureDetector::create(15);
    cv::Ptr<cv::FastFeatureDetector> FAST_AOP_DT = cv::FastFeatureDetector::create(80);

    /** GFTT 特征检测器（备选方案） */
    cv::Ptr<cv::GFTTDetector> GFTT_DT = cv::GFTTDetector::create(
        500,     // maxCorners
        0.03,    // qualityLevel
        7,       // minDistance
        3,       // blockSize
        false    // harrisDetector
    );

    /** LK 光流窗口大小 */
    cv::Size LK_WIN_SIZE = cv::Size(21, 21);

    /** LK 光流金字塔最大层数 */
    int LK_MAX_LEVEL = 6;

    /** LK 光流终止条件 */
    cv::TermCriteria LK_CRITERIA = cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 0.01);

    /** LK 预测位置与 KD-tree 匹配的最大距离（像素） */
    double LK_MATCH_THRESHOLD = 10.0;

    /** NMS 抑制半径（像素） */
    int NMS_RADIUS = 21;

    /** RANSAC 重投影误差阈值（像素） */
    double RANSAC_REMAP_THR = 2.0;

    /** RANSAC 置信度 */
    double RANSAC_CONFIDENCE = 0.99;

    PolarConfig() = default;
};

/**
 * @brief 带通道信息的特征点
 *
 * 扩展了 OpenCV 的 KeyPoint，增加通道标识、唯一 ID、
 * 跟踪计数和评分，用于跨帧特征关联。
 */
struct PolarKeyPoint {
    std::string channel;  ///< 所属通道（"s0"/"dop"/"aopsin"/"aopcos"）
    int id;               ///< 特征点唯一 ID（跨帧保持一致）
    cv::KeyPoint kp;      ///< OpenCV 关键点（坐标、响应值等）
    int track_cnt;        ///< 连续跟踪帧数
    float score;          ///< 特征点评分（用于筛选高质量点）
};

/** 匹配的点对（前一帧 → 当前帧） */
using MatchedPair  = std::pair<PolarKeyPoint, PolarKeyPoint>;

/** 按通道分组的特征点 */
using ChannelKeyPoints = std::map<std::string, std::vector<PolarKeyPoint>>;

/** 按通道分组的匹配点对 */
using ChannelPairs = std::map<std::string, std::vector<MatchedPair>>;

/** 按通道命名的图像集合 */
using ChannelImages = std::map<std::string, cv::Mat>;


/**
 * @brief 偏振多通道特征跟踪器主类
 *
 * 作为 VINS-Fusion 前端 FeatureTracker 的直接替代品，
 * 处理偏振图像并完成特征检测 → 光流匹配 → 异常值剔除 → VINS 格式输出。
 */
class PolarFeatureTracker {
public:
    PolarFeatureTracker();
    ~PolarFeatureTracker();

    /**
     * @brief 主入口函数：处理一帧新图像，返回特征帧
     * @param _cur_time 当前帧时间戳
     * @param _img 当前帧图像（偏振原始图）
     * @param _img1 双目右图（暂未实现，忽略）
     * @return map<特征点ID, vector<相机索引, [x,y,z,u,v,vx,vy]>>
     */
    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(
        double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());

    /**
     * @brief 读取相机内参（从 YAML 标定文件）
     * @param calib_file 标定文件路径列表（单目=1个，双目=2个）
     */
    void readIntrinsicParameter(const std::vector<std::string> &calib_file);

    /**
     * @brief 设置外部预测点（可选，由后端提供）
     * @param predictPts 特征点 ID → 3D 预测位置的映射
     */
    void setPrediction(std::map<int, Eigen::Vector3d> &predictPts);

    /**
     * @brief 获取用于 ROS 发布的跟踪可视化图像
     * @return 田字格拼接的 4 通道跟踪图
     */
    cv::Mat getTrackImage();

    /**
     * @brief 移除被后端标记为异常的点
     * @param removePtsIds 需要移除的特征点 ID 集合
     */
    void removeOutliers(std::set<int> &removePtsIds);

    /** 更新配置参数 */
    void setConfig(const PolarConfig& new_config) { config = new_config; }

    /** 获取当前配置参数 */
    PolarConfig getConfig() const { return config; }

    /** 获取前一帧特征点列表（调试用） */
    const std::vector<PolarKeyPoint>& getPrevPolarPts() const { return prev_polar_pts; }

    /** 获取当前帧特征点列表（调试用） */
    const std::vector<PolarKeyPoint>& getCurPolarPts() const { return cur_polar_pts; }

    /** 向后兼容：提取前一帧 2D 坐标 */
    std::vector<cv::Point2f> getPrevPts() const { return extractPoints(prev_polar_pts); }

    /** 向后兼容：提取当前帧 2D 坐标 */
    std::vector<cv::Point2f> getCurPts() const { return extractPoints(cur_polar_pts); }

    /** 向后兼容：提取当前帧特征点 ID */
    std::vector<int> getIds() const { return extractIds(cur_polar_pts); }

    /** 静态辅助：从 PolarKeyPoint 列表提取 2D 坐标 */
    static std::vector<cv::Point2f> extractPoints(const std::vector<PolarKeyPoint>& pkps);

    /** 静态辅助：从 PolarKeyPoint 列表提取 ID 列表 */
    static std::vector<int> extractIds(const std::vector<PolarKeyPoint>& pkps);

private:
    // ====== 从 VINS 继承的核心状态 ======
    std::vector<camodocal::CameraPtr> m_camera;    ///< 相机模型（畸变校正/投影）
    cv::Mat prev_img, cur_img;                     ///< 原始图像（用于可视化）
    std::vector<PolarKeyPoint> prev_polar_pts;     ///< 前一帧合并特征点（含 ID、通道、坐标、track_cnt）
    std::vector<PolarKeyPoint> cur_polar_pts;      ///< 当前帧合并特征点
    double prev_time = 0.0;                        ///< 前一帧时间戳
    double cur_time = 0.0;                         ///< 当前帧时间戳
    bool hasPrediction = false;                    ///< 是否有外部预测点
    std::vector<cv::Point2f> predict_pts;          ///< 外部预测点坐标
    cv::Mat imTrack;                               ///< 跟踪可视化图像
    cv::Mat mask;                                  ///< 特征点 NMS 掩码
    bool stereo_cam = false;                       ///< 是否为双目相机
    int n_id = 0;                                  ///< 下一个新特征点的 ID
    std::mt19937 rng;                              ///< 随机数生成器

    // ====== 偏振处理特有状态 ======
    PolarConfig config;                                ///< 配置参数
    ChannelImages prev_polar_images;                   ///< 前一帧多通道偏振图像
    ChannelImages cur_polar_images;                    ///< 当前帧多通道偏振图像
    std::map<std::string, std::vector<int>> channel_to_ids;  ///< 通道 → ID 列表的映射

    // ====== 临时生成分组结构 ======
    static ChannelKeyPoints groupByChannel(const std::vector<PolarKeyPoint>& pts);

    // ====== 内部实现函数 ======

    /**
     * @brief 生成特征点 NMS 掩码
     *
     * 按 track_cnt 降序排列当前特征点，在掩码上以每个点为中心画圆，
     * 确保新检测的特征点不会与已跟踪点过于接近。
     */
    void setMask();

    /**
     * @brief 对图像应用三次中值滤波
     * @param image 输入图像
     * @return 滤波后图像（debug_no_filter=true 时直接返回原图）
     */
    cv::Mat applyFilter(const cv::Mat& image);

    /**
     * @brief 将偏振原始图分解为多通道图像并滤波
     * @param raw_image 输入偏振原始图
     * @return 包含 s0/dop/aopsin/aopcos 的 ChannelImages
     */
    ChannelImages getPolarizationImage(const cv::Mat& raw_image);

    /**
     * @brief 在指定通道图像上检测特征点
     * @param image 输入通道图像
     * @param channel_type 通道名称（"s0"/"dop"/"aopsin"/"aopcos"）
     * @param start_id 起始 ID（>=0 时按序分配，-1 时不分配）
     * @return PolarKeyPoint 列表
     */
    std::vector<PolarKeyPoint> extractFeatures(const cv::Mat& image, const std::string& channel_type, int start_id = -1);

    /**
     * @brief 对特征点执行非极大值抑制（NMS）
     * @param pkps_list 输入特征点列表
     * @param nms_mask 掩码（掩码值 >0 的位置直接排除）
     * @return NMS 筛选后的特征点列表
     */
    std::vector<PolarKeyPoint> efficientNMSKeypoints(const std::vector<PolarKeyPoint>& pkps_list, const cv::Mat& nms_mask);

    /**
     * @brief 特征匹配：LK 光流 + 反向验证 + KD-tree 最近邻
     * @param image_prev 前一帧通道图像
     * @param pkp_prev 前一帧特征点
     * @param image_curr 当前帧通道图像
     * @param pkp_curr 当前帧特征点
     * @return 匹配的点对列表
     */
    std::vector<MatchedPair> matchFeatures(
        const cv::Mat& image_prev, const std::vector<PolarKeyPoint>& pkp_prev,
        const cv::Mat& image_curr, const std::vector<PolarKeyPoint>& pkp_curr);

    /**
     * @brief 多通道联合 RANSAC 异常值剔除
     *
     * 将所有通道的匹配点合并，计算基础矩阵 F，
     * 用 RANSAC 一次性剔除所有通道的异常值。
     *
     * @param pairs_mc 按通道分组的匹配点对
     * @return RANSAC 筛选后的匹配点对（按通道分组）
     */
    ChannelPairs combinedFeatureRANSAC(const ChannelPairs& pairs_mc);

    // ====== 格式转换与状态管理 ======

    /**
     * @brief 将当前帧特征点转换为 VINS-Fusion 后端所需的格式
     * @return map<特征点ID, vector<相机索引, [x,y,z,u,v,vx,vy]>>
     *         其中 x,y,z 为归一化相机坐标，u,v 为像素坐标，vx,vy 为像素速度
     */
    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>
        convertToVINSFormat();

    /**
     * @brief 更新跟踪状态：合并匹配点和新检测点
     * @param matched_pairs RANSAC 筛选后的匹配点对
     * @param new_kp 新检测的特征点（未匹配的）
     */
    void updateTrackingState(const ChannelPairs& matched_pairs, const ChannelKeyPoints& new_kp);

    /**
     * @brief 为新检测的特征点分配唯一 ID
     * @param channel 通道名称
     * @param pkps 输入特征点列表
     * @return 分配好 ID 的特征点列表
     */
    std::vector<PolarKeyPoint> assignNewIds(const std::string& channel, std::vector<PolarKeyPoint> pkps);

    // ====== 可视化 ======

    /**
     * @brief 绘制跟踪轨迹到可视化图像
     *
     * 将 4 个通道拼接为田字格，在每个通道图像上绘制：
     *   - 特征点（彩色圆点）
     *   - 轨迹线（从前一帧到当前帧的连线）
     *
     * @param image 参考图像
     * @param pairs 匹配点对（用于绘制轨迹线）
     */
    void drawTrack(const cv::Mat &image, ChannelPairs &pairs);
};

#endif // POLARFP_TRACKER_H
