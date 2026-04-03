#ifndef POLARFP_TRACKER_H
#define POLARFP_TRACKER_H

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

// 配置结构体
struct PolarConfig {
    std::string FP_METHOD = "fast";
    std::vector<std::string> ALL_CHANNELS = {"s0", "dop", "aopsin", "aopcos"};
    std::vector<std::string> FP_CHANNELS = {"s0", "dop","aopsin", "aopcos"}; // , "dop",  "aopsin", "aopcos"
    std::map<std::string, cv::Scalar> CHANNEL_COLORS = {
        {"s0", cv::Scalar(0, 255, 255)},    // yellow
        {"dop", cv::Scalar(0, 255, 0)},     // green
        {"aopsin", cv::Scalar(0, 0, 255)},  // red
        {"aopcos", cv::Scalar(255, 0, 0)}   // blue
    };
    int KP_NUM_TARGET = 1000;

    // FAST特征点参数
    cv::Ptr<cv::FastFeatureDetector> FAST_S0_DT = cv::FastFeatureDetector::create(5);
    cv::Ptr<cv::FastFeatureDetector> FAST_DOP_DT = cv::FastFeatureDetector::create(15);
    cv::Ptr<cv::FastFeatureDetector> FAST_AOP_DT = cv::FastFeatureDetector::create(80);

    // GFTT特征点参数
    cv::Ptr<cv::GFTTDetector> GFTT_DT = cv::GFTTDetector::create(
        500,     // maxCorners
        0.03,      // qualityLevel
        7,        // minDistance
        3,        // blockSize
        false     // harrisDetector
    );

    // LK光流参数
    cv::Size LK_WIN_SIZE = cv::Size(21, 21);
    int LK_MAX_LEVEL = 6;
    cv::TermCriteria LK_CRITERIA = cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 0.01);
    double LK_MATCH_THRESHOLD = 10.0;

    // NMS与RANSAC参数
    int NMS_RADIUS = 21;
    double RANSAC_REMAP_THR = 2.0;
    double RANSAC_CONFIDENCE = 0.99;

    PolarConfig() = default;
};

// core数据结构
struct PolarKeyPoint {
    std::string channel;
    int id;
    cv::KeyPoint kp;
    int track_cnt;
    float score;
};
using MatchedPair  = std::pair<PolarKeyPoint, PolarKeyPoint>;
using ChannelKeyPoints = std::map<std::string, std::vector<PolarKeyPoint>>;
using ChannelPairs = std::map<std::string, std::vector<MatchedPair>>;
using ChannelImages = std::map<std::string, cv::Mat>;


class PolarFeatureTracker {
public:
    PolarFeatureTracker();
    ~PolarFeatureTracker();

    // 与 VINS 的 trackImage 签名完全兼容的入口
    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(
        double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());

    // 复用 VINS 的相机内参读取接口
    void readIntrinsicParameter(const std::vector<std::string> &calib_file);

    // 设置外部预测点（可选）
    void setPrediction(std::map<int, Eigen::Vector3d> &predictPts);

    // 获取用于可视化的图像
    cv::Mat getTrackImage();

    // 移除异常点(TODO: historical remove)
    void removeOutliers(std::set<int> &removePtsIds);

    // 配置访问
    void setConfig(const PolarConfig& new_config) { config = new_config; }
    PolarConfig getConfig() const { return config; }

    // 成员变量访问（用于调试）
    const std::vector<PolarKeyPoint>& getPrevPolarPts() const { return prev_polar_pts; }
    const std::vector<PolarKeyPoint>& getCurPolarPts() const { return cur_polar_pts; }

    // 向后兼容的访问函数
    std::vector<cv::Point2f> getPrevPts() const { return extractPoints(prev_polar_pts); }
    std::vector<cv::Point2f> getCurPts() const { return extractPoints(cur_polar_pts); }
    std::vector<int> getIds() const { return extractIds(cur_polar_pts); }

    // 静态辅助函数
    static std::vector<cv::Point2f> extractPoints(const std::vector<PolarKeyPoint>& pkps);
    static std::vector<int> extractIds(const std::vector<PolarKeyPoint>& pkps);

private:
    // 从VINS继承的核心状态
    std::vector<camodocal::CameraPtr> m_camera;
    cv::Mat prev_img, cur_img;                    // 原始图像（用于可视化）
    std::vector<PolarKeyPoint> prev_polar_pts, cur_polar_pts;  // 合并后的特征点（包含ID、通道、坐标、track_cnt）
    double prev_time = 0.0;
    double cur_time = 0.0;
    bool hasPrediction = false;
    std::vector<cv::Point2f> predict_pts;
    cv::Mat imTrack;
    cv::Mat mask;
    bool stereo_cam = false;
    int n_id = 0;  // 下一个特征点ID
    std::mt19937 rng;  // 随机数生成器

    // 偏振处理特有状态
    PolarConfig config;                                // 配置参数
    ChannelImages prev_polar_images;                   // 前一帧多通道偏振图像
    ChannelImages cur_polar_images;                    // 当前帧多通道偏振图像
    std::map<std::string, std::vector<int>> channel_to_ids;  // 通道到ID列表的映射

    // 临时生成分组结构（从单一真相源 polar_pts 生成）
    static ChannelKeyPoints groupByChannel(const std::vector<PolarKeyPoint>& pts);

    // 内部实现函数
    void setMask();
    cv::Mat applyFilter(const cv::Mat& image);
    ChannelImages getPolarizationImage(const cv::Mat& raw_image);
    std::vector<PolarKeyPoint> extractFeatures(const cv::Mat& image, const std::string& channel_type, int start_id = -1);
    std::vector<PolarKeyPoint> efficientNMSKeypoints(const std::vector<PolarKeyPoint>& pkps_list, const cv::Mat& nms_mask);
    std::vector<MatchedPair> matchFeatures(
        const cv::Mat& image_prev, const std::vector<PolarKeyPoint>& pkp_prev,
        const cv::Mat& image_curr, const std::vector<PolarKeyPoint>& pkp_curr);
    ChannelPairs combinedFeatureRANSAC(const ChannelPairs& pairs_mc);

    // 格式转换与状态管理
    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>
        convertToVINSFormat();
    void updateTrackingState(const ChannelPairs& matched_pairs, const ChannelKeyPoints& new_kp);
    std::vector<PolarKeyPoint> assignNewIds(const std::string& channel, std::vector<PolarKeyPoint> pkps);

    // 可视化
    void drawTrack(const cv::Mat &image, ChannelPairs &pairs);
};

#endif // POLARFP_TRACKER_H