/**
 * @file polarfp_tracker.cpp
 * @brief 偏振多通道特征跟踪器的实现文件。
 *
 * 实现完整的特征跟踪管线：
 *   1. 偏振图像分解 → 2. 多通道特征检测 → 3. LK 光流 + KD-tree 匹配
 *   → 4. 多通道联合 RANSAC → 5. 状态更新 → 6. VINS 格式转换
 *
 * 与 VINS-Fusion 后端完全兼容，直接替换原有 FeatureTracker。
 */

#include "polarfp_tracker.h"
#include "PolarChannel.h"
#include "nanoflann.hpp"
#include <algorithm>
#include <random>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <limits>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"


namespace {
/**
 * @brief 点云适配器 —— 将 cv::Point2f 向量包装为 nanoflann 所需的接口
 *
 * nanoflann 需要数据源提供 kdtree_get_point_count() 和 kdtree_get_pt()，
 * 此结构体桥接 OpenCV 点集与 nanoflann KD-tree。
 */
struct PointCloud {
    const std::vector<cv::Point2f>& pts;
    PointCloud(const std::vector<cv::Point2f>& pts_) : pts(pts_) {}

    /// 返回点集大小
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    /// 返回第 idx 个点的第 dim 维坐标（0=x, 1=y）
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return dim == 0 ? pts[idx].x : pts[idx].y;
    }

    /// 不支持边界框查询
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};
} // 匿名命名空间


// ============================================================
// 静态辅助函数
// ============================================================

/** 从 PolarKeyPoint 列表提取 2D 坐标 */
std::vector<cv::Point2f> PolarFeatureTracker::extractPoints(const std::vector<PolarKeyPoint>& pkps) {
    std::vector<cv::Point2f> pts;
    pts.reserve(pkps.size());
    for (const auto& pkp : pkps) {
        pts.push_back(pkp.kp.pt);
    }
    return pts;
}

/** 从 PolarKeyPoint 列表提取 ID 列表 */
std::vector<int> PolarFeatureTracker::extractIds(const std::vector<PolarKeyPoint>& pkps) {
    std::vector<int> ids;
    ids.reserve(pkps.size());
    for (const auto& pkp : pkps) {
        ids.push_back(pkp.id);
    }
    return ids;
}

/** 按通道名称分组 PolarKeyPoint */
ChannelKeyPoints PolarFeatureTracker::groupByChannel(const std::vector<PolarKeyPoint>& pts) {
    ChannelKeyPoints grouped;
    for (const auto& pkp : pts) {
        grouped[pkp.channel].push_back(pkp);
    }
    return grouped;
}


// ============================================================
// KD-tree 匹配
// ============================================================

/**
 * @brief 基于 KD-tree 的二维点集最近邻匹配
 *
 * 对 points_a 中的每个点在 points_b 中查找最近邻，
 * 使用贪心策略避免一对多冲突：优先匹配距离近的点对。
 *
 * @param points_a 源点集（如 LK 光流预测位置）
 * @param points_b 目标点集（如当前帧检测到的特征点）
 * @param match_radius 最大匹配距离（超过此距离视为无匹配）
 * @return 匹配对列表 <points_a 索引, points_b 索引>
 */
static std::vector<std::pair<int, int>> matchPointsKdTree(
    const std::vector<cv::Point2f>& points_a,
    const std::vector<cv::Point2f>& points_b,
    float match_radius) {
    if (points_a.empty() || points_b.empty()) {
        return {};
    }

    // 构建 KD-tree（使用 points_b 作为索引）
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud>,
        PointCloud, 2>;
    PointCloud cloud_b(points_b);
    KDTree tree(2, cloud_b, {10});  // 叶子节点大小 10
    tree.buildIndex();

    std::vector<std::pair<int, int>> matches;
    std::vector<bool> used_b(points_b.size(), false);
    std::vector<std::pair<float, int>> best_match_for_a;  // (距离平方, points_b 索引)

    // 第一轮：为每个 points_a 的点找到最近邻
    float radius_sq = match_radius * match_radius;
    for (size_t i = 0; i < points_a.size(); ++i) {
        const float query[2] = {points_a[i].x, points_a[i].y};
        size_t num_results = 1;
        size_t ret_index;
        float out_dist_sqr;
        nanoflann::KNNResultSet<float> resultSet(num_results);
        resultSet.init(&ret_index, &out_dist_sqr);
        if (tree.findNeighbors(resultSet, query, nanoflann::SearchParameters()) &&
            out_dist_sqr <= radius_sq) {
            best_match_for_a.emplace_back(out_dist_sqr, (int)ret_index);
        } else {
            best_match_for_a.emplace_back(-1.0f, -1);  // 无匹配
        }
    }

    // 第二轮：贪心匹配，避免冲突
    // 按距离排序，优先匹配距离近的点对
    std::vector<size_t> order(points_a.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(),
        [&best_match_for_a](size_t i1, size_t i2) {
            if (best_match_for_a[i1].first < 0) return false;
            if (best_match_for_a[i2].first < 0) return true;
            return best_match_for_a[i1].first < best_match_for_a[i2].first;
        });

    // 执行匹配（每个 points_b 的点只能被匹配一次）
    for (size_t idx_a : order) {
        auto& [dist_sqr, idx_b] = best_match_for_a[idx_a];
        if (dist_sqr >= 0 && idx_b >= 0 && !used_b[idx_b]) {
            matches.emplace_back((int)idx_a, idx_b);
            used_b[idx_b] = true;
        }
    }
    return matches;
}


// ============================================================
// 构造 / 析构
// ============================================================

PolarFeatureTracker::PolarFeatureTracker() {
    // 初始化随机数生成器
    rng = std::mt19937(std::random_device{}());
}

PolarFeatureTracker::~PolarFeatureTracker() {}


// ============================================================
// 公共接口
// ============================================================

/** 读取相机内参（从 YAML 标定文件） */
void PolarFeatureTracker::readIntrinsicParameter(const std::vector<std::string> &calib_file) {
    for (size_t i = 0; i < calib_file.size(); i++) {
        camodocal::CameraPtr camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
    stereo_cam = (calib_file.size() == 2);
}

/** 设置外部预测点（由后端优化器提供，暂未实际使用） */
void PolarFeatureTracker::setPrediction(std::map<int, Eigen::Vector3d> &predictPts) {
    hasPrediction = true;
    predict_pts.clear();
    for (const auto& pkp : prev_polar_pts) {
        int id = pkp.id;
        auto it = predictPts.find(id);
        if (it != predictPts.end()) {
            Eigen::Vector2d uv;
            m_camera[0]->spaceToPlane(it->second, uv);
            predict_pts.push_back(cv::Point2f((float)uv.x(), (float)uv.y()));
        } else {
            predict_pts.push_back(pkp.kp.pt);
        }
    }
}

/** 移除被后端优化器标记为异常的点 */
void PolarFeatureTracker::removeOutliers(std::set<int> &removePtsIds) {
    std::vector<PolarKeyPoint> new_prev_pts, new_cur_pts;
    // 过滤当前帧
    for (size_t i = 0; i < cur_polar_pts.size(); i++) {
        if (removePtsIds.find(cur_polar_pts[i].id) == removePtsIds.end()) {
            new_cur_pts.push_back(cur_polar_pts[i]);
        }
    }
    // 过滤前一帧
    for (const auto& pkp : prev_polar_pts) {
        if (removePtsIds.find(pkp.id) == removePtsIds.end()) {
            new_prev_pts.push_back(pkp);
        }
    }
    prev_polar_pts = new_prev_pts;
    cur_polar_pts = new_cur_pts;
}

/** 获取跟踪可视化图像（田字格 4 通道拼接图） */
cv::Mat PolarFeatureTracker::getTrackImage() {
    if (imTrack.empty()) {
        imTrack = cur_img.clone();
        if (imTrack.channels() == 1) {
            cv::cvtColor(imTrack, imTrack, cv::COLOR_GRAY2BGR);
        }
    }
    return imTrack;
}


// ============================================================
// 内部辅助函数
// ============================================================

/**
 * @brief 生成特征点 NMS 掩码
 *
 * 按 track_cnt 降序排列当前特征点，以每个点为中心画圆覆盖掩码，
 * 确保新检测的特征点不会与已跟踪的高质量特征点重叠。
 */
void PolarFeatureTracker::setMask() {
    mask = cv::Mat(cur_img.rows, cur_img.cols, CV_8UC1, cv::Scalar(255));
    std::vector<std::tuple<int, cv::Point2f, int, std::string>> cnt_pts_id;  // track_cnt, pt, id, channel
    for (size_t i = 0; i < cur_polar_pts.size(); i++)
        cnt_pts_id.emplace_back(cur_polar_pts[i].track_cnt, cur_polar_pts[i].kp.pt, cur_polar_pts[i].id, cur_polar_pts[i].channel);
    // 按跟踪帧数降序排列，优先保留长期跟踪的点
    std::sort(cnt_pts_id.begin(), cnt_pts_id.end(),
        [](const auto &a, const auto &b) {
            return std::get<0>(a) > std::get<0>(b);
        });
    cur_polar_pts.clear();
    channel_to_ids.clear();
    for (auto &it : cnt_pts_id) {
        cv::Point2f pt = std::get<1>(it);
        if (mask.at<uchar>(pt) == 255) {
            PolarKeyPoint pkp;
            pkp.kp.pt = pt;
            pkp.id = std::get<2>(it);
            pkp.channel = std::get<3>(it);
            pkp.track_cnt = std::get<0>(it);
            cur_polar_pts.push_back(pkp);
            channel_to_ids[pkp.channel].push_back(pkp.id);
            // 在掩码上画圆，覆盖该点周围的区域
            cv::circle(mask, pt, config.KP_NUM_TARGET > 0 ? std::min(10, config.KP_NUM_TARGET/50) : 10, 0, -1);
        }
    }
}

/**
 * @brief 对图像应用三次中值滤波（3x3 窗口）
 *
 * 用于平滑偏振通道图像，减少检测器噪声。
 * debug_no_filter 标志可用于跳过滤波（调试用）。
 */
cv::Mat PolarFeatureTracker::applyFilter(const cv::Mat& image) {
    bool debug_no_filter = false;
    if (debug_no_filter){
        return image;
    }
    else {
        cv::Mat image_1, image_2, image_3;
        cv::medianBlur(image, image_1, 3);
        cv::medianBlur(image_1, image_2, 3);
        cv::medianBlur(image_2, image_3, 3);
        return image_3;
    }
}

/**
 * @brief 将偏振原始图分解为 4 个通道图像
 *
 * 调用 raw2polar() 获取 S0/DoP/sin(AoP)/cos(AoP)，
 * 然后对 DoP 和 AoP 通道应用中值滤波。
 */
ChannelImages PolarFeatureTracker::getPolarizationImage(const cv::Mat& raw_image) {
    if (raw_image.empty()) {
        std::cerr << "Error: empty input image in getPolarizationImage" << std::endl;
        return {};
    }
    PolarChannelResult chnl_images = raw2polar(raw_image);
    cv::Mat S0_img = chnl_images.S0_img;
    cv::Mat dop_img = chnl_images.dop_img;
    cv::Mat aopsin_img = chnl_images.sin_img;
    cv::Mat aopcos_img = chnl_images.cos_img;
    try {
        // 对偏振通道做滤波以减少噪声
        dop_img = applyFilter(dop_img);
        aopsin_img = applyFilter(aopsin_img);
        aopcos_img = applyFilter(aopcos_img);
    } catch (const std::exception& e) {
        std::cerr << "Exception while filtering polarization image: " << e.what() << std::endl;
    }
    ChannelImages result;
    result["s0"] = S0_img;
    result["dop"] = dop_img;
    result["aopsin"] = aopsin_img;
    result["aopcos"] = aopcos_img;
    return result;
}

/**
 * @brief 在指定通道图像上检测特征点
 *
 * 根据 config.FP_METHOD 选择 FAST 或 GFTT 检测器。
 * FAST 检测器根据通道类型选择不同阈值：
 *   - S0: 低阈值（5），强度图特征丰富
 *   - DoP: 中阈值（15），偏振度特征适中
 *   - AoP: 高阈值（80），只保留强偏振角特征
 *
 * @param image 输入通道图像（8bit 灰度）
 * @param channel_type 通道名称
 * @param start_id 起始 ID（>=0 时按序分配，-1 时不分配）
 * @return PolarKeyPoint 列表
 */
std::vector<PolarKeyPoint> PolarFeatureTracker::extractFeatures(const cv::Mat& image, const std::string& channel_type, int start_id) {
    std::vector<cv::KeyPoint> cv_kps;

    if (config.FP_METHOD == "fast") {
        cv::Ptr<cv::FastFeatureDetector> detector;
        if (channel_type == "s0") {
            detector = config.FAST_S0_DT;
        } else if (channel_type == "dop") {
            detector = config.FAST_DOP_DT;
        } else {
            detector = config.FAST_AOP_DT;
        }
        detector->detect(image, cv_kps);
    } else if (config.FP_METHOD == "gftt") {
        config.GFTT_DT->detect(image, cv_kps);
    } else {
        std::cerr << "Error: unknown FP type: " << config.FP_METHOD << std::endl;
        return {};
    }

    // 转换为 PolarKeyPoint 格式
    std::vector<PolarKeyPoint> result;
    result.reserve(cv_kps.size());
    int id = start_id;
    for (const auto& kp : cv_kps) {
        PolarKeyPoint pkp;
        pkp.channel = channel_type;
        pkp.id = (start_id >= 0) ? id++ : -1;
        pkp.kp = kp;
        pkp.track_cnt = 0;
        // GFTT 使用统一评分，FAST 使用响应值
        if (config.FP_METHOD == "gftt") {
            pkp.score = 1.0;
        } else {
            pkp.score = kp.response / 100.0;
        }
        result.push_back(pkp);
    }
    return result;
}

/**
 * @brief 特征匹配：LK 光流 + 反向验证 + KD-tree 最近邻
 *
 * 匹配流程：
 *   1. 正向 LK 光流：从前一帧预测到当前帧
 *   2. 反向 LK 光流：从预测位置回推到前一帧
 *   3. 一致性检查：正向+反向均成功且回推误差 < 0.5 像素
 *   4. KD-tree 匹配：在通过一致性检查的预测位置与当前帧特征点间找最近邻
 *
 * @param image_prev 前一帧通道图像
 * @param pkp_prev 前一帧特征点
 * @param image_curr 当前帧通道图像
 * @param pkp_curr 当前帧特征点
 * @return 匹配的点对列表
 */
std::vector<MatchedPair> PolarFeatureTracker::matchFeatures(
    const cv::Mat& image_prev, const std::vector<PolarKeyPoint>& pkp_prev,
    const cv::Mat& image_curr, const std::vector<PolarKeyPoint>& pkp_curr) {
    TicToc tic_step;
    tic_step.tic();

    if (pkp_prev.empty() || pkp_curr.empty()) return {};

    // 1. LK 光流正向追踪
    std::vector<cv::Point2f> pts_prev = extractPoints(pkp_prev);
    std::vector<cv::Point2f> pts_pred_lk;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(image_prev, image_curr, pts_prev, pts_pred_lk, status, err,
                             config.LK_WIN_SIZE, config.LK_MAX_LEVEL, config.LK_CRITERIA);

    // 2. 反向 LK 光流验证（Bidirectional Check）
    std::vector<uchar> reverse_status;
    std::vector<cv::Point2f> reverse_pts = pts_pred_lk;
    cv::calcOpticalFlowPyrLK(image_curr, image_prev, pts_pred_lk, reverse_pts, reverse_status, err,
                             config.LK_WIN_SIZE, 1, config.LK_CRITERIA, cv::OPTFLOW_USE_INITIAL_FLOW);

    // 3. 一致性检查：正向+反向成功且回推距离 < 0.5 像素
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i] && reverse_status[i]) {
            double dist = cv::norm(pts_prev[i] - reverse_pts[i]);
            if (dist <= 0.5) {
                status[i] = 1;  // 保留：双向光流一致
            } else {
                status[i] = 0;  // 剔除：反向不一致（漂移过大）
            }
        } else {
            status[i] = 0;  // 剔除：正向或反向光流失败
        }
    }

    // 4. KD-tree 匹配：在通过验证的预测位置与当前帧特征点间找最近邻
    tic_step.tic();
    std::vector<cv::Point2f> pts_curr_kp = extractPoints(pkp_curr);
    size_t n_prev = pts_pred_lk.size();
    size_t n_curr = pts_curr_kp.size();
    ROS_DEBUG("matchFeatures: n_prev: %ld, n_curr: %ld", n_prev, n_curr);
    std::vector<MatchedPair> pairs;
    if (n_prev > 0 && n_curr > 0) {
        auto kd_matches = matchPointsKdTree(pts_pred_lk, pts_curr_kp, config.LK_MATCH_THRESHOLD);
        // 筛选有效匹配
        for (const auto& match : kd_matches) {
            int i = match.first;   // pts_pred_lk 索引
            int j = match.second;  // pts_curr_kp 索引
            if (i >= 0 && i < (int)n_prev && j >= 0 && j < (int)n_curr && status[i]) {
                double d = cv::norm(pts_pred_lk[i] - pts_curr_kp[j]);
                if (d <= config.LK_MATCH_THRESHOLD) {
                    pairs.emplace_back(pkp_prev[i], pkp_curr[j]);
                }
            }
        }
    }
    return pairs;
}


/**
 * @brief 多通道联合 RANSAC 异常值剔除
 *
 * 将所有通道的匹配点合并为一个集合，统一计算基础矩阵 F，
 * 用 RANSAC 一次性剔除所有通道的异常值。
 * 相比逐通道 RANSAC，联合方式利用了更多约束，更鲁棒。
 *
 * @param pairs_mc 按通道分组的匹配点对
 * @return RANSAC 筛选后的匹配点对（按通道分组）
 */
ChannelPairs PolarFeatureTracker::combinedFeatureRANSAC(const ChannelPairs& pairs_mc) {
    ChannelPairs result;
    for (const auto& channel : config.FP_CHANNELS) {
        result[channel] = {};
    }

    // 合并所有通道的匹配点，记录每个通道的范围
    std::vector<MatchedPair> all_pairs;
    std::map<std::string, std::pair<int, int>> ranges;
    int start = 0;
    for (const auto &c : pairs_mc) {
        int len = (int)c.second.size();
        ranges[c.first] = {start, start + len};
        start += len;
        all_pairs.insert(all_pairs.end(), c.second.begin(), c.second.end());
    }

    if (all_pairs.size() < 8) return result;  // 基础矩阵至少需要 8 对点

    // 提取左右点集
    std::vector<cv::Point2f> L, R;
    for (auto &pr : all_pairs) {
        L.push_back(pr.first.kp.pt);
        R.push_back(pr.second.kp.pt);
    }

    // 计算基础矩阵 F（RANSAC）
    std::vector<uchar> mask;
    cv::Mat F = cv::findFundamentalMat(L, R, mask, cv::FM_RANSAC, config.RANSAC_REMAP_THR, config.RANSAC_CONFIDENCE);
    if (mask.empty()) return result;

    // 根据 mask 筛选内点，按通道分回结果
    for (const auto &rng : ranges) result[rng.first] = {};
    for (size_t i = 0; i < mask.size(); ++i) {
        if (!mask[i]) continue;
        for (const auto &rng : ranges) {
            if ((int)i >= rng.second.first && (int)i < rng.second.second) {
                result[rng.first].push_back(all_pairs[i]);
                break;
            }
        }
    }
    return result;
}


/**
 * @brief 对特征点执行非极大值抑制（NMS）
 *
 * 两步策略：
 *   1. 先用掩码排除已跟踪点周围的区域
 *   2. 按响应值降序遍历，对每个保留点抑制其 NMS_RADIUS 范围内的其他点
 *
 * @param pkps_list 输入特征点列表
 * @param nms_mask 掩码（>0 的位置直接排除）
 * @return NMS 筛选后的特征点列表
 */
std::vector<PolarKeyPoint> PolarFeatureTracker::efficientNMSKeypoints(const std::vector<PolarKeyPoint>& pkps_list, const cv::Mat& nms_mask) {
    if (pkps_list.empty()) return {};

    // 按响应值排序
    std::vector<int> indices(pkps_list.size());
    std::vector<float> responses;
    for (size_t i = 0; i < pkps_list.size(); ++i) { indices[i] = (int)i; responses.push_back(pkps_list[i].kp.response); }
    std::sort(indices.begin(), indices.end(), [&responses](int a, int b){ return responses[a] > responses[b]; });

    // 标记被掩码覆盖的点
    std::vector<bool> keep(pkps_list.size(), true);
    if (!nms_mask.empty()) {
        for (size_t i = 0; i < pkps_list.size(); ++i) {
            cv::Point pt = pkps_list[i].kp.pt;
            if (pt.x >= 0 && pt.x < nms_mask.cols && pt.y >= 0 && pt.y < nms_mask.rows) {
                if (nms_mask.at<uchar>(pt) > 0) keep[i] = false;
            }
        }
    }

    // NMS 抑制
    std::vector<PolarKeyPoint> out;
    float radius_sq = (float)config.NMS_RADIUS * (float)config.NMS_RADIUS;
    std::vector<cv::Point2f> points;
    for (const auto &pkp : pkps_list) points.push_back(pkp.kp.pt);
    for (int idx : indices) {
        if (!keep[idx]) continue;
        out.push_back(pkps_list[idx]);
        // 抑制该点半径范围内的其他点
        cv::Point2f center = points[idx];
        for (size_t j = 0; j < points.size(); ++j) {
            if (keep[j]) {
                float dx = points[j].x - center.x;
                float dy = points[j].y - center.y;
                if (dx*dx + dy*dy <= radius_sq) keep[j] = false;
            }
        }
        keep[idx] = true;  // 自身保留
    }
    return out;
}


/**
 * @brief 为新检测的特征点分配唯一 ID
 * @param channel 通道名称
 * @param pkps 输入特征点列表
 * @return 分配好 ID 的特征点列表
 */
std::vector<PolarKeyPoint> PolarFeatureTracker::assignNewIds(const std::string& channel, std::vector<PolarKeyPoint> pkps) {
    for (auto& pkp : pkps) {
        pkp.id = n_id++;
        pkp.channel = channel;
    }
    return pkps;
}


/**
 * @brief 更新跟踪状态：合并 RANSAC 筛选后的匹配点和新检测点
 *
 * 1. 匹配点：继承前一帧 ID，track_cnt +1，score 累加
 * 2. 新检测点：分配新 ID，track_cnt = 1
 * 3. 如果总点数超过目标值，按 score 降序筛选，保留高质量点
 *
 * @param matched_pairs RANSAC 筛选后的匹配点对
 * @param new_kp 新检测的特征点
 */
void PolarFeatureTracker::updateTrackingState(const ChannelPairs& matched_pairs, const ChannelKeyPoints& new_kp) {
    cur_polar_pts.clear();
    channel_to_ids.clear();

    // 构建前一帧 ID → track_cnt / score 的映射
    std::map<int, int> prev_id_to_track_cnt;
    std::map<int, float> prev_id_to_score;
    for (const auto& pkp : prev_polar_pts) {
        prev_id_to_track_cnt[pkp.id] = pkp.track_cnt;
        prev_id_to_score[pkp.id] = pkp.score;
    }

    // 1. 处理匹配点：继承 ID，更新跟踪计数和评分
    for (const auto& [channel, pairs] : matched_pairs) {
        for (const auto& match : pairs) {
            PolarKeyPoint curr_kp = match.second;
            const PolarKeyPoint& prev_kp = match.first;
            curr_kp.id = prev_kp.id;
            // 跟踪计数 +1
            auto it = prev_id_to_track_cnt.find(prev_kp.id);
            curr_kp.track_cnt = (it != prev_id_to_track_cnt.end()) ? (it->second + 1) : 1;
            // 累加评分：旧分数 + 新检测分数
            auto score_it = prev_id_to_score.find(prev_kp.id);
            if (score_it != prev_id_to_score.end()) {
                curr_kp.score += score_it->second;
            }
            cur_polar_pts.push_back(curr_kp);
            channel_to_ids[channel].push_back(curr_kp.id);
        }
    }

    // 2. 处理新检测点：分配新 ID
    for (const auto& [channel, pkps] : new_kp) {
        std::vector<PolarKeyPoint> new_pkps = assignNewIds(channel, pkps);
        for (auto& pkp : new_pkps) {
            pkp.track_cnt = 1;  // 新检测点跟踪计数初始化为 1
            cur_polar_pts.push_back(pkp);
            channel_to_ids[channel].push_back(pkp.id);
        }
    }

    // 3. 如果总点数超过目标值，按 score 降序筛选
    if ((int)cur_polar_pts.size() > config.KP_NUM_TARGET) {
        std::vector<std::pair<float, size_t>> score_index_pairs;  // (score, index)
        for (size_t i = 0; i < cur_polar_pts.size(); ++i) {
            score_index_pairs.emplace_back(cur_polar_pts[i].score, i);
        }
        std::sort(score_index_pairs.begin(), score_index_pairs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        // 重新构建状态，只保留前 KP_NUM_TARGET 个高分点
        std::vector<PolarKeyPoint> new_polar_pts;
        std::map<std::string, std::vector<int>> new_channel_to_ids;
        for (int i = 0; i < config.KP_NUM_TARGET && i < (int)score_index_pairs.size(); ++i) {
            const auto& pkp = cur_polar_pts[score_index_pairs[i].second];
            new_polar_pts.push_back(pkp);
            new_channel_to_ids[pkp.channel].push_back(pkp.id);
        }
        cur_polar_pts = std::move(new_polar_pts);
        channel_to_ids = std::move(new_channel_to_ids);
    }
}


/**
 * @brief 将当前帧特征点转换为 VINS-Fusion 后端所需的格式
 *
 * 输出格式：map<特征点ID, vector<相机索引, [x,y,z,u,v,vx,vy]>>
 *   - x, y, z: 归一化相机坐标（z=1）
 *   - u, v: 像素坐标
 *   - vx, vy: 像素速度（当前帧 - 前一帧）
 *
 * 如果有相机模型，使用 liftProjective 做去畸变和归一化；
 * 否则使用恒等变换（仅用于调试）。
 *
 * @return VINS 格式的特征帧
 */
std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>
PolarFeatureTracker::convertToVINSFormat() {
    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    if (m_camera.empty()) {
        // 如果没有相机模型，使用简单转换（调试模式）
        ROS_WARN("Warning: camera model not initialized, using identity transformation");
        for (const auto& pkp : cur_polar_pts) {
            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            double x = pkp.kp.pt.x;
            double y = pkp.kp.pt.y;
            xyz_uv_velocity << x, y, 1.0, pkp.kp.pt.x, pkp.kp.pt.y, 0.0, 0.0;
            featureFrame[pkp.id].emplace_back(0, xyz_uv_velocity);
        }
        return featureFrame;
    }

    camodocal::CameraPtr cam = m_camera[0]; // 单目假设
    for (const auto& pkp : cur_polar_pts) {
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        double p_u = pkp.kp.pt.x;
        double p_v = pkp.kp.pt.y;

        // 去畸变 + 归一化
        Eigen::Vector3d un_pt;
        cam->liftProjective(Eigen::Vector2d(p_u, p_v), un_pt);
        double x = un_pt.x() / un_pt.z();
        double y = un_pt.y() / un_pt.z();
        double z = 1.0;

        // 计算像素速度（当前帧 - 前一帧）
        double velocity_x = 0.0;
        double velocity_y = 0.0;
        for (const auto& prev_pkp : prev_polar_pts) {
            if (prev_pkp.id == pkp.id) {
                velocity_x = pkp.kp.pt.x - prev_pkp.kp.pt.x;
                velocity_y = pkp.kp.pt.y - prev_pkp.kp.pt.y;
                break;
            }
        }

        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[pkp.id].emplace_back(0, xyz_uv_velocity);  // 相机 ID 设为 0
    }
    return featureFrame;
}


// ============================================================
// 可视化
// ============================================================

/**
 * @brief 绘制跟踪轨迹到田字格 4 通道拼接图
 *
 * 将 s0/dop/aopsin/aopcos 四个通道拼接为 2x2 网格，
 * 在每个通道图像上绘制：
 *   - 彩色圆点标记当前帧特征点
 *   - 线段连接前一帧和当前帧位置（轨迹）
 *
 * @param image 参考图像
 * @param pairs 匹配点对（用于绘制轨迹线）
 */
void PolarFeatureTracker::drawTrack(const cv::Mat &image, ChannelPairs &pairs) {
    cv::Mat s0_img, dop_img, aopsin_img, aopcos_img;

    auto it_s0 = cur_polar_images.find("s0");
    auto it_dop = cur_polar_images.find("dop");
    auto it_aopsin = cur_polar_images.find("aopsin");
    auto it_aopcos = cur_polar_images.find("aopcos");

    // 检查 4 个通道是否都存在，缺少则回退到单图显示
    if (it_s0 == cur_polar_images.end() || it_dop == cur_polar_images.end() ||
        it_aopsin == cur_polar_images.end() || it_aopcos == cur_polar_images.end()) {
        imTrack = image.clone();
        if (imTrack.channels() == 1) {
            cv::cvtColor(imTrack, imTrack, cv::COLOR_GRAY2BGR);
        }
        return;
    }

    s0_img = it_s0->second.clone();
    dop_img = it_dop->second.clone();
    aopsin_img = it_aopsin->second.clone();
    aopcos_img = it_aopcos->second.clone();

    // 确保所有通道为彩色图
    if (s0_img.channels() == 1) cv::cvtColor(s0_img, s0_img, cv::COLOR_GRAY2BGR);
    if (dop_img.channels() == 1) cv::cvtColor(dop_img, dop_img, cv::COLOR_GRAY2BGR);
    if (aopsin_img.channels() == 1) cv::cvtColor(aopsin_img, aopsin_img, cv::COLOR_GRAY2BGR);
    if (aopcos_img.channels() == 1) cv::cvtColor(aopcos_img, aopcos_img, cv::COLOR_GRAY2BGR);

    // 获取各通道可视化颜色
    cv::Scalar s0_color = config.CHANNEL_COLORS.count("s0") ? config.CHANNEL_COLORS.at("s0") : cv::Scalar(0, 255, 255);
    cv::Scalar dop_color = config.CHANNEL_COLORS.count("dop") ? config.CHANNEL_COLORS.at("dop") : cv::Scalar(0, 255, 0);
    cv::Scalar aopsin_color = config.CHANNEL_COLORS.count("aopsin") ? config.CHANNEL_COLORS.at("aopsin") : cv::Scalar(0, 0, 255);
    cv::Scalar aopcos_color = config.CHANNEL_COLORS.count("aopcos") ? config.CHANNEL_COLORS.at("aopcos") : cv::Scalar(255, 0, 0);

    // 在 s0 图像上绘制 s0 通道的特征点和轨迹
    auto it_s0_pairs = pairs.find("s0");
    if (it_s0_pairs != pairs.end()) {
        for (const MatchedPair& pair : it_s0_pairs->second) {
            const auto& curr_pt = pair.second.kp.pt;
            const auto& prev_pt = pair.first.kp.pt;
            cv::circle(s0_img, curr_pt, 3, s0_color, -1);
            cv::line(s0_img, prev_pt, curr_pt, s0_color, 1, cv::LINE_8);
        }
    }

    // 在 dop 图像上绘制 dop 通道的特征点和轨迹
    auto it_dop_pairs = pairs.find("dop");
    if (it_dop_pairs != pairs.end()) {
        for (const MatchedPair& pair : it_dop_pairs->second) {
            const auto& curr_pt = pair.second.kp.pt;
            const auto& prev_pt = pair.first.kp.pt;
            cv::circle(dop_img, curr_pt, 3, dop_color, -1);
            cv::line(dop_img, prev_pt, curr_pt, dop_color, 1, cv::LINE_8);
        }
    }

    // 在 aopsin 图像上绘制 aopsin 通道的特征点和轨迹
    auto it_aopsin_pairs = pairs.find("aopsin");
    if (it_aopsin_pairs != pairs.end()) {
        for (const MatchedPair& pair : it_aopsin_pairs->second) {
            const auto& curr_pt = pair.second.kp.pt;
            const auto& prev_pt = pair.first.kp.pt;
            cv::circle(aopsin_img, curr_pt, 3, aopsin_color, -1);
            cv::line(aopsin_img, prev_pt, curr_pt, aopsin_color, 1, cv::LINE_8);
        }
    }

    // 在 aopcos 图像上绘制 aopcos 通道的特征点和轨迹
    auto it_aopcos_pairs = pairs.find("aopcos");
    if (it_aopcos_pairs != pairs.end()) {
        for (const MatchedPair& pair : it_aopcos_pairs->second) {
            const auto& curr_pt = pair.second.kp.pt;
            const auto& prev_pt = pair.first.kp.pt;
            cv::circle(aopcos_img, curr_pt, 3, aopcos_color, -1);
            cv::line(aopcos_img, prev_pt, curr_pt, aopcos_color, 1, cv::LINE_8);
        }
    }

    // 构建田字格拼接图 (2h x 2w)
    int h = s0_img.rows;
    int w = s0_img.cols;
    imTrack = cv::Mat::zeros(h * 2, w * 2, CV_8UC3);

    // 左上: s0, 右上: dop, 左下: aopsin, 右下: aopcos
    s0_img.copyTo(imTrack(cv::Rect(0, 0, w, h)));
    dop_img.copyTo(imTrack(cv::Rect(w, 0, w, h)));
    aopsin_img.copyTo(imTrack(cv::Rect(0, h, w, h)));
    aopcos_img.copyTo(imTrack(cv::Rect(w, h, w, h)));

    // 添加通道标签
    cv::putText(imTrack, "S0", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, s0_color, 2);
    cv::putText(imTrack, "DoP", cv::Point(w + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, dop_color, 2);
    cv::putText(imTrack, "AoP Sin", cv::Point(10, h + 30), cv::FONT_HERSHEY_SIMPLEX, 1, aopsin_color, 2);
    cv::putText(imTrack, "AoP Cos", cv::Point(w + 10, h + 30), cv::FONT_HERSHEY_SIMPLEX, 1, aopcos_color, 2);

    // 画分隔线
    cv::line(imTrack, cv::Point(w, 0), cv::Point(w, 2 * h), cv::Scalar(128, 128, 128), 2);
    cv::line(imTrack, cv::Point(0, h), cv::Point(2 * w, h), cv::Scalar(128, 128, 128), 2);
}


// ============================================================
// 主入口：trackImage
// ============================================================

/** 是否打印每步耗时统计 */
bool show_time_stats = true;

/**
 * @brief 主入口函数：处理一帧新图像，完成从偏振解码到 VINS 格式输出的完整流程
 *
 * 处理步骤：
 *   1. 偏振图像分解（raw2polar → 4 通道）
 *   2. 多通道特征检测（FAST/GFTT）
 *   3. 特征匹配（LK 光流 + KD-tree）+ 新点 NMS
 *   4. 多通道联合 RANSAC 异常值剔除
 *   5. 更新跟踪状态（合并匹配点和新点，评分筛选）
 *   6. 转换为 VINS 后端格式
 *   7. 可视化（如果启用 SHOW_TRACK）
 *   8. 更新前一帧状态
 *
 * @param _cur_time 当前帧时间戳
 * @param _img 当前帧偏振原始图像
 * @param _img1 双目右图（暂未实现）
 * @return VINS 格式的特征帧
 */
std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>
PolarFeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1) {
    TicToc tic_total;
    TicToc tic_step;

    // Step 1: 偏振图像分解
    tic_step.tic();
    cur_time = _cur_time;
    cur_polar_images = getPolarizationImage(_img);
    cur_img = cur_polar_images["dop"].clone();

    if (cur_polar_images.empty()) {
        std::cerr << "Error: failed to process polarization image" << std::endl;
        return {};
    }
    if(show_time_stats)
        ROS_DEBUG("Step 1 (Polarization image decomposition) costs: %fms", tic_step.toc());

    // Step 2: 多通道特征检测
    tic_step.tic();
    ChannelKeyPoints cur_channel_kps;  // 临时变量，仅在当前帧使用
    for (const auto& channel : config.ALL_CHANNELS) {
        if (cur_polar_images.find(channel) != cur_polar_images.end()) {
            cur_channel_kps[channel] = extractFeatures(cur_polar_images[channel], channel);
        }
    }
    if(show_time_stats)
        ROS_DEBUG("Step 2 (Multi-channel feature extraction) costs: %fms", tic_step.toc());

    // Step 3: 特征匹配 + 新点 NMS
    tic_step.tic();
    ChannelPairs matched_pairs;
    ChannelKeyPoints new_kp;
    if (!prev_polar_images.empty()) {
        // 3.1 从 prev_polar_pts 生成分组结构（携带正确 ID）
        ChannelKeyPoints prev_channel_kps = groupByChannel(prev_polar_pts);
        for (const auto& channel : config.FP_CHANNELS) {
            const cv::Mat& curr_img = cur_polar_images[channel];
            std::vector<PolarKeyPoint>& curr_pkp = cur_channel_kps[channel];
            cv::Mat nms_mask = cv::Mat::zeros(curr_img.size(), CV_8UC1);

            // 3.2 LK 光流 + KD-tree 匹配
            matched_pairs[channel] = matchFeatures(
                prev_polar_images[channel], prev_channel_kps[channel],
                curr_img, curr_pkp
            );
            if(show_time_stats)
                ROS_DEBUG("Step 3.1 (matchFeatures) costs: %fms", tic_step.toc());

            // 3.3 构建已匹配点的 NMS 掩码
            tic_step.tic();
            for (const auto& match : matched_pairs[channel]) {
                cv::Point pt(match.second.kp.pt.x, match.second.kp.pt.y);
                cv::circle(nms_mask, pt, config.NMS_RADIUS, cv::Scalar(255), -1);
            }
            // 3.4 对未匹配的新点做 NMS
            new_kp[channel] = efficientNMSKeypoints(curr_pkp, nms_mask);
            if(show_time_stats)
                ROS_DEBUG("Step 3.2 (NMS) costs: %fms", tic_step.toc());
        }
    } else {
        // 第一帧：没有前一帧，所有检测点都是新点
        for (const auto& channel : config.FP_CHANNELS) {
            matched_pairs[channel] = {};
            new_kp[channel] = cur_channel_kps[channel];
        }
        if(show_time_stats)
            ROS_DEBUG("Step 3 (Feature matching) costs: %fms", tic_step.toc());
    }

    // Step 4: 多通道联合 RANSAC 异常值剔除
    tic_step.tic();
    ChannelPairs pairs_ransac_mc = combinedFeatureRANSAC(matched_pairs);
    if(show_time_stats)
        ROS_DEBUG("Step 4 (Multi-channel RANSAC) costs: %fms", tic_step.toc());

    // Step 5: 更新跟踪状态
    tic_step.tic();
    updateTrackingState(pairs_ransac_mc, new_kp);
    if(show_time_stats)
        ROS_DEBUG("Step 5 (Update tracking state) costs: %fms", tic_step.toc());

    // Step 6: 转换为 VINS 格式
    tic_step.tic();
    auto featureFrame = convertToVINSFormat();
    if(show_time_stats)
        ROS_DEBUG("Step 6 (Convert to VINS format) costs: %fms", tic_step.toc());

    // Step 7: 可视化（如果启用）
    if (SHOW_TRACK) {
        tic_step.tic();
        drawTrack(cur_img, pairs_ransac_mc);
        if(show_time_stats)
            ROS_DEBUG("Step 7 (Visualization) costs: %fms", tic_step.toc());
    }

    // Step 8: 更新前一帧状态
    tic_step.tic();
    prev_img = cur_img.clone();
    prev_polar_pts = cur_polar_pts;
    prev_polar_images = cur_polar_images;
    prev_time = cur_time;
    hasPrediction = false;
    if(show_time_stats)
        ROS_DEBUG("Step 8 (Update previous frame state) costs: %fms", tic_step.toc());

    ROS_DEBUG("PolarFP feature tracker total costs: %fms", tic_total.toc());
    return featureFrame;
}
