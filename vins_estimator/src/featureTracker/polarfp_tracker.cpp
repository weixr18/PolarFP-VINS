#include "polarfp_tracker.h"
#include "PolarChannel.h"
#include "nanoflann.hpp"  // KD-tree库
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
// 点云适配器，用于nanoflann
struct PointCloud {
    const std::vector<cv::Point2f>& pts;
    PointCloud(const std::vector<cv::Point2f>& pts_) : pts(pts_) {}

    inline size_t kdtree_get_point_count() const { return pts.size(); }

    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return dim == 0 ? pts[idx].x : pts[idx].y;
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};
} // 匿名命名空间

// 静态辅助函数：从 PolarKeyPoint 提取 Point2f 坐标
std::vector<cv::Point2f> PolarFeatureTracker::extractPoints(const std::vector<PolarKeyPoint>& pkps) {
    std::vector<cv::Point2f> pts;
    pts.reserve(pkps.size());
    for (const auto& pkp : pkps) {
        pts.push_back(pkp.kp.pt);
    }
    return pts;
}

// 静态辅助函数：从 PolarKeyPoint 提取 ID 列表
std::vector<int> PolarFeatureTracker::extractIds(const std::vector<PolarKeyPoint>& pkps) {
    std::vector<int> ids;
    ids.reserve(pkps.size());
    for (const auto& pkp : pkps) {
        ids.push_back(pkp.id);
    }
    return ids;
}

// 静态辅助函数：按通道分组 PolarKeyPoint
ChannelKeyPoints PolarFeatureTracker::groupByChannel(const std::vector<PolarKeyPoint>& pts) {
    ChannelKeyPoints grouped;
    for (const auto& pkp : pts) {
        grouped[pkp.channel].push_back(pkp);
    }
    return grouped;
}

// KD-tree匹配函数
static std::vector<std::pair<int, int>> matchPointsKdTree(
    const std::vector<cv::Point2f>& points_a,
    const std::vector<cv::Point2f>& points_b,
    float match_radius) {
    if (points_a.empty() || points_b.empty()) {
        return {};
    }
    // 构建kd-tree（使用nanoflann）
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud>,
        PointCloud, 2>;
    PointCloud cloud_b(points_b);
    KDTree tree(2, cloud_b, {10});  // 叶子节点大小10
    tree.buildIndex();
    // 存储匹配结果
    std::vector<std::pair<int, int>> matches;
    std::vector<bool> used_b(points_b.size(), false);
    std::vector<std::pair<float, int>> best_match_for_a;  // (distance_squared, idx_b)
    // 第一轮：为每个点找到最近邻
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
    // 第二轮：贪心匹配（避免冲突）
    std::vector<size_t> order(points_a.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    // 按距离排序，优先匹配距离近的
    std::sort(order.begin(), order.end(),
        [&best_match_for_a](size_t i1, size_t i2) {
            if (best_match_for_a[i1].first < 0) return false;
            if (best_match_for_a[i2].first < 0) return true;
            return best_match_for_a[i1].first < best_match_for_a[i2].first;
        });
    // 执行匹配
    for (size_t idx_a : order) {
        auto& [dist_sqr, idx_b] = best_match_for_a[idx_a];
        if (dist_sqr >= 0 && idx_b >= 0 && !used_b[idx_b]) {
            matches.emplace_back((int)idx_a, idx_b);
            used_b[idx_b] = true;
        }
    }
    return matches;
}


PolarFeatureTracker::PolarFeatureTracker() {
    // 初始化随机数生成器
    rng = std::mt19937(std::random_device{}());
    // 其他默认初始化
}

PolarFeatureTracker::~PolarFeatureTracker() {}

// 读取相机内参
void PolarFeatureTracker::readIntrinsicParameter(const std::vector<std::string> &calib_file) {
    for (size_t i = 0; i < calib_file.size(); i++) {
        camodocal::CameraPtr camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
    stereo_cam = (calib_file.size() == 2);
}

// set mask of tracked points
void PolarFeatureTracker::setMask() {
    mask = cv::Mat(cur_img.rows, cur_img.cols, CV_8UC1, cv::Scalar(255));
    std::vector<std::tuple<int, cv::Point2f, int, std::string>> cnt_pts_id;  // track_cnt, pt, id, channel
    for (size_t i = 0; i < cur_polar_pts.size(); i++)
        cnt_pts_id.emplace_back(cur_polar_pts[i].track_cnt, cur_polar_pts[i].kp.pt, cur_polar_pts[i].id, cur_polar_pts[i].channel);
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
            cv::circle(mask, pt, config.KP_NUM_TARGET > 0 ? std::min(10, config.KP_NUM_TARGET/50) : 10, 0, -1);
        }
    }
}

// filter the origin points
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

// 1. get polar image (+ filtering)
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

// 2. extract key-points
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

    // 转换为 PolarKeyPoint
    std::vector<PolarKeyPoint> result;
    result.reserve(cv_kps.size());
    int id = start_id;
    for (const auto& kp : cv_kps) {
        PolarKeyPoint pkp;
        pkp.channel = channel_type;
        pkp.id = (start_id >= 0) ? id++ : -1;
        pkp.kp = kp;
        pkp.track_cnt = 0;
        if (config.FP_METHOD == "gftt") {
            pkp.score = 1.0;
        } else {
            pkp.score = kp.response / 100.0;
        }
        result.push_back(pkp);
    }
    return result;
}

// 3. L-K opt-flow 特征匹配
std::vector<MatchedPair> PolarFeatureTracker::matchFeatures(
    const cv::Mat& image_prev, const std::vector<PolarKeyPoint>& pkp_prev,
    const cv::Mat& image_curr, const std::vector<PolarKeyPoint>& pkp_curr) {
    TicToc tic_step;
    tic_step.tic();
    // 1. LK optflow
    if (pkp_prev.empty() || pkp_curr.empty()) return {};
    std::vector<cv::Point2f> pts_prev = extractPoints(pkp_prev);
    std::vector<cv::Point2f> pts_pred_lk;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(image_prev, image_curr, pts_prev, pts_pred_lk, status, err,
                             config.LK_WIN_SIZE, config.LK_MAX_LEVEL, config.LK_CRITERIA);
    // 2. KD-tree匹配
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
            int i = match.first;  // pts_pred_lk索引
            int j = match.second; // pts_curr_kp索引
            if (i >= 0 && i < (int)n_prev && j >= 0 && j < (int)n_curr && status[i]) {
                // 计算实际距离验证
                double d = cv::norm(pts_pred_lk[i] - pts_curr_kp[j]);
                if (d <= config.LK_MATCH_THRESHOLD) {
                    // successfull matched pair
                    pairs.emplace_back(pkp_prev[i], pkp_curr[j]);
                }
            }
        }
    }
    return pairs;
}


// 4. 多通道联合RANSAC
ChannelPairs PolarFeatureTracker::combinedFeatureRANSAC(const ChannelPairs& pairs_mc) {
    ChannelPairs result;
    for (const auto& channel : config.FP_CHANNELS) {
        result[channel] = {};
    }
    std::vector<MatchedPair> all_pairs;
    std::map<std::string, std::pair<int, int>> ranges;
    int start = 0;
    for (const auto &c : pairs_mc) {
        int len = (int)c.second.size();
        ranges[c.first] = {start, start + len};
        start += len;
        all_pairs.insert(all_pairs.end(), c.second.begin(), c.second.end());
    }
    if (all_pairs.size() < 8) return result;
    std::vector<cv::Point2f> L, R;
    for (auto &pr : all_pairs) { 
        L.push_back(pr.first.kp.pt); 
        R.push_back(pr.second.kp.pt); 
    }
    std::vector<uchar> mask;
    cv::Mat F = cv::findFundamentalMat(L, R, mask, cv::FM_RANSAC, config.RANSAC_REMAP_THR, config.RANSAC_CONFIDENCE);
    if (mask.empty()) return result;
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


// NMS new point filter
std::vector<PolarKeyPoint> PolarFeatureTracker::efficientNMSKeypoints(const std::vector<PolarKeyPoint>& pkps_list, const cv::Mat& nms_mask) {
    if (pkps_list.empty()) return {};
    std::vector<int> indices(pkps_list.size());
    std::vector<float> responses;
    for (size_t i = 0; i < pkps_list.size(); ++i) { indices[i] = (int)i; responses.push_back(pkps_list[i].kp.response); }
    std::sort(indices.begin(), indices.end(), [&responses](int a, int b){ return responses[a] > responses[b]; });
    std::vector<bool> keep(pkps_list.size(), true);
    if (!nms_mask.empty()) {
        for (size_t i = 0; i < pkps_list.size(); ++i) {
            cv::Point pt = pkps_list[i].kp.pt;
            if (pt.x >= 0 && pt.x < nms_mask.cols && pt.y >= 0 && pt.y < nms_mask.rows) {
                if (nms_mask.at<uchar>(pt) > 0) keep[i] = false;
            }
        }
    }
    std::vector<PolarKeyPoint> out;
    float radius_sq = (float)config.NMS_RADIUS * (float)config.NMS_RADIUS;
    std::vector<cv::Point2f> points;
    for (const auto &pkp : pkps_list) points.push_back(pkp.kp.pt);
    for (int idx : indices) {
        if (!keep[idx]) continue;
        out.push_back(pkps_list[idx]);
        cv::Point2f center = points[idx];
        for (size_t j = 0; j < points.size(); ++j) {
            if (keep[j]) {
                float dx = points[j].x - center.x;
                float dy = points[j].y - center.y;
                if (dx*dx + dy*dy <= radius_sq) keep[j] = false;
            }
        }
        keep[idx] = true;
    }
    return out;
}


// 为指定通道的特征点分配新ID
std::vector<PolarKeyPoint> PolarFeatureTracker::assignNewIds(const std::string& channel, std::vector<PolarKeyPoint> pkps) {
    for (auto& pkp : pkps) {
        pkp.id = n_id++;
        pkp.channel = channel;
    }
    return pkps;
}

// 5. 合并匹配点和新检测点
void PolarFeatureTracker::updateTrackingState(const ChannelPairs& matched_pairs, const ChannelKeyPoints& new_kp) {
    // 清空当前状态
    cur_polar_pts.clear();
    channel_to_ids.clear();
    // 构建前一帧ID到track_cnt和score的映射，方便查找
    std::map<int, int> prev_id_to_track_cnt;
    std::map<int, float> prev_id_to_score;
    for (const auto& pkp : prev_polar_pts) {
        prev_id_to_track_cnt[pkp.id] = pkp.track_cnt;
        prev_id_to_score[pkp.id] = pkp.score;
    }
    // 1. 处理匹配点（更新跟踪计数）
    for (const auto& [channel, pairs] : matched_pairs) {
        for (const auto& match : pairs) {
            PolarKeyPoint curr_kp = match.second; // Create a copy of match.second
            const PolarKeyPoint& prev_kp = match.first;
            // update curr_kp id with prev point
            curr_kp.id = prev_kp.id;
            // 直接从映射中查找跟踪计数
            auto it = prev_id_to_track_cnt.find(prev_kp.id);
            curr_kp.track_cnt = (it != prev_id_to_track_cnt.end()) ? (it->second + 1) : 1;
            // 累加 score：旧分数 + 新检测分数
            auto score_it = prev_id_to_score.find(prev_kp.id);
            if (score_it != prev_id_to_score.end()) {
                curr_kp.score += score_it->second;
            }
            // 更新当前状态
            cur_polar_pts.push_back(curr_kp);
            channel_to_ids[channel].push_back(curr_kp.id);
        }
    }

    // 2. 处理新检测点
    for (const auto& [channel, pkps] : new_kp) {
        std::vector<PolarKeyPoint> new_pkps = assignNewIds(channel, pkps);
        for (auto& pkp : new_pkps) {
            pkp.track_cnt = 1;  // 新检测点跟踪计数初始化为1
            cur_polar_pts.push_back(pkp);
            channel_to_ids[channel].push_back(pkp.id);
        }
    }

    // 3. 确保特征点数量不超过目标值
    if ((int)cur_polar_pts.size() > config.KP_NUM_TARGET) {
        // 按 score 排序，保留高质量点（跟踪历史分数累加高的点）
        std::vector<std::pair<float, size_t>> score_index_pairs;  // (score, index)
        for (size_t i = 0; i < cur_polar_pts.size(); ++i) {
            score_index_pairs.emplace_back(cur_polar_pts[i].score, i);
        }
        std::sort(score_index_pairs.begin(), score_index_pairs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        // 重新构建状态
        std::vector<PolarKeyPoint> new_polar_pts;
        std::map<std::string, std::vector<int>> new_channel_to_ids;
        for (int i = 0; i < config.KP_NUM_TARGET && i < (int)score_index_pairs.size(); ++i) {
            const auto& pkp = cur_polar_pts[score_index_pairs[i].second];
            new_polar_pts.push_back(pkp);  // score 已包含在对象中
            new_channel_to_ids[pkp.channel].push_back(pkp.id);
        }
        cur_polar_pts = std::move(new_polar_pts);
        channel_to_ids = std::move(new_channel_to_ids);
    }
}

// 6. 转换为VINS格式
std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>
PolarFeatureTracker::convertToVINSFormat() {
    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    if (m_camera.empty()) {
        // 如果没有相机模型，使用简单转换
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
        // 像素坐标
        double p_u = pkp.kp.pt.x;
        double p_v = pkp.kp.pt.y;
        // 去畸变归一化坐标
        Eigen::Vector3d un_pt;
        cam->liftProjective(Eigen::Vector2d(p_u, p_v), un_pt);
        double x = un_pt.x() / un_pt.z();
        double y = un_pt.y() / un_pt.z();
        double z = 1.0;
        // 归一化速度
        double velocity_x = 0.0;
        double velocity_y = 0.0;
        // 查找前一帧对应点计算速度
        for (const auto& prev_pkp : prev_polar_pts) {
            if (prev_pkp.id == pkp.id) {
                velocity_x = pkp.kp.pt.x - prev_pkp.kp.pt.x;
                velocity_y = pkp.kp.pt.y - prev_pkp.kp.pt.y;
                break;
            }
        }
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[pkp.id].emplace_back(0, xyz_uv_velocity);  // 相机ID设为0
    }

    return featureFrame;
}



// 设置预测点
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

// 移除异常点
void PolarFeatureTracker::removeOutliers(std::set<int> &removePtsIds) {
    std::vector<PolarKeyPoint> new_prev_pts, new_cur_pts;
    // 过滤当前帧
    for (size_t i = 0; i < cur_polar_pts.size(); i++) {
        if (removePtsIds.find(cur_polar_pts[i].id) == removePtsIds.end()) {
            new_cur_pts.push_back(cur_polar_pts[i]);  // track_cnt 随对象保留
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

// 获取跟踪图像
cv::Mat PolarFeatureTracker::getTrackImage() {
    if (imTrack.empty()) {
        imTrack = cur_img.clone();
        if (imTrack.channels() == 1) {
            cv::cvtColor(imTrack, imTrack, cv::COLOR_GRAY2BGR);
        }
    }
    return imTrack;
}

// 绘制跟踪轨迹
void PolarFeatureTracker::drawTrack(const cv::Mat &image, ChannelPairs &pairs) {
    imTrack = image.clone();
    if (imTrack.channels() == 1) {
        cv::cvtColor(imTrack, imTrack, cv::COLOR_GRAY2BGR);
    }
    // 绘制特征点（按通道颜色）
    for (const auto &c: pairs){
        std::string channel = c.first;
        cv::Scalar color = cv::Scalar(0, 255, 0);  // 默认绿色
        auto it = config.CHANNEL_COLORS.find(channel);
        if (it != config.CHANNEL_COLORS.end()) {
            color = it->second;
        }
        for (const MatchedPair& pair: c.second) {
            const auto& curr_pt = pair.second.kp.pt;
            const auto& prev_pt = pair.first.kp.pt;
            // 绘制特征点
            cv::circle(imTrack, curr_pt, 3, color, -1); // 绘制特征点
            // 绘制跟踪轨迹
            cv::line(imTrack, prev_pt, curr_pt, color, 1, cv::LINE_8);
        }
    }
}

bool show_time_stats = true;


// 主函数：trackImage
std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>
PolarFeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1) {
    TicToc tic_total;
    TicToc tic_step;

    // 1. 偏振图像分解
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

    // 2. 多通道特征提取
    tic_step.tic();
    ChannelKeyPoints cur_channel_kps;  // 临时变量，仅在当前帧使用
    for (const auto& channel : config.ALL_CHANNELS) {
        if (cur_polar_images.find(channel) != cur_polar_images.end()) {
            cur_channel_kps[channel] = extractFeatures(cur_polar_images[channel], channel);
        }
    }
    if(show_time_stats)
        ROS_DEBUG("Step 2 (Multi-channel feature extraction) costs: %fms", tic_step.toc());

    // 3. 特征匹配
    tic_step.tic();
    ChannelPairs matched_pairs;
    ChannelKeyPoints new_kp;
    if (!prev_polar_images.empty()) {
        // 3.1 如果有前一帧，从 prev_polar_pts 临时生成分组结构（携带正确ID）
        ChannelKeyPoints prev_channel_kps = groupByChannel(prev_polar_pts);
        for (const auto& channel : config.FP_CHANNELS) {
            const cv::Mat& curr_img = cur_polar_images[channel];
            std::vector<PolarKeyPoint>& curr_pkp = cur_channel_kps[channel];
            cv::Mat nms_mask = cv::Mat::zeros(curr_img.size(), CV_8UC1);
            // 3.2 特征匹配
            matched_pairs[channel] = matchFeatures(
                prev_polar_images[channel], prev_channel_kps[channel],
                curr_img, curr_pkp
            );
            if(show_time_stats)
                ROS_DEBUG("Step 3.1 (matchFeatures) costs: %fms", tic_step.toc());
            // 3.3 新点NMS过滤
            tic_step.tic();
            for (const auto& match : matched_pairs[channel]) {
                // ROS_DEBUG("Step 3.1 match.first.id = %d", match.first.id);
                cv::Point pt(match.second.kp.pt.x, match.second.kp.pt.y);
                cv::circle(nms_mask, pt, config.NMS_RADIUS, cv::Scalar(255), -1);
            }
            new_kp[channel] = efficientNMSKeypoints(curr_pkp, nms_mask);
            if(show_time_stats)
                ROS_DEBUG("Step 3.2 (NMS) costs: %fms", tic_step.toc());
        }
    } else {
        // 如果第一帧，没有匹配
        for (const auto& channel : config.FP_CHANNELS) {
            matched_pairs[channel] = {};
            new_kp[channel] = cur_channel_kps[channel];
        }
        if(show_time_stats)
            ROS_DEBUG("Step 3 (Feature matching) costs: %fms", tic_step.toc());
    }

    // 4. 多通道联合RANSAC
    tic_step.tic();
    ChannelPairs pairs_ransac_mc = combinedFeatureRANSAC(matched_pairs);
    if(show_time_stats)
        ROS_DEBUG("Step 4 (Multi-channel RANSAC) costs: %fms", tic_step.toc());

    // 5. 更新跟踪状态
    tic_step.tic();
    updateTrackingState(pairs_ransac_mc, new_kp);
    if(show_time_stats)
        ROS_DEBUG("Step 5 (Update tracking state) costs: %fms", tic_step.toc());

    // 6. 转换为VINS格式
    tic_step.tic();
    auto featureFrame = convertToVINSFormat();
    if(show_time_stats)
        ROS_DEBUG("Step 6 (Convert to VINS format) costs: %fms", tic_step.toc());

    // 7. FP 可视化（如果启用）
    if (SHOW_TRACK) {
        tic_step.tic();
        drawTrack(cur_img, pairs_ransac_mc);
        if(show_time_stats)
            ROS_DEBUG("Step 7 (Visualization) costs: %fms", tic_step.toc());
    }

    // 8. 更新前一帧状态
    tic_step.tic();
    prev_img = cur_img.clone();
    prev_polar_pts = cur_polar_pts;  // track_cnt 随对象自动拷贝
    prev_polar_images = cur_polar_images;
    prev_time = cur_time;
    hasPrediction = false;
    if(show_time_stats)
        ROS_DEBUG("Step 8 (Update previous frame state) costs: %fms", tic_step.toc());

    ROS_DEBUG("PolarFP feature tracker total costs: %fms", tic_total.toc());
    return featureFrame;
}