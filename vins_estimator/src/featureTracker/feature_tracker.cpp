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

#include "feature_tracker.h"
#include "feature_tracker_detector.h"
#include "feature_tracker_matcher.h"
#include "superpoint_detector.h"

/**
 * @brief 判断特征点是否在图像边界内
 * @param pt 特征点坐标
 * @return true 如果在边界内，false 否则
 *
 * 使用 BORDER_SIZE=1 作为安全边距，避免特征点过于靠近图像边缘
 * 导致后续处理（如光流、去畸变）出现不稳定。
 */
bool FeatureTracker::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

// (global distance function removed — replaced by FeatureTracker::distance member function)

/**
 * @brief 根据 status 状态压缩 Point2f 向量
 * @param v 待压缩的向量（原地修改）
 * @param status 状态标记，非零表示保留对应位置的元素
 *
 * 该函数在光流跟踪后使用，用于剔除跟踪失败或超界的特征点。
 */
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

/**
 * @brief 根据 status 状态压缩 int 向量
 * @param v 待压缩的向量（原地修改）
 * @param status 状态标记，非零表示保留对应位置的元素
 *
 * 用于剔除对应特征点的 ID、跟踪计数等整数属性。
 */
void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

/**
 * @brief FeatureTracker 构造函数
 *
 * 初始化双目标志为0、特征点ID计数器为0、预测标志为false。
 */
FeatureTracker::FeatureTracker()
{
    stereo_cam = 0;
    n_id = 0;
    hasPrediction = false;
}

/**
 * @brief 设置 mask 以筛选特征点，保证特征点在图像中均匀分布
 *
 * 算法流程：
 * 1. 将所有 (跟踪次数, 特征点坐标, ID) 打包并按跟踪次数降序排序
 * 2. 优先保留跟踪时间长的特征点
 * 3. 对每个保留的特征点，在 mask 上以 MIN_DIST 为半径画圆（置0）
 * 4. 后续特征点只有在 mask 值为255的区域才会被保留
 *
 * 这样可以有效避免特征点过度聚集在纹理丰富的区域。
 */
void FeatureTracker::setMask()
{
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            cur_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

/** @brief FeatureTracker 成员函数版本的两点间欧氏距离 */
double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

// =============================================
// 检测器/匹配器初始化
// =============================================

/** @brief 根据全局配置初始化检测器和匹配器 */
void FeatureTracker::initDetectorAndMatcher()
{
    DetectorConfig det_cfg;
    if (FEATURE_DETECTOR_TYPE == 1) {
        det_cfg.type = DetectorType::FAST;
        det_cfg.fast_threshold = FAST_THRESHOLD;
        det_cfg.fast_nonmax = FAST_NONMAX_SUPPRESSION;
    } else if (FEATURE_DETECTOR_TYPE == 2) {
        det_cfg.type = DetectorType::SUPERPOINT;
        det_cfg.sp_model_path = SUPERPOINT_MODEL_PATH;
        det_cfg.sp_use_gpu = (SUPERPOINT_USE_GPU != 0);
        det_cfg.sp_keypoint_threshold = SUPERPOINT_KEYPOINT_THRESHOLD;
        det_cfg.sp_nms_radius = SUPERPOINT_NMS_RADIUS;
    } else {
        det_cfg.type = DetectorType::GFTT;
        det_cfg.gftt_quality = 0.01;
        det_cfg.min_dist = MIN_DIST;
    }
    detector_ = createDetector(det_cfg);
    ROS_INFO("[PolarFP] Detector: %s", detector_->name().c_str());

    MatcherConfig match_cfg;
    if (FEATURE_MATCHER_TYPE == 1) {
        match_cfg.type = MatcherType::BRIEF_FLANN;
        match_cfg.brief_bytes = BRIEF_DESCRIPTOR_BYTES;
        match_cfg.flann_lsh_tables = FLANN_LSH_TABLES;
        match_cfg.flann_lsh_key_size = FLANN_LSH_KEY_SIZE;
        match_cfg.flann_multi_probe = FLANN_MULTI_PROBE;
        match_cfg.brief_match_dist_ratio = BRIEF_MATCH_DIST_RATIO;
    } else {
        match_cfg.type = MatcherType::LK_FLOW;
        match_cfg.lk_win_size = 21;
        match_cfg.lk_max_level = 3;
        match_cfg.flow_back = (FLOW_BACK != 0);
        match_cfg.back_dist_thresh = 0.5;
    }
    matcher_ = createMatcher(match_cfg);
    ROS_INFO("[PolarFP] Matcher: %s", matcher_->name().c_str());
}

// =============================================
// 偏振模式辅助函数实现
// =============================================

/** @brief 设置启用的偏振通道 */
void FeatureTracker::setPolarChannels(const vector<string>& channel_names)
{
    channels.clear();
    for (const auto& name : channel_names) {
        channels.emplace_back(name);
    }
    polar_mode = true;
    ROS_INFO("[PolarFP] setPolarChannels: %zu channels", channels.size());
}

/** @brief 设置偏振通道滤波配置 */
void FeatureTracker::setPolarFilterConfig(const PolarFilterConfig& cfg)
{
    polar_filter_cfg = cfg;
    if (cfg.filter_type == FILTER_BILATERAL) {
        ROS_INFO("[PolarFP] Bilateral filter: d=%d sigmaColor=%.1f sigmaSpace=%.1f",
                 cfg.bilateral_d, cfg.bilateral_sigmaColor, cfg.bilateral_sigmaSpace);
    } else if (cfg.filter_type == FILTER_GUIDED) {
        ROS_INFO("[PolarFP] Guided filter: radius=%d eps=%.4f",
                 cfg.guided_radius, cfg.guided_eps);
    } else if (cfg.filter_type == FILTER_MEDIAN) {
        ROS_INFO("[PolarFP] Median filter: kernel_size=%d",
                 cfg.median_kernel_size);
    }
}

/** @brief 从 PolarChannelResult 提取指定通道的 8bit 图像 */
cv::Mat FeatureTracker::getChannelImage(const PolarChannelResult& result, const string& channel)
{
    if (channel == "s0")      return result.S0_img.clone();
    if (channel == "dop")     return result.dop_img.clone();
    if (channel == "aopsin")  return result.sin_img.clone();
    if (channel == "aopcos")  return result.cos_img.clone();
    ROS_WARN("[PolarFP] unknown channel: %s", channel.c_str());
    return cv::Mat();
}

/** @brief 边界检查的通道版本 */
bool FeatureTracker::inBorderImpl(const ChannelState& ch, const cv::Point2f& pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < ch.col - BORDER_SIZE &&
           BORDER_SIZE <= img_y && img_y < ch.row - BORDER_SIZE;
}

/** @brief setMask 的单通道版本 */
void FeatureTracker::setMaskForChannel(ChannelState& ch)
{
    ch.mask = cv::Mat(ch.row, ch.col, CV_8UC1, cv::Scalar(255));

    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
    for (unsigned int i = 0; i < ch.cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(ch.track_cnt[i], make_pair(ch.cur_pts[i], ch.ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(),
         [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b) {
             return a.first > b.first;
         });

    ch.cur_pts.clear();
    ch.ids.clear();
    ch.track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (ch.mask.at<uchar>(it.second.first) == 255)
        {
            ch.cur_pts.push_back(it.second.first);
            ch.ids.push_back(it.second.second);
            ch.track_cnt.push_back(it.first);
            cv::circle(ch.mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

/** @brief 多通道可视化 */
void FeatureTracker::drawTrackPolar()
{
    if (channels.empty()) return;

    // Per-channel HSV: {H(0-179), S, V}
    // H from original channel color; S varies with track_cnt; V=255
    static const vector<cv::Vec3b> channelHSV = {
        { 30, 255, 255 },  // yellow - s0
        { 60, 255, 255 },  // green  - dop
        {  0, 255, 255 },  // red    - aopsin
        { 120, 255, 255 }, // blue   - aopcos
    };

    auto getColorByTrackCnt = [](int hue, int trackCnt) -> cv::Scalar {
        int sat = std::min(255, trackCnt * 255 / 20);
        cv::Mat hsv(1, 1, CV_8UC3, cv::Vec3b((uchar)hue, (uchar)sat, 255));
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        cv::Vec3b bgrVal = bgr.at<cv::Vec3b>(0, 0);
        return cv::Scalar(bgrVal[0], bgrVal[1], bgrVal[2]);
    };

    if (channels.size() == 1) {
        // 单通道:直接在该图像上绘制
        const auto& ch = channels[0];
        if (ch.cur_img.empty()) return;
        cv::cvtColor(ch.cur_img, imTrack, cv::COLOR_GRAY2BGR);
        for (size_t j = 0; j < ch.cur_pts.size(); j++) {
            cv::Scalar color = getColorByTrackCnt(channelHSV[0][0], ch.track_cnt[j]);
            cv::circle(imTrack, ch.cur_pts[j], 2, color, 2);
        }
    } else {
        // 多通道:2x2 网格拼接
        int h = channels[0].row;
        int w = channels[0].col;
        imTrack = cv::Mat(h * 2, w * 2, CV_8UC3, cv::Scalar(0));

        for (size_t c = 0; c < channels.size() && c < 4; c++) {
            const auto& ch = channels[c];
            if (ch.cur_img.empty()) continue;

            int gridRow = c / 2;
            int gridCol = c % 2;
            cv::Mat chBGR;
            cv::cvtColor(ch.cur_img, chBGR, cv::COLOR_GRAY2BGR);
            cv::Mat roi = imTrack(cv::Rect(gridCol * w, gridRow * h, w, h));
            chBGR.copyTo(roi);

            // 绘制特征点(饱和度随跟踪帧数变化)
            int hue = channelHSV[c][0];
            for (size_t j = 0; j < ch.cur_pts.size(); j++) {
                cv::Point2f pt(ch.cur_pts[j].x + gridCol * w, ch.cur_pts[j].y + gridRow * h);
                cv::Scalar color = getColorByTrackCnt(hue, ch.track_cnt[j]);
                cv::circle(imTrack, pt, 2, color, 2);
            }
            // 绘制轨迹箭头(固定通道颜色)
            for (size_t j = 0; j < ch.cur_pts.size(); j++) {
                auto it = ch.prevLeftPtsMap.find(ch.ids[j]);
                if (it != ch.prevLeftPtsMap.end()) {
                    cv::Point2f from(ch.cur_pts[j].x + gridCol * w, ch.cur_pts[j].y + gridRow * h);
                    cv::Point2f to(it->second.x + gridCol * w, it->second.y + gridRow * h);
                    cv::Scalar fullColor = getColorByTrackCnt(hue, 20);
                    cv::arrowedLine(imTrack, from, to, fullColor, 1, 8, 0, 0.2);
                }
            }
        }
    }
}

/**
 * @brief 核心跟踪函数：对输入图像进行特征点跟踪
 *
 * 偏振模式下:
 * 1. raw2polar 分解原始偏振图像为多通道
 * 2. 每通道独立运行检测器+匹配器管线
 * 3. 合并所有通道结果输出给后端
 *
 * 非偏振模式: 返回空 featureFrame (当前仅支持偏振模式)
 */
map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
FeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &/*_img1*/)
{
    cur_time = _cur_time;
    cur_img = _img;

    // ---- 偏振模式分支 ----
    if (!isPolarMode())
        return {};  // 非偏振模式当前未实现

    if (channels.empty()) {
        ROS_ERROR("[PolarFP] polar mode enabled but no channels configured");
        return {};
    }

    // =============================================
    // 1. 偏振图像分解: raw → 多通道
    // =============================================
    PolarChannelResult polar_result = raw2polar(cur_img, polar_filter_cfg);

    for (auto& ch : channels) {
        ch.cur_img = getChannelImage(polar_result, ch.name);
        if (ch.cur_img.empty()) {
            ROS_ERROR("[PolarFP] channel %s image is empty!", ch.name.c_str());
            continue;
        }
        ch.row = ch.cur_img.rows;
        ch.col = ch.cur_img.cols;
    }

    // =============================================
    // 2. 每通道独立跟踪 (模块化: detector_ + matcher_)
    // =============================================
    for (auto& ch : channels) {
        if (ch.cur_img.empty()) continue;

        ch.cur_time = cur_time;

        // ---- 2a. 匹配: prev → cur (LK光流 或 BRIEF+FLANN) ----
        if (!ch.prev_pts.empty() && !ch.prev_img.empty()) {
            MatchResult mr = matcher_->track(
                ch.prev_img, ch.cur_img,
                ch.prev_pts, ch.ids, ch.track_cnt,
                ch.prev_brief_desc);

            ch.prev_pts = std::move(mr.prev_pts);
            ch.cur_pts  = std::move(mr.cur_pts);
            ch.ids      = std::move(mr.ids);
            ch.track_cnt = std::move(mr.track_cnt);
        }

        // ---- 2b. 边界检查 ----
        if (!ch.cur_pts.empty()) {
            vector<uchar> status(ch.cur_pts.size(), 1);
            for (size_t i = 0; i < ch.cur_pts.size(); i++) {
                if (!inBorderImpl(ch, ch.cur_pts[i]))
                    status[i] = 0;
            }
            reduceVector(ch.prev_pts, status);
            reduceVector(ch.cur_pts, status);
            reduceVector(ch.ids, status);
            reduceVector(ch.track_cnt, status);
        }

        // ---- 2c. track_cnt++ ----
        for (auto &n : ch.track_cnt)
            n++;

        // ---- 2d. setMask: 空间均匀分布 ----
        setMaskForChannel(ch);
    }

    // =============================================
    // 2e. 新特征检测（分 SuperPoint 和 GFTT/FAST 两条路径）
    // =============================================
    if (detector_->name() == "SuperPoint") {
        // 收集所有通道的图像 + mask + max_cnt
        std::vector<cv::Mat> batch_imgs, batch_masks;
        std::vector<int> batch_max_cnts;
        for (const auto& ch : channels) {
            if (ch.cur_img.empty()) continue;
            batch_imgs.push_back(ch.cur_img);
            batch_masks.push_back(ch.mask);
            batch_max_cnts.push_back(MAX_CNT - static_cast<int>(ch.cur_pts.size()));
        }
        // Batch 推理一次，结果存到 detector 内部
        auto* sp_det = dynamic_cast<SuperPointFeatureDetector*>(detector_.get());
        if (sp_det) {
            sp_det->detectBatchForChannels(batch_imgs, batch_masks, batch_max_cnts);
        }
    }

    // 逐通道提取（统一入口：SuperPoint 从 batch 缓存取，GFTT/FAST 直接 detect）
    for (auto& ch : channels) {
        if (ch.cur_img.empty()) continue;

        int n_max_cnt = MAX_CNT - static_cast<int>(ch.cur_pts.size());
        if (n_max_cnt > 0 && !ch.mask.empty()) {
            ch.n_pts = detector_->detect(ch.cur_img, ch.mask, n_max_cnt);
        } else {
            ch.n_pts.clear();
        }

        // ---- 2f. 新特征加入, 分配全局唯一 ID ----
        for (auto &p : ch.n_pts) {
            ch.cur_pts.push_back(p);
            ch.ids.push_back(n_id++);
            ch.track_cnt.push_back(1);
        }

        // ---- 2g. 提取描述子 (BRIEF 模式) ----
        if (matcher_->name() == "BRIEF_FLANN" && !ch.cur_pts.empty()) {
            ch.prev_brief_desc = matcher_->extractDescriptors(ch.cur_img, ch.cur_pts);
        } else {
            ch.prev_brief_desc.clear();
        }

        // ---- 2h. 归一化坐标 + 速度 ----
        ch.cur_un_pts = undistortedPts(ch.cur_pts, m_camera[0]);

        // 构建当前帧 undistorted map
        map<int, cv::Point2f> cur_un_pts_map;
        for (size_t i = 0; i < ch.ids.size(); i++)
            cur_un_pts_map[ch.ids[i]] = ch.cur_un_pts[i];

        // 计算速度
        vector<cv::Point2f> vel;
        if (!ch.prev_un_pts_map.empty()) {
            double dt = ch.cur_time - ch.prev_time;
            for (size_t i = 0; i < ch.ids.size(); i++) {
                auto it = ch.prev_un_pts_map.find(ch.ids[i]);
                if (it != ch.prev_un_pts_map.end()) {
                    double vx = (ch.cur_un_pts[i].x - it->second.x) / dt;
                    double vy = (ch.cur_un_pts[i].y - it->second.y) / dt;
                    vel.push_back(cv::Point2f(vx, vy));
                } else {
                    vel.push_back(cv::Point2f(0, 0));
                }
            }
        } else {
            vel.assign(ch.cur_pts.size(), cv::Point2f(0, 0));
        }
        ch.pts_velocity = std::move(vel);

        // ---- 2i. 更新 prev 状态 ----
        ch.prev_img = ch.cur_img.clone();
        ch.prev_pts = ch.cur_pts;
        ch.prev_un_pts = ch.cur_un_pts;
        ch.prev_un_pts_map = std::move(ch.cur_un_pts_map);
        ch.prev_time = ch.cur_time;
        ch.prevLeftPtsMap.clear();
        for (size_t i = 0; i < ch.cur_pts.size(); i++)
            ch.prevLeftPtsMap[ch.ids[i]] = ch.cur_pts[i];
    }

    // =============================================
    // 3. 合并所有通道结果 → VINS 格式
    // =============================================
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (const auto& ch : channels) {
        if (ch.cur_img.empty()) continue;
        for (size_t i = 0; i < ch.ids.size(); i++) {
            int feature_id = ch.ids[i];
            double x = ch.cur_un_pts[i].x;
            double y = ch.cur_un_pts[i].y;
            double z = 1;
            double p_u = ch.cur_pts[i].x;
            double p_v = ch.cur_pts[i].y;
            double velocity_x = ch.pts_velocity[i].x;
            double velocity_y = ch.pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(0, xyz_uv_velocity);
        }
    }

    // =============================================
    // 4. 可视化
    // =============================================
    if (SHOW_TRACK)
        drawTrackPolar();

    // =============================================
    // 5. 更新全局状态
    // =============================================
    hasPrediction = false;

    if (!channels.empty() && !channels[0].cur_img.empty())
        cur_img = channels[0].cur_img;

    return featureFrame;
}

/**
 * @brief 使用基础矩阵 F 的 RANSAC 方法剔除外点
 *
 * 不同于 OpenCV 直接在像素坐标上计算 F 矩阵，本函数先将特征点
 * 通过相机模型投影到归一化平面，再缩放到像素尺度进行 F 矩阵估计。
 * 这样得到的 F 矩阵更准确地反映了相机运动的几何约束。
 *
 * 注意：该函数在当前代码中未被调用（被注释掉了）。
 */
void FeatureTracker::rejectWithF()
{
    if (cur_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

/**
 * @brief 读取相机内参配置文件
 * @param calib_file 相机标定文件路径列表
 *
 * 支持单目（1个文件）和双目（2个文件）配置。
 * 使用 camodocal 的 CameraFactory 从 YAML 文件生成相机模型，
 * 自动识别相机类型（针孔、梅涅特/鱼眼等）。
 */
void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
}

/**
 * @brief 显示去畸变后的图像（调试功能，默认关闭）
 * @param name 显示窗口名称
 *
 * 遍历图像每个像素，通过相机模型将其映射到归一化平面，
 * 再将归一化坐标按焦距和主点偏移到去畸变图像位置。
 * 该函数可用于直观验证相机标定参数的正确性。
 */
void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    // turn the following code on if you need
    // cv::imshow(name, undistortedImg);
    // cv::waitKey(0);
}

/**
 * @brief 将像素坐标通过相机模型转换到归一化平面
 * @param pts 待转换的像素坐标点集
 * @param cam 相机模型指针
 * @return 归一化平面坐标点集
 *
 * 使用 camodocal 库的 liftProjective 函数，将畸变图像上的
 * 像素坐标 (u, v) 映射到归一化相机平面上的 (x, y, 1)。
 * 归一化坐标消除了相机内参的影响，后续可直接用于几何计算。
 */
vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}

/**
 * @brief 计算特征点在归一化平面上的速度
 * @param ids 特征点ID列表
 * @param pts 当前帧特征点归一化坐标
 * @param cur_id_pts 输出：当前帧 ID -> 归一化坐标 的映射
 * @param prev_id_pts 上一帧 ID -> 归一化坐标 的映射
 * @return 每个特征点的速度 (vx, vy)，单位：归一化坐标/秒
 *
 * 通过查找上一帧中相同 ID 的特征点，计算坐标差除以时间差得到速度。
 * 如果某特征点在前一帧中不存在（新检测到的特征），速度设为 (0, 0)。
 * 速度信息用于后续滑动窗口中的关键帧筛选和初始化估计。
 */
vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
                                            map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
{
    vector<cv::Point2f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    // caculate points velocity
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time;

        for (unsigned int i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end())
            {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0));

        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}

/**
 * @brief 绘制特征点跟踪可视化图像
 * @param imLeft 左目图像
 * @param imRight 右目图像
 * @param curLeftIds 当前左目特征点ID
 * @param curLeftPts 当前左目特征点坐标
 * @param curRightPts 当前右目特征点坐标
 * @param prevLeftPtsMap 上一帧左目特征点 ID->坐标 映射
 *
 * 可视化内容：
 * - 左目特征点：颜色从红（新）到绿（跟踪时间长），阈值20帧
 * - 右目特征点（双目模式）：绿色圆点
 * - 运动箭头：从当前点指向上一帧对应位置（绿色）
 */
void FeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts,
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap)
{
    //int rows = imLeft.rows;
    int cols = imLeft.cols;
    if (!imRight.empty() && stereo_cam)
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();
    cv::cvtColor(imTrack, imTrack, cv::COLOR_GRAY2RGB);

    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    if (!imRight.empty() && stereo_cam)
    {
        for (size_t i = 0; i < curRightPts.size(); i++)
        {
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            //cv::Point2f leftPt = curLeftPtsTrackRight[i];
            //cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }

    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++)
    {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if(mapIt != prevLeftPtsMap.end())
        {
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }

    //draw prediction
    /*
    for(size_t i = 0; i < predict_pts_debug.size(); i++)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
    //printf("predict pts size %d \n", (int)predict_pts_debug.size());

    //cv::Mat imCur2Compress;
    //cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}


/**
 * @brief 设置特征点预测位置，用于加速光流匹配
 * @param predictPts 预测的3D点映射（ID -> 3D坐标）
 *
 * 当后端优化或IMU预积分提供了特征点的3D位置预测时，
 * 将3D点通过相机模型投影到像素平面作为光流的初始猜测值。
 * 使用 OPTFLOW_USE_INITIAL_FLOW 标志可以显著提高光流匹配
 * 在大幅运动或快速旋转场景下的成功率。
 * 若某特征点没有预测值，则使用其上一帧位置作为默认猜测。
 */
void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
    if (isPolarMode()) {
        hasPrediction = true;
        // 偏振模式下,预测暂时存储为全局,后续可改为 per-channel
        // 当前简化:设置标志但不做具体投影,光流不使用预测
    } else {
        hasPrediction = true;
        predict_pts.clear();
        predict_pts_debug.clear();
        map<int, Eigen::Vector3d>::iterator itPredict;
        for (size_t i = 0; i < ids.size(); i++)
        {
            //printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
            int id = ids[i];
            itPredict = predictPts.find(id);
            if (itPredict != predictPts.end())
            {
                Eigen::Vector2d tmp_uv;
                m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
                predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
                predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
            }
            else
                predict_pts.push_back(prev_pts[i]);
        }
    }
}


/**
 * @brief 移除被标记为外点的特征点
 * @param removePtsIds 需要移除的特征点ID集合
 *
 * 通常在后端优化（如滑动窗口BA）或基础矩阵RANSAC剔除后调用，
 * 将判定为外点的特征从跟踪器中删除，防止错误特征影响后续跟踪。
 * 注意：此处只更新 prev_pts、ids 和 track_cnt，不处理 cur_pts，
 * 因为外点剔除发生在 trackImage 之后、下一帧跟踪之前。
 */
void FeatureTracker::removeOutliers(set<int> &removePtsIds)
{
    if (isPolarMode()) {
        for (auto& ch : channels) {
            vector<uchar> status;
            for (size_t i = 0; i < ch.ids.size(); i++) {
                if (removePtsIds.count(ch.ids[i]))
                    status.push_back(0);
                else
                    status.push_back(1);
            }
            reduceVector(ch.prev_pts, status);
            reduceVector(ch.ids, status);
            reduceVector(ch.track_cnt, status);
        }
    } else {
        std::set<int>::iterator itSet;
        vector<uchar> status;
        for (size_t i = 0; i < ids.size(); i++)
        {
            itSet = removePtsIds.find(ids[i]);
            if(itSet != removePtsIds.end())
                status.push_back(0);
            else
                status.push_back(1);
        }

        reduceVector(prev_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
    }
}


/** @brief 返回跟踪可视化图像，供 ROS 话题发布或调试查看 */
cv::Mat FeatureTracker::getTrackImage()
{
    return imTrack;
}
