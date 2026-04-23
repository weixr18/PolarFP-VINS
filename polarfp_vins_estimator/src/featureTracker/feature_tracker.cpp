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
 * @brief 根据status状态压缩Point2f向量（原地修改）
 * @param v 待压缩向量，status非零位置元素被保留
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
 * @brief 根据status状态压缩int向量（原地修改）
 * @param v 待压缩向量，status非零位置元素被保留
 */
void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

/** @brief 构造函数：初始化双目标志和特征点ID计数器 */
FeatureTracker::FeatureTracker()
{
    stereo_cam = 0;
    n_id = 0;
}

// =============================================
// 核心函数实现
// =============================================


/**
 * @brief 核心跟踪函数：偏振图像分解→每通道独立跟踪→新特征检测→全局池合并
 *
 * 流程：(1)raw2polar解码原始图像为多通道 (2)匹配器跟踪+边界检查
 *      (3)检测器补充新特征 (4)去畸变+速度计算 (5)全局特征池注册输出
 */
map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
FeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &/*_img1*/)
{
    cur_time = _cur_time;
    cur_img = _img;

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
    // 2. 每通道时序跟踪: prev → cur
    // =============================================
    for (auto& ch : channels) {
        if (ch.cur_img.empty()) continue;
        ch.cur_time = cur_time;

        // 2.1 匹配跟踪 (LK 光流)
        if (!ch.prev_pts.empty() && !ch.prev_img.empty()) {
            MatchResult mr = matcher_->track(
                ch.prev_img, ch.cur_img,
                ch.prev_pts, ch.local_ids, ch.track_cnt,
                {});

            ch.prev_pts = std::move(mr.prev_pts);
            ch.cur_pts  = std::move(mr.cur_pts);
            ch.local_ids = std::move(mr.ids);
            ch.track_cnt = std::move(mr.track_cnt);
        }

        // 2.2 边界检查: 剔除越界点
        if (!ch.cur_pts.empty()) {
            vector<uchar> status(ch.cur_pts.size(), 1);
            for (size_t i = 0; i < ch.cur_pts.size(); i++) {
                if (!inBorderImpl(ch, ch.cur_pts[i]))
                    status[i] = 0;
            }
            reduceVector(ch.prev_pts, status);
            reduceVector(ch.cur_pts, status);
            reduceVector(ch.local_ids, status);
            reduceVector(ch.track_cnt, status);
        }

        // 2.3 跟踪计数 +1
        for (auto &n : ch.track_cnt)
            n++;

        // 2.4 生成掩码: 已有特征周围禁止检测新点，保证空间均匀
        setMaskForChannel(ch);
    }

    // =============================================
    // 3. 新特征检测与补充
    // =============================================

    // 3.1 SuperPoint: 跨通道 Batch 推理
    if (detector_->name() == "SuperPoint") {
        std::vector<cv::Mat> batch_imgs, batch_masks;
        std::vector<int> batch_max_cnts;
        for (const auto& ch : channels) {
            if (ch.cur_img.empty()) continue;
            batch_imgs.push_back(ch.cur_img);
            batch_masks.push_back(ch.mask);
            batch_max_cnts.push_back(MAX_CNT - static_cast<int>(ch.cur_pts.size()));
        }
        auto* sp_det = dynamic_cast<SuperPointFeatureDetector*>(detector_.get());
        if (sp_det) {
            sp_det->detectBatchForChannels(batch_imgs, batch_masks, batch_max_cnts);
        }
    }
    // 3.2-3.4 新特征检测与补充
    for (auto& ch : channels) {
        if (ch.cur_img.empty()) continue;
        // 3.2 逐通道提取新特征 (SuperPoint 从 batch 缓存取; GFTT 直接 detect)
        int n_max_cnt = MAX_CNT - static_cast<int>(ch.cur_pts.size());
        if (n_max_cnt > 0 && !ch.mask.empty()) {
            ch.n_pts = detector_->detect(ch.cur_img, ch.mask, n_max_cnt);
        } else {
            ch.n_pts.clear();
        }
        // 3.3 新特征加入当前帧，分配局部唯一 ID
        for (auto &p : ch.n_pts) {
            ch.cur_pts.push_back(p);
            ch.local_ids.push_back(ch.next_local_id++);
            ch.track_cnt.push_back(1);
        }
    }

    // =============================================
    // 4. 坐标计算与状态更新 (供下一帧使用)
    // =============================================
    for (auto& ch : channels) {
        if (ch.cur_img.empty()) continue;

        // 4.1 去畸变到归一化平面
        ch.cur_un_pts = undistortedPts(ch.cur_pts, m_camera[0]);

        // 4.2 计算像素速度 (用于后端初始速度估计)
        map<int, cv::Point2f> cur_un_pts_map;
        for (size_t i = 0; i < ch.local_ids.size(); i++)
            cur_un_pts_map[ch.local_ids[i]] = ch.cur_un_pts[i];

        vector<cv::Point2f> vel;
        if (!ch.prev_un_pts_map.empty()) {
            double dt = ch.cur_time - ch.prev_time;
            for (size_t i = 0; i < ch.local_ids.size(); i++) {
                auto it = ch.prev_un_pts_map.find(ch.local_ids[i]);
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

        // 4.3 更新 prev 状态
        ch.prev_img = ch.cur_img.clone();
        ch.prev_pts = ch.cur_pts;
        ch.prev_un_pts = ch.cur_un_pts;
        ch.prev_un_pts_map = std::move(cur_un_pts_map);
        ch.prev_time = ch.cur_time;
        ch.prevLeftPtsMap.clear();
        for (size_t i = 0; i < ch.cur_pts.size(); i++)
            ch.prevLeftPtsMap[ch.local_ids[i]] = ch.cur_pts[i];
    }

    // =============================================
    // 5. 全局特征池注册与合并
    // =============================================
    global_pool_.beginFrame();
    global_pool_.propagateTracked(channels);
    global_pool_.registerUnboundFeatures(channels, MIN_DIST);
    auto featureFrame = global_pool_.buildFeatureFrame(channels);
    global_pool_.endFrame();

    // =============================================
    // 6. 可视化
    // =============================================
    if (SHOW_TRACK)
        drawTrackPolar();

    // =============================================
    // 7. 更新全局状态
    // =============================================
    if (!channels.empty() && !channels[0].cur_img.empty())
        cur_img = channels[0].cur_img;
    return featureFrame;
}




// =============================================
// 辅助函数实现
// =============================================

/** @brief 检测器和匹配器初始化
    根据全局配置（FEATURE_DETECTOR_TYPE/FEATURE_MATCHER_TYPE）创建 */
void FeatureTracker::initDetectorAndMatcher()
{
    DetectorConfig det_cfg;
    if (FEATURE_DETECTOR_TYPE == 2) {
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
    match_cfg.type = MatcherType::LK_FLOW;
    match_cfg.lk_win_size = 21;
    matcher_ = createMatcher(match_cfg);
    ROS_INFO("[PolarFP] Matcher: %s", matcher_->name().c_str());
}


/** @brief 设置启用的偏振通道列表 */
void FeatureTracker::setPolarChannels(const vector<string>& channel_names)
{
    channels.clear();
    for (const auto& name : channel_names) {
        channels.emplace_back(name);
    }
    ROS_INFO("[PolarFP] setPolarChannels: %zu channels", channels.size());
}

/** @brief 设置偏振通道滤波配置并打印参数信息 */
void FeatureTracker::setPolarFilterConfig(const PolarFilterConfig& cfg)
{
    polar_filter_cfg = cfg;
    if (cfg.filter_type == FILTER_GUIDED) {
        ROS_INFO("[PolarFP] Guided filter: radius=%d eps=%.4f",
                 cfg.guided_radius, cfg.guided_eps);
    }
}

/** @brief 从PolarChannelResult中提取指定通道（s0/dop/aopsin/aopcos）的8bit图像 */
cv::Mat FeatureTracker::getChannelImage(const PolarChannelResult& result, const string& channel)
{
    if (channel == "s0")      return result.S0_img.clone();
    if (channel == "dop")     return result.dop_img.clone();
    if (channel == "aopsin")  return result.sin_img.clone();
    if (channel == "aopcos")  return result.cos_img.clone();
    ROS_WARN("[PolarFP] unknown channel: %s", channel.c_str());
    return cv::Mat();
}

/** @brief 边界检查：判断点是否在通道图像内部（留1像素边距） */
bool FeatureTracker::inBorderImpl(const ChannelState& ch, const cv::Point2f& pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < ch.col - BORDER_SIZE &&
           BORDER_SIZE <= img_y && img_y < ch.row - BORDER_SIZE;
}

/** @brief 对单通道特征点施加mask：按跟踪帧数降序选取，保证空间均匀分布 */
void FeatureTracker::setMaskForChannel(ChannelState& ch)
{
    ch.mask = cv::Mat(ch.row, ch.col, CV_8UC1, cv::Scalar(255));

    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
    for (unsigned int i = 0; i < ch.cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(ch.track_cnt[i], make_pair(ch.cur_pts[i], ch.local_ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(),
         [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b) {
             return a.first > b.first;
         });

    ch.cur_pts.clear();
    ch.local_ids.clear();
    ch.track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (ch.mask.at<uchar>(it.second.first) == 255)
        {
            ch.cur_pts.push_back(it.second.first);
            ch.local_ids.push_back(it.second.second);
            ch.track_cnt.push_back(it.first);
            cv::circle(ch.mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

/** @brief 多通道跟踪可视化：单通道直接绘制，多通道按2x2网格拼接，颜色随跟踪帧数变化 */
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
                auto it = ch.prevLeftPtsMap.find(ch.local_ids[j]);
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
 * @brief 读取相机内参配置文件，生成相机模型（支持单目和双目）
 * @param calib_file 相机标定文件路径列表
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
 * @brief 将像素坐标通过相机模型投影到归一化平面
 * @param pts 待转换的像素坐标点集
 * @param cam 相机模型指针
 * @return 归一化坐标点集（消除相机内参影响）
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
 * @brief 移除后端判定为外点的特征
 * @param removePtsIds 需要移除的全局ID集合
 *
 * 将全局ID映射到各通道局部ID后，逐通道剔除对应特征点。
 */
void FeatureTracker::removeOutliers(set<int> &removePtsIds)
{
    auto global_to_local = global_pool_.getGlobalToLocalMap();

    map<string, set<int>> removeLocalIdsPerChannel;
    for (int gid : removePtsIds) {
        auto it = global_to_local.find(gid);
        if (it != global_to_local.end()) {
            for (const auto& [ch_name, local_id] : it->second) {
                removeLocalIdsPerChannel[ch_name].insert(local_id);
            }
        }
    }

    for (auto& ch : channels) {
        vector<uchar> status;
        for (size_t i = 0; i < ch.local_ids.size(); i++) {
            auto it = removeLocalIdsPerChannel.find(ch.name);
            if (it != removeLocalIdsPerChannel.end() && it->second.count(ch.local_ids[i]))
                status.push_back(0);
            else
                status.push_back(1);
        }
        reduceVector(ch.prev_pts, status);
        reduceVector(ch.local_ids, status);
        reduceVector(ch.track_cnt, status);
    }
}


/** @brief 返回跟踪可视化图像 */
cv::Mat FeatureTracker::getTrackImage()
{
    return imTrack;
}
