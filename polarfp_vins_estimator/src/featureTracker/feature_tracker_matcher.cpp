/*******************************************************
 * Copyright (C) 2026, Bionic Intelligence Research Team, Beihang University
 *
 * This file is part of PolarFP-VINS.
 * Auther: Wei Xinran (github.com/weixr18; weixr0605@sina.com)
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "feature_tracker_matcher.h"
#include <opencv2/features2d.hpp>

// BRIEF定义在opencv_contrib(xfeatures2d)中，尝试包含；否则回退到ORB
#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d.hpp>
#endif

// ============================================================
// FeatureMatcher base
// ============================================================

std::vector<uchar> FeatureMatcher::extractDescriptors(
    const cv::Mat& /*image*/, const std::vector<cv::Point2f>& /*pts*/) const
{
    return {};  // LK doesn't need descriptors
}

// ============================================================
// LKFlowMatcher
// ============================================================

/**
 * @brief LK光流匹配器跟踪实现：前向光流 + 可选反向检查，剔除误匹配
 * @param prev_img 上一帧图像
 * @param cur_img 当前帧图像
 * @param prev_pts 上一帧特征点
 * @param prev_ids 上一帧特征点ID
 * @param prev_track_cnt 上一帧跟踪计数
 * @param prev_desc 上一帧描述子（LK模式未使用）
 * @return 匹配结果（前后帧对应点、ID、跟踪计数）
 */
MatchResult LKFlowMatcher::track(
    const cv::Mat& prev_img, const cv::Mat& cur_img,
    const std::vector<cv::Point2f>& prev_pts,
    const std::vector<int>& prev_ids,
    const std::vector<int>& prev_track_cnt,
    const std::vector<uchar>& /*prev_desc*/)
{
    MatchResult result;
    if (prev_pts.empty() || prev_img.empty() || cur_img.empty())
        return result;

    // Forward LK flow
    std::vector<uchar> status;
    std::vector<float> err;
    std::vector<cv::Point2f> cur_pts;
    cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts,
                             status, err, cv::Size(win_size_, win_size_),
                             max_level_);

    // Backward check
    if (flow_back_) {
        std::vector<uchar> reverse_status;
        std::vector<cv::Point2f> reverse_pts = prev_pts;
        cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts,
                                 reverse_status, err,
                                 cv::Size(win_size_, win_size_), 1,
                                 cv::TermCriteria(cv::TermCriteria::COUNT +
                                                  cv::TermCriteria::EPS, 30, 0.01),
                                 cv::OPTFLOW_USE_INITIAL_FLOW);
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i] && reverse_status[i]) {
                double dx = prev_pts[i].x - reverse_pts[i].x;
                double dy = prev_pts[i].y - reverse_pts[i].y;
                if (dx * dx + dy * dy <= back_dist_thresh_ * back_dist_thresh_)
                    status[i] = 1;
                else
                    status[i] = 0;
            } else {
                status[i] = 0;
            }
        }
    }

    // Compact
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) {
            result.prev_pts.push_back(prev_pts[i]);
            result.cur_pts.push_back(cur_pts[i]);
            result.ids.push_back(prev_ids[i]);
            result.track_cnt.push_back(prev_track_cnt[i]);
        }
    }

    return result;
}

// ============================================================
// BRIEFFLANNMatcher
// ============================================================

/**
 * @brief BRIEF+FLANN匹配器构造函数：初始化BRIEF描述子提取器（或ORB回退）
 * @param brief_bytes 描述子字节数
 * @param flann_lsh_tables FLANN LSH哈希表数量
 * @param flann_lsh_key_size 哈希键大小
 * @param flann_multi_probe 多探测层级
 * @param match_dist_ratio 匹配距离比率阈值（Lowe's ratio test）
 */
BRIEFFLANNMatcher::BRIEFFLANNMatcher(
    int brief_bytes, int flann_lsh_tables,
    int flann_lsh_key_size, int flann_multi_probe,
    float match_dist_ratio)
    : brief_bytes_(brief_bytes), flann_lsh_tables_(flann_lsh_tables),
      flann_lsh_key_size_(flann_lsh_key_size),
      flann_multi_probe_(flann_multi_probe),
      match_dist_ratio_(match_dist_ratio)
{
#ifdef HAVE_OPENCV_XFEATURES2D
    brief_extractor_ = cv::xfeatures2d::BriefDescriptorExtractor::create(brief_bytes_);
#else
    // ORB回退：也产生二进制描述子，兼容Hamming/FLANN-LSH
    brief_extractor_ = cv::ORB::create(5000, 1.2f, 8, 31, 0, 2,
                                        cv::ORB::HARRIS_SCORE, 31, 20);
    // 注意：ORB使用32字节（256位），与BriefDescriptorExtractor::BYTES相同
#endif
}

/**
 * @brief BRIEF+FLANN匹配器跟踪实现：检测当前帧关键点，提取描述子，
 *        构建FLANN-LSH索引后用Hamming距离 + ratio test进行匹配
 * @param prev_img 上一帧图像（未使用）
 * @param cur_img 当前帧图像
 * @param prev_pts 上一帧特征点
 * @param prev_ids 上一帧特征点ID
 * @param prev_track_cnt 上一帧跟踪计数
 * @param prev_desc 上一帧扁平描述子字节数组
 * @return 匹配结果
 */
MatchResult BRIEFFLANNMatcher::track(
    const cv::Mat& /*prev_img*/, const cv::Mat& cur_img,
    const std::vector<cv::Point2f>& prev_pts,
    const std::vector<int>& prev_ids,
    const std::vector<int>& prev_track_cnt,
    const std::vector<uchar>& prev_desc)
{
    MatchResult result;
    if (prev_pts.empty() || prev_desc.empty() || cur_img.empty())
        return result;

    // Number of previous features
    int n_prev = static_cast<int>(prev_pts.size());
    if (static_cast<int>(prev_desc.size()) != n_prev * brief_bytes_)
        return result;

    // 1. Detect keypoints in current image
    std::vector<cv::KeyPoint> cur_kps;
    cv::FAST(cur_img, cur_kps, 20, true);
    if (cur_kps.empty())
        return result;

    // 2. Compute descriptors for current keypoints
    cv::Mat cur_desc_mat;
    brief_extractor_->compute(cur_img, cur_kps, cur_desc_mat);
    if (cur_desc_mat.empty())
        return result;

    // 3. Build FLANN LSH index from current descriptors
    // FLANN LSH params: table_number, key_size, multi_probe_level
    cv::flann::Index flann_index(
        cur_desc_mat,
        cv::flann::LshIndexParams(flann_lsh_tables_,
                                   flann_lsh_key_size_,
                                   flann_multi_probe_),
        cvflann::FLANN_DIST_HAMMING);

    // 4. Build query matrix from previous descriptors
    cv::Mat query_desc(n_prev, brief_bytes_, CV_8UC1, const_cast<uchar*>(prev_desc.data()));

    // 5. kNN search with k=2 for ratio test
    cv::Mat indices(n_prev, 2, CV_32SC1);
    cv::Mat dists(n_prev, 2, CV_32FC1);
    flann_index.knnSearch(query_desc, indices, dists, 2,
                          cv::flann::SearchParams(32));

    // 6. Match with ratio test
    for (int i = 0; i < n_prev; i++) {
        int idx1 = indices.at<int>(i, 0);
        float d1 = dists.at<float>(i, 0);
        float d2 = dists.at<float>(i, 1);

        if (idx1 < 0 || idx1 >= static_cast<int>(cur_kps.size()))
            continue;
        if (d2 < 1e-5f) continue;  // avoid division by zero

        float ratio = d1 / d2;
        if (ratio < match_dist_ratio_) {
            result.prev_pts.push_back(prev_pts[i]);
            result.cur_pts.push_back(cur_kps[idx1].pt);
            result.ids.push_back(prev_ids[i]);
            result.track_cnt.push_back(prev_track_cnt[i]);
        }
    }

    return result;
}

/**
 * @brief 提取BRIEF/ORB二进制描述子：将Point2f转为KeyPoint后调用提取器
 * @param image 当前帧图像
 * @param pts 特征点坐标列表
 * @return 扁平描述子字节数组 [feat0_desc, feat1_desc, ...]
 */
std::vector<uchar> BRIEFFLANNMatcher::extractDescriptors(
    const cv::Mat& image, const std::vector<cv::Point2f>& pts) const
{
    std::vector<uchar> desc_flat;
    if (pts.empty() || image.empty())
        return desc_flat;

    // Convert points to keypoints
    std::vector<cv::KeyPoint> kps;
    kps.reserve(pts.size());
    for (const auto& p : pts)
        kps.emplace_back(p, 31.f);  // 31px patch size for BRIEF

    cv::Mat desc_mat;
    brief_extractor_->compute(image, kps, desc_mat);

    if (desc_mat.empty())
        return desc_flat;

    // Flatten: each row is a descriptor of brief_bytes_ bytes
    int n = std::min(static_cast<int>(pts.size()), desc_mat.rows);
    desc_flat.assign(desc_mat.ptr<uchar>(), desc_mat.ptr<uchar>() + n * brief_bytes_);

    return desc_flat;
}

// ============================================================
// Factory
// ============================================================

/**
 * @brief 工厂函数：根据配置创建对应的特征匹配器
 * @param cfg 匹配器配置（包含类型及对应参数）
 * @return 具体匹配器实例（LK_FLOW/BRIEF_FLANN）
 */
std::shared_ptr<FeatureMatcher> createMatcher(const MatcherConfig& cfg)
{
    switch (cfg.type) {
        case MatcherType::LK_FLOW:
            return std::make_shared<LKFlowMatcher>(
                cfg.lk_win_size, cfg.lk_max_level,
                cfg.flow_back, cfg.back_dist_thresh);
        case MatcherType::BRIEF_FLANN:
            return std::make_shared<BRIEFFLANNMatcher>(
                cfg.brief_bytes, cfg.flann_lsh_tables,
                cfg.flann_lsh_key_size, cfg.flann_multi_probe,
                cfg.brief_match_dist_ratio);
        default:
            return std::make_shared<LKFlowMatcher>(
                cfg.lk_win_size, cfg.lk_max_level,
                cfg.flow_back, cfg.back_dist_thresh);
    }
}
