/*******************************************************
 * Copyright (C) 2026, Bionic Intelligence Research Team, Beihang University
 *
 * This file is part of PolarFP-VINS.
 * Auther: Wei Xinran (github.com/weixr18; weixr0605@sina.com)
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "feature_tracker_matcher.h"

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
// Factory
// ============================================================

/**
 * @brief 工厂函数：创建LK光流匹配器
 */
std::shared_ptr<FeatureMatcher> createMatcher(const MatcherConfig& cfg)
{
    return std::make_shared<LKFlowMatcher>(
        cfg.lk_win_size, cfg.lk_max_level,
        cfg.flow_back, cfg.back_dist_thresh);
}
