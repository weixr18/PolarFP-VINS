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

// BRIEF is in opencv_contrib (xfeatures2d). Try to include it; fall back to ORB.
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
    // ORB fallback: also produces binary descriptors compatible with Hamming/FLANN-LSH
    brief_extractor_ = cv::ORB::create(5000, 1.2f, 8, 31, 0, 2,
                                        cv::ORB::HARRIS_SCORE, 31, 20);
    // Note: ORB uses 32 bytes (256 bits), same as BriefDescriptorExtractor::BYTES
#endif
}

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
