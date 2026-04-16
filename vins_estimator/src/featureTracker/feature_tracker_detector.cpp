/*******************************************************
 * Copyright (C) 2026, Bionic Intelligence Research Team, Beihang University
 *
 * This file is part of PolarFP-VINS.
 * Auther: Wei Xinran (github.com/weixr18; weixr0605@sina.com)
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "feature_tracker_detector.h"
#include "superpoint_detector.h"

std::vector<cv::Point2f> GFTTDetector::detect(
    const cv::Mat& image, const cv::Mat& mask, int max_cnt) const
{
    std::vector<cv::Point2f> pts;
    if (max_cnt <= 0 || image.empty())
        return pts;

    cv::goodFeaturesToTrack(image, pts, max_cnt,
                            quality_level_, min_distance_, mask);
    return pts;
}

std::vector<cv::Point2f> FASTDetector::detect(
    const cv::Mat& image, const cv::Mat& mask, int max_cnt) const
{
    std::vector<cv::Point2f> pts;
    if (max_cnt <= 0 || image.empty())
        return pts;

    std::vector<cv::KeyPoint> keypoints;
    cv::FAST(image, keypoints, threshold_, nonmax_suppression_);

    // Mask 过滤
    if (!mask.empty()) {
        std::vector<cv::KeyPoint> masked;
        masked.reserve(keypoints.size());
        for (const auto& kp : keypoints) {
            int x = cvRound(kp.pt.x);
            int y = cvRound(kp.pt.y);
            if (x >= 0 && x < image.cols && y >= 0 && y < image.rows
                && mask.at<uchar>(y, x) > 0) {
                masked.push_back(kp);
            }
        }
        keypoints = std::move(masked);
    }

    // 按响应值降序排序
    std::sort(keypoints.begin(), keypoints.end(),
              [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                  return a.response > b.response;
              });

    // top-N 截断
    if (static_cast<int>(keypoints.size()) > max_cnt)
        keypoints.resize(max_cnt);

    pts.reserve(keypoints.size());
    for (const auto& kp : keypoints)
        pts.push_back(kp.pt);

    return pts;
}

std::shared_ptr<FeatureDetector> createDetector(const DetectorConfig& cfg)
{
    switch (cfg.type) {
        case DetectorType::GFTT:
            return std::make_shared<GFTTDetector>(cfg.gftt_quality, cfg.min_dist);
        case DetectorType::FAST:
            return std::make_shared<FASTDetector>(cfg.fast_threshold, cfg.fast_nonmax);
        case DetectorType::SUPERPOINT:
            return std::make_shared<SuperPointFeatureDetector>(
                cfg.sp_model_path, cfg.sp_use_gpu,
                cfg.sp_keypoint_threshold, cfg.sp_nms_radius);
        default:
            return std::make_shared<GFTTDetector>(cfg.gftt_quality, cfg.min_dist);
    }
}
