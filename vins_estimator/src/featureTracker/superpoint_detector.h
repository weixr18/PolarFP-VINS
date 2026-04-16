/*******************************************************
 * Copyright (C) 2026, Bionic Intelligence Research Team, Beihang University
 *
 * This file is part of PolarFP-VINS.
 * Auther: Wei Xinran (github.com/weixr18; weixr0605@sina.com)
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include "feature_tracker_detector.h"
#include <memory>
#include <string>
#include <vector>

// --- Forward declaration (PIMPL) ---
// Keeps <torch/script.h> out of headers to avoid c10::nullopt vs std::nullopt conflicts.
struct SuperPointDetectorImpl;

/**
 * @class SuperPointFeatureDetector
 * @brief SuperPoint feature detector wrapped as FeatureDetector.
 *
 * Supports both single-image detect() and batch detectBatchForChannels().
 * In batch mode: all channel images are inferred at once, then per-channel
 * detect() calls return cached results.
 * In single-image mode (fallback for non-polar): performs single-image inference.
 */
class SuperPointFeatureDetector : public FeatureDetector {
public:
    SuperPointFeatureDetector(const std::string& model_path, bool use_gpu,
                              float kp_thresh, int nms_radius);
    ~SuperPointFeatureDetector();

    std::string name() const override { return "SuperPoint"; }

    // FeatureDetector interface: per-channel detect
    // Batch mode: returns cached result from detectBatchForChannels
    // Fallback: single-image inference + mask filter + max_cnt truncation
    std::vector<cv::Point2f> detect(const cv::Mat& image, const cv::Mat& mask,
                                    int max_cnt) const override;

    // Batch inference: all channels at once, stores results internally
    void detectBatchForChannels(
        const std::vector<cv::Mat>& images,
        const std::vector<cv::Mat>& masks,
        const std::vector<int>& max_cnts);

private:
    // Internal single-image detection
    std::vector<cv::Point2f> detectSingleImage(const cv::Mat& image) const;

    // PIMPL: isolates <torch/script.h>
    std::unique_ptr<SuperPointDetectorImpl> impl_;
    bool initialized_ = false;

    // Batch inference cache
    mutable bool batch_cached_ = false;
    mutable std::vector<std::vector<cv::Point2f>> batch_results_;
    mutable size_t current_channel_idx_ = 0;

    // Config
    float keypoint_threshold_;
    int nms_radius_;
};
