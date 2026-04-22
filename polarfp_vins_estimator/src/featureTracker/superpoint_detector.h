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
 * @brief SuperPoint特征检测器，继承自FeatureDetector
 *
 * 支持单张图像detect()和批量detectBatchForChannels()两种模式。
 * 批量模式下：所有通道图像一次性推理，之后per-channel detect()返回缓存结果。
 * 单图像模式（非偏振回退）：执行单张图像推理。
 */
class SuperPointFeatureDetector : public FeatureDetector {
public:
    SuperPointFeatureDetector(const std::string& model_path, bool use_gpu,
                              float kp_thresh, int nms_radius);
    ~SuperPointFeatureDetector();

    std::string name() const override { return "SuperPoint"; }

    // FeatureDetector接口：per-channel detect
    // 批量模式：返回detectBatchForChannels的缓存结果
    // 回退模式：单张推理 + mask过滤 + max_cnt截断
    std::vector<cv::Point2f> detect(const cv::Mat& image, const cv::Mat& mask,
                                    int max_cnt) const override;

    // 批量推理：所有通道一次性推理，内部存储结果
    void detectBatchForChannels(
        const std::vector<cv::Mat>& images,
        const std::vector<cv::Mat>& masks,
        const std::vector<int>& max_cnts);

private:
    // 内部单张图像检测
    std::vector<cv::Point2f> detectSingleImage(const cv::Mat& image) const;

    // PIMPL：隔离<torch/script.h>
    std::unique_ptr<SuperPointDetectorImpl> impl_;
    bool initialized_ = false;

    // 批量推理缓存
    mutable bool batch_cached_ = false;
    mutable std::vector<std::vector<cv::Point2f>> batch_results_;
    mutable size_t current_channel_idx_ = 0;

    // Config
    float keypoint_threshold_;
    int nms_radius_;
};
