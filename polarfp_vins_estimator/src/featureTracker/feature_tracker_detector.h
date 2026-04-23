/*******************************************************
 * Copyright (C) 2026, Bionic Intelligence Research Team, Beihang University
 *
 * This file is part of PolarFP-VINS.
 * Auther: Wei Xinran (github.com/weixr18; weixr0605@sina.com)
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/
 
#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>

/** @brief 特征检测器类型 */
enum class DetectorType { GFTT, SUPERPOINT };

/** @brief 检测器配置 */
struct DetectorConfig {
    DetectorType type = DetectorType::GFTT;
    // GFTT params
    double gftt_quality = 0.01;
    int min_dist = 20;
    // SuperPoint params
    std::string sp_model_path;
    bool sp_use_gpu = true;
    float sp_keypoint_threshold = 0.015f;
    int sp_nms_radius = 4;
};

/**
 * @class FeatureDetector
 * @brief 特征检测器抽象接口
 */
class FeatureDetector {
public:
    virtual ~FeatureDetector() = default;
    virtual std::string name() const = 0;
    virtual std::vector<cv::Point2f> detect(
        const cv::Mat& image, const cv::Mat& mask, int max_cnt) const = 0;
};

/**
 * @class GFTTDetector
 * @brief Shi-Tomasi (Good Features To Track) 检测器
 *
 * 直接包装 cv::goodFeaturesToTrack，与原版 VINS 行为一致。
 */
class GFTTDetector : public FeatureDetector {
public:
    GFTTDetector(double quality_level, int min_distance)
        : quality_level_(quality_level), min_distance_(min_distance) {}
    std::string name() const override { return "GFTT"; }
    std::vector<cv::Point2f> detect(
        const cv::Mat& image, const cv::Mat& mask, int max_cnt) const override;
private:
    double quality_level_;
    int min_distance_;
};

/** @brief 工厂函数：根据配置创建检测器 */
std::shared_ptr<FeatureDetector> createDetector(const DetectorConfig& cfg);
