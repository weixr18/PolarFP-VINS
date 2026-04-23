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

/**
 * @brief GFTT检测器实现：调用cv::goodFeaturesToTrack提取角点
 * @param image 输入图像（单通道8bit）
 * @param mask 掩码图像，仅在非零区域检测
 * @param max_cnt 最大返回特征点数
 * @return 检测到的特征点坐标列表
 */
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

/**
 * @brief 工厂函数：根据配置创建对应的特征检测器
 * @param cfg 检测器配置（包含类型及对应参数）
 * @return 具体检测器实例（GFTT/FAST/SuperPoint）
 */
std::shared_ptr<FeatureDetector> createDetector(const DetectorConfig& cfg)
{
    switch (cfg.type) {
        case DetectorType::GFTT:
            return std::make_shared<GFTTDetector>(cfg.gftt_quality, cfg.min_dist);
        case DetectorType::SUPERPOINT:
            return std::make_shared<SuperPointFeatureDetector>(
                cfg.sp_model_path, cfg.sp_use_gpu,
                cfg.sp_keypoint_threshold, cfg.sp_nms_radius);
        default:
            return std::make_shared<GFTTDetector>(cfg.gftt_quality, cfg.min_dist);
    }
}
