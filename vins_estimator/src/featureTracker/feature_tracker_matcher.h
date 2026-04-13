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

/** @brief 特征匹配器类型 */
enum class MatcherType { LK_FLOW, BRIEF_FLANN };

/** @brief 匹配器配置 */
struct MatcherConfig {
    MatcherType type = MatcherType::LK_FLOW;
    // LK flow params
    int lk_win_size = 21;
    int lk_max_level = 3;
    bool flow_back = true;
    double back_dist_thresh = 0.5;
    // BRIEF params
    int brief_bytes = 32;
    int flann_lsh_tables = 20;
    int flann_lsh_key_size = 20;
    int flann_multi_probe = 20;
    float brief_match_dist_ratio = 0.75f;
};

/** @brief 匹配结果 */
struct MatchResult {
    std::vector<cv::Point2f> prev_pts;
    std::vector<cv::Point2f> cur_pts;
    std::vector<int> ids;
    std::vector<int> track_cnt;
};

/**
 * @class FeatureMatcher
 * @brief 特征匹配器抽象接口
 */
class FeatureMatcher {
public:
    virtual ~FeatureMatcher() = default;
    virtual std::string name() const = 0;
    virtual MatchResult track(
        const cv::Mat& prev_img, const cv::Mat& cur_img,
        const std::vector<cv::Point2f>& prev_pts,
        const std::vector<int>& prev_ids,
        const std::vector<int>& prev_track_cnt,
        const std::vector<uchar>& prev_desc) = 0;
    /**
     * @brief 提取描述子 (仅 BRIEF 模式需要，LK 返回空向量)
     * @param image 当前图像
     * @param pts 特征点位置
     * @return 扁平描述子字节: [feat0_desc, feat1_desc, ...]
     */
    virtual std::vector<uchar> extractDescriptors(
        const cv::Mat& image, const std::vector<cv::Point2f>& pts) const;
};

/**
 * @class LKFlowMatcher
 * @brief Lucas-Kanade 光流匹配器 (含双向检查)
 *
 * 等价于原版 VINS 的 calcOpticalFlowPyrLK + reverse check。
 */
class LKFlowMatcher : public FeatureMatcher {
public:
    LKFlowMatcher(int win_size, int max_level, bool flow_back,
                  double back_dist_thresh)
        : win_size_(win_size), max_level_(max_level),
          flow_back_(flow_back), back_dist_thresh_(back_dist_thresh) {}
    std::string name() const override { return "LK_FLOW"; }
    MatchResult track(
        const cv::Mat& prev_img, const cv::Mat& cur_img,
        const std::vector<cv::Point2f>& prev_pts,
        const std::vector<int>& prev_ids,
        const std::vector<int>& prev_track_cnt,
        const std::vector<uchar>& prev_desc) override;
private:
    int win_size_;
    int max_level_;
    bool flow_back_;
    double back_dist_thresh_;
};

/**
 * @class BRIEFFLANNMatcher
 * @brief BRIEF 描述子 + FLANN LSH 匹配器
 *
 * 提取 BRIEF (或 ORB fallback) 二进制描述子，使用 FLANN LSH index
 * 进行快速 Hamming 距离匹配。
 */
class BRIEFFLANNMatcher : public FeatureMatcher {
public:
    BRIEFFLANNMatcher(int brief_bytes, int flann_lsh_tables,
                      int flann_lsh_key_size, int flann_multi_probe,
                      float match_dist_ratio);
    std::string name() const override { return "BRIEF_FLANN"; }
    MatchResult track(
        const cv::Mat& prev_img, const cv::Mat& cur_img,
        const std::vector<cv::Point2f>& prev_pts,
        const std::vector<int>& prev_ids,
        const std::vector<int>& prev_track_cnt,
        const std::vector<uchar>& prev_desc) override;
    std::vector<uchar> extractDescriptors(
        const cv::Mat& image, const std::vector<cv::Point2f>& pts) const override;
private:
    int brief_bytes_;
    int flann_lsh_tables_;
    int flann_lsh_key_size_;
    int flann_multi_probe_;
    float match_dist_ratio_;

    // BRIEF 提取器 (xfeatures2d 或 ORB fallback)
    cv::Ptr<cv::Feature2D> brief_extractor_;
};

/** @brief 工厂函数：根据配置创建匹配器 */
std::shared_ptr<FeatureMatcher> createMatcher(const MatcherConfig& cfg);
