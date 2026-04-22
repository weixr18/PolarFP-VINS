/*******************************************************
 * Copyright (C) 2026, Bionic Intelligence Research Team, Beihang University
 *
 * This file is part of PolarFP-VINS.
 * Auther: Wei Xinran (github.com/weixr18; weixr0605@sina.com)
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

/**
 * @file PolarChannel.cpp
 * @brief PolarChannel.h 的实现文件。
 *
 * 负责从偏振相机的 2x2 微偏振片阵列原始图像中解码出：
 *   - S0（总强度）、DoP（偏振度）、AoP（偏振角）
 *   - sin(AoP)、cos(AoP) 分量（量化为 8bit）
 *   - 彩色可视化图像
 *
 * 偏振信息在暗光条件下比强度信息更稳定，是本系统鲁棒跟踪的核心。
 *
 * Version: 2026.3.6
 */

#include "PolarChannel.h"
#include <vector>
#include <algorithm>

/** @brief 盒滤波辅助函数，使用BORDER_REPLICATE处理边界 */
static cv::Mat boxFilter2D(const cv::Mat& src, int radius) {
    cv::Mat dst;
    cv::boxFilter(src, dst, -1, cv::Size(radius, radius), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    return dst;
}

/** @brief 单通道导向滤波实现（基于均值滤波的局部线性模型） */
cv::Mat guidedFilterSingle(const cv::Mat& I, const cv::Mat& p, int r, double eps) {
    cv::Mat mean_I  = boxFilter2D(I, 2 * r + 1);
    cv::Mat mean_p  = boxFilter2D(p, 2 * r + 1);
    cv::Mat mean_II = boxFilter2D(I.mul(I), 2 * r + 1);
    cv::Mat mean_Ip = boxFilter2D(I.mul(p), 2 * r + 1);
    cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    cv::Mat var_I  = mean_II - mean_I.mul(mean_I);
    cv::Mat a = cov_Ip / (var_I + eps);
    cv::Mat b = mean_p - a.mul(mean_I);
    cv::Mat mean_a = boxFilter2D(a, 2 * r + 1);
    cv::Mat mean_b = boxFilter2D(b, 2 * r + 1);
    return mean_a.mul(I) + mean_b;
}

/** @brief 将4个原始通道按BT.601系数融合为灰度图，输出经双三次插值放大2x */
cv::Mat _raw_chnl_to_gray(const std::vector<cv::Mat>& img_xx) {
    int itp = cv::INTER_CUBIC;
    const cv::Mat& img_xx_r = img_xx[0];
    const cv::Mat& img_xx_g1 = img_xx[1];
    const cv::Mat& img_xx_g2 = img_xx[2];
    const cv::Mat& img_xx_b = img_xx[3];
    int h_4 = img_xx_r.rows;
    int w_4 = img_xx_r.cols;
    int h_half = h_4 * 2;
    int w_half = w_4 * 2;
    cv::Mat img_xx_r_resized, img_xx_g1_resized, img_xx_g2_resized, img_xx_b_resized;
    cv::resize(img_xx_r, img_xx_r_resized, cv::Size(w_half, h_half), 0, 0, itp);
    cv::resize(img_xx_g1, img_xx_g1_resized, cv::Size(w_half, h_half), 0, 0, itp);
    cv::resize(img_xx_g2, img_xx_g2_resized, cv::Size(w_half, h_half), 0, 0, itp);
    cv::resize(img_xx_b, img_xx_b_resized, cv::Size(w_half, h_half), 0, 0, itp);
    double k_r = 0.299, k_g = 0.587, k_b = 0.114;
    cv::Mat img_xx_gray;
    img_xx_r_resized.convertTo(img_xx_gray, CV_64F);
    img_xx_gray *= k_r;
    cv::Mat temp;
    img_xx_g1_resized.convertTo(temp, CV_64F);
    img_xx_gray += temp * k_g * 0.5;
    img_xx_g2_resized.convertTo(temp, CV_64F);
    img_xx_gray += temp * k_g * 0.5;
    img_xx_b_resized.convertTo(temp, CV_64F);
    img_xx_gray += temp * k_b;
    return img_xx_gray;
}





/** @brief 核心函数：从2x2微偏振阵列原始图像解码Stokes参数（S0/DoP/AoP）及可视化 */
PolarChannelResult raw2polar(const cv::Mat& img_raw, const PolarFilterConfig& cfg) {
    assert(img_raw.channels() == 1 && img_raw.dims == 2);

    int new_rows = img_raw.rows / 4;
    int new_cols = img_raw.cols / 4;

    // 辅助 lambda：从超像素网格中按两对偏移量采样 4 个通道，返回灰度图
    auto sample_polar_channels = [&](
        int row_offset1, int col_offset1, int row_offset2, int col_offset2
    ) -> cv::Mat {
        cv::Mat r_channel(new_rows, new_cols, img_raw.type());
        cv::Mat g1_channel(new_rows, new_cols, img_raw.type());
        cv::Mat g2_channel(new_rows, new_cols, img_raw.type());
        cv::Mat b_channel(new_rows, new_cols, img_raw.type());
        for (int i = 0; i < new_rows; i++) {
            for (int j = 0; j < new_cols; j++) {
                r_channel.at<uchar>(i, j) = img_raw.at<uchar>(i * 4 + row_offset1, j * 4 + col_offset1);
                g1_channel.at<uchar>(i, j) = img_raw.at<uchar>(i * 4 + row_offset1, j * 4 + col_offset2);
                g2_channel.at<uchar>(i, j) = img_raw.at<uchar>(i * 4 + row_offset2, j * 4 + col_offset1);
                b_channel.at<uchar>(i, j) = img_raw.at<uchar>(i * 4 + row_offset2, j * 4 + col_offset2);
            }
        }
        std::vector<cv::Mat> channels = {r_channel, g1_channel, g2_channel, b_channel};
        return _raw_chnl_to_gray(channels);
    };

    // 采样 4 个偏振角度的子图像
    cv::Mat img_90_gray = sample_polar_channels(0, 0, 2, 2);    // 90°
    cv::Mat img_45_gray = sample_polar_channels(0, 1, 2, 3);    // 45°
    cv::Mat img_135_gray = sample_polar_channels(1, 0, 3, 2);  // 135°
    cv::Mat img_0_gray = sample_polar_channels(1, 1, 3, 3);      // 0°

    // 计算 Stokes 向量
    cv::Mat S_0 = (img_90_gray + img_45_gray + img_135_gray + img_0_gray) / 4.0;
    cv::Mat S_1 = img_0_gray - img_90_gray;
    cv::Mat S_2 = img_45_gray - img_135_gray;

    // 计算 sin(AoP) 和 cos(AoP)，分母加 EPSILON 防止除零
    cv::Mat denominator;
    cv::sqrt(S_1.mul(S_1) + S_2.mul(S_2) + EPSILON, denominator);
    cv::Mat sinaop = S_2 / denominator;
    cv::Mat cosaop = S_1 / denominator;

    // 计算 AoP = 0.5 * atan2(S_2, S_1)，范围 [-π/2, π/2]
    cv::Mat aop;
    cv::phase(S_1, S_2, aop);
    aop *= 0.5;

    // 将 AoP 可视化：映射为 HSV 色相环，再转 RGB
    cv::Mat aop_vis_hsv(img_90_gray.size(), CV_8UC3);
    for (int i = 0; i < aop_vis_hsv.rows; i++) {
        for (int j = 0; j < aop_vis_hsv.cols; j++) {
            double aop_val = aop.at<double>(i, j);
            // 将 [-π/2, π/2] 映射到 [0, 180] 作为 Hue
            float hue = static_cast<float>((aop_val + CV_PI/2) * 180.0 / CV_PI);
            aop_vis_hsv.at<cv::Vec3b>(i, j) = cv::Vec3b(
                static_cast<uchar>(hue),      // H
                180,                          // S
                255                           // V
            );
        }
    }
    cv::Mat aop_vis;
    cv::cvtColor(aop_vis_hsv, aop_vis, cv::COLOR_HSV2RGB);

    // 计算 DoP = sqrt(S1²+S2²) / S0
    cv::Mat dop;
    cv::sqrt(S_1.mul(S_1) + S_2.mul(S_2) + EPSILON, dop);
    dop /= (S_0 + EPSILON);

    // 处理无效区域（S0 接近 0 的地方置零）
    cv::Mat MASK = cv::abs(S_0) < EPSILON;
    sinaop.setTo(0.0, MASK);
    cosaop.setTo(0.0, MASK);
    dop.setTo(0.0, MASK);

    // DoP mask：去除异常高值
    cv::Mat MASK_2 = dop > 0.999;
    sinaop.setTo(0.0, MASK_2);
    cosaop.setTo(0.0, MASK_2);
    dop.setTo(0.0, MASK_2);

    // 量化输出为 8bit 图像
    cv::Mat dop_img, sin_img, cos_img, S0_img;
    dop.convertTo(dop_img, CV_8U, 255.0);           // DoP: [0,1] → [0,255]
    sinaop.convertTo(sin_img, CV_8U, 127.5, 127.5); // sin: [-1,1] → [0,255]
    cosaop.convertTo(cos_img, CV_8U, 127.5, 127.5); // cos: [-1,1] → [0,255]
    S_0.convertTo(S0_img, CV_8U);                   // S0: 直接截断

    // 可选：对 DoP/sin/cos 施加滤波，降低低光照噪声
    if (cfg.filter_type == FILTER_BILATERAL) {
        cv::Mat dop_filt, sin_filt, cos_filt;
        cv::bilateralFilter(dop_img, dop_filt, cfg.bilateral_d, cfg.bilateral_sigmaColor, cfg.bilateral_sigmaSpace);
        cv::bilateralFilter(sin_img, sin_filt, cfg.bilateral_d, cfg.bilateral_sigmaColor, cfg.bilateral_sigmaSpace);
        cv::bilateralFilter(cos_img, cos_filt, cfg.bilateral_d, cfg.bilateral_sigmaColor, cfg.bilateral_sigmaSpace);
        dop_img = dop_filt;
        sin_img = sin_filt;
        cos_img = cos_filt;
    } else if (cfg.filter_type == FILTER_GUIDED) {
        cv::Mat dop_f, sin_f, cos_f, s0_f;
        dop_img.convertTo(dop_f, CV_64F, 1.0 / 255.0);
        sin_img.convertTo(sin_f, CV_64F, 1.0 / 255.0);
        cos_img.convertTo(cos_f, CV_64F, 1.0 / 255.0);
        S0_img.convertTo(s0_f, CV_64F, 1.0 / 255.0);

        cv::Mat dop_g = guidedFilterSingle(s0_f, dop_f, cfg.guided_radius, cfg.guided_eps);
        cv::Mat sin_g = guidedFilterSingle(s0_f, sin_f, cfg.guided_radius, cfg.guided_eps);
        cv::Mat cos_g = guidedFilterSingle(s0_f, cos_f, cfg.guided_radius, cfg.guided_eps);

        dop_g.convertTo(dop_img, CV_8U, 255.0);
        sin_g.convertTo(sin_img, CV_8U, 255.0);
        cos_g.convertTo(cos_img, CV_8U, 255.0);
    } else if (cfg.filter_type == FILTER_NLM) {
        cv::Mat dop_nlm, sin_nlm, cos_nlm;
        cv::fastNlMeansDenoising(dop_img, dop_nlm, cfg.nlm_h, cfg.nlm_template, cfg.nlm_search);
        cv::fastNlMeansDenoising(sin_img, sin_nlm, cfg.nlm_h, cfg.nlm_template, cfg.nlm_search);
        cv::fastNlMeansDenoising(cos_img, cos_nlm, cfg.nlm_h, cfg.nlm_template, cfg.nlm_search);
        dop_img = dop_nlm;
        sin_img = sin_nlm;
        cos_img = cos_nlm;
    } else if (cfg.filter_type == FILTER_MEDIAN) {
        cv::Mat dop_median, sin_median, cos_median;
        cv::medianBlur(dop_img, dop_median, cfg.median_kernel_size);
        cv::medianBlur(sin_img, sin_median, cfg.median_kernel_size);
        cv::medianBlur(cos_img, cos_median, cfg.median_kernel_size);
        dop_img = dop_median;
        sin_img = sin_median;
        cos_img = cos_median;
    }

    // 组装结果
    PolarChannelResult result;
    result.S0_img = S0_img;
    result.aop_vis = aop_vis;
    result.dop_img = dop_img;
    result.sin_img = sin_img;
    result.cos_img = cos_img;
    result.aop = aop.clone();
    result.dop = dop.clone();
    return result;
}
