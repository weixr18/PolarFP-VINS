/*******************************************************
 * Copyright (C) 2026, Bionic Intelligence Research Team, Beihang University
 *
 * This file is part of PolarFP-VINS.
 * Auther: Wei Xinran (github.com/weixr18; weixr0605@sina.com)
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef _POLAR_CHANNEL_H
#define _POLAR_CHANNEL_H

/**
 * @file PolarChannel.h
 * @brief 偏振图像解码模块 —— 将微偏振片阵列（2x2 拜耳式）原始图像
 *        转换为 Stokes 参数（S0/DoP/AoP）及其可视化通道。
 *
 * 输入图像像素排布（每 2x2 像素为一个偏振超像素）：
 *   (0,0)=90°  (0,1)=45°
 *   (1,0)=135° (1,1)=0°
 *
 * 输出通道说明：
 *   - S0:      总强度（等效灰度图），暗光下退化
 *   - DoP:     偏振度 [0,1]，量化为 0-255，材质/边缘信息，对光照鲁棒
 *   - AoP:     偏振角 [-π/2, π/2]，表面法向信息，对光照鲁棒
 *   - sin/cos: AoP 的正弦/余弦分量，量化为 0-255，用于特征提取
 */

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>


/** 防止除零的小常数 */
const double EPSILON = 1e-6;

/** @brief 将4通道原始数据融合为灰度图（BT.601加权），输出经双三次插值放大2x */
cv::Mat _raw_chnl_to_gray(const std::vector<cv::Mat>& img_xx);

/**
 * @brief raw2polar 的输出结构体
 *
 * 包含原始计算结果（double 精度）和可视化版本（8bit）。
 * 所有输出图像尺寸均为输入的一半（宽高各 /2）。
 */
struct PolarChannelResult {
    cv::Mat S0_color;  ///< (h/2, w/2, 3) RGB 彩色 S0 图，用于可视化
    cv::Mat S0_img;    ///< (h/2, w/2) 灰度 S0 强度图，等效灰度图
    cv::Mat aop_vis;   ///< (h/2, w/2, 3) AoP 的 HSV 彩色可视化
    cv::Mat dop_img;   ///< (h/2, w/2) 量化后的 DoP 图 [0, 255]
    cv::Mat sin_img;   ///< (h/2, w/2) 量化后的 sin(AoP) 图 [0, 255]
    cv::Mat cos_img;   ///< (h/2, w/2) 量化后的 cos(AoP) 图 [0, 255]
    cv::Mat aop;       ///< (h/2, w/2) 原始 AoP 值（double 精度，单位：弧度）
    cv::Mat dop;       ///< (h/2, w/2) 原始 DoP 值（double 精度，范围 [0, 1]）
};

/**
 * @brief 滤波器类型枚举
 *
 * 用于在 raw2polar() 中选择不同的去噪滤波算法。
 */
enum PolarFilterType {
    FILTER_NONE = 0,
    FILTER_BILATERAL = 1,
    FILTER_GUIDED = 2,
    FILTER_NLM = 3,
    FILTER_MEDIAN = 4,
};

/**
 * @brief 偏振通道滤波配置参数
 *
 * 用于在 raw2polar() 输出的 DoP/sin/cos 通道上施加滤波，
 * 降低低光照条件下的噪声，提升特征点检测稳定性。
 */
struct PolarFilterConfig {
    // 滤波器类型选择
    PolarFilterType filter_type = FILTER_NONE;  ///< 滤波器类型（默认不启用）

    // 双边滤波参数
    int bilateral_d = 9;                ///< 邻域直径
    double bilateral_sigmaColor = 200;  ///< 颜色空间标准差
    double bilateral_sigmaSpace = 30;   ///< 空间域标准差

    // 导向滤波参数
    int guided_radius = 4;              ///< 局部窗口半径（核大小 = 2*radius+1）
    double guided_eps = 0.01;           ///< 正则化参数，越大平滑越强

    // NLM（非局部均值）滤波参数
    float nlm_h = 50.0f;                ///< 滤波强度
    int nlm_template = 5;               ///< 模板窗口大小
    int nlm_search = 21;                ///< 搜索窗口大小

    // 中值滤波参数
    int median_kernel_size = 5;         ///< 中值滤波核大小（必须为奇数且 > 1）
};

/**
 * @brief 核心函数：将微偏振片阵列原始图像转换为 Stokes 参数
 *
 * 处理流程：
 *   1. 从 2x2 超像素网格中采样出 4 个偏振角度图像（0°/45°/90°/135°）
 *   2. 计算 Stokes 向量：S0 = (I0+I45+I90+I135)/4, S1 = I0-I90, S2 = I45-I135
 *   3. 计算 DoP = sqrt(S1²+S2²) / S0
 *   4. 计算 AoP = 0.5 * atan2(S2, S1)
 *   5. 将 AoP 映射为 HSV 彩色图以便可视化
 *   6. 对 DoP 做百分位归一化
 *   7. 量化输出为 8bit 图像
 *   8. 若 cfg.filter_type != FILTER_NONE，对 DoP/sin/cos 施加指定类型的滤波
 *
 * @param img_raw 输入原始偏振图像（单通道 8bit，2x2 超像素排布）
 * @param cfg     滤波配置参数（默认不启用滤波）
 * @return PolarChannelResult 包含所有通道和可视化结果
 */

/** @brief 单通道导向滤波，以S0强度图为引导，在平滑噪声的同时保留边缘 */
cv::Mat guidedFilterSingle(const cv::Mat& I, const cv::Mat& p, int r, double eps);

/** @brief 核心函数：将微偏振阵列原始图像解码为Stokes参数（S0/DoP/AoP）及各通道可视化 */
PolarChannelResult raw2polar(const cv::Mat& img_raw, const PolarFilterConfig& cfg = {});

#endif // _POLAR_CHANNEL_H
