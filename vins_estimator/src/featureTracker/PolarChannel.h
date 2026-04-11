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

/** DoP 归一化时使用的百分位阈值，用于去除异常高值 */
const double DOP_PERCENTILE = 99.0;

/**
 * @brief 计算矩阵的指定百分位值
 * @param mat 输入矩阵（任意类型）
 * @param percentile 百分位 [0, 100]
 * @return 排序后对应位置的元素值
 */
double _calculatePercentile(const cv::Mat& mat, double percentile);

/**
 * @brief 将 4 通道原始数据转换为灰度图（加权平均）
 * @param img_xx 长度为 4 的 Mat 向量，分别对应 R, G1, G2, B 通道
 * @return 融合后的灰度图，尺寸放大 2x（双三次插值）
 */
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
 *
 * @param img_raw 输入原始偏振图像（单通道 8bit，2x2 超像素排布）
 * @return PolarChannelResult 包含所有通道和可视化结果
 */
PolarChannelResult raw2polar(const cv::Mat& img_raw);

#endif // _POLAR_CHANNEL_H
