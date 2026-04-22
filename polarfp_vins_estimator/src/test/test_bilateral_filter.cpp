/**
 * @file test_bilateral_filter.cpp
 * @brief 三种双边滤波管线对比测试节点
 *
 * 功能概述：
 *   1. 订阅原始偏振相机图像（mono8 格式的微偏振阵列 RAW 数据）
 *   2. 对比三种不同的双边滤波管线：
 *      - 管线 0：raw2polar 得到 DoP/sin/cos 后做双边滤波
 *      - 管线 1：计算 S0/S1/S2，对 S1/S2 做双边滤波，再算 DoP/sin/cos
 *      - 管线 2：先在 I0/I45/I90/I135 上做双边滤波，再算 DoP/sin/cos
 *   3. 以 4 行 3 列布局展示对比结果
 *   4. 支持 Shi-Tomasi 角点检测叠加
 *
 * 编译运行：
 *   rosrun polarfp_vins test_bilateral_filter
 *
 * 订阅话题：
 *   /arena_cam_qc2/image_raw
 *
 * 键盘控制：
 *   SPACE  - 暂停/单步/恢复
 *   'f'    - 开关特征点检测叠加
 *   'w'/'W' - 切换当前显示的管线（循环切换）
 *   '+'/'=' - 增大双边滤波 d（邻域直径）
 *   '-'    - 减小双边滤波 d
 *   'b'    - 减小 sigmaColor
 *   'B'    - 增大 sigmaColor
 *   's'    - 保存当前视图到 PNG
 *   ESC/Q  - 退出
 *
 * 三种管线的区别：
 *   管线 0（DOP 后滤波）：标准流程，但 DoP/sin/cos 量化后丢失符号信息
 *   管线 1（S1/S2 滤波）：在中间计算结果上滤波，保持符号连续性，更合理
 *   管线 2（RAW 预滤波）：最激进，在原始强度图上降噪后再计算
 */

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

#include "../featureTracker/PolarChannel.h"

// ============================================================================
// 辅助函数：从原始偏振图像提取 I0/I45/I90/I135 四张子采样强度图
// ============================================================================

/**
 * @brief 从 2x2 微偏振阵列 RAW 图像中提取四个偏振角度子图像
 *         并做 2x 双三次插值，输出半分辨率图像（与 raw2polar 一致）。
 *
 * 超像素布局（4x4 网格中的位置）：
 *   (0,0)=90°  (0,1)=45°  (0,2)=90°  (0,3)=45°
 *   (1,0)=135° (1,1)=0°   (1,2)=135° (1,3)=0°
 *   ...
 *
 * @param img_raw 输入 RAW 图像（单通道 8bit）
 * @param out_I0  输出 0° 子采样图（半分辨率，cubic 插值）
 * @param out_I45 输出 45° 子采样图（半分辨率，cubic 插值）
 * @param out_I90 输出 90° 子采样图（半分辨率，cubic 插值）
 * @param out_I135 输出 135° 子采样图（半分辨率，cubic 插值）
 */
static void extractPolarAngles(const cv::Mat& img_raw,
                               cv::Mat& out_I0, cv::Mat& out_I45,
                               cv::Mat& out_I90, cv::Mat& out_I135) {
    int h_sub = img_raw.rows / 4;
    int w_sub = img_raw.cols / 4;

    // 先 4 倍下采样提取子像素
    cv::Mat I90_sub(h_sub, w_sub, CV_8U);
    cv::Mat I45_sub(h_sub, w_sub, CV_8U);
    cv::Mat I135_sub(h_sub, w_sub, CV_8U);
    cv::Mat I0_sub(h_sub, w_sub, CV_8U);

    for (int i = 0; i < h_sub; i++) {
        for (int j = 0; j < w_sub; j++) {
            I90_sub.at<uchar>(i, j)   = img_raw.at<uchar>(i * 4,     j * 4);
            I45_sub.at<uchar>(i, j)   = img_raw.at<uchar>(i * 4,     j * 4 + 1);
            I135_sub.at<uchar>(i, j)  = img_raw.at<uchar>(i * 4 + 1, j * 4);
            I0_sub.at<uchar>(i, j)    = img_raw.at<uchar>(i * 4 + 1, j * 4 + 1);
        }
    }

    // 2x 双三次插值 → 半分辨率，与 raw2polar 中的 _raw_chnl_to_gray 一致
    cv::resize(I90_sub,  out_I90,  cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);
    cv::resize(I45_sub,  out_I45,  cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);
    cv::resize(I135_sub, out_I135, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);
    cv::resize(I0_sub,   out_I0,   cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);
}

// ============================================================================
// 双边滤波管线结果
// ============================================================================

struct PipelineResult {
    // 管线 0：raw2polar 后滤波
    cv::Mat p0_dop_raw, p0_sin_raw, p0_cos_raw, p0_S0;
    cv::Mat p0_dop_f, p0_sin_f, p0_cos_f;

    // 管线 1：S1/S2 滤波
    cv::Mat p1_dop_raw, p1_sin_raw, p1_cos_raw, p1_S0;
    cv::Mat p1_dop_f, p1_sin_f, p1_cos_f;

    // 管线 2：I0/I45/I90/I135 滤波
    cv::Mat p2_dop_raw, p2_sin_raw, p2_cos_raw, p2_S0;
    cv::Mat p2_dop_f, p2_sin_f, p2_cos_f;
};

/**
 * @brief 三种滤波管线的结果容器
 *
 * 每条管线保存 4 张原始未滤波图（S0 + DoP/sin/cos）和 3 张滤波后图（DoP/sin/cos）。
 * 所有图像尺寸均为输入 RAW 的半分辨率（宽高各 /2）。
 */
struct PipelineResult {
    // 管线 0：raw2polar 后滤波
    cv::Mat p0_dop_raw, p0_sin_raw, p0_cos_raw, p0_S0;
    cv::Mat p0_dop_f, p0_sin_f, p0_cos_f;

    // 管线 1：S1/S2 滤波
    cv::Mat p1_dop_raw, p1_sin_raw, p1_cos_raw, p1_S0;
    cv::Mat p1_dop_f, p1_sin_f, p1_cos_f;

    // 管线 2：I0/I45/I90/I135 滤波
    cv::Mat p2_dop_raw, p2_sin_raw, p2_cos_raw, p2_S0;
    cv::Mat p2_dop_f, p2_sin_f, p2_cos_f;
};

/**
 * @brief 对三个偏振通道施加双边滤波
 *
 * 将双边滤波器并行作用于 DoP、sin(AoP)、cos(AoP) 三张 8-bit 图。
 * 该函数被管线 0 直接调用，也用于管线 1 和管线 2 中 32F 数据的滤波（外部转精度）。
 *
 * @param dop_in      输入 DoP 图（8-bit）
 * @param sin_in      输入 sin(AoP) 图（8-bit）
 * @param cos_in      输入 cos(AoP) 图（8-bit）
 * @param dop_out     输出 DoP 滤波结果
 * @param sin_out     输出 sin(AoP) 滤波结果
 * @param cos_out     输出 cos(AoP) 滤波结果
 * @param d           邻域直径，>0 则使用固定值；≤0 时从 sigmaSpace 自动计算
 * @param sigmaColor  颜色空间标准差，控制强度差异对权重的影响
 * @param sigmaSpace  空间域标准差，控制像素距离对权重的影响
 */
static void applyBilateral(const cv::Mat& dop_in, const cv::Mat& sin_in, const cv::Mat& cos_in,
                           cv::Mat& dop_out, cv::Mat& sin_out, cv::Mat& cos_out,
                           int d, double sigmaColor, double sigmaSpace) {
    cv::bilateralFilter(dop_in, dop_out, d, sigmaColor, sigmaSpace);
    cv::bilateralFilter(sin_in, sin_out, d, sigmaColor, sigmaSpace);
    cv::bilateralFilter(cos_in, cos_out, d, sigmaColor, sigmaSpace);
}

/**
 * @brief 将 double 精度的 DoP/sin/cos 量化为 8-bit 显示图像
 *
 * DoP 范围 [0,1] → 线性映射到 [0,255]
 * sin/cos 范围 [-1,1] → 线性映射到 [0,255]（127.5 为零点）
 *
 * @param dop64  double 精度 DoP
 * @param sin64  double 精度 sin(AoP)
 * @param cos64  double 精度 cos(AoP)
 * @param dop8   输出 8-bit DoP
 * @param sin8   输出 8-bit sin(AoP)
 * @param cos8   输出 8-bit cos(AoP)
 */
static void quantizePolar(const cv::Mat& dop64, const cv::Mat& sin64, const cv::Mat& cos64,
                          cv::Mat& dop8, cv::Mat& sin8, cv::Mat& cos8) {
    dop64.convertTo(dop8, CV_8U, 255.0);
    sin64.convertTo(sin8, CV_8U, 127.5, 127.5);
    cos64.convertTo(cos8, CV_8U, 127.5, 127.5);
}

// ============================================================================
// 三种管线
// ============================================================================

/**
 * @brief 管线 0 — raw2polar 解码后对 DoP/sin/cos 做双边滤波
 *
 * 这是最直接的流程：先用 raw2polar 完整解码偏振信息，
 * 然后在量化后的 8-bit DoP/sin/cos 图上直接施加双边滤波。
 * 缺点：量化会丢失 sin/cos 的符号信息（[-1,1] → [0,255]）。
 */
static void runPipeline0(const PolarChannelResult& polar, PipelineResult& res,
                         int d, double sc, double ss) {
    polar.dop_img.copyTo(res.p0_dop_raw);
    polar.sin_img.copyTo(res.p0_sin_raw);
    polar.cos_img.copyTo(res.p0_cos_raw);
    polar.S0_img.copyTo(res.p0_S0);
    applyBilateral(res.p0_dop_raw, res.p0_sin_raw, res.p0_cos_raw,
                   res.p0_dop_f, res.p0_sin_f, res.p0_cos_f, d, sc, ss);
}

/**
 * @brief 管线 1 — 计算 S0/S1/S2，对 S1/S2 做双边滤波，再推导 DoP/sin/cos
 *
 * 核心思路：S1 = I0 - I90、S2 = I45 - I135 是带符号的中间量，
 * 在此阶段滤波可以避免量化引入的符号信息丢失。
 * 双边滤波需要 32F 以支持负值，所以先转 CV_32F → 滤波 → 转回 CV_64F → 计算 DoP/sin/cos。
 * 最后同样做掩膜（S0 接近 0 和 DoP 异常值）处理。
 */
static void runPipeline1(const cv::Mat& I0, const cv::Mat& I45,
                         const cv::Mat& I90, const cv::Mat& I135,
                         PipelineResult& res,
                         int d, double sc, double ss) {
    // 先转 64F 再做算术运算，否则 8-bit 运算不提升精度
    cv::Mat I0_64, I45_64, I90_64, I135_64;
    I0.convertTo(I0_64, CV_64F);
    I45.convertTo(I45_64, CV_64F);
    I90.convertTo(I90_64, CV_64F);
    I135.convertTo(I135_64, CV_64F);

    cv::Mat S0_64 = (I0_64 + I45_64 + I90_64 + I135_64) / 4.0;
    cv::Mat S1_64 = (I0_64 - I90_64);
    cv::Mat S2_64 = (I45_64 - I135_64);

    // 保存未滤波的显示结果
    S0_64.convertTo(res.p1_S0, CV_8U);
    cv::Mat den_64;
    cv::sqrt(S1_64.mul(S1_64) + S2_64.mul(S2_64) + EPSILON, den_64);
    cv::Mat sin64 = S2_64 / den_64;
    cv::Mat cos64 = S1_64 / den_64;
    cv::Mat dop64 = den_64 / (S0_64 + EPSILON);
    quantizePolar(dop64, sin64, cos64, res.p1_dop_raw, res.p1_sin_raw, res.p1_cos_raw);

    // S1/S2 双边滤波（32F 以支持负值）
    cv::Mat S1_f, S2_f;
    cv::Mat S1_32f, S2_32f;
    S1_64.convertTo(S1_32f, CV_32F);
    S2_64.convertTo(S2_32f, CV_32F);
    cv::bilateralFilter(S1_32f, S1_f, d, sc, ss);
    cv::bilateralFilter(S2_32f, S2_f, d, sc, ss);
    cv::Mat S1_filt, S2_filt;
    S1_f.convertTo(S1_filt, CV_64F);
    S2_f.convertTo(S2_filt, CV_64F);

    // 从滤波后的 S1/S2 计算 DoP/sin/cos
    cv::Mat den_f;
    cv::sqrt(S1_filt.mul(S1_filt) + S2_filt.mul(S2_filt) + EPSILON, den_f);
    cv::Mat mask = cv::abs(S0_64) < EPSILON;
    cv::Mat sin_f = S2_filt / (den_f + EPSILON);
    cv::Mat cos_f = S1_filt / (den_f + EPSILON);
    cv::Mat dop_f = den_f / (S0_64 + EPSILON);
    sin_f.setTo(0.0, mask);
    cos_f.setTo(0.0, mask);
    dop_f.setTo(0.0, mask);
    cv::Mat mask2 = dop_f > 0.999;
    sin_f.setTo(0.0, mask2);
    cos_f.setTo(0.0, mask2);
    dop_f.setTo(0.0, mask2);

    quantizePolar(dop_f, sin_f, cos_f, res.p1_dop_f, res.p1_sin_f, res.p1_cos_f);
}

/**
 * @brief 管线 2 — 先在 I0/I45/I90/I135 上做双边滤波，再计算 DoP/sin/cos
 *
 * 最激进的方案：在四个偏振角度原始强度图（8-bit，经 2x 插值后）上
 * 分别施加双边滤波，然后从滤波后的强度值推导 S0/S1/S2 和 DoP/sin/cos。
 * 注意：不对 S0 做滤波，只对四个角度的原始图像做滤波。
 * 这种方案在源头降噪，但可能改变 Stokes 参数之间的物理关系。
 */
static void runPipeline2(const cv::Mat& I0, const cv::Mat& I45,
                         const cv::Mat& I90, const cv::Mat& I135,
                         PipelineResult& res,
                         int d, double sc, double ss) {
    cv::Mat I0_64, I45_64, I90_64, I135_64;
    I0.convertTo(I0_64, CV_64F);
    I45.convertTo(I45_64, CV_64F);
    I90.convertTo(I90_64, CV_64F);
    I135.convertTo(I135_64, CV_64F);

    cv::Mat S0_64 = (I0_64 + I45_64 + I90_64 + I135_64) / 4.0;
    S0_64.convertTo(res.p2_S0, CV_8U);
    cv::Mat S1_64 = (I0_64 - I90_64);
    cv::Mat S2_64 = (I45_64 - I135_64);
    cv::Mat den_64;
    cv::sqrt(S1_64.mul(S1_64) + S2_64.mul(S2_64) + EPSILON, den_64);
    cv::Mat sin64 = S2_64 / den_64;
    cv::Mat cos64 = S1_64 / den_64;
    cv::Mat dop64 = den_64 / (S0_64 + EPSILON);
    quantizePolar(dop64, sin64, cos64, res.p2_dop_raw, res.p2_sin_raw, res.p2_cos_raw);

    // I0/I45/I90/I135 双边滤波（32F 以精确处理强度值）
    cv::Mat I0_f, I45_f, I90_f, I135_f;
    cv::Mat I0_32f, I45_32f, I90_32f, I135_32f;
    I0.convertTo(I0_32f, CV_32F);
    I45.convertTo(I45_32f, CV_32F);
    I90.convertTo(I90_32f, CV_32F);
    I135.convertTo(I135_32f, CV_32F);
    cv::bilateralFilter(I0_32f, I0_f, d, sc, ss);
    cv::bilateralFilter(I45_32f, I45_f, d, sc, ss);
    cv::bilateralFilter(I90_32f, I90_f, d, sc, ss);
    cv::bilateralFilter(I135_32f, I135_f, d, sc, ss);
    cv::Mat I0_filt, I45_filt, I90_filt, I135_filt;
    I0_f.convertTo(I0_filt, CV_64F);
    I45_f.convertTo(I45_filt, CV_64F);
    I90_f.convertTo(I90_filt, CV_64F);
    I135_f.convertTo(I135_filt, CV_64F);

    // 从滤波后的 I 值计算 S0/S1/S2 → DoP/sin/cos
    cv::Mat S0_f = (I0_filt + I45_filt + I90_filt + I135_filt) / 4.0;
    cv::Mat S1_f = I0_filt - I90_filt;
    cv::Mat S2_f = I45_filt - I135_filt;
    cv::Mat den_f;
    cv::sqrt(S1_f.mul(S1_f) + S2_f.mul(S2_f) + EPSILON, den_f);
    cv::Mat mask = cv::abs(S0_f) < EPSILON;
    cv::Mat sin_f = S2_f / (den_f + EPSILON);
    cv::Mat cos_f = S1_f / (den_f + EPSILON);
    cv::Mat dop_f = den_f / (S0_f + EPSILON);
    sin_f.setTo(0.0, mask);
    cos_f.setTo(0.0, mask);
    dop_f.setTo(0.0, mask);

    quantizePolar(dop_f, sin_f, cos_f, res.p2_dop_f, res.p2_sin_f, res.p2_cos_f);
}

// ============================================================================
// 可视化
// ============================================================================

/**
 * @brief 构建 4 行 4 列的三管线对比视图
 *
 * 布局：
 *   列 → S0 | DoP | sin(AoP) | cos(AoP)
 *   行0 → 原始未滤波（P0 的 raw 通道）
 *   行1 → 管线 0（raw2polar → 双边滤波）
 *   行2 → 管线 1（S1/S2 双边滤波 → 推导 DoP/sin/cos）
 *   行3 → 管线 2（I0/I45/I90/I135 双边滤波 → 推导 DoP/sin/cos）
 *
 * 顶部有彩色列标题，当前活动管线行用红色边框高亮。
 *
 * @param res            三管线的滤波结果
 * @param out            输出的拼接图像（BGR 格式）
 * @param active_pipeline 当前活动的管线索引（0/1/2）
 */
static void buildCompareView(const PipelineResult& res, cv::Mat& out,
                             int active_pipeline) {
    const cv::Scalar txtColor(0, 255, 0);
    auto labelAndGray = [&](const cv::Mat& gray, const std::string& lbl) -> cv::Mat {
        cv::Mat c;
        cv::cvtColor(gray, c, cv::COLOR_GRAY2BGR);
        cv::putText(c, lbl, cv::Point(5, 14), cv::FONT_HERSHEY_SIMPLEX, 0.35, txtColor, 1, cv::LINE_AA);
        return c;
    };

    const char* pnames[] = {"raw2polar→Bilateral", "S1/S2→Bilateral", "I0-135→Bilateral"};

    // Row 0: raw unfiltered
    cv::Mat r0c0 = labelAndGray(res.p0_S0, "S0 raw");
    cv::Mat r0c1 = labelAndGray(res.p0_dop_raw, "DoP raw");
    cv::Mat r0c2 = labelAndGray(res.p0_sin_raw, "sin(AoP) raw");
    cv::Mat r0c3 = labelAndGray(res.p0_cos_raw, "cos(AoP) raw");
    cv::Mat row0, row0r;
    cv::hconcat(r0c0, r0c1, row0);
    cv::hconcat(r0c2, r0c3, row0r);
    cv::hconcat(row0, row0r, row0);

    // Row 1: Pipeline 0
    cv::Mat r1c0 = labelAndGray(res.p0_S0, "P0 S0");
    cv::Mat r1c1 = labelAndGray(res.p0_dop_f, "DoP");
    cv::Mat r1c2 = labelAndGray(res.p0_sin_f, "sin(AoP)");
    cv::Mat r1c3 = labelAndGray(res.p0_cos_f, "cos(AoP)");
    cv::Mat row1, row1r;
    cv::hconcat(r1c0, r1c1, row1);
    cv::hconcat(r1c2, r1c3, row1r);
    cv::hconcat(row1, row1r, row1);

    // Row 2: Pipeline 1
    cv::Mat r2c0 = labelAndGray(res.p1_S0, "P1 S0");
    cv::Mat r2c1 = labelAndGray(res.p1_dop_f, "DoP");
    cv::Mat r2c2 = labelAndGray(res.p1_sin_f, "sin(AoP)");
    cv::Mat r2c3 = labelAndGray(res.p1_cos_f, "cos(AoP)");
    cv::Mat row2, row2r;
    cv::hconcat(r2c0, r2c1, row2);
    cv::hconcat(r2c2, r2c3, row2r);
    cv::hconcat(row2, row2r, row2);

    // Row 3: Pipeline 2
    cv::Mat r3c0 = labelAndGray(res.p2_S0, "P2 S0");
    cv::Mat r3c1 = labelAndGray(res.p2_dop_f, "DoP");
    cv::Mat r3c2 = labelAndGray(res.p2_sin_f, "sin(AoP)");
    cv::Mat r3c3 = labelAndGray(res.p2_cos_f, "cos(AoP)");
    cv::Mat row3, row3r;
    cv::hconcat(r3c0, r3c1, row3);
    cv::hconcat(r3c2, r3c3, row3r);
    cv::hconcat(row3, row3r, row3);

    cv::vconcat(row0, row1, out);
    cv::Mat bottom;
    cv::vconcat(row2, row3, bottom);
    cv::vconcat(out, bottom, out);

    // Highlight active pipeline row
    int rowH = out.rows / 4;
    if (active_pipeline < 3) {
        cv::Mat rowROI = out(cv::Rect(0, active_pipeline * rowH, out.cols, rowH));
        cv::rectangle(out, cv::Point(0, active_pipeline * rowH),
                      cv::Point(out.cols - 1, (active_pipeline + 1) * rowH - 1),
                      cv::Scalar(0, 0, 255), 2);
    }

    // Add column headers at the top
    int barH = 25;
    cv::Mat header = cv::Mat::zeros(barH, out.cols, CV_8UC3);
    int colW = out.cols / 4;
    const char* colLabels[] = {"S0", "DoP", "sin(AoP)", "cos(AoP)"};
    const cv::Scalar colColors[] = {
        cv::Scalar(255, 255, 255),
        cv::Scalar(0, 255, 255),
        cv::Scalar(255, 0, 255),
        cv::Scalar(255, 255, 0)
    };
    for (int i = 0; i < 4; i++) {
        cv::putText(header, colLabels[i], cv::Point(i * colW + 5, 17),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, colColors[i], 1, cv::LINE_AA);
    }
    cv::vconcat(header, out, out);
}

/**
 * @brief 在单通道图像上执行 Shi-Tomasi 角点检测并绘制结果
 *
 * 用于评估不同管线和滤波处理后特征点提取的数量和质量变化。
 *
 * @param gray     输入灰度图
 * @param out      输出带角点标记的 BGR 图像
 * @param label    图像标签（显示在左上角）
 * @param max_cnt  最大角点数量
 * @param quality  Shi-Tomasi 质量阈值（最小可接受特征值）
 * @param min_dist 角点之间的最小像素距离
 */
static void detectAndDrawFeatures(const cv::Mat& gray, cv::Mat& out, const std::string& label,
                                  int max_cnt, double quality, double min_dist) {
    cv::cvtColor(gray, out, cv::COLOR_GRAY2BGR);
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(gray, corners, max_cnt, quality, min_dist);
    for (const auto& pt : corners) {
        cv::circle(out, pt, 3, cv::Scalar(0, 255, 0), -1);
    }
    cv::putText(out, label + " (" + std::to_string(corners.size()) + " pts)",
                cv::Point(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
}

/**
 * @brief 构建特征点检测对比视图
 *
 * 展示 S0 原始图 + DoP 原始图 + 当前活动管线 DoP 滤波结果上的角点检测效果，
 * 用于比较哪种管线能提取到更多/更稳定的特征点。
 *
 * @param res           三管线的滤波结果
 * @param out           输出的拼接图像（2 行 2 列：S0 | DoP raw / DoP filtered | 空白）
 * @param max_cnt       最大角点数
 * @param quality       角点质量阈值
 * @param min_dist      角点最小间距
 * @param active_pipeline 当前活动的管线索引
 */
static void buildFeatureView(const PipelineResult& res, cv::Mat& out,
                             int max_cnt, double quality, double min_dist,
                             int active_pipeline) {
    cv::Mat s0_f, dop_rf, dop_filt;
    detectAndDrawFeatures(res.p0_S0, s0_f, "S0", max_cnt, quality, min_dist);
    detectAndDrawFeatures(res.p0_dop_raw, dop_rf, "DoP raw", max_cnt, quality, min_dist);

    if (active_pipeline == 0)
        detectAndDrawFeatures(res.p0_dop_f, dop_filt, "DoP P0-bilateral", max_cnt, quality, min_dist);
    else if (active_pipeline == 1)
        detectAndDrawFeatures(res.p1_dop_f, dop_filt, "DoP P1-S1S2-bilateral", max_cnt, quality, min_dist);
    else
        detectAndDrawFeatures(res.p2_dop_f, dop_filt, "DoP P2-RAW-bilateral", max_cnt, quality, min_dist);

    cv::Mat row1, row2;
    cv::hconcat(s0_f, dop_rf, row1);
    cv::hconcat(dop_filt, cv::Mat::zeros(dop_filt.size(), dop_filt.type()), row2);
    cv::vconcat(row1, row2, out);
}

// ============================================================================
// ROS Node
// ============================================================================

/// 订阅的原始偏振相机图像话题（需与实际硬件发布的话题一致）
const std::string IMAGE_TOPIC = "/arena_cam_qc2/image_raw";

/// 保护最新帧的互斥锁
std::mutex img_mutex;
/// 最新一帧原始图像
cv::Mat latest_raw;
/// 是否有新帧待处理
bool new_frame = false;

/**
 * @brief ROS 图像话题回调函数
 *
 * 将 sensor_msgs/Image 转换为 OpenCV Mat 并缓存到 latest_raw。
 * 使用 toCvShare 避免内存拷贝，但需立即 copyTo 脱离 ROS 消息生命周期。
 */
void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    try {
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "mono8");
        std::lock_guard<std::mutex> lock(img_mutex);
        cv_ptr->image.copyTo(latest_raw);
        new_frame = true;
    } catch (cv_bridge::Exception& e) {
        ROS_WARN("cv_bridge error: %s", e.what());
    }
}

/**
 * @brief 主函数 — 初始化 ROS 节点、滤波参数、OpenCV 窗口及主循环
 *
 * 主循环流程：
 *   1. ros::spinOnce() 处理 ROS 回调（接收新图像）
 *   2. 线程安全地取出最新帧
 *   3. raw2polar + extractPolarAngles 解码偏振数据
 *   4. 并行运行三条管线
 *   5. 构建对比视图（或特征点视图）并显示
 *   6. 等待键盘输入处理交互命令
 */
int main(int argc, char** argv) {
    ros::init(argc, argv, "test_bilateral_filter");
    ROS_INFO("=== Polar Bilateral Filter Pipeline Comparison ===");
    ROS_INFO("Subscribing to %s", IMAGE_TOPIC.c_str());

    // 双边滤波参数
    int bilateral_d = 9;
    double bilateral_sigmaColor = 200;
    double bilateral_sigmaSpace = 30;

    // 角点检测参数
    int max_cnt = 200;
    double quality = 0.01;
    double min_dist = 10;

    ROS_INFO("Bilateral params: d=%d sigmaColor=%.1f sigmaSpace=%.1f",
             bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace);

    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe(IMAGE_TOPIC, 10, imageCallback);

    const std::string win_name = "Bilateral Pipeline Comparison";
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(win_name, 1600, 1000);

    bool show_features = false;
    bool paused = false;
    int frame_count = 0;
    int saved_count = 0;
    int active_pipeline = 0;
    const char* pipeline_names[] = {"raw2polar→Bilateral", "S1/S2→Bilateral", "I0-135→Bilateral"};
    ROS_INFO("Active pipeline: %s (press 'w' to switch)", pipeline_names[active_pipeline]);

    cv::Mat I0, I45, I90, I135;

    while (ros::ok()) {
        ros::spinOnce();

        std::lock_guard<std::mutex> lock(img_mutex);
        bool has_new = new_frame;
        new_frame = false;
        cv::Mat raw;
        if (has_new) {
            latest_raw.copyTo(raw);
        }

        cv::Mat display;
        if (has_new) {
            auto t_start = std::chrono::high_resolution_clock::now();

            // raw2polar for Pipeline 0 (and raw reference)
            PolarChannelResult polar = raw2polar(raw);

            // Extract I0/I45/I90/I135 for Pipeline 1 and 2
            extractPolarAngles(raw, I0, I45, I90, I135);

            PipelineResult pr;
            runPipeline0(polar, pr, bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace);
            runPipeline1(I0, I45, I90, I135, pr, bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace);
            runPipeline2(I0, I45, I90, I135, pr, bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace);

            auto t_end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
            printf("total polar processing time: %.2f ms\n", elapsed_ms);

            if (show_features) {
                buildFeatureView(pr, display, max_cnt, quality, min_dist, active_pipeline);
            } else {
                buildCompareView(pr, display, active_pipeline);
            }

            cv::setWindowTitle(win_name, "Pipeline: " + std::string(pipeline_names[active_pipeline]));
            cv::imshow(win_name, display);
            frame_count++;
        }

        int key = cv::waitKey(20) & 0xFF;

        if (key == 27 || key == 'q') {
            break;
        } else if (key == 32) {
            paused = !paused;
            ROS_INFO("%s", paused ? "PAUSED" : "RESUMED");
        } else if (paused) {
            continue;
        } else if (key == 's') {
            std::string name = "bilateral_frame_" + std::to_string(frame_count) + "_" + std::to_string(saved_count) + ".png";
            cv::imwrite(name, display);
            ROS_INFO("Saved %s", name.c_str());
            saved_count++;
        } else if (key == 'f') {
            show_features = !show_features;
            ROS_INFO("Feature overlay: %s", show_features ? "ON" : "OFF");
        } else if (key == 'w' || key == 'W') {
            active_pipeline = (active_pipeline + 1) % 3;
            ROS_INFO("Pipeline -> %s", pipeline_names[active_pipeline]);
        } else if (key == '=' || key == '+') {
            bilateral_d = std::min(bilateral_d + 1, 25);
            ROS_INFO("bilateral_d -> %d", bilateral_d);
        } else if (key == '-') {
            bilateral_d = std::max(bilateral_d - 1, 1);
            ROS_INFO("bilateral_d -> %d", bilateral_d);
        } else if (key == 'b') {
            bilateral_sigmaColor = std::max(bilateral_sigmaColor - 5.0, 1.0);
            ROS_INFO("bilateral_sigmaColor -> %.1f", bilateral_sigmaColor);
        } else if (key == 'B') {
            bilateral_sigmaColor = std::min(bilateral_sigmaColor + 5.0, 500.0);
            ROS_INFO("bilateral_sigmaColor -> %.1f", bilateral_sigmaColor);
        }
    }

    cv::destroyAllWindows();
    ROS_INFO("Processed %d frames", frame_count);
    return 0;
}
