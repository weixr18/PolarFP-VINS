/*******************************************************
 * Copyright (C) 2026, Bionic Intelligence Research Team, Beihang University
 *
 * This file is part of PolarFP-VINS.
 * Author: Wei Xinran (github.com/weixr18; weixr0605@sina.com)
 * Licensed under the GNU General Public License v3.0.
 *******************************************************/

/**
 * @file main.cpp
 * @brief 偏振通道滤波方法对比图生成
 *
 * 对原始偏振图像解码出的 P0(DoP)/P1(sinAoP)/P2(cosAoP) 三个通道，
 * 分别施加 4 种滤波算法（中值 / 导向 / 双边 / NLM），生成 3行×5列 对比图。
 *
 * 用法:
 *   mkdir build && cd build && cmake .. && make
 *   ./filter_compare <input_dir> <output_dir>
 *
 * input_dir 中包含从 ROSbag 提取的 PNG 文件。
 * 每张 PNG 生成一张对比图保存到 output_dir。
 */

#include "PolarChannel.h"
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <string>
#include <vector>

// ============================================================================
// 滤波器参数（论文中表1）
// ============================================================================
const int MEDIAN_K = 3;                 // 中值滤波核大小（3x3，执行3轮）

const int GUIDED_R = 8;                 // 导向滤波局部窗口半径
const double GUIDED_EPS = 0.0001;       // 导向滤波正则化参数

const int BILATERAL_D = 9;              // 双边滤波邻域直径
const double BILATERAL_SIGMA_COLOR = 200;  // 双边滤波颜色空间标准差
const double BILATERAL_SIGMA_SPACE = 30;   // 双边滤波空间域标准差

const double NLM_H = 50.0;              // NLM 去噪强度
const int NLM_TEMPLATE_WINDOW = 5;      // NLM 模板窗口大小
const int NLM_SEARCH_WINDOW = 21;       // NLM 搜索窗口大小

// ============================================================================
// 通道与标签定义
// ============================================================================
const std::vector<std::string> ROW_LABELS = {"P0 (DoP)", "P1 (sin(AoP))", "P2 (cos(AoP))"};
const std::vector<std::string> COL_LABELS = {"Raw", "Median", "Guided", "Bilateral", "NLM"};
const int NUM_ROWS = 3;
const int NUM_COLS = 5;

// ============================================================================
// 滤波结果容器
// ============================================================================
struct FilteredChannels {
    cv::Mat raw;
    cv::Mat median;
    cv::Mat guided;
    cv::Mat bilateral;
    cv::Mat nlm;
};

/**
 * @brief 对一个偏振通道施加全部 4 种滤波器
 *
 * @param ch        输入通道（CV_8U）
 * @param s0_guide  S0 强度图（用于导向滤波的引导图, CV_8U）
 * @return FilteredChannels 包含原始 + 4 种滤波结果
 */
static FilteredChannels applyAllFilters(const cv::Mat& ch, const cv::Mat& s0_guide) {
    FilteredChannels res;
    ch.copyTo(res.raw);

    // ── 中值滤波: 3x3 核, 执行 3 轮 ──
    cv::Mat med = ch.clone();
    for (int i = 0; i < 3; i++) {
        cv::medianBlur(med, med, MEDIAN_K);
    }
    res.median = med;

    // ── 导向滤波（以 S0 为引导图）──
    // 需要转换到 [0,1] double 精度
    cv::Mat ch_f, s0_f, guided_f;
    ch.convertTo(ch_f, CV_64F, 1.0 / 255.0);
    s0_guide.convertTo(s0_f, CV_64F, 1.0 / 255.0);
    guided_f = guidedFilterSingle(s0_f, ch_f, GUIDED_R, GUIDED_EPS);
    guided_f.convertTo(res.guided, CV_8U, 255.0);

    // ── 双边滤波 ──
    cv::bilateralFilter(ch, res.bilateral, BILATERAL_D,
                        BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE);

    // ── 非局部均值去噪 (NLM) ──
    cv::fastNlMeansDenoising(ch, res.nlm, NLM_H,
                             NLM_TEMPLATE_WINDOW, NLM_SEARCH_WINDOW);

    return res;
}

/**
 * @brief 获取目录下所有 .png 文件（按字母序排列）
 */
static std::vector<std::string> listPngFiles(const std::string& dir) {
    std::vector<std::string> files;
    DIR* dp = opendir(dir.c_str());
    if (!dp) {
        std::cerr << "[ERROR] Cannot open directory: " << dir << std::endl;
        return files;
    }
    struct dirent* entry;
    while ((entry = readdir(dp)) != nullptr) {
        const char* name = entry->d_name;
        size_t len = strlen(name);
        if (len > 4 && strcasecmp(name + len - 4, ".png") == 0) {
            files.push_back(name);
        }
    }
    closedir(dp);
    std::sort(files.begin(), files.end());
    return files;
}

/**
 * @brief 构建 3行×5列 的滤波方法对比图
 *
 * 布局:
 *           Raw    Median  Guided  Bilateral  NLM
 *   P0(DoP)  [img]  [img]   [img]   [img]     [img]
 *   P1(sin)  [img]  [img]   [img]   [img]     [img]
 *   P2(cos)  [img]  [img]   [img]   [img]     [img]
 *
 * @param channels 按行排列的滤波结果 [row][col] = FilteredChannels
 * @param canvas   输出的拼接画布
 */
static void buildCompareFigure(
    const std::vector<FilteredChannels>& channels,
    cv::Mat& canvas) {

    // 取第一个通道的尺寸作为单元格基准
    int cell_w = channels[0].raw.cols;
    int cell_h = channels[0].raw.rows;

    const int PAD = 6;              // 单元格内边距
    const int GAP_H = 10;           // 列间距
    const int GAP_V = 10;           // 行间距
    const int TOP_MARGIN = 60;      // 顶部标题栏高度
    const int LEFT_MARGIN = 130;    // 左侧行标签宽度
    const int HEADER_H = 36;        // 列标题高度

    int canvas_w = LEFT_MARGIN + NUM_COLS * (cell_w + 2 * PAD) + (NUM_COLS - 1) * GAP_H;
    int canvas_h = TOP_MARGIN + HEADER_H + NUM_ROWS * (cell_h + 2 * PAD) + (NUM_ROWS - 1) * GAP_V;

    canvas = cv::Mat(canvas_h, canvas_w, CV_8UC3, cv::Scalar(255, 255, 255));
    int font = cv::FONT_HERSHEY_SIMPLEX;

    // ── 顶部大标题 ──
    {
        std::string title = "Polarization Channel Filter Comparison";
        int baseline = 0;
        cv::Size ts = cv::getTextSize(title, font, 0.7, 2, &baseline);
        cv::putText(canvas, title,
                    cv::Point((canvas_w - ts.width) / 2, TOP_MARGIN - 18),
                    font, 0.7, cv::Scalar(0, 0, 0), 2);
    }

    // ── 列标题 ──
    for (int c = 0; c < NUM_COLS; c++) {
        int cx = LEFT_MARGIN + c * (cell_w + 2 * PAD + GAP_H) + PAD + cell_w / 2;
        int baseline = 0;
        cv::Size ts = cv::getTextSize(COL_LABELS[c], font, 0.65, 2, &baseline);
        cv::putText(canvas, COL_LABELS[c],
                    cv::Point(cx - ts.width / 2, TOP_MARGIN + HEADER_H - 8),
                    font, 0.65, cv::Scalar(0, 0, 0), 2);
    }

    // ── 行标签 + 单元格填充 ──
    for (int r = 0; r < NUM_ROWS; r++) {
        // 行标签（垂直居中）
        int ry = TOP_MARGIN + HEADER_H + r * (cell_h + 2 * PAD + GAP_V) + PAD + cell_h / 2;
        int baseline = 0;
        cv::Size ts = cv::getTextSize(ROW_LABELS[r], font, 0.55, 1, &baseline);
        cv::putText(canvas, ROW_LABELS[r],
                    cv::Point(LEFT_MARGIN - ts.width - 12, ry + ts.height / 3),
                    font, 0.55, cv::Scalar(0, 0, 0), 1);

        // 获取该行各列图像
        std::vector<cv::Mat> imgs = {
            channels[r].raw,
            channels[r].median,
            channels[r].guided,
            channels[r].bilateral,
            channels[r].nlm
        };

        for (int c = 0; c < NUM_COLS; c++) {
            int x0 = LEFT_MARGIN + c * (cell_w + 2 * PAD + GAP_H);
            int y0 = TOP_MARGIN + HEADER_H + r * (cell_h + 2 * PAD + GAP_V);

            // 浅灰色背景边框
            cv::rectangle(canvas,
                          cv::Rect(x0, y0, cell_w + 2 * PAD, cell_h + 2 * PAD),
                          cv::Scalar(200, 200, 200), cv::FILLED);

            // 转为 BGR 显示
            cv::Mat bgr;
            cv::cvtColor(imgs[c], bgr, cv::COLOR_GRAY2BGR);

            // 在单元格左上角标注方法名缩写（小字）
            cv::putText(bgr, COL_LABELS[c],
                        cv::Point(3, 14), font, 0.4, cv::Scalar(0, 200, 0), 1);

            bgr.copyTo(canvas(cv::Rect(x0 + PAD, y0 + PAD, cell_w, cell_h)));
        }
    }
}

/**
 * @brief 主函数
 *
 * 用法: ./filter_compare <input_dir> <output_dir>
 */
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_dir> <output_dir>" << std::endl;
        return 1;
    }

    std::string input_dir = argv[1];
    std::string output_dir = argv[2];

    // 确保输出目录存在
    std::string mkdir_cmd = "mkdir -p " + output_dir;
    system(mkdir_cmd.c_str());

    // 列出输入目录下所有 PNG
    std::vector<std::string> png_files = listPngFiles(input_dir);
    if (png_files.empty()) {
        std::cerr << "[ERROR] No PNG files found in " << input_dir << std::endl;
        return 1;
    }
    std::cout << "Found " << png_files.size() << " PNG files." << std::endl;

    for (size_t fi = 0; fi < png_files.size(); fi++) {
        std::string img_path = input_dir + "/" + png_files[fi];
        std::cout << "[" << (fi + 1) << "/" << png_files.size() << "] "
                  << png_files[fi] << " ... " << std::flush;

        cv::Mat raw = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        if (raw.empty()) {
            std::cerr << "ERROR: cannot read " << img_path << std::endl;
            continue;
        }

        // Step 1: 偏振解码
        PolarChannelResult polar = raw2polar(raw);

        // Step 2: 对 P0/DoP, P1/sinAoP, P2/cosAoP 分别施加 4 种滤波
        std::vector<FilteredChannels> channels(NUM_ROWS);
        channels[0] = applyAllFilters(polar.dop_img, polar.S0_img);   // P0 = DoP
        channels[1] = applyAllFilters(polar.sin_img, polar.S0_img);   // P1 = sin(AoP)
        channels[2] = applyAllFilters(polar.cos_img, polar.S0_img);   // P2 = cos(AoP)

        // Step 3: 构建对比图
        cv::Mat figure;
        buildCompareFigure(channels, figure);

        // Step 4: 保存
        std::string out_path = output_dir + "/" + png_files[fi];
        cv::imwrite(out_path, figure);
        std::cout << "saved to " << out_path
                  << " (" << figure.cols << "x" << figure.rows << ")" << std::endl;
    }

    std::cout << "\nAll done. " << png_files.size() << " figures generated." << std::endl;
    return 0;
}
