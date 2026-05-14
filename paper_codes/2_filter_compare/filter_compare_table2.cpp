/*******************************************************
 * Copyright (C) 2026, Bionic Intelligence Research Team, Beihang University
 *
 * This file is part of PolarFP-VINS.
 * Author: Wei Xinran (github.com/weixr18; weixr0605@sina.com)
 * Licensed under the GNU General Public License v3.0.
 *******************************************************/

/**
 * @file filter_compare_table2.cpp
 * @brief 偏振通道滤波方法量化对比评估 — 论文表2
 *
 * 对连续帧偏振图像解码出 DoP/sin(AoP)/cos(AoP) 三个通道，
 * 分别施加 4 种滤波算法（中值/导向/双边/NLM），对每个组合：
 *   - GFTT 角点检测 → 帧平均检测数
 *   - 双向 LK 光流匹配（相邻帧）→ 帧平均匹配数
 *   - RANSAC 基础矩阵估计 → 帧平均内点数
 *   - 处理时间
 *
 * 用法:
 *   ./filter_compare_table2 <input_dir> [output.csv]
 *
 * input_dir 包含连续帧 PNG 文件（frame_00000.png, frame_00001.png ...）。
 */

#include "PolarChannel.h"
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ============================================================================
// 滤波器参数（论文表1）
// ============================================================================
const int MEDIAN_K = 3;                 // 中值滤波核大小
const int GUIDED_R = 8;                 // 导向滤波半径
const double GUIDED_EPS = 0.0001;       // 导向滤波正则化
const int BILATERAL_D = 9;              // 双边滤波直径
const double BILATERAL_SIGMA_COLOR = 200;
const double BILATERAL_SIGMA_SPACE = 30;
const double NLM_H = 50.0;
const int NLM_TEMPLATE_WINDOW = 5;
const int NLM_SEARCH_WINDOW = 21;

// ============================================================================
// GFTT 角点检测参数
// ============================================================================
const int GFTT_MAX_CORNERS = 500;
const double GFTT_QUALITY = 0.01;
const double GFTT_MIN_DIST = 10;
const int GFTT_BLOCK_SIZE = 3;

// ============================================================================
// LK 光流参数
// ============================================================================
const int LK_WIN_SIZE = 21;
const int LK_MAX_LEVEL = 3;
const double LK_BACK_DIST_THRESH = 0.5;

// ============================================================================
// RANSAC 基础矩阵参数
// ============================================================================
const double RANSAC_THRESH = 1.0;
const double RANSAC_CONFIDENCE = 0.99;

// ============================================================================
// 通道与滤波方法定义
// ============================================================================
const int NUM_CHS = 3;
const int NUM_FLTS = 5;  // Raw, Median, Guided, Bilateral, NLM

static const char* CH_NAMES[] = {"DoP", "sinAoP", "cosAoP"};
static const char* FLT_NAMES[] = {"Raw", "Median", "Guided", "Bilateral", "NLM"};

// ============================================================================
// 文件列表
// ============================================================================
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

// ============================================================================
// 获取滤波后图像
// ============================================================================
static cv::Mat getFilteredImage(const PolarChannelResult& polar, int ch, int fl) {
    const cv::Mat* src = nullptr;
    if (ch == 0)      src = &polar.dop_img;
    else if (ch == 1) src = &polar.sin_img;
    else              src = &polar.cos_img;

    switch (fl) {
        case 0:  // Raw
            return src->clone();

        case 1: {  // Median: 3x3, 3 rounds
            cv::Mat med = src->clone();
            for (int i = 0; i < 3; i++)
                cv::medianBlur(med, med, MEDIAN_K);
            return med;
        }

        case 2: {  // Guided (guided by S0)
            cv::Mat ch_f, s0_f, guided_f;
            src->convertTo(ch_f, CV_64F, 1.0 / 255.0);
            polar.S0_img.convertTo(s0_f, CV_64F, 1.0 / 255.0);
            guided_f = guidedFilterSingle(s0_f, ch_f, GUIDED_R, GUIDED_EPS);
            cv::Mat result;
            guided_f.convertTo(result, CV_8U, 255.0);
            return result;
        }

        case 3: {  // Bilateral
            cv::Mat result;
            cv::bilateralFilter(*src, result, BILATERAL_D,
                                BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE);
            return result;
        }

        case 4: {  // NLM
            cv::Mat result;
            cv::fastNlMeansDenoising(*src, result, NLM_H,
                                     NLM_TEMPLATE_WINDOW, NLM_SEARCH_WINDOW);
            return result;
        }

        default:
            return cv::Mat();
    }
}

// ============================================================================
// 双向 LK 光流匹配 — 返回匹配上的特征点数量
// ============================================================================
static int matchBidirectionalLK(const cv::Mat& prev_img, const cv::Mat& cur_img,
                                const std::vector<cv::Point2f>& prev_pts,
                                std::vector<cv::Point2f>& matched_prev,
                                std::vector<cv::Point2f>& matched_cur) {
    matched_prev.clear();
    matched_cur.clear();
    if (prev_pts.empty())
        return 0;

    std::vector<cv::Point2f> cur_pts;
    std::vector<uchar> status;
    std::vector<float> err;

    // Forward LK
    cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts,
                             status, err,
                             cv::Size(LK_WIN_SIZE, LK_WIN_SIZE), LK_MAX_LEVEL);

    // Backward check
    std::vector<uchar> rev_status;
    std::vector<cv::Point2f> rev_pts = prev_pts;  // initial guess for backward flow
    cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, rev_pts,
                             rev_status, err,
                             cv::Size(LK_WIN_SIZE, LK_WIN_SIZE), 1,
                             cv::TermCriteria(cv::TermCriteria::COUNT +
                                              cv::TermCriteria::EPS, 30, 0.01),
                             cv::OPTFLOW_USE_INITIAL_FLOW);

    int count = 0;
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i] && rev_status[i]) {
            double dx = prev_pts[i].x - rev_pts[i].x;
            double dy = prev_pts[i].y - rev_pts[i].y;
            if (dx * dx + dy * dy <= LK_BACK_DIST_THRESH * LK_BACK_DIST_THRESH) {
                count++;
                matched_prev.push_back(prev_pts[i]);
                matched_cur.push_back(cur_pts[i]);
            }
        }
    }
    return count;
}

// ============================================================================
// RANSAC 基础矩阵内点统计
// ============================================================================
static int countFMInliers(const std::vector<cv::Point2f>& pts1,
                          const std::vector<cv::Point2f>& pts2) {
    if (pts1.size() < 8)
        return 0;
    std::vector<uchar> mask;
    cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC,
                           RANSAC_THRESH, RANSAC_CONFIDENCE, mask);
    return cv::countNonZero(mask);
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_dir> [output.csv]" << std::endl;
        return 1;
    }

    std::string input_dir = argv[1];
    std::string csv_path = (argc > 2) ? argv[2] : "";

    // Extract dataset name from input directory
    std::string dataset_name = input_dir;
    if (dataset_name.back() == '/')
        dataset_name.pop_back();
    size_t pos = dataset_name.find_last_of('/');
    if (pos != std::string::npos)
        dataset_name = dataset_name.substr(pos + 1);

    // ------------------------------------------------------------------
    // List PNG files
    // ------------------------------------------------------------------
    std::vector<std::string> files = listPngFiles(input_dir);
    if (files.empty()) {
        std::cerr << "[ERROR] No PNG files found in " << input_dir << std::endl;
        return 1;
    }

    size_t N = std::min(files.size(), size_t(200));
    if (files.size() > N)
        files.resize(N);

    std::cout << "Dataset: " << dataset_name
              << "  |  Frames: " << N << std::endl;

    // ------------------------------------------------------------------
    // Accumulators: [channel][filter]
    // ------------------------------------------------------------------
    long long detect_sum[NUM_CHS][NUM_FLTS] = {{0}};
    long long match_sum[NUM_CHS][NUM_FLTS] = {{0}};
    long long inlier_sum[NUM_CHS][NUM_FLTS] = {{0}};
    double    time_sum[NUM_CHS][NUM_FLTS] = {{0.0}};

    // Previous-frame data
    cv::Mat              prev_filt[NUM_CHS][NUM_FLTS];
    std::vector<cv::Point2f> prev_pts[NUM_CHS][NUM_FLTS];
    bool has_prev = false;

    // ------------------------------------------------------------------
    // Process each frame
    // ------------------------------------------------------------------
    for (size_t fi = 0; fi < N; fi++) {
        std::string img_path = input_dir + "/" + files[fi];
        cv::Mat raw = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        if (raw.empty()) {
            std::cerr << "\n[ERROR] Cannot read " << img_path << std::endl;
            continue;
        }

        // Decode polarization channels
        PolarChannelResult polar = raw2polar(raw);

        for (int ch = 0; ch < NUM_CHS; ch++) {
            for (int fl = 0; fl < NUM_FLTS; fl++) {
                auto t0 = std::chrono::high_resolution_clock::now();

                // Apply filter
                cv::Mat cur_img = getFilteredImage(polar, ch, fl);

                // GFTT corner detection
                std::vector<cv::Point2f> corners;
                cv::goodFeaturesToTrack(cur_img, corners,
                                        GFTT_MAX_CORNERS, GFTT_QUALITY,
                                        GFTT_MIN_DIST, cv::noArray(),
                                        GFTT_BLOCK_SIZE);
                detect_sum[ch][fl] += static_cast<long long>(corners.size());

                // Bidirectional LK matching with previous frame
                if (has_prev && !prev_pts[ch][fl].empty()) {
                    std::vector<cv::Point2f> matched_prev, matched_cur;
                    int n_match = matchBidirectionalLK(
                        prev_filt[ch][fl], cur_img, prev_pts[ch][fl],
                        matched_prev, matched_cur);
                    match_sum[ch][fl] += n_match;
                    if (n_match >= 8) {
                        int n_inlier = countFMInliers(matched_prev, matched_cur);
                        inlier_sum[ch][fl] += n_inlier;
                    }
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                double dt = std::chrono::duration_cast<
                    std::chrono::microseconds>(t1 - t0).count() / 1000.0;
                time_sum[ch][fl] += dt;

                // Store for next frame
                cur_img.copyTo(prev_filt[ch][fl]);
                prev_pts[ch][fl] = std::move(corners);
            }
        }

        has_prev = true;

        if ((fi + 1) % 20 == 0 || fi == 0)
            std::cout << "\r  Processed " << (fi + 1) << "/" << N << " frames"
                      << std::flush;
    }
    std::cout << "\r  Processed " << N << "/" << N << " frames - done."
              << std::endl;

    // ------------------------------------------------------------------
    // Output results
    // ------------------------------------------------------------------
    int detect_cnt = N;
    int match_cnt  = static_cast<int>(N - 1);
    if (match_cnt < 1) match_cnt = 1;

    printf("\n");
    printf("===============================================================================\n");
    printf("Dataset: %s\n", dataset_name.c_str());
    printf("Frames: %zu  (matching pairs: %d)\n", N, match_cnt);
    printf("===============================================================================\n");
    printf("%-12s %-15s %12s %12s %12s %14s\n",
           "Channel", "Filter", "Avg Detect", "Avg Match", "Avg Inlier", "Avg Time(ms)");
    printf("------------------------------------------------------------------------------\n");

    for (int ch = 0; ch < NUM_CHS; ch++) {
        for (int fl = 0; fl < NUM_FLTS; fl++) {
            double avg_detect = static_cast<double>(detect_sum[ch][fl]) / detect_cnt;
            double avg_match  = static_cast<double>(match_sum[ch][fl])  / match_cnt;
            double avg_inlier = static_cast<double>(inlier_sum[ch][fl]) / match_cnt;
            double avg_time   = time_sum[ch][fl] / detect_cnt;

            printf("%-12s %-15s %12.1f %12.1f %12.1f %14.2f\n",
                   CH_NAMES[ch], FLT_NAMES[fl], avg_detect, avg_match, avg_inlier, avg_time);
        }
    }
    printf("===============================================================================\n");

    // ------------------------------------------------------------------
    // CSV output
    // ------------------------------------------------------------------
    if (!csv_path.empty()) {
        std::ofstream csv(csv_path);
        if (csv.is_open()) {
            csv << "dataset,channel,filter,avg_detect,avg_match,avg_inlier,avg_time_ms\n";
            for (int ch = 0; ch < NUM_CHS; ch++) {
                for (int fl = 0; fl < NUM_FLTS; fl++) {
                    double avg_detect = static_cast<double>(detect_sum[ch][fl]) / detect_cnt;
                    double avg_match  = static_cast<double>(match_sum[ch][fl])  / match_cnt;
                    double avg_inlier = static_cast<double>(inlier_sum[ch][fl]) / match_cnt;
                    double avg_time   = time_sum[ch][fl] / detect_cnt;
                    csv << dataset_name << ","
                        << CH_NAMES[ch] << ","
                        << FLT_NAMES[fl] << ","
                        << avg_detect << ","
                        << avg_match << ","
                        << avg_inlier << ","
                        << avg_time << "\n";
                }
            }
            csv.close();
            std::cout << "CSV saved to: " << csv_path << std::endl;
        } else {
            std::cerr << "[ERROR] Cannot write " << csv_path << std::endl;
        }
    }

    return 0;
}
