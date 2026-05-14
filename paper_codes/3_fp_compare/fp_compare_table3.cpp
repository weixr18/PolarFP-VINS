/*******************************************************
 * Copyright (C) 2026, Bionic Intelligence Research Team, Beihang University
 *
 * This file is part of PolarFP-VINS.
 * Author: Wei Xinran (github.com/weixr18; weixr0605@sina.com)
 * Licensed under the GNU General Public License v3.0.
 *******************************************************/

/**
 * @file fp_compare_table3.cpp
 * @brief 特征点检测方法量化对比评估 — 论文表3
 *
 * 对连续帧偏振图像解码出 DoP/sinAoP/cosAoP 三个通道，
 * 施加导向滤波后，分别运行 FAST / GFTT / SuperPoint 检测器，对每个组合：
 *   - 帧平均检测数
 *   - 双向 LK 光流匹配数（相邻帧）
 *   - RANSAC 本质矩阵内点数
 *   - 处理时间
 *
 * 用法:
 *   ./fp_compare_table3 <input_dir> [output.csv]
 */

#include "PolarChannel.h"
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ============================================================================
// Guided filter parameters
// ============================================================================
const int GUIDED_R = 8;
const double GUIDED_EPS = 0.0001;

// ============================================================================
// Detector parameters
// ============================================================================
const int GFTT_MAX_CORNERS = 500;
const double GFTT_QUALITY = 0.01;
const double GFTT_MIN_DIST = 10;
const int GFTT_BLOCK_SIZE = 3;

const int FAST_THRESHOLD = 10;
const int FAST_TOP_K = 500;

const float SP_KP_THRESH = 0.005f;
const int SP_NMS_RADIUS = 3;
const int SP_TOP_K = 500;

// ============================================================================
// SuperPoint model path
// ============================================================================
static const char* SP_MODEL_PATH =
    "/home/dhz/ws/vi_catkin_ws/src/PolarFP-VINS/polarfp_vins_estimator/nn/superpoint_v1.pt";

// ============================================================================
// LK optical flow parameters
// ============================================================================
const int LK_WIN_SIZE = 21;
const int LK_MAX_LEVEL = 3;
const double LK_BACK_DIST_THRESH = 0.5;

// ============================================================================
// RANSAC fundamental matrix parameters
// ============================================================================
const double RANSAC_THRESH = 1.0;
const double RANSAC_CONFIDENCE = 0.99;

// ============================================================================
// Channel and detector names
// ============================================================================
const int NUM_CHS = 3;
const int NUM_DET = 3;

static const char* CH_NAMES[] = {"DoP", "sinAoP", "cosAoP"};
static const char* DET_NAMES[] = {"FAST", "GFTT", "SuperPoint"};

// ============================================================================
// File listing
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
        if (len > 4 && strcasecmp(name + len - 4, ".png") == 0)
            files.push_back(name);
    }
    closedir(dp);
    std::sort(files.begin(), files.end());
    return files;
}

// ============================================================================
// Guided filter
// ============================================================================
static cv::Mat applyGuidedFilter(const cv::Mat& ch, const cv::Mat& s0) {
    cv::Mat ch_f, s0_f, guided_f;
    ch.convertTo(ch_f, CV_64F, 1.0 / 255.0);
    s0.convertTo(s0_f, CV_64F, 1.0 / 255.0);
    guided_f = guidedFilterSingle(s0_f, ch_f, GUIDED_R, GUIDED_EPS);
    cv::Mat result;
    guided_f.convertTo(result, CV_8U, 255.0);
    return result;
}

// ============================================================================
// FAST detection
// ============================================================================
static std::vector<cv::Point2f> detectFAST(const cv::Mat& image) {
    std::vector<cv::KeyPoint> keypoints;
    cv::FAST(image, keypoints, FAST_THRESHOLD, true);
    std::sort(keypoints.begin(), keypoints.end(),
              [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                  return a.response > b.response;
              });
    if (static_cast<int>(keypoints.size()) > FAST_TOP_K)
        keypoints.resize(FAST_TOP_K);
    std::vector<cv::Point2f> pts;
    pts.reserve(keypoints.size());
    for (const auto& kp : keypoints)
        pts.emplace_back(kp.pt);
    return pts;
}

// ============================================================================
// GFTT detection
// ============================================================================
static std::vector<cv::Point2f> detectGFTT(const cv::Mat& image) {
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(image, corners, GFTT_MAX_CORNERS,
                            GFTT_QUALITY, GFTT_MIN_DIST, cv::noArray(),
                            GFTT_BLOCK_SIZE);
    return corners;
}

// ============================================================================
// SuperPoint batch detection — returns keypoints for 3 channels
// ============================================================================
struct SPKeypoint { float x, y, score; };

static std::vector<std::vector<cv::Point2f>> detectSuperPointBatch(
    const std::vector<cv::Mat>& images,
    torch::jit::script::Module& model,
    torch::Device& device) {
    std::vector<std::vector<cv::Point2f>> results(images.size());
    if (images.empty()) return results;

    int n = static_cast<int>(images.size());
    int h = images[0].rows;
    int w = images[0].cols;

    // Pack into batch tensor [N, H, W]
    auto batch = torch::empty({n, h, w}, torch::kFloat32);
    for (int i = 0; i < n; i++) {
        cv::Mat img_f;
        images[i].convertTo(img_f, CV_32FC1, 1.0 / 255.0);
        auto img_tensor = torch::from_blob(img_f.data, {h, w}, torch::kFloat32).clone();
        batch[i] = img_tensor;
    }

    // Forward: [N, 1, H, W]
    auto input = batch.unsqueeze(1).to(device);
    auto output = model.forward({input});
    auto det_maps = output.toTuple()->elements()[0].toTensor().cpu();  // [N, 65, H/8, W/8]

    // Per-channel post-processing
    for (int i = 0; i < n; i++) {
        auto det = det_maps[i];
        det = torch::softmax(det, 0);
        auto det_no_dust = det.slice(0, 0, 64);
        auto max_vals = std::get<0>(det_no_dust.max(0));
        auto max_indices = std::get<1>(det_no_dust.max(0));

        auto hh = max_vals.size(0);
        auto ww = max_vals.size(1);
        auto acc = max_vals.accessor<float, 2>();
        auto indices = max_indices.accessor<int64_t, 2>();

        std::vector<SPKeypoint> kpts;
        for (int64_t y = 0; y < hh; ++y) {
            for (int64_t x = 0; x < ww; ++x) {
                if (acc[y][x] < SP_KP_THRESH) continue;

                float val = acc[y][x];
                bool is_max = true;
                for (int dy = -SP_NMS_RADIUS; dy <= SP_NMS_RADIUS && is_max; ++dy) {
                    for (int dx = -SP_NMS_RADIUS; dx <= SP_NMS_RADIUS; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        int ny = y + dy, nx = x + dx;
                        if (ny >= 0 && ny < hh && nx >= 0 && nx < ww) {
                            if (acc[ny][nx] > val) { is_max = false; break; }
                        }
                    }
                }
                if (!is_max) continue;

                int bin = static_cast<int>(indices[y][x]);
                kpts.push_back({x * 8.0f + (bin % 8), y * 8.0f + (bin / 8), val});
            }
        }

        std::sort(kpts.begin(), kpts.end(),
                  [](const SPKeypoint& a, const SPKeypoint& b) { return a.score > b.score; });
        if (static_cast<int>(kpts.size()) > SP_TOP_K)
            kpts.resize(SP_TOP_K);

        for (const auto& kp : kpts)
            results[i].emplace_back(kp.x, kp.y);
    }
    return results;
}

// ============================================================================
// Bidirectional LK matching
// ============================================================================
static int matchBidirectionalLK(const cv::Mat& prev_img, const cv::Mat& cur_img,
                                const std::vector<cv::Point2f>& prev_pts,
                                std::vector<cv::Point2f>& matched_prev,
                                std::vector<cv::Point2f>& matched_cur) {
    matched_prev.clear();
    matched_cur.clear();
    if (prev_pts.empty()) return 0;

    std::vector<cv::Point2f> cur_pts;
    std::vector<uchar> status;
    std::vector<float> err;

    // Forward LK
    cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts,
                             status, err,
                             cv::Size(LK_WIN_SIZE, LK_WIN_SIZE), LK_MAX_LEVEL);

    // Backward check with initial flow
    std::vector<uchar> rev_status;
    std::vector<cv::Point2f> rev_pts = prev_pts;
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
// RANSAC fundamental matrix inliers
// ============================================================================
static int countFMInliers(const std::vector<cv::Point2f>& pts1,
                           const std::vector<cv::Point2f>& pts2) {
    if (pts1.size() < 8) return 0;
    std::vector<uchar> mask;
    cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC,
                           RANSAC_THRESH, RANSAC_CONFIDENCE, mask);
    return cv::countNonZero(mask);
}

// ============================================================================
// Main
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
    if (dataset_name.back() == '/') dataset_name.pop_back();
    size_t pos = dataset_name.find_last_of('/');
    if (pos != std::string::npos)
        dataset_name = dataset_name.substr(pos + 1);

    // ------------------------------------------------------------------
    // Load SuperPoint model
    // ------------------------------------------------------------------
    std::cout << "Loading SuperPoint model from " << SP_MODEL_PATH << " ..." << std::endl;

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    bool use_gpu = (device_count > 0);
    torch::Device device(use_gpu ? torch::kCUDA : torch::kCPU);
    std::cout << "  Using " << (use_gpu ? "GPU" : "CPU") << std::endl;

    torch::jit::script::Module sp_model;
    try {
        sp_model = torch::jit::load(SP_MODEL_PATH);
        sp_model.eval();
        sp_model.to(device);
    } catch (const c10::Error& e) {
        std::cerr << "[ERROR] Failed to load SuperPoint model: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "  Model loaded." << std::endl;

    // Warm-up inference
    if (use_gpu) {
        auto dummy = torch::ones({1, 1, 480, 640}, torch::kFloat32).to(device);
        sp_model.forward({dummy});
        cudaDeviceSynchronize();
        std::cout << "  GPU warm-up done." << std::endl;
    }

    // ------------------------------------------------------------------
    // List PNG files
    // ------------------------------------------------------------------
    std::vector<std::string> files = listPngFiles(input_dir);
    if (files.empty()) {
        std::cerr << "[ERROR] No PNG files found in " << input_dir << std::endl;
        return 1;
    }
    size_t N = std::min(files.size(), size_t(200));
    if (files.size() > N) files.resize(N);

    std::cout << "Dataset: " << dataset_name << "  |  Frames: " << N << std::endl;

    // ------------------------------------------------------------------
    // Accumulators: [channel][detector]
    // ------------------------------------------------------------------
    long long detect_sum[NUM_CHS][NUM_DET] = {{0}};
    long long match_sum[NUM_CHS][NUM_DET] = {{0}};
    long long inlier_sum[NUM_CHS][NUM_DET] = {{0}};
    double    time_sum[NUM_CHS][NUM_DET] = {{0.0}};

    // Previous-frame data
    cv::Mat prev_filt[NUM_CHS];
    std::vector<cv::Point2f> prev_pts[NUM_CHS][NUM_DET];
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

        // Apply guided filter once per channel (cached)
        cv::Mat channels[NUM_CHS];
        for (int ch = 0; ch < NUM_CHS; ch++) {
            const cv::Mat* src = (ch == 0) ? &polar.dop_img :
                                 (ch == 1) ? &polar.sin_img :
                                             &polar.cos_img;
            channels[ch] = applyGuidedFilter(*src, polar.S0_img);
        }

        // --- FAST and GFTT: per-channel, per-detector ---
        for (int ch = 0; ch < NUM_CHS; ch++) {
            for (int det = 0; det < 2; det++) {  // 0=FAST, 1=GFTT
                auto t0 = std::chrono::high_resolution_clock::now();

                std::vector<cv::Point2f> corners;
                if (det == 0)
                    corners = detectFAST(channels[ch]);
                else
                    corners = detectGFTT(channels[ch]);

                detect_sum[ch][det] += static_cast<long long>(corners.size());

                if (has_prev && !prev_pts[ch][det].empty()) {
                    std::vector<cv::Point2f> matched_prev, matched_cur;
                    int n_match = matchBidirectionalLK(
                        prev_filt[ch], channels[ch], prev_pts[ch][det],
                        matched_prev, matched_cur);
                    match_sum[ch][det] += n_match;
                    if (n_match >= 8) {
                        int n_inlier = countFMInliers(matched_prev, matched_cur);
                        inlier_sum[ch][det] += n_inlier;
                    }
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                double dt = std::chrono::duration_cast<
                    std::chrono::microseconds>(t1 - t0).count() / 1000.0;
                time_sum[ch][det] += dt;

                prev_pts[ch][det] = std::move(corners);
            }
        }

        // --- SuperPoint: batch inference on 3 channels ---
        {
            auto t0 = std::chrono::high_resolution_clock::now();

            std::vector<cv::Mat> sp_images = {channels[0], channels[1], channels[2]};
            auto sp_results = detectSuperPointBatch(sp_images, sp_model, device);

            for (int ch = 0; ch < NUM_CHS; ch++) {
                int det = 2;  // SuperPoint
                detect_sum[ch][det] += static_cast<long long>(sp_results[ch].size());

                if (has_prev && !prev_pts[ch][det].empty()) {
                    std::vector<cv::Point2f> matched_prev, matched_cur;
                    int n_match = matchBidirectionalLK(
                        prev_filt[ch], channels[ch], prev_pts[ch][det],
                        matched_prev, matched_cur);
                    match_sum[ch][det] += n_match;
                    if (n_match >= 8) {
                        int n_inlier = countFMInliers(matched_prev, matched_cur);
                        inlier_sum[ch][det] += n_inlier;
                    }
                }

                prev_pts[ch][det] = std::move(sp_results[ch]);
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration_cast<
                std::chrono::microseconds>(t1 - t0).count() / 1000.0;
            // Distribute batch time evenly across channels
            for (int ch = 0; ch < NUM_CHS; ch++)
                time_sum[ch][2] += dt / NUM_CHS;
        }

        // Store filtered images for next frame
        for (int ch = 0; ch < NUM_CHS; ch++)
            channels[ch].copyTo(prev_filt[ch]);
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
    printf("%-12s %-15s %12s %12s %12s %12s %14s\n",
           "Channel", "Detector", "Avg Detect", "Avg Match", "Avg Inlier",
           "Inlier Ratio", "Avg Time(ms)");
    printf("------------------------------------------------------------------------------\n");

    for (int ch = 0; ch < NUM_CHS; ch++) {
        for (int det = 0; det < NUM_DET; det++) {
            double avg_detect = static_cast<double>(detect_sum[ch][det]) / detect_cnt;
            double avg_match  = static_cast<double>(match_sum[ch][det])  / match_cnt;
            double avg_inlier = static_cast<double>(inlier_sum[ch][det]) / match_cnt;
            double avg_time   = time_sum[ch][det] / detect_cnt;
            double inlier_ratio = (avg_match > 0) ? avg_inlier / avg_match : 0.0;

            printf("%-12s %-15s %12.1f %12.1f %12.1f %11.3f %14.2f\n",
                   CH_NAMES[ch], DET_NAMES[det],
                   avg_detect, avg_match, avg_inlier, inlier_ratio, avg_time);
        }
    }
    printf("===============================================================================\n");

    // ------------------------------------------------------------------
    // CSV output
    // ------------------------------------------------------------------
    if (!csv_path.empty()) {
        std::ofstream csv(csv_path);
        if (csv.is_open()) {
            csv << "dataset,channel,detector,avg_detect,avg_match,avg_inlier,avg_time_ms\n";
            for (int ch = 0; ch < NUM_CHS; ch++) {
                for (int det = 0; det < NUM_DET; det++) {
                    double avg_detect = static_cast<double>(detect_sum[ch][det]) / detect_cnt;
                    double avg_match  = static_cast<double>(match_sum[ch][det])  / match_cnt;
                    double avg_inlier = static_cast<double>(inlier_sum[ch][det]) / match_cnt;
                    double avg_time   = time_sum[ch][det] / detect_cnt;
                    csv << dataset_name << ","
                        << CH_NAMES[ch] << ","
                        << DET_NAMES[det] << ","
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
