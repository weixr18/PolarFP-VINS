/*******************************************************
 * Copyright (C) 2026, Bionic Intelligence Research Team, Beihang University
 *
 * This file is part of PolarFP-VINS.
 * Author: Wei Xinran (github.com/weixr18; weixr0605@sina.com)
 * Licensed under the GNU General Public License v3.0.
 *******************************************************/

/**
 * @file fp_compare_1.cpp
 * @brief 特征点检测方法对比可视化 — 3×3 特征点叠加图
 *
 * 对每组光照条件的代表性帧，在 DoP/sinAoP/cosAoP 三个偏振通道上
 * 分别运行 FAST / GFTT / SuperPoint 检测器，生成 3×3 对比图。
 *
 * 用法:
 *   ./fp_compare_1 <input_dir> <output_dir>
 */

#include "PolarChannel.h"
#include <opencv2/opencv.hpp>
#include <torch/script.h>

#include <algorithm>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <string>
#include <vector>

// ============================================================================
// Guided filter parameters (same as 2_filter_compare)
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
// Layout
// ============================================================================
static const std::vector<std::string> ROW_LABELS = {"DoP (P0)", "sinAoP (P1)", "cosAoP (P2)"};
static const std::vector<std::string> COL_LABELS = {"FAST", "GFTT", "SuperPoint"};
const int NUM_ROWS = 3;
const int NUM_COLS = 3;

const cv::Scalar KP_COLOR(0, 255, 0);  // green dots for keypoints

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
// Guided filter (single channel, S0-guided)
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
// FAST detection: threshold + sort by response + top-K
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
// SuperPoint single-image detection
// ============================================================================
struct SPKeypoint { float x, y, score; };

static std::vector<cv::Point2f> detectSuperPoint(const cv::Mat& image,
                                                   torch::jit::script::Module& model,
                                                   torch::Device& device) {
    if (image.empty()) return {};

    cv::Mat img_float;
    image.convertTo(img_float, CV_32FC1, 1.0 / 255.0);

    auto img_tensor = torch::from_blob(
        img_float.data, {1, 1, image.rows, image.cols}, torch::kFloat32).clone();

    auto output = model.forward({img_tensor.to(device)});
    auto det_map = output.toTuple()->elements()[0].toTensor();

    auto det = det_map.squeeze(0).cpu();
    det = torch::softmax(det, 0);
    auto det_no_dust = det.slice(0, 0, 64);

    auto max_vals = std::get<0>(det_no_dust.max(0));
    auto max_indices = std::get<1>(det_no_dust.max(0));

    auto h = max_vals.size(0);
    auto w = max_vals.size(1);
    auto acc = max_vals.accessor<float, 2>();
    auto indices = max_indices.accessor<int64_t, 2>();

    std::vector<SPKeypoint> kpts;
    for (int64_t y = 0; y < h; ++y) {
        for (int64_t x = 0; x < w; ++x) {
            if (acc[y][x] < SP_KP_THRESH) continue;

            float val = acc[y][x];
            bool is_max = true;
            for (int dy = -SP_NMS_RADIUS; dy <= SP_NMS_RADIUS && is_max; ++dy) {
                for (int dx = -SP_NMS_RADIUS; dx <= SP_NMS_RADIUS; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    int ny = y + dy, nx = x + dx;
                    if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
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

    std::vector<cv::Point2f> pts;
    pts.reserve(kpts.size());
    for (const auto& kp : kpts)
        pts.emplace_back(kp.x, kp.y);
    return pts;
}

// ============================================================================
// Build 3×3 comparison figure
// ============================================================================
static void buildCompareFigure(
    const std::vector<cv::Mat>& channel_imgs,
    const std::vector<std::vector<cv::Point2f>>& kpts_fast,
    const std::vector<std::vector<cv::Point2f>>& kpts_gftt,
    const std::vector<std::vector<cv::Point2f>>& kpts_sp,
    cv::Mat& canvas) {

    int cell_w = channel_imgs[0].cols;
    int cell_h = channel_imgs[0].rows;

    const int PAD = 6;
    const int GAP_H = 10;
    const int GAP_V = 10;
    const int TOP_MARGIN = 60;
    const int LEFT_MARGIN = 140;
    const int HEADER_H = 36;

    int canvas_w = LEFT_MARGIN + NUM_COLS * (cell_w + 2 * PAD) + (NUM_COLS - 1) * GAP_H;
    int canvas_h = TOP_MARGIN + HEADER_H + NUM_ROWS * (cell_h + 2 * PAD) + (NUM_ROWS - 1) * GAP_V;

    canvas = cv::Mat(canvas_h, canvas_w, CV_8UC3, cv::Scalar(255, 255, 255));
    int font = cv::FONT_HERSHEY_SIMPLEX;

    // Title
    {
        std::string title = "Feature Detector Comparison (Guided Filter, r=8)";
        int baseline = 0;
        cv::Size ts = cv::getTextSize(title, font, 0.7, 2, &baseline);
        cv::putText(canvas, title,
                    cv::Point((canvas_w - ts.width) / 2, TOP_MARGIN - 18),
                    font, 0.7, cv::Scalar(0, 0, 0), 2);
    }

    // Column headers
    for (int c = 0; c < NUM_COLS; c++) {
        int cx = LEFT_MARGIN + c * (cell_w + 2 * PAD + GAP_H) + PAD + cell_w / 2;
        int baseline = 0;
        cv::Size ts = cv::getTextSize(COL_LABELS[c], font, 0.65, 2, &baseline);
        cv::putText(canvas, COL_LABELS[c],
                    cv::Point(cx - ts.width / 2, TOP_MARGIN + HEADER_H - 8),
                    font, 0.65, cv::Scalar(0, 0, 0), 2);
    }

    // Row labels + cells
    for (int r = 0; r < NUM_ROWS; r++) {
        int ry = TOP_MARGIN + HEADER_H + r * (cell_h + 2 * PAD + GAP_V) + PAD + cell_h / 2;
        int baseline = 0;
        cv::Size ts = cv::getTextSize(ROW_LABELS[r], font, 0.55, 1, &baseline);
        cv::putText(canvas, ROW_LABELS[r],
                    cv::Point(LEFT_MARGIN - ts.width - 12, ry + ts.height / 3),
                    font, 0.55, cv::Scalar(0, 0, 0), 1);

        const std::vector<cv::Point2f>* kpt_sets[3] = {
            &kpts_fast[r], &kpts_gftt[r], &kpts_sp[r]
        };

        for (int c = 0; c < NUM_COLS; c++) {
            int x0 = LEFT_MARGIN + c * (cell_w + 2 * PAD + GAP_H);
            int y0 = TOP_MARGIN + HEADER_H + r * (cell_h + 2 * PAD + GAP_V);

            cv::rectangle(canvas,
                          cv::Rect(x0, y0, cell_w + 2 * PAD, cell_h + 2 * PAD),
                          cv::Scalar(200, 200, 200), cv::FILLED);

            // Convert channel to BGR and draw keypoints
            cv::Mat bgr;
            cv::cvtColor(channel_imgs[r], bgr, cv::COLOR_GRAY2BGR);
            for (const auto& pt : *kpt_sets[c])
                cv::circle(bgr, pt, 2, KP_COLOR, -1);

            bgr.copyTo(canvas(cv::Rect(x0 + PAD, y0 + PAD, cell_w, cell_h)));

            // Keypoint count in bottom-left corner
            std::string cnt_str = std::to_string(kpt_sets[c]->size()) + " kpts";
            cv::putText(canvas, cnt_str,
                        cv::Point(x0 + PAD + 4, y0 + PAD + cell_h - 6),
                        font, 0.35, cv::Scalar(0, 255, 0), 1);
        }
    }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_dir> <output_dir>" << std::endl;
        return 1;
    }

    std::string input_dir = argv[1];
    std::string output_dir = argv[2];

    std::string mkdir_cmd = "mkdir -p " + output_dir;
    system(mkdir_cmd.c_str());

    // --- Load SuperPoint model ---
    std::cout << "Loading SuperPoint model from " << SP_MODEL_PATH << " ..." << std::endl;
    torch::jit::script::Module sp_model;
    try {
        sp_model = torch::jit::load(SP_MODEL_PATH);
        sp_model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "[ERROR] Failed to load SuperPoint model: " << e.what() << std::endl;
        return 1;
    }
    torch::Device device(torch::kCPU);
    sp_model.to(device);
    std::cout << "  Model loaded (CPU)." << std::endl;

    // List PNG files
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

        // Decode polarization
        PolarChannelResult polar = raw2polar(raw);

        // Apply guided filter to DoP/sin/cos
        std::vector<cv::Mat> channels(3);
        channels[0] = applyGuidedFilter(polar.dop_img, polar.S0_img);
        channels[1] = applyGuidedFilter(polar.sin_img, polar.S0_img);
        channels[2] = applyGuidedFilter(polar.cos_img, polar.S0_img);

        // Detect features on each channel
        std::vector<std::vector<cv::Point2f>> kpts_fast(3), kpts_gftt(3), kpts_sp(3);
        for (int ch = 0; ch < 3; ch++) {
            kpts_fast[ch] = detectFAST(channels[ch]);
            kpts_gftt[ch] = detectGFTT(channels[ch]);
            kpts_sp[ch]   = detectSuperPoint(channels[ch], sp_model, device);
        }

        // Build comparison figure
        cv::Mat figure;
        buildCompareFigure(channels, kpts_fast, kpts_gftt, kpts_sp, figure);

        // Save
        std::string stem = png_files[fi].substr(0, png_files[fi].rfind('.'));
        std::string out_path = output_dir + "/feature_points_" + stem + ".png";
        cv::imwrite(out_path, figure);
        std::cout << "saved (" << figure.cols << "x" << figure.rows << ")" << std::endl;
    }

    std::cout << "\nAll done. " << png_files.size() << " figures generated." << std::endl;
    return 0;
}
