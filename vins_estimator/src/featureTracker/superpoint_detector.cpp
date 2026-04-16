/*******************************************************
 * Copyright (C) 2026, Bionic Intelligence Research Team, Beihang University
 *
 * This file is part of PolarFP-VINS.
 * Auther: Wei Xinran (github.com/weixr18; weixr0605@sina.com)
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

/**
 * @file superpoint_detector.cpp
 * @brief SuperPoint detector implementing FeatureDetector interface.
 *
 * Supports batch inference for multi-channel polar frontend:
 *   detectBatchForChannels() — batch N images → per-channel post-process → cache
 *   detect() — returns cached per-channel result (batch mode) or single-image inference (fallback)
 */

#ifdef USE_SUPERPOINT

#include "superpoint_detector.h"

#include <torch/script.h>
#include <cuda_runtime.h>
#include <iostream>

// Internal structs (not exposed in header)
struct SuperPointDetectorImpl {
    torch::jit::script::Module model;
    torch::Device device;

    SuperPointDetectorImpl(torch::jit::script::Module m, torch::Device d)
        : model(std::move(m)), device(std::move(d)) {}
};

struct SuperPointKeypoint {
    float x, y, score;
};

// =============================================
// Constructor / Destructor
// =============================================

SuperPointFeatureDetector::SuperPointFeatureDetector(
    const std::string& model_path, bool use_gpu,
    float kp_thresh, int nms_radius)
    : keypoint_threshold_(kp_thresh), nms_radius_(nms_radius)
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    bool actual_use_gpu = use_gpu && (device_count > 0);
    torch::Device dev(actual_use_gpu ? torch::kCUDA : torch::kCPU);

    if (actual_use_gpu) {
        std::cout << "[SuperPoint] CUDA available (" << device_count
                  << " device(s)), running on GPU" << std::endl;
    } else {
        std::cout << "[SuperPoint] Running on CPU (use_gpu=" << use_gpu
                  << ", cuda devices=" << device_count << ")" << std::endl;
    }

    try {
        impl_ = std::make_unique<SuperPointDetectorImpl>(
            torch::jit::load(model_path), dev);
        impl_->model.eval();
        impl_->model.to(impl_->device);
        std::cout << "[SuperPoint] Model loaded from " << model_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "[SuperPoint] Could not load model: " << e.what() << std::endl;
        initialized_ = false;
        return;
    }

    // Warm-up inference to initialize CUDA kernels
    if (impl_->device.is_cuda()) {
        auto dummy = torch::ones({1, 1, 480, 640}, torch::kFloat32).to(impl_->device);
        impl_->model.forward({dummy});
        cudaDeviceSynchronize();
        std::cout << "[SuperPoint] Warm-up inference done" << std::endl;
    }

    initialized_ = true;
}

SuperPointFeatureDetector::~SuperPointFeatureDetector() = default;

// =============================================
// Internal: single-image keypoint detection
// =============================================

std::vector<cv::Point2f> SuperPointFeatureDetector::detectSingleImage(
    const cv::Mat& image) const
{
    if (!initialized_ || image.empty()) return {};

    // Preprocess: 8UC1 → float32 [0, 1]
    cv::Mat img_float;
    image.convertTo(img_float, CV_32FC1, 1.0 / 255.0);

    // .clone() is important — from_blob doesn't own data
    auto img_tensor = torch::from_blob(
        img_float.data, {1, 1, image.rows, image.cols}, torch::kFloat32).clone();

    // Forward: returns tuple (detection_map, descriptor_map)
    auto output = impl_->model.forward({img_tensor.to(impl_->device)});
    auto outputs = output.toTuple();
    auto det_map = outputs->elements()[0].toTensor();

    // Post-process: squeeze batch dim, move to CPU
    auto det = det_map.squeeze(0).cpu();  // [65, H/8, W/8]

    // Softmax over 65 classes
    det = torch::softmax(det, /*dim=*/0);

    // Remove "dust" class (index 64), keep first 64 bins
    auto det_no_dust = det.slice(/*dim=*/0, /*start=*/0, /*end=*/64);

    // Get max probability and corresponding bin index per spatial location
    auto max_vals = std::get<0>(det_no_dust.max(/*dim=*/0));  // [H/8, W/8]
    auto max_indices = std::get<1>(det_no_dust.max(/*dim=*/0));

    auto h = max_vals.size(0);
    auto w = max_vals.size(1);

    auto acc = max_vals.accessor<float, 2>();
    auto indices = max_indices.accessor<int64_t, 2>();

    std::vector<SuperPointKeypoint> kpts;

    for (int64_t y = 0; y < h; ++y) {
        for (int64_t x = 0; x < w; ++x) {
            if (acc[y][x] < keypoint_threshold_) continue;

            float val = acc[y][x];
            bool is_max = true;
            for (int dy = -nms_radius_; dy <= nms_radius_; ++dy) {
                for (int dx = -nms_radius_; dx <= nms_radius_; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    int ny = y + dy, nx = x + dx;
                    if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
                        if (acc[ny][nx] > val) {
                            is_max = false;
                            break;
                        }
                    }
                }
                if (!is_max) break;
            }

            if (is_max) {
                int bin = static_cast<int>(indices[y][x]);
                int bx = bin % 8;
                int by = bin / 8;
                SuperPointKeypoint kp;
                kp.x = x * 8 + bx;
                kp.y = y * 8 + by;
                kp.score = val;
                kpts.push_back(kp);
            }
        }
    }

    // Convert to cv::Point2f
    std::vector<cv::Point2f> pts;
    pts.reserve(kpts.size());
    for (const auto& kp : kpts) {
        pts.emplace_back(kp.x, kp.y);
    }
    return pts;
}

// =============================================
// Batch inference for multi-channel
// =============================================

void SuperPointFeatureDetector::detectBatchForChannels(
    const std::vector<cv::Mat>& images,
    const std::vector<cv::Mat>& masks,
    const std::vector<int>& max_cnts)
{
    batch_cached_ = false;

    if (!initialized_ || images.empty()) return;

    int n = static_cast<int>(images.size());
    int h = images[0].rows;
    int w = images[0].cols;

    // 1. Pack images into batch tensor [N, H, W] (single-channel, no extra dim needed)
    auto batch = torch::empty({n, h, w}, torch::kFloat32);
    for (int i = 0; i < n; i++) {
        cv::Mat img_f;
        images[i].convertTo(img_f, CV_32FC1, 1.0 / 255.0);
        auto img_tensor = torch::from_blob(img_f.data, {h, w}, torch::kFloat32).clone();
        batch[i] = img_tensor;
    }

    // 2. Forward → (detection_maps, descriptor_maps)
    // Model expects 4D input [N, 1, H, W], so unsqueeze channel dim
    auto input = batch.unsqueeze(1).to(impl_->device);
    auto output = impl_->model.forward({input});
    auto det_maps = output.toTuple()->elements()[0].toTensor().cpu();  // [N, 65, H/8, W/8]

    // 3. Per-image post-processing: softmax → NMS → threshold
    std::vector<std::vector<SuperPointKeypoint>> all_kpts(n);
    for (int i = 0; i < n; i++) {
        auto det = det_maps[i];  // [65, H/8, W/8]
        det = torch::softmax(det, 0);
        auto det_no_dust = det.slice(0, 0, 64);
        auto max_vals = std::get<0>(det_no_dust.max(0));
        auto max_indices = std::get<1>(det_no_dust.max(0));

        auto hh = max_vals.size(0);
        auto ww = max_vals.size(1);
        auto acc = max_vals.accessor<float, 2>();
        auto indices = max_indices.accessor<int64_t, 2>();

        for (int64_t y = 0; y < hh; ++y) {
            for (int64_t x = 0; x < ww; ++x) {
                if (acc[y][x] < keypoint_threshold_) continue;

                float val = acc[y][x];
                bool is_max = true;
                for (int dy = -nms_radius_; dy <= nms_radius_; ++dy) {
                    for (int dx = -nms_radius_; dx <= nms_radius_; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        int ny = y + dy, nx = x + dx;
                        if (ny >= 0 && ny < hh && nx >= 0 && nx < ww) {
                            if (acc[ny][nx] > val) {
                                is_max = false;
                                break;
                            }
                        }
                    }
                    if (!is_max) break;
                }

                if (is_max) {
                    int bin = static_cast<int>(indices[y][x]);
                    int bx = bin % 8;
                    int by = bin / 8;
                    SuperPointKeypoint kp;
                    kp.x = x * 8 + bx;
                    kp.y = y * 8 + by;
                    kp.score = val;
                    all_kpts[i].push_back(kp);
                }
            }
        }
    }

    // 4. Per-channel: mask filter + max_cnt truncation → batch_results_
    batch_results_.resize(n);
    for (size_t i = 0; i < n; i++) {
        std::vector<cv::Point2f> pts;
        const auto& mask = masks[i];
        const auto& img = images[i];
        for (const auto& kp : all_kpts[i]) {
            int xr = cvRound(kp.x), yr = cvRound(kp.y);
            if (xr < 0 || xr >= img.cols || yr < 0 || yr >= img.rows) continue;
            if (!mask.empty() && mask.at<uchar>(yr, xr) == 0) continue;
            pts.emplace_back(kp.x, kp.y);
        }
        if (max_cnts[i] > 0 && static_cast<int>(pts.size()) > max_cnts[i])
            pts.resize(max_cnts[i]);
        batch_results_[i] = std::move(pts);
    }
    batch_cached_ = true;
    current_channel_idx_ = 0;
}

// =============================================
// detect() — FeatureDetector interface
// =============================================

std::vector<cv::Point2f> SuperPointFeatureDetector::detect(
    const cv::Mat& image, const cv::Mat& mask, int max_cnt) const
{
    if (batch_cached_ && current_channel_idx_ < batch_results_.size()) {
        return batch_results_[current_channel_idx_++];
    }

    // Fallback: single-image inference (non-polar mode)
    std::vector<cv::Point2f> pts = detectSingleImage(image);

    // Mask filter
    if (!mask.empty()) {
        std::vector<cv::Point2f> masked_pts;
        masked_pts.reserve(pts.size());
        for (const auto& pt : pts) {
            int xr = cvRound(pt.x), yr = cvRound(pt.y);
            if (xr >= 0 && xr < image.cols && yr >= 0 && yr < image.rows
                && mask.at<uchar>(yr, xr) > 0) {
                masked_pts.push_back(pt);
            }
        }
        pts = std::move(masked_pts);
    }

    // Top-N truncation
    if (max_cnt > 0 && static_cast<int>(pts.size()) > max_cnt)
        pts.resize(max_cnt);

    return pts;
}

#endif // USE_SUPERPOINT
