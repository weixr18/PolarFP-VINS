/**
 * @file test_guide_filter.cpp
 * @brief 导向滤波+组合滤波方案对比测试节点
 *
 * 三种滤波方案：
 *   Mode 0: 各通道 -> 导向滤波
 *   Mode 1: 各通道 -> 导向滤波 -> 双边滤波
 *   Mode 2: 各通道 -> 导向滤波 -> 双边滤波 -> 拉普拉斯锐化
 *
 * 编译运行：
 *   rosrun polarfp_vins test_guide_filter
 *
 * 订阅话题：
 *   /arena_cam_qc2/image_raw (硬编码)
 *
 * 键盘控制：
 *   w/W   - 切换模式
 *   +/-   - 导向滤波半径
 *   []    - 导向滤波 eps
 *   b/B   - 双边滤波 sigmaColor (Mode 1/2)
 *   n/N   - 双边滤波 sigmaSpace (Mode 1/2)
 *   a/A   - 拉普拉斯锐化强度 alpha (Mode 2)
 *   f     - 开关特征点叠加
 *   s     - 保存当前视图
 *   ESC/Q - 退出
 */

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>

#include "../featureTracker/PolarChannel.h"

// ============================================================================
// 滤波器
// ============================================================================

/**
 * @brief 双边滤波 + 拉普拉斯锐化
 *
 * 双边平滑保持边缘 → 拉普拉斯提取细节 → 叠加锐化
 */
static cv::Mat bilateralLaplacianSharpen(const cv::Mat& src,
    double sigmaColor = 75, double sigmaSpace = 75, double alpha = 1.5) {
    cv::Mat smooth, laplacian;

    cv::bilateralFilter(src, smooth, -1, sigmaColor, sigmaSpace);

    cv::Laplacian(smooth, laplacian, CV_16S, 3);

    cv::Mat absLap;
    cv::convertScaleAbs(laplacian, absLap);

    cv::Mat sharpened;
    addWeighted(smooth, 1.0, absLap, -alpha / 255.0, 0, sharpened, CV_8U);

    return sharpened;
}

// ============================================================================
// 滤波流水线
// ============================================================================

struct FilterResult {
    cv::Mat dop_out, sin_out, cos_out;
    cv::Mat S0_img;
    cv::Mat dop_raw, sin_raw, cos_raw;
};

/**
 * @brief 对偏振解码结果施加导向滤波 + 可选后处理
 *
 * @param mode        0=guided, 1=guided+bilateral, 2=guided+bilateral+laplacian
 */
static FilterResult applyFilters(const PolarChannelResult& polar,
                                 int guided_r, double guided_eps,
                                 int mode,
                                 double bilateral_sigmaColor,
                                 double bilateral_sigmaSpace,
                                 double lap_alpha) {
    FilterResult res;

    polar.dop_img.copyTo(res.dop_raw);
    polar.sin_img.copyTo(res.sin_raw);
    polar.cos_img.copyTo(res.cos_raw);
    polar.S0_img.copyTo(res.S0_img);

    // Step 1: 导向滤波（以 S0 为引导图）
    cv::Mat dop_f, sin_f, cos_f, s0_f;
    polar.dop_img.convertTo(dop_f, CV_64F, 1.0 / 255.0);
    polar.sin_img.convertTo(sin_f, CV_64F, 1.0 / 255.0);
    polar.cos_img.convertTo(cos_f, CV_64F, 1.0 / 255.0);
    polar.S0_img.convertTo(s0_f, CV_64F, 1.0 / 255.0);

    cv::Mat dop_g = guidedFilterSingle(s0_f, dop_f, guided_r, guided_eps);
    cv::Mat sin_g = guidedFilterSingle(s0_f, sin_f, guided_r, guided_eps);
    cv::Mat cos_g = guidedFilterSingle(s0_f, cos_f, guided_r, guided_eps);

    cv::Mat dop8, sin8, cos8;
    dop_g.convertTo(dop8, CV_8U, 255.0);
    sin_g.convertTo(sin8, CV_8U, 255.0);
    cos_g.convertTo(cos8, CV_8U, 255.0);

    // Step 2: 后处理（根据 mode）
    if (mode == 0) {
        // 仅导向滤波
        dop8.copyTo(res.dop_out);
        sin8.copyTo(res.sin_out);
        cos8.copyTo(res.cos_out);
    } else if (mode == 1) {
        // 导向滤波 + 双边滤波
        cv::bilateralFilter(dop8, res.dop_out, -1, bilateral_sigmaColor, bilateral_sigmaSpace);
        cv::bilateralFilter(sin8, res.sin_out, -1, bilateral_sigmaColor, bilateral_sigmaSpace);
        cv::bilateralFilter(cos8, res.cos_out, -1, bilateral_sigmaColor, bilateral_sigmaSpace);
    } else {
        // 导向滤波 + 双边滤波 + 拉普拉斯锐化
        res.dop_out = bilateralLaplacianSharpen(dop8, bilateral_sigmaColor, bilateral_sigmaSpace, lap_alpha);
        res.sin_out = bilateralLaplacianSharpen(sin8, bilateral_sigmaColor, bilateral_sigmaSpace, lap_alpha);
        res.cos_out = bilateralLaplacianSharpen(cos8, bilateral_sigmaColor, bilateral_sigmaSpace, lap_alpha);
    }

    return res;
}

// ============================================================================
// 可视化
// ============================================================================

static const char* modeNames[] = {"Guided", "Guided+Bilateral", "Guided+Bilateral+Laplacian"};

static void buildCompareView(const PolarChannelResult& polar, const FilterResult& fr,
                             int mode, cv::Mat& out) {
    const cv::Scalar txtColor(0, 255, 0);
    auto labelAndGray = [&](const cv::Mat& gray, const std::string& lbl) -> cv::Mat {
        cv::Mat c;
        cv::cvtColor(gray, c, cv::COLOR_GRAY2BGR);
        cv::putText(c, lbl, cv::Point(5, 14), cv::FONT_HERSHEY_SIMPLEX, 0.35, txtColor, 1, cv::LINE_AA);
        return c;
    };

    // 第 0 行：原始偏振通道
    cv::Mat r0c0 = labelAndGray(polar.dop_img, "DoP raw");
    cv::Mat r0c1 = labelAndGray(polar.sin_img, "sin(AoP) raw");
    cv::Mat r0c2 = labelAndGray(polar.cos_img, "cos(AoP) raw");
    cv::Mat row0, row0r;
    cv::hconcat(r0c0, r0c1, row0);
    cv::hconcat(row0, r0c2, row0r);

    // 第 1 行：当前模式结果
    const std::string suffix = std::string(" ") + modeNames[mode];
    cv::Mat r1c0 = labelAndGray(fr.dop_out, "DoP" + suffix);
    cv::Mat r1c1 = labelAndGray(fr.sin_out, "sin(AoP)" + suffix);
    cv::Mat r1c2 = labelAndGray(fr.cos_out, "cos(AoP)" + suffix);
    cv::Mat row1, row1r;
    cv::hconcat(r1c0, r1c1, row1);
    cv::hconcat(row1, r1c2, row1r);

    cv::vconcat(row0r, row1r, out);
}

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

static void buildFeatureView(const FilterResult& fr, cv::Mat& out,
                             int max_cnt, double quality, double min_dist,
                             int mode) {
    cv::Mat s0_f, dop_rf, dop_filt;
    detectAndDrawFeatures(fr.S0_img, s0_f, "S0", max_cnt, quality, min_dist);
    detectAndDrawFeatures(fr.dop_raw, dop_rf, "DoP raw", max_cnt, quality, min_dist);

    std::string filtLabel = std::string("DoP ") + modeNames[mode];
    detectAndDrawFeatures(fr.dop_out, dop_filt, filtLabel, max_cnt, quality, min_dist);

    cv::Mat row1, row2;
    cv::hconcat(s0_f, dop_rf, row1);
    cv::hconcat(dop_filt, cv::Mat::zeros(dop_filt.size(), dop_filt.type()), row2);
    cv::vconcat(row1, row2, out);
}

// ============================================================================
// ROS 节点
// ============================================================================

const std::string IMAGE_TOPIC = "/arena_cam_qc2/image_raw";

std::mutex img_mutex;
cv::Mat latest_raw;
bool new_frame = false;

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

int main(int argc, char** argv) {
    ros::init(argc, argv, "test_guide_filter");
    ROS_INFO("=== Polar Guided Filter Test ===");
    ROS_INFO("Subscribing to %s", IMAGE_TOPIC.c_str());

    // 导向滤波参数
    int guided_r = 8;
    double guided_eps = 0.0001;

    // 双边滤波参数
    double bilateral_sigmaColor = 75;
    double bilateral_sigmaSpace = 75;

    // 拉普拉斯锐化强度
    double lap_alpha = 1.5;

    // 特征点检测参数
    int max_cnt = 200;
    double quality = 0.01;
    double min_dist = 10;

    int mode = 0;
    ROS_INFO("Mode: %s (w/W to switch)", modeNames[mode]);
    ROS_INFO("Guided: r=%d eps=%.3f | Bilateral: sc=%.0f ss=%.0f | Lap alpha=%.1f",
             guided_r, guided_eps, bilateral_sigmaColor, bilateral_sigmaSpace, lap_alpha);

    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe(IMAGE_TOPIC, 10, imageCallback);

    const std::string win_name = "Polar Filter Test";
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(win_name, 1280, 720);

    bool show_features = false;
    int frame_count = 0;
    int saved_count = 0;

    while (ros::ok()) {
        ros::spinOnce();

        cv::Mat raw;
        {
            std::lock_guard<std::mutex> lock(img_mutex);
            if (new_frame) {
                latest_raw.copyTo(raw);
                new_frame = false;
            }
        }

        cv::Mat display;
        if (!raw.empty()) {
            auto t_start = std::chrono::high_resolution_clock::now();

            PolarChannelResult polar = raw2polar(raw);
            FilterResult fr = applyFilters(polar, guided_r, guided_eps,
                                           mode, bilateral_sigmaColor, bilateral_sigmaSpace,
                                           lap_alpha);

            auto t_end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
            printf("filter time: %.2f ms\n", elapsed_ms);

            if (show_features) {
                buildFeatureView(fr, display, max_cnt, quality, min_dist, mode);
            } else {
                buildCompareView(polar, fr, mode, display);
            }

            cv::setWindowTitle(win_name, std::string("Mode: ") + modeNames[mode]);
            cv::imshow(win_name, display);
            frame_count++;
        }

        int key = cv::waitKey(20) & 0xFF;

        if (key == 27 || key == 'q') {
            break;
        } else if (key == 'w') {
            mode = (mode + 1) % 3;
            ROS_INFO("Mode -> %s", modeNames[mode]);
        } else if (key == 'W') {
            mode = (mode + 2) % 3;
            ROS_INFO("Mode -> %s", modeNames[mode]);
        } else if (key == '=' || key == '+') {
            guided_r = std::min(guided_r + 1, 20);
            ROS_INFO("guided_r -> %d", guided_r);
        } else if (key == '-') {
            guided_r = std::max(guided_r - 1, 1);
            ROS_INFO("guided_r -> %d", guided_r);
        } else if (key == '[') {
            guided_eps = std::max(guided_eps * 0.5, 0.0001);
            ROS_INFO("guided_eps -> %.4f", guided_eps);
        } else if (key == ']') {
            guided_eps = std::min(guided_eps * 2.0, 1.0);
            ROS_INFO("guided_eps -> %.4f", guided_eps);
        } else if (key == 'b') {
            bilateral_sigmaColor = std::max(bilateral_sigmaColor - 5.0, 1.0);
            ROS_INFO("bilateral_sigmaColor -> %.0f", bilateral_sigmaColor);
        } else if (key == 'B') {
            bilateral_sigmaColor = std::min(bilateral_sigmaColor + 5.0, 300.0);
            ROS_INFO("bilateral_sigmaColor -> %.0f", bilateral_sigmaColor);
        } else if (key == 'n') {
            bilateral_sigmaSpace = std::max(bilateral_sigmaSpace - 5.0, 1.0);
            ROS_INFO("bilateral_sigmaSpace -> %.0f", bilateral_sigmaSpace);
        } else if (key == 'N') {
            bilateral_sigmaSpace = std::min(bilateral_sigmaSpace + 5.0, 200.0);
            ROS_INFO("bilateral_sigmaSpace -> %.0f", bilateral_sigmaSpace);
        } else if (key == 'a') {
            lap_alpha = std::max(lap_alpha - 0.1, 0.0);
            ROS_INFO("lap_alpha -> %.1f", lap_alpha);
        } else if (key == 'A') {
            lap_alpha = std::min(lap_alpha + 0.1, 5.0);
            ROS_INFO("lap_alpha -> %.1f", lap_alpha);
        } else if (key == 'f') {
            show_features = !show_features;
            ROS_INFO("Feature overlay: %s", show_features ? "ON" : "OFF");
        } else if (key == 's') {
            std::string name = "guide_frame_" + std::to_string(frame_count) + "_" + std::to_string(saved_count) + ".png";
            cv::imwrite(name, display);
            ROS_INFO("Saved %s", name.c_str());
            saved_count++;
        }
    }

    cv::destroyAllWindows();
    ROS_INFO("Processed %d frames", frame_count);
    return 0;
}
