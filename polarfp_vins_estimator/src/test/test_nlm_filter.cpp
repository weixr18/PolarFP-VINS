/**
 * @file test_nlm_filter.cpp
 * @brief 非局部均值（NLM）滤波器独立测试节点
 *
 * 功能概述：
 *   1. 订阅原始偏振相机图像（mono8 格式的微偏振阵列 RAW 数据）
 *   2. 通过 raw2polar() 解码出 S0、DoP、sin(AoP)、cos(AoP) 四个偏振通道
 *   3. 对 DoP / sin(AoP) / cos(AoP) 三个通道施加 fastNlMeansDenoising 滤波
 *   4. 三个 OpenCV 参数均可实时调节：h、templateWindowSize、searchWindowSize
 *   5. 支持特征点对比模式：无滤波 vs NLM 滤波后提取的 Shi-Tomasi 角点数量对比
 *
 * 编译运行：
 *   rosrun polarfp_vins test_nlm_filter
 *
 * 订阅话题：
 *   /arena_cam_q2/image_raw (硬编码，需与实际相机话题一致)
 *
 * 键盘控制（在图像窗口中操作）：
 *   SPACE  - 暂停/单步/恢复
 *   'f'    - 开关特征点对比模式
 *   '+'/'=' - 增大 NLM h 参数
 *   '-'    - 减小 NLM h 参数
 *   '['    - 减小 NLM templateWindowSize
 *   ']'    - 增大 NLM templateWindowSize
 *   '{'    - 减小 NLM searchWindowSize
 *   '}'    - 增大 NLM searchWindowSize
 *   's'    - 保存当前视图到 PNG
 *   ESC/Q  - 退出
 */

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "../featureTracker/PolarChannel.h"

// ============================================================================
// 可视化 — 构建滤波对比视图
// ============================================================================

/**
 * @brief 构建 3 行对比视图：原始 DoP/sin/cos vs NLM 滤波后
 *
 * @param polar raw2polar() 输出的原始偏振通道
 * @param dop_nlm  NLM 滤波后的 DoP 图
 * @param sin_nlm  NLM 滤波后的 sin(AoP) 图
 * @param cos_nlm  NLM 滤波后的 cos(AoP) 图
 * @param out      输出的拼接图像（BGR 格式）
 */
static void buildCompareView(const PolarChannelResult& polar,
                             const cv::Mat& dop_nlm, const cv::Mat& sin_nlm, const cv::Mat& cos_nlm,
                             cv::Mat& out) {
    const cv::Scalar txtColor(0, 255, 0);
    auto labelAndGray = [&](const cv::Mat& gray, const std::string& lbl) -> cv::Mat {
        cv::Mat c;
        cv::cvtColor(gray, c, cv::COLOR_GRAY2BGR);
        cv::putText(c, lbl, cv::Point(5, 14), cv::FONT_HERSHEY_SIMPLEX, 0.35, txtColor, 1, cv::LINE_AA);
        return c;
    };

    // 第 0 行：原始偏振通道
    cv::Mat r0c0 = labelAndGray(polar.S0_img, "S0 raw");
    cv::Mat r0c1 = labelAndGray(polar.dop_img, "DoP raw");
    cv::Mat r0c2 = labelAndGray(polar.sin_img, "sin(AoP) raw");
    cv::Mat r0c3 = labelAndGray(polar.cos_img, "cos(AoP) raw");
    cv::Mat row0, row0r;
    cv::hconcat(r0c0, r0c1, row0);
    cv::hconcat(r0c2, r0c3, row0r);
    cv::hconcat(row0, row0r, row0);

    // 第 1 行：NLM 滤波结果
    cv::Mat r1c0 = labelAndGray(polar.S0_img, "S0 (guide)");
    cv::Mat r1c1 = labelAndGray(dop_nlm, "DoP NLM");
    cv::Mat r1c2 = labelAndGray(sin_nlm, "sin(AoP) NLM");
    cv::Mat r1c3 = labelAndGray(cos_nlm, "cos(AoP) NLM");
    cv::Mat row1, row1r;
    cv::hconcat(r1c0, r1c1, row1);
    cv::hconcat(r1c2, r1c3, row1r);
    cv::hconcat(row1, row1r, row1);

    cv::vconcat(row0, row1, out);
}

// ============================================================================
// 特征点对比视图：无滤波 vs NLM 滤波
// ============================================================================

/**
 * @brief 在单通道图像上执行 Shi-Tomasi 角点检测并绘制结果
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
 * @brief 构建特征点对比视图
 *
 * 左半部分：DoP/sin/cos 无滤波提取的角点
 * 右半部分：DoP/sin/cos NLM 滤波后提取的角点
 */
static void buildFeatureView(const PolarChannelResult& polar,
                             const cv::Mat& dop_nlm, const cv::Mat& sin_nlm, const cv::Mat& cos_nlm,
                             cv::Mat& out,
                             int max_cnt, double quality, double min_dist) {
    cv::Mat dop_rf, sin_rf, cos_rf;   // raw features
    cv::Mat dop_nf, sin_nf, cos_nf;   // NLM features

    detectAndDrawFeatures(polar.dop_img, dop_rf, "DoP raw", max_cnt, quality, min_dist);
    detectAndDrawFeatures(polar.sin_img, sin_rf, "sin raw", max_cnt, quality, min_dist);
    detectAndDrawFeatures(polar.cos_img, cos_rf, "cos raw", max_cnt, quality, min_dist);

    detectAndDrawFeatures(dop_nlm, dop_nf, "DoP NLM", max_cnt, quality, min_dist);
    detectAndDrawFeatures(sin_nlm, sin_nf, "sin NLM", max_cnt, quality, min_dist);
    detectAndDrawFeatures(cos_nlm, cos_nf, "cos NLM", max_cnt, quality, min_dist);

    cv::Mat left, right;
    cv::vconcat(dop_rf, sin_rf, left);
    cv::vconcat(left, cos_rf, left);

    cv::Mat right_col, right_tmp;
    cv::vconcat(dop_nf, sin_nf, right_tmp);
    cv::vconcat(right_tmp, cos_nf, right);

    cv::hconcat(left, right, out);
}

// ============================================================================
// ROS 节点 — 主循环
// ============================================================================

/// 订阅的原始偏振相机图像话题
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
    ros::init(argc, argv, "test_nlm_filter");
    ROS_INFO("=== NLM Filter Test ===");
    ROS_INFO("Subscribing to %s", IMAGE_TOPIC.c_str());

    // NLM 三参数（均可实时调节）
    float nlm_h = 50.0f;            // 滤波强度
    int nlm_template = 5;           // 模板窗口大小
    int nlm_search = 21;            // 搜索窗口大小

    // Shi-Tomasi 角点检测参数
    int max_cnt = 200;
    double quality = 0.01;
    double min_dist = 10;

    ROS_INFO("NLM params: h=%.1f template=%d search=%d", nlm_h, nlm_template, nlm_search);

    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe(IMAGE_TOPIC, 10, imageCallback);

    const std::string win_name = "NLM Filter Test";
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(win_name, 1280, 720);

    bool show_features = false;
    bool paused = false;
    int frame_count = 0;
    int saved_count = 0;

    while (ros::ok()) {
        ros::spinOnce();

        // Thread-safe frame grab
        cv::Mat raw;
        bool has_new;
        {
            std::lock_guard<std::mutex> lock(img_mutex);
            has_new = new_frame;
            new_frame = false;
            if (has_new) {
                latest_raw.copyTo(raw);
            }
        }

        if (!has_new) {
            // No new frame — still respond to keyboard
            int key = cv::waitKey(20) & 0xFF;
            if (key == 27 || key == 'q') break;
            if (key == 32) { paused = !paused; ROS_INFO("%s", paused ? "PAUSED" : "RESUMED"); }
            if (paused) continue;
            if (key == 'f') {
                show_features = !show_features;
                ROS_INFO("Feature mode: %s", show_features ? "ON" : "OFF");
            }
            if (key == 's') {
                ROS_INFO("No frame to save yet");
            }
            continue;
        }

        // ====== 处理流水线 ======
        auto t_start = std::chrono::high_resolution_clock::now();

        PolarChannelResult polar = raw2polar(raw);

        // NLM 滤波
        cv::Mat dop_nlm, sin_nlm, cos_nlm;
        cv::fastNlMeansDenoising(polar.dop_img, dop_nlm, nlm_h, nlm_template, nlm_search);
        cv::fastNlMeansDenoising(polar.sin_img, sin_nlm, nlm_h, nlm_template, nlm_search);
        cv::fastNlMeansDenoising(polar.cos_img, cos_nlm, nlm_h, nlm_template, nlm_search);

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
        printf("NLM filter time: %.2f ms | h=%.1f tpl=%d srch=%d\n",
               elapsed_ms, nlm_h, nlm_template, nlm_search);

        cv::Mat display;
        if (show_features) {
            buildFeatureView(polar, dop_nlm, sin_nlm, cos_nlm, display, max_cnt, quality, min_dist);
        } else {
            buildCompareView(polar, dop_nlm, sin_nlm, cos_nlm, display);
        }

        // 在图像底部叠加参数信息
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1)
                << "h=" << nlm_h << " tpl=" << nlm_template << " srch=" << nlm_search
                << " | " << std::setprecision(2) << elapsed_ms << "ms";
            cv::Mat bar = cv::Mat::zeros(30, display.cols, CV_8UC3);
            cv::putText(bar, oss.str(), cv::Point(10, 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            cv::vconcat(display, bar, display);
        }

        std::string mode_str = show_features ? " [FEATURE MODE]" : "";
        std::string pause_str = paused ? " [PAUSED]" : "";
        cv::setWindowTitle(win_name, "NLM Filter" + mode_str + pause_str);

        cv::imshow(win_name, display);
        frame_count++;

        // ====== 键盘交互 ======
        int key = cv::waitKey(1) & 0xFF;

        if (key == 27 || key == 'q') {
            break;
        } else if (key == 32) {
            paused = !paused;
            ROS_INFO("%s", paused ? "PAUSED" : "RESUMED");
        } else if (paused) {
            continue;
        } else if (key == 'f') {
            show_features = !show_features;
            ROS_INFO("Feature mode: %s", show_features ? "ON" : "OFF");
        } else if (key == 's') {
            std::string name = "nlm_frame_" + std::to_string(frame_count) + "_" + std::to_string(saved_count) + ".png";
            cv::imwrite(name, display);
            ROS_INFO("Saved %s", name.c_str());
            saved_count++;
        } else if (key == '=' || key == '+') {
            nlm_h = std::min(nlm_h + 1.0f, 100.0f);
            ROS_INFO("nlm_h -> %.1f", nlm_h);
        } else if (key == '-') {
            nlm_h = std::max(nlm_h - 1.0f, 1.0f);
            ROS_INFO("nlm_h -> %.1f", nlm_h);
        } else if (key == '[') {
            nlm_template = std::max(nlm_template - 1, 1);
            ROS_INFO("nlm_template -> %d", nlm_template);
        } else if (key == ']') {
            nlm_template = std::min(nlm_template + 1, 25);
            ROS_INFO("nlm_template -> %d", nlm_template);
        } else if (key == '{') {
            nlm_search = std::max(nlm_search - 2, nlm_template + 2);
            ROS_INFO("nlm_search -> %d", nlm_search);
        } else if (key == '}') {
            nlm_search = std::min(nlm_search + 2, 51);
            ROS_INFO("nlm_search -> %d", nlm_search);
        }
    }

    cv::destroyAllWindows();
    ROS_INFO("Processed %d frames", frame_count);
    return 0;
}
