/*******************************************************
 * Copyright (C) 2026, Bionic Intelligence Research Team, Beihang University
 *
 * This file is part of PolarFP-VINS.
 * Auther: Wei Xinran (github.com/weixr18; weixr0605@sina.com)
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License
 *******************************************************/

/**
 * @file test_visualize.cpp
 * @brief 将三个偏振通道映射到RGB彩色图的可视化测试节点
 *
 * 左侧：DoP / sin(AoP) / cos(AoP) 直接作为 R / G / B 通道
 * 右侧：三个通道先以 S0 为引导图做导向滤波，再作为 R / G / B 通道
 *
 * 编译：
 *   catkin_make
 * 运行：
 *   rosrun polarfp_vins test_visualize
 *
 * 订阅：/arena_cam_qc2/image_raw （硬编码）
 *
 * 键盘：
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
// 常量与全局状态
// ============================================================================

static const std::string IMAGE_TOPIC = "/arena_cam_qc2/image_raw";

std::mutex img_mutex;
cv::Mat latest_raw;
bool new_frame = false;

// ============================================================================
// ROS 回调
// ============================================================================

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

// ============================================================================
// 可视化
// ============================================================================

/**
 * @brief 将三个灰度通道合成 RGB 彩色图
 *
 * @param ch0  R 通道（8bit）
 * @param ch1  G 通道（8bit）
 * @param ch2  B 通道（8bit）
 */
static cv::Mat mergeRGB(const cv::Mat& ch0, const cv::Mat& ch1, const cv::Mat& ch2) {
    std::vector<cv::Mat> planes = {ch0, ch1, ch2};
    cv::Mat rgb;
    cv::merge(planes, rgb);
    return rgb;
}

/**
 * @brief 构建并排对比视图：左侧 Raw RGB，右侧 Guided RGB
 */
static cv::Mat buildCompareView(const cv::Mat& rgb_raw, const cv::Mat& rgb_guided) {
    const cv::Scalar txtColor(0, 255, 0);
    cv::Mat left = rgb_raw.clone();
    cv::Mat right = rgb_guided.clone();

    cv::putText(left,  "Raw", cv::Point(5, 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, txtColor, 1, cv::LINE_AA);
    cv::putText(right, "Guided", cv::Point(5, 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, txtColor, 1, cv::LINE_AA);

    // 中间画一条白色分隔线
    cv::Mat sep(left.rows, 1, CV_8UC3, cv::Scalar(255, 255, 255));

    cv::Mat combined;
    cv::hconcat(left, sep, combined);
    cv::hconcat(combined, right, combined);
    return combined;
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char** argv) {
    ros::init(argc, argv, "test_visualize");
    ROS_INFO("=== Polar RGB Visualization Test ===");
    ROS_INFO("Subscribing to %s", IMAGE_TOPIC.c_str());

    // 导向滤波参数
    const int guided_r = 8;
    const double guided_eps = 0.0001;
    ROS_INFO("DoP/R, sin(AoP)/G, cos(AoP)/B  |  Guided r=%d eps=%.5f", guided_r, guided_eps);

    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe(IMAGE_TOPIC, 10, imageCallback);

    const std::string win_name = "Polar RGB Visualization";
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(win_name, 1400, 480);

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

        if (raw.empty()) {
            cv::waitKey(20);
            continue;
        }

        auto t_start = std::chrono::high_resolution_clock::now();

        PolarChannelResult polar = raw2polar(raw);

        // 直接合成
        cv::Mat rgb_raw = mergeRGB(polar.dop_img, polar.sin_img, polar.cos_img);

        // 导向滤波后合成
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

        cv::Mat rgb_guided = mergeRGB(dop8, sin8, cos8);

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            t_end - t_start).count() / 1000.0;
        printf("processing time: %.2f ms\n", elapsed_ms);

        cv::Mat display = buildCompareView(rgb_raw, rgb_guided);

        cv::imshow(win_name, display);
        frame_count++;

        int key = cv::waitKey(20) & 0xFF;

        if (key == 27 || key == 'q') {
            break;
        } else if (key == 's') {
            std::string name = "vis_frame_" + std::to_string(frame_count) + "_" + std::to_string(saved_count) + ".png";
            cv::imwrite(name, display);
            ROS_INFO("Saved %s", name.c_str());
            saved_count++;
        }
    }

    cv::destroyAllWindows();
    ROS_INFO("Processed %d frames", frame_count);
    return 0;
}
