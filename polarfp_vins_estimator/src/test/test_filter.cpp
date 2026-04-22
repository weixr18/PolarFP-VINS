/**
 * @file test_filter.cpp
 * @brief 偏振通道滤波效果对比测试节点
 *
 * 功能概述：
 *   1. 订阅原始偏振相机图像（mono8 格式的微偏振阵列 RAW 数据）
 *   2. 通过 raw2polar() 解码出 S0、DoP、sin(AoP)、cos(AoP) 四个偏振通道
 *   3. 对 DoP / sin(AoP) / cos(AoP) 三个通道分别施加三种滤波算法：
 *      - 导向滤波（Guided Filter）：以 S0 强度图为引导图，保边去噪
 *      - 双边滤波（Bilateral Filter）：经典保边滤波器
 *      - 非局部均值去噪（NLM）：利用图像自相似性去噪
 *   4. 以 2 行 4 列布局展示原始/当前滤波结果，便于直观对比
 *   5. 支持 Shi-Tomasi 角点检测叠加，评估不同滤波对特征点提取数量的影响
 *
 * 编译运行：
 *   rosrun polarfp_vins test_filter
 *
 * 订阅话题：
 *   /arena_cam_q2/image_raw (硬编码，需与实际相机话题一致)
 *
 * 键盘控制（在图像窗口中操作）：
 *   SPACE  - 暂停/单步/恢复
 *   'f'    - 开关特征点检测叠加
 *   '+'/'=' - 增大导向滤波半径
 *   '-'    - 减小导向滤波半径
 *   '['    - 减小导向滤波正则化参数 eps
 *   ']'    - 增大导向滤波正则化参数 eps
 *   'w'    - 切换滤波方法（导向→双边→NLM→循环）
 *   'b'/'B' - 减小/增大双边滤波的 sigmaColor
 *   'n'/'N' - 减小/增大 NLM 去噪强度 h
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

#include "../featureTracker/PolarChannel.h"

// ============================================================================
// 导向滤波（手动实现，不依赖 ximgproc 模块）
// 参考文献: He et al., "Guided Image Filtering", ECCV 2010 / TPAMI 2013
// ============================================================================

/**
 * @brief 盒滤波（均值滤波）辅助函数，使用 BORDER_REPLICATE 复制边界
 */
static cv::Mat boxFilter2D(const cv::Mat& src, int radius) {
    cv::Mat dst;
    cv::boxFilter(src, dst, -1, cv::Size(radius, radius), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    return dst;
}

/**
 * @brief 单通道导向滤波
 *
 * 核心思想：以引导图 I 的局部线性模型来滤波输入图 p。
 * 在每个局部窗口内，输出 q = a*I + b，使得 q 接近 p 且平滑。
 * 由于使用 S0（强度图）作为引导图，可以在平滑噪声的同时保留
 * 与强度边缘对齐的偏振通道边缘，避免 DoP/AoP 在物体边界处模糊。
 *
 * @param I   引导图像（CV_64F，通常使用 S0 强度图）
 * @param p   待滤波图像（CV_64F，如 DoP / sin(AoP) / cos(AoP)）
 * @param r   局部窗口半径（核大小 = 2r+1）
 * @param eps 正则化参数，越大平滑越强（防止 var_I 接近零时 a 过大）
 * @return    滤波后图像（CV_64F）
 */
static cv::Mat guidedFilterSingle(const cv::Mat& I, const cv::Mat& p, int r, double eps) {
    cv::Mat mean_I  = boxFilter2D(I, 2 * r + 1);
    cv::Mat mean_p  = boxFilter2D(p, 2 * r + 1);
    cv::Mat mean_II = boxFilter2D(I.mul(I), 2 * r + 1);
    cv::Mat mean_Ip = boxFilter2D(I.mul(p), 2 * r + 1);

    cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    cv::Mat var_I  = mean_II - mean_I.mul(mean_I);

    cv::Mat a = cov_Ip / (var_I + eps);
    cv::Mat b = mean_p - a.mul(mean_I);

    cv::Mat mean_a = boxFilter2D(a, 2 * r + 1);
    cv::Mat mean_b = boxFilter2D(b, 2 * r + 1);

    return mean_a.mul(I) + mean_b;
}

// ============================================================================
// 滤波流水线 — 对三个偏振通道并行施加三种滤波器
// ============================================================================

/**
 * @brief 三种滤波器的输出结果容器
 *
 * 每种滤波器（guided / bilateral / NLM）分别作用于 DoP、sin(AoP)、cos(AoP)，
 * 共 9 个滤波结果 + 1 张 S0 参考图 + 3 张原始通道图。
 */
struct FilterResult {
    cv::Mat dop_guided, dop_bilateral, dop_nlm;
    cv::Mat sin_guided, sin_bilateral, sin_nlm;
    cv::Mat cos_guided, cos_bilateral, cos_nlm;
    cv::Mat S0_img;
    cv::Mat dop_raw, sin_raw, cos_raw;
};

/**
 * @brief 对偏振解码结果施加三种滤波器
 *
 * 处理流程：
 *   1. 将 8-bit 偏振通道归一化到 [0,1] 的 double 精度（导向滤波需要）
 *   2. 保存原始通道副本用于对比展示
 *   3. 以 S0 强度图作为引导图，对 DoP/sin/cos 分别做导向滤波
 *   4. 对 DoP/sin/cos 分别做双边滤波
 *   5. 对 DoP/sin/cos 分别做 NLM 非局部均值去噪
 *
 * @param polar             raw2polar() 解码后的偏振通道结果
 * @param guided_r          导向滤波窗口半径
 * @param guided_eps        导向滤波正则化参数
 * @param bilateral_d       双边滤波邻域直径
 * @param bilateral_sigmaColor 双边滤波颜色空间标准差
 * @param bilateral_sigmaSpace 双边滤波空间域标准差
 * @param nlm_h             NLM 去噪强度参数
 * @param filter_mode       当前滤波方法: 0=guided, 1=bilateral, 2=nlm
 * @return                  包含所有滤波结果的 FilterResult 结构体（仅当前模式字段有效）
 */
static FilterResult applyFilters(const PolarChannelResult& polar,
                                 int guided_r, double guided_eps,
                                 int bilateral_d, double bilateral_sigmaColor, double bilateral_sigmaSpace,
                                 int nlm_h,
                                 int filter_mode) {
    FilterResult res;

    polar.dop_img.copyTo(res.dop_raw);
    polar.sin_img.copyTo(res.sin_raw);
    polar.cos_img.copyTo(res.cos_raw);
    polar.S0_img.copyTo(res.S0_img);

    if (filter_mode == 0) {
        // 导向滤波（以 S0 强度图为引导）
        cv::Mat dop_f, sin_f, cos_f, s0_f;
        polar.dop_img.convertTo(dop_f, CV_64F, 1.0 / 255.0);
        polar.sin_img.convertTo(sin_f, CV_64F, 1.0 / 255.0);
        polar.cos_img.convertTo(cos_f, CV_64F, 1.0 / 255.0);
        polar.S0_img.convertTo(s0_f, CV_64F, 1.0 / 255.0);

        cv::Mat dop_g = guidedFilterSingle(s0_f, dop_f, guided_r, guided_eps);
        cv::Mat sin_g = guidedFilterSingle(s0_f, sin_f, guided_r, guided_eps);
        cv::Mat cos_g = guidedFilterSingle(s0_f, cos_f, guided_r, guided_eps);
        dop_g.convertTo(res.dop_guided, CV_8U, 255.0);
        sin_g.convertTo(res.sin_guided, CV_8U, 255.0);
        cos_g.convertTo(res.cos_guided, CV_8U, 255.0);
    } else if (filter_mode == 1) {
        // 双边滤波
        cv::bilateralFilter(polar.dop_img, res.dop_bilateral, bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace);
        cv::bilateralFilter(polar.sin_img, res.sin_bilateral, bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace);
        cv::bilateralFilter(polar.cos_img, res.cos_bilateral, bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace);
    } else {
        // 非局部均值去噪
        cv::fastNlMeansDenoising(polar.dop_img, res.dop_nlm, nlm_h);
        cv::fastNlMeansDenoising(polar.sin_img, res.sin_nlm, nlm_h);
        cv::fastNlMeansDenoising(polar.cos_img, res.cos_nlm, nlm_h);
    }

    return res;
}

// ============================================================================
// 可视化 — 构建 2 行对比视图（原始 vs 当前滤波）
// ============================================================================

/**
 * @brief 构建 2 行 4 列的对比视图
 *
 * 布局说明：
 *   列 → S0 | DoP | sin(AoP) | cos(AoP)
 *   行 → 原始 | 当前滤波方法
 *
 * @param polar       raw2polar() 输出的原始偏振通道
 * @param fr          applyFilters() 输出的滤波结果
 * @param filter_mode 当前滤波方法: 0=guided, 1=bilateral, 2=nlm
 * @param out         输出的拼接图像（BGR 格式）
 */
static void buildCompareView(const PolarChannelResult& polar, const FilterResult& fr,
                             int filter_mode, cv::Mat& out) {
    const cv::Scalar txtColor(0, 255, 0);
    auto labelAndGray = [&](const cv::Mat& gray, const std::string& lbl) -> cv::Mat {
        cv::Mat c;
        cv::cvtColor(gray, c, cv::COLOR_GRAY2BGR);
        cv::putText(c, lbl, cv::Point(5, 14), cv::FONT_HERSHEY_SIMPLEX, 0.35, txtColor, 1, cv::LINE_AA);
        return c;
    };

    const char* method_name = (filter_mode == 0) ? "Guided" : (filter_mode == 1) ? "Bilateral" : "NLM";

    // 第 0 行：原始偏振通道
    cv::Mat r0c0 = labelAndGray(polar.S0_img, "S0 raw");
    cv::Mat r0c1 = labelAndGray(polar.dop_img, "DoP raw");
    cv::Mat r0c2 = labelAndGray(polar.sin_img, "sin(AoP) raw");
    cv::Mat r0c3 = labelAndGray(polar.cos_img, "cos(AoP) raw");
    cv::Mat row0, row0r;
    cv::hconcat(r0c0, r0c1, row0);
    cv::hconcat(r0c2, r0c3, row0r);
    cv::hconcat(row0, row0r, row0);

    // 第 1 行：当前滤波方法结果
    cv::Mat r1c1, r1c2, r1c3;
    if (filter_mode == 0) {
        r1c1 = labelAndGray(fr.dop_guided, std::string("DoP ") + method_name);
        r1c2 = labelAndGray(fr.sin_guided, std::string("sin(AoP) ") + method_name);
        r1c3 = labelAndGray(fr.cos_guided, std::string("cos(AoP) ") + method_name);
    } else if (filter_mode == 1) {
        r1c1 = labelAndGray(fr.dop_bilateral, std::string("DoP ") + method_name);
        r1c2 = labelAndGray(fr.sin_bilateral, std::string("sin(AoP) ") + method_name);
        r1c3 = labelAndGray(fr.cos_bilateral, std::string("cos(AoP) ") + method_name);
    } else {
        r1c1 = labelAndGray(fr.dop_nlm, std::string("DoP ") + method_name);
        r1c2 = labelAndGray(fr.sin_nlm, std::string("sin(AoP) ") + method_name);
        r1c3 = labelAndGray(fr.cos_nlm, std::string("cos(AoP) ") + method_name);
    }
    cv::Mat r1c0 = labelAndGray(fr.S0_img, "S0 raw");
    cv::Mat row1, row1r;
    cv::hconcat(r1c0, r1c1, row1);
    cv::hconcat(r1c2, r1c3, row1r);
    cv::hconcat(row1, row1r, row1);

    cv::vconcat(row0, row1, out);
}

/**
 * @brief 在单通道图像上执行 Shi-Tomasi 角点检测并绘制结果
 *
 * 用于评估不同滤波处理后特征点提取的数量和质量变化。
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
 * 展示 S0 原始图 + DoP 原始图 + 三种 DoP 滤波结果上的角点检测效果，
 * 用于直观比较哪种滤波方式能提取到更多/更稳定的特征点。
 *
 * @param fr    滤波结果
 * @param out   输出的拼接图像
 * @param max_cnt 最大角点数
 * @param quality 角点质量阈值
 * @param min_dist 角点最小间距
 */
static void buildFeatureView(const FilterResult& fr, cv::Mat& out,
                             int max_cnt, double quality, double min_dist,
                             int filter_mode) {
    cv::Mat s0_f, dop_rf, dop_filt;
    detectAndDrawFeatures(fr.S0_img, s0_f, "S0", max_cnt, quality, min_dist);
    detectAndDrawFeatures(fr.dop_raw, dop_rf, "DoP raw", max_cnt, quality, min_dist);

    const char* names[] = {"guided", "bilateral", "NLM"};
    if (filter_mode == 0)
        detectAndDrawFeatures(fr.dop_guided, dop_filt, "DoP guided", max_cnt, quality, min_dist);
    else if (filter_mode == 1)
        detectAndDrawFeatures(fr.dop_bilateral, dop_filt, "DoP bilateral", max_cnt, quality, min_dist);
    else
        detectAndDrawFeatures(fr.dop_nlm, dop_filt, "DoP NLM", max_cnt, quality, min_dist);

    cv::Mat row1, row2;
    cv::hconcat(s0_f, dop_rf, row1);
    cv::hconcat(dop_filt, cv::Mat::zeros(dop_filt.size(), dop_filt.type()), row2);
    cv::vconcat(row1, row2, out);
}

// ============================================================================
// ROS 节点 — 主循环：订阅图像 → 解码 → 滤波 → 显示 → 键盘交互
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
 * 将 ROS sensor_msgs/Image 转换为 OpenCV Mat 并缓存到 latest_raw。
 * 使用 toCvShare 避免内存拷贝（零拷贝共享 ROS 消息缓冲区），
 * 但需要立即 copyTo 以脱离 ROS 消息的生命周期。
 */
void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    try {
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "mono8");
        img_mutex.lock();
        cv_ptr->image.copyTo(latest_raw);
        new_frame = true;
        img_mutex.unlock();
    } catch (cv_bridge::Exception& e) {
        ROS_WARN("cv_bridge error: %s", e.what());
    }
}

/**
 * @brief 主函数 — 初始化 ROS 节点、滤波器参数、OpenCV 窗口及主循环
 *
 * 主循环流程：
 *   1. ros::spinOnce() 处理 ROS 回调（接收新图像）
 *   2. 线程安全地取出最新帧
 *   3. raw2polar() 解码偏振通道
 *   4. applyFilters() 施加三种滤波
 *   5. 构建对比视图（或特征点视图）并显示
 *   6. 等待键盘输入处理交互命令
 */
int main(int argc, char** argv) {
    ros::init(argc, argv, "test_filter");
    ROS_INFO("=== Polar Channel Filter Test ===");
    ROS_INFO("Subscribing to %s", IMAGE_TOPIC.c_str());

    // 滤波器参数（可通过键盘实时调节）
    int guided_r = 4;
    double guided_eps = 0.01;
    int bilateral_d = 9;
    double bilateral_sigmaColor = 100;
    double bilateral_sigmaSpace = 30;
    int nlm_h = 50;

    // Shi-Tomasi 角点检测参数
    int max_cnt = 200;
    double quality = 0.01;
    double min_dist = 10;

    ROS_INFO("Filter params: guided_r=%d eps=%.3f | bilateral_d=%d sc=%.1f ss=%.1f | nlm_h=%d",
             guided_r, guided_eps, bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace, nlm_h);

    // 创建 ROS 订阅器，队列深度 10
    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe(IMAGE_TOPIC, 10, imageCallback);

    // 创建可缩放窗口
    const std::string win_name = "Filter Comparison";
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(win_name, 1280, 720);

    // 运行状态标志
    bool show_features = false;  // 是否显示特征点叠加视图
    bool paused = false;          // 是否暂停
    int frame_count = 0;          // 已处理帧数
    int saved_count = 0;          // 已保存帧数
    int filter_mode = 0;          // 当前滤波方法: 0=guided, 1=bilateral, 2=NLM
    const char* filter_names[] = {"Guided", "Bilateral", "NLM"};
    ROS_INFO("Active filter: %s (press 'w' to switch)", filter_names[filter_mode]);

    // 主循环：ROS 事件处理 + 图像处理 + 键盘交互
    while (ros::ok()) {
        ros::spinOnce();

        // 线程安全地取出最新帧
        img_mutex.lock();
        bool has_new = new_frame;
        new_frame = false;
        cv::Mat raw;
        if (has_new) {
            latest_raw.copyTo(raw);
        }
        img_mutex.unlock();

        // 核心处理流水线：解码偏振通道 → 施加滤波器
        cv::Mat display;
        if (has_new) {
            auto t_start = std::chrono::high_resolution_clock::now();

            PolarChannelResult polar = raw2polar(raw);
            FilterResult fr = applyFilters(polar, guided_r, guided_eps,
                                           bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace,
                                           nlm_h, filter_mode);

            auto t_end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
            printf("filter time: %.2f ms\n", elapsed_ms);

            // 构建显示视图（滤波对比模式 或 特征点检测模式）
            if (show_features) {
                buildFeatureView(fr, display, max_cnt, quality, min_dist, filter_mode);
            } else {
                buildCompareView(polar, fr, filter_mode, display);
            }

            // 在窗口标题中显示当前滤波方法
            cv::setWindowTitle(win_name, "Filter: " + std::string(filter_names[filter_mode]));

            cv::imshow(win_name, display);
            frame_count++;
        }

        // 每次循环统一等待 20ms，降低 CPU 占用并响应键盘
        int key = cv::waitKey(20) & 0xFF;

        // ====== 键盘交互处理 ======
        if (key == 27 || key == 'q') {
            // ESC 或 Q 退出
            break;
        } else if (key == 32) { // SPACE：暂停/恢复
            paused = !paused;
            ROS_INFO("%s", paused ? "PAUSED" : "RESUMED");
        } else if (paused) {
            // 暂停状态下按键不处理其他操作，继续等待
            continue;
        } else if (key == 's') {
            // 保存当前视图到 PNG
            std::string name = "filter_frame_" + std::to_string(frame_count) + "_" + std::to_string(saved_count) + ".png";
            cv::imwrite(name, display);
            ROS_INFO("Saved %s", name.c_str());
            saved_count++;
        } else if (key == 'f') {
            // 切换特征点叠加显示
            show_features = !show_features;
            ROS_INFO("Feature overlay: %s", show_features ? "ON" : "OFF");
        } else if (key == 'w') {
            // 切换滤波方法
            filter_mode = (filter_mode + 1) % 3;
            ROS_INFO("Filter method -> %s", filter_names[filter_mode]);
        } else if (key == '=' || key == '+') {
            // 增大导向滤波半径
            guided_r = std::min(guided_r + 1, 20);
            ROS_INFO("guided_r -> %d", guided_r);
        } else if (key == '-') {
            // 减小导向滤波半径
            guided_r = std::max(guided_r - 1, 1);
            ROS_INFO("guided_r -> %d", guided_r);
        } else if (key == '[') {
            // 减小导向滤波正则化参数（halve）
            guided_eps = std::max(guided_eps * 0.5, 0.0001);
            ROS_INFO("guided_eps -> %.4f", guided_eps);
        } else if (key == ']') {
            // 增大导向滤波正则化参数（double）
            guided_eps = std::min(guided_eps * 2.0, 1.0);
            ROS_INFO("guided_eps -> %.4f", guided_eps);
        } else if (key == 'b') {
            // 减小双边滤波 sigmaColor（降低保边敏感度）
            bilateral_sigmaColor = std::max(bilateral_sigmaColor - 5.0, 1.0);
            ROS_INFO("bilateral_sigmaColor -> %.1f", bilateral_sigmaColor);
        } else if (key == 'B') {
            // 增大双边滤波 sigmaColor（提高保边敏感度）
            bilateral_sigmaColor = std::min(bilateral_sigmaColor + 5.0, 250.0);
            ROS_INFO("bilateral_sigmaColor -> %.1f", bilateral_sigmaColor);
        } else if (key == 'n') {
            // 减小 NLM 去噪强度
            nlm_h = std::max(nlm_h - 2, 1);
            ROS_INFO("nlm_h -> %d", nlm_h);
        } else if (key == 'N') {
            // 增大 NLM 去噪强度
            nlm_h = std::min(nlm_h + 2, 50);
            ROS_INFO("nlm_h -> %d", nlm_h);
        }
    }

    cv::destroyAllWindows();
    ROS_INFO("Processed %d frames", frame_count);
    return 0;
}
