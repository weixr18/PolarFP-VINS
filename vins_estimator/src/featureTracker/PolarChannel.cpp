#include "PolarChannel.h"
#include <vector>
#include <algorithm>
// version: 2026.3.6 

double _calculatePercentile(const cv::Mat& mat, double percentile) {
    cv::Mat flattened = mat.reshape(1, 1);
    std::vector<double> sorted;
    flattened.copyTo(sorted);
    std::sort(sorted.begin(), sorted.end());
    int index = static_cast<int>((percentile / 100.0) * (sorted.size() - 1));
    return sorted[index];
}

cv::Mat _raw_chnl_to_gray(const std::vector<cv::Mat>& img_xx) {
    int itp = cv::INTER_CUBIC;
    const cv::Mat& img_xx_r = img_xx[0];
    const cv::Mat& img_xx_g1 = img_xx[1];
    const cv::Mat& img_xx_g2 = img_xx[2];
    const cv::Mat& img_xx_b = img_xx[3];
    int h_4 = img_xx_r.rows;
    int w_4 = img_xx_r.cols;
    int h_half = h_4 * 2;
    int w_half = w_4 * 2;
    cv::Mat img_xx_r_resized, img_xx_g1_resized, img_xx_g2_resized, img_xx_b_resized;
    cv::resize(img_xx_r, img_xx_r_resized, cv::Size(w_half, h_half), 0, 0, itp);
    cv::resize(img_xx_g1, img_xx_g1_resized, cv::Size(w_half, h_half), 0, 0, itp);
    cv::resize(img_xx_g2, img_xx_g2_resized, cv::Size(w_half, h_half), 0, 0, itp);
    cv::resize(img_xx_b, img_xx_b_resized, cv::Size(w_half, h_half), 0, 0, itp);
    double k_r = 0.299, k_g = 0.587, k_b = 0.114;
    cv::Mat img_xx_gray;
    img_xx_r_resized.convertTo(img_xx_gray, CV_64F);
    img_xx_gray *= k_r;
    cv::Mat temp;
    img_xx_g1_resized.convertTo(temp, CV_64F);
    img_xx_gray += temp * k_g * 0.5;
    img_xx_g2_resized.convertTo(temp, CV_64F);
    img_xx_gray += temp * k_g * 0.5;
    img_xx_b_resized.convertTo(temp, CV_64F);
    img_xx_gray += temp * k_b;
    return img_xx_gray;
}

cv::Mat _raw_chnl_to_rgb(const std::vector<cv::Mat>& img_xx) {
    int itp = cv::INTER_CUBIC;
    const cv::Mat& img_xx_r = img_xx[0];
    const cv::Mat& img_xx_g1 = img_xx[1];
    const cv::Mat& img_xx_g2 = img_xx[2];
    const cv::Mat& img_xx_b = img_xx[3];
    int h_4 = img_xx_r.rows;
    int w_4 = img_xx_r.cols;
    int h_half = h_4 * 2;
    int w_half = w_4 * 2;
    
    cv::Mat r_resized, g1_resized, g2_resized, b_resized;
    cv::resize(img_xx_r, r_resized, cv::Size(w_half, h_half), 0, 0, itp);
    cv::resize(img_xx_g1, g1_resized, cv::Size(w_half, h_half), 0, 0, itp);
    cv::resize(img_xx_g2, g2_resized, cv::Size(w_half, h_half), 0, 0, itp);
    cv::resize(img_xx_b, b_resized, cv::Size(w_half, h_half), 0, 0, itp);
    
    // Combine channels to RGB (R, G, B)
    std::vector<cv::Mat> channels;
    channels.push_back(r_resized);  // R
    cv::Mat g_combined = (g1_resized + g2_resized) / 2;  // G
    channels.push_back(g_combined);  // G
    channels.push_back(b_resized);   // B
    
    cv::Mat rgb;
    cv::merge(channels, rgb);
    return rgb;
}


PolarChannelResult raw2polar(const cv::Mat& img_raw) {
    assert(img_raw.channels() == 1 && img_raw.dims == 2);
    
    int new_rows = img_raw.rows / 4;
    int new_cols = img_raw.cols / 4;
    
    // Helper function to sample polar channels and return both gray and RGB
    auto sample_polar_channels = [&](
        int row_offset1, int col_offset1, int row_offset2, int col_offset2
    ) -> std::pair<cv::Mat, cv::Mat> {
        cv::Mat r_channel(new_rows, new_cols, img_raw.type());
        cv::Mat g1_channel(new_rows, new_cols, img_raw.type());
        cv::Mat g2_channel(new_rows, new_cols, img_raw.type());
        cv::Mat b_channel(new_rows, new_cols, img_raw.type());
        for (int i = 0; i < new_rows; i++) {
            for (int j = 0; j < new_cols; j++) {
                r_channel.at<uchar>(i, j) = img_raw.at<uchar>(i * 4 + row_offset1, j * 4 + col_offset1);
                g1_channel.at<uchar>(i, j) = img_raw.at<uchar>(i * 4 + row_offset1, j * 4 + col_offset2);
                g2_channel.at<uchar>(i, j) = img_raw.at<uchar>(i * 4 + row_offset2, j * 4 + col_offset1);
                b_channel.at<uchar>(i, j) = img_raw.at<uchar>(i * 4 + row_offset2, j * 4 + col_offset2);
            }
        }
        std::vector<cv::Mat> channels = {r_channel, g1_channel, g2_channel, b_channel};
        cv::Mat gray = _raw_chnl_to_gray(channels);
        cv::Mat rgb = _raw_chnl_to_rgb(channels);
        return {gray, rgb};
    };
    
    // Sample all four polarizations
    auto [img_90_gray, img_90_rgb] = sample_polar_channels(0, 0, 2, 2);  // 90°
    auto [img_45_gray, img_45_rgb] = sample_polar_channels(0, 1, 2, 3);  // 45°
    auto [img_135_gray, img_135_rgb] = sample_polar_channels(1, 0, 3, 2); // 135°
    auto [img_0_gray, img_0_rgb] = sample_polar_channels(1, 1, 3, 3);    // 0°
    
    // 2. Stokes vector
    cv::Mat S_0 = (img_90_gray + img_45_gray + img_135_gray + img_0_gray) / 4.0;
    cv::Mat S_1 = img_0_gray - img_90_gray;
    cv::Mat S_2 = img_45_gray - img_135_gray;
    
    // Color Stokes vector
    cv::Mat S0_color = (img_90_rgb + img_45_rgb + img_135_rgb + img_0_rgb) / 4.0;
    S0_color.convertTo(S0_color, CV_8U);
    
    // 3. Calculate sinaop and cosaop
    cv::Mat denominator;
    cv::sqrt(S_1.mul(S_1) + S_2.mul(S_2) + EPSILON, denominator);
    cv::Mat sinaop = S_2 / denominator;  // Note: swapped from previous version to match Python
    cv::Mat cosaop = S_1 / denominator;
    
    // 4. Calculate AoP
    cv::Mat aop;
    cv::phase(S_1, S_2, aop);  // arctan2(S_2, S_1)
    aop *= 0.5;  // [-pi/2, pi/2]
    
    // 5. Visualize AoP as HSV then convert to RGB
    cv::Mat aop_vis_hsv(img_90_gray.size(), CV_8UC3);
    for (int i = 0; i < aop_vis_hsv.rows; i++) {
        for (int j = 0; j < aop_vis_hsv.cols; j++) {
            double aop_val = aop.at<double>(i, j);
            // Convert from [-pi/2, pi/2] to [0, 180] for Hue
            float hue = static_cast<float>((aop_val + CV_PI/2) * 180.0 / CV_PI);
            aop_vis_hsv.at<cv::Vec3b>(i, j) = cv::Vec3b(
                static_cast<uchar>(hue),      // H
                180,                          // S
                255                           // V
            );
        }
    }
    cv::Mat aop_vis;
    cv::cvtColor(aop_vis_hsv, aop_vis, cv::COLOR_HSV2RGB);
    
    // 6. Calculate DoP
    cv::Mat dop;
    cv::sqrt(S_1.mul(S_1) + S_2.mul(S_2) + EPSILON, dop);
    dop /= (S_0 + EPSILON);
    // Handle mask for invalid regions
    cv::Mat MASK = cv::abs(S_0) < EPSILON;
    sinaop.setTo(0.0, MASK);
    cosaop.setTo(0.0, MASK);
    dop.setTo(0.0, MASK);
    // Apply DoP percentile clipping
    double max_dop = _calculatePercentile(dop, DOP_PERCENTILE);
    cv::min(dop, max_dop, dop);
    if (max_dop > 1.0) {
        dop /= max_dop;
    }
    
    // 7. Quantize outputs
    cv::Mat dop_img, sin_img, cos_img, S0_img;
    dop.convertTo(dop_img, CV_8U, 255.0);  // * 255
    sinaop.convertTo(sin_img, CV_8U, 127.5, 127.5); // (x + 1) / 2 * 255
    cosaop.convertTo(cos_img, CV_8U, 127.5, 127.5);
    S_0.convertTo(S0_img, CV_8U);
    PolarChannelResult result;
    result.S0_color = S0_color;
    result.S0_img = S0_img;
    result.aop_vis = aop_vis;
    result.dop_img = dop_img;
    result.sin_img = sin_img;
    result.cos_img = cos_img;
    result.aop = aop.clone();
    result.dop = dop.clone();
    return result;
}