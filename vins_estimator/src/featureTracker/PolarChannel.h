#ifndef _POLAR_CHANNEL_H
#define _POLAR_CHANNEL_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>


const double EPSILON = 1e-6;
const double DOP_PERCENTILE = 99.0;

// Match the implementations in PolarChannel.cpp
double _calculatePercentile(const cv::Mat& mat, double percentile);
cv::Mat _raw_chnl_to_gray(const std::vector<cv::Mat>& img_xx);
struct PolarChannelResult {
    cv::Mat S0_color;  // (h/2, w/2, 3) RGB color
    cv::Mat S0_img;    // (h/2, w/2) grayscale
    cv::Mat aop_vis;   // (h/2, w/2, 3) RGB visualization of AoP
    cv::Mat dop_img;   // (h/2, w/2) quantized DoP
    cv::Mat sin_img;   // (h/2, w/2) quantized sin(AoP)
    cv::Mat cos_img;   // (h/2, w/2) quantized cos(AoP)
    cv::Mat aop;       // (h/2, w/2) raw AoP values
    cv::Mat dop;       // (h/2, w/2) raw DoP values
};
PolarChannelResult raw2polar(const cv::Mat& img_raw);

#endif // _POLAR_CHANNEL_H