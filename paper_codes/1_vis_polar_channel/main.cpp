/*******************************************************
 * Copyright (C) 2026, Bionic Intelligence Research Team, Beihang University
 *
 * Polar channel visualization — generates Figure 2 for the paper.
 *
 * Reads raw polarization frames (PNG extracted from ROS bags), computes
 * Stokes parameters and polar channels, saves individual channel images
 * and a composite 3x6 table.
 *
 * Usage: cd paper_codes/1_vis_polar_channel && ./build/vis_polar_channel
 *******************************************************/

#include "PolarChannel.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

struct SceneInfo {
    std::string name;   // subdirectory / bag name
    std::string label;  // lighting condition label
};

int main() {
    // ── Scene definitions ──
    const std::vector<SceneInfo> scenes = {
        {"13-27-22", "94-205 lux"},
        {"13-54-30", "2.6-18.6 lux"},
        {"14-15-03", "0.9-4.8 lux"},
    };

    const std::vector<std::string> ch_names = {"S0", "S1", "S2", "P0", "P1", "P2"};
    const std::vector<std::string> ch_dirs  = {"s0", "s1", "s2", "p0", "p1", "p2"};
    const int NUM_SCENES = 3;
    const int NUM_CHANNELS = 6;

    // ── Create output directories ──
    system("mkdir -p output/s0 output/s1 output/s2 output/p0 output/p1 output/p2");

    // Store quantized (CV_8U) channel images for composite: [scene][channel]
    std::vector<std::vector<cv::Mat>> all_images(NUM_SCENES);

    // ── Process each scene ──
    for (int i = 0; i < NUM_SCENES; i++) {
        std::string img_path = "data/" + scenes[i].name + ".png";
        cv::Mat raw = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        if (raw.empty()) {
            std::cerr << "[ERROR] Cannot read " << img_path << std::endl;
            std::cerr << "       Run extract_first_frames.py first, or check path." << std::endl;
            return 1;
        }
        std::cout << "Processing " << scenes[i].name
                  << " (" << scenes[i].label << ")"
                  << "  [" << raw.cols << "x" << raw.rows << "]" << std::endl;

        PolarChannelResult res = raw2polar(raw, PolarFilterConfig{});

        // Collect 6 quantized channels: S0, S1, S2, P0(DoP), P1(sinAoP), P2(cosAoP)
        all_images[i] = {res.S0_img, res.S1_img, res.S2_img,
                         res.dop_img, res.sin_img, res.cos_img};

        // Save individual channel images
        for (int c = 0; c < NUM_CHANNELS; c++) {
            std::string out = "output/" + ch_dirs[c] + "/" + scenes[i].name + ".png";
            cv::imwrite(out, all_images[i][c]);
            std::cout << "  -> " << out << std::endl;
        }
    }

    // ── Build composite table ──
    // Layout parameters
    const int CELL_W = all_images[0][0].cols;   // 306
    const int CELL_H = all_images[0][0].rows;   // 256
    const int PAD = 6;           // padding around each cell (border)
    const int GAP_H = 10;        // horizontal gap between cells
    const int GAP_V = 10;        // vertical gap between rows
    const int LEFT_MARGIN = 130; // space for row labels
    const int TOP_MARGIN = 50;   // space for column headers

    int canvas_w = LEFT_MARGIN + NUM_CHANNELS * (CELL_W + 2 * PAD) + (NUM_CHANNELS - 1) * GAP_H;
    int canvas_h = TOP_MARGIN + NUM_SCENES * (CELL_H + 2 * PAD) + (NUM_SCENES - 1) * GAP_V;

    cv::Mat canvas(canvas_h, canvas_w, CV_8UC3, cv::Scalar(255, 255, 255));

    // Font settings
    int font = cv::FONT_HERSHEY_SIMPLEX;

    // Draw column headers (centered above each column)
    for (int c = 0; c < NUM_CHANNELS; c++) {
        int cx = LEFT_MARGIN + c * (CELL_W + 2 * PAD + GAP_H) + PAD + CELL_W / 2;
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(ch_names[c], font, 0.7, 2, &baseline);
        cv::putText(canvas, ch_names[c],
                    cv::Point(cx - text_size.width / 2, TOP_MARGIN - 10),
                    font, 0.7, cv::Scalar(0, 0, 0), 2);
    }

    // Draw row labels and place images
    for (int r = 0; r < NUM_SCENES; r++) {
        // Row label (centered vertically on the left)
        int ry = TOP_MARGIN + r * (CELL_H + 2 * PAD + GAP_V) + PAD + CELL_H / 2;
        int baseline = 0;
        cv::Size lbl_size = cv::getTextSize(scenes[r].label, font, 0.6, 1, &baseline);
        cv::putText(canvas, scenes[r].label,
                    cv::Point(LEFT_MARGIN - lbl_size.width - 12, ry + lbl_size.height / 2),
                    font, 0.6, cv::Scalar(0, 0, 0), 1);

        for (int c = 0; c < NUM_CHANNELS; c++) {
            int x0 = LEFT_MARGIN + c * (CELL_W + 2 * PAD + GAP_H);
            int y0 = TOP_MARGIN + r * (CELL_H + 2 * PAD + GAP_V);

            // Light gray border
            cv::rectangle(canvas,
                          cv::Rect(x0, y0, CELL_W + 2 * PAD, CELL_H + 2 * PAD),
                          cv::Scalar(200, 200, 200), cv::FILLED);

            // Convert single channel to BGR for display
            cv::Mat img_bgr;
            cv::cvtColor(all_images[r][c], img_bgr, cv::COLOR_GRAY2BGR);

            // Place image
            img_bgr.copyTo(canvas(cv::Rect(x0 + PAD, y0 + PAD, CELL_W, CELL_H)));
        }
    }

    std::string composite_path = "output/composite.png";
    cv::imwrite(composite_path, canvas);
    std::cout << "\nComposite saved: " << composite_path
              << " (" << canvas_w << "x" << canvas_h << ")" << std::endl;

    return 0;
}
