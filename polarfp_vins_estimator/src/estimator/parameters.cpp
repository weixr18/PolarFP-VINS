/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"
#include <sstream>

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string OUTPUT_NAME;
std::string IMU_TOPIC;
int ROW, COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
int MULTIPLE_THREAD;
map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
double F_THRESHOLD;
int SHOW_TRACK;
int FLOW_BACK;

// Polar mode
std::vector<std::string> POLAR_CHANNELS;
PolarFilterConfig POLAR_FILTER_CFG;
int POLAR_HASH_GRID_SIZE = 5;

// Feature detector/matcher configuration
int FEATURE_DETECTOR_TYPE = 0;   // 0=GFTT, 1=FAST
int FAST_THRESHOLD = 20;
int FAST_NONMAX_SUPPRESSION = 1;
int FEATURE_MATCHER_TYPE = 0;    // 0=LK_FLOW, 1=BRIEF_FLANN
int BRIEF_DESCRIPTOR_BYTES = 32;
float BRIEF_MATCH_DIST_RATIO = 0.75f;
int RANDOM_SEED = -1;

// SuperPoint parameters
std::string SUPERPOINT_MODEL_PATH;
int SUPERPOINT_USE_GPU = 1;
float SUPERPOINT_KEYPOINT_THRESHOLD = 0.015f;
int SUPERPOINT_NMS_RADIUS = 4;


template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(std::string config_file)
{
    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        ROS_BREAK();
        return;          
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    FLOW_BACK = fsSettings["flow_back"];

    MULTIPLE_THREAD = fsSettings["multiple_thread"];

    USE_IMU = fsSettings["imu"];
    printf("USE_IMU: %d\n", USE_IMU);
    if(USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];
    }

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    fsSettings["output_path"] >> OUTPUT_FOLDER;
    fsSettings["output_name"] >> OUTPUT_NAME;
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/" + OUTPUT_NAME + ".csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream foutC(VINS_RESULT_PATH, std::ios::out);
    foutC << "time,tx,ty,tz,qw,qx,qy,qz" << std::endl;
    foutC.close();

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_T;
        fsSettings["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    } 
    
    NUM_OF_CAM = fsSettings["num_of_cam"];
    printf("camera number %d\n", NUM_OF_CAM);

    if(NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
    {
        printf("num_of_cam should be 1 or 2\n");
        assert(0);
    }


    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);
    
    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    CAM_NAMES.push_back(cam0Path);

    if(NUM_OF_CAM == 2)
    {
        STEREO = 1;
        std::string cam1Calib;
        fsSettings["cam1_calib"] >> cam1Calib;
        std::string cam1Path = configPath + "/" + cam1Calib; 
        //printf("%s cam1 path\n", cam1Path.c_str() );
        CAM_NAMES.push_back(cam1Path);
        
        cv::Mat cv_T;
        fsSettings["body_T_cam1"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %d COL: %d ", ROW, COL);

    if(!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    // Polar mode parameters
    if (!fsSettings["polar_hash_grid_size"].empty())
        POLAR_HASH_GRID_SIZE = (int)fsSettings["polar_hash_grid_size"];
    else
        POLAR_HASH_GRID_SIZE = 5;

    std::string channels_str;
    fsSettings["polar_channels"] >> channels_str;
    std::stringstream ss(channels_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        POLAR_CHANNELS.push_back(item);
    }
    if (POLAR_CHANNELS.empty()) {
        POLAR_CHANNELS = {"s0", "dop"};
        ROS_WARN("polar_channels not specified, using default: s0,dop");
    }
    ROS_INFO("Polar mode enabled, channels: %zu", POLAR_CHANNELS.size());
    for (const auto& ch : POLAR_CHANNELS)
        ROS_INFO("  channel: %s", ch.c_str());

    // 偏振通道滤波参数
    if (!fsSettings["polar_filter_type"].empty())
        POLAR_FILTER_CFG.filter_type = static_cast<PolarFilterType>((int)fsSettings["polar_filter_type"]);

    // 双边滤波参数
    if (POLAR_FILTER_CFG.filter_type == FILTER_BILATERAL) {
        if (!fsSettings["polar_bilateral_d"].empty())
            POLAR_FILTER_CFG.bilateral_d = (int)fsSettings["polar_bilateral_d"];
        if (!fsSettings["polar_bilateral_sigma_color"].empty())
            POLAR_FILTER_CFG.bilateral_sigmaColor = (double)fsSettings["polar_bilateral_sigma_color"];
        if (!fsSettings["polar_bilateral_sigma_space"].empty())
            POLAR_FILTER_CFG.bilateral_sigmaSpace = (double)fsSettings["polar_bilateral_sigma_space"];
        ROS_INFO("[PolarFP] Bilateral filter: d=%d sigmaColor=%.1f sigmaSpace=%.1f",
                 POLAR_FILTER_CFG.bilateral_d, POLAR_FILTER_CFG.bilateral_sigmaColor,
                 POLAR_FILTER_CFG.bilateral_sigmaSpace);
    }

    // 导向滤波参数
    if (POLAR_FILTER_CFG.filter_type == FILTER_GUIDED) {
        if (!fsSettings["polar_guided_radius"].empty())
            POLAR_FILTER_CFG.guided_radius = (int)fsSettings["polar_guided_radius"];
        if (!fsSettings["polar_guided_eps"].empty())
            POLAR_FILTER_CFG.guided_eps = (double)fsSettings["polar_guided_eps"];
        ROS_INFO("[PolarFP] Guided filter: radius=%d eps=%.4f",
                 POLAR_FILTER_CFG.guided_radius, POLAR_FILTER_CFG.guided_eps);
    }

    // NLM 滤波参数
    if (POLAR_FILTER_CFG.filter_type == FILTER_NLM) {
        if (!fsSettings["polar_nlm_h"].empty())
            POLAR_FILTER_CFG.nlm_h = (float)(double)fsSettings["polar_nlm_h"];
        if (!fsSettings["polar_nlm_template"].empty())
            POLAR_FILTER_CFG.nlm_template = (int)fsSettings["polar_nlm_template"];
        if (!fsSettings["polar_nlm_search"].empty())
            POLAR_FILTER_CFG.nlm_search = (int)fsSettings["polar_nlm_search"];
        ROS_INFO("[PolarFP] NLM filter: h=%.1f template=%d search=%d",
                 POLAR_FILTER_CFG.nlm_h, POLAR_FILTER_CFG.nlm_template,
                 POLAR_FILTER_CFG.nlm_search);
    }

    // 中值滤波参数
    if (POLAR_FILTER_CFG.filter_type == FILTER_MEDIAN) {
        if (!fsSettings["polar_median_kernel_size"].empty())
            POLAR_FILTER_CFG.median_kernel_size = (int)fsSettings["polar_median_kernel_size"];
        ROS_INFO("[PolarFP] Median filter: kernel_size=%d",
                 POLAR_FILTER_CFG.median_kernel_size);
    }

    // Feature detector/matcher parameters
    if (!fsSettings["feature_detector_type"].empty())
        FEATURE_DETECTOR_TYPE = (int)fsSettings["feature_detector_type"];
    if (!fsSettings["fast_threshold"].empty())
        FAST_THRESHOLD = (int)fsSettings["fast_threshold"];
    if (!fsSettings["fast_nonmax_suppression"].empty())
        FAST_NONMAX_SUPPRESSION = (int)fsSettings["fast_nonmax_suppression"];
    if (!fsSettings["feature_matcher_type"].empty())
        FEATURE_MATCHER_TYPE = (int)fsSettings["feature_matcher_type"];
    if (!fsSettings["brief_descriptor_bytes"].empty())
        BRIEF_DESCRIPTOR_BYTES = (int)fsSettings["brief_descriptor_bytes"];
    if (!fsSettings["brief_match_dist_ratio"].empty())
        BRIEF_MATCH_DIST_RATIO = (float)(double)fsSettings["brief_match_dist_ratio"];
    if (!fsSettings["random_seed"].empty())
        RANDOM_SEED = (int)fsSettings["random_seed"];

    const char* det_names[] = {"GFTT", "FAST", "SUPERPOINT"};
    const char* match_names[] = {"LK_FLOW", "BRIEF_FLANN"};
    ROS_INFO("[PolarFP] Detector: %s, Matcher: %s",
             FEATURE_DETECTOR_TYPE < 3 ? det_names[FEATURE_DETECTOR_TYPE] : "unknown",
             FEATURE_MATCHER_TYPE < 2 ? match_names[FEATURE_MATCHER_TYPE] : "unknown");

    // SuperPoint parameters
    if (FEATURE_DETECTOR_TYPE == 2) {
        if (!fsSettings["superpoint_model_path"].empty()) {
            fsSettings["superpoint_model_path"] >> SUPERPOINT_MODEL_PATH;
            // Resolve relative path
            if (SUPERPOINT_MODEL_PATH[0] != '/') {
                SUPERPOINT_MODEL_PATH = configPath + "/" + SUPERPOINT_MODEL_PATH;
            }
        }
        if (!fsSettings["superpoint_use_gpu"].empty())
            SUPERPOINT_USE_GPU = (int)fsSettings["superpoint_use_gpu"];
        if (!fsSettings["superpoint_keypoint_threshold"].empty())
            SUPERPOINT_KEYPOINT_THRESHOLD = (float)(double)fsSettings["superpoint_keypoint_threshold"];
        if (!fsSettings["superpoint_nms_radius"].empty())
            SUPERPOINT_NMS_RADIUS = (int)fsSettings["superpoint_nms_radius"];
        ROS_INFO("[PolarFP] SuperPoint model: %s, GPU: %d", SUPERPOINT_MODEL_PATH.c_str(), SUPERPOINT_USE_GPU);
    }

    fsSettings.release();
}
