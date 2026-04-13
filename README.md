# PolarFP-VINS

基于偏振视觉的 VINS（Visual-Inertial Navigation System），在 [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) 基础上扩展，利用偏振相机提供的偏振信息（Stokes 参数、偏振度 DoP、偏振角 AoP）替代/增强传统强度特征，**在暗光条件下仍能实现鲁棒的视觉惯性里程计估计**。

## 核心特性

- **偏振特征跟踪**：从偏振相机 2×2 微偏振片阵列原始图像中解码 S0（总强度）、DoP（偏振度）、AoP（偏振角），支持多通道组合跟踪（如 `s0,dop,aopsin,aopcos`）
- **暗光鲁棒性**：偏振信息在低照度下比纯强度信息更稳定
- **VINS-Fusion 兼容**：保持与 VINS-Fusion 相同的滑动窗口优化架构（IMU 预积分 + 视觉重投影误差 + 边缘化）
- **单目/双目 + IMU**：支持单目和双目相机配置

## 项目结构

```
PolarFP-VINS/
├── config/                     # 配置文件（YAML）
├── docker/                     # Docker 运行环境
├── polarfp_camera_models/     # 相机模型库（pinhole、fisheye、Scaramuzza 等）
│   ├── CMakeLists.txt
│   ├── include/
│   └── src/
├── vins_estimator/            # VINS 估计器主包
│   ├── CMakeLists.txt
│   ├── cmake/
│   └── src/
│       ├── estimator/         # 滑动窗口状态估计器
│       ├── factor/            # Ceres 因子（IMU 预积分、投影误差、边缘化）
│       ├── featureTracker/    # 特征跟踪器 + 偏振通道解码
│       ├── initial/           # 初始化（SFM、外参标定、陀螺仪对齐）
│       └── utility/           # 可视化工具
└── README.md
```

## 依赖

- ROS（Melodic / Noetic）
- OpenCV 3/4
- Ceres Solver
- Eigen3

## 编译

```bash
cd ~/ws/vi_catkin_ws
catkin build polarfp_vins   # 或 catkin_make
source devel/setup.bash
```

## 运行

```bash
roslaunch polarfp_vins run.launch config:=dark/dark_mono_imu_config.yaml
```

## 配置说明

YAML 配置文件中的关键字段：

| 参数 | 说明 |
|------|------|
| `use_polar` | 是否启用偏振通道（1=启用） |
| `polar_channels` | 使用的偏振通道组合，如 `"s0,dop,aopsin,aopcos"` |
| `polar_filter_type` | 滤波器类型：0=无，1=双边滤波，2=导向滤波 |
| `polar_bilateral_d` | 双边滤波邻域直径（filter_type=1 时） |
| `polar_bilateral_sigma_color` | 双边滤波颜色空间标准差（filter_type=1 时） |
| `polar_bilateral_sigma_space` | 双边滤波空间域标准差（filter_type=1 时） |
| `polar_guided_radius` | 导向滤波窗口半径（filter_type=2 时） |
| `polar_guided_eps` | 导向滤波正则化参数（filter_type=2 时） |
| `num_of_cam` | 相机数量（1=单目，2=双目） |
| `imu` | 是否使用 IMU |
| `max_cnt` | 跟踪特征点最大数量 |
| `freq` | 跟踪结果发布频率（Hz） |

## 图像滤波

`raw2polar()` 在 8-bit 量化后可选地对 DoP/sin(AoP)/cos(AoP) 通道施加去噪滤波，以提升暗光下 GFTT 特征点检测的稳定性。

### 可用滤波器

| 类型 | `polar_filter_type` | 说明 | 核心参数 |
|------|---------------------|------|----------|
| 无 | 0 | 默认，不滤波 | — |
| 双边滤波 | 1 | 保边去噪，OpenCV 内置 `cv::bilateralFilter` | `d`, `sigma_color`, `sigma_space` |
| 导向滤波 | 2 | 以 S0 强度图为引导，保边效果更强，自定义实现（不依赖 ximgproc） | `guided_radius`, `guided_eps` |

**导向滤波说明**：以 S0（强度图）作为引导图，在 DoP/sin/cos 上执行局部线性滤波。可以在平滑噪声的同时保留与强度边缘对齐的偏振通道边缘，避免 DoP/AoP 在物体边界处模糊。

### 测试节点

```bash
# 交互式滤波器对比（支持键盘切换/调参）
rosrun polarfp_vins test_filter
```

支持 `w` 切换滤波方法，`+/-` 调导向滤波半径，`[/]` 调 eps，`b/B` 调双边 sigmaColor，`n/N` 调 NLM 强度。
