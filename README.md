# PolarFP-VINS：**基于偏振的视觉惯性导航系统**

在 [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) 基础上进行系统性改进，核心创新在于**将偏振相机提供的多通道偏振信息（S0 强度、DoP 偏振度、AoP 偏振角）引入 VINS 前端特征跟踪管线**，使视觉惯性里程计在低照度、高动态等传统强度视觉失效的恶劣条件下仍能保持鲁棒跟踪。

> 本项目为科研性质的算法验证平台，旨在探索偏振信息在 VSLAM 前端中的应用潜力。

## 1 创新点

- **多通道偏振特征跟踪** — 从偏振相机 2×2 微偏振片阵列原始图像中解码 Stokes 参数，同时提取 S0（总强度）、DoP（偏振度）、sin(AoP)/cos(AoP)（偏振角）四个通道，每个通道独立运行 VINS 的成熟检测/跟踪管线，最后合并输出给后端
- **可配置特征检测器** — 前端特征检测器可在 GFTT、SuperPoint 之间切换（通过 `feature_detector_type`），SuperPoint 基于 LibTorch 推理，支持多通道批量单次前向传播
- **可配置图像滤波** — 8-bit 量化后可选地对 DoP/sin/cos 通道施加导向滤波，抑制偏振传感器特有的噪声，提升暗光下特征检测稳定性
- **后端零侵入** — 后端滑动窗口优化器（IMU 预积分 + 视觉重投影误差 + 边缘化）完全不知道特征点来自哪个偏振通道，所有偏振信息在前端融合，保持与 VINS-Fusion 后端的完全兼容
- **暗光鲁棒性** — 偏振信息（特别是 DoP 和 AoP）在低照度下比纯强度信息更稳定，能捕获强度图中不可见的纹理和边缘

## 2 project informations

### 2.1 项目结构

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
│   ├── docs/                  # 重构文档
│   │   ├── compare_polarfpv1_1_vins.md    # V1.1 与原版 VINS-Fusion 对比分析
│   │   └── polarfp_v2_reconstruction_plan.md   # V2 前端重构计划
│   ├── nn/                    # SuperPoint 模型文件
│   │   ├── superpoint_v1.pt              # TorchScript 模型
│   │   ├── superpoint_v1.pth             # PyTorch 权重
│   │   └── convert_to_torchscript.py     # 转换脚本
│   └── src/
│       ├── estimator/         # 滑动窗口状态估计器（后端，未修改）
│       ├── factor/            # Ceres 因子（IMU 预积分、投影误差、边缘化，未修改）
│       ├── featureTracker/    # 特征跟踪器 + 偏振通道解码（核心修改区域）
│       │   ├── feature_tracker.h/cpp           # 主跟踪器，V2 重构为多通道
│       │   ├── feature_tracker_detector.h/cpp  # 特征检测器抽象（GFTT/SuperPoint）
│       │   ├── feature_tracker_matcher.h/cpp   # 特征匹配器（LK光流）
│       │   ├── superpoint_detector.h/cpp       # SuperPoint 检测器实现
│       │   └── PolarChannel.h/cpp              # 偏振解码 + 可选滤波
│       ├── initial/           # 初始化（SFM、外参标定、陀螺仪对齐，未修改）
│       └── utility/           # 可视化工具
└── README.md
```

### 2.2 依赖

- ROS（Melodic / Noetic）
- OpenCV 3/4
- Ceres Solver
- Eigen3
- LibTorch（可选，启用 SuperPoint 时需要）

### 2.3 编译

```bash
cd ~/ws/vi_catkin_ws
catkin_make
source devel/setup.bash
```

**SuperPoint 编译说明：** CMake 选项 `USE_SUPERPOINT`（默认 ON）控制是否启用 SuperPoint 支持。启用时会自动查找 LibTorch 并链接相关库。LibTorch 路径硬编码在 CMakeLists.txt 中。

### 2.4 运行

```bash
roslaunch polarfp_vins run.launch config:=dark/dark_mono_imu_config.yaml
```

## 3 偏振相机原理与偏振通道计算

### 3.1 微偏振片阵列（DoFP）相机成像原理

本项目使用的传感器为**分焦平面型（Division-of-Focal-Plane, DoFP）偏振相机**，其核心结构是在 CMOS 像面上直接集成一层 **2×2 微偏振片阵列（Micro-Polarizer Array, MPA）**。每个 2×2 超像素单元内，4 个相邻像素分别覆盖不同透振方向的偏振片：

```
超像素单元（2×2）：
┌─────────┬─────────┐
│  90°    │  45°    │    ← 第 i 行
│ (0,0)   │ (0,1)   │
├─────────┼─────────┤
│  135°   │   0°    │    ← 第 i+1 行
│ (1,0)   │ (1,1)   │
└─────────┴─────────┘
  第 j 列    第 j+1 列
```

这种结构与拜耳（Bayer）彩色滤波阵列类似，但每个像素记录的是**特定偏振方向的光强分量** $I_\theta$，而非颜色信息。四个偏振角度（0°/45°/90°/135°）同时曝光采集，无需旋转偏振片，因此可以实时获取场景的偏振信息。

### 3.2 从原始图像到 Stokes 参数

偏振光在物理上由 **Stokes 向量** $[S_0, S_1, S_2, S_3]^T$ 完整描述。对于线偏振（$S_3 = 0$），只需前三个分量：

$$S_0 = \frac{I_0 + I_{45} + I_{90} + I_{135}}{4}$$
$$S_1 = I_0 - I_{90}$$
$$S_2 = I_{45} - I_{135}$$

代码中从原始图像解码 Stokes 参数的完整流程在 [`PolarChannel.cpp:172-323`](vins_estimator/src/featureTracker/PolarChannel.cpp#L172-L323) 的 `raw2polar()` 函数中实现：

```
原始图像（单通道 8bit）
    │
    ▼ ① 解复用：从 2×2 超像素网格中按偏移量采样 4 个偏振角度子图像
    │   (0,0)→90°, (0,1)→45°, (1,0)→135°, (1,1)→0°  [PolarChannel.cpp:205-208]
    │
    ▼ ② 计算 Stokes 向量
    │   S0 = (I90+I45+I135+I0)/4    [PolarChannel.cpp:211]
    │   S1 = I0 - I90               [PolarChannel.cpp:212]
    │   S2 = I45 - I135             [PolarChannel.cpp:213]
    │
    ▼ ③ 派生物理量
    │   DoP   = √(S1²+S2²) / S0     [PolarChannel.cpp:249-250] 偏振度 [0,1]
    │   AoP   = 0.5·atan2(S2, S1)   [PolarChannel.cpp:227-228] 偏振角 [-π/2, π/2]
    │   sin(AoP) = S2/√(S1²+S2²)   [PolarChannel.cpp:222]
    │   cos(AoP) = S1/√(S1²+S2²)   [PolarChannel.cpp:223]
    │
    ▼ ④ 无效区域剔除
    │   S0 接近 0 或 DoP > 0.999 的像素置零  [PolarChannel.cpp:253-262]
    │
    ▼ ⑤ 量化为 8bit 图像
    │   DoP:   [0,1] → [0,255]       [PolarChannel.cpp:266]
    │   sin/cos: [-1,1] → [0,255]    [PolarChannel.cpp:267-268]
    │   S0:    直接截断               [PolarChannel.cpp:269]
    │
    ▼ ⑥ 可选滤波去噪（双边/导向/NLM/中值） [PolarChannel.cpp:272-310]
    │
    ▼ 输出：S0_img, dop_img, sin_img, cos_img
```

**为什么使用 sin(AoP) 和 cos(AoP) 分量而非直接使用 AoP？**

AoP 是一个角度量，存在 $\pm \pi/2$ 周期性跳变（winding discontinuity），直接作为图像通道会在边界处产生数值不连续，导致特征检测器误检。将其分解为正弦/余弦分量后，每个分量都是连续的标量场，更适合作为传统特征检测器的输入。

### 3.3 为什么将偏振相机引入 VSLAM

传统 VSLAM 系统仅依赖强度（灰度）图像进行特征提取与跟踪，在以下场景中面临严重退化：

| 场景 | 强度图像的缺陷 | 偏振通道的优势 |
|------|----------------|----------------|
| **低照度** | 信噪比急剧下降，纹理模糊，特征点数量骤减 | DoP 和 AoP 是光强的**比值/角度量**，对绝对亮度不敏感，在暗光下仍能保留纹理对比度 |
| **光照突变** | 同一场景在不同光照下强度差异大，特征匹配容易失败 | 偏振属性由物体表面材质和几何决定，与环境光照强度弱相关，跨帧一致性更好 |
| **弱纹理/同色区域** | 强度图中缺乏梯度变化，无法提取足够特征点 | 不同材质（如金属、玻璃、塑料）的偏振特性差异显著，偏振通道能"看到"强度图中不可见的边缘 |
| **反光/高光** | 镜面反射区域强度饱和，特征跟踪丢失 | 偏振信息可以有效区分漫反射和镜面反射成分，抑制高光干扰 |

**系统设计哲学**：偏振通道不是要替代强度通道，而是**互补增强**。S0（强度）在光照充足时仍然是最优的特征提取通道，而 DoP/AoP 在强度退化时提供额外的纹理线索。我们将 4 个通道各自独立运行 VINS-Fusion 成熟的检测/跟踪管线，然后合并结果，使后端在任意时刻都能从最"健康"的通道中获取特征——这正是 PolarFP-VINS 在暗光条件下实现鲁棒跟踪的核心机制。

## 4 配置说明

YAML 配置文件中的关键字段：

### 4.1 基础参数

| 参数 | 说明 |
|------|------|
| `use_polar` | 是否启用偏振通道（1=启用，0=退化为原始 VINS-Fusion 行为） |
| `polar_channels` | 使用的偏振通道组合，如 `"s0,dop,aopsin,aopcos"` |
| `num_of_cam` | 相机数量（1=单目） |
| `imu` | 是否使用 IMU |
| `max_cnt` | 每通道跟踪特征点最大数量 |
| `freq` | 跟踪结果发布频率（Hz） |

### 4.2 偏振滤波

| 参数 | 说明 |
|------|------|
| `polar_filter_type` | 滤波器类型：0=无，2=导向滤波 |
| `polar_guided_radius` | 导向滤波窗口半径（filter_type=2 时） |
| `polar_guided_eps` | 导向滤波正则化参数（filter_type=2 时） |


### 4.3 特征检测器

| 参数 | 说明 |
|------|------|
| `feature_detector_type` | 检测器类型：0=GFTT，2=SuperPoint |
| `superpoint_model_path` | SuperPoint TorchScript 模型路径（支持相对于配置目录的相对路径） |
| `superpoint_use_gpu` | SuperPoint 是否使用 GPU：1=GPU，0=CPU |
| `superpoint_keypoint_threshold` | SuperPoint 特征点概率阈值 |
| `superpoint_nms_radius` | SuperPoint NMS 半径（1/8 输出分辨率下） |

## 5 Implementation details

### 5.1 图像滤波

`raw2polar()` 在 8-bit 量化后可选地对 DoP/sin(AoP)/cos(AoP) 通道施加去噪滤波，以提升暗光下特征点检测的稳定性。

**可用滤波器**:

| 类型 | `polar_filter_type` | 说明 | 核心参数 |
|------|---------------------|------|----------|
| 无 | 0 | 默认，不滤波 | — |
| 导向滤波 | 2 | 以 S0 强度图为引导，保边效果更强，自定义实现（不依赖 ximgproc） | `guided_radius`, `guided_eps` |

**导向滤波说明**：以 S0（强度图）作为引导图，在 DoP/sin/cos 上执行局部线性滤波。可以在平滑噪声的同时保留与强度边缘对齐的偏振通道边缘，避免 DoP/AoP 在物体边界处模糊。

### 5.2 特征检测器

前端通过 `FeatureDetector` 抽象接口支持两种检测器：

| 检测器 | 类型值 | 说明 |
|--------|--------|------|
| GFTT | 0 | OpenCV `cv::goodFeaturesToTrack`，原始 VINS-Fusion 默认 |
| SuperPoint | 2 | LibTorch 推理，支持多通道批量单次前向传播，需 `USE_SUPERPOINT` CMake 选项 |

#### SuperPoint 集成

SuperPoint 作为 `FeatureDetector` 的实现通过工厂模式接入。架构采用 PIMPL（指向实现的指针）惯用法，将 `<torch/script.h>` 隔离在源文件中，避免 `c10::nullopt` 与 `std::nullopt` 冲突。

- **非偏振模式**：单张图像调用 `detect()`，返回 SuperPoint 检测的关键点
- **偏振模式**：调用 `detectBatchForChannels()` 将所有 N 个通道图像打包为一次 LibTorch 前向传播，然后逐通道后处理（softmax、NMS、阈值过滤、掩码过滤、max_cnt 截断），后续逐通道调用 `detect()` 从缓存返回结果


## 迭代历史

| 版本 | 状态 | 说明 |
|------|------|------|
| V0 (`663e403`) | 基线 | 从 VINS-Fusion 重置，编译通过，功能等价于原版 |
| V2 (`1dde542`) | 历史 | 最小化改动策略：复用 VINS 成熟管线，每通道独立运行后合并结果 |
| V2.1 (`e304552`) | 历史 | 在 V2 基础上加入可配置图像滤波模块（双边/导向/NLM） |
| V2.2 (`6cc5f1d`) | 历史 | 前端模块化：FeatureDetector 抽象（GFTT/FAST），FeatureMatcher 抽象（LK/BRIEF+FLANN），文件拆分为 detector/matcher 模块 |
| V2.3 (`4e56be4`) | 当前 | 在 V2.2 基础上集成 SuperPoint 检测器（LibTorch，PIMPL 模式，多通道批量推理） |
