# Plan: 偏振通道可视化 Figure 2 生成程序

## Context

论文 Figure 2 需要展示不同光照条件下同一场景的斯托克斯矢量 S0/S1/S2 和偏振通道 P0/P1/P2 的对比表图（3行×6列）。输入数据为 ROS bag 中的原始偏振帧（612×512, mono8, 2×2 micropolarizer），需复用 PolarChannel 计算管线。

**C++ 代码不依赖 ROS**：用 Python 脚本从 rosbag 提取首帧存为 PNG，C++ 仅读取 PNG 图片。

## 光照条件

| bag 文件名 | 光照范围 |
|-----------|---------|
| `13-27-22` | 94–205 lux |
| `13-54-30` | 2.6–18.6 lux |
| `14-15-03` | 0.9–4.8 lux |

## 目录结构

```
paper_codes/1_vis_polar_channel/
├── CMakeLists.txt              # 独立 CMake，仅依赖 OpenCV（无 ROS）
├── extract_first_frames.py     # Python 脚本：rosbag → 首帧 PNG
├── PolarChannel.h              # 从 featureTracker/ 复制，去除 ROS 依赖
├── PolarChannel.cpp            # 同上
├── main.cpp                    # 主程序：读取 PNG + 计算 + 合成图
├── data/                       # Python 提取的原始帧 PNG
│   ├── 13-27-22.png
│   ├── 13-54-30.png
│   └── 14-15-03.png
└── output/                     # 输出目录
    ├── s0/
    ├── s1/
    ├── s2/
    ├── p0/
    ├── p1/
    ├── p2/
    └── composite.png
```

## Step 1: Python 提取首帧

`extract_first_frames.py`：
- 依赖：`rosbag`, `sensor_msgs`, `cv2`（使用系统 ROS Python 环境）
- 遍历三个 bag 文件（路径从命令行参数或硬编码读取）
- 打开 bag → `rosbag.View` → 取第一条 `/arena_cam_qc2/image_raw`
- `cv_bridge.imgmsg_to_cv2` → 存为 `data/<bag_name>.png`
- 注：若 conda 环境中无 rosbag，使用 `/usr/bin/python3` 或 `PYTHONPATH` 指向 ROS 库

## Step 2: C++ 主程序

### CMakeLists.txt

独立 CMake 项目，仅依赖 OpenCV：
- `find_package(OpenCV REQUIRED)`
- `add_executable(vis_polar_channel main.cpp PolarChannel.cpp)`
- `target_link_libraries(vis_polar_channel ${OpenCV_LIBS})`
- C++14 标准

### PolarChannel.h/cpp 修改点

1. **去 ROS 依赖**：移除 `ros/ros.h`、`sensor_msgs::ImageConstPtr` 相关接口，`ROS_INFO` → `printf`/`std::cout`
2. **PolarChannelResult 增加 S1_raw / S2_raw 输出**：PolarChannel 内部已计算 S1/S2，将其暴露出来
3. **增加 8-bit 量化接口或辅助函数**：直接输出各通道的 CV_8U 版本，方便保存

### PolarChannelResult 结构体（修改后）

```cpp
struct PolarChannelResult {
    cv::Mat S0_img;      // CV_64F, [0, 255]
    cv::Mat S1_raw;      // CV_64F, ≈[-255, 255]  — 新增
    cv::Mat S2_raw;      // CV_64F, ≈[-255, 255]  — 新增
    cv::Mat dop_img;     // CV_64F, [0, 1]
    cv::Mat sin_img;     // CV_64F, [-1, 1]
    cv::Mat cos_img;     // CV_64F, [-1, 1]
};
```

### 各通道到 8-bit 的映射

所有映射在 PolarChannel 内完成或者 PolarChannel 对外提供量化函数，main.cpp 直接拿 CV_8U 结果保存。

| 通道 | float64 来源 | uint8 映射 |
|------|-------------|-----------|
| S0 | S0_img | `round(v)`, clamp [0,255] |
| S1 | S1_raw | `v * 0.5 + 127.5`, clamp [0,255] |
| S2 | S2_raw | `v * 0.5 + 127.5`, clamp [0,255] |
| P0 (DoP) | dop_img | `v * 255` |
| P1 (sin(AoP)) | sin_img | `(v + 1) * 127.5` |
| P2 (cos(AoP)) | cos_img | `(v + 1) * 127.5` |

### 流程

```
for each image in {13-27-22, 13-54-30, 14-15-03}:
  1. imread PNG → cv::Mat (CV_8U, 612×512)
  2. PolarChannel::raw2polar → PolarChannelResult (含 S1_raw, S2_raw)
  3. 各通道量化到 CV_8U
  4. 保存各通道独立图像到 output/<ch>/

构建合成表图 composite.png:
  - 3 行 × 6 列 (S0 | S1 | S2 | P0 | P1 | P2)
  - 行标注光照条件（英文，如 "94-205 lux"），列标注通道名（英文）
  - 输出为高分辨率 PNG
```

## 关键文件清单

| 文件 | 操作 |
|------|------|
| `paper_codes/1_vis_polar_channel/extract_first_frames.py` | 新建 |
| `paper_codes/1_vis_polar_channel/main.cpp` | 新建 |
| `paper_codes/1_vis_polar_channel/CMakeLists.txt` | 新建 |
| `paper_codes/1_vis_polar_channel/PolarChannel.h` | 从 `featureTracker/` 复制 + 去 ROS 依赖 |
| `paper_codes/1_vis_polar_channel/PolarChannel.cpp` | 从 `featureTracker/` 复制 + 去 ROS 依赖 |

## 验证方式

1. `python3 extract_first_frames.py` → 检查 `data/` 下生成三个 PNG
2. `cd paper_codes/1_vis_polar_channel && mkdir -p build && cd build && cmake .. && make`
3. 运行 `./vis_polar_channel`
4. 检查 `output/composite.png` 是否为 3×6 布局（3 行光照条件，6 列通道）
5. 人工确认各通道语义正确：
   - S0 = 强度图（类灰度照片）
   - S1/S2 = 偏振差图像（高偏振区域亮）
   - P0 (DoP) = 偏振度（[0,1] → [0,255]）
   - P1/P2 = sin/cos(AoP)（偏振角的正弦/余弦分量）
6. 图中标注均为英文
