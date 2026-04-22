# V2.2: 模块化前端 — FAST + BRIEF(completed)

## 概述

| 项目 | 内容 |
|------|------|
| **目标** | 将 `trackImagePolar()` 中硬编码的预处理/GFTT检测/LK光流拆分为可插拔模块 |
| **新增管线** | `{GFTT, FAST} × {LK_FLOW, BRIEF_FLANN}` 四种组合 |
| **精简** | 删除原版 `trackImage()` 非偏振 VINS 代码，精简代码库 |
| **状态** | 完成 (2026-04-13) |

## 设计架构

### 模块接口

```
FeatureTracker
├── FeatureDetector (抽象接口)
│   ├── GFTTDetector   — cv::goodFeaturesToTrack 包装
│   └── FASTDetector   — cv::FAST + 响应值排序 + mask 过滤
│
├── FeatureMatcher (抽象接口)
│   ├── LKFlowMatcher      — Lucas-Kanade 光流 + 双向检查
│   └── BRIEFFLANNMatcher  — BRIEF/ORB 描述子 + FLANN LSH Hamming 匹配
│
└── ChannelState (每通道状态)
    ├── prev_brief_desc     — BRIEF 模式下存储上一帧描述子
    └── brief_bytes = 32    — 描述子字节数
```

### 新增文件

| 文件 | 内容 |
|------|------|
| `feature_tracker_detector.h` | 检测器接口 + `DetectorConfig` + `GFTTDetector` / `FASTDetector` 声明 |
| `feature_tracker_detector.cpp` | 两检测器实现 + `createDetector()` 工厂函数 |
| `feature_tracker_matcher.h` | 匹配器接口 + `MatcherConfig` + `LKFlowMatcher` / `BRIEFFLANNMatcher` 声明 |
| `feature_tracker_matcher.cpp` | 两匹配器实现 + FLANN LSH 匹配 + `createMatcher()` 工厂函数 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `feature_tracker.h` | 新增 `detector_`, `matcher_` 指针; `ChannelState` 新增 `prev_brief_desc`; `initDetectorAndMatcher()` 移入 public |
| `feature_tracker.cpp` | **删除**旧 `trackImagePolar()` + 非偏振 `trackImage()` (~390行); **新增**模块化 `trackImage()` |
| `parameters.h` | 新增 8 个 extern 变量 |
| `parameters.cpp` | 新增 YAML 读取 + 日志 |
| `estimator.cpp` | `setParameter()` 中调用 `initDetectorAndMatcher()` |
| `CMakeLists.txt` | 新增 2 个源文件到 `vins_lib` |
| `dark_mono_imu_config.yaml` | 新增 9 个配置项 |

## 模块化 trackImage 流程

```
trackImage(_cur_time, _img, _img1)
│
├─ 1. raw2polar() — 原始偏振图像分解为多通道
│
├─ 2. 每通道独立管线
│   ├─ 2a. matcher_->track()      — LK光流 或 BRIEF+FLANN (替代硬编码光流)
│   ├─ 2b. 边界检查                — inBorderImpl()
│   ├─ 2c. setMaskForChannel()     — 空间均匀分布
│   ├─ 2d. detector_->detect()     — GFTT 或 FAST (替代硬编码 goodFeaturesToTrack)
│   ├─ 2e. 新特征 + n_id++         — 全局唯一ID
│   ├─ 2f. 提取描述子               — BRIEF模式下 extractDescriptors()
│   ├─ 2g. undistortedPts()        — 归一化坐标
│   ├─ 2h. 速度计算                 — 内联实现 (替代 ptsVelocityForChannel)
│   └─ 2i. 更新 prev 状态           — 含 prev_brief_desc
│
├─ 3. 合并通道 → VINS 格式          — 7维向量 (x,y,z,u,v,vx,vy)
├─ 4. 可视化                        — drawTrackPolar()
└─ 5. 更新全局状态                  — hasPrediction = false
```

## YAML 配置

### 新增参数 (`dark_mono_imu_config.yaml`)

```yaml
# 特征检测器
feature_detector_type: 0        # 0=GFTT, 1=FAST
fast_threshold: 20
fast_nonmax_suppression: 1

# 特征匹配器
feature_matcher_type: 0         # 0=LK_FLOW, 1=BRIEF_FLANN
brief_descriptor_bytes: 32
flann_lsh_tables: 20
flann_lsh_key_size: 20
flann_multi_probe: 20
brief_match_dist_ratio: 0.75
```

## 代码位置 (refactoring 后)

| 模块 | 位置 |
|------|------|
| 模块化 `trackImage()` | `feature_tracker.cpp` L317–499 |
| 辅助函数 (setPolarChannels 等) | `feature_tracker.cpp` L168–315 |
| `rejectWithF()` 等保留函数 | `feature_tracker.cpp` L501+ |
| 检测器/匹配器初始化 | `feature_tracker.cpp` L140–173 |

### 已删除

- `trackImagePolar()` — 被模块化 `trackImage()` 替代
- 旧非偏振 `trackImage()` — 当前项目仅使用偏振模式
- 全局 `distance()` 函数 — 已被 `FeatureTracker::distance()` 成员函数替代
- `ptsVelocityForChannel()` — 速度计算逻辑内联到 `trackImage()`

## 编译修复记录

1. **BRIEFFLANN 构造函数重定义** — header 中 inline 实现与 .cpp 定义冲突 → header 改为纯声明
2. **`initDetectorAndMatcher()` 访问权限** — 原为 private → 移入 public
3. **未使用变量 `j`** — LKFlowMatcher 中残留变量 → 删除
4. **未使用变量 `idx2`** — BRIEFFLANN ratio test 中多余变量 → 删除
