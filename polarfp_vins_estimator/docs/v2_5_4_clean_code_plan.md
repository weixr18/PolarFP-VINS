# V2.5.4 Clean Code Plan — 精简滤波/检测器/匹配器

## Context

PolarFP-VINS 当前集成了多种滤波器、特征检测器和匹配器，其中大部分实际不使用。本次清理仅保留经过验证的最佳组合：
- **滤波**：仅保留 `0=无滤波` 和 `2=导向滤波`，移除 1(双边)、3(NLM)、4(中值)
- **检测器**：仅保留 `0=GFTT` 和 `2=SUPERPOINT`，移除 1(FAST)，代号 1 空缺
- **匹配器**：仅保留 `0=LK光流`，移除 1(BRIEF_FLANN)，`feature_matcher_type` 仅支持 0

目标：删除死代码、减少编译依赖、简化配置。

---

## Step 1: 移除滤波类型 1/3/4

### 1.1 PolarChannel.h
**文件**: `polarfp_vins_estimator/src/featureTracker/PolarChannel.h`
- 枚举 `PolarFilterType`（L62-68）：移除 `FILTER_BILATERAL = 1`, `FILTER_NLM = 3`, `FILTER_MEDIAN = 4`
- 结构体 `PolarFilterConfig`（L76-96）：移除 bilateral/NLM/median 相关成员字段，保留 `guided_radius`, `guided_eps`

### 1.2 PolarChannel.cpp
**文件**: `polarfp_vins_estimator/src/featureTracker/PolarChannel.cpp`
- 保留 `boxFilter2D()` 和 `guidedFilterSingle()`（导向滤波依赖）
- 在 `raw2polar()` 的滤波分发块（L175-213）中，移除 BILATERAL、NLM、MEDIAN 三个 `else if` 分支

### 1.3 parameters.cpp
**文件**: `polarfp_vins_estimator/src/estimator/parameters.cpp`
- 移除 bilateral 参数加载块（L251-261）
- 移除 NLM 参数加载块（L274-284）
- 移除 median 参数加载块（L287-292）
- 保留 guided 参数加载块（L264-271）

### 1.4 dark_mono_imu_config.yaml
**文件**: `config/dark/dark_mono_imu_config.yaml`
- L24: 更新注释为 `0=无滤波, 2=导向滤波`
- 移除 L27-30（双边参数）
- 移除 L36-37（median 参数）

---

## Step 2: 移除 FAST 检测器（类型 1）

### 2.1 feature_tracker_detector.h
**文件**: `polarfp_vins_estimator/src/featureTracker/feature_tracker_detector.h`
- `enum class DetectorType`（L18）：改为 `{ GFTT, SUPERPOINT }`，移除 `FAST`
- `DetectorConfig` 结构体：移除 `fast_threshold`, `fast_nonmax` 字段
- 移除整个 `class FASTDetector` 声明（L67-82）

### 2.2 feature_tracker_detector.cpp
**文件**: `polarfp_vins_estimator/src/featureTracker/feature_tracker_detector.cpp`
- 移除 `FASTDetector::detect()` 实现（L39-79）
- 移除 factory 中 `case DetectorType::FAST:` 分支（L91-92）

### 2.3 feature_tracker.cpp
**文件**: `polarfp_vins_estimator/src/featureTracker/feature_tracker.cpp`
- `initDetectorAndMatcher()` 中移除 `if (FEATURE_DETECTOR_TYPE == 1)` 分支（L249-252）
- L150 注释更新：`GFTT/FAST` → `GFTT`
- L164 注释更新：移除 BRIEF+FLANN 相关提及

### 2.4 parameters.h / parameters.cpp
**文件**: `polarfp_vins_estimator/src/estimator/parameters.h` 和 `parameters.cpp`
- parameters.h L76: 更新注释
- parameters.h L77-78: 移除 `extern int FAST_THRESHOLD` 和 `extern int FAST_NONMAX_SUPPRESSION`
- parameters.cpp L57: 更新注释
- parameters.cpp L58-59: 移除 FAST 默认值初始化
- parameters.cpp L297-300: 移除 FAST 参数 YAML 加载
- parameters.cpp L310: 更新 `det_names` 数组为 `{"GFTT", "SUPERPOINT"}`

### 2.5 dark_mono_imu_config.yaml
**文件**: `config/dark/dark_mono_imu_config.yaml`
- L44: 更新注释为 `# 0=GFTT, 1=SUPERPOINT`
- 移除 L45 `fast_threshold: 20`
- 移除 L46 `fast_nonmax_suppression: 1`

---

## Step 3: 移除 BRIEF_FLANN 匹配器（类型 1）

### 3.1 feature_tracker_matcher.h
**文件**: `polarfp_vins_estimator/src/featureTracker/feature_tracker_matcher.h`
- `enum class MatcherType`（L18）：移除 `BRIEF_FLANN`，仅保留 `LK_FLOW`
- `MatcherConfig` 结构体（L21-31）：移除 `brief_bytes`, `brief_match_dist_ratio`
- 移除整个 `class BRIEFFLANNMatcher` 声明（L98-116）
- `FeatureMatcher::extractDescriptors()`（L61-62）：移除 virtual 方法声明（仅 BRIEF 需要）

### 3.2 feature_tracker_matcher.cpp
**文件**: `polarfp_vins_estimator/src/featureTracker/feature_tracker_matcher.cpp`
- 移除 `#include <opencv2/xfeatures2d.hpp>` 块（L14-16）
- 移除 `FeatureMatcher::extractDescriptors()` 基类实现（L22-26）
- 移除整个 `BRIEFFLANNMatcher` 类实现（L99-221：构造函数、track()、extractDescriptors()）
- 简化 `createMatcher()` factory（L232-248）：直接返回 `LKFlowMatcher`，移除 switch

### 3.3 feature_tracker.h
**文件**: `polarfp_vins_estimator/src/featureTracker/feature_tracker.h`
- `ChannelState` 结构体：移除 `prev_brief_desc` 和 `brief_bytes` 字段（L77-80）

### 3.4 feature_tracker.cpp
**文件**: `polarfp_vins_estimator/src/featureTracker/feature_tracker.cpp`
- `trackImage()` 中 `matcher_->track()` 调用（L96-99）：移除 `ch.prev_brief_desc` 参数
- 移除 BRIEF 描述子提取块（L163-168）
- `initDetectorAndMatcher()` 中移除 `if (FEATURE_MATCHER_TYPE == 1)` 分支（L268-271）

### 3.5 parameters.h / parameters.cpp
**文件**: `polarfp_vins_estimator/src/estimator/parameters.h` 和 `parameters.cpp`
- parameters.h L79: 移除 `extern int FEATURE_MATCHER_TYPE`
- parameters.h L80: 移除 `extern int BRIEF_DESCRIPTOR_BYTES`
- parameters.h L81: 移除 `extern float BRIEF_MATCH_DIST_RATIO`
- parameters.cpp L60: 移除 `int FEATURE_MATCHER_TYPE = 0;`
- parameters.cpp L61: 移除 `int BRIEF_DESCRIPTOR_BYTES = 32;`
- parameters.cpp L62: 移除 `float BRIEF_MATCH_DIST_RATIO = 0.75f;`
- parameters.cpp L301-306: 移除 matcher 相关 YAML 加载
- parameters.cpp L311: 移除 `match_names` 数组和对应日志输出

### 3.6 dark_mono_imu_config.yaml
**文件**: `config/dark/dark_mono_imu_config.yaml`
- 移除 L62-67 整个 matcher params 区域（`feature_matcher_type`, `brief_descriptor_bytes`, `flann_lsh_*`, `brief_match_dist_ratio`）

---

## Step 4: 更新 CLAUDE.md & README.md

**文件**: `CLAUDE.md`（项目根目录）
- L20: 更新 featureTracker/ 目录描述，移除 BRIEF/FAST 提及
- L57: 更新 `feature_detector_type` 说明为 `0=GFTT, 2=SUPERPOINT`
- L59: 更新 `polar_filter_type` 说明为 `0=none, 2=guided`
- 移除 `feature_matcher_type` 相关文档行（因仅支持 0，可不再提及或标注为 fixed）


**文件**: `README.md`（项目根目录）

---

## Verification

1. **编译检查**：`catkin_make` 应成功编译，无未定义引用或找不到符号的错误
2. **配置检查**：`config/dark/dark_mono_imu_config.yaml` 中不应出现已删除的参数名
3. **功能检查**：使用默认配置（guided filter + GFTT + LK flow）运行一段数据，确保输出与清理前一致
