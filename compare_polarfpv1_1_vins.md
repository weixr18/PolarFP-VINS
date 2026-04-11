# PolarFP-VINS (master) vs VINS-Fusion (org-vins) 前端对比分析

## 1. 整体架构差异

### 1.1 文件组织

| 维度 | org-vins（原始 VINS-Fusion） | master（PolarFP-VINS） |
|------|------------------------------|------------------------|
| 前端入口 | `feature_tracker.h/cpp` | `polarfp_tracker.h/cpp` |
| 偏振处理 | 无 | `PolarChannel.h/cpp` — 原始 2×2 微偏振阵列 → Stokes 参数 |
| 深度学习检测 | 无 | `superpoint_detector.h/cpp` — LibTorch SuperPoint |
| 最近邻搜索 | 无（纯光流） | `nanoflann.hpp` — KD-tree 加速匹配 |
| C++ 标准 | C++14 | C++17 |
| 外部依赖 | OpenCV, Ceres, Eigen | + LibTorch (CUDA 11.3), `polarfp_camera_models` |

### 1.2 核心类对比

| 维度 | org-vins: `FeatureTracker` | master: `PolarFeatureTracker` |
|------|---------------------------|-------------------------------|
| 数据表示 | `vector<cv::Point2f>` 存坐标 + `vector<int>` 存 ID/track_cnt | `vector<PolarKeyPoint>` 统一结构体（含 channel, id, kp, track_cnt, score） |
| 通道概念 | 单灰度通道 | 4 偏振通道（S0, DoP, AoP sin, AoP cos），按通道分组管理 |
| 配置参数 | 全局参数（`MAX_CNT`, `MIN_DIST` 等） | 封装在 `PolarConfig` 结构体中，每通道独立阈值 |
| 状态管理 | 多个平行 vector（`ids`, `cur_pts`, `track_cnt` 等） | 单一 `polar_pts` vector 为真实来源，按需分组/提取 |
| 检测器切换 | 仅 GFTT | FAST / GFTT / SuperPoint 三种可选 |

### 1.3 数据流对比

**org-vins:**
```
灰度图像 → LK 光流(prev→cur) → 反向光流检查 → setMask()掩码
    → goodFeaturesToTrack补充 → 去畸变 → 速度计算 → VINS格式输出
```

**master:**
```
原始偏振图像 → PolarChannel::raw2polar() → 4通道图像(S0/DoP/AoP sin/AoP cos)
    → 每通道独立检测(FAST/GFTT/SuperPoint) → LK光流+反向检查
    → KD-tree最近邻匹配 → 多通道联合RANSAC → 评分/NMS筛选
    → ID分配/评分剪枝 → VINS格式输出
```

---

## 2. 图像预处理差异

### 2.1 预处理方法

| 维度 | org-vins（原始 VINS-Fusion） | master（PolarFP-VINS） |
|------|------------------------------|------------------------|
| 方法 | **CLAHE**（代码存在但已注释，**实际未启用**） | **三重中值滤波**（核大小 3，连续 3 次） |
| 作用通道 | 灰度图（左/右相机） | 仅偏振导出通道（dop, aopsin, aopcos），**S0 通道不滤波** |
| 目的 | 对比度增强，改善光照不均 | 去传感器噪声，稳定偏振参数 |

**org-vins CLAHE 代码（已注释）:**
```cpp
// feature_tracker.cpp, trackImage() 中:
/*
{
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(cur_img, cur_img);
    if(!rightImg.empty())
        clahe->apply(rightImg, rightImg);
}
*/
```

**master 中值滤波代码:**
```cpp
// polarfp_tracker.cpp, applyFilter():
cv::medianBlur(image, image_1, 3);
cv::medianBlur(image_1, image_2, 3);
cv::medianBlur(image_2, image_3, 3);
return image_3;
```

该滤波仅在 `getPolarizationImage()` 中应用于偏振通道：
```cpp
dop_img = applyFilter(dop_img);
aopsin_img = applyFilter(aopsin_img);
aopcos_img = applyFilter(aopcos_img);
// S0 通道不做任何滤波
```

### 2.2 影响分析

- **org-vins**: 无实际预处理，直接对原始灰度图做光流跟踪
- **master**: 偏振导出通道（DoP/AoP）对中值滤波敏感，连续 3 次平滑可有效抑制偏振传感器特有的椒盐噪声；S0 通道保持原始强度信息不变，以保证 GFTT/FAST 检测器对强度角点的灵敏度

---

## 3. 特征点检测差异

### 3.1 检测方法

| 维度 | org-vins | master |
|------|----------|--------|
| 唯一方法 | **GFTT** (`cv::goodFeaturesToTrack`) | **三种可选**: FAST / GFTT / SuperPoint |
| 输入 | 单张灰度图 | 4 张偏振通道图（分别处理） |
| 空间分布 | 通过 `setMask()` 圆形掩码保证 | NMS 半径抑制 + 掩码 + 评分排序 |
| 目标数量 | 全局 `MAX_CNT`（所有特征总数上限） | `KP_NUM_TARGET = 1000`（跨通道目标总数） |

### 3.2 具体参数对比

**org-vins GFTT 参数:**
```cpp
// feature_tracker.cpp
goodFeaturesToTrack(cur_img, n_pts, MAX_CNT - cnt, 0.01, MIN_DIST, mask);
// qualityLevel = 0.01, minDistance = MIN_DIST（来自 YAML 配置）
```

**master - FAST 模式（每通道独立阈值）:**
```cpp
// polarfp_tracker.cpp / PolarConfig
FAST_S0_DT   = FastFeatureDetector::create(5);    // S0: 低阈值，捕获更多特征
FAST_DOP_DT  = FastFeatureDetector::create(15);   // DoP: 中等阈值
FAST_AOP_DT  = FastFeatureDetector::create(80);   // AoP: 高阈值，仅强极化角
```

**master - GFTT 模式:**
```cpp
GFTT_DT = GFTTDetector::create(500, 0.03, 7, 3, false);
// maxCorners=500, qualityLevel=0.03, minDistance=7, blockSize=3, harris=false
// 注意：每通道最多 500 个点，4 通道理论上最多 2000 个点
```

**master - SuperPoint 模式（默认）:**
```cpp
SP_KEYPOINT_THRESHOLD = 0.015f;  // 最小检测概率
SP_NMS_RADIUS = 4;               // 1/8 尺度下的 NMS 半径
SP_USE_GPU = true;               // 使用 CUDA 推理
```

### 3.3 检测策略差异

| 维度 | org-vins | master |
|------|----------|--------|
| 优先级 | `setMask()` 按 track_cnt 降序，长跟踪特征优先占位 | 评分排序 + NMS，新特征可通过评分竞争 |
| 特征补充 | 光流跟踪后，在 mask 空白区域调用 goodFeaturesToTrack 补满 | 每通道独立检测，合并后通过 NMS 和评分筛选 |
| 多样性 | 仅灰度角点 | 4 种互补信息（强度+极化度+极化角） |
| 自适应 | 无 | 不同通道有不同检测器/阈值，可根据噪声特性调整 |

---

## 4. 特征点匹配差异

### 4.1 光流跟踪

| 维度 | org-vins | master |
|------|----------|--------|
| 基本方法 | LK 光流 (`cv::calcOpticalFlowPyrLK`) | LK 光流（同左），**但按通道分别执行** |
| 窗口大小 | 21×21 | 21×21（`LK_WIN_SIZE`） |
| 金字塔层数 | 3 层 | **6 层**（`LK_MAX_LEVEL`） |
| 收敛条件 | 默认 | EPS+COUNT, 10次迭代, 0.01精度（`LK_CRITERIA`） |
| 反向检查 | 可选（`FLOW_BACK` 参数控制） | **始终启用**，往返误差 < 0.5px |
| IMU 预测 | 有：`cv::OPTFLOW_USE_INITIAL_FLOW` 标志，1层金字塔 | 有：通过 `setPrediction()` 传入 |

### 4.2 匹配策略

**org-vins — 纯光流跟踪:**
```
prev_pts → calcOpticalFlowPyrLK → status筛选 → 保留成功跟踪的点
```
- 无描述子匹配，无 KD-tree
- 仅依赖光流的连续性，新帧新特征通过 goodFeaturesToTrack 检测后直接分配新 ID

**master — 光流 + KD-tree 混合匹配:**
```
1. LK 光流预测当前帧位置 (prev → cur)
2. 反向光流验证 (cur → prev)，筛选可靠跟踪
3. 在当前帧检测新关键点
4. KD-tree 搜索：在 LK 预测位置附近找最近邻（max dist = 10px）
5. 贪婪冲突解决：最近对优先
```
- 关键差异：引入了**描述符无关的 KD-tree 空间匹配**，即使光流失败也能通过空间位置找到对应关系
- `nanoflann` 库用于加速最近邻搜索

### 4.3 匹配结果组织

**org-vins:**
```cpp
// 全局并行 vector
vector<int> ids;              // 特征 ID
vector<cv::Point2f> cur_pts;  // 当前像素坐标
vector<cv::Point2f> prev_pts; // 上一帧像素坐标
vector<int> track_cnt;        // 跟踪计数
```

**master:**
```cpp
// 按通道分组的匹配对
ChannelPairs matched_pairs;  // map<string, vector<MatchedPair>>
// MatchedPair = pair<PolarKeyPoint, PolarKeyPoint>  // (prev_kp, curr_kp)
```

---

## 5. RANSAC 与外点剔除差异

### 5.1 前端 RANSAC

| 维度 | org-vins | master |
|------|----------|--------|
| 方法 | `cv::findFundamentalMat` + FM_RANSAC | `cv::findFundamentalMat` + FM_RANSAC |
| 阈值 | `F_THRESHOLD`（来自 YAML，通常 ~1.0） | `RANSAC_REMAP_THR = 2.0` |
| 置信度 | 0.99 | `RANSAC_CONFIDENCE = 0.99` |
| **是否启用** | **默认禁用**（`rejectWithF()` 在 `trackImage()` 中被注释掉） | **始终启用**（`combinedFeatureRANSAC()` 在匹配后自动调用） |
| 数据范围 | 单一灰度通道的全部匹配点 | **所有 4 通道匹配点拼接后统一计算**一个基础矩阵 |

**关键差异**: org-vins 中 F 矩阵 RANSAC 虽然存在但被注释掉未使用；master 中将其作为核心外点剔除步骤，且是多通道联合使用。

### 5.2 外点剔除流程

**org-vins:**
```
光流成功 → setMask() 空间筛选 → (可选 F 矩阵 RANSAC，已禁用)
    → 边界检查(inBorder) → 输出
```

**master:**
```
LK 光流 + 反向检查 → KD-tree 匹配
    → combinedFeatureRANSAC()（多通道联合 F 矩阵 RANSAC）
    → NMS 空间抑制 → 评分排序剪枝 → 输出
```

### 5.3 后端反馈剔除

两者都有 `removeOutliers()` 方法，由后端 `outliersRejection()` 调用：
- **org-vins**: 通过 `ids` vector 查找并移除对应 ID 的点，同时清理 `prev_pts`/`cur_pts`
- **master**: 通过 `removePtsIds` 集合过滤 `prev_polar_pts`/`cur_polar_pts`

---

## 6. 特征点管理差异

### 6.1 空间分布控制

**org-vins — `setMask()` 掩码法:**
```cpp
// 1. 初始化全白 mask
// 2. 按 track_cnt 降序排序已有特征
// 3. 遍历：若该位置 mask 为白，保留特征并在该位置画黑圆（半径 MIN_DIST）
// 4. 新特征检测在 mask 的白区进行
```
- **优点**: 简单、保证长跟踪特征优先
- **缺点**: 二值决策（要么保留要么丢弃），无质量评估

**master — NMS + 评分法:**
```cpp
// 1. 按通道独立检测特征
// 2. 合并所有通道特征
// 3. efficientNMS() 空间抑制（半径 21px）
// 4. 评分排序，保留 top-KP_NUM_TARGET
// 5. mask 阻止已占区域
```
- **优点**: 多维度评估（评分累积）、NMS 更精细
- **缺点**: 计算量更大

### 6.2 特征质量评估

| 维度 | org-vins | master |
|------|----------|--------|
| 质量指标 | 仅 `track_cnt`（跟踪帧数） | `score`（累积评分）+ `track_cnt` |
| 评分累积 | 无 | 每次跟踪成功累加到 `score` |
| 淘汰策略 | 掩码覆盖（被长跟踪特征的空间位置覆盖） | 评分排名，低于阈值被淘汰 |
| 新特征竞争 | 在掩码空白区域补充 | 通过 NMS 和评分与旧特征竞争 |

### 6.3 ID 管理

| 维度 | org-vins | master |
|------|----------|--------|
| ID 分配 | 全局单调递增 `n_id++` | 全局单调递增 `n_id++`（相同） |
| ID 持久性 | 光流跟踪成功保留 ID | 匹配成功保留 ID，新检测分配新 ID |
| ID 与通道 | 无通道概念 | `PolarKeyPoint` 记录所属 `channel` |

---

## 7. 单双目支持差异

### 7.1 总体差异

这是两个分支最显著的架构差异之一：**master 分支完全移除了双目支持**。

### 7.2 逐项对比

| 维度 | org-vins | master |
|------|----------|--------|
| `STEREO` 全局变量 | 有 | **已删除** |
| 双目相机加载 | `NUM_OF_CAM==2` 时加载 cam1 标定 + 外参 | **已删除**，即使配置 2 个相机也不加载第二个 |
| 前端立体匹配 | `cv::calcOpticalFlowPyrLK(left→right)` 同帧匹配 | `trackImage()` 中 `_img1` 参数**直接忽略** |
| 立体特征 ID | 左图 ID 和右图 ID 分开维护（`ids`/`ids_right`） | 无立体 ID 概念 |
| 后端立体三角化 | `FeatureManager::triangulate()` 中有同帧左/右三角化分支 | **已删除**，仅保留多帧时序三角化 |
| 后端立体残差 | `ProjectionTwoFrameTwoCamFactor`、`ProjectionOneFrameTwoCamFactor` | **已删除**，仅用 `ProjectionTwoFrameOneCamFactor` |
| 立体初始化 | 有 `STEREO+IMU` 和 `STEREO-only` 两条初始化路径 | **已删除**，仅保留 `MONO+IMU` |
| 立体外点检查 | `outliersRejection()` 中有右相机重投影检查 | **已删除** |

### 7.3 代码层面证据

**org-vins (estimator.cpp) — 三种初始化路径:**
```cpp
if (!STEREO && USE_IMU) { /* 单目+IMU */ }
if (STEREO && USE_IMU)  { /* 双目+IMU */ }
if (STEREO && !USE_IMU) { /* 仅双目 */ }
```

**master (estimator.cpp) — 仅一种:**
```cpp
if (frame_count == WINDOW_SIZE) { /* 仅单目+IMU，无 STEREO 判断 */ }
```

---

## 8. 与后端交互差异

### 8.1 特征数据格式

两者输出格式相同（保持向后兼容）：
```cpp
map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
// key: camera_id (0 or 1)
// value: vector of {feature_id, [x, y, z, u, v, vel_x, vel_y]}
```

### 8.2 预测反馈

| 维度 | org-vins | master |
|------|----------|--------|
| 预测来源 | `predictPtsInNextFrame()` 用常速度模型投影 3D 特征到下一帧 | 相同 |
| 预测使用 | `setPrediction()` → 光流 `USE_INITIAL_FLOW` | `setPrediction()` → 光流初始化 |

### 8.3 后端外点剔除反馈

**两者流程相同:**
```
optimization() → outliersRejection()（重投影误差 > 阈值）
    → featureTracker.removeOutliers() + f_manager.removeOutlier()
```

差异在于 master 的 `outliersRejection()` 仅检查单目重投影误差（无立体检查）。

### 8.4 FeatureManager 变化

| 维度 | org-vins | master |
|------|----------|--------|
| 类定义 | 不变 | 不变（仅 CRLF→LF 换行符差异） |
| `addFeatureCheckParallax()` | assert 硬检查相机 ID | 改为软检查 + `ROS_WARN`，增加健壮性 |
| `triangulate()` | 含同帧双目三角化分支 | 删除双目分支，仅保留时序三角化 |

---

## 9. 参数配置差异

### 9.1 新增/删除参数

| 参数 | org-vins | master |
|------|----------|--------|
| `STEREO` | 有 | **删除** |
| `POLAR_CHANNEL` | 无 | **新增**（声明但未在 readParameters 中读取） |
| `VINS_RESULT_NAME` | 无（硬编码 "vio.csv"） | **新增**（可配置，输出 `{name}.txt`） |

### 9.2 关键参数对照表

| 功能 | org-vins 参数 | master 参数 | 备注 |
|------|---------------|-------------|------|
| 最大特征数 | `MAX_CNT` (YAML) | `KP_NUM_TARGET` = 1000 | master 为固定值 |
| 特征间距 | `MIN_DIST` (YAML) | `NMS_RADIUS` = 21 | master 为固定值 |
| F 矩阵阈值 | `F_THRESHOLD` (YAML) | `RANSAC_REMAP_THR` = 2.0 | master 为固定值 |
| LK 窗口 | 硬编码 21×21 | `LK_WIN_SIZE` = 21×21 | 相同 |
| LK 金字塔 | 硬编码 3 层 | `LK_MAX_LEVEL` = 6 | **master 翻倍** |
| 反向光流 | `FLOW_BACK` (YAML, 可选) | 始终启用 | master 无此开关 |

---

## 10. 总结

### 10.1 master 相对 org-vins 的主要变更

1. **前端完全重写**: `FeatureTracker` → `PolarFeatureTracker`，从单一灰度跟踪变为 4 通道极化跟踪
2. **图像预处理替换**: 无（org-vins 的 CLAHE 代码已注释未启用）→ 三重中值滤波（仅偏振通道）
3. **检测方法多样化**: 仅 GFTT → FAST/GFTT/SuperPoint 三种可选
4. **匹配策略升级**: 纯光流 → 光流 + KD-tree 空间匹配
5. **RANSAC 从禁用变为核心**: F 矩阵 RANSAC 从注释状态变为核心外点剔除手段
6. **特征管理精细化**: 简单 track_cnt → 评分累积 + NMS + 多通道联合管理
7. **双目支持完全删除**: 所有双目相关代码（前端、后端、初始化、优化因子）被移除
8. **健壮性提升**: assert 硬检查 → 软检查 + 警告日志
9. **计算成本增加**: 多通道处理约 2-3 倍于单通道

### 10.2 后续开发建议

- **若需恢复双目支持**: 需重写 `PolarFeatureTracker::trackImage()` 处理 `_img1`，并在后端恢复立体残差因子
- **若需调整特征检测**: 可通过 `PolarConfig` 切换 FAST/GFTT/SuperPoint，或修改每通道阈值
- **若需新增通道**: 修改 `PolarChannel::raw2polar()` 并在 `PolarConfig.ALL_CHANNELS`/`FP_CHANNELS` 中注册
- **性能优化方向**: LK 光流按通道分别执行是主要耗时点，可考虑并行化或减少通道数
- **若需启用 CLAHE**: org-vins 中已有代码（已注释），可对 S0 通道启用以增强暗光对比度
