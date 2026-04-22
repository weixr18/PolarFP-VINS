# PolarFP-VINS 双目支持方案

## 背景

VINS-Fusion（上游）原生支持单目、单目+IMU、双目、双目+IMU 四种配置。PolarFP-VINS 从它 fork 出来后只实现了单目偏振前端，**后端（估计器、Ceres 因子、特征管理器）已经是完整双目就绪的**——只是前端从未向它发送过右目观测。

**目标**：扩展 PolarFP 前端，使其同时处理左、右两路原始偏振图像，各自解码为偏振通道，左目做时序跟踪，左→右做立体匹配，向后端同时发送左右观测，使其能立即三角化深度。

## 可行性评估

**高可行性**。唯一缺失的环节是前端（`feature_tracker.cpp`）。后端无需任何改动：
- `FeatureManager` 已有 `rightObservation()`、`is_stereo`、`pointRight/uvRight`
- 三种立体 Ceres 因子已存在：`ProjectionOneFrameTwoCamFactor`、`ProjectionTwoFrameTwoCamFactor`、`ProjectionTwoFrameOneCamFactor`
- `processImage()` 中已有双目初始化路径
- ROS 节点已订阅左右两个图像话题并做时间同步

---

## 架构

```
              立体（空间）
    左相机 <--------> 右相机
         |                |
    raw2polar()       raw2polar()
         |                |
  S0  DoP  sin  cos  S0  DoP  sin  cos
       通道                  通道
  （时序 LK 跟踪）     （立体 LK 匹配）
         |                |
  发送 camera_id=0   发送 camera_id=1
```

每个偏振通道：左目做 prev→cur 时序 LK 跟踪，然后 cur_left→cur_right 立体 LK 匹配。新特征仅在左目检测，再立体匹配到右目。

---

## 实施步骤

### 步骤 1：在 ChannelState 中添加右目字段

**文件：** `polarfp_vins_estimator/src/featureTracker/feature_tracker.h`

在第 82 行（`brief_bytes = 32;`）之后、`ChannelState() = default;` 之前添加：

```cpp
// --- 右目（立体）字段 ---
cv::Mat cur_right_img;                       ///< 当前帧右目通道图像
vector<cv::Point2f> cur_right_pts;           ///< 右目特征点（从左目立体匹配而来）
vector<cv::Point2f> cur_un_right_pts;        ///< 右目去畸变点
vector<int> ids_right;                       ///< 右目特征 ID（与匹配的左目 ID 相同）
vector<cv::Point2f> right_pts_velocity;      ///< 右目光流速度
map<int, cv::Point2f> cur_un_right_pts_map;  ///< 右目：ID → 归一化坐标映射
```

不需要 `prev_right_pts`——立体匹配总是从当前左目特征重新开始，右目侧没有时序持续性。

### 步骤 2：在 FeatureMatcher 接口中添加 stereoMatch()

**文件：** `polarfp_vins_estimator/src/featureTracker/feature_tracker_matcher.h`

在 `FeatureMatcher` 类中添加纯虚方法：

```cpp
virtual std::pair<std::vector<cv::Point2f>, std::vector<uchar>> stereoMatch(
    const cv::Mat& left_img, const cv::Mat& right_img,
    const std::vector<cv::Point2f>& left_pts,
    const std::vector<int>& left_ids);
```

**文件：** `polarfp_vins_estimator/src/featureTracker/feature_tracker_matcher.cpp`

在 `LKFlowMatcher` 中实现——双向 LK 光流（与 VINS-Fusion 原版一致）：
- 前向：`calcOpticalFlowPyrLK(left_img, right_img, left_pts, ...)`
- 反向：`calcOpticalFlowPyrLK(right_img, left_img, right_pts, ...)` 做一致性校验
- 返回全尺寸 `right_pts` + `status` 向量（调用方用 `reduceVector` 压缩）

双目模式下**仅支持 LK 光流匹配**。BRIEF+FLANN 不支持立体匹配。

### 步骤 2b：双目 + 非 LK 配置的启动检查

**文件：** `polarfp_vins_estimator/src/featureTracker/feature_tracker.cpp` — `initDetectorAndMatcher()`

在检测器和匹配器创建完成后，添加校验：

```cpp
// 双目模式下仅支持 LK 光流匹配
if (stereo_cam && FEATURE_MATCHER_TYPE != 0) {
    ROS_ERROR("[PolarFP] Stereo mode only supports LK flow matcher (feature_matcher_type=0). "
              "Current config uses type %d. Please set feature_matcher_type=0 or num_of_cam=1.",
              FEATURE_MATCHER_TYPE);
    ros::shutdown();
    exit(EXIT_FAILURE);
}
```

因此 `BRIEFFLANNMatcher` **无需实现** `stereoMatch()`。

### 步骤 3：重写 trackImage() 支持双目偏振

**文件：** `polarfp_vins_estimator/src/featureTracker/feature_tracker.cpp`（第 337-529 行）

函数签名改动：
- `const cv::Mat &/*_img1*/` → `const cv::Mat &_img1`（去掉注释）

每个通道的处理流程：

```
[1] 偏振解码
    left_polar  = raw2polar(_img,  polar_filter_cfg)
    right_polar = raw2polar(_img1, polar_filter_cfg)   // stereo_cam && !_img1.empty()
    每个 ch: ch.cur_img  = getChannelImage(left_polar,  ch.name)
    每个 ch: ch.cur_right_img = getChannelImage(right_polar, ch.name)

[2a] 时序跟踪：matcher_->track(ch.prev_img, ch.cur_img, ...)

[2b] 左目边界检查

[2c] track_cnt++

[2d] setMaskForChannel(ch)

[2e] 立体匹配：左 → 右
    if (stereo_cam && !ch.cur_right_img.empty() && !ch.cur_pts.empty()):
        auto [right_pts, status] = matcher_->stereoMatch(ch.cur_img, ch.cur_right_img, ch.cur_pts, ch.ids)
        reduceVector(ch.cur_pts, status)         // 左目压缩为匹配子集
        reduceVector(ch.ids, status)
        reduceVector(ch.track_cnt, status)
        reduceVector(ch.cur_un_pts, status)
        reduceVector(ch.pts_velocity, status)
        ch.cur_right_pts = right_pts（按 status 过滤）
        ch.ids_right = ch.ids
        // 与原版 VINS-Fusion 一致：立体匹配失败的特征从当前帧丢弃，
        // 确保保留的每个特征都有左右观测

[2f] 左目新特征检测（不变——GFTT/FAST/SuperPoint）

[2g] 新特征立体匹配
    对每个新特征（track_cnt == 1）：
        LK 左→右 + 双向校验
        成功则 push 到 ch.cur_right_pts, ch.ids_right

[2h] 去畸变：ch.cur_un_pts = undistortedPts(ch.cur_pts, m_camera[0])
     右目去畸变：ch.cur_un_right_pts = undistortedPts(ch.cur_right_pts, m_camera[1])
     左目速度：ch.pts_velocity = ptsVelocity(...)
     右目速度：ch.right_pts_velocity = ptsVelocity(ids_right, ...)

[3] 构建 featureFrame
    每个 ch，每个特征：
        featureFrame[id].emplace_back(0, left_obs)   // 总是
        if (有右目观测):
            featureFrame[id].emplace_back(1, right_obs)  // 双目
```

**关键输出格式**：后端期望 `featureFrame[feature_id]` 按顺序包含 `[(camera_id=0, left_obs), (camera_id=1, right_obs)]`。当 `id_pts.second.size() == 2` 时，`addFeatureCheckParallax()` 会调用 `rightObservation()`。

### 步骤 4：更新 removeOutliers()

**文件：** `polarfp_vins_estimator/src/featureTracker/feature_tracker.cpp`

在 polar 模式的 `removeOutliers()` 分支中，同步压缩右目数据：

```cpp
reduceVector(ch.ids_right, status);
```

### 步骤 5：更新可视化

**文件：** `polarfp_vins_estimator/src/featureTracker/feature_tracker.cpp` — `drawTrackPolar()`

双目模式下，在左目特征旁绘制右目特征（绿色圆点），可选择绘制左右匹配连线。

### 步骤 6：后端无需改动

零改动。后端已完整就绪。

---

## 配置

无需新增 YAML 参数。使用现有双目配置：

```yaml
num_of_cam: 2
cam0_calib: "cam0_polar.yaml"
cam1_calib: "cam1_polar.yaml"
body_T_cam0: !!opencv-matrix ...   # IMU→左目外参
body_T_cam1: !!opencv-matrix ...   # IMU→右目外参
use_polar: 1
polar_channels: "s0,dop,aopsin,aopcos"
```

---

## 边界情况处理

| 场景 | 处理方式 |
|------|---------|
| 左右图像尺寸不同 | LK 光流使用实际通道图像；边界检查使用左目通道尺寸 |
| 左右相机畸变模型不同 | `m_camera[0]` 左目去畸变，`m_camera[1]` 右目去畸变 |
| 部分特征立体匹配失败 | 从当前帧丢弃（与 VINS-Fusion 一致） |
| 右目某通道图像为空 | 跳过该通道立体匹配；特征仅有左目观测 |
| 首帧（无前帧数据） | 所有特征新检测；立体匹配在新特征上运行 |
| SuperPoint 批量检测 | 不变——仅在左目通道上检测；立体匹配在检测后进行 |

---

## 需修改文件清单

| 文件 | 改动内容 |
|------|---------|
| `polarfp_vins_estimator/src/featureTracker/feature_tracker.h` | `ChannelState` 添加右目字段 |
| `polarfp_vins_estimator/src/featureTracker/feature_tracker.cpp` | 重写 `trackImage()`、更新 `removeOutliers()`、更新 `drawTrackPolar()` |
| `polarfp_vins_estimator/src/featureTracker/feature_tracker_matcher.h` | 添加 `stereoMatch()` 虚方法 |
| `polarfp_vins_estimator/src/featureTracker/feature_tracker_matcher.cpp` | 实现 `LKFlowMatcher::stereoMatch()` |

## 验证方式

1. `catkin_make` 编译通过
2. 使用双目偏振数据集（两路同步偏振相机）运行
3. 检查 `featureFrame` 中是否同时包含 `camera_id=0` 和 `camera_id=1` 的观测
4. 确认 `FeatureManager` 对特征设置 `is_stereo = true`
5. 检查深度初始化使用立体三角化（单帧，而非运动视差）
6. 监控优化过程：三种因子类型都应有残差加入
