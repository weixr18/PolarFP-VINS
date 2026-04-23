# PolarFP-VINS V2 前端重构计划

## 0. 重构背景与目标

### 0.1 项目简介

PolarFP 是一个科研项目,在 VINS-Fusion 基础上修改前端,支持**多偏振通道特征点检测+联合使用**,从而提升暗光环境下 VSLAM 效果。VINS-Fusion 是对比的 baseline。

相关分支:
- `org-vins`: 原版 VINS-Fusion(baseline)
- `polarfp-v1.1`: 上一版 PolarFP 方案(完全重写前端,效果不佳)
- `feat/superpoint`: 支持 SuperPoint 特征点的分支
- `master`: 当前工作分支,已从 org-vins 重置,基于原版 VINS-Fusion

### 0.2 为什么要重构(核心问题)

**上一版方案(polarfp-v1.1)效果不如原版 VINS-Fusion。**

在同一份数据上,VINS-Fusion 在纯光强通道上能做到的事情(成功初始化、成功跟踪),polarfp-v1.1 的前端反而做不到。

详细对比报告见项目中的 `compare_polarfpv1_1_vins.md`。

**根因分析**: v1.1 完全重写了特征跟踪器(`PolarFeatureTracker`),引入了大量偏离 VINS 原版的设计:
- 自定义 `PolarKeyPoint` 结构体替代 VINS 的平行 vector
- KD-tree 最近邻匹配替代纯 LK 光流
- 评分排序系统替代 track_cnt 排序
- 自定义 NMS 替代 setMask 掩码机制
- 多通道联合 RANSAC 替代后端 RANSAC

这些改动偏离了 VINS 经过验证的成熟管线,导致可靠性下降。**偏离越远,效果越差**。

### 0.3 本次重构方案

**最小化改动策略**: 尽可能沿用 VINS-Fusion 的全套特征跟踪管线(FeatureTracker),仅增加一处改动:

> **从多个偏振通道同时解析特征点,每通道独立运行 VINS 管线,合并结果输出给后端。**

具体来说:
- 特征点解析方法与 VINS-Fusion 保持一致(GFTT + LK 光流 + 双向检查 + setMask)
- 多 channel 分别独立进行光流匹配
- 多 channel 的特征点同时输出给后端
- **后端无需知道前端特征点来自什么通道**,它们都是同一帧原始图像出来的

### 0.4 用户原始指示

> "polarfp_v2_reconstruction_plan.md作为你的工作记录本。你来做重构计划，把计划写到这里。你在执行重构的时候，也要按照这个文件来列checklist。"

### 0.5 重构约束

1. 不改动后端(estimator/factor/initial 等任何文件)
2. 不改 ROS 节点入口
3. 不改相机标定参数
4. 非偏振模式下,功能与原版 VINS-Fusion 完全一致(可切换)
5. 参考 polarfp-v1.1 中的可复用组件(PolarChannel 模块),但不沿用其前端架构

---

## 1. 当前代码状态(master 分支)

### 1.1 可编译基线
- 已从 org-vins(原版 VINS-Fusion)重置
- V0 commit: `663e403`
- 编译通过,功能等同于原版 VINS-Fusion
- 仅修改了项目名称(`polarfp_vins`)和 catkin 依赖(`polarfp_camera_models`)
- **Important!!!** Compile command: `cd ~/ws/vi_catkin_ws && catkin_make`


### 1.2 关键文件

| 文件 | 当前状态 | 变更计划 |
|------|----------|----------|
| `polarfp_vins_estimator/src/featureTracker/feature_tracker.h` | 原版 VINS `FeatureTracker` | **大幅修改** — 增加多通道状态 |
| `polarfp_vins_estimator/src/featureTracker/feature_tracker.cpp` | 原版 VINS `FeatureTracker` 实现 | **大幅修改** — 核心逻辑改为多通道 |
| `polarfp_vins_estimator/src/estimator/estimator.h` | 引用 `feature_tracker.h`,成员 `FeatureTracker featureTracker` | **小改** — 增加 `setPolarChannels()` 方法声明 |
| `polarfp_vins_estimator/src/estimator/estimator.cpp` | 调用 `featureTracker.trackImage()` | **小改** — 在 `setParameter()` 中调用通道配置 |
| `polarfp_vins_estimator/src/estimator/parameters.h` | 原版参数声明 | **小改** — 增加极化相关 extern 变量 |
| `polarfp_vins_estimator/src/estimator/parameters.cpp` | 原版参数读取 | **小改** — 增加极化相关参数读取 |
| `polarfp_vins_estimator/src/rosNodeTest.cpp` | ROS 节点入口 | **基本不变** |
| `polarfp_vins_estimator/CMakeLists.txt` | 编译配置 | **小改** — 增加 `PolarChannel.cpp` |
| `polarfp_vins_estimator/src/featureTracker/PolarChannel.h` | **不存在于当前分支** | **新增**(从 polarfp-v1.1 引入) |
| `polarfp_vins_estimator/src/featureTracker/PolarChannel.cpp` | **不存在于当前分支** | **新增**(从 polarfp-v1.1 引入) |

---

## 2. 与 polarfp-v1.1 分支的差异对比

### 2.1 通道管理策略

| 维度 | v1.1 做法 | v2 建议做法 | 状态 |
|------|-----------|-------------|------|
| 数据表示 | `PolarKeyPoint` 自定义结构体(channel + cv::KeyPoint + id + score + track_cnt) | **沿用 VINS 的平行 vector**(pts/ids/track_cnt),但改为 per-channel | **建议沿用 VINS,需确认** |
| 通道配置 | `PolarConfig` 结构体,`FP_CHANNELS = {"s0","dop"}` | YAML 配置 `polar_channels: [s0, dop]` | **需确认通道列表** |
| 特征检测 | 每通道不同 FAST 阈值(S0=5, DoP=15, AoP=80) | **全部用 GFTT**,与 VINS 一致 | 用户已确认 |

### 2.2 匹配策略

| 维度 | v1.1 做法 | v2 建议做法 | 状态 |
|------|-----------|-------------|------|
| 帧间匹配 | LK光流 + KD-tree 最近邻 | **仅 LK 光流**(与 VINS 一致) | 用户已确认 |
| 外点剔除 | 多通道联合 RANSAC | **沿用 VINS**(setMask 空间筛选,后端 RANSAC) | 建议 |

### 2.3 特征筛选

| 维度 | v1.1 做法 | v2 建议做法 | 状态 |
|------|-----------|-------------|------|
| 空间分布 | NMS(21px半径) + 评分排序 | **沿用 VINS setMask** (圆形掩码,track_cnt 降序) | 建议 |
| 数量控制 | `KP_NUM_TARGET = 1000`,评分排序截断 | **沿用 VINS**(每通道独立 MAX_CNT) | **需确认** |

---

### 2.4 用户决策汇总(2026-04-11 确认)

| 编号 | 问题 | 决策 | 备注 |
|------|------|------|------|
| A1 | 特征数量分配 | **每通道独立 MAX_CNT** | S0 通道最多 MAX_CNT 个, DoP 通道也最多 MAX_CNT 个 |
| A2 | 启用通道 | **`[s0, dop]`**, YAML 可配 | 后续可添加 `aopsin`, `aopcos` |
| A3 | ID 管理 | **全局共享 `n_id`** | 每个通道之间不能有两个特征点用同一个 id |
| A4 | 分辨率处理 | **直接使用通道分辨率,不 upscale** | 相机标定已经是 612x512,与通道分辨率一致,坐标系统天然匹配 |
| A5 | setMask | **每通道独立** | 后续可能改为跨通道联合,重构阶段保持独立 |
| A6 | 通道间关联 | **不关联** | 后端自行处理同一物理点多通道检测 |

### 2.5 分辨率关键发现

相机标定文件 `config/dark/TRI050S-spin4.yaml` 中:
```yaml
image_width: 612
image_height: 512
fx: 625.87976
fy: 621.14111
cx: 313.91154
cy: 250.14958
```

这恰好与 `raw2polar()` 输出的通道分辨率一致(raw 1024x1224 → 通道 512x612,即 height=512, width=612)。
**标定是在解码后的通道图像上做的**,坐标系统天然匹配。因此:
- `liftProjective` 可以直接使用通道像素坐标
- 输出给后端的像素坐标无需缩放
- `ROW`(512) 和 `COL`(612) 已在 config 中正确设置
- 整个管线中**没有任何地方需要知道原始 1024x1224 分辨率**

---

## 4. 详细修改计划

### Step 1: 引入 PolarChannel 模块

**文件操作**:
- 从 `polarfp-v1.1` 分支复制 `PolarChannel.h` 和 `PolarChannel.cpp` 到 `polarfp_vins_estimator/src/featureTracker/`
- 不需要 `nanoflann.hpp`(不用 KD-tree)
- 不需要 `polarfp_tracker.h/cpp`(我们修改现有 `feature_tracker`)

**`polarfp_vins_estimator/CMakeLists.txt` 修改**:
```cmake
# 在 vins_lib 的源文件列表中, feature_tracker.cpp 后面增加:
src/featureTracker/PolarChannel.cpp
```

**编译验证**: 确保 `PolarChannel` 能正常编译(它只依赖 OpenCV)。

---

### Step 2: 增加Polar相关配置参数

**`parameters.h` 修改** — 增加:
```cpp
extern int USE_POLAR;     // 是否启用偏振模式(0/1)
extern std::vector<std::string> POLAR_CHANNELS;   // 启用的偏振通道列表
```

**`parameters.cpp` 修改** — 增加变量定义:
```cpp
int USE_POLAR = 0;
std::vector<std::string> POLAR_CHANNELS;
```

**`parameters.cpp` — `readParameters()` 中增加**:
```cpp
// 在现有参数读取之后, fsSettings.release() 之前:
if (fsSettings["use_polar"].isNoString())
    USE_POLAR = 0;
else
    USE_POLAR = (int)fsSettings["use_polar"];

if (USE_POLAR) {
    std::string channels_str;
    fsSettings["polar_channels"] >> channels_str;
    // 解析逗号分隔字符串,如 "s0,dop" → vector<string>{"s0","dop"}
    std::stringstream ss(channels_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // 去除首尾空格
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        POLAR_CHANNELS.push_back(item);
    }
    if (POLAR_CHANNELS.empty()) {
        POLAR_CHANNELS = {"s0", "dop"};  // 默认值
        ROS_WARN("polar_channels not specified, using default: s0,dop");
    }
    ROS_INFO("Polar mode enabled, channels: %zu", POLAR_CHANNELS.size());
    for (const auto& ch : POLAR_CHANNELS)
        ROS_INFO("  channel: %s", ch.c_str());
}
```

**需要 `<sstream>` include** (如果 parameters.cpp 中没有的话)。

**YAML 配置示例** — 在现有 config 文件中增加两行:
```yaml
use_polar: 1
polar_channels: "s0,dop"
```

---

### Step 3: 修改 FeatureTracker 头文件

**`feature_tracker.h` 修改清单**:

#### 3a. 增加 include
```cpp
#include "PolarChannel.h"
```

#### 3b. 增加 ChannelState 结构体(在 FeatureTracker 类定义之前或之内):
```cpp
/**
 * @brief 单个偏振通道的跟踪状态
 *
 * 复用 VINS-Fusion FeatureTracker 的数据组织方式(平行 vector),
 * 每个通道独立维护自己的跟踪状态。
 */
struct ChannelState {
    std::string name;                           // 通道名称 "s0"/"dop"/...
    int row = 0, col = 0;                       // 通道图像尺寸
    cv::Mat prev_img, cur_img;                  // 前后帧图像
    vector<cv::Point2f> prev_pts, cur_pts;      // 前后帧特征点坐标
    vector<cv::Point2f> prev_un_pts, cur_un_pts; // undistorted 特征点坐标
    vector<cv::Point2f> n_pts;                  // 新检测特征点
    vector<int> ids;                            // 特征 ID(全局共享 n_id 分配)
    vector<int> track_cnt;                      // 跟踪帧数
    vector<cv::Point2f> pts_velocity;           // 像素速度
    map<int, cv::Point2f> cur_un_pts_map;       // ID → 归一化坐标
    map<int, cv::Point2f> prev_un_pts_map;      // 上一帧 ID → 归一化坐标
    map<int, cv::Point2f> prevLeftPtsMap;       // 上一帧 ID → 像素坐标
    cv::Mat mask;                               // 空间分布掩码

    ChannelState() = default;
    explicit ChannelState(const string& n) : name(n) {}
};
```

#### 3c. FeatureTracker 类中增加成员变量:
```cpp
    // ---- 偏振模式特有成员 ----
    vector<ChannelState> channels;      // 启用的通道状态列表
    bool polar_mode = false;            // 是否处于偏振模式
```

#### 3d. FeatureTracker 类中增加公共方法:
```cpp
    /**
     * @brief 设置启用的偏振通道
     * @param channel_names 通道名称列表,如 {"s0", "dop"}
     *
     * 在 estimator::setParameter() 中调用。
     * 调用后 trackImage 将输入视为原始偏振图像,先解码为多通道再分别处理。
     */
    void setPolarChannels(const vector<string>& channel_names);

    /** @brief 判断是否处于偏振模式 */
    bool isPolarMode() const { return polar_mode; }
```

#### 3e. FeatureTracker 类中增加私有方法:
```cpp
    /**
     * @brief 偏振模式跟踪管线
     * @return 标准 VINS 格式特征帧
     */
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImagePolar(double _cur_time);

    /**
     * @brief 从 PolarChannelResult 提取指定通道的 8bit 图像
     */
    static cv::Mat getChannelImage(const PolarChannelResult& result, const string& channel);
```

#### 3f. 保留原版成员变量不变:
现有的 `row, col, mask, prev_img, cur_img, prev_pts, cur_pts, ids, track_cnt` 等在**非偏振模式**下仍然使用(用于兼容原始灰度图输入)。偏振模式使用 `channels` vector 中的 per-channel 状态。

---

### Step 4: 修改 FeatureTracker 实现

**`feature_tracker.cpp` 修改清单**:

#### 4a. trackImage 改为分支入口
```cpp
map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
FeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1)
{
    cur_time = _cur_time;
    cur_img = _img;

    if (isPolarMode())
        return trackImagePolar(_cur_time);

    // ====== 以下为原版 VINS 管线,完全不变 ======
    // ... 现有的全部代码,从 cur_pts.clear() 到最后 return featureFrame ...
}
```

**注意**: 原 `trackImage` 中的代码整体后移到一个 `else` 分支或直接在 `if (!isPolarMode())` 之后继续执行。`cur_time` 和 `cur_img` 的赋值保留在分支之前(两个模式共用)。

#### 4b. 新增 setPolarChannels 方法:
```cpp
void FeatureTracker::setPolarChannels(const vector<string>& channel_names)
{
    channels.clear();
    for (const auto& name : channel_names) {
        channels.emplace_back(name);
    }
    polar_mode = true;
    ROS_INFO("[PolarFP] setPolarChannels: %zu channels", channels.size());
}
```

#### 4c. 新增 getChannelImage 辅助函数(静态):
```cpp
cv::Mat FeatureTracker::getChannelImage(const PolarChannelResult& result, const string& channel)
{
    if (channel == "s0")      return result.S0_img.clone();
    if (channel == "dop")     return result.dop_img.clone();
    if (channel == "aopsin")  return result.sin_img.clone();
    if (channel == "aopcos")  return result.cos_img.clone();
    ROS_WARN("[PolarFP] unknown channel: %s", channel.c_str());
    return cv::Mat();
}
```

#### 4d. 新增 trackImagePolar 核心函数:

这是整个重构的核心。每通道完整复用 VINS 管线:

```cpp
map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
FeatureTracker::trackImagePolar(double _cur_time)
{
    // =============================================
    // 1. 偏振图像分解: raw → 4 通道
    // =============================================
    PolarChannelResult polar_result = raw2polar(cur_img);

    for (auto& ch : channels) {
        ch.cur_img = getChannelImage(polar_result, ch.name);
        if (ch.cur_img.empty()) {
            ROS_ERROR("[PolarFP] channel %s image is empty!", ch.name.c_str());
            continue;
        }
        ch.row = ch.cur_img.rows;
        ch.col = ch.cur_img.cols;
    }

    // =============================================
    // 2. 对每个通道,独立运行 VINS 跟踪管线
    // =============================================
    for (auto& ch : channels) {
        if (ch.cur_img.empty()) continue;

        // ---- 2a. LK 光流跟踪(prev → cur) ----
        if (!ch.prev_pts.empty() && !ch.prev_img.empty()) {
            vector<uchar> status;
            vector<float> err;

            // 有预测位置时使用 OPTFLOW_USE_INITIAL_FLOW
            if (hasPrediction) {
                // 需要从全局 predict_pts 中提取该通道的预测位置
                // 这里简化处理:直接用 prev_pts 作为预测(无外部预测加速)
                // TODO: 后续可加入 per-channel 预测
            }

            cv::calcOpticalFlowPyrLK(ch.prev_img, ch.cur_img, ch.prev_pts, ch.cur_pts,
                                     status, err, cv::Size(21, 21), 3);

            // ---- 2b. 反向光流检查 ----
            if (FLOW_BACK) {
                vector<uchar> reverse_status;
                vector<cv::Point2f> reverse_pts = ch.prev_pts;
                cv::calcOpticalFlowPyrLK(ch.cur_img, ch.prev_img, ch.cur_pts, reverse_pts,
                                         reverse_status, err, cv::Size(21, 21), 1,
                                         cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                                         cv::OPTFLOW_USE_INITIAL_FLOW);
                for (size_t i = 0; i < status.size(); i++) {
                    if (status[i] && reverse_status[i] &&
                        distance(ch.prev_pts[i], reverse_pts[i]) <= 0.5) {
                        status[i] = 1;
                    } else {
                        status[i] = 0;
                    }
                }
            }

            // ---- 2c. 边界检查 ----
            for (int i = 0; i < int(ch.cur_pts.size()); i++)
                if (status[i] && !inBorderImpl(ch, ch.cur_pts[i]))  // 注意用通道的 row/col
                    status[i] = 0;

            reduceVector(ch.prev_pts, status);
            reduceVector(ch.cur_pts, status);
            reduceVector(ch.ids, status);
            reduceVector(ch.track_cnt, status);
        }

        // ---- 2d. track_cnt++ ----
        for (auto &n : ch.track_cnt)
            n++;

        // ---- 2e. setMask:保证空间均匀分布 ----
        setMaskForChannel(ch);

        // ---- 2f. 新特征检测(GFTT) ----
        int n_max_cnt = MAX_CNT - static_cast<int>(ch.cur_pts.size());
        if (n_max_cnt > 0) {
            if (ch.mask.empty())
                cout << "mask is empty for channel " << ch.name << endl;
            if (ch.mask.type() != CV_8UC1)
                cout << "mask type wrong for channel " << ch.name << endl;
            cv::goodFeaturesToTrack(ch.cur_img, ch.n_pts, n_max_cnt, 0.01, MIN_DIST, ch.mask);
        } else {
            ch.n_pts.clear();
        }

        // ---- 2g. 新特征加入,分配全局唯一 ID ----
        for (auto &p : ch.n_pts) {
            ch.cur_pts.push_back(p);
            ch.ids.push_back(n_id++);   // 全局计数器,保证跨通道 ID 唯一
            ch.track_cnt.push_back(1);
        }

        // ---- 2h. 归一化坐标(相机标定已匹配通道分辨率,直接使用) ----
        ch.cur_un_pts = undistortedPts(ch.cur_pts, m_camera[0]);
        ch.pts_velocity = ptsVelocityForChannel(ch);

        // ---- 2i. 更新 prev 状态 ----
        ch.prev_img = ch.cur_img.clone();
        ch.prev_pts = ch.cur_pts;
        ch.prev_un_pts = ch.cur_un_pts;
        ch.prev_un_pts_map = ch.cur_un_pts_map;
        ch.prev_time = ch.cur_time;
        ch.prevLeftPtsMap.clear();
        for (size_t i = 0; i < ch.cur_pts.size(); i++)
            ch.prevLeftPtsMap[ch.ids[i]] = ch.cur_pts[i];
    }

    // =============================================
    // 3. 合并所有通道结果 → VINS 格式
    // =============================================
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (const auto& ch : channels) {
        if (ch.cur_img.empty()) continue;
        for (size_t i = 0; i < ch.ids.size(); i++) {
            int feature_id = ch.ids[i];
            double x = ch.cur_un_pts[i].x;
            double y = ch.cur_un_pts[i].y;
            double z = 1;
            double p_u = ch.cur_pts[i].x;
            double p_v = ch.cur_pts[i].y;
            double velocity_x = ch.pts_velocity[i].x;
            double velocity_y = ch.pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(0, xyz_uv_velocity);
        }
    }

    // =============================================
    // 4. 可视化
    // =============================================
    if (SHOW_TRACK)
        drawTrackPolar();

    // =============================================
    // 5. 更新全局状态(用于后端交互)
    // =============================================
    hasPrediction = false;

    // 将第一个通道的图像作为全局 cur_img 用于可视化
    if (!channels.empty() && !channels[0].cur_img.empty())
        cur_img = channels[0].cur_img;

    return featureFrame;
}
```

#### 4e. 新增 per-channel 辅助函数:

**`setMaskForChannel(ChannelState& ch)`** — 复用原版 setMask 逻辑:
```cpp
void FeatureTracker::setMaskForChannel(ChannelState& ch)
{
    ch.mask = cv::Mat(ch.row, ch.col, CV_8UC1, cv::Scalar(255));

    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
    for (unsigned int i = 0; i < ch.cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(ch.track_cnt[i], make_pair(ch.cur_pts[i], ch.ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(),
         [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b) {
             return a.first > b.first;
         });

    ch.cur_pts.clear();
    ch.ids.clear();
    ch.track_cnt.clear();

    for (auto &it : cnt_pts_id) {
        if (ch.mask.at<uchar>(it.second.first) == 255) {
            ch.cur_pts.push_back(it.second.first);
            ch.ids.push_back(it.second.second);
            ch.track_cnt.push_back(it.first);
            cv::circle(ch.mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}
```

**`inBorderImpl(const ChannelState& ch, const cv::Point2f& pt)`** — 边界检查的通道版本:
```cpp
static bool inBorderImpl(const ChannelState& ch, const cv::Point2f& pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < ch.col - BORDER_SIZE &&
           BORDER_SIZE <= img_y && img_y < ch.row - BORDER_SIZE;
}
```

**`ptsVelocityForChannel(ChannelState& ch)`** — 速度计算的通道版本:
```cpp
vector<cv::Point2f> FeatureTracker::ptsVelocityForChannel(ChannelState& ch)
{
    vector<cv::Point2f> pts_velocity;
    ch.cur_un_pts_map.clear();
    for (unsigned int i = 0; i < ch.ids.size(); i++)
        ch.cur_un_pts_map.insert(make_pair(ch.ids[i], ch.cur_un_pts[i]));

    if (!ch.prev_un_pts_map.empty()) {
        double dt = ch.cur_time - ch.prev_time;
        for (unsigned int i = 0; i < ch.ids.size(); i++) {
            auto it = ch.prev_un_pts_map.find(ch.ids[i]);
            if (it != ch.prev_un_pts_map.end()) {
                double vx = (ch.cur_un_pts[i].x - it->second.x) / dt;
                double vy = (ch.cur_un_pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(vx, vy));
            } else {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    } else {
        for (unsigned int i = 0; i < ch.cur_pts.size(); i++)
            pts_velocity.push_back(cv::Point2f(0, 0));
    }
    return pts_velocity;
}
```

**`drawTrackPolar()`** — 多通道可视化:
```cpp
void FeatureTracker::drawTrackPolar()
{
    if (channels.empty()) return;

    static const vector<cv::Scalar> channelColors = {
        cv::Scalar(0, 255, 255),  // yellow - s0
        cv::Scalar(0, 255, 0),    // green  - dop
        cv::Scalar(0, 0, 255),    // red    - aopsin
        cv::Scalar(255, 0, 0)     // blue   - aopcos
    };

    if (channels.size() == 1) {
        // 单通道:直接在该图像上绘制
        const auto& ch = channels[0];
        if (ch.cur_img.empty()) return;
        cv::cvtColor(ch.cur_img, imTrack, cv::COLOR_GRAY2RGB);
        for (size_t j = 0; j < ch.cur_pts.size(); j++) {
            double len = std::min(1.0, 1.0 * ch.track_cnt[j] / 20);
            cv::circle(imTrack, ch.cur_pts[j], 2,
                       cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
        }
    } else {
        // 多通道:2x2 网格拼接
        int h = channels[0].row;
        int w = channels[0].col;
        imTrack = cv::Mat(h * 2, w * 2, CV_8UC3, cv::Scalar(0));

        for (size_t c = 0; c < channels.size() && c < 4; c++) {
            const auto& ch = channels[c];
            if (ch.cur_img.empty()) continue;

            int gridRow = c / 2;
            int gridCol = c % 2;
            cv::Mat chBGR;
            cv::cvtColor(ch.cur_img, chBGR, cv::COLOR_GRAY2BGR);
            cv::Mat roi = imTrack(cv::Rect(gridCol * w, gridRow * h, w, h));
            chBGR.copyTo(roi);

            // 绘制特征点
            cv::Scalar color = channelColors[c % channelColors.size()];
            for (size_t j = 0; j < ch.cur_pts.size(); j++) {
                cv::Point2f pt(ch.cur_pts[j].x + gridCol * w, ch.cur_pts[j].y + gridRow * h);
                cv::circle(imTrack, pt, 2, color, 2);
            }
            // 绘制轨迹箭头
            for (size_t j = 0; j < ch.cur_pts.size(); j++) {
                auto it = ch.prevLeftPtsMap.find(ch.ids[j]);
                if (it != ch.prevLeftPtsMap.end()) {
                    cv::Point2f from(ch.cur_pts[j].x + gridCol * w, ch.cur_pts[j].y + gridRow * h);
                    cv::Point2f to(it->second.x + gridCol * w, it->second.y + gridRow * h);
                    cv::arrowedLine(imTrack, from, to, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
                }
            }
        }
    }
}
```

---

### Step 5: 修改 removeOutliers 支持偏振模式

原版 `removeOutliers` 操作的是全局 `prev_pts`/`ids`/`track_cnt`。偏振模式下需要操作每个 channel 的状态:

```cpp
void FeatureTracker::removeOutliers(set<int> &removePtsIds)
{
    if (isPolarMode()) {
        for (auto& ch : channels) {
            vector<uchar> status;
            for (size_t i = 0; i < ch.ids.size(); i++) {
                if (removePtsIds.count(ch.ids[i]))
                    status.push_back(0);
                else
                    status.push_back(1);
            }
            reduceVector(ch.prev_pts, status);
            reduceVector(ch.ids, status);
            reduceVector(ch.track_cnt, status);
        }
    } else {
        // 原版逻辑
        std::set<int>::iterator itSet;
        vector<uchar> status;
        for (size_t i = 0; i < ids.size(); i++) {
            itSet = removePtsIds.find(ids[i]);
            if (itSet != removePtsIds.end())
                status.push_back(0);
            else
                status.push_back(1);
        }
        reduceVector(prev_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
    }
}
```

---

### Step 6: 修改 setPrediction 支持偏振模式

原版 `setPrediction` 将预测 3D 点投影到像素平面存入 `predict_pts`。偏振模式下,每个通道有自己的 prev_pts,需要分别处理:

```cpp
void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
    if (isPolarMode()) {
        hasPrediction = true;
        // 偏振模式下,预测暂时存储为全局,后续可改为 per-channel
        // 当前简化:设置标志但不做具体投影,光流不使用预测
        // TODO: per-channel prediction when needed
    } else {
        // 原版逻辑
        hasPrediction = true;
        predict_pts.clear();
        predict_pts_debug.clear();
        for (size_t i = 0; i < ids.size(); i++) {
            int id = ids[i];
            auto itPredict = predictPts.find(id);
            if (itPredict != predictPts.end()) {
                Eigen::Vector2d tmp_uv;
                m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
                predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
                predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
            } else {
                predict_pts.push_back(prev_pts[i]);
            }
        }
    }
}
```

---

### Step 7: 修改 estimator 集成

**`estimator.cpp` — `setParameter()` 中增加**:
```cpp
void Estimator::setParameter()
{
    // ... 原有代码不变 ...
    featureTracker.readIntrinsicParameter(CAM_NAMES);

    // 新增:配置偏振通道
    if (USE_POLAR) {
        featureTracker.setPolarChannels(POLAR_CHANNELS);
    }
    // ... 原有代码不变 ...
}
```

`estimator.h` 不需要修改 — `FeatureTracker featureTracker` 成员类型不变。

---

### Step 8: 更新 ROS 节点的图像编码处理

由于输入图像可能是原始的 1024x1224 Bayer-like 图像,需要确保 `rosNodeTest.cpp` 能正确处理。但检查发现 `getImageFromMsg` 已经是通用的 mono8 转换,不需要修改。

唯一需要注意的是:当前 `inputImage` 接收的图像是原始分辨率(1024x1224),`raw2polar` 会在 `trackImagePolar` 内部解码为通道分辨率(512x612)。这不需要任何修改,`raw2polar` 已经处理了分辨率减半。

**确认**: `rosNodeTest.cpp` 不需要修改。

---

### Step 9: 可选 — 保留 stereo 双目兼容性

虽然偏振模式下双目支持可能不适用,但非偏振模式下双目仍然可以工作。现有的 `trackImage(double, const cv::Mat&, const cv::Mat&)` 签名和双目逻辑保持不变,偏振模式下 `_img1` 参数被忽略即可。

---

## 5. 文件变更清单

### 5.1 新增文件(从 polarfp-v1.1 分支复制)
- `polarfp_vins_estimator/src/featureTracker/PolarChannel.h`
- `polarfp_vins_estimator/src/featureTracker/PolarChannel.cpp`

### 5.2 修改文件
| 文件 | 变更类型 | 预估改动量 |
|------|----------|-----------|
| `polarfp_vins_estimator/src/featureTracker/feature_tracker.h` | 增加 `ChannelState` 结构体 + `setPolarChannels`/`trackImagePolar`/`getChannelImage`/`drawTrackPolar`/`setMaskForChannel`/`ptsVelocityForChannel` 方法声明 + `channels`/`polar_mode` 成员 | ~80 行增加 |
| `polarfp_vins_estimator/src/featureTracker/feature_tracker.cpp` | `trackImage` 改为分支入口; 新增 `trackImagePolar`/`setPolarChannels`/`getChannelImage`/`drawTrackPolar`/`setMaskForChannel`/`ptsVelocityForChannel`; 修改 `removeOutliers`/`setPrediction` 支持偏振模式 | ~400 行增加, 原版 `trackImage` 代码后移 |
| `polarfp_vins_estimator/src/estimator/parameters.h` | 增加 `USE_POLAR`, `POLAR_CHANNELS` extern | ~2 行增加 |
| `polarfp_vins_estimator/src/estimator/parameters.cpp` | 增加变量定义 + YAML 参数读取 + 字符串解析 | ~20 行增加 |
| `polarfp_vins_estimator/src/estimator/estimator.cpp` | `setParameter()` 中调用 `featureTracker.setPolarChannels()` | ~4 行增加 |
| `polarfp_vins_estimator/CMakeLists.txt` | 增加 `PolarChannel.cpp` | ~1 行增加 |

### 5.3 不变文件
- `polarfp_vins_estimator/src/rosNodeTest.cpp` — 不需要修改
- `polarfp_vins_estimator/src/estimator/estimator.h` — 不需要修改(`FeatureTracker` 类型不变)
- `polarfp_vins_estimator/src/estimator/feature_manager.cpp` — 不需要修改
- 所有 factor 文件 — 不需要修改
- 所有 initial 文件 — 不需要修改
- 后端优化文件 — 不需要修改
- 所有配置文件(YAML) — 可选添加 `use_polar` 和 `polar_channels`,不添加则默认为非偏振模式

---

## 6. 执行步骤规划(下一个 session 执行)

1. **Step 1**: 从 polarfp-v1.1 复制 `PolarChannel.h/cpp`, 修改 `CMakeLists.txt`, 编译验证
2. **Step 2**: 修改 `parameters.h/cpp` 增加 `USE_POLAR`/`POLAR_CHANNELS`
3. **Step 3**: 修改 `feature_tracker.h` — 增加 `ChannelState`, 新成员, 新方法声明
4. **Step 4**: 修改 `feature_tracker.cpp` — 核心实现 `trackImagePolar` 及辅助函数
5. **Step 5**: 修改 `estimator.cpp` — `setParameter()` 中调用 `setPolarChannels`
6. **Step 6**: 编译验证, 确认非偏振模式(原版)仍能正常工作
7. **Step 7**: 在 config YAML 中添加 `use_polar: 1` 和 `polar_channels: "s0,dop"`, 测试偏振模式
