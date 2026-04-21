# PolarFP-VINS V2.5 全局特征池重构计划

## 1. Context & Motivation

当前 V2.4 中，所有偏振通道共享一个全局 ID 计数器 `FeatureTracker::n_id`。每个通道独立检测/跟踪特征，新特征统一从全局计数器取号。最终合并时，不同通道在同一像素位置检测到的特征会获得不同全局 ID，后端将它们视为完全独立的 3D 点，导致：
- **特征冗余**：同一物理角点被多个通道重复发布。
- **性能下降**：后端 BA 因子数量随通道数线性膨胀。
- **逻辑不一致**：从 VSLAM 语义上，同一空间角点不应因通道不同而分裂为多个路标。

本重构引入**前端全局特征池（Global Feature Pool）**，在不修改后端的前提下，将多通道局部特征通过空间位置去重，映射为统一、连续的全局 ID 序列发给后端。

## 2. Design Overview

### 2.1 核心概念

- **局部 ID（Local ID）**：每个偏振通道拥有自己独立的 ID 计数器。`s0:45` 表示 s0 通道的第 45 号局部特征，`dop:127` 表示 dop 通道的第 127 号。局部 ID 仅在本通道内唯一，由通道自己的 LK 光流/BRIEF 匹配保持帧间连续性。
- **全局 ID（Global ID）**：由前端全局特征池统一分配。后端只看到全局 ID。一个全局 ID 可以跨帧、跨通道绑定多个局部特征，但**同一帧中，一个全局 ID 最多向后端发送一个观测**。
- **全局特征池状态**：维护上一帧所有全局 ID 到各通道局部 ID 的绑定关系，以及对应的像素坐标。新一帧到来时，优先继承已有绑定；无法继承的局部特征再通过空间哈希匹配到已有全局特征或创建新全局 ID。

### 2.2 关键决策

1. **不修改后端**：`feature_manager`、`estimator`、Ceres 因子、滑动窗口逻辑完全不动。前端输出接口保持 `map<int, vector<pair<int, Matrix<double,7,1>>>>` 不变。
2. **每帧单观测**：若同一全局 ID 在当前帧同时绑定了 `s0:45`（像素 [45,72]）和 `dop:127`（像素 [46,71]），只保留一个观测发给后端。保留优先级由 YAML `polar_channels` 配置顺序决定（例如 `s0,dop,aopsin,aopcos` 则 s0 优先级最高）。
3. **空间哈希去重**：使用像素坐标构建网格哈希。网格大小由 YAML 参数 `polar_hash_grid_size` 控制（默认 **5 px**），查询时搜索 3×3 邻域，匹配阈值取 `MIN_DIST / 2`。小网格在暗弱光场景下能发现更多可合并特征。
4. **两阶段注册**：
   - **Propagate（继承）**：对于上一帧的每个全局 ID，检查其绑定的各通道局部 ID 是否在当前帧该通道中仍然被 LK 跟踪成功。若成功，直接继承绑定关系。
   - **Register（新注册）**：按通道优先级遍历各通道当前帧中**尚未绑定到任何全局 ID** 的局部特征（主要是本帧新检测的点，也可能包括上一帧未成功绑定的遗留点）。对每个未绑定局部特征，查询空间哈希网格。若找到附近已有全局 ID 且该全局 ID 在当前帧的**该通道**尚未被绑定，则 attach；否则创建新全局 ID。

### 2.3 帧间连续性示例

| 帧 | s0 局部点 | dop 局部点 | 全局 1 的绑定 | 后端看到的观测 |
|---|---|---|---|---|
| 1 | `s0:1` | — | `s0:1` | 全局 1（来自 s0） |
| 2 | `s0:1` | — | `s0:1` | 全局 1（来自 s0） |
| 3 | `s0:1` | `dop:2` | `s0:1`, `dop:2` | 全局 1（来自 s0，s0 优先级高） |
| 5 | `s0:1` | `dop:2` | `s0:1`, `dop:2` | 全局 1（来自 s0） |
| 6 | — | `dop:2` | `dop:2` | 全局 1（来自 dop） |
| 7 | `s0:3` | `dop:2` | `s0:3`, `dop:2` | 全局 1（来自 s0） |
| 8 | `s0:3` | `dop:2` | `s0:3`, `dop:2` | 全局 1（来自 s0） |
| 9 | `s0:3` | — | `s0:3` | 全局 1（来自 s0） |

后端在帧 1–9 始终只看到**同一个**全局 ID `1`，其轨迹从未中断，尽管前端内部先后绑定了三个不同的局部点。

## 3. Data Structures

### 3.1 `ChannelState` 改动

文件：`vins_estimator/src/featureTracker/feature_tracker.h`（约第 62–86 行）

在 `ChannelState` 中增加局部 ID 计数器，并修改 `ids` 字段语义：

```cpp
struct ChannelState {
    std::string name;                           ///< 通道名称 "s0"/"dop"/...
    int row = 0, col = 0;                       ///< 通道图像尺寸
    cv::Mat prev_img, cur_img;                  ///< 前后帧图像
    vector<cv::Point2f> prev_pts, cur_pts;      ///< 前后帧特征点像素坐标
    vector<cv::Point2f> prev_un_pts, cur_un_pts; ///< 前后帧归一化平面坐标
    vector<cv::Point2f> n_pts;                  ///< 本帧新检测到的特征点
    vector<int> local_ids;                      ///< 【语义变更】局部特征 ID（仅在当前通道内唯一）
    vector<int> track_cnt;                      ///< 特征点连续跟踪帧数
    vector<cv::Point2f> pts_velocity;           ///< 像素速度
    map<int, cv::Point2f> cur_un_pts_map;       ///< 当前帧 local_id -> 归一化坐标
    map<int, cv::Point2f> prev_un_pts_map;      ///< 上一帧 local_id -> 归一化坐标
    map<int, cv::Point2f> prevLeftPtsMap;       ///< 上一帧 local_id -> 像素坐标（可视化用）
    double cur_time = 0;                        ///< 当前帧时间戳
    double prev_time = 0;                       ///< 上一帧时间戳
    cv::Mat mask;                               ///< 空间分布掩码

    // BRIEF/ORB 描述子(上一帧), 扁平存储: [feat0_desc, feat1_desc, ...]
    std::vector<uchar> prev_brief_desc;
    int brief_bytes = 32;

    int next_local_id = 0;                      ///< 【新增】本通道局部 ID 计数器，新特征从此取值

    ChannelState() = default;
    explicit ChannelState(const string& n) : name(n) {}
};
```

说明：
- `local_ids` 字段原名 `ids`，物理类型不变（`vector<int>`），语义从“全局唯一”变为“通道局部唯一”。LK 匹配器返回的 `MatchResult.ids` 本来就是上一帧的 ID，在局部 ID 体系下天然兼容。
- `cur_un_pts_map`、`prev_un_pts_map`、`prevLeftPtsMap` 的 key 类型保持 `int`，但含义变为 local_id。
- `next_local_id` 在每个 `ChannelState` 构造时初始化为 0，新特征检测时自增。

### 3.2 `GlobalFeaturePool` 新增结构

在 `feature_tracker.h` 中新增（可放在 `FeatureTracker` 类内部作为 `private struct`）：

```cpp
struct GlobalFeaturePool {
    // 单帧内一个全局特征的状态
    struct GlobalFeature {
        int global_id;
        // 当前帧：通道名 -> local_id
        std::map<std::string, int> local_ids;
        // 当前帧：通道名 -> 像素坐标（用于空间哈希与 featureFrame 构建）
        std::map<std::string, cv::Point2f> pixel_pts;
    };

    int next_global_id = 0;

    // 上一帧的全局绑定关系（帧间连续性依赖此状态）
    std::map<int, std::map<std::string, int>> prev_bindings;   // global_id -> {ch_name -> local_id}
    std::map<int, std::map<std::string, cv::Point2f>> prev_pts; // global_id -> {ch_name -> pixel_pt}

    // 当前帧处理过程中逐步构建
    std::map<int, GlobalFeature> cur_globals;                  ///< 当前帧活跃的全局特征
    std::map<std::pair<int, int>, std::vector<int>> grid;      ///< 空间哈希 (cell_x, cell_y) -> [global_id]

    int grid_size_ = 5;   ///< 哈希网格边长（像素），由 YAML polar_hash_grid_size 注入，默认 5

    // ---------- 核心方法 ----------

    /** @brief 开始处理新一帧，清空当前帧状态，保留上一帧绑定关系 */
    void beginFrame();

    /**
     * @brief Step A: 继承跟踪
     * @details 遍历 prev_bindings，若某全局特征在上帧绑定的局部 ID
     *          仍存在于当前帧同通道的 ids 中，则直接继承绑定关系。
     *          继承成功的全局特征会被插入空间哈希网格。
     */
    void propagateTracked(const std::vector<ChannelState>& channels);

    /**
     * @brief Step B: 注册未绑定的局部特征
     * @param channels 所有偏振通道状态（顺序即优先级）
     * @param min_dist 特征点最小间距（来自 YAML），用于判断空间匹配阈值
     * @details 按通道优先级遍历各通道当前帧中尚未绑定到全局特征的局部点，
     *          查询 3×3 邻域空间哈希网格，若找到距离满足阈值且该通道未绑定的全局特征，则 attach；
     *          否则创建新全局 ID。
     */
    void registerUnboundFeatures(std::vector<ChannelState>& channels, int min_dist);

    /**
     * @brief 构建最终发给后端的 featureFrame
     * @return map<global_id, vector<(camera_id, 7D_obs)>>
     * @details 对每个全局特征，在当前帧绑定的通道中按优先级选一个观测，
     *          打包为 VINS 标准格式。未绑定任何通道的全局特征被跳过。
     */
    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>
    buildFeatureFrame(const std::vector<ChannelState>& channels) const;

    /**
     * @brief 获取全局 ID 到局部 ID 的反向映射
     * @return map<global_id, vector<(channel_name, local_id)>>
     * @details 供 FeatureTracker::removeOutliers 使用，将后端返回的全局 outlier ID
     *          映射为各通道的 local_id 以便在前端剔除。
     */
    std::map<int, std::vector<std::pair<std::string, int>>> getGlobalToLocalMap() const;

    /**
     * @brief 结束当前帧处理
     * @details 将 cur_globals 中至少有一个通道绑定的特征保存到 prev_bindings/prev_pts；
     *          完全丢失绑定的全局特征被丢弃，不再参与下一帧 propagate。
     *          空间哈希同时清空。
     */
    void endFrame();

private:
    std::pair<int, int> getGridCell(const cv::Point2f& pt) const;
    void insertToGrid(int global_id, const cv::Point2f& pt);
    std::vector<int> queryNearby(const cv::Point2f& pt) const;
};
```

## 4. Step-by-Step Implementation

### Step 1: 修改 `ChannelState`（feature_tracker.h）

- 将字段 `vector<int> ids;` 重命名为 `vector<int> local_ids;`。
- 为 `local_ids` 及所有其他字段补全/更新 Doxygen 风格注释，明确说明 key/value 含义。
- 在末尾添加 `int next_local_id = 0;` 并写注释。
- 将 `cur_un_pts_map`、`prev_un_pts_map`、`prevLeftPtsMap` 的注释明确标注 key 为 local_id。

**注意**：`feature_tracker.cpp` 中所有引用 `ch.ids` 的地方需同步改为 `ch.local_ids`（包括 `setMaskForChannel`、LK 匹配结果处理、速度计算、`removeOutliers` 等）。

### Step 2: 新增 YAML 参数（parameters.cpp / parameters.h）

在 `parameters.h` 新增全局变量：
```cpp
extern int POLAR_HASH_GRID_SIZE;   // 默认 5 (px)
```

在 `parameters.cpp` 读取：
```cpp
if (!fsSettings["polar_hash_grid_size"].empty())
    POLAR_HASH_GRID_SIZE = (int)fsSettings["polar_hash_grid_size"];
else
    POLAR_HASH_GRID_SIZE = 5;
```

并在 `featureTracker.setPolarHashGridSize(POLAR_HASH_GRID_SIZE)` 中注入到 `FeatureTracker` / `GlobalFeaturePool`。

### Step 3: 移除/隔离全局 `n_id` 对 Polar 模式的影响（feature_tracker.h / feature_tracker.cpp）

- `feature_tracker.h` 第 256 行 `int n_id;` 保留，但仅用于 legacy（非 polar）模式。
- `feature_tracker.cpp` 第 75 行构造函数 `n_id = 0;` 保留。
- **第 441–444 行**（新特征分配 ID）：
  ```cpp
  // 旧代码
  // ch.local_ids.push_back(n_id++);

  // 新代码
  ch.local_ids.push_back(ch.next_local_id++);
  ```
  每个通道独立递增自己的 `next_local_id`。

### Step 3: 实现 `GlobalFeaturePool`（feature_tracker.h）

新增 `GlobalFeaturePool` 结构体定义（详见 3.2 节）。

在 `FeatureTracker` 类中：
- 添加成员 `GlobalFeaturePool global_pool_;`（`feature_tracker.h`，约在 `channels` 向量附近）。
- 添加 `void setPolarHashGridSize(int size) { global_pool_.grid_size_ = size; }` 供 `parameters.cpp` 注入参数。
- 若 `n_id` 仍需要保留给 legacy 模式，可保持原样；`global_pool_` 仅在 `isPolarMode()` 下使用。

### Step 4: 实现 `GlobalFeaturePool` 方法体（feature_tracker.cpp，新增函数）

在 `feature_tracker.cpp` 的偏振辅助函数区域（约第 175 行之后）新增实现：

#### 4.1 `beginFrame()`
```cpp
cur_globals.clear();
grid.clear();
```

#### 4.2 `propagateTracked(const vector<ChannelState>& channels)`
遍历 `prev_bindings` 中每个 `(global_id, channel_map)`：
- 对每个 `(ch_name, prev_local_id)`：
  - 在 `channels` 中找到对应名称的通道 `ch`。
  - 在 `ch.local_ids` 中查找 `prev_local_id`。
  - 若找到（索引为 `idx`），说明 LK 跟踪成功：
    - `cur_globals[global_id].global_id = global_id`
    - `cur_globals[global_id].local_ids[ch_name] = prev_local_id`
    - `cur_globals[global_id].pixel_pts[ch_name] = ch.cur_pts[idx]`
    - `insertToGrid(global_id, ch.cur_pts[idx])`

#### 4.3 `registerUnboundFeatures(vector<ChannelState>& channels, int min_dist)`
```cpp
for (auto& ch : channels) {  // channels 顺序即优先级
    for (size_t i = 0; i < ch.local_ids.size(); ++i) {
        int local_id = ch.local_ids[i];
        // 检查本局部特征是否已在当前帧被绑定到某个全局 ID
        bool already_bound = false;
        for (auto& [gid, gf] : cur_globals) {
            if (gf.local_ids.count(ch.name) && gf.local_ids.at(ch.name) == local_id) {
                already_bound = true;
                break;
            }
        }
        if (already_bound) continue;

        cv::Point2f pt = ch.cur_pts[i];
        auto candidates = queryNearby(pt);

        bool attached = false;
        for (int gid : candidates) {
            auto& gf = cur_globals[gid];
            // 该全局特征在当前帧的此通道是否已有绑定？
            if (gf.local_ids.count(ch.name)) continue;

            // 距离检查：取该全局特征在此帧的任意已有像素坐标作为参考
            // 如果全局特征在当前帧还没有任何像素坐标（理论上不会发生，因为 propagate 或之前注册会放坐标），跳过
            if (gf.pixel_pts.empty()) continue;

            // 计算与最近已有坐标的距离
            double min_d = std::numeric_limits<double>::max();
            for (const auto& [_, other_pt] : gf.pixel_pts) {
                double dx = pt.x - other_pt.x;
                double dy = pt.y - other_pt.y;
                min_d = std::min(min_d, dx*dx + dy*dy);
            }
            if (min_d <= (min_dist / 2.0) * (min_dist / 2.0)) {
                gf.local_ids[ch.name] = local_id;
                gf.pixel_pts[ch.name] = pt;
                insertToGrid(gid, pt);
                attached = true;
                break;
            }
        }

        if (!attached) {
            int new_gid = next_global_id++;
            GlobalFeature gf;
            gf.global_id = new_gid;
            gf.local_ids[ch.name] = local_id;
            gf.pixel_pts[ch.name] = pt;
            cur_globals[new_gid] = std::move(gf);
            insertToGrid(new_gid, pt);
        }
    }
}
```

注意：如果全局特征在当前帧已有绑定（例如来自更高优先级的通道），其像素坐标已经存入 `pixel_pts`。新局部特征 attach 时，距离阈值以该全局特征当前帧已有坐标的最近者为准。

#### 4.4 `buildFeatureFrame(const vector<ChannelState>& channels) const`
```cpp
map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;

// 建立通道名 -> 通道索引的映射，方便快速查找
map<string, const ChannelState*> ch_map;
for (const auto& ch : channels) ch_map[ch.name] = &ch;

for (const auto& [gid, gf] : cur_globals) {
    // 按 channels 的优先级顺序，找到第一个有绑定的通道
    const ChannelState* selected_ch = nullptr;
    int selected_local_id = -1;
    size_t selected_idx = 0;

    for (const auto& ch : channels) {
        auto it = gf.local_ids.find(ch.name);
        if (it != gf.local_ids.end()) {
            selected_local_id = it->second;
            selected_ch = ch_map.at(ch.name);
            // 在 selected_ch->local_ids 中找索引
            for (size_t i = 0; i < selected_ch->local_ids.size(); ++i) {
                if (selected_ch->local_ids[i] == selected_local_id) {
                    selected_idx = i;
                    break;
                }
            }
            break;
        }
    }

    if (!selected_ch) continue;  // 无绑定，跳过

    const auto& ch = *selected_ch;
    double x = ch.cur_un_pts[selected_idx].x;
    double y = ch.cur_un_pts[selected_idx].y;
    double z = 1;
    double p_u = ch.cur_pts[selected_idx].x;
    double p_v = ch.cur_pts[selected_idx].y;
    double velocity_x = ch.pts_velocity[selected_idx].x;
    double velocity_y = ch.pts_velocity[selected_idx].y;

    Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
    xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
    featureFrame[gid].emplace_back(0, xyz_uv_velocity);
}
return featureFrame;
```

#### 4.5 `getGlobalToLocalMap() const`
```cpp
map<int, vector<pair<string, int>>> result;
for (const auto& [gid, gf] : cur_globals) {
    for (const auto& [ch_name, local_id] : gf.local_ids) {
        result[gid].emplace_back(ch_name, local_id);
    }
}
return result;
```

#### 4.6 `endFrame()`
```cpp
prev_bindings.clear();
prev_pts.clear();
for (const auto& [gid, gf] : cur_globals) {
    // 只保留在当前帧至少还有一个通道绑定的全局特征；
    // 完全丢失所有局部绑定的全局特征被直接删除，
    // 下一帧同网格出现的新局部特征将视为全新的全局特征。
    if (!gf.local_ids.empty()) {
        prev_bindings[gid] = gf.local_ids;
        prev_pts[gid] = gf.pixel_pts;
    }
}
```

#### 4.7 辅助函数
```cpp
pair<int, int> GlobalFeaturePool::getGridCell(const cv::Point2f& pt) const {
    return {static_cast<int>(pt.x) / grid_size_, static_cast<int>(pt.y) / grid_size_};
}

void GlobalFeaturePool::insertToGrid(int global_id, const cv::Point2f& pt) {
    auto cell = getGridCell(pt);
    grid[cell].push_back(global_id);
}

vector<int> GlobalFeaturePool::queryNearby(const cv::Point2f& pt) const {
    vector<int> result;
    auto [cx, cy] = getGridCell(pt);
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            auto it = grid.find({cx + dx, cy + dy});
            if (it != grid.end()) {
                result.insert(result.end(), it->second.begin(), it->second.end());
            }
        }
    }
    return result;
}
```

### Step 5: 修改 `trackImage()` 主流程（feature_tracker.cpp，第 492–512 行）

将原有“合并所有通道结果 → VINS 格式”代码块替换为全局池调用：

```cpp
    // =============================================
    // 3. 全局特征池注册与合并
    // =============================================
    global_pool_.beginFrame();
    global_pool_.propagateTracked(channels);
    global_pool_.registerUnboundFeatures(channels, MIN_DIST);
    auto featureFrame = global_pool_.buildFeatureFrame(channels);
    global_pool_.endFrame();
```

旧代码（约第 495–512 行）全部删除。

### Step 6: 修改 `removeOutliers()`（feature_tracker.cpp，第 833–864 行）

当前 polar 分支直接比较 `ch.ids[i]`（全局 ID）和 `removePtsIds`。改为通过全局池查询：

```cpp
void FeatureTracker::removeOutliers(set<int> &removePtsIds)
{
    if (isPolarMode()) {
        // 获取当前帧全局 ID -> [(channel_name, local_id)] 映射
        auto global_to_local = global_pool_.getGlobalToLocalMap();

        // 为每个通道构建一个待移除的 local_id set
        map<string, set<int>> removeLocalIdsPerChannel;
        for (int gid : removePtsIds) {
            auto it = global_to_local.find(gid);
            if (it != global_to_local.end()) {
                for (const auto& [ch_name, local_id] : it->second) {
                    removeLocalIdsPerChannel[ch_name].insert(local_id);
                }
            }
        }

        for (auto& ch : channels) {
            vector<uchar> status;
            for (size_t i = 0; i < ch.local_ids.size(); i++) {
                auto it = removeLocalIdsPerChannel.find(ch.name);
                if (it != removeLocalIdsPerChannel.end() && it->second.count(ch.local_ids[i]))
                    status.push_back(0);
                else
                    status.push_back(1);
            }
            reduceVector(ch.prev_pts, status);
            reduceVector(ch.local_ids, status);
            reduceVector(ch.track_cnt, status);
        }
    } else {
        // legacy 逻辑不变
        ...
    }
}
```

注意：`removeOutliers` 修改的是 `prev_pts`，因为该函数在 `processImage` 之后、下一帧 `trackImage` 之前调用，此时 `prev_pts` 仍是本帧跟踪结束后的“上一帧点集”。清除后，下一帧 LK 光流不会从这些 outlier 位置发起。

### Step 7: `setPrediction()`（feature_tracker.cpp，第 794–821 行）

当前 polar 模式下为 no-op。新架构下全局池理论上可以支持 prediction，但实现较复杂（需要将全局 ID 的预测位置映射到各通道的 local ID）。**本版本保持 no-op**，后续若有需求可扩展：
- 在 `GlobalFeaturePool` 中维护 `map<int, cv::Point2f> predictions_`（key 为 global_id，value 为预测的像素坐标）。
- `setPrediction` 接收 `map<int, Vector3d>`（归一化平面坐标），反投影为像素坐标后存入 predictions_。
- 在 `registerUnboundFeatures` 的 LK 匹配阶段使用预测位置加速搜索。

### Step 8: 可视化代码检查（feature_tracker.cpp，drawTrackPolar）

`drawTrackPolar` 使用 `ch.local_ids` 作为 key 来查 `ch.prevLeftPtsMap`。由于字段已改名且语义一致，`prevLeftPtsMap` 在 `trackImage` 末尾（第 488–489 行）仍以 local_id 为 key 插入即可。**无需修改**。

但需注意：`prevLeftPtsMap` 当前在 `trackImage` 中用于可视化连线。如果某全局特征在当前帧换了通道（例如从 dop 继承变为 s0 继承），可视化上该点会显示为 s0 的颜色，这是预期行为。

## 5. 边界情况与处理

### 5.1 第一帧（无 prev 状态）

`prev_bindings` 为空，`propagateTracked` 什么都不做。所有局部特征都进入 `registerUnboundFeatures`。由于空间哈希为空，所有局部特征都会创建新的全局 ID。s0 先创建一批，dop 后创建一批并可能 attach 到 s0 创建的相近全局特征上。最终全局特征数量 ≈ 空间去重后的实际物理角点数量，远小于旧架构的“通道数 × 角点数”。

### 5.2 某通道所有特征丢失

若某帧 dop 通道 LK 跟踪全部失败，`propagateTracked` 不会为 dop 继承任何绑定。但该帧 dop 新检测的特征仍可在 `registerUnboundFeatures` 中 attach 到已有的全局特征（由 s0 或其他通道创建/继承），保证全局 ID 不漂移。

### 5.3 全局特征数量爆炸

**已解决**：`endFrame()` 只将当前帧仍至少有一个通道绑定的全局特征保留到 `prev_bindings`；完全丢失所有局部绑定的全局特征被直接删除。因此全局 ID 不会无限累积——丢失即释放。下一帧若同网格出现新局部特征，会分配新全局 ID，语义正确。

### 5.4 两个通道在同一帧检测到同一新角点

假设 s0 和 dop 同时检测到同一新角点（例如物体突然进入视野）。s0 优先级高，先处理，空间哈希无匹配 → 创建新全局 ID `G`。dop 后处理，查询空间哈希发现 `G` 且距离满足阈值 → attach 到 `G`。最终 `G` 同时绑定 `s0:local_x` 和 `dop:local_y`。后端只看到 `G` 的一个观测（来自 s0）。完美。

### 5.5 同一通道内两个局部特征都想 attach 到同一全局特征

每个通道的局部特征在检测阶段已经过 `setMaskForChannel`，保证彼此间距 ≥ `MIN_DIST`。全局特征的空间哈希匹配阈值是 `MIN_DIST / 2`。因此同一通道内不可能有两个未绑定局部特征同时满足 attach 条件到同一全局特征。即使发生（数值误差），由于代码中是顺序遍历，先处理的会成功 attach，后处理的会查询到“该全局特征在此通道已绑定”而跳过，最终创建新全局 ID 或 attach 到其他全局特征。

## 6. 接口契约

### 6.1 Frontend → Backend（不变）

```cpp
map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
```

- key: `global_id`（全局特征池分配）
- value 内 pair.first: camera_id，始终为 0（单目）
- value 内 pair.second: 7D 观测向量

### 6.2 Backend → Frontend 反馈

#### `removeOutliers(set<int> &removePtsIds)`
- 输入：`removePtsIds` 为全局 ID 集合（语义不变）。
- 行为：FeatureTracker 通过全局池查询每个全局 ID 对应的 `(channel_name, local_id)`，在各通道的 `prev_pts/ids/track_cnt` 中剔除。

#### `setPrediction(map<int, Vector3d> &predictPts)`
- 输入：key 为全局 ID。
- 当前版本：no-op。

## 7. 文件变更清单

| 文件 | 变更类型 | 说明 |
|---|---|---|
| `vins_estimator/src/featureTracker/feature_tracker.h` | 修改 + 新增 | `ChannelState` 重命名 `ids`->`local_ids`，添加 `next_local_id`；新增 `GlobalFeaturePool`（含完整注释）；`FeatureTracker` 添加 `global_pool_` |
| `vins_estimator/src/featureTracker/feature_tracker.cpp` | 修改 | 新特征分配改用 `ch.next_local_id++`；`trackImage` 合并逻辑替换为全局池三步调用；新增 `GlobalFeaturePool` 全部方法体（含注释）；修改 `removeOutliers`；所有旧注释同步更新 |
| `vins_estimator/src/estimator/parameters.h` | 修改 | 新增 `extern int POLAR_HASH_GRID_SIZE;` |
| `vins_estimator/src/estimator/parameters.cpp` | 修改 | 从 YAML 读取 `polar_hash_grid_size`，默认 5 |
| `config/dark/dark_mono_imu_config.yaml` | 修改 | 新增 `polar_hash_grid_size: 5` 示例行 |
| 其他所有文件 | 不变 | 后端、CMake 等均不改动 |

## 8. Verification Plan

### 8.1 编译验证
```bash
cd ~/ws/vi_catkin_ws
catkin_make
```
确保 `vins_estimator` 编译通过，无 warning。

### 8.2 单元逻辑验证（建议增加临时日志）

在 `trackImage` 的全局池处理前后增加临时日志（建议用 `ROS_INFO`，方便直接看）：
- 打印每帧各通道的局部特征数量（`ch.local_ids.size()`）。
- 打印 `propagateTracked` 后继承的全局特征数量（`cur_globals` 大小）。
- 打印 `registerUnboundFeatures` 后新创建的全局特征数量（`next_global_id` 增量或 `cur_globals` 大小）。
- 打印最终 `featureFrame.size()`（发给后端的特征数）。
- 打印 `endFrame` 后保留到 `prev_bindings` 的全局特征数量（验证清理机制）。

**预期行为**：
- 在 4 通道场景下，旧架构每帧发给后端的特征数 ≈ `4 × MAX_CNT`（去重前）；新架构应显著下降，接近单通道场景的特征数（例如 `~MAX_CNT` 或稍多，取决于场景纹理丰富度）。
- `featureFrame.size()` 应远小于 `sum(ch.ids.size())`。

### 8.3 运行时轨迹验证

使用已知轨迹数据集（如 EuRoC MH_01_easy 的偏振合成版本，或实际采集的 dark room 数据）：
1. **定性**：RViz 中特征轨迹线（`SHOW_TRACK = 1`）应更“干净”，同一物理角点不会被画出 4 条不同颜色的平行线。
2. **定量**：对比 V2.4 与新版本的 `feature_manager` 特征数（日志或 CSV 输出）。新版本在滑动窗口稳定后，`f_manager.feature.size()` 应明显小于 V2.4。
3. **精度**：对比最终位姿估计的 RMSE。理论上全局池减少了冗余观测，BA 更稳定，RMSE 应持平或略降。若出现明显退化，需检查是否因“每帧只发单观测”导致有效观测数不足（可通过调低 `MIN_DIST` 或提高 `MAX_CNT` 补偿）。

### 8.4 边界情况测试

- **单通道模式**（`polar_channels: s0`）：全局池退化为直接透传，行为应与 V2.4 完全一致。
- **两通道模式**（`s0,dop`）：验证空间去重效果最明显。
- **动态场景**：快速运动导致通道间特征位置差异大时，验证全局池不会错误 merge（应保证距离阈值合理）。

## 9. 风险与回滚

| 风险 | 缓解措施 |
|---|---|
| 空间哈希阈值过严，导致应合并的特征未合并 | 网格大小默认 5px（远小于 `MIN_DIST`），允许更细粒度的空间查询；距离阈值仍使用 `MIN_DIST / 2`，可通过 YAML `polar_hash_grid_size` 调节 |
| 空间哈希阈值过松，导致不同角点错误合并 | 距离阈值保持 `MIN_DIST / 2`；`setMaskForChannel` 已保证通道内特征间距 ≥ `MIN_DIST` |
| `removeOutliers` 映射错误，导致前端误删正常特征 | 仔细检查 `getGlobalToLocalMap` 与 `removeOutliers` 的通道名匹配逻辑 |
| 后端 feature_manager 因全局 ID 数量变少而判定为关键帧过于频繁 | 观察 `last_track_num` 和 `new_feature_num` 统计；若退化，可适当提高 `MAX_CNT` |

回滚方式：由于仅修改 `feature_tracker.h/cpp`，可直接 `git checkout` 这两个文件恢复 V2.4 行为。

---

*计划版本: V2.5-draft-1*
*日期: 2026-04-21*
