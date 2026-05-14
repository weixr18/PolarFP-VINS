# Plan: 融合特征点统计收集（修订版）

## Context

为验证 Global Feature Pool 设计中跨通道融合的效果，需要统计每帧**融合（全局）特征点**的指标。核心关注点：多个偏振通道检测到的同一空间点的特征，经全局池合并后记为 1 个全局特征，指标反映融合后的效果而非分通道效果。

## CSV 文件名

格式: `polarfp_stats_{yyyymmdd_hhmmss}_{ch_names}.csv`
- `yyyymmdd_hhmmss`: 第一帧 ROS 时间戳转换的本地时间
- `ch_names`: 通道名用下划线连接，例如 `s0_dop_aopsin_aopcos`
- 输出到 `OUTPUT_FOLDER` 目录下

示例: `polarfp_stats_20260514_143021_s0_dop_aopsin_aopcos.csv`

## CSV 列定义（仅全局指标，无分通道统计）

| 列名 | 含义 | 收集时机 |
|------|------|---------|
| `frame_idx` | 帧序号 | 从 0 计数 |
| `timestamp` | 时间戳 | Unix 时间，保留 6 位小数 |
| `num_global_matched` | 匹配的融合特征点数目 | `propagateTracked` 后 `cur_globals` 大小（从上一帧成功传播的全局特征） |
| `num_global_total` | 融合特征点总数 | `registerUnboundFeatures` 后 `cur_globals` 大小（匹配 + 新注册） |
| `avg_global_track` | 融合特征点的平均跟踪帧数 | 遍历 `cur_globals`，对每个全局特征取绑定的最大 `track_cnt`，做平均 |

## avg_global_track 计算

全局特征本身没有 `track_cnt`。方法：对每个**传播过来的**全局特征（即 `propagateTracked` 后的 `cur_globals`），遍历其 `local_ids` 绑定的各通道 `(ch_name, local_id)`，在各通道的 `track_cnt` 向量中找到对应值，取最大值作为该全局特征的等效跟踪帧数，然后做平均。**新注册的特征点（跟踪帧数为 0）不参与统计。**

## YAML 开关

新增 `polarfp_stats_enabled: 0`（默认 0，关闭）。设为 1 时启用。

## 修改文件

### 1. `estimator/parameters.h`
- 新增 `extern bool POLARFP_STATS_ENABLED;`

### 2. `estimator/parameters.cpp`
- 定义 `bool POLARFP_STATS_ENABLED = false;`
- `readParameters()` 中读取 YAML

### 3. `featureTracker/feature_tracker.h`
FeatureTracker 私有区新增：
```cpp
bool stats_initialized_ = false;
std::string stats_csv_path_;
int stats_frame_idx_ = 0;
double stats_first_timestamp_ = 0.0;
void initStatsFile();
void writeStatsRow(int matched, int total, double avg_track);
```

### 4. `featureTracker/feature_tracker.cpp`
- `initStatsFile()`: 时间戳 → `yyyymmdd_hhmmss` + 通道名 → 文件名，写表头
- `writeStatsRow()`: 写一行数据
- `trackImage()`: 在全局池处理流程中插入

## trackImage() 插入点

```cpp
global_pool_.beginFrame();
global_pool_.propagateTracked(channels);
int num_matched = global_pool_.cur_globals.size();       // ← 收集 matched
double avg_track = computeAvgTrack();                      // ← 计算 avg（只对传播来的特征）
global_pool_.registerUnboundFeatures(channels, MIN_DIST);
int num_total = global_pool_.cur_globals.size();          // ← 收集 total
auto featureFrame = global_pool_.buildFeatureFrame(channels);
global_pool_.endFrame();

if (POLARFP_STATS_ENABLED) {
    if (!stats_initialized_) { stats_first_timestamp_ = cur_time; initStatsFile(); }
    writeStatsRow(num_matched, num_total, avg_track);
    stats_frame_idx_++;
}
```

## 验证

1. `catkin_make` 编译通过
2. config 中设置 `polarfp_stats_enabled: 1`，运行后检查 CSV 文件
3. CSV 行数 == 帧数，`num_global_matched <= num_global_total`，`avg_global_track >= 1`
