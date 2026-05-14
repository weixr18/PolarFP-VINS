# 特征点检测方法对比

在导向滤波预处理（r=8, eps=0.0001，以 S0 为引导图）后的 DoP/sinAoP/cosAoP 三个偏振通道上，对比 FAST / GFTT / SuperPoint 三种特征点检测方法的性能。

## 文件说明

| 文件 | 说明 |
|------|------|
| `get_data_2.py` | 软链接 → `../2_filter_compare/get_data_2.py`，从 ROSbag 随机抽取帧 |
| `get_data_2_stream.py` | 软链接 → `../2_filter_compare/get_data_2_stream.py`，提取连续帧 |
| `fp_compare_1.cpp` | C++ 程序：读 PNG → 偏振解码 → 导向滤波 → 三种检测器 → 3×3 特征点叠加图 |
| `fp_compare_table3.cpp` | C++ 程序：读连续帧 PNG → 偏振解码 → 导向滤波 → 检测 + 双向 LK 光流匹配 → 量化指标统计 |
| `CMakeLists.txt` | CMake 编译配置，包含 `fp_compare_1` 和 `fp_compare_table3` 两个目标，依赖 LibTorch |
| `visualize_res.py` | Python 可视化：分组柱状图 + 综合柱状图 + Markdown 汇总表 |
| 共享的 `PolarChannel.h/cpp` | 位于 `../utilities/`，偏振通道解码模块 |

---

## 一、可视化对比（fp_compare_1）

生成 3行×3列 特征点叠加对比图，直观展示各通道 × 各检测器的特征点分布。

### 图布局

- **行**: DoP (P0) | sinAoP (P1) | cosAoP (P2)
- **列**: FAST | GFTT | SuperPoint

每格显示通道灰度图，绿色圆点叠加表示检测到的特征点，左下角标注特征点数量。

### 使用流程

#### 1. 准备数据

```bash
# 从 2_filter_compare 复制代表性帧（每场景 1 帧）
cp ../2_filter_compare/data/*_00.png data/

# 或重新从 bag 中提取
python3 get_data_3.py \
  --bags ~/data/dark/0416/13-27-22/2026-04-16-13-27-22.bag \
         ~/data/dark/0416/13-54-30/2026-04-16-13-54-30.bag \
         ~/data/dark/0416/14-09-36/2026-04-16-14-09-36.bag \
  --output data/ --k 1
```

#### 2. 编译

```bash
mkdir -p build && cd build && cmake .. && make -j$(nproc) && cd ..
```

依赖：OpenCV + LibTorch（SuperPoint 需要）。

#### 3. 生成对比图

```bash
./build/fp_compare_1 data/ figures/
```

输出：`figures/feature_points_<场景>_<序号>.png`

---

## 二、量化评估（fp_compare_table3）

对每种 channel × detector 组合，统计帧平均指标，产出论文 Table 3 所需数据。

### 量化指标

| 指标 | 方法 |
|------|------|
| 帧平均检测数 (Avg Detect) | 检测器直接提取的特征点数量 |
| 帧平均匹配数 (Avg Match) | 相邻帧间双向 LK 光流匹配（正向+反向验证，回环距离 ≤ 0.5px） |
| 帧平均内点数 (Avg Inlier) | RANSAC 本质矩阵内点数量（阈值 1.0，置信度 0.99） |
| 帧平均内点比例 (Inlier Ratio) | 内点数 / 匹配数 |
| 帧平均处理时间 (Avg Time) | 每帧检测+LK匹配+RANSAC 的耗时（ms） |

### 检测器参数

| 检测器 | 参数 |
|--------|------|
| FAST | threshold=10, nonmaxSuppression=true, 按 response 取 top-500 |
| GFTT | max_corners=500, quality=0.01, min_dist=10, block=3 |
| SuperPoint | TorchScript 模型, kp_thresh=0.005, nms_radius=4, 按 score 取 top-500 |

### LK 光流参数（与 2_filter_compare 一致）

| 参数 | 取值 |
|------|------|
| 正向 LK | win=(21,21), maxLevel=3 |
| 反向 LK | win=(21,21), maxLevel=1, OPTFLOW_USE_INITIAL_FLOW |
| 反向阈值 | 回环距离 ≤ 0.5 px |

### 使用流程

#### 1. 连续帧数据

```bash
# 已软链接到 ../2_filter_compare/data_stream/，无需重新提取
ls data_stream/
# 13-27-22/  13-54-30/  14-09-36/

# 或重新提取
python3 get_data_2_stream.py \
  ~/data/dark/0416/13-27-22/2026-04-16-13-27-22.bag \
  ~/data/dark/0416/13-54-30/2026-04-16-13-54-30.bag \
  ~/data/dark/0416/14-09-36/2026-04-16-14-09-36.bag \
  data_stream/ --n 200
```

#### 2. 编译（同上）

#### 3. 运行量化评估

```bash
for d in data_stream/*/; do
  echo "=== Processing $(basename $d) ==="
  ./build/fp_compare_table3 "$d" "output/table3_$(basename $d).csv"
done
```

CSV 输出格式：
```csv
dataset,channel,detector,avg_detect,avg_match,avg_inlier,avg_time_ms
13-27-22,DoP,FAST,88.7,77.8,73.4,1.0
```

---

## 三、Python 可视化（visualize_res.py）

基于 CSV 数据生成论文用图表。

```bash
python3 visualize_res.py
```

### 输出

| 文件 | 说明 |
|------|------|
| `figures/planA_fp_metrics_grid.png` | 分组柱状图网格：3 行（光照）× 3 列（avg_inlier / inlier_ratio / avg_time_ms） |
| `figures/fp_summary_bars.png` | 综合柱状图：3 个子图（每种光照一个），展示 3 检测器 × 3 通道的 avg_inlier |
| `figures/fp_summary_table.md` | Markdown 汇总表，按光照条件列出各组合指标及最佳方法 |

---

## 关键设计决策

1. **滤波固定为引导滤波**：r=8, eps=0.0001，与 2_filter_compare 一致，以 S0 为引导图
2. **检测器参数不做逐通道调优**：对比的是检测器本身对不同数据分布的适应能力
3. **SuperPoint 使用批量推理**：fp_compare_table3 中将 3 个通道打包为 [3, H, W] 张量一次前向推理，提升效率
4. **SuperPoint 计时**：首帧前执行 GPU warm-up，正式循环中 batch 耗时平分到各通道
5. **FAST threshold=10**（原计划 20）：针对偏振通道低对比度特点降低阈值，保证暗光下有足够检测数

---

## 光照条件

| 场景 | 目录 | 照度范围 |
|------|------|----------|
| 正常光 | `13-27-22` | 94–205 lux |
| 暗光 | `13-54-30` | 2.6–18.6 lux |
| 极暗 | `14-09-36` | 0.9–4.8 lux |
