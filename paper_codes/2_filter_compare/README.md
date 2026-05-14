# 偏振通道滤波方法对比

对原始偏振图像解码出的 P0(DoP)、P1(sin(AoP))、P2(cos(AoP)) 三个通道，分别施加中值滤波、导向滤波（以 S0 为引导）、双边滤波、非局部均值去噪（NLM）四种算法。

## 文件说明

| 文件 | 说明 |
|------|------|
| `get_data_2.py` | 从 ROSbag 中**随机抽取** K 帧保存为 PNG（用于可视化对比） |
| `get_data_2_stream.py` | 从 ROSbag 中提取**前 N 帧连续帧**保存为 PNG（用于量化评估） |
| `filter_compare_1.cpp` | C++ 程序：读 PNG → 偏振解码 → 四种滤波 → 拼图输出（3行×5列可视化对比） |
| `filter_compare_table2.cpp` | C++ 程序：读连续帧 PNG → 偏振解码 → 滤波 → GFTT角点检测 + 双向LK光流匹配 → 量化指标统计 |
| `CMakeLists.txt` | CMake 编译配置，包含 `filter_compare` 和 `filter_compare_table2` 两个目标 |
| `table1_params.tex` | 论文表1：滤波算法参数 LaTeX 代码 |
| 共享的 `PolarChannel.h/cpp` | 位于 `../utilities/`，偏振通道解码模块 |

---

## 一、可视化对比（filter_compare_1）

生成 3行×5列 对比图，用于论文 Fig. 2 定性展示。

### 使用流程

#### 1. 从 ROSbag 随机抽帧

```bash
python3 get_data_2.py \
  --bags ~/data/dark/0416/13-27-22/2026-04-16-13-27-22.bag \
         ~/data/dark/0416/13-54-30/2026-04-16-13-54-30.bag \
         ~/data/dark/0416/14-09-36/2026-04-16-14-09-36.bag \
  --output data/ --k 10
```

每组 bag 随机抽取 10 帧，保存为 `<场景名>_<序号>.png`。

#### 2. 编译

```bash
mkdir -p build && cd build && cmake .. && make -j$(nproc)
cd ..
```

依赖：OpenCV（无需 ROS）。

#### 3. 生成对比图

```bash
./build/filter_compare data/ output/
```

每张 PNG 输入生成一张 3行×5列 的对比图：

- **行**: P0 (DoP) | P1 (sin(AoP)) | P2 (cos(AoP))
- **列**: Raw | Median | Guided | Bilateral | NLM

---

## 二、量化评估（filter_compare_table2）

对每种 channel × filter 组合，使用 GFTT 角点检测 + 双向 LK 光流匹配，统计帧平均指标，产出论文 Table 2 所需数据。

### 量化指标

| 指标 | 方法 |
|------|------|
| 帧平均检测数 (Avg Detect) | GFTT 角点检测 (`cv::goodFeaturesToTrack`) |
| 帧平均匹配数 (Avg Match) | 相邻帧间双向 LK 光流匹配（正向+反向验证，回环距离 ≤ 0.5px） |
| 帧平均处理时间 (Avg Time) | 每帧滤波+GFTT+LK 的耗时（ms） |

### 关键参数

| 组件 | 参数 |
|------|------|
| 滤波 | 同表1：Median 3×3×3轮 / Guided r=8,eps=0.0001 / Bilateral d=9,σc=200,σs=30 / NLM h=50,t=5,s=21 |
| GFTT | max_corners=500, quality=0.01, min_dist=10, block=3 |
| 正向 LK | win=(21,21), maxLevel=3 |
| 反向 LK | win=(21,21), maxLevel=1, 使用 OPTFLOW_USE_INITIAL_FLOW，TermCriteria(COUNT+EPS,30,0.01) |
| 反向阈值 | 回环距离 ≤ 0.5 px |

### 使用流程

#### 1. 提取连续帧数据

从三个光照条件下的 ROS bag 中各提取前 200 帧连续图像：

```bash
python3 get_data_2_stream.py \
  ~/data/dark/0416/13-27-22/2026-04-16-13-27-22.bag \
  ~/data/dark/0416/13-54-30/2026-04-16-13-54-30.bag \
  ~/data/dark/0416/14-09-36/2026-04-16-14-09-36.bag \
  data_stream/ --n 200
```

输出结构：
```
data_stream/
├── 13-27-22/       # 94–205 lux
│   ├── frame_00000.png
│   ├── frame_00001.png
│   └── ...
├── 13-54-30/       # 2.6–18.6 lux
│   └── ...
└── 14-09-36/       # 0.9–4.8 lux
    └── ...
```

#### 2. 编译（同可视化对比，CMakeLists.txt 已包含两个目标）

```bash
mkdir -p build && cd build && cmake .. && make -j$(nproc) && cd ..
```

#### 3. 运行量化评估

```bash
for d in data_stream/*/; do
  echo "=== Processing $(basename $d) ==="
  ./build/filter_compare_table2 "$d" "output/table2_$(basename $d).csv"
done
```

输出控制台表格示例：
```
Dataset: 13-27-22
Frames: 200  (matching pairs: 199)
===============================================================================
Channel      Filter          Avg Detect    Avg Match    Avg Time(ms)
------------------------------------------------------------------------------
DoP          Raw                   123.4         98.7           5.23
DoP          Median                130.1        105.2           6.15
...
```

同时输出 CSV 文件（`output/table2_<scene>.csv`），格式：
```
dataset,channel,filter,avg_detect,avg_match,avg_time_ms
13-27-22,DoP,Raw,123.4,98.7,5.23
...
```

---

## 滤波参数

| 算法 | 参数 | 取值 |
|------|------|------|
| 中值滤波 | 核大小 | 3×3（执行 3 轮） |
| 导向滤波 | 窗口半径 r / 正则化 ε | 8 / 0.0001（以 S0 引导） |
| 双边滤波 | 邻域直径 d / σ_color / σ_space | 9 / 200 / 30 |
| NLM | 滤波强度 h / 模板窗口 / 搜索窗口 | 50 / 5×5 / 21×21 |

详见 `table1_params.tex`。

---

## 光照条件

| 场景 | 目录 | 照度范围 |
|------|------|----------|
| 正常光 | `13-27-22` | 94–205 lux |
| 暗光 | `13-54-30` | 2.6–18.6 lux |
| 极暗 | `14-09-36` | 0.9–4.8 lux |
