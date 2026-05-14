# 3_fp_compare — 特征点检测方法对比方案

## 目标

对比三种特征点检测方法（FAST / GFTT / SuperPoint）在经过导向滤波预处理的偏振通道上的表现。**滤波方法固定为引导滤波**（r=8, eps=0.0001，以 S0 为引导图），参数与 2_filter_compare 一致。

## 数据来源

复用 2_filter_compare 的 3 个 bag：

| 场景 | 光照条件 | 照度范围 |
|------|----------|----------|
| 13-27-22 | 正常光 | 94–205 lux |
| 13-54-30 | 暗光 | 2.6–18.6 lux |
| 14-09-36 | 极暗 | 0.9–4.8 lux |

- **可视化部分**：复用 `get_data_2.py`，每个 bag 随机抽取 1 帧（或手动指定代表性帧）→ `data/`
- **量化评估部分**：复用 `get_data_2_stream.py`，每个 bag 提取前 200 帧连续帧 → `data_stream/`（可直接软链接 2_filter_compare 已有的数据）

---

## 第一部分 — 可视化：特征点叠加图（fp_compare_1.cpp）

**目的**：对每组光照条件，生成 1 张对比图，直观展示各通道 × 各检测器检测到的特征点分布。

**图布局**：3 行（通道：DoP / sinAoP / cosAoP）× 3 列（检测器：FAST / GFTT / SuperPoint），共 9 个子图。每格显示通道图像，并将检测到的特征点以彩色圆点叠加在图像上。

**各检测器参数**：
- GFTT：max_corners=500, quality=0.01, min_dist=10, block=3（与 2_filter_compare 一致）
- FAST：threshold=20, nonmaxSuppression=true，随后按 response 分数取 top-500
- SuperPoint：TorchScript 模型，kp_thresh=0.015, nms_radius=4（与 PolarFP-VINS 配置文件一致）

**每帧处理流程**：
1. 读取原始 PNG → `raw2polar()` → S0、DoP、sinAoP、cosAoP
2. 对 3 个偏振通道均施加导向滤波（S0 引导，r=8, eps=0.0001）
3. 对每个滤波后通道分别运行 FAST / GFTT / SuperPoint → 得到特征点坐标
4. 在通道图像上绘制特征点（彩色圆点）
5. 拼接为 3×3 对比图，注明行标签（通道名）和列标签（检测器名）

**SuperPoint 依赖**：需要链接 LibTorch，路径与 VINS 主工程一致（`/home/dhz/SLAM/libs/libtorch-cxx11-abi-shared-with-deps-1.12.1+cu113/libtorch`）。C++ 代码需 `#include <torch/script.h>` 并加载 TorchScript 模型。这是相比 2_filter_compare 的主要新增复杂度。

**输出**：`figures/feature_points_13-27-22.png`、`feature_points_13-54-30.png`、`feature_points_14-09-36.png`

---

## 第二部分 — 量化评估（fp_compare_table3.cpp）

**目的**：对每种 通道 × 检测器 组合，在 200 帧连续帧上统计帧平均指标。

**量化指标**（与 2_filter_compare 表2 一致）：
- 帧平均检测数（Avg Detect）：每帧检测到的特征点数量均值
- 帧平均匹配数（Avg Match）：相邻帧间双向 LK 光流匹配成功的点对数量均值
- 帧平均内点数（Avg Inlier）：经 RANSAC 本质矩阵验证的内点数量均值
- 帧平均内点比例（Inlier Ratio）= 内点数 / 匹配数
- 帧平均处理时间（Avg Time, ms）：每帧（解码+引导滤波+检测+LK匹配+RANSAC）的总耗时

**测试矩阵**：3 通道 × 3 检测器 = 每组光照 9 个组合：

```
通道       检测器
DoP        FAST
DoP        GFTT
DoP        SuperPoint
sinAoP     FAST
sinAoP     GFTT
sinAoP     SuperPoint
cosAoP     FAST
cosAoP     GFTT
cosAoP     SuperPoint
```

**每帧处理流程**：
1. 读取 PNG → `raw2polar()` → 对 DoP/sin/cos 施加导向滤波
2. 对每个通道：
   a. 用当前检测器提取特征点
   b. 与前帧做双向 LK 光流匹配（参数同 2_filter_compare：正向 win=21, maxLevel=3；反向 win=21, maxLevel=1, 使用 OPTFLOW_USE_INITIAL_FLOW；回环距离 ≤ 0.5px）
   c. RANSAC 本质矩阵估计内点数（参数同：阈值=1.0，置信度=0.99）
   d. 累加数量和耗时

**各检测器参数**：

| 检测器 | 参数 |
|--------|------|
| FAST | threshold=20, nonmaxSuppression=true, 按 response 取 top-500 |
| GFTT | max_corners=500, quality=0.01, min_dist=10, block=3 |
| SuperPoint | TorchScript 模型, kp_thresh=0.015, nms_radius=4, 按 score 取 top-500 |

**输出**：控制台表格 + 每个光照条件一个 CSV 文件（如 `output/table3_13-27-22.csv`）

CSV 格式：
```csv
dataset,channel,detector,avg_detect,avg_match,avg_inlier,avg_time_ms
13-27-22,DoP,FAST,...
```

---

## 第三部分 — Python 数据可视化（visualize_res.py）

复用 `2_filter_compare/visualize_res.py` 的可视化模式，将"滤波方法"维度替换为"检测器"：

**Plan A — 分组柱状图网格**：3 行（光照）× 3 列（avg_inlier / inlier_ratio / avg_time_ms）。每张子图：x 轴 = 3 种检测器，分组柱 = 3 个通道。

**补充 — 综合柱状图（推荐用于论文）**：1 张大图包含 3 个子图（每种光照 1 个），每个子图展示 3 检测器 × 3 通道 = 9 根柱。

**Markdown 汇总表**：按光照条件列出最佳检测器组合。

**输出**：`figures/planA_fp_metrics_grid.png`、`figures/fp_summary_table.md`

---

## 文件清单

```
paper_codes/3_fp_compare/
├── plan_3.md                  # 本方案文档
├── CMakeLists.txt             # 编译配置（2 个目标：fp_compare_1, fp_compare_table3）
├── get_data_2.py              # 软链接 → ../2_filter_compare/get_data_2.py（复用）
├── get_data_2_stream.py       # 软链接 → ../2_filter_compare/get_data_2_stream.py（复用）
├── fp_compare_1.cpp           # 可视化：3×3 特征点叠加图
├── fp_compare_table3.cpp      # 量化评估：检测器指标统计
├── visualize_res.py           # Python 柱状图 + markdown 表格
├── data/                      # 从 bag 随机抽取的帧（用于可视化）
├── data_stream/               # 连续帧数据（软链接到 ../2_filter_compare/data_stream/）
├── output/                    # fp_compare_table3 输出的 CSV
└── figures/                   # 生成的图表
```

---

## CMakeLists.txt 要点

需要引入 LibTorch 以支持 SuperPoint。两个编译目标：

```cmake
# 查找 LibTorch
find_package(Torch REQUIRED PATHS
    /home/dhz/SLAM/libs/libtorch-cxx11-abi-shared-with-deps-1.12.1+cu113/libtorch)

# 目标 1：可视化
add_executable(fp_compare_1
    fp_compare_1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../utilities/PolarChannel.cpp
)
target_include_directories(fp_compare_1 PRIVATE
    ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../utilities)
target_link_libraries(fp_compare_1 ${OpenCV_LIBS} ${TORCH_LIBRARIES})

# 目标 2：量化评估（配置同上）
add_executable(fp_compare_table3
    fp_compare_table3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../utilities/PolarChannel.cpp
)
target_include_directories(fp_compare_table3 PRIVATE
    ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../utilities)
target_link_libraries(fp_compare_table3 ${OpenCV_LIBS} ${TORCH_LIBRARIES})
```

---

## 关键设计决策

1. **GFTT 保持 max_corners=500**（与 2_filter_compare 一致）。虽然 VINS 原始配置用 max_cnt=150 做跟踪，但此处测试的是原始检测能力，500 能更充分体现检测器差异。

2. **FAST 采用 threshold=20 + 按 response 取 top-500**。`cv::FAST()` 返回的点按 response 降序排列，取前 500 个与 GFTT 输出规模对齐，保证公平对比。threshold=20 是 OpenCV 默认值，对 8bit 图像是合理起点。

3. **SuperPoint 模型路径**：待确认实际路径（可能在 `/home/dhz/SLAM/libs/superpoint/` 下），编码前先 `ls` 确认。

4. **所有通道使用相同检测器参数**，不做逐通道调优。这样对比的是检测器本身对不同数据分布的适应能力，而非调参水平。

5. **SuperPoint 计时**：首帧包含 CUDA kernel 编译和模型预热，耗时不可比。在正式循环前跑一次 dummy 推理预热，所有帧统一计时。
   - 补充：在 `fp_compare_1.cpp`（可视化）中不需要关心计时，只需加载模型正常推理即可。

6. **两次 C++ 程序中 SuperPoint 推理方式不同**：
   - `fp_compare_1.cpp`（可视化）：每个通道单独调用一次单张推理，无需批量模式。3 通道 × 1 帧 = 3 次 forward。
   - `fp_compare_table3.cpp`（量化评估）：为提升效率，可将 3 个通道打包为批量张量 [3, H, W] 一次推理，参考 `SuperPointFeatureDetector::detectBatchForChannels()` 的实现。

---

## 使用流程

```bash
# 1. 提取数据（如果尚未从 2_filter_compare 获得）
cd paper_codes/3_fp_compare
python3 get_data_2.py --bags <bag1> <bag2> <bag3> --output data/ --k 1
python3 get_data_2_stream.py <bag1> <bag2> <bag3> data_stream/ --n 200
# 或直接软链接 2_filter_compare 已有数据：
# ln -s ../2_filter_compare/data_stream .

# 2. 编译
mkdir -p build && cd build && cmake .. && make -j$(nproc) && cd ..

# 3. 生成可视化对比图
./build/fp_compare_1 data/ figures/

# 4. 运行量化评估（三种光照各输出一个 CSV）
for d in data_stream/*/; do
  ./build/fp_compare_table3 "$d" "output/table3_$(basename $d).csv"
done

# 5. 绘制 Python 图表
python3 visualize_res.py
```
