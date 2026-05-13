# 滤波方法对比图生成 — 实现计划

## 目标

为论文生成 3 组对比图（对应 3 种光照条件），每组从 ROSbag 中随机抽取 K=10 帧，每帧生成一张 3行×5列 的对比图。

## 布局（每张图）

- **行 (3)**：P0 (DoP) | P1 (sinAoP) | P2 (cosAoP)
- **列 (5)**：原始(无滤波) | 中值滤波(3轮) | 导向滤波(以S0引导) | 双边滤波 | NLM

## 实现步骤

### 1. Python 脚本 `extract_random_frames.py`

从每个 ROSbag 中随机抽取 K=10 帧，保存为 PNG。

- 输入：ROSbag 文件路径列表 + 输出目录
- 读取 bag 中所有 `/image_raw` 消息的时间戳
- 随机抽取 K=10 个时间戳，提取对应帧
- 输出：`<输出目录>/<场景名>_<idx>.png`

### 2. C++ 程序 (`main.cpp` + `PolarChannel.h/cpp`)

独立程序，不依赖 ROS。复用 `paper_codes/1_vis_polar_channel/PolarChannel.h/cpp`。

**处理流程：**
1. 遍历输入目录下所有 `<场景名>_*.png`
2. 每张图调用 `raw2polar()` 解码出 S0/DoP/sinAoP/cosAoP
3. 对 DoP, sinAoP, cosAoP 三个偏振通道分别施加 4 种滤波（共 12 个滤波结果）：
   - **中值滤波**：`cv::medianBlur()`，kernel=3，连续 3 轮
   - **导向滤波**：`guidedFilterSingle()`，以 S0 为引导图，参数 `r=8, eps=0.0001`
   - **双边滤波**：`cv::bilateralFilter()`，参数 `d=9, sigmaColor=200, sigmaSpace=30`
   - **NLM**：`cv::fastNlMeansDenoising()`，参数 `h=50.0, templateWindowSize=5, searchWindowSize=21`
4. 拼图输出：
   - 5 列 = 原始 + 4 种滤波方法
   - 3 行 = P0(DoP) / P1(sinAoP) / P2(cosAoP)
   - 带行列标签
   - 保存为 `output/<场景名>_<idx>.png`

**CMakeLists.txt：** 与 1_vis_polar_channel 相同结构，链接 OpenCV。

### 3. 文件组织

```
paper_codes/
├── utilities/                  ← 共享的公共代码
│   ├── PolarChannel.h          ← 从 1_vis_polar_channel/ 移入
│   └── PolarChannel.cpp        ← 从 1_vis_polar_channel/ 移入
├── 1_vis_polar_channel/
│   ├── CMakeLists.txt          ← 更新 include path 指向 ../utilities/
│   ├── main.cpp                ← 更新 #include "PolarChannel.h" 路径
│   ├── PolarChannel.h          ← 删除（移入 utilities/）
│   ├── PolarChannel.cpp        ← 删除（移入 utilities/）
│   └── ...
└── 2_filter_compare/
    ├── plan_2.md               ← 本文件
    ├── extract_random_frames.py ← Python 从 ROSbag 随机抽帧
    ├── main.cpp                ← C++ 入口，读 PNG → 滤波 → 拼图
    ├── CMakeLists.txt          ← 编译配置，include ../utilities/
    ├── data/                   ← Python 脚本抽取的 PNG 存放位置
    └── output/                 ← 生成的对比图存放位置
```

**迁移步骤：**
1. 创建 `paper_codes/utilities/` 目录
2. 将 `paper_codes/1_vis_polar_channel/PolarChannel.h/cpp` 移动到 `paper_codes/utilities/`
3. 更新 `paper_codes/1_vis_polar_channel/CMakeLists.txt` 的 `target_include_directories` 路径
4. 更新 `paper_codes/1_vis_polar_channel/main.cpp` 的 `#include "PolarChannel.h"`

### 4. 使用流程

```bash
# Step 1: 从 ROSbag 随机抽帧
cd paper_codes/2_filter_compare
python3 extract_random_frames.py /path/to/bag1.bag /path/to/bag2.bag /path/to/bag3.bag data/

# Step 2: 编译
mkdir -p build && cd build
cmake ..
make

# Step 3: 运行
./build/filter_compare data/ output/
```

### 5. 表1 LaTeX 代码

程序输出后，我会额外生成一个 `table1_params.tex` 文件，包含所有滤波算法参数表格的 LaTeX 代码。

### 6. 验证

- 输出目录中有 30 张 PNG（3 组 × 10 张）
- 每张图 3 行 5 列，布局正确
- 肉眼确认滤波效果差异明显
