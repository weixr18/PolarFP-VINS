# 偏振通道滤波方法对比

对原始偏振图像解码出的 P0(DoP)、P1(sin(AoP))、P2(cos(AoP)) 三个通道，分别施加中值滤波、导向滤波（以 S0 为引导）、双边滤波、非局部均值去噪（NLM）四种算法，生成 3行×5列 对比图。

## 文件说明

| 文件 | 说明 |
|------|------|
| `get_data_2.py` | 从 ROSbag 中随机抽取 K 帧保存为 PNG |
| `filter_compare_main.cpp` | C++ 程序：读 PNG → 偏振解码 → 四种滤波 → 拼图输出 |
| `CMakeLists.txt` | CMake 编译配置，依赖 OpenCV |
| `table1_params.tex` | 论文表1：滤波算法参数 LaTeX 代码 |
| 共享的 `PolarChannel.h/cpp` | 位于 `../utilities/`，偏振通道解码模块 |

## 使用流程

### 1. 从 ROSbag 随机抽帧

```bash
python3 get_data_2.py \
  --bags ~/data/dark/0416/13-27-22/2026-04-16-13-27-22.bag \
         ~/data/dark/0416/13-54-30/2026-04-16-13-54-30.bag \
         ~/data/dark/0416/14-09-36/2026-04-16-14-09-36.bag \
  --output data/ --k 10
```

每组 bag 随机抽取 10 帧，保存为 `<场景名>_<序号>.png`。

### 2. 编译

```bash
mkdir -p build && cd build && cmake .. && make -j$(nproc)
cd ..
```

依赖：OpenCV（无需 ROS）。

### 3. 生成对比图

```bash
./build/filter_compare data/ output/
```

每张 PNG 输入生成一张 3行×5列 的对比图：

- **行**: P0 (DoP) | P1 (sin(AoP)) | P2 (cos(AoP))
- **列**: Raw | Median | Guided | Bilateral | NLM

## 滤波参数

| 算法 | 参数 | 取值 |
|------|------|------|
| 中值滤波 | 核大小 | 3×3（执行 3 轮） |
| 导向滤波 | 窗口半径 r / 正则化 ε | 8 / 0.0001（以 S0 引导） |
| 双边滤波 | 邻域直径 d / σ_color / σ_space | 9 / 200 / 30 |
| NLM | 滤波强度 h / 模板窗口 / 搜索窗口 | 50 / 5×5 / 21×21 |

详见 `table1_params.tex`。
