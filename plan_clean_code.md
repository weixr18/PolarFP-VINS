# featureTracker 模块死代码清理计划

## 背景

对 `src/featureTracker/` 下所有 .h/.cpp 文件逐函数检查，发现 10 个完全未被调用的函数和 1 个未使用的常量。本计划描述如何安全清理它们。

---

## 待清理项

### feature_tracker.h / feature_tracker.cpp

| # | 函数 | 声明:行 | 定义:行 | 清理动作 |
|---|------|---------|---------|----------|
| 1 | `setMask()` | .h:160 | .cpp:89 | 删除声明和定义 |
| 2 | `showUndistortion()` | .h:169 | .cpp:768 | 删除声明和定义 |
| 3 | `rejectWithF()` | .h:177 | .cpp:706 | 删除声明和定义 |
| 4 | `undistortedPoints()` | .h:180 | 无定义 | 仅删除声明 |
| 5 | `ptsVelocity()` | .h:198 | .cpp:840 | 删除声明和定义 |
| 6 | `showTwoImage()` | .h:208 | 无定义 | 仅删除声明 |
| 7 | `drawTrack()` | .h:224 | .cpp:894 | 删除声明和定义 |
| 8 | `distance()` | .h:231 | .cpp:121 | 删除声明和定义 |
| 9 | `inBorder()` 成员函数 | .h:243 | .cpp:25 | 删除声明和定义 |

### PolarChannel.h / PolarChannel.cpp

| # | 函数/常量 | 声明:行 | 定义:行 | 清理动作 |
|---|-----------|---------|---------|----------|
| 10 | `_calculatePercentile()` | .h:47 | .cpp:68 | 删除声明和定义 |
| 11 | `DOP_PERCENTILE` | .h:39 | — | 删除常量声明 |

---

## 确认安全的函数（不删除）

以下函数曾被怀疑但未使用，实际均被调用：

- `undistortedPts()` — 在 `trackImage()` 内调用
- `drawTrackPolar()` — 在 `trackImage()` 内调用
- `inBorderImpl()` — 在 `trackImage()` 内调用（替代了废弃的 `inBorder()`）
- `setMaskForChannel()` — 在 `trackImage()` 内调用（替代了废弃的 `setMask()`）
- `extractDescriptors()` 基类实现 — 虽返回空，但被 `BRIEFFLANNMatcher` 虚函数覆盖，通过多态调用

---

## 验证

1. 修改后执行 `catkin_make` 确认编译通过
2. 运行 VINS 节点确认功能无回归
