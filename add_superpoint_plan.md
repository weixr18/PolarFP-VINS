# SuperPoint 迁移计划：集成到模块化前端

## Context

feat/superpoint 分支之前实现了一个独立的 `PolarFeatureTracker` 类，将 SuperPoint 作为三选一检测器集成其中。**旧分支有一个设计缺陷**：SuperPoint 仅在 S0 通道推理一次，结果缓存后复用于所有通道。这导致 DoP/AoP 通道的独特纹理无法产生额外特征点，失去了多通道的意义。

当前 master 分支是重构后的模块化前端：`FeatureDetector` 抽象接口 + 工厂模式 + `FeatureMatcher` 抽象接口。GFTT/FAST 已经是每个通道独立检测。**SuperPoint 应该与它们行为一致** — 四个通道各自独立检测，像四个重叠的相机一样互补特征。

## 架构设计总览

```
DetectorType enum: { GFTT, FAST, SUPERPOINT }
                                        |
createDetector() ───────────────────────┤
        ├── GFTTDetector       (existing, per-channel detect)
        ├── FASTDetector       (existing, per-channel detect)
        └── SuperPointFeatureDetector   (NEW, PIMPL + FeatureDetector + batch)
                └── SuperPointDetectorImpl  (PIMPL: LibTorch model)
                        └── detectBatchForChannels()
                            → batch=N 一次推理 → 按通道 mask/max_cnt → 缓存
                            → 逐通道 detect() 从缓存取出
```

**核心思路**：
- `trackImage()` 通道循环拆为两阶段：2a-2d（per-channel 跟踪） + 2e（检测分支） + 2f-2i（per-channel 后续处理）
- GFTT/FAST：2e 保持逐通道 `detect()`
- SuperPoint：2e 收集所有通道图像，调用 `detectBatch()` 一次推理 batch=N，按通道拆分 + 各自 mask 过滤 + max_cnt 截断
- **三种 Detector 对外行为一致**：`trackImage()` 看到的只是 `detector_->detect()` 返回 `vector<Point2f>`

---

## Step 1: 新建文件

### 1.1 `vins_estimator/src/featureTracker/superpoint_detector.h`

PIMPL 模式，**一个类**同时完成 LibTorch 封装和 `FeatureDetector` 适配。`SuperPointKeypoint` struct 不暴露，放在 `.cpp` 内部。

```cpp
// --- 前向声明（PIMPL） ---
struct SuperPointDetectorImpl;

// --- 单一类：LibTorch + FeatureDetector ---
class SuperPointFeatureDetector : public FeatureDetector {
public:
    SuperPointFeatureDetector(const std::string& model_path, bool use_gpu,
                              float kp_thresh, int nms_radius);
    ~SuperPointFeatureDetector();

    std::string name() const override { return "SuperPoint"; }

    // 与 GFTT/FAST 完全一致的接口
    // batch 模式下：从 detectBatchForChannels 预计算的结果中按序取出
    // 非 batch 模式（回退）：单图推理 + mask 过滤 + max_cnt 截断
    std::vector<cv::Point2f> detect(const cv::Mat& image, const cv::Mat& mask,
                                    int max_cnt) const override;

    // trackImage() 调用此方法：所有通道图像一次性 batch 推理
    // 内部存储拆分后的结果，后续逐通道 detect() 从中取出
    void detectBatchForChannels(
        const std::vector<cv::Mat>& images,
        const std::vector<cv::Mat>& masks,
        const std::vector<int>& max_cnts);

private:
    // PIMPL: 隔离 <torch/script.h> 避免 c10::nullopt vs std::nullopt 冲突
    std::unique_ptr<SuperPointDetectorImpl> impl_;
    bool initialized_ = false;

    // batch 推理结果缓存
    mutable bool batch_cached_ = false;
    mutable std::vector<std::vector<cv::Point2f>> batch_results_;
    mutable size_t current_channel_idx_ = 0;

    // 配置参数
    float keypoint_threshold_;
    int nms_radius_;
};
```

### 1.2 `vins_estimator/src/featureTracker/superpoint_detector.cpp`

从旧分支 `superpoint_detector.cpp` 移植核心推理代码，合并到一个类中。

**内部 struct（仅 cpp 使用）**：
```cpp
struct SuperPointDetectorImpl {
    torch::jit::script::Module model;
    torch::Device device;
    SuperPointDetectorImpl(torch::jit::script::Module m, torch::Device d)
        : model(std::move(m)), device(std::move(d)) {}
};

struct SuperPointKeypoint { float x, y, score; };  // 不暴露到 header
```

**`initialize()`**（构造函数中调用）：同旧分支，加载 TorchScript 模型 + 设置设备 + GPU 预热。

**`detectBatchForChannels()` 实现**：
```cpp
void SuperPointFeatureDetector::detectBatchForChannels(
    const std::vector<cv::Mat>& images,
    const std::vector<cv::Mat>& masks,
    const std::vector<int>& max_cnts)
{
    batch_cached_ = false;

    // 1. Pack images into batch tensor [N, 1, H, W]
    int n = images.size();
    int h = images[0].rows, w = images[0].cols;
    auto batch = torch::empty({n, 1, h, w}, torch::kFloat32);
    for (int i = 0; i < n; i++) {
        cv::Mat img_f;
        images[i].convertTo(img_f, CV_32FC1, 1.0 / 255.0);
        auto img_tensor = torch::from_blob(img_f.data, {1, 1, h, w}, torch::kFloat32);
        batch[i] = img_tensor;
    }

    // 2. Forward → (detection_maps, descriptor_maps)
    auto output = impl_->model.forward({batch.to(impl_->device)});
    auto det_maps = output.toTuple()->elements()[0].toTensor().cpu();  // [N, 65, H/8, W/8]

    // 3. Per-image post-processing: softmax → NMS → threshold
    std::vector<std::vector<SuperPointKeypoint>> all_kpts(n);
    for (int i = 0; i < n; i++) {
        auto det = det_maps[i];
        det = torch::softmax(det, 0);
        auto det_no_dust = det.slice(0, 0, 64);
        auto max_vals = std::get<0>(det_no_dust.max(0));
        auto max_indices = std::get<1>(det_no_dust.max(0));
        // NMS + threshold (同旧分支逻辑)
        // → 填充 all_kpts[i]
    }

    // 4. Per-channel: mask 过滤 + max_cnt 截断 → 存为 batch_results_
    batch_results_.resize(n);
    for (size_t i = 0; i < n; i++) {
        std::vector<cv::Point2f> pts;
        for (const auto& kp : all_kpts[i]) {
            int x = cvRound(kp.x), y = cvRound(kp.y);
            if (!masks[i].empty() && masks[i].at<uchar>(y, x) == 0) continue;
            if (x < 0 || x >= images[i].cols || y < 0 || y >= images[i].rows) continue;
            pts.emplace_back(kp.x, kp.y);
        }
        if (max_cnts[i] > 0 && static_cast<int>(pts.size()) > max_cnts[i])
            pts.resize(max_cnts[i]);
        batch_results_[i] = std::move(pts);
    }
    batch_cached_ = true;
    current_channel_idx_ = 0;
}
```

**`detect()` 实现**（与 GFTT/FAST 接口一致）：
```cpp
std::vector<cv::Point2f> SuperPointFeatureDetector::detect(
    const cv::Mat& image, const cv::Mat& mask, int max_cnt) const
{
    if (batch_cached_ && current_channel_idx_ < batch_results_.size()) {
        return batch_results_[current_channel_idx_++];
    }
    // 未 batch 预推理时回退：单图推理（兼容非 polar 模式）
    auto kpts = detectKeypointsInternal(image);  // 内部函数
    // ... mask 过滤 + max_cnt 截断 ...
    return pts;
}
```

---

## Step 2: 修改 `feature_tracker.cpp` — 通道循环拆分

**核心改动**：`trackImage()` 的步骤 2e（新特征检测）需要区分 SuperPoint 和 GFTT/FAST。

将 for 循环拆为两阶段：

```cpp
// =============================================
// 2. 每通道跟踪 (步骤 2a-2d: 跟踪/边界/track_cnt/mask)
// =============================================
for (auto& ch : channels) {
    // 2a-2d: 与现有逻辑完全一致
    ...
}

// =============================================
// 2e. 新特征检测（分 SuperPoint 和 GFTT/FAST 两条路径）
// =============================================
if (detector_->name() == "SuperPoint") {
    // 收集所有通道的图像 + mask + max_cnt
    std::vector<cv::Mat> batch_imgs, batch_masks;
    std::vector<int> batch_max_cnts;
    for (const auto& ch : channels) {
        if (ch.cur_img.empty()) continue;
        batch_imgs.push_back(ch.cur_img);
        batch_masks.push_back(ch.mask);
        batch_max_cnts.push_back(MAX_CNT - static_cast<int>(ch.cur_pts.size()));
    }
    // Batch 推理一次，结果存到 detector 内部
    auto* sp_det = dynamic_cast<SuperPointFeatureDetector*>(detector_.get());
    if (sp_det) {
        sp_det->detectBatchForChannels(batch_imgs, batch_masks, batch_max_cnts);
    }
}

// 逐通道提取（统一入口：SuperPoint 从 batch 缓存取，GFTT/FAST 直接 detect）
for (auto& ch : channels) {
    if (ch.cur_img.empty()) continue;

    int n_max_cnt = MAX_CNT - static_cast<int>(ch.cur_pts.size());
    if (n_max_cnt > 0 && !ch.mask.empty()) {
        ch.n_pts = detector_->detect(ch.cur_img, ch.mask, n_max_cnt);
    } else {
        ch.n_pts.clear();
    }

    // 2f-2i: 与现有逻辑完全一致（分配 ID、描述子、归一化、速度、更新 prev）
    ...
}
```

**设计要点**：
- 对 GFTT/FAST：`detectBatchForChannels` 不存在，逐通道直接 `detect()`
- 对 SuperPoint：第一次 `detectBatchForChannels()` 做完整 batch 推理，后续逐通道 `detect()` 只是从内部缓存取对应通道的结果
- **2f-2i 完全不变**，保持代码最小改动

---

## Step 3: 修改现有文件

### 3.1 `vins_estimator/src/featureTracker/feature_tracker_detector.h`

```cpp
enum class DetectorType { GFTT, FAST, SUPERPOINT };

struct DetectorConfig {
    // ... existing fields ...
    // SuperPoint params
    std::string sp_model_path;
    bool sp_use_gpu = true;
    float sp_keypoint_threshold = 0.015f;
    int sp_nms_radius = 4;
};
```

### 3.2 `vins_estimator/src/featureTracker/feature_tracker_detector.cpp`

1. `#include "superpoint_detector.h"`
2. `createDetector()` 添加 SUPERPOINT case

### 3.3 `vins_estimator/src/estimator/parameters.h`

```cpp
extern std::string SUPERPOINT_MODEL_PATH;
extern int SUPERPOINT_USE_GPU;
extern float SUPERPOINT_KEYPOINT_THRESHOLD;
extern int SUPERPOINT_NMS_RADIUS;
```

### 3.4 `vins_estimator/src/estimator/parameters.cpp`

- 变量定义 + YAML 读取 + 相对路径解析 + 日志（同之前方案）

### 3.5 `vins_estimator/src/featureTracker/feature_tracker.cpp` — `initDetectorAndMatcher()`

添加 SuperPoint 配置分支（同之前方案）。

---

## Step 4: CMakeLists.txt 修改

同之前方案：C++14→17、可选 LibTorch（`LIBTORCH_DIR` 环境变量）、条件编译链接。

---

## Step 5: 模型文件

从 `feat/superpoint` 复制：
| 文件 | 必需? |
|------|-------|
| `vins_estimator/nn/superpoint_v1.pt` | 必需 |
| `vins_estimator/nn/superpoint_v1.pth` | 可选 |
| `vins_estimator/nn/convert_to_torchscript.py` | 可选 |
| `vins_estimator/src/featureTracker/test_superpoint.cpp` | 可选 |

**不需要** `nanoflann.hpp`（属于旧 tracker 的 KD-tree 匹配）。

---

## Step 6: 验证计划

### 6.1 编译验证
```bash
export LIBTORCH_DIR=/home/dhz/SLAM/libs/libtorch-cxx11-abi-shared-with-deps-1.12.1+cu113/libtorch
catkin_make
```

### 6.2 回归测试
- `feature_detector_type: 0` → GFTT + LK
- `feature_detector_type: 1` → FAST + LK

### 6.3 SuperPoint + LK 集成测试
```yaml
feature_detector_type: 2
feature_matcher_type: 0
```
预期输出：
```
[PolarFP] Detector: SuperPoint, Matcher: LK_FLOW
[SuperPoint] CUDA available (1 device(s)), running on GPU
[SuperPoint] Model loaded from .../superpoint_v1.pt
[SuperPoint] Warm-up inference done
```

### 6.4 多通道独立检测验证
临时在 `detectBatchForChannels()` 中打印每个通道检测到的特征点数：
```
[SuperPoint] batch=4 inference done: s0=120, dop=85, aopsin=45, aopcos=62
```
预期：各通道检测点数不同，证明是独立检测而非缓存复用。

### 6.5 组合测试矩阵
| 检测器 | 匹配器 | 预期 |
|--------|--------|------|
| SuperPoint | LK_FLOW | 四通道独立检测 + LK 跟踪 |
| SuperPoint | BRIEF_FLANN | 四通道独立检测 + BRIEF 描述子提取 + FLANN |
| GFTT/FAST | LK_FLOW | 回归 |

### 6.6 CPU 回退
`superpoint_use_gpu: 0` → CPU 推理，慢但可用。

### 6.7 无 SuperPoint 编译
`catkin_make -DUSE_SUPERPOINT=OFF` → 无 LibTorch 依赖。

---

## 文件变更清单

### 新建文件（2 个）
| 文件 | 说明 |
|------|------|
| `vins_estimator/src/featureTracker/superpoint_detector.h` | PIMPL + FeatureDetector 适配器 + batch 推理接口 |
| `vins_estimator/src/featureTracker/superpoint_detector.cpp` | LibTorch batch 推理 + mask/max_cnt 过滤 |

### 修改文件（5 个）
| 文件 | 变更 |
|------|------|
| `vins_estimator/src/featureTracker/feature_tracker_detector.h` | 枚举添加 SUPERPOINT，Config 添加 SP 字段 |
| `vins_estimator/src/featureTracker/feature_tracker_detector.cpp` | 工厂函数添加 SP case，include 头文件 |
| `vins_estimator/src/featureTracker/feature_tracker.cpp` | **trackImage 拆为两阶段循环**（2a-2d + 2e 分支 + 2f-2i），initDetectorAndMatcher 添加 SP 配置 |
| `vins_estimator/src/estimator/parameters.h` | SuperPoint 全局变量 |
| `vins_estimator/src/estimator/parameters.cpp` | 变量定义、YAML 读取、路径解析 |
| `vins_estimator/CMakeLists.txt` | C++17、可选 LibTorch、条件编译链接 |
