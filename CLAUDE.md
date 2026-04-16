# CLAUDE.md — PolarFP-VINS

## Project Overview

Polarization-based Visual-Inertial Navigation System, forked from [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion). Core innovation: the VINS frontend decodes a raw polarization camera image into multiple channels (S0 intensity, DoP, sin(AoP), cos(AoP)), runs the **original VINS feature tracking pipeline independently on each channel**, then merges results for the backend. This provides robustness in low-light conditions where intensity-only tracking fails.

Built with **ROS catkin**, C++14/17. SuperPoint detection requires LibTorch.

## Key Differences from VINS-Fusion

### What was changed (V2 frontend through SuperPoint integration)

1. **PolarChannel module** (`PolarChannel.h/cpp`) — NEW. Decodes 2×2 micro-polarizer array raw image into S0, DoP, AoP channels. V2.1 adds optional bilateral/guided/NLM filtering.
2. **FeatureTracker multi-channel** (`feature_tracker.h/cpp`) — MODIFIED. `trackImage()` branches to polar mode which runs per-channel tracking then merges results. Non-polar mode returns empty (original VINS-Fusion path removed during V2 reconstruction).
3. **FeatureDetector abstraction** (`feature_tracker_detector.h/cpp`) — NEW. Abstract `FeatureDetector` interface with three implementations: `GFTTDetector`, `FASTDetector`, `SuperPointFeatureDetector`.
4. **FeatureMatcher abstraction** (`feature_tracker_matcher.h/cpp`) — NEW. Abstract `FeatureMatcher` interface with two implementations: `LKFlowMatcher`, `BRIEFFLANNMatcher`.
5. **SuperPoint integration** (`superpoint_detector.h/cpp`, `nn/`) — NEW. LibTorch-based keypoint detection with PIMPL pattern, batch multi-channel inference.
6. **Parameter loading** (`parameters.h/cpp`) — MODIFIED. Added `USE_POLAR`, `POLAR_CHANNELS`, `POLAR_FILTER_CFG`, detector/matcher type selection, SuperPoint parameters.
7. **Estimator integration** (`estimator.cpp`) — MODIFIED. Calls `featureTracker.setPolarChannels()`, `featureTracker.setPolarFilterConfig()`, `featureTracker.initDetectorAndMatcher()`.
8. **CMakeLists.txt** — MODIFIED. Added `PolarChannel.cpp`, optional `superpoint_detector.cpp` via `USE_SUPERPOINT` option with conditional LibTorch linking.

### What was NOT changed (intentionally)

- **All backend files** — estimator sliding window, Ceres factors (IMU preintegration, projection, marginalization), initialization (SFM, gyro bias, velocity/gravity) are untouched.
- **ROS node** — `rosNodeTest.cpp` is unchanged.
- **Camera calibration** — no changes to calibration parameters or camera models.

## Build Commands

```bash
cd ~/ws/vi_catkin_ws
catkin_make
source devel/setup.bash
```

SuperPoint requires LibTorch at `/home/dhz/SLAM/libs/libtorch-cxx11-abi-shared-with-deps-1.12.1+cu113/libtorch`. CMake option `USE_SUPERPOINT` (default ON) gates compilation.

## Code Architecture

### Two catkin packages

- **`polarfp_camera_models`** — Camera model library (pinhole/fisheye/Scaramuzza etc.) and calibration tools. C++17, builds the `camera_models` library.
- **`vins_estimator`** — Main VINS estimator package. C++14/17, builds `vins_lib` library + `polarfp_vins_node` executable.

### vins_estimator key modules

| Directory | Responsibility |
|-----------|----------------|
| `featureTracker/` | Feature detection (GFTT/FAST/SuperPoint), matching (LK/BRIEF+FLANN), **polarization channel decoding** (`PolarChannel.cpp`) |
| `estimator/` | Sliding-window nonlinear optimization (`estimator.cpp`), parameter loading, feature management |
| `factor/` | Ceres cost functions: IMU preintegration (`imu_factor.h`), projection errors, marginalization, local parameterization |
| `initial/` | System initialization: SFM, extrinsic rotation calibration, gyro bias alignment, velocity/gravity initialization |
| `utility/` | Visualization, utilities |

### Polarization channel pipeline (primary dev area)

#### 1. `PolarChannel.h/cpp` — Raw image decoding

- Samples 4 polarization angle sub-images (0°, 45°, 90°, 135°) from 2×2 superpixel offsets
- Computes Stokes parameters S0, S1, S2
- Exports DoP, AoP, sin(AoP), cos(AoP) as 8-bit images
- Supports configurable denoising: FILTER_NONE, FILTER_BILATERAL, FILTER_GUIDED (custom box-filter implementation), FILTER_NLM
- Key function: `raw2polar()`

#### 2. `feature_tracker.h/cpp` — Multi-channel feature tracking

- **Polar mode**: `trackImage()` calls `trackImagePolar()`:
  - Phase 1 (per-channel): `raw2polar()` → matcher track → border check → setMask
  - Phase 2 (batch): if SuperPoint, collects all channel images, calls `detectBatchForChannels()` for single LibTorch forward pass
  - Phase 3 (per-channel): detector detect (SuperPoint returns cached results, GFTT/FAST direct inference) → ID assignment → descriptor extraction → undistortion → velocity
  - Merges all channels' results into the standard VINS output format
  - Backend receives features without knowing which channel they came from
- **Non-polar mode**: returns empty `featureFrame` (original VINS-Fusion tracking path removed during V2 reconstruction)

#### 3. `feature_tracker_detector.h/cpp` — Feature detector abstraction

- Abstract interface: `detect(image, mask, max_cnt)`
- **`GFTTDetector`**: wraps `cv::goodFeaturesToTrack` (original VINS-Fusion default)
- **`FASTDetector`**: `cv::FAST` + response sort + mask filter + top-N truncate
- **`SuperPointFeatureDetector`**: LibTorch inference with PIMPL, supports batch mode for polar channels
- Factory: `createDetector(DetectorConfig)`

#### 4. `feature_tracker_matcher.h/cpp` — Feature matcher abstraction

- Abstract interface: `track()`, `extractDescriptors()`
- **`LKFlowMatcher`**: Lucas-Kanade optical flow with optional backward check (original VINS-Fusion default)
- **`BRIEFFLANNMatcher`**: BRIEF/ORB binary descriptors + FLANN LSH index for Hamming distance matching with ratio test. Requires `opencv_contrib` for BRIEF; falls back to ORB.

#### 5. `ChannelState` structure

Each enabled polar channel maintains its own copy of VINS tracking state:
```cpp
struct ChannelState {
    std::string name;              // "s0", "dop", "aopsin", "aopcos"
    cv::Mat prev_img, cur_img;     // per-channel images
    vector<cv::Point2f> prev_pts, cur_pts;  // feature coordinates
    vector<int> ids;               // feature IDs (global n_id shared across channels)
    vector<int> track_cnt;         // tracking count
    // ... plus undistorted points, velocity, maps, mask
};
```

### Configuration

YAML configs in `config/`. Key fields:
- `use_polar` (1 = enable polarization channels, 0 = returns empty featureFrame — non-polar VINS-Fusion path removed)
- `polar_channels` (e.g. `"s0,dop,aopsin,aopcos"`)
- `feature_detector_type` (0=GFTT, 1=FAST, 2=SUPERPOINT)
- `feature_matcher_type` (0=LK_FLOW, 1=BRIEF_FLANN)
- `superpoint_model_path`, `superpoint_use_gpu`, `superpoint_keypoint_threshold`, `superpoint_nms_radius`
- `polar_filter_type` (0=none, 1=bilateral, 2=guided, 3=NLM)
- `num_of_cam`, `imu`, `max_cnt`, `freq`, etc.

### Image filtering (polarization channels, V2.1)

The `raw2polar()` pipeline supports configurable denoising filters applied to DoP/sin/cos channels after 8-bit quantization. This improves feature detection stability under low-light conditions.

**Architecture (4 layers):**

1. **Config struct** — `PolarFilterConfig` in `PolarChannel.h`:
   ```cpp
   enum PolarFilterType { FILTER_NONE = 0, FILTER_BILATERAL = 1, FILTER_GUIDED = 2, FILTER_NLM = 3 };
   struct PolarFilterConfig {
       PolarFilterType filter_type = FILTER_NONE;
       // Bilateral params
       int bilateral_d = 9;
       double bilateral_sigmaColor = 200;
       double bilateral_sigmaSpace = 30;
       // Guided filter params
       int guided_radius = 4;
       double guided_eps = 0.01;
       // NLM params
       double nlm_h = 10;
       int nlm_template = 7;
       int nlm_search = 21;
   };
   ```

2. **Filter application** — Inside `raw2polar()` in `PolarChannel.cpp`, gated by `cfg.filter_type`:
   - `FILTER_BILATERAL`: calls `cv::bilateralFilter()` on dop_img/sin_img/cos_img
   - `FILTER_GUIDED`: converts to CV_64F, uses S0 as guidance image, calls `guidedFilterSingle()` (box-filter based implementation, no ximgproc dependency), converts back to CV_8U
   - `FILTER_NLM`: calls `cv::fastNlMeansDenoising()` on dop_img/sin_img/cos_img
   - `FILTER_NONE`: pass-through (default)

3. **Parameter loading** — `parameters.cpp::readParameters()` reads YAML into global `POLAR_FILTER_CFG`

4. **Call chain** — `readParameters()` → `POLAR_FILTER_CFG` → `FeatureTracker::setPolarFilterConfig()` → `FeatureTracker::polar_filter_cfg` → `raw2polar(cur_img, polar_filter_cfg)` in `trackImagePolar()`

**Test nodes** in `vins_estimator/src/test/`:
- `test_filter.cpp`: Interactive comparison of guided/bilateral/NLM with keyboard controls
- `test_bilateral_filter.cpp`: Compares three bilateral filter pipeline stages
- `test_nlm_filter.cpp`: NLM filter testing

### SuperPoint integration

**Architecture:**
- `SuperPointFeatureDetector` extends `FeatureDetector` via PIMPL (forward-declared `Impl` struct in header, full definition in `.cpp`)
- `<torch/script.h>` is isolated in `superpoint_detector.cpp`, avoiding `c10::nullopt` vs `std::nullopt` conflicts in other headers
- CMake option `USE_SUPERPOINT` (default ON) controls:
  - Whether LibTorch is found at hardcoded path
  - Whether `superpoint_detector.cpp` is compiled via `target_sources()`
  - Whether `TORCH_LIBRARIES` is linked
  - Whether `-DUSE_SUPERPOINT` preprocessor define is set

**Batch inference for polar channels:**
- `detectBatchForChannels(images)` batches N channel images into single LibTorch tensor [N, 1, H, W]
- Single forward pass through SuperPoint network
- Per-channel post-processing: softmax over 65 classes (descriptor + confidence), NMS at 1/8 resolution, threshold filtering, mask filtering, max_cnt truncation
- Results cached per channel; subsequent `detect()` calls return from cache
- Model loaded from TorchScript `.pt` file in `nn/` directory

**File dependencies:**
```
YAML config → parameters.cpp (read YAML) → parameters.h (extern SUPERPOINT_* vars)
→ feature_tracker.cpp (initDetectorAndMatcher, two-phase channel loop)
→ feature_tracker_detector.h/cpp (DetectorConfig, createDetector, SUPERPOINT case)
→ superpoint_detector.h/cpp (SuperPointFeatureDetector, PIMPL, batch inference)
→ nn/superpoint_v1.pt (TorchScript model)
```

## VINS-Fusion comparison summary

| Dimension | VINS-Fusion | PolarFP-VINS |
|-----------|-------------|--------------|
| Input | Grayscale image | Raw polarization image (2×2 micro-polarizer array) |
| Frontend channels | 1 (intensity) | 4 (S0, DoP, sin(AoP), cos(AoP)), independently tracked |
| Feature detection | GFTT | GFTT / FAST / SuperPoint (configurable) |
| Tracking | LK optical flow + backward check | LK flow or BRIEF+FLANN (configurable) per channel |
| Spatial distribution | setMask (circular mask) | setMask per channel |
| Backend | Sliding window BA + IMU preintegration | **Identical** — backend is unchanged |
| Stereo support | Yes (mono/stereo) | Mono only (stereo removed in current branch) |
| Low-light robustness | Degrades with intensity | Improved via DoP/AoP channels |
| Filter pipeline | None | Optional bilateral/guided/NLM filter on polar channels |
| Neural detection | No | SuperPoint via LibTorch (optional) |

## Tech Stack

- C++14/17 (vins_estimator) / C++17 (polarfp_camera_models)
- ROS (roscpp, cv_bridge, image_transport, tf, std_msgs, geometry_msgs, nav_msgs)
- OpenCV 4
- Ceres Solver
- Eigen3
- LibTorch (optional, for SuperPoint)

## Workflow

- **Compilation**: User compiles manually and reviews the full output. After code modifications, just report what was changed — do NOT run compile commands unless the user explicitly asks or pastes an error for you to fix.

## Notes

- This is a **catkin workspace**, not a standalone CMake project
- Compile command: `cd ~/ws/vi_catkin_ws && catkin_make`, DO NOT use `catkin build`
- `vins_lib` is a library target linked by `polarfp_vins_node`
- VS Code workspace file: `PolarFP-VINS.code-workspace`
- Current state: V2.2+ — modular frontend (detector/matcher abstraction) with SuperPoint integration
- V1.1 (`polarfp-v1.1` branch) was a full frontend rewrite that performed worse than baseline — **design principle: stay close to VINS-Fusion's proven pipeline**
- Python + PyTorch testing: use conda env `torch1.12` at `/home/dhz/anaconda3/envs/torch1.12`
- Detailed reconstruction docs in `vins_estimator/docs/`
