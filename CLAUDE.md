# CLAUDE.md — PolarFP-VINS

## Project Overview

Polarization-based Visual-Inertial Navigation System, forked from [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion). Core innovation: the VINS frontend decodes a raw polarization camera image into multiple channels (S0 intensity, DoP, sin(AoP), cos(AoP)), runs the **original VINS feature tracking pipeline independently on each channel**, then merges results for the backend. This provides robustness in low-light conditions where intensity-only tracking fails.

Built with **ROS catkin**, C++14/17.

## Key Differences from VINS-Fusion

### What was changed (V2 frontend, commits `663e403` → `1dde542`)

1. **PolarChannel module** (`PolarChannel.h/cpp`) — NEW. Decodes 2×2 micro-polarizer array raw image into S0, DoP, AoP channels. V2.1 adds optional bilateral/guided filtering.
2. **FeatureTracker multi-channel** (`feature_tracker.h/cpp`) — MODIFIED. Original `trackImage()` now branches: non-polar mode is identical to VINS-Fusion; polar mode calls `trackImagePolar()` which runs per-channel GFTT+LK optical flow+setMask, then merges results.
3. **Parameter loading** (`parameters.h/cpp`) — MODIFIED. Added `USE_POLAR`, `POLAR_CHANNELS`, `POLAR_FILTER_CFG`.
4. **Estimator integration** (`estimator.cpp`) — MODIFIED. Calls `featureTracker.setPolarChannels()` when polar mode is enabled.
5. **CMakeLists.txt** — MODIFIED. Added `PolarChannel.cpp` to `vins_lib` sources.

### What was NOT changed (intentionally)

- **All backend files** — estimator sliding window, Ceres factors (IMU preintegration, projection, marginalization), initialization (SFM, gyro bias, velocity/gravity) are untouched.
- **ROS node** — `rosNodeTest.cpp` is unchanged.
- **Camera calibration** — no changes to calibration parameters or camera models.
- Non-polar mode behavior is **bit-for-bit identical** to original VINS-Fusion.

## Build Commands

```bash
cd ~/ws/vi_catkin_ws
catkin_make
source devel/setup.bash
```

## Code Architecture

### Two catkin packages

- **`polarfp_camera_models`** — Camera model library (pinhole/fisheye/Scaramuzza etc.) and calibration tools. C++17, builds the `camera_models` library.
- **`vins_estimator`** — Main VINS estimator package. C++14, builds `vins_lib` library + `polarfp_vins_node` executable.

### vins_estimator key modules

| Directory | Responsibility |
|-----------|----------------|
| `featureTracker/` | Feature detection, optical flow tracking, **polarization channel decoding** (`PolarChannel.cpp`) |
| `estimator/` | Sliding-window nonlinear optimization (`estimator.cpp`), parameter loading, feature management |
| `factor/` | Ceres cost functions: IMU preintegration (`imu_factor.h`), projection errors, marginalization, local parameterization |
| `initial/` | System initialization: SFM, extrinsic rotation calibration, gyro bias alignment, velocity/gravity initialization |
| `utility/` | Visualization, utilities |

### Polarization channel pipeline (primary dev area)

#### 1. `PolarChannel.h/cpp` — Raw image decoding

- Samples 4 polarization angle sub-images (0°, 45°, 90°, 135°) from 2×2 superpixel offsets
- Computes Stokes parameters S0, S1, S2
- Exports DoP, AoP, sin(AoP), cos(AoP) as 8-bit images
- Key function: `raw2polar()`

#### 2. `feature_tracker.h/cpp` — Multi-channel feature tracking

- **Non-polar mode**: identical to VINS-Fusion (single grayscale image, GFTT + LK flow + setMask)
- **Polar mode**: `trackImage()` branches to `trackImagePolar()`:
  - Calls `raw2polar()` to decode raw image into 4 channels
  - For each enabled channel, independently runs the full VINS pipeline (LK optical flow, backward check, setMask spatial distribution, GFTT new feature detection, ID assignment via global `n_id`)
  - Merges all channels' results into the standard VINS output format
  - Backend receives features without knowing which channel they came from

#### 3. `ChannelState` structure

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
- `use_polar` (1 = enable polarization channels, 0 = VINS-Fusion behavior)
- `polar_channels` (e.g. `"s0,dop,aopsin,aopcos"`)
- `num_of_cam`, `imu`, `max_cnt`, `freq`, etc.

### Image filtering (polarization channels, V2.1)

The `raw2polar()` pipeline supports configurable denoising filters applied to DoP/sin/cos channels after 8-bit quantization. This improves GFTT feature detection stability under low-light conditions.

**Architecture (4 layers):**

1. **Config struct** — `PolarFilterConfig` in `PolarChannel.h`:
   ```cpp
   enum PolarFilterType { FILTER_NONE = 0, FILTER_BILATERAL = 1, FILTER_GUIDED = 2 };
   struct PolarFilterConfig {
       PolarFilterType filter_type = FILTER_NONE;
       // Bilateral params
       int bilateral_d = 9;
       double bilateral_sigmaColor = 200;
       double bilateral_sigmaSpace = 30;
       // Guided filter params
       int guided_radius = 4;
       double guided_eps = 0.01;
   };
   ```

2. **Filter application** — Inside `raw2polar()` in `PolarChannel.cpp`, gated by `cfg.filter_type`:
   - `FILTER_BILATERAL`: calls `cv::bilateralFilter()` on dop_img/sin_img/cos_img
   - `FILTER_GUIDED`: converts to CV_64F, uses S0 as guidance image, calls `guidedFilterSingle()` (box-filter based implementation, no ximgproc dependency), converts back to CV_8U
   - `FILTER_NONE`: pass-through (default)

3. **Parameter loading** — `parameters.cpp::readParameters()` reads YAML into global `POLAR_FILTER_CFG`:
   - `polar_filter_type`: 0=none, 1=bilateral, 2=guided
   - `polar_bilateral_d`, `polar_bilateral_sigma_color`, `polar_bilateral_sigma_space`
   - `polar_guided_radius`, `polar_guided_eps`

4. **Call chain** — `readParameters()` → `POLAR_FILTER_CFG` → `FeatureTracker::setPolarFilterConfig()` → `FeatureTracker::polar_filter_cfg` → `raw2polar(cur_img, polar_filter_cfg)` in `trackImagePolar()`

**Test nodes** in `vins_estimator/src/test/`:
- `test_filter.cpp`: Interactive comparison of guided/bilateral/NLM with keyboard controls (`w` to switch, `+/−` to adjust radius, `[]` to adjust eps)
- `test_bilateral_filter.cpp`: Compares three bilateral filter pipeline stages

**File dependencies:**
```
YAML config → parameters.cpp (read YAML) → parameters.h (extern POLAR_FILTER_CFG)
→ feature_tracker.h/cpp (setPolarFilterConfig, trackImagePolar)
→ PolarChannel.h/cpp (PolarFilterConfig struct, raw2polar, guidedFilterSingle)
```

## VINS-Fusion comparison summary

| Dimension | VINS-Fusion | PolarFP-VINS |
|-----------|-------------|--------------|
| Input | Grayscale image | Raw polarization image (2×2 micro-polarizer array) |
| Frontend channels | 1 (intensity) | 4 (S0, DoP, sin(AoP), cos(AoP)), independently tracked |
| Feature detection | GFTT | GFTT per channel (same algorithm, different input) |
| Tracking | LK optical flow + backward check | LK optical flow + backward check per channel |
| Spatial distribution | setMask (circular mask) | setMask per channel |
| Backend | Sliding window BA + IMU preintegration | **Identical** — backend is unchanged |
| Stereo support | Yes (mono/stereo) | Mono only (stereo removed in current branch) |
| Low-light robustness | Degrades with intensity | Improved via DoP/AoP channels |
| Filter pipeline | None | Optional bilateral/guided filter on polar channels |

## Tech Stack

- C++14 (vins_estimator) / C++17 (polarfp_camera_models)
- ROS (roscpp, cv_bridge, image_transport, tf, std_msgs, geometry_msgs, nav_msgs)
- OpenCV 4
- Ceres Solver
- Eigen3

## Workflow

- **Compilation**: User compiles manually and reviews the full output. After code modifications, just report what was changed — do NOT run compile commands unless the user explicitly asks or pastes an error for you to fix.

## Notes

- This is a **catkin workspace**, not a standalone CMake project
- Compile command: `cd ~/ws/vi_catkin_ws && catkin_make`, DO NOT use `catkin build`
- `vins_lib` is a library target linked by `polarfp_vins_node`
- Two VS Code workspace files exist: `VINS-Fusion.code-workspace` and `PolarFP-VINS.code-workspace`
- Current state: V2.1 — multi-polar channel frontend with optional image filtering (commit `e304552`)
- V1.1 (`polarfp-v1.1` branch) was a full frontend rewrite that performed worse than baseline — **design principle: stay close to VINS-Fusion's proven pipeline**
- Detailed reconstruction docs in `vins_estimator/docs/`
