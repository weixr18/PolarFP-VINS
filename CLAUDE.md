# CLAUDE.md — PolarFP-VINS

## Project Overview

Polarization-based Visual-Inertial Navigation System, forked from [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion). Core innovation: the VINS frontend uses multi-channel polarization information (S0, DoP, sin(AoP), cos(AoP)) decoded from a raw polarization camera for feature tracking, providing robustness in low-light conditions where intensity-only tracking fails.

Built with **ROS catkin**, C++14/17.

## Build Commands

```bash
cd ~/ws/vi_catkin_ws
catkin build polarfp_vins polarfp_camera_models   # or catkin_make
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

1. **`PolarChannel.h/cpp`** — Decodes 2×2 micro-polarizer array raw image:
   - Samples 4 polarization angle sub-images (0°, 45°, 90°, 135°) via superpixel offsets
   - Computes Stokes parameters S0, S1, S2
   - Exports DoP, AoP, sin(AoP), cos(AoP)
   - Quantizes all outputs to 8-bit images
   - Key function: `raw2polar()`

2. **`feature_tracker.h/cpp`** — Multi-channel feature tracking:
   - Selects tracking channels based on configured `polar_channels`
   - Tracks features independently/jointly across polarization channels
   - Publishes tracking results to the estimator

### Configuration

YAML configs in `config/`. Key fields:
- `use_polar` (1 = enable polarization channels)
- `polar_channels` (e.g. `"s0,dop,aopsin,aopcos"`)
- `num_of_cam`, `imu`, `max_cnt`, `freq`, etc.

### Image filtering (polarization channels)

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
- Current state: V2 frontend multi-polar channel support completed (commit `1dde542`)
