# CLAUDE.md — PolarFP-VINS

## Overview

Polarization-based Visual-Inertial Navigation System, forked from [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion). Core innovation: decodes a raw polarization camera image (2×2 micro-polarizer array) into 4 channels (S0 intensity, DoP, sin(AoP), cos(AoP)), runs feature tracking independently on each, then merges results for the backend — unchanged from VINS-Fusion. Provides robustness in low-light where intensity-only tracking fails.

ROS catkin, C++14/17. SuperPoint detection requires LibTorch.

## Key Architecture

### Two catkin packages

- **`polarfp_camera_models`** — Camera model library (pinhole/fisheye/Scaramuzza) and calibration tools.(Same as in VINS-Fusion)
- **`polarfp_vins_estimator`** — Main package. Builds `vins_lib` library + `polarfp_vins_node` executable.

### vins_estimator modules

| Directory | Responsibility |
|-----------|----------------|
| `featureTracker/` | Polar channel decoding (`PolarChannel`), detection (GFTT/FAST/SuperPoint), matching (LK/BRIEF+FLANN) |
| `estimator/` | Sliding-window BA, IMU preintegration, initialization, parameter loading |
| `factor/` | Ceres cost functions (IMU, projection, marginalization) |
| `initial/` | SFM, gyro bias, velocity/gravity initialization |
| `utility/` | Visualization, helpers |

### Key files

| File | Description |
|------|-------------|
| `featureTracker/PolarChannel.h/cpp` | Raw image → S0/DoP/AoP channels via Stokes parameters. Supports bilateral/guided/NLM filtering |
| `featureTracker/feature_tracker.h/cpp` | Multi-channel tracking: per-channel track → batch SuperPoint detect → merge results |
| `featureTracker/feature_tracker_detector.h/cpp` | Abstract `FeatureDetector`: GFTT / FAST / SuperPoint |
| `featureTracker/feature_tracker_matcher.h/cpp` | Abstract `FeatureMatcher`: LK flow / BRIEF+FLANN |
| `featureTracker/superpoint_detector.h/cpp` | LibTorch SuperPoint with PIMPL, batch multi-channel inference |
| `featureTracker/parameters.h/cpp` | YAML parameter loading |
| `estimator/estimator.cpp` | Main estimator, wires polar channels into FeatureTracker |

## Build & Environment

```bash
cd ~/ws/vi_catkin_ws
catkin_make
source devel/setup.bash
```

- **DO NOT use `catkin build`** — this is a catkin workspace using `catkin_make`
- SuperPoint requires LibTorch at `/home/dhz/SLAM/libs/libtorch-cxx11-abi-shared-with-deps-1.12.1+cu113/libtorch`
- CMake option `USE_SUPERPOINT` (default ON) gates SuperPoint compilation
- Python + PyTorch testing: conda env `torch1.12` at `/home/dhz/anaconda3/envs/torch1.12`

## Configuration

YAML configs in `config/`. Key fields:

- `use_polar`: 1 = enable polarization, 0 = returns empty featureFrame (non-polar VINS-Fusion path removed)
- `polar_channels`: e.g. `"s0,dop,aopsin,aopcos"`
- `feature_detector_type`: 0=GFTT, 1=FAST, 2=SUPERPOINT
- `feature_matcher_type`: 0=LK_FLOW, 1=BRIEF_FLANN
- `polar_filter_type`: 0=none, 1=bilateral, 2=guided, 3=NLM(slow, not recommanded)

## Workflow & Principles

- **Compilation**: User compiles manually. After code modifications, just report what changed — do NOT run compile commands unless asked.
- **Design principle**: Stay close to VINS-Fusion's proven pipeline. V1.1 was a full frontend rewrite that performed worse.
- **Backend is untouched**: Ceres factors, initialization, sliding window are identical to VINS-Fusion.

## Tech Stack

C++14/17 · ROS (roscpp, cv_bridge, tf) · OpenCV 4 · Ceres Solver · Eigen3 · LibTorch (optional)
