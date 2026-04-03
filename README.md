# PolarFP-VINS

PolarFP-VINS (Polarization Feature Processing VINS) is a Visual-Inertial Navigation System (VINS) specifically designed for polarization cameras. Based on [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion), it replaces the traditional frontend with a novel multi-channel polarization feature tracker that significantly improves visual navigation capabilities in dark and low-light scenes.

## Overview

Traditional VINS systems rely on intensity-based feature detection, which struggles in dark environments where image contrast is poor. PolarFP-VINS leverages the physical properties of polarized light to extract robust features even when conventional cameras fail.

### Key Features

- **Multi-Channel Polarization Processing**: Decomposes raw polarization images into Stokes parameters (S0, DoP, AoP) for feature extraction
- **Dark Scene Robustness**: Maintains tracking capability in low-light conditions where traditional VINS fails
- **Multi-Channel Feature Fusion**: Combines features from intensity (S0), degree of polarization (DoP), and angle of polarization (AoP) channels
- **Custom Polarization Tracker**: Replaces VINS-Fusion's feature tracker with `PolarFeatureTracker` optimized for polarization cameras
- **IMU Integration**: Tightly-coupled visual-inertial odometry for accurate state estimation

## Polarization Camera Support

The system supports polarization cameras with a 2x2 micropolarizer array (0°, 45°, 90°, 135°), such as:
- Lucid Vision TRI050S-P (Spin4 configuration)
- Other Sony IMX250MZR-based polarization cameras

### Raw Image Format

Input images should be raw Bayer-like format where each 2x2 pixel block contains:
```
+-----+-----+
| 90° | 45° |
+-----+-----+
| 135°|  0° |
+-----+-----+
```

## Architecture

```
Raw Polarization Image (612x512 or 1224x1024)
           |
           v
+-----------------------------+
|    PolarChannel (raw2polar) |
+-----------------------------+
           |
    +------+------+------+
    |      |      |      |
    v      v      v      v
   S0     DoP   AoPSin AoPCos
(Intensity) (Degree) (Angle)
    |      |      |      |
    +------+------+------+
           |
           v
+-----------------------------+
|  PolarFeatureTracker        |
|  - Multi-channel FAST/GFTT  |
|  - LK Optical Flow          |
|  - KD-Tree Matching         |
|  - Multi-channel RANSAC     |
+-----------------------------+
           |
           v
+-----------------------------+
|      VINS-Fusion Backend    |
|  (Estimator/Optimization)   |
+-----------------------------+
           |
           v
        Odometry Output
```

## Installation

### Dependencies

- Ubuntu 18.04/20.04
- ROS Melodic/Noetic
- OpenCV 3.2+ or 4.x
- Eigen3
- Ceres Solver
- Boost

### Build

```bash
cd ~/catkin_ws/src
git clone <repository-url> PolarFP-VINS
cd ..
catkin build polarfp_vins polarfp_camera_models
# or: catkin_make
```

## Usage

### 1. Calibration

Calibrate your polarization camera using the provided camera models:

```bash
# Example for TRI050S-P camera
# Camera intrinsics should be saved to config/dark/TRI050S-spin4.yaml
```

### 2. Configuration

Edit the config file for your setup. See `config/dark/dark_mono_imu_config.yaml` for reference:

```yaml
# Key parameters
imu_topic: "/your/imu/topic"
image0_topic: "/your/camera/image_raw"
cam0_calib: "TRI050S-spin4.yaml"
image_width: 612
image_height: 512

# IMU-Camera extrinsics
body_T_cam0: !!opencv-matrix
   rows: 4
   cols: 4
   dt: d
   data: [ ... ]
```

### 3. Run

```bash
# Source your workspace
source ~/catkin_ws/devel/setup.bash

# Launch the estimator
rosrun polarfp_vins polarfp_vins_node \
    ~/catkin_ws/src/PolarFP-VINS/config/dark/dark_mono_imu_config.yaml
```

### 4. Visualization

```bash
# Launch RViz with VINS configuration
roslaunch vins_estimator vins_rviz.launch
```

## Configuration Files

| File | Description |
|------|-------------|
| `config/dark/dark_mono_imu_config.yaml` | Dark scene mono camera + IMU config |
| `config/dark/TRI050S-spin4.yaml` | Camera intrinsics for 612x512 mode |
| `config/arena/arena_imu_config.yaml` | Arena camera configuration |
| `config/arena/TRI050S-spin2.yaml` | Camera intrinsics for 1224x1024 mode |

## Algorithm Details

### Stokes Parameters Computation

From the 4 polarization angles (0°, 45°, 90°, 135°), we compute:

- **S0** (Total intensity): S0 = (I₀ + I₄₅ + I₉₀ + I₁₃₅) / 4
- **S1** (Horizontal/Vertical): S1 = I₀ - I₉₀
- **S2** (Diagonal): S2 = I₄₅ - I₁₃₅

Then derive:
- **DoP** (Degree of Polarization): √(S₁² + S₂²) / S₀
- **AoP** (Angle of Polarization): 0.5 × arctan2(S₂, S₁)

### Multi-Channel Feature Tracking

The tracker processes each polarization channel independently:

1. **Feature Detection**: FAST or GFTT on each channel (S0, DoP, AoPSin, AoPCos)
2. **Optical Flow**: LK pyramid tracking within each channel
3. **KD-Tree Matching**: Fast nearest-neighbor matching for feature correspondence
4. **Multi-Channel RANSAC**: Joint outlier rejection using fundamental matrix estimation
5. **Score-Based Management**: Features ranked by accumulated tracking quality scores

### Channel-Specific Parameters

Different channels use different detection thresholds:
- **S0 (Intensity)**: FAST threshold 5
- **DoP**: FAST threshold 15
- **AoP**: FAST threshold 80


## Differences from VINS-Fusion

| Component | VINS-Fusion | PolarFP-VINS |
|-----------|-------------|--------------|
| Feature Tracker | `FeatureTracker` (single channel) | `PolarFeatureTracker` (multi-channel) |
| Input | Grayscale/RGB | Raw polarization (4 angles) |
| Channel Processing | N/A | S0, DoP, AoP decomposition |
| Matching | LK Flow + Fundamental Matrix | LK Flow + KD-Tree + Multi-channel RANSAC |

## File Structure

```
PolarFP-VINS/
├── config/                     # Configuration files
│   ├── dark/                   # Dark scene configs
│   └── arena/                  # Arena camera configs
├── vins_estimator/
│   └── src/
│       ├── featureTracker/
│       │   ├── polarfp_tracker.cpp   # Main polarization tracker
│       │   ├── polarfp_tracker.h
│       │   ├── PolarChannel.cpp      # Stokes parameter computation
│       │   └── PolarChannel.h
│       └── estimator/
│           └── estimator.cpp         # Modified to use PolarFeatureTracker
└── polarfp_camera_models/      # Camera calibration models
```

## License

This project follows the same license as VINS-Fusion (GPL v3.0). See [LICENCE](LICENCE) for details.

## Acknowledgements

- Original [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) by HKUST Aerial Robotics Group
- [camodocal](https://github.com/hengli/camodocal) for camera models
- [nanoflann](https://github.com/jlblancoc/nanoflann) for KD-tree implementation
