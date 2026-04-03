# CLAUDE.md - PolarFP-VINS Developer Guide

This file provides context for Claude Code when working on the PolarFP-VINS project.

## Project Overview

PolarFP-VINS is a modified VINS-Fusion that replaces the frontend feature tracker with a polarization-aware multi-channel tracker. The core innovation is processing raw polarization camera data (4-angle micropolarizer array) into Stokes parameters for robust feature tracking in dark scenes.

## Architecture Summary

### Key Components

1. **PolarChannel** (`vins_estimator/src/featureTracker/PolarChannel.cpp`)
   - Converts raw 2x2 polarization grid to Stokes parameters
   - Outputs: S0 (intensity), DoP (degree of polarization), AoP sin/cos components

2. **PolarFeatureTracker** (`vins_estimator/src/featureTracker/polarfp_tracker.cpp`)
   - Multi-channel feature detector (FAST/GFTT per channel)
   - LK optical flow within each channel
   - KD-tree matching using nanoflann
   - Multi-channel joint RANSAC for outlier rejection
   - Score-based feature management

3. **Estimator** (`vins_estimator/src/estimator/estimator.cpp`)
   - Standard VINS-Fusion backend
   - Uses `PolarFeatureTracker` instead of original `FeatureTracker`

### Data Flow

```
Raw Polarization Image
    -> PolarChannel::raw2polar() -> ChannelImages {s0, dop, aopsin, aopcos}
    -> PolarFeatureTracker::extractFeatures() (per channel)
    -> PolarFeatureTracker::matchFeatures() (LK + KD-tree)
    -> PolarFeatureTracker::combinedFeatureRANSAC() (joint outlier rejection)
    -> Estimator::processImage() (standard VINS optimization)
```

## Key Files and Their Roles

| File | Purpose |
|------|---------|
| `polarfp_tracker.h/cpp` | Main tracker class, drop-in replacement for VINS feature tracker |
| `PolarChannel.h/cpp` | Stokes parameter computation from raw polarization images |
| `nanoflann.hpp` | Header-only KD-tree for fast feature matching |
| `estimator.h` | Changed `featureTracker` type to `PolarFeatureTracker` |
| `estimator.cpp` | Unchanged logic, uses new tracker interface |
| `rosNodeTest.cpp` | ROS node entry point |

## Important Implementation Details

### Polarization Image Format

Input images are expected to have this 2x2 pixel pattern:
```
Row 0: [90°, 45°, 90°, 45°, ...]
Row 1: [135°, 0°, 135°, 0°, ...]
Row 2: [90°, 45°, 90°, 45°, ...]
...
```

The `raw2polar()` function subsamples this into 4 separate images and computes Stokes parameters.

### Channel Processing

Tracker processes these channels (defined in `PolarConfig.ALL_CHANNELS`):
- `s0`: Total intensity (grayscale equivalent)
- `dop`: Degree of polarization (0-1, quantized to 0-255)
- `aopsin`: sin(AoP), quantized to [0, 255]
- `aopcos`: cos(AoP), quantized to [0, 255]

### Feature Matching Pipeline

1. LK optical flow predicts feature positions
2. KD-tree finds nearest neighbors in current frame
3. Multi-channel RANSAC computes fundamental matrix using ALL channels jointly
4. Inliers kept, outliers discarded

### Configuration Parameters

Key params in `PolarConfig` (in `polarfp_tracker.h`):
```cpp
std::string FP_METHOD = "fast";  // "fast" or "gftt"
int KP_NUM_TARGET = 1000;        // Target feature count across all channels
cv::Size LK_WIN_SIZE = cv::Size(21, 21);
int LK_MAX_LEVEL = 6;
double LK_MATCH_THRESHOLD = 10.0;
int NMS_RADIUS = 21;
double RANSAC_REMAP_THR = 2.0;
```

FAST thresholds per channel:
- S0: 5 (low, captures many features)
- DoP: 15 (medium)
- AoP: 80 (high, only strong polarization angles)

## Build System

### CMakeLists.txt Changes

- C++17 required (for structured bindings in `PolarChannel.cpp`)
- Added `polarfp_tracker.cpp` and `PolarChannel.cpp` to library sources

### Package Names

- ROS package: `polarfp_vins`
- Library: `polarfp_vins_lib`
- Node: `polarfp_vins_node`

## Common Tasks

### Adding a New Polarization Channel

1. Update `PolarChannel::raw2polar()` to compute new channel
2. Add channel name to `PolarConfig.ALL_CHANNELS`
3. Optionally add to `PolarConfig.FP_CHANNELS` if features should be extracted
4. Add color to `PolarConfig.CHANNEL_COLORS` for visualization

### Tuning Feature Detection

Adjust thresholds in `PolarConfig` constructor:
```cpp
FAST_S0_DT = cv::FastFeatureDetector::create(5);   // Intensity
FAST_DOP_DT = cv::FastFeatureDetector::create(15); // DoP
FAST_AOP_DT = cv::FastFeatureDetector::create(80); // AoP
```

### Debugging Feature Tracking

Enable timing stats in `polarfp_tracker.cpp`:
```cpp
bool show_time_stats = true;  // Shows per-step timing in ROS_DEBUG
```

Enable track visualization in config:
```yaml
show_track: 1  # Publishes tracking image to topic
```

## Testing Configurations

| Config File | Camera Mode | Use Case |
|-------------|-------------|----------|
| `dark_mono_imu_config.yaml` | 612x512 (spin4) | Dark scenes, lower res |
| `arena_imu_config.yaml` | 1224x1024 (spin2) | Standard lighting |

## Known Limitations

1. **Mono only**: Stereo support not yet implemented in PolarFeatureTracker
2. **Single camera**: Multi-camera setup not tested
3. **Image size**: Assumes specific polarization grid layout (2x2)
4. **Computational cost**: Multi-channel processing ~2-3x slower than single channel

## Git Branches

- `master`: Main development branch
- `org-vins`: Original VINS-Fusion (for comparison)
- `feat/grid`: Development branch with current polarization features

## Related Research Context

This implementation is based on the principle that polarization information remains stable in low-light conditions where intensity images degrade. The Stokes parameters provide complementary information:
- S0: Standard intensity (fails in dark)
- DoP: Material/edge properties (robust to lighting)
- AoP: Surface orientation (robust to lighting)

When working on improvements, consider:
1. Channel weighting based on scene conditions
2. Adaptive thresholding per channel
3. Deep learning-based polarization feature extraction
