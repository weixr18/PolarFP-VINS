# Polar Channel Visualization — Figure 2

Generates a 3×6 composite table showing Stokes parameters (S0/S1/S2) and polar channels (P0/P1/P2) under three lighting conditions for the paper.

## Data

Three scenes from `~/data/dark/0416/`:

| Scene | Lighting | Bag path |
|-------|----------|----------|
| 13-27-22 | 94–205 lux | `13-27-22/2026-04-16-13-27-22.bag` |
| 13-54-30 | 2.6–18.6 lux | `13-54-30/2026-04-16-13-54-30.bag` |
| 14-09-36 | 0.9–4.8 lux | `14-09-36/2026-04-16-14-09-36.bag` |

## Dependencies

- **C++**: OpenCV 4 (no ROS)
- **Python** (frame extraction only): ROS Noetic (`rosbag`, `cv_bridge`)

## Usage

### 1. Extract first frames from ROS bags

```bash
cd paper_codes/1_vis_polar_channel
/usr/bin/python3 get_data_1.py \
  ~/data/dark/0416/13-27-22/2026-04-16-13-27-22.bag \
  ~/data/dark/0416/13-54-30/2026-04-16-13-54-30.bag \
  ~/data/dark/0416/14-09-36/2026-04-16-14-09-36.bag \
  data/
```

This saves `data/13-27-22.png`, `data/13-54-30.png`, `data/14-09-36.png`.

Note: if `conda` is active, deactivate first (`conda deactivate`) so the system Python's ROS bindings are visible.

### 2. Build the C++ program

```bash
cd paper_codes/1_vis_polar_channel
mkdir -p build && cd build && cmake .. && make -j$(nproc)
cd ..
```

### 3. Generate visualizations

```bash
cd paper_codes/1_vis_polar_channel
./build/vis_polar_channel
```

### Output

Individual channel images in `output/<ch>/<scene>.png`:

- `output/s0/` — S0 (intensity)
- `output/s1/` — S1 (Stokes, fixed map: v×0.5+127.5)
- `output/s2/` — S2 (Stokes, fixed map: v×0.5+127.5)
- `output/p0/` — P0 (DoP)
- `output/p1/` — P1 (sin(AoP))
- `output/p2/` — P2 (cos(AoP))

Composite table at `output/composite.png`:

|       | S0 | S1 | S2 | P0 | P1 | P2 |
|-------|----|----|----|----|----|----|
| 94-205 lux | | | | | | |
| 2.6-18.6 lux | | | | | | |
| 0.9-4.8 lux | | | | | | |

## Files

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Standalone CMake project (OpenCV only) |
| `get_data_1.py` | Extract first frame from ROS bags to PNG |
| `vis_polar_channel_main.cpp` | Read PNG → compute channels → save images + composite |
| `PolarChannel.h` | Polar channel computation (copied from `polarfp_vins_estimator`, modified to expose S1/S2) |
| `PolarChannel.cpp` | Implementation |
| `data/` | Input raw frames (PNG, 612×512) |
| `output/` | Generated channel images and composite |

## Channel definitions (from paper)

- S0 = (I0 + I45 + I90 + I135) / 4 (total intensity)
- S1 = I0 − I90
- S2 = I45 − I135
- DoP = √(S1² + S2²) / S0
- AoP = 0.5 × atan2(S2, S1)
- P0 = DoP × 255
- P1 = sin(AoP) × 127.5 + 127.5
- P2 = cos(AoP) × 127.5 + 127.5
