# Paper Codes — Figure Generation for PolarFP-VINS

This directory contains standalone code for generating figures used in the paper. Each subdirectory is a self-contained project numbered by figure order.

## Directory Structure

| Directory | Figure | Description |
|-----------|--------|-------------|
| `1_vis_polar_channel/` | Figure 2 | Polar channel visualization: raw image → S0/S1/S2/P0/P1/P2 composite table |
| `2_filter_compare/` | Table 1 | Filter method comparison: median/guided/bilateral/NLM on P0/P1/P2 |

## Shared Module

`utilities/PolarChannel.h/cpp` — shared polarization decoding library used by both folders.

## Naming Convention

Files follow `{number}_{keyword}.{ext}` or `get_data_{number}.py` to keep names distinct across folders:

| File | Description |
|------|-------------|
| `1_vis_polar_channel/get_data_1.py` | Extract first frame from ROS bags to PNG |
| `1_vis_polar_channel/vis_polar_channel_main.cpp` | Read PNG → compute channels → save composite |
| `2_filter_compare/get_data_2.py` | Extract K random frames from ROS bags to PNG |
| `2_filter_compare/filter_compare_main.cpp` | Read PNG → decode → 4 filters → save comparison |

When adding a new figure, follow this pattern: name the folder `{number}_{keyword}`, the Python data script `get_data_{number}.py`, and the C++ main file `{keyword}_main.cpp`.
