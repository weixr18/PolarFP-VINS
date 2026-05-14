# Feature Detector Comparison Summary

Each cell: **avg_inlier** / inlier_ratio / time(ms)

## Bright (94–205 lux)

| Channel | FAST | GFTT | SuperPoint |
|---|---|---|---|
| DoP (P0) | **73.4** / 94.30% / 1.0ms | **152.7** / 90.95% / 3.8ms | **25.5** / 89.78% / 9.9ms |
| sinAoP (P1) | **121.9** / 90.99% / 1.1ms | **188.4** / 83.93% / 4.0ms | **23.0** / 84.72% / 9.9ms |
| cosAoP (P2) | **304.6** / 91.11% / 1.6ms | **213.2** / 88.25% / 3.9ms | **25.9** / 88.45% / 9.9ms |

## Medium (2.6–18.6 lux)

| Channel | FAST | GFTT | SuperPoint |
|---|---|---|---|
| DoP (P0) | **1.5** / 36.96% / 0.8ms | **62.8** / 70.97% / 4.2ms | **12.9** / 76.25% / 10.6ms |
| sinAoP (P1) | **1.0** / 29.91% / 0.7ms | **24.4** / 65.32% / 4.7ms | **2.4** / 36.11% / 10.6ms |
| cosAoP (P2) | **20.1** / 87.79% / 1.1ms | **53.2** / 78.09% / 3.5ms | **10.3** / 72.00% / 10.6ms |

## Dark (0.9–4.8 lux)

| Channel | FAST | GFTT | SuperPoint |
|---|---|---|---|
| DoP (P0) | **0.0** / 3.65% / 0.4ms | **37.2** / 62.21% / 5.3ms | **5.0** / 55.53% / 10.2ms |
| sinAoP (P1) | **0.0** / 0.00% / 0.3ms | **11.5** / 54.89% / 6.3ms | **0.1** / 3.25% / 10.2ms |
| cosAoP (P2) | **1.9** / 46.55% / 0.9ms | **20.6** / 64.99% / 4.6ms | **2.7** / 41.33% / 10.2ms |

## Best per Lighting

| Lighting | Best Inlier | Best Ratio | Fastest |
|---|---|---|---|
| Bright | cosAoP+FAST (304.6) | DoP+FAST (94.30%) | DoP+FAST (1.0ms) |
| Medium | DoP+GFTT (62.8) | cosAoP+FAST (87.79%) | sinAoP+FAST (0.7ms) |
| Dark | DoP+GFTT (37.2) | cosAoP+GFTT (64.99%) | sinAoP+FAST (0.3ms) |