# Recovery robustness: data error, smoothing, and a computable indicator

Exact field: boundary potential of a unit sphere translating along +x;
the reference is the exact gradient projected on each panel plane.

## 1. Recovery of an exact trace (order)

| level | panels | h_mean | absolute RMS error | relative L2 | observed order |
|---:|---:|---:|---:|---:|---:|
| 1 | 80 | 0.5823 | 8.5221e-03 | 1.9658e-02 | -- |
| 2 | 320 | 0.2993 | 4.1277e-03 | 9.9585e-03 | 1.046 |
| 3 | 1280 | 0.1507 | 2.0819e-03 | 5.0802e-03 | 0.987 |
| 4 | 5120 | 0.0755 | 1.0341e-03 | 2.5307e-03 | 1.009 |

## 2. White (panel-to-panel) data error

| level | panels | h_mean | sigma | error, raw | error, 1 smoothing | gain | amplification, raw (sigma/h) | predicted | actual/predicted |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 80 | 0.5823 | 1.0e-02 | 1.505e-02 | 4.986e-02 | 0.30 | 0.877 | 1.262e-02 | 1.19 |
| 2 | 320 | 0.2993 | 1.0e-02 | 1.957e-02 | 1.803e-02 | 1.09 | 0.586 | 1.984e-02 | 0.99 |
| 3 | 1280 | 0.1507 | 1.0e-02 | 4.174e-02 | 2.141e-02 | 1.95 | 0.629 | 4.226e-02 | 0.99 |
| 4 | 5120 | 0.0755 | 1.0e-02 | 8.048e-02 | 3.849e-02 | 2.09 | 0.608 | 8.216e-02 | 0.98 |

## 3. Correlated (smooth) data error of the same amplitude

| level | panels | amplitude | error, raw | error after smoothing |
|---:|---:|---:|---:|---:|
| 1 | 80 | 1.0e-02 | 1.039e-02 | 5.105e-02 |
| 2 | 320 | 1.0e-02 | 4.731e-03 | 1.485e-02 |
| 3 | 1280 | 1.0e-02 | 2.270e-03 | 1.040e-02 |
| 4 | 5120 | 1.0e-02 | 1.110e-03 | 7.352e-03 |

## 4. Summary

- observed order of the recovery on exact data: 1.05, 0.99, 1.01
- slope of log(error) against log(h) for a fixed white sigma: -0.850 (pure 1/h is -1)
- smoothing gain on white noise: 0.30, 1.09, 1.95, 2.09
- smoothing penalty on exact data: 5.87, 3.44, 4.95, 7.09
- correlated error is smaller than the white error of the same amplitude by: 1.4, 4.1, 18.4, 72.5
- indicator on raw noisy data (actual/predicted): 1.19, 0.99, 0.99, 0.98
- indicator on smoothed noisy data (actual/predicted): 27.41, 4.08, 3.39, 3.45
