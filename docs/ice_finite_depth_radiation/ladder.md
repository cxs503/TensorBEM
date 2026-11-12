### Depth ladder (omega = 1 rad/s, ice thickness 0.9 m)

| H (m) | k_i H | K (1/m) | A_33 (kg) | B_33 (kg/s) | vs deep-water ice: dA | dB |
|---|---|---|---|---|---|---|
| 10 | 1.29 | 0.11109 | 1742.67 | 8.132 | guard (k_iH<3) | guard (k_iH<3) |
| 15 | 1.77 | 0.11109 | 1744.04 | 8.508 | guard (k_iH<3) | guard (k_iH<3) |
| 20 | 2.27 | 0.11109 | 1744.55 | 8.953 | guard (k_iH<3) | guard (k_iH<3) |
| 30 | 3.34 | 0.11109 | 1744.65 | 9.434 | 5.860e-05 | 8.033e-03 |
| 50 | 5.55 | 0.11109 | 1744.54 | 9.558 | 5.150e-05 | 2.672e-04 |
| 100 | 11.11 | 0.11109 | 1744.53 | 9.561 | 5.459e-05 | 7.495e-05 |

### Thickness ladder (omega = 1 rad/s, H = 30 m)

| h_i (m) | mu (m) | k_i H | A_33 (kg) | B_33 (kg/s) |
|---|---|---|---|---|
| 0.20 | 0.179 | 3.13 | 1740.67 | 7.822 |
| 0.50 | 0.447 | 3.22 | 1742.30 | 8.466 |
| 0.90 | 0.805 | 3.34 | 1744.65 | 9.434 |
| 1.50 | 1.342 | 3.55 | 1748.56 | 11.171 |
| 2.50 | 2.237 | 3.97 | 1756.37 | 15.108 |

### Analytic limits

| check | quantity | value |
|---|---|---|
| dispersion `k tanh(kH) = K` | residual (1/m) | 9.576e-16 |
| mu -> 0 vs open-water solver | added mass rel. diff | 1.410e-12 |
| mu -> 0 vs open-water solver | damping rel. diff | 2.389e-10 |
| mu -> 0 vs `omega^2/g` | `K` rel. diff | 9.123e-11 |

### Mesh sensitivity (H = 20 m, one frequency)

| subdivisions | panels | A_33 (kg) | B_33 (kg/s) | A_33 rel. change | B_33 rel. change |
|---|---|---|---|---|---|
| 0 | 20 | 1744.55 | 8.953 | - | - |
| 1 | 80 | 2274.42 | 16.580 | +3.037e-01 | +8.520e-01 |
