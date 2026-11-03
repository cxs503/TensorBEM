# Near-wall error budget (generated)

## 1. Near-wall velocity error and the density-amplification model

| panels | clearance/h_max | velocity error | \|dCp\|/scale | model prediction | max amplification \|gradOmega\|/4pi |
|---:|---:|---:|---:|---:|---:|
| 320 | 1 | 3.473e-04 | 0.000 | 1.744e-02 | 8.954e-02 |
| 320 | 0.5 | 8.458e-03 | 0.008 | 5.651e-02 | 3.061e-01 |
| 320 | 0.1 | 3.585e-01 | 0.319 | 3.869e-01 | 2.118e+00 |
| 320 | 0.03 | 1.626e+00 | 1.446 | 1.327e+00 | 7.267e+00 |
| 1280 | 1 | 6.594e-05 | 0.000 | 1.940e-02 | 1.966e-01 |
| 1280 | 0.5 | 7.388e-03 | 0.007 | 6.205e-02 | 6.786e-01 |
| 1280 | 0.1 | 3.670e-01 | 0.326 | 4.137e-01 | 4.581e+00 |
| 1280 | 0.03 | 1.676e+00 | 1.489 | 1.407e+00 | 1.558e+01 |

## 2. Closed-form solid-angle gradient vs the finite-difference oracle

| case | panels | max \|error\| | max relative error | \|sum_j grad Omega_j\| |
|---|---:|---:|---:|---:|
| equator_d0.1 | 1280 | 3.671e-09 | 1.943e-08 | 1.924e-13 |
| equator_d0.03 | 1280 | 7.045e-08 | 2.451e-08 | 2.153e-13 |

## 3. Quadrature independence at the failing clearance

| clearance/h_max | error, order 8 | error, order 12 | ratio |
|---:|---:|---:|---:|
| 1 | 6.594e-05 | 6.594e-05 | 1.000 |
| 0.5 | 7.388e-03 | 7.388e-03 | 1.000 |
| 0.1 | 3.670e-01 | 3.670e-01 | 1.000 |
| 0.03 | 1.676e+00 | 1.676e+00 | 1.000 |

## 4. Mesh ladder

| clearance/h_max | levels (panels: error) | observed orders |
|---:|---|---|
| 0.1 | 320: 3.585e-01, 1280: 3.670e-01 | -0.03 |
| 0.03 | 320: 1.626e+00, 1280: 1.676e+00 | -0.04 |
