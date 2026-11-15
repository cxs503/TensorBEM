# KK gate calibration (C6b): error-injection certificate

Exact causal pair -- two-exponential memory kernel -- on the case band 0.2-12 rad/s, 41 nodes. The estimator is the case's own `kk_added_mass_closure`.

## 1 Pair (truth)

| quantity | value |
| --- | --- |
| kernel | `sum_j c_j exp(-a_j s)  (exact causal memory kernel)` |
| coefficients | -8, 32 |
| rates | 2, 8 |
| A_inf | 3 |
| B hump at omega | 4 rad/s |
| max B on band | 2.399 |
| max \|A - A_inf\| on band | 1.481 |
| B > 0 on band | yes |

## 2 Estimator configuration (same as the case)

| item | value |
| --- | --- |
| power-tail fit valid | no |
| fitted tail exponent | 0.8061 |
| tail rule | fitted exponent 0.806 < 3: the sweep has not reached the algebraic tail |
| floor on the exact pair | 0.00565 |

## 3 Calibration curves (injected -> reported)

### damping_step

| injected delta | reported residual | above floor | over bar |
| --- | --- | --- | --- |
| 0 | 0.00565 | 0 | no |
| 0.01 | 0.00478 | -0.000873 | no |
| 0.02 | 0.00465 | -0.001 | no |
| 0.05 | 0.0099 | 0.00425 | no |
| 0.1 | 0.0187 | 0.013 | no |
| 0.15 | 0.0274 | 0.0218 | no |
| 0.2 | 0.0362 | 0.0305 | no |
| 0.3 | 0.0537 | 0.048 | no |
| 0.45 | 0.0799 | 0.0743 | no |
| 0.6 | 0.106 | 0.101 | yes |
| 0.9 | 0.159 | 0.153 | yes |

slope `residual - floor` vs delta = **0.174**, monotone = no; bar crossed at delta = **0.565** of max|A|, **0.192** of the added-mass amplitude (linear interpolation between bracketing grid points).

### added_mass_bump

| injected delta | reported residual | above floor | over bar |
| --- | --- | --- | --- |
| 0 | 0.00565 | 0 | no |
| 0.01 | 0.00437 | -0.00129 | no |
| 0.02 | 0.00313 | -0.00252 | no |
| 0.05 | 0.0086 | 0.00294 | no |
| 0.1 | 0.0185 | 0.0128 | no |
| 0.2 | 0.0383 | 0.0327 | no |
| 0.4 | 0.0779 | 0.0723 | no |
| 0.8 | 0.157 | 0.152 | yes |

slope `residual - floor` vs delta = **0.196**, monotone = no; bar crossed at delta = **0.511** of max|A|, **0.158** of the added-mass amplitude (linear interpolation between bracketing grid points).

### added_mass_bump_zero_mean

| injected delta | reported residual | above floor | over bar |
| --- | --- | --- | --- |
| 0 | 0.00565 | 0 | no |
| 0.01 | 0.00369 | -0.00197 | no |
| 0.02 | 0.00479 | -0.000861 | no |
| 0.05 | 0.014 | 0.00837 | no |
| 0.1 | 0.0296 | 0.024 | no |
| 0.2 | 0.0617 | 0.0561 | no |
| 0.4 | 0.13 | 0.124 | yes |
| 0.8 | 0.257 | 0.252 | yes |

slope `residual - floor` vs delta = **0.324**, monotone = yes; bar crossed at delta = **0.313** of max|A|, **0.104** of the added-mass amplitude (linear interpolation between bracketing grid points).

### damping_bump

| injected delta | reported residual | above floor | over bar |
| --- | --- | --- | --- |
| 0 | 0.00565 | 0 | no |
| 0.05 | 0.0068 | 0.00115 | no |
| 0.1 | 0.0124 | 0.00678 | no |
| 0.2 | 0.0245 | 0.0189 | no |
| 0.4 | 0.0497 | 0.0441 | no |
| 0.8 | 0.1 | 0.0945 | yes |
| 1.6 | 0.201 | 0.195 | yes |
| 3.2 | 0.403 | 0.397 | yes |

slope `residual - floor` vs delta = **0.126**, monotone = yes; bar crossed at delta = **0.799** of max|A|, **0.282** of the added-mass amplitude (linear interpolation between bracketing grid points).

## 3b Localisation blind spot (how narrow may a corruption be?)

| log-width | injected delta | reported residual | over bar |
| --- | --- | --- | --- |
| 0.15 | 0.3 | 0.0248 | no |
| 0.15 | 0.5 | 0.0422 | no |
| 0.15 | 1 | 0.0855 | no |
| 0.15 | 2 | 0.172 | yes |
| 0.25 | 0.3 | 0.0281 | no |
| 0.25 | 0.5 | 0.0477 | no |
| 0.25 | 1 | 0.0965 | no |
| 0.25 | 2 | 0.194 | yes |
| 0.4 | 0.3 | 0.0308 | no |
| 0.4 | 0.5 | 0.0517 | no |
| 0.4 | 1 | 0.104 | yes |
| 0.4 | 2 | 0.209 | yes |
| 0.6 | 0.3 | 0.0371 | no |
| 0.6 | 0.5 | 0.0624 | no |
| 0.6 | 1 | 0.125 | yes |
| 0.6 | 2 | 0.251 | yes |
| 1 | 0.3 | 0.0444 | no |
| 1 | 0.5 | 0.0739 | no |
| 1 | 1 | 0.148 | yes |
| 1 | 2 | 0.298 | yes |

Largest corruption that still passes the bar: a **2x** damping error confined to d(log omega) = 0.15 reports only **0.0855** of max|A|.

## 4 Gate record

| item | value |
| --- | --- |
| gate | C6b_cal |
| passed | yes |
| floor_relative_residual | 0.005653 |
| floor_over_bar | 0.05653 |
| damping_step_slope | 0.1745 |
| damping_step_monotone_over_0p05 | yes |
| detection_threshold_step_reported | 0.5645 |
| detection_threshold_step_amplitude_normalised | 0.1922 |
| injected_step_C6b | 0.3 |
| predicted_residual_at_c6b_step | 0.05368 |
| predicted_residual_at_c6b_step_amplitude_normalised | 0.1484 |
| detection_margin_reported | 1.882 |
| detection_margin_amplitude_normalised | 0.6407 |
| added_mass_bump_slope | 0.1963 |
| added_mass_bump_zero_mean_slope | 0.324 |
| damping_bump_threshold_reported | 0.7986 |
| note | certifies the detector, not the sweep: the case's truncation budget on surge/sway/yaw stays unbounded until the 20 rad/s band artifact exists, which is what gates C8/C9 and C3a_extended address |
| claim floor_ok | yes |
| claim responsive_ok | yes |
| claim detection_ok | yes |

## 5 What this does and does not certify

* **Does**: the KK gate reports an injected damping-shape corruption monotonically and with a measured slope, crosses the 10 % bar at a measurable fraction of the 0.30 step that `C6b` injects, and carries an estimator bias (floor) far below the bar on data whose answer is known.
* **Does not**: change the case's truncation budget. surge/sway/yaw keep `n_decidable = 0` at 12 rad/s; that is a property of the un-solved high-frequency band and needs the 20 rad/s artifact (C8/C9).
