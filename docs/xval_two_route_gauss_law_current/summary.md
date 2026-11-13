# Two-route Gauss-law current cross-validation

| case | panels | surface I (A) | far-field I (A) | analytic I (A) | surface err | far err | two-route |
|---|---|---|---|---|---|---|---|
| sphere_r1 | 320 | -43.41021 | -43.41018 | -43.98230 | 1.301e-02 | 1.301e-02 | 6.325e-07 |
| prolate_2to1 | 320 | -57.09244 | -57.09181 | -57.84511 | 1.301e-02 | 1.302e-02 | 1.111e-05 |
| prolate_4to1 | 320 | -81.48001 | -81.47830 | -82.55290 | 1.300e-02 | 1.302e-02 | 2.098e-05 |
| oblate_2to1 | 320 | -35.90003 | -35.90082 | -36.37307 | 1.301e-02 | 1.298e-02 | 2.194e-05 |

| scored criterion | value | budget | pass |
|---|---|---|---|
| sphere_analytic_rel | 1.301e-02 | 3.000e-02 | yes |
| prolate_2to1_analytic_rel | 1.301e-02 | 3.000e-02 | yes |
| prolate_4to1_analytic_rel | 1.300e-02 | 5.000e-02 | yes |
| oblate_2to1_analytic_rel | 1.301e-02 | 5.000e-02 | yes |
| two_route_rel | 4.105e-05 | 1.000e-02 | yes |
| radius_independence_rel | 5.667e-05 | 1.000e-02 | yes |
| direction_sweep_max_rel | 2.929e-04 | 1.000e-02 | yes |
| mutant_detection_min_rel | 4.165e-02 | 1.000e-02 | yes |
| interference_ratio_low | 7.297e-01 | 6.931e-01 | yes |
| interference_ratio_high | 8.593e-01 | 1.000e+00 | yes |
| two_sphere_monotone | 1.000e+00 | 1.000e+00 | yes |

Overall: **PASS** (39.2 s)
