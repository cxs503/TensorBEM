# Wave-ice-ship numbers (generated)

Every table below is generated from the JSON evidence written by `python -m tensorbem.benchmarks.ice_ship_verification` and rendered by `tools/ice_ship_report.py`; no value is transcribed by hand, so the report and the machine-readable evidence cannot drift apart.

## A. Mass-loading ice-cover dispersion (no BEM)

Ice cover: h_i = 0.90 m, rho_i = 917 kg/m^3, H = 200 m, mu = 0.8052 m, cut-off omega = 3.490 rad/s (T = 1.800 s).

| T (s) | k_ice (1/m) | k_open (1/m) | k_ice/k_open - 1 | lambda_ice (m) | c_ice (m/s) | cg_ice (m/s) | A_ice/A_open | Robin residual (1/m) | RK4 shooting error |
|---|---|---|---|---|---|---|---|---|---|
| 3.00 | 0.69907 | 0.44730 | 0.5629 | 9.0 | 3.00 | 0.96 | 1.5629 | 2.22e-15 | 1.06e-15 |
| 4.00 | 0.31553 | 0.25160 | 0.2541 | 19.9 | 4.98 | 1.98 | 1.2541 | 1.83e-15 | 2.40e-15 |
| 6.00 | 0.12289 | 0.11182 | 0.0989 | 51.1 | 8.52 | 3.88 | 1.0989 | 1.04e-15 | 4.03e-15 |
| 8.00 | 0.06626 | 0.06290 | 0.0533 | 94.8 | 11.85 | 5.63 | 1.0533 | 2.43e-15 | 1.74e-14 |
| 10.00 | 0.04161 | 0.04026 | 0.0335 | 151.0 | 15.10 | 7.31 | 1.0335 | 4.47e-15 | 5.05e-14 |
| 12.00 | 0.02860 | 0.02796 | 0.0230 | 219.7 | 18.31 | 8.95 | 1.0231 | 1.63e-16 | 4.45e-14 |
| 14.00 | 0.02089 | 0.02055 | 0.0167 | 300.7 | 21.48 | 10.60 | 1.0170 | 3.74e-15 | 1.03e-13 |

Independent references (the verification gate of this leg):

| reference | regime | error |
|---|---|---|
| deep-water Weitz-Keller omega^2 = g k/(1+mu k) | T = 10.0 s | 2.17e-14 |
| shallow-water omega^2 = g k^2 H/(1+mu k^2 H) | T = 12.0 s, kH = 0.426 | 3.14e-14 |
| RK4 shooting of W'' = k^2 W (no cosh) | T = 4-14 s | 1.03e-13 (worst) |
| Robin residual k tanh(kH) - omega^2/(g - mu omega^2) | T = 3-14 s | 4.47e-15 1/m |
| independent Robin root for k(h_i) | h_i = 0.3-2.5 m, T = 6.0 s | 1.11e-14 (worst) |

Ice-thickness study, T = 6.0 s (gate: < 5 %):

| h_i (m) | mu (m) | k_ice (1/m) | k analytic (1/m) | relative difference |
|---|---|---|---|---|
| 0.30 | 0.2684 | 0.11528 | 0.11528 | 3.49e-15 |
| 0.60 | 0.5368 | 0.11897 | 0.11897 | 1.11e-14 |
| 0.90 | 0.8052 | 0.12289 | 0.12289 | 8.47e-15 |
| 1.50 | 1.3420 | 0.13157 | 0.13157 | 7.17e-15 |
| 2.50 | 2.2366 | 0.14912 | 0.14912 | 1.49e-15 |

Evanescent branch above the cut-off (no propagating root exists; k = i kappa):

| omega/cut-off | kappa (1/m) | decay length (m) | amplitude after 100 m | residual of kappa tan(kappa H) |
|---|---|---|---|---|
| 1.02 | 0.00785 | 127.3 | 0.4560 | 7.51e-08 |
| 1.20 | 0.00784 | 127.5 | 0.4564 | 1.31e-09 |
| 2.00 | 0.00783 | 127.7 | 0.4570 | 1.93e-10 |

## B. Open-water chain against a classical reference (Lamb sphere)

Submerged sphere r = 5.0 m at 15.0 m depth; reference: Lamb infinite-fluid heave added mass (2/3) pi rho R^3.

| mesh | panels | omega (rad/s) | nu (1/m) | A33 (kg) | B33 (kg/s) | A33/A33_ref - 1 | A33/(rho V) |
|---|---|---|---|---|---|---|---|
| coarse | 168 | 0.30 | 0.00918 | 2.5176e+05 | 48.062 | 0.0618 | 0.4691 |
| coarse | 168 | 0.50 | 0.02549 | 2.5412e+05 | 1059.450 | 0.0530 | 0.4735 |
| coarse | 168 | 0.80 | 0.06526 | 2.5274e+05 | 8609.494 | 0.0581 | 0.4709 |
| fine | 352 | 0.30 | 0.00918 | 2.6137e+05 | 51.855 | 0.0260 | 0.4870 |
| fine | 352 | 0.50 | 0.02549 | 2.6391e+05 | 1143.281 | 0.0165 | 0.4917 |
| fine | 352 | 0.80 | 0.06526 | 2.6242e+05 | 9287.229 | 0.0221 | 0.4890 |

Ice chain in the mu -> 0 limit versus the open-water solve (consistency gate):

| mesh | wavenumber difference | added-mass difference | damping difference |
|---|---|---|---|
| coarse | 8.21e-12 | 3.03e-14 | 2.24e-11 |
| fine | 8.21e-12 | 3.21e-14 | 2.24e-11 |

Worst fine-mesh A33 error: 0.0260 (gate < 15 %: yes). Worst mu -> 0 difference: 2.24e-11.

## C. Idealised PC6-like hull: geometry gate and 6-DOF RAO

Hull: L = 120.0 m, B = 24.0 m, T = 9.0 m, waterline exponent n = 4.0, section exponent q = 2.0, keel ratio = 0.70, end ratio = 0.02.

Geometry gate (mesh versus independent parametric quadrature):

| mesh | panels | Cb (mesh) | Cb (reference) | Cb error | L error | B error | T error | Awp error |
|---|---|---|---|---|---|---|---|---|
| coarse | 314 | 0.71071 | 0.72178 | 0.0153 | 0.0 | 0.0 | 0.0 | 0.0096 |
| medium | 590 | 0.71631 | 0.72178 | 0.0076 | 0.0 | 0.0 | 0.0 | 0.0030 |
| fine | 1056 | 0.71816 | 0.72178 | 0.0050 | 0.0 | 0.0 | 0.0 | 0.0011 |

Mass = 1.9031e+07 kg at the design draft, CG = (0.00, 0.00, -4.50) m; assumptions: CG height = 0.50 T; k_xx = 0.35 B; k_yy = k_zz = 0.25 L (engineering estimates).
Ice cover: h_i = 0.90 m, H = 200.0 m, mu = 0.8052 m; head seas (wave angle 180 deg).

| T (s) | propagating | k_ice (1/m) | k_open (1/m) | k_i H | heave RAO open | heave RAO ice | pitch RAO open (rad/m) | pitch RAO ice (rad/m) | A33 open (kg) | A33 ice (kg) | B33 open (kg/s) | B33 ice (kg/s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.0 | yes | 0.31553 | 0.25160 | 63.1 | 0.0011 | 5.9302e-04 | 7.1217e-05 | 3.5392e-05 | 1.7260e+07 | 1.7924e+07 | 1.3337e+06 | 7.0651e+05 |
| 6.0 | yes | 0.12289 | 0.11182 | 24.6 | 0.0422 | 0.0330 | 0.0034 | 0.0029 | 1.4656e+07 | 1.4862e+07 | 5.2025e+06 | 4.4271e+06 |
| 8.0 | yes | 0.06626 | 0.06290 | 13.3 | 0.1595 | 0.1580 | 0.0304 | 0.0266 | 1.4896e+07 | 1.4722e+07 | 8.5406e+06 | 8.0396e+06 |
| 10.0 | yes | 0.04161 | 0.04026 | 8.3 | 0.5209 | 0.4801 | 0.0308 | 0.0298 | 1.8633e+07 | 1.8228e+07 | 1.0152e+07 | 9.9509e+06 |
| 14.0 | yes | 0.02088 | 0.02055 | 4.2 | 0.8703 | 0.8577 | 0.0182 | 0.0183 | 2.8111e+07 | 2.7878e+07 | 8.4161e+06 | 8.4229e+06 |

Mesh convergence at T = 8 s (heave RAO, head seas):

| mesh | panels | A33 ice (kg) | A33 open (kg) | heave RAO ice | heave RAO open |
|---|---|---|---|---|---|
| coarse | 314 | 1.4625e+07 | 1.4802e+07 | 0.1580 | 0.1618 |
| medium | 590 | 1.4722e+07 | 1.4896e+07 | 0.1580 | 0.1595 |

Added resistance in ice: **withheld** -- Ice-zone added resistance requires a second-order drift-force operator and/or a floe-scattering (attenuation) model; neither exists in this repository and no independent reference is available to validate a number, so R_AW is not reported. First-order drivers are reported instead.

Runtime of this stage: 1126 s.
