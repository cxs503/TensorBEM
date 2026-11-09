# Multibody frequency dependence of added mass and damping (3-D Wigley hulls)

Campaign node `P106W2-N04`.  Study record: `benchmarks/multibody_frequency_dependence_3d/results.json` (wall clock 571.6 s).

## 1. Case, geometry and parameters

Two and three identical Wigley III hulls (Journee 1992) side by side, zero forward speed, deep water, linear potential flow.

| quantity | value |
|---|---|
| length L | 3.0 m |
| beam B | 0.3 m |
| draft T | 0.1875 m |
| modification alpha | 1.1088 |
| block coefficient (analytic) | 0.5430 |
| L/B, B/T | 10.0, 1.60 |
| reference gap (hulls abreast) | 0.50 B = 0.050 L (centre distance 0.150 L) |
| heave natural frequency omega_n | 5.1619 rad/s |
| displaced mass | 92.64 kg |
| waterplane area | 0.5956 m^2 |
| A33(0) used in omega_n | 132.06 kg |

Geometry gate (discretisation before hydrodynamics):

| nx | nz | panels | displaced volume [m^3] | analytic | rel. error | waterplane rel. error |
|---|---|---|---|---|---|---|
| 12 | 6 | 264 | 0.089379 | 0.091631 | 2.458 % | 1.346 % |
| 16 | 8 | 480 | 0.090385 | 0.091631 | 1.360 % | 0.727 % |
| 24 | 8 | 736 | 0.090764 | 0.091631 | 0.946 % | 0.310 % |
| 40 | 12 | 1872 | 0.091272 | 0.091631 | 0.392 % | 0.108 % |

## 2. Method

* **Frequency domain** -- `tensorbem.solve_radiation_3d` (Kelvin/Havelock free-surface Green function, direct collocation with CHIEF rows for irregular-frequency suppression) on a boundary surface that contains all hulls of the configuration.  Modes solved: sway (1), heave (2), pitch (4) about the common origin, so a catamaran/trimaran pitches and drifts as one rigid body and the difference from `n_hulls x single` is pure hydrodynamic interference.
* **Zero-frequency limit** (`dphi/dz = 0`, rigid wall): exact Laplace collocation on the mirror-closed double body with `sigma = +1` continuation (`tensorbem.multibody3d.static_added_mass`).
* **Infinite-frequency limit** (`phi = 0`, pressure release): the same solve with `sigma = -1`; the odd continuation of a surface-piercing hull is exactly the radiation problem of the closed double body.
* **Infinite-frequency check from the sweep**: fit `A(w) = A_inf + c/w^2` (the Kramers-Kronig branch) to the resolved high-frequency ladder and compare `A_inf` with the pressure-release solve.
* **Mesh policy**: the longitudinal panel is tied to the radiated wavelength (target 8 panels per wavelength, floor 6); every row reports `panels per wavelength` and rows below the floor are flagged `resolved = false` and excluded from the criteria.
* **Grid convergence**: Richardson triples (`nx = 12, 16, 24`) at two frequencies, refinement 1.5.

## 3. Results

### 3.1 Frequency scan (single hull / catamaran / trimaran)

| w/w_n | ppw | resolved | A_33 single | A_33 cat | A_33 tri | B_33 single | B_33 cat | B_33 tri |
|---|---|---|---|---|---|---|---|---|
| 0.10 | 847.9 | yes | 134.512 | 429.513 | 829.204 | 2.467 | 9.797 | 21.920 |
| 0.20 | 212.0 | yes | 139.344 | 444.721 | 856.690 | 18.470 | 71.603 | 157.220 |
| 0.30 | 94.2 | yes | 138.158 | 434.925 | 827.241 | 55.361 | 206.170 | 439.064 |
| 0.45 | 41.9 | yes | 122.238 | 374.808 | 698.935 | 141.679 | 491.707 | 996.819 |
| 0.60 | 23.6 | yes | 95.401 | 294.819 | 555.888 | 230.264 | 764.291 | 1513.094 |
| 0.80 | 13.2 | yes | 59.683 | 209.613 | 436.287 | 285.766 | 997.327 | 2138.129 |
| 1.00 | 8.5 | yes | 41.399 | 194.867 | 299.819 | 253.641 | 1302.600 | 4278.629 |
| 1.20 | 8.0 | yes | 37.798 | -180.910 | -367.463 | 203.188 | 1382.474 | 1225.222 |
| 1.50 | 6.5 | yes | 39.675 | 48.758 | 57.418 | 126.499 | 104.602 | 106.059 |
| 2.00 | 3.7 | no | 47.644 | 103.909 | n/a | 42.917 | 47.129 | n/a |
| 3.00 | 1.6 | no | 56.388 | 133.103 | n/a | 2.105 | 3.355 | n/a |

### 3.2 Asymptotic limits (exact potential solves)

| configuration | A_22(0) | A_33(0) | A_55(0) | A_22(inf) | A_33(inf) | A_55(inf) | parity residual | cond |
|---|---|---|---|---|---|---|---|---|
| single | 151.029 | 132.058 | 27.181 | 61.615 | 61.860 | 18.013 | 1.8e-02 | 26.6 |
| catamaran | 250.585 | 417.742 | 73.892 | 116.847 | 151.228 | 40.753 | 1.1e-02 | 27.8 |
| trimaran | 346.202 | 793.011 | 125.666 | 174.050 | 245.181 | 62.816 | 8.8e-03 | 17.5 |

Mesh study of the asymptotes (rigid-wall heave, closure panel count):

| configuration | nx | closure panels | A_33(0) | A_33(inf) | parity |
|---|---|---|---|---|---|
| single | 12 | 528 | 130.570 | 60.919 | 1.8e-02 |
| single | 16 | 960 | 132.058 | 61.860 | 1.8e-02 |
| catamaran | 12 | 1056 | 413.084 | 148.695 | 1.1e-02 |
| catamaran | 16 | 1920 | 417.742 | 151.228 | 1.1e-02 |
| trimaran | 12 | 1584 | 793.011 | 245.181 | 8.8e-03 |

### 3.3 High-frequency ladder and the infinite-frequency extrapolation

| w/w_n | ppw | resolved | A_33 single | A_33 catamaran |
|---|---|---|---|---|
| 1.00 | 8.5 | yes | 41.399 | 194.867 |
| 1.25 | 8.4 | yes | 37.773 | -136.326 |
| 1.50 | 6.5 | yes | 39.675 | 48.758 |
| 2.00 | 3.7 | no | 47.644 | 103.909 |

`infinite_frequency_added_mass`: extrapolated -58.129, 36.858, 7.099 vs exact 61.615, 61.860, 18.013 (modes sway, heave, pitch), R^2 0.9951, 0.3803, 0.8952.

`infinite_frequency_added_mass_catamaran`: extrapolated 91.090, -203.681, -48.829 vs exact 116.847, 151.228, 40.753 (modes sway, heave, pitch), R^2 0.9794, 0.3424, 0.4070.

### 3.4 Interference in the frequency domain

| w/w_n | sigma_A cat (sway/heave/pitch) | sigma_B cat | sigma_A tri |
|---|---|---|---|
| 0.10 | 0.830 / 1.597 / 1.366 | 1.598 / 1.985 / 2.040 | 0.757 / 2.055 / 1.587 |
| 0.20 | 0.829 / 1.596 / 1.376 | 1.588 / 1.938 / 2.030 | 0.756 / 2.049 / 1.606 |
| 0.30 | 0.830 / 1.574 / 1.394 | 1.569 / 1.862 / 2.009 | 0.757 / 1.996 / 1.641 |
| 0.45 | 0.832 / 1.533 / 1.428 | 1.511 / 1.735 / 1.938 | 0.756 / 1.906 / 1.705 |
| 0.60 | 0.817 / 1.545 / 1.430 | 1.360 / 1.660 / 1.801 | 0.722 / 1.942 / 1.696 |
| 0.80 | 0.713 / 1.756 / 1.431 | 0.940 / 1.745 / 1.650 | 0.587 / 2.437 / 1.679 |
| 1.00 | 0.648 / 2.354 / 1.702 | 0.603 / 2.568 / 1.971 | 0.549 / 2.414 / 1.798 |
| 1.20 | 0.772 / -2.393 / -1.612 | 0.470 / 3.402 / 3.154 | 0.970 / -3.241 / -1.544 |
| 1.50 | 1.693 / 0.614 / 0.686 | 0.435 / 0.413 / 0.436 | 1.906 / 0.482 / 0.566 |
| 2.00 | -0.040 / 1.090 / 1.055 | 1.204 / 0.549 / 0.526 | n/a |
| 3.00 | 0.852 / 1.180 / 1.105 | 1.345 / 0.797 / 0.944 | n/a |

### 3.5 Distance decay

| w/w_n | gap d/L | sigma_A (sway/heave/pitch) | cond |
|---|---|---|---|
| 0.6 | 0.000 | 0.722 / 1.681 / 1.566 | 29.8 |
| 0.6 | 0.005 | 0.733 / 1.665 / 1.549 | 25.0 |
| 0.6 | 0.010 | 0.744 / 1.650 / 1.533 | 24.8 |
| 0.6 | 0.025 | 0.776 / 1.609 / 1.489 | 24.4 |
| 0.6 | 0.050 | 0.817 / 1.545 / 1.430 | 23.8 |
| 0.6 | 0.100 | 0.868 / 1.429 / 1.340 | 22.9 |
| 0.6 | 0.200 | 0.908 / 1.209 / 1.215 | 21.7 |
| 0.6 | 0.500 | 0.934 / 0.629 / 1.022 | 22.3 |
| 0.6 | 1.000 | 0.981 / 0.678 / 0.936 | 22.1 |
| 0.6 | 2.500 | 0.966 / 0.949 / 1.016 | 20.6 |
| 0.6 | 5.000 | 0.970 / 0.798 / 0.999 | 21.1 |
| 0.6 | 10.000 | 1.015 / 1.039 / 0.998 | 20.3 |
| 1.2 | 0.000 | 0.579 / 2.821 / 2.701 | 79.2 |
| 1.2 | 0.005 | 0.597 / 2.971 / 2.714 | 57.0 |
| 1.2 | 0.010 | 0.616 / 3.107 / 2.664 | 57.0 |
| 1.2 | 0.025 | 0.674 / 2.615 / 1.604 | 60.3 |
| 1.2 | 0.050 | 0.772 / -2.393 / -1.612 | 51.9 |
| 1.2 | 0.100 | 0.985 / -0.210 / -0.294 | 51.5 |
| 1.2 | 0.200 | 1.724 / 0.820 / 0.698 | 52.3 |
| 1.2 | 0.500 | 0.484 / 1.891 / 1.604 | 45.8 |
| 1.2 | 1.000 | 0.178 / 1.582 / 1.256 | 44.7 |
| 1.2 | 2.500 | 0.273 / 1.132 / 0.951 | 43.8 |
| 1.2 | 5.000 | 1.256 / 0.616 / 0.978 | 40.8 |
| 1.2 | 10.000 | 1.179 / 1.082 / 0.985 | 40.4 |

Fitted power law `|sigma_A - 1| = C (d/L)^(-n)`:

| w/w_n | mode | n | C | points used/total | scatter (dex) |
|---|---|---|---|---|---|
| 0.6 | heave | 0.35 | 0.1582 | 5/11 | 0.198 |
| 0.6 | pitch | 0.85 | 0.0191 | 2/11 | 0.128 |
| 0.6 | sway | 0.38 | 0.0484 | 10/11 | 0.166 |
| 1.2 | heave | 0.40 | 0.3645 | 3/11 | 0.156 |
| 1.2 | pitch | 0.64 | 0.1255 | 1/11 | n/a |
| 1.2 | sway | -0.04 | 0.3161 | 3/11 | 0.191 |

### 3.6 Negative control

| gap/beam | sigma_A heave | sigma_A sway | cond |
|---|---|---|---|
| 0.0000 | 2.412 | 0.950 | 74.7 |
| 0.0020 | 2.418 | 0.953 | 75.3 |
| 0.0100 | 2.429 | 0.962 | 78.5 |
| 0.0200 | 2.410 | 0.975 | 83.3 |
| 0.0500 | 1.852 | 1.013 | 99.5 |

Superposed (coincident) hulls: status `solved`, condition number 101.983 , relative residual 0.036941766526274095.

### 3.7 Energy checks

| w/w_n | configuration | B_33 near | B_33 far (Haskind) | rel. difference (sway/heave/pitch) |
|---|---|---|---|---|
| 0.60 | single | 230.267 | 230.379 | 0.023 / 0.000 / 0.002 |
| 0.60 | catamaran | 764.323 | 764.789 | 0.023 / 0.001 / 0.002 |
| 1.00 | single | 253.644 | 254.031 | 0.026 / 0.002 / 0.011 |
| 1.00 | catamaran | 1303.326 | 1309.879 | 0.024 / 0.005 / 0.007 |

Kramers-Kronig moment over the resolved band (0.52-7.74 rad/s, 9 rows): `(2/pi) int B/w^2 dw = 66.11` kg.  The exact relation gives `A(0) - A(inf) = 70.20` kg for heave; the deficit is the part of the integral above the resolved band.

### 3.8 Grid convergence

| w/w_n | quantity | observed order | GCI |
|---|---|---|---|
| 1.0 | added_mass_heave_single | 1.59 | 0.0113 |
| 1.0 | damping_heave_single | n/a | n/a |
| 1.0 | added_mass_heave_catamaran | 1.72 | 0.0053 |
| 1.0 | sigma_added_mass_heave | 1.52 | 0.0058 |
| 1.5 | added_mass_heave_single | 1.60 | 0.0153 |
| 1.5 | damping_heave_single | 1.72 | 0.0067 |
| 1.5 | added_mass_heave_catamaran | 1.68 | 0.0307 |
| 1.5 | sigma_added_mass_heave | 1.86 | 0.0146 |

## 4. Criterion verdicts

| criterion | measured | limit | verdict |
|---|---|---|---|
| distance_decay_far_field | 0.384013 | 0.05 | FAIL |
| energy_far_field_agreement | 0.025829 | 0.1 | PASS |
| energy_non_negative_damping | 0.000149 | 0.0 | PASS |
| geometry | 0.003925 | 0.02 | PASS |
| infinite_frequency_added_mass | 1.943412 | 0.02 | FAIL |
| infinite_frequency_added_mass_catamaran | 2.346851 | 0.02 | FAIL |
| infinite_frequency_from_kramers_kronig | 0.538143 | 0.02 | FAIL |
| negative_control_singularity | {"monotone_growth": true, "superposed_status": "solved", "superposition_singular": true} | None | PASS |
| zero_frequency_added_mass | 0.030189 | 0.02 | FAIL |
| zero_frequency_damping | 0.008634 | 0.01 | PASS |

**all gated criteria pass: False**

## 5. Notes, limits and residual risk

* Zero forward speed and deep water only: the Kelvin kernel is the zero-speed kernel, so no forward-speed interference (the classical multihull design point) is covered.
* The two hulls interact through the linearised free surface only: no viscous coupling, no wave breaking in the gap, no hull-to-hull diffraction.
* Rows flagged `resolved = false` are reported but excluded from the criteria: at those frequencies the radiated wavelength is shorter than the affordable panel length, and the number would be a discretisation artefact.
* The infinite-frequency value obtained from the sweep is an *extrapolation* along the Kramers-Kronig `1/w^2` branch; its agreement with the pressure-release solve is a statement about the branch, not about a computed `A(inf)` sample.
* The interference factor uses the mode's diagonal only; off-diagonal (sway-heave, heave-pitch) coupling of the pair is computed but not part of the criteria.

## 6. Error judgement against references

| quantity | reference | measured | relative error | below 3 %? |
|---|---|---|---|---|
| displaced volume (nx=40) | Cb L B T = 0.091631 m^3 | 0.091272 m^3 | 0.392 % | yes |
| waterplane area (nx=40) | 2/3 B L = 0.6000 m^2 | 0.5994 m^2 | 0.108 % | yes |
| A_33 at omega_min vs rigid-wall solve | 130.570 kg | 134.512 kg | 3.019 % | no |
| A_33(inf) extrapolated vs pressure-release solve | 61.860 kg | 36.858 kg | 40.418 % | no |
| B (near field) vs B (Haskind far field) | - | worst 2.583 % | 2.583 % | yes |

No experimental or published Wigley multihull added-mass table is available in this repository, so the comparisons above are against analytic hydrostatics and against two independent numerical formulations of the same linear problem (Kelvin Green function vs Laplace double/single layer).  Where no external reference exists the table says so instead of quoting a self-consistency difference as an experimental error.
