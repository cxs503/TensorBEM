# Finite-depth frequency-domain potential flow: three-anchor verification loop

**Case** `benchmarks/verified/bem_finite_depth` · **repository** TensorBEM · **all gates passed** `True`

**Evidence level.** REAL for every number in this record: each one is a computed value compared with an analytic solution, a quadrature of the same definition along a different numerical path, or an exact conservation identity. No experimental data set ships with this repository, so the engineering (dimensional) magnitudes remain WITHHELD; what is established here is the correctness of the solver chain against those references.

## 1. Case, geometry and parameters

Linear potential flow in water of constant depth `h` with a flat impermeable bottom at `z = -h` and a free surface at `z = 0`, time factor `exp(-i omega t)`. The vertical eigenproblem of the strip has one propagating mode `phi_0 = cosh(k0 (z+h))`, `k0 tanh(k0 h) = nu = omega^2/g`, and a countable set of evanescent modes. The kernel and the two solvers are the existing `tensorbem.finite_depth`, `tensorbem.finite_depth_solver` and `tensorbem.finite_depth_diffraction`; this case measures them, it does not re-derive them.

### Hull types (leg A)

| hull | L (m) | B (m) | T (m) | Cb (published) | Cb (mesh) | Cb error | Cm (section) |
|---|---|---|---|---|---|---|---|
| Wigley | 6.000 | 0.600 | 0.375 | 0.4444 | 0.4439 | 0.0012 | 0.6667 |
| Series 60 (idealised) | 120.000 | 18.500 | 7.500 | 0.6000 | 0.5976 | 0.0041 | 0.9600 |
| KVLCC2 (idealised) | 320.000 | 58.000 | 20.800 | 0.8098 | 0.8072 | 0.0032 | 0.9980 |

Wigley is the closed-form parabolic hull. Series 60 and KVLCC2 are the parametric hulls of `tensorbem.pc6_hull`, calibrated (waterline exponent) so that the quadrature of the parametric definition reproduces the published block coefficient; KVLCC2 additionally carries the published principal dimensions Lpp/B/T = 320/58/20.8 and Cm = 0.9980. They are *idealised* hulls, not the official offset tables.

### Wave matrix (legs B-E)

| parameter | value |
|---|---|
| section family | superellipse `\|y/a\|^p + \|z/b\|^p = 1`, `Cm = Gamma(1+1/p)^2/Gamma(1+2/p)` |
| sections | Wigley `Cm = 2/3`, Series 60 `Cm = 0.96`, KVLCC2 `Cm = 0.998` |
| semi-draught / submergence | 1.0 m / 0.50 m (total draught 2.50 m) |
| frequencies `nu*T` | 0.30, 0.60, 1.00, 1.50, 2.00 |
| depth ratios `h/T` | 1.50, 3.00, inf |
| panels per section | 80 (constant panels, arclength-uniform) |
| gravity / density | 9.80665 m/s^2 / 1025.0 kg/m^3 |

The sections are rigidly submerged: the shallowest point sits `0.5 b` below the still water level. That is a stated approximation -- a real hull section pierces the free surface -- so the wave results are *shape* hydrodynamics and not ship RAOs.

## 2. Method

**Leg A -- geometry before hydrodynamics.** The mesh volume is integrated by the divergence theorem over the triangulated surface, and compared with a quadrature of the parametric half-breadth definition (a different path: two-dimensional trapezoid on `(u, v)` for the parametric hulls, closed forms `V = 4LBT/9`, `Cb = 4/9`, `A_wp = 2LB/3` for Wigley). Principal dimensions and the waterplane area are re-derived from the mesh. Rule: a hull that fails this gate must not be used for hydrodynamics.
**Leg B -- causality.** A frequency-domain radiation solution is a causal linear system, so `(A(omega), B(omega))` must satisfy the hydrodynamic Kramers-Kronig relations. Two sweeps are reported: Ursell's multipole closed form (already a causal pair, so its residual is the *transform* floor) and the panel solve. The damping direction is band-limited: the truncated high-frequency band enters through `s^2/(s^2-omega^2) -> 1`, i.e. as an additive constant in `B`. The constant is measured and separated, and the gate is applied to the frequency-dependent part; the raw number is reported as well, not hidden.
**Leg C -- diffraction energy.** For a closed body in a strip the incident, reflected and transmitted fluxes balance, and the group velocity is the same on both sides, so `|R|^2 + |T|^2 = 1`. Reported per case.
**Leg D -- depth limit.** At fixed `nu*T` the finite-depth coefficients are compared with the separately implemented deep-water solver (`tensorbem.free_surface_solver`, Havelock kernel) and the deviation is tracked against `h/T`; the convergence slope is fitted on the four deepest stations.
**Leg E -- second energy identity per case.** The damping from the radiated wave amplitude (`B = 2 rho g c_g |eta_a|^2` in finite depth, Haskind form in infinite depth) is compared with the damping from the body pressure, `omega rho Im(force)`. Two independent routes to the same number.

## 3. Results

### 3.1 Hydrostatics and geometry (leg A)

Worst block-coefficient error `0.0041`, worst principal-dimension / waterplane error `6.47e-04`.

| hull | V (m^3) | V reference | V error | Cb | Cb reference |
|---|---|---|---|---|---|
| Wigley | 0.5993 | 0.6000 | 0.0012 | 0.4439 | 0.4444 |
| Series 60 (idealised) | 9949.2626 | 9990.0000 | 0.0041 | 0.5976 | 0.6000 |
| KVLCC2 (idealised) | 311618.0359 | 312621.6704 | 0.0032 | 0.8072 | 0.8098 |

### 3.2 Kramers-Kronig causality (leg B)

Sweep `nu*a` in [0.010, 30.0], 80 geometric nodes, submerged circular section `a = 1.00 m`, `f = 2.00 m`.

| case | A(omega) residual | B(omega) shape residual | B constant offset | B raw residual |
|---|---|---|---|---|
| Ursell closed form (transform floor) | 9.98e-04 | 0.0065 | 0.0513 | 0.0535 |
| panel solve, 32 panels | 0.0010 | 0.0064 | 0.0511 | 0.0533 |
| panel solve, 64 panels | 1.00e-03 | 0.0064 | 0.0511 | 0.0533 |

Panel accuracy against Ursell's closed form over the same band: added mass `0.0228`, damping `0.0282` at 64 panels. The KK residual of the panel sweep is *not larger* than the transform floor measured on the exact pair, which is the statement the gate can support.

### 3.3 Section x depth x frequency matrix (legs C and E)

45 cases, 80 panels per section.

| hull | h/T | nu*T | lambda/T | A33 (kg) | B33 (kg/s) | B far-field | diffraction energy | radiation energy |
|---|---|---|---|---|---|---|---|---|
| Wigley | 1.50 | 0.30 | 12.991 | 2332.014 | 97.186 | 98.349 | 4.14e-07 | 0.0120 |
| Wigley | 1.50 | 0.60 | 8.443 | 2438.151 | 451.349 | 456.270 | 2.81e-06 | 0.0109 |
| Wigley | 1.50 | 1.00 | 5.811 | 2376.180 | 1304.957 | 1318.701 | 1.30e-05 | 0.0105 |
| Wigley | 1.50 | 1.50 | 4.105 | 1985.234 | 2218.599 | 2241.738 | 3.91e-05 | 0.0104 |
| Wigley | 1.50 | 2.00 | 3.126 | 1618.784 | 2359.547 | 2384.223 | 6.97e-05 | 0.0105 |
| Wigley | 3.00 | 0.30 | 16.886 | 2333.704 | 148.101 | 149.816 | 1.06e-07 | 0.0116 |
| Wigley | 3.00 | 0.60 | 10.000 | 2433.943 | 764.266 | 772.391 | 1.74e-06 | 0.0106 |
| Wigley | 3.00 | 1.00 | 6.253 | 2125.344 | 1828.134 | 1847.106 | 1.32e-05 | 0.0104 |
| Wigley | 3.00 | 1.50 | 4.188 | 1638.375 | 2336.784 | 2361.003 | 4.18e-05 | 0.0104 |
| Wigley | 3.00 | 2.00 | 3.142 | 1355.156 | 2231.705 | 2254.985 | 7.18e-05 | 0.0104 |
| Wigley | inf | 0.30 | 20.944 | 2352.801 | 223.383 | 220.822 | 4.44e-16 | 0.0115 |
| Wigley | inf | 0.60 | 10.472 | 2350.041 | 904.146 | 894.338 | 0.0000 | 0.0108 |
| Wigley | inf | 1.00 | 6.283 | 2021.181 | 1814.799 | 1795.548 | 0.0000 | 0.0106 |
| Wigley | inf | 1.50 | 4.189 | 1582.044 | 2272.737 | 2249.290 | 2.22e-16 | 0.0103 |
| Wigley | inf | 2.00 | 3.142 | 1315.110 | 2179.580 | 2157.934 | -1.11e-16 | 0.0099 |
| Series 60 | 1.50 | 0.30 | 12.991 | 12033.489 | 1502.013 | 1496.760 | 6.28e-07 | 0.0085 |
| Series 60 | 1.50 | 0.60 | 8.443 | 12711.260 | 8361.744 | 8358.947 | 3.85e-05 | 0.0085 |
| Series 60 | 1.50 | 1.00 | 5.811 | 7283.065 | 17900.427 | 17910.800 | 2.61e-04 | 0.0086 |
| Series 60 | 1.50 | 1.50 | 4.105 | 2958.460 | 14303.289 | 14307.866 | 5.93e-04 | 0.0086 |
| Series 60 | 1.50 | 2.00 | 3.126 | 2506.150 | 9127.388 | 9119.183 | 8.90e-04 | 0.0085 |
| Series 60 | 3.00 | 0.30 | 16.886 | 12197.689 | 2236.736 | 2231.063 | 1.60e-06 | 0.0084 |
| Series 60 | 3.00 | 0.60 | 10.000 | 11205.770 | 12983.400 | 12986.336 | 4.98e-05 | 0.0085 |
| Series 60 | 3.00 | 1.00 | 6.253 | 3670.781 | 17008.896 | 17021.701 | 2.69e-04 | 0.0086 |
| Series 60 | 3.00 | 1.50 | 4.188 | 1458.327 | 11630.306 | 11631.943 | 5.79e-04 | 0.0086 |
| Series 60 | 3.00 | 2.00 | 3.142 | 1508.924 | 7421.218 | 7411.529 | 8.77e-04 | 0.0085 |
| Series 60 | inf | 0.30 | 20.944 | 12285.966 | 3356.558 | 3365.563 | -1.11e-15 | 0.0087 |
| Series 60 | inf | 0.60 | 10.472 | 9426.480 | 13209.088 | 13223.425 | -5.55e-16 | 0.0095 |
| Series 60 | inf | 1.00 | 6.283 | 3363.926 | 15569.017 | 15578.595 | 6.66e-16 | 0.0107 |
| Series 60 | inf | 1.50 | 4.189 | 1377.638 | 11048.669 | 11034.683 | 1.55e-15 | 0.0112 |
| Series 60 | inf | 2.00 | 3.142 | 1396.798 | 7111.739 | 7075.013 | 6.66e-16 | 0.0101 |
| KVLCC2 | 1.50 | 0.30 | 12.991 | 19567.749 | 3625.677 | 3601.766 | 3.48e-06 | 0.0066 |
| KVLCC2 | 1.50 | 0.60 | 8.443 | 18167.855 | 20682.165 | 20613.516 | 1.71e-04 | 0.0055 |
| KVLCC2 | 1.50 | 1.00 | 5.811 | 5085.974 | 26484.035 | 26416.820 | 6.63e-04 | 0.0056 |
| KVLCC2 | 1.50 | 1.50 | 4.105 | 2428.083 | 15259.482 | 15208.085 | 0.0011 | 0.0058 |
| KVLCC2 | 1.50 | 2.00 | 3.126 | 3219.311 | 8542.141 | 8493.090 | 0.0017 | 0.0057 |
| KVLCC2 | 3.00 | 0.30 | 16.886 | 20084.763 | 5410.739 | 5379.704 | 7.09e-06 | 0.0057 |
| KVLCC2 | 3.00 | 0.60 | 10.000 | 12379.686 | 28025.142 | 27943.255 | 1.96e-04 | 0.0055 |
| KVLCC2 | 3.00 | 1.00 | 6.253 | 930.332 | 21581.607 | 21525.354 | 6.08e-04 | 0.0056 |
| KVLCC2 | 3.00 | 1.50 | 4.188 | 701.793 | 11841.675 | 11794.954 | 0.0011 | 0.0058 |
| KVLCC2 | 3.00 | 2.00 | 3.142 | 1770.869 | 6617.680 | 6572.995 | 0.0017 | 0.0068 |
| KVLCC2 | inf | 0.30 | 20.944 | 19935.399 | 8033.163 | 8082.551 | 3.97e-12 | 0.0061 |
| KVLCC2 | inf | 0.60 | 10.472 | 9328.442 | 25554.043 | 25678.176 | 1.55e-11 | 0.0072 |
| KVLCC2 | inf | 1.00 | 6.283 | 974.886 | 19772.908 | 19863.831 | 2.19e-11 | 0.0089 |
| KVLCC2 | inf | 1.50 | 4.189 | 624.924 | 11234.295 | 11247.936 | 2.71e-11 | 0.0086 |
| KVLCC2 | inf | 2.00 | 3.142 | 1604.701 | 6295.766 | 6257.881 | 3.22e-11 | 0.0060 |

Worst diffraction energy residual `0.0017`, worst radiation energy residual `0.0120`.

### 3.4 Finite depth to infinite depth (leg D)

Hull `Wigley`, `nu*T = 0.60`. Deep-water reference: A33 = 2350.0414 kg, B33 = 904.1465 kg/s.

| h/T | k0 h | A33 dev | B33 dev | sway A dev | sway B dev |
|---|---|---|---|---|---|
| 1.50 | 1.116 | 0.0375 | 0.5008 | 0.0535 | 0.3392 |
| 2.00 | 1.367 | 0.0338 | 0.3477 | 0.0458 | 0.0831 |
| 3.00 | 1.885 | 0.0357 | 0.1547 | 0.0164 | 0.0286 |
| 4.00 | 2.437 | 0.0276 | 0.0557 | 0.0029 | 0.0259 |
| 6.00 | 3.605 | 0.0116 | 2.00e-04 | 0.0011 | 0.0046 |
| 8.00 | 4.801 | 0.0051 | 0.0037 | 4.09e-04 | 3.56e-04 |
| 12.00 | 7.200 | 0.0017 | 0.0022 | 4.67e-05 | 5.72e-05 |
| 20.00 | 12.000 | 5.13e-04 | 0.0015 | 1.69e-06 | 2.49e-05 |

Fitted added-mass convergence slope in `h/T`: `-2.58` (the finite-depth correction is algebraic in the depth, as it must be for a wave-bearing limit).

### 3.5 Panel refinement (leg F)

Hull `KVLCC2 (idealised)`, `h/T = 1.50`, `nu*T = 2.00`.

| panels | diffraction energy residual | |R| | |T| | cond(A) |
|---|---|---|---|---|
| 40 | 0.0066 | 0.3152 | 0.9525 | 6.53 |
| 80 | 0.0017 | 0.3263 | 0.9462 | 6.63 |
| 120 | 7.53e-04 | 0.3293 | 0.9446 | 6.71 |

## 4. Figures

![energy identities](figures/fig_energy_identities.png)

*Fig. 1* -- diffraction energy residual (left) and radiation energy residual (right) for every case; markers are sections, colours are depth ratios, the dashed line is the gate.

![Kramers-Kronig](figures/fig_kramers_kronig.png)

*Fig. 2* -- the damping sweep against Ursell's closed form (left) and the Kramers-Kronig damping residual with the constant band-truncation term removed (right).

![deep water limit](figures/fig_deep_water_limit.png)

*Fig. 3* -- relative deviation from the deep-water solution against `h/T`.

![matrix](figures/fig_section_matrix.png)

*Fig. 4* -- heave added mass and damping of the three sections; colour is the depth ratio. The shallow case lifts the damping (the radiated wave carries less energy per unit elevation) and reduces the added mass.

## 5. Error verdict

| gate | quantity | threshold | measured | verdict |
|---|---|---|---|---|
| `G1_geometry` | hull Cb / L / B / T / waterplane area vs quadrature of the parametric definition | 0.01 relative | 0.004078 relative | PASS |
| `G2_kkr_added_mass` | KK relation (2): added mass predicted from the damping sweep | 0.02 relative | 0.0009999 relative | PASS |
| `G3_kkr_damping_shape` | KK relation (1): frequency-dependent part of the damping residual | 0.02 relative | 0.006408 relative | PASS |
| `G4_diffraction_energy` | \|R\|^2 + \|T\|^2 = 1 over the section x depth x frequency matrix | 0.01 relative | 0.001674 relative | PASS |
| `G5_radiation_energy` | damping from the radiated wave amplitude vs damping from the body pressure | 0.02 relative | 0.01197 relative | PASS |
| `G6_deep_water_limit` | finite-depth coefficients converge to the deep-water solution | 0.01 relative | 0.001456 relative | PASS |
| `G7_matrix_completeness` | 3 sections x 3 depths x 5 frequencies present | 45 cases | 45 cases | PASS |
| `G8_panel_refinement` | diffraction energy residual decreases with panel count | 0.01 relative | 0.006603 relative | PASS |

**Explicit answer to the 3 % question.** Every gated quantity is below 3 %; the worst is `0.0120` and the worst energy-identity residual is `0.0120`. The one number that exceeds 2 % is the *band-limited* KK damping residual (`0.0533`), which is the transform's own floor measured on the exact Ursell pair (`0.0535`) and not a solver error; it is reported here rather than hidden, and it is the reason the gate is stated on the frequency-dependent part.

## 6. Notes, limitations and what is not claimed

1. The wave legs use a superellipse idealisation of each midship section, rigidly submerged below the free surface. Real hull sections intersect the free surface; the section results are shape hydrodynamics, not ship RAOs, and no seakeeping prediction is claimed.
2. Series 60 and KVLCC2 are idealised parametric hulls built by tensorbem.pc6_hull calibrated to the published Cb (and, for KVLCC2, the published Lpp/B/T); they are not the official offset tables, so the geometry-fidelity error is unknown rather than zero.
3. The KK damping direction is band-limited: the truncated high-frequency band enters as an additive constant in B and is reported separately. The gate is applied to the frequency-dependent part; the raw number is reported as well.
4. Constant-panel collocation with the finite-depth kernel converges slowly on boxy sections (the source density is weakly singular at the bilge). The refinement leg measures the residual at the working mesh instead of assuming it.
5. All comparisons are against analytic or independent numerical references. No experimental data set ships with this repository, so the evidence level for the absolute engineering magnitudes remains WITHHELD.

Two further points worth recording for the next reader. First, the constant-panel collocation divides by the perpendicular distance from each panel to the collocation panel's line; collinear panels (a flat bottom) make that distance vanish, and the assembled matrix used to be NaN. The limit is zero and is now taken explicitly -- `tests/test_bem_finite_depth_anchors.py` locks that in. Second, the sections here must be *closed* contours: closing a half-section with one long chord panel turns the far-field amplitude quadrature into the dominant error (measured: energy residual 0.5-0.9 for a closed-on-a-chord section against 1e-4 for the same shape meshed with a distributed lid), which is a mesh rule, not a solver limitation.

## 7. Reproduce

```
PYTHONPATH=src python -m tensorbem.benchmarks.bem_finite_depth_anchors --stage all
PYTHONPATH=src python -m pytest tests/test_bem_finite_depth_anchors.py -q
```

The heavy legs can be run separately (`--stage hydro|kkr|matrix|asympt|refine`) and merged with `--stage merge`; each stage record is kept as `stage_<leg>.json` next to this report, with its wall-clock time.
