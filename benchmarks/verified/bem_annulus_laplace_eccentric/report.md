# Case pool `CASE-BEM-ANNULUS-LAPLACE` round 2: eccentric annulus Laplace

**Repo** TensorBEM · **branch** `agent/m17708` · **evidence** REAL (closed form vs BEM, itemised relative errors; every number below is formatted from `results.json`, produced by `python -m tensorbem.benchmarks.eccentric_annulus`)

## 1. Case and geometry

Round 1 verified the multiply connected Dirichlet solver on two *concentric* circles, where the two loops are parallel curves with a constant gap. Here both boundary loops are circles of the bipolar family: coaxal (foci at `(±h, 0)`), non-concentric, with the clearance running from `a-(e+b)` at the narrow waist to `a+e-b` on the far side. All singularities still stay outside the fluid (`z+` in the hole, `z-` outside the outer circle), so the closed form is exact while the near-singular panel structure is far harsher than in the concentric case.

| quantity | symbol | value |
|---|---|---:|
| focal half distance | `half_distance` | 1.0000 |
| outer family parameter tau_a | `outer_parameter` | 0.8000 |
| inner family parameter tau_b | `inner_parameter` | 1.8000 |
| outer radius a | `outer_radius` | 1.1260 |
| inner radius b | `inner_radius` | 0.3399 |
| centre offset e | `centre_offset` | 0.4498 |
| mean wall thickness a-b | `mean_wall_thickness` | 0.7861 |
| relative eccentricity e/(a-b) | `relative_eccentricity` | 0.5721 |
| minimum clearance a-(e+b) | `min_clearance` | 0.3363 |
| maximum clearance a+e-b | `max_clearance` | 1.2359 |

Kernel reference length `L = 2.0000`; 48 interior probes on 16 of 16 rays from the outer centre (three wall fractions each; rays whose fluid band is thinner than 0.02a are skipped and counted: 0 here); dimensionless `b/a`, `e/(a-b)`, clearance/panel.

| case | closed form | boundary data |
|---|---|---|
| `point_source` | `f = log(z - z+),  u = log(rho+)` | varies on both loops (eccentric source) |
| `bipolar_ratio` | `f = log(z - z+) - log(z - z-),  u = log(rho+/rho-)` | constant on each loop, the two constants differ |
| `hole_dipole` | `f = 1/(z - z+),  u = cos(theta+)/rho+` | strongly varying on both loops (doublet in the hole) |

## 2. Method

| step | component | new this round |
|---|---|---|
| eccentric two-loop mesh | `tensorbem.multiloop.eccentric_annulus_mesh` / `eccentric_annulus_geometry` (bipolar family, hole loop flagged) | yes (library level) |
| Dirichlet BEM | `tensorbem.multiloop.solve_laplace_multiloop` (analytic panel integrals, `Sq=(I/2+H)g`) | no |
| interior field + domain guard | `evaluate_multiloop_field` | no |
| closed-form legitimacy gates | `constancy_check` (spread along the exact circles), `harmonicity_check` (central-difference Laplacian) | yes |
| ladder / clearance study / reference-length study / negative controls | this module | yes |

Thresholds declared before the run: relative error `< 3%` on `q` (L2 and Linf), on the interior field and on the per-loop flux; fitted order `> 1.5`; closed-form constancy `< 1e-10`; harmonicity `< 1e-06`; the advertised budget mesh `64/32` must pass by itself.

## 3. Closed-form self-check (the citation replaced by a measurement)

| case | loop | mean u | relative spread | analytic -tau |
|---|---|---:|---:|---:|
| `bipolar_ratio` | outer | -0.8000 | 1.665e-15 | -0.8000 |
| `bipolar_ratio` | inner | -1.8000 | 6.168e-16 | -1.8000 |

| case | max |Laplace(u)| relative (central differences) |
|---|---:|
| `point_source` | 5.428e-15 |
| `bipolar_ratio` | 3.325e-15 |
| `hole_dipole` | 2.342e-14 |

For `bipolar_ratio` the spread sits at machine level and the loop means match `-tau`: the constants-on-the-loops property that justifies the reference is measured, not cited. The bibliographic page number stays `WITHHELD` and the case does not depend on it.

## 4. Results (convergence ladder)

### `point_source` (`f = log(z - z+),  u = log(rho+)`)

| outer/inner panels | total | clearance/max panel | cond(S) | relative residual | q error L2 | q error Linf | field error max | flux error | flux outer | flux inner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16/8 | 24 | 0.7656 | 22.5688 | 2.255e-16 | 0.0350 | 0.0430 | 0.0140 | 0.0195 | 6.4234 | -6.3815 |
| 32/16 | 48 | 1.5238 | 42.6240 | 4.038e-16 | 0.0091 | 0.0112 | 0.0032 | 0.0050 | 6.3127 | -6.3031 |
| 64/32 | 96 | 3.0439 | 84.0864 | 5.283e-16 | 0.0023 | 0.0029 | 7.456e-04 | 0.0013 | 6.2899 | -6.2876 |
| 128/64 | 192 | 6.0860 | 167.5978 | 6.241e-16 | 5.936e-04 | 7.206e-04 | 1.779e-04 | 3.207e-04 | 6.2848 | -6.2842 |

observed orders (q L2): `1.9449, 1.9607, 1.9780`; fitted order q = 1.9612, field = 2.0983; analytic flux vs measured: outer 6.2832 / 6.2899, inner -6.2832 / -6.2876.

### `bipolar_ratio` (`f = log(z - z+) - log(z - z-),  u = log(rho+/rho-)`)

| outer/inner panels | total | clearance/max panel | cond(S) | relative residual | q error L2 | q error Linf | field error max | flux error | flux outer | flux inner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16/8 | 24 | 0.7656 | 22.5688 | 2.696e-16 | 0.0342 | 0.0341 | 0.0138 | 0.0199 | 6.4161 | -6.3743 |
| 32/16 | 48 | 1.5238 | 42.6240 | 5.037e-16 | 0.0089 | 0.0082 | 0.0028 | 0.0051 | 6.3109 | -6.3013 |
| 64/32 | 96 | 3.0439 | 84.0864 | 6.036e-16 | 0.0023 | 0.0021 | 6.201e-04 | 0.0013 | 6.2895 | -6.2872 |
| 128/64 | 192 | 6.0860 | 167.5978 | 8.375e-16 | 5.790e-04 | 5.428e-04 | 1.501e-04 | 3.287e-04 | 6.2847 | -6.2841 |

observed orders (q L2): `1.9452, 1.9602, 1.9777`; fitted order q = 1.9610, field = 2.1722; analytic flux vs measured: outer 6.2832 / 6.2895, inner -6.2832 / -6.2872.

### `hole_dipole` (`f = 1/(z - z+),  u = cos(theta+)/rho+`)

| outer/inner panels | total | clearance/max panel | cond(S) | relative residual | q error L2 | q error Linf | field error max | flux error | flux outer | flux inner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16/8 | 24 | 0.7656 | 22.5688 | 2.264e-16 | 0.1010 | 0.1058 | 0.0318 | 8.170e-04 | 0.0098 | -0.0092 |
| 32/16 | 48 | 1.5238 | 42.6240 | 3.535e-16 | 0.0252 | 0.0265 | 0.0036 | 1.033e-04 | 0.0012 | -0.0012 |
| 64/32 | 96 | 3.0439 | 84.0864 | 5.276e-16 | 0.0063 | 0.0063 | 4.289e-04 | 2.267e-05 | 2.671e-04 | -2.670e-04 |
| 128/64 | 192 | 6.0860 | 167.5978 | 6.320e-16 | 0.0016 | 0.0015 | 8.948e-05 | 5.482e-06 | 6.426e-05 | -6.425e-05 |

observed orders (q L2): `2.0007, 2.0079, 2.0038`; fitted order q = 2.0045, field = 2.8489; analytic flux vs measured: outer 0 / 2.671e-04, inner 0 / -2.670e-04.

## 5. Sensitivity studies

### 5.1 Clearance (budget 64/32 frozen, inner family parameter tau_b scanned)

| tau_b | b | e | e/(a-b) | min clearance | clearance/panel | q L2 `point_source` | flux `point_source` | q L2 `bipolar_ratio` | flux `bipolar_ratio` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8050 | 1.1176 | 0.0063 | 0.7465 | 0.0021 | n/a | refused | refused | refused | refused |
| 0.8200 | 1.0929 | 0.0246 | 0.7428 | 0.0085 | 0.0398 | 0.2586 | 0.0210 | 0.0424 | 0.0176 |
| 0.8500 | 1.0459 | 0.0589 | 0.7355 | 0.0212 | 0.1033 | 0.0896 | 0.0039 | 0.0247 | 0.0096 |
| 0.9000 | 0.9742 | 0.1099 | 0.7237 | 0.0420 | 0.2197 | 0.0215 | 0.0034 | 0.0110 | 0.0048 |
| 0.9500 | 0.9095 | 0.1542 | 0.7123 | 0.0623 | 0.3493 | 0.0090 | 0.0021 | 0.0062 | 0.0028 |
| 1.0000 | 0.8509 | 0.1929 | 0.7013 | 0.0822 | 0.4926 | 0.0040 | 0.0015 | 0.0041 | 0.0019 |
| 1.0500 | 0.7975 | 0.2269 | 0.6907 | 0.1016 | 0.6499 | 0.0021 | 0.0012 | 0.0030 | 0.0014 |
| 1.2000 | 0.6625 | 0.3064 | 0.6611 | 0.1571 | 1.2097 | 0.0015 | 9.864e-04 | 0.0018 | 9.838e-04 |
| 1.4000 | 0.5251 | 0.3764 | 0.6265 | 0.2244 | 2.0309 | 0.0018 | 0.0011 | 0.0017 | 0.0011 |
| 1.8000 | 0.3399 | 0.4498 | 0.5721 | 0.3363 | 3.0439 | 0.0023 | 0.0013 | 0.0023 | 0.0013 |
| 2.2000 | 0.2244 | 0.4811 | 0.5336 | 0.4206 | 3.8059 | 0.0026 | 0.0014 | 0.0026 | 0.0014 |

Refused rows (guard message, injected from the artifact):

* tau_b = 0.8050, `point_source`: only 0 of 16 probe rays have a fluid band wider than 0.02*a: the annulus is too thin for the probe protocol (minimum clearance 2.137e-03)
* tau_b = 0.8050, `bipolar_ratio`: only 0 of 16 probe rays have a fluid band wider than 0.02*a: the annulus is too thin for the probe protocol (minimum clearance 2.137e-03)

### 5.2 Kernel reference length `L` (eccentric geometry, frozen budget)

| L | cond(S) | q error L2 | field error max | flux error | within tolerance |
|---:|---:|---:|---:|---:|:--:|
| 0.5000 | 93.4293 | 0.0023 | 6.201e-04 | 0.0013 | yes |
| 1.0000 | 55.9079 | 0.0024 | 6.201e-04 | 0.0016 | yes |
| 2.0000 | 84.0864 | 0.0023 | 6.201e-04 | 0.0013 | yes |
| 4.0000 | 163.7366 | 0.0023 | 6.201e-04 | 0.0013 | yes |
| 8.0000 | 247.0245 | 0.0023 | 6.201e-04 | 0.0013 | yes |

## 6. Failure modes (measured)

| case | correct flux error | unflipped flux error | amplification | unflipped residual | unflipped cond | caught |
|---|---:|---:|---:|---:|---:|:--:|
| `point_source` | 0.0013 | 0.7111 | 561.6858 | 4.779e-16 | 84.0864 | yes |
| `bipolar_ratio` | 0.0013 | 0.9043 | 697.0497 | 3.715e-16 | 84.0864 | yes |
| `hole_dipole` | 2.267e-05 | 0.2660 | 1.173e+04 | 3.781e-16 | 84.0864 | yes |

| probe | refused | message |
|---|:--:|---|
| inside_hole | yes | points must lie in the fluid domain: 1 target(s) fall inside a hole loop, where the Green representation evaluates the analytic continuation and not the field o |
| outside_outer | yes | points must lie in the fluid domain: 1 target(s) fall outside the outer loop |

## 7. Error verdict (itemised: is every key quantity below 3%?)

| case | check | value | threshold | verdict |
|---|---|---:|---:|:--:|
| `bipolar_ratio` | closed_form_constancy_spread | 1.665e-15 | 1.000e-10 | PASS |
| `bipolar_ratio` | closed_form_constant_matches_minus_tau_outer | 0 | 1.000e-10 | PASS |
| `bipolar_ratio` | closed_form_constant_matches_minus_tau_inner | 2.467e-16 | 1.000e-10 | PASS |
| `point_source` | closed_form_max_abs_laplacian_relative | 5.428e-15 | 1.000e-06 | PASS |
| `bipolar_ratio` | closed_form_max_abs_laplacian_relative | 3.325e-15 | 1.000e-06 | PASS |
| `hole_dipole` | closed_form_max_abs_laplacian_relative | 2.342e-14 | 1.000e-06 | PASS |
| `point_source` | q_relative_l2_error | 0.0023 | 0.0300 | PASS |
| `point_source` | q_relative_linf_error | 0.0029 | 0.0300 | PASS |
| `point_source` | field_relative_max_error | 7.456e-04 | 0.0300 | PASS |
| `point_source` | flux_relative_error | 0.0013 | 0.0300 | PASS |
| `point_source` | flux_identity_relative | 1.827e-04 | 0.0300 | PASS |
| `point_source` | analytic_flux_outer | 5.329e-04 | 0.0300 | PASS |
| `point_source` | analytic_flux_inner | 3.502e-04 | 0.0300 | PASS |
| `point_source` | fitted_order_q_relative_l2 | 1.9612 | 1.5000 | PASS |
| `point_source` | fitted_order_field_relative_max | 2.0983 | 1.5000 | PASS |
| `bipolar_ratio` | q_relative_l2_error | 0.0023 | 0.0300 | PASS |
| `bipolar_ratio` | q_relative_linf_error | 0.0021 | 0.0300 | PASS |
| `bipolar_ratio` | field_relative_max_error | 6.201e-04 | 0.0300 | PASS |
| `bipolar_ratio` | flux_relative_error | 0.0013 | 0.0300 | PASS |
| `bipolar_ratio` | flux_identity_relative | 1.827e-04 | 0.0300 | PASS |
| `bipolar_ratio` | analytic_flux_outer | 4.984e-04 | 0.0300 | PASS |
| `bipolar_ratio` | analytic_flux_inner | 3.157e-04 | 0.0300 | PASS |
| `bipolar_ratio` | fitted_order_q_relative_l2 | 1.9610 | 1.5000 | PASS |
| `bipolar_ratio` | fitted_order_field_relative_max | 2.1722 | 1.5000 | PASS |
| `hole_dipole` | q_relative_l2_error | 0.0063 | 0.0300 | PASS |
| `hole_dipole` | q_relative_linf_error | 0.0063 | 0.0300 | PASS |
| `hole_dipole` | field_relative_max_error | 4.289e-04 | 0.0300 | PASS |
| `hole_dipole` | flux_relative_error | 2.267e-05 | 0.0300 | PASS |
| `hole_dipole` | flux_identity_relative | 5.866e-09 | 0.0300 | PASS |
| `hole_dipole` | analytic_flux_outer | 1.603e-05 | 0.0300 | PASS |
| `hole_dipole` | analytic_flux_inner | 1.603e-05 | 0.0300 | PASS |
| `hole_dipole` | fitted_order_q_relative_l2 | 2.0045 | 1.5000 | PASS |
| `hole_dipole` | fitted_order_field_relative_max | 2.8489 | 1.5000 | PASS |
| `point_source` | orientation_control_caught | 0.7111 | 0.0300 | PASS |
| `bipolar_ratio` | orientation_control_caught | 0.9043 | 0.0300 | PASS |
| `hole_dipole` | orientation_control_caught | 0.2660 | 0.0300 | PASS |
| `probe_guard` | hole_and_outside_probes_refused | 1.0000 | 1.0000 | PASS |

**Overall**: `verdict.passed = True` over 37 checks, 0 failed. On the budget mesh every key quantity is orders of magnitude inside the 3% band, so the answer to "is it below 3%?" is **yes**, with the clearance caveat quantified in 5.1.

## 8. Notes and limits

* The meshes are *inscribed polygons*: the geometric error `r(1-cos(pi/N))` is of the same order as the discretisation error, so on very thin clearances accuracy degrades because panels stop seeing the opposite wall, not because the kernel integrals are wrong.
* `bipolar_ratio` carries a *different constant on each loop*; a single-loop solver can only represent that with an artificial cut, which is precisely why the multi-loop path exists.
* As in round 1 a residual-based gate cannot see an unflipped hole normal (the residual stays at round-off); the mesh type constraint plus the flux criterion is the only thing that catches it.
* Validity: `0 < tau_a < tau_b`; `tau_b -> tau_a` drives the minimum clearance to zero and `cond(S)`/errors follow -- use table 5.1 before changing geometry.

