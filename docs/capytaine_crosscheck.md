# Cross-code verification of the 3-D free-surface BEM solver against capytaine

*Anchor **FEAT-OSS-INVENTORY-BEM** (deep-dive wave: the Tier-1 repository `capytaine` stops being a catalogue entry and becomes an executable reference).  Every number in this document is rendered from `benchmarks/verified/capytaine_crosscheck/results.json` by `tensorbem.benchmarks.capytaine_crosscheck.render_document`; `tests/test_capytaine_crosscheck.py` re-runs the study and fails if code, artifacts and prose disagree.*

**Verdict: PASS** -- worst deviation from the independent code 4.97 % against a fixed 5 % band.

## 1. Case, geometry and reference

| item | value |
|---|---|
| body A | surface-piercing hemisphere, R = 1.0 m, displaced volume 2/3*pi*R^3 = 2.0944 m^3 |
| body B | floating sphere R = 1.0 m, centre z = -0.5 m, waterline radius sqrt(3)/2 = 0.8660 m |
| fluid | rho = 1025.0 kg/m^3, g = 9.80665 m/s^2, infinite depth |
| frequencies | nu = 0.5, nu = 1.0, nu = 2.0 (nu = omega^2/g, m^-1) |
| independent code | capytaine 3.1.dev0 (commit `c0516db5b0df`), loongarch64, Python 3.11.6 |
| independent code meshes | finest capytaine level: 1296 panels (hemisphere), 1120 panels (sphere) |
| TensorBEM meshes | hemisphere 64 / 256 / 1024 panels; sphere 270 / 1026 panels |
| third axis | Hulme (1982) semi-analytical multipole solution for the hemisphere (`tensorbem.analytic.hulme_hemisphere_radiation`) |

## 2. Method

1. `scripts/capytaine_crosscheck_ref.py` builds capytaine from source, mesh-converges it on the two smooth bodies and freezes the values, the panel counts and the provenance (version, git commit, platform) into `capytaine_ref.json`.  capytaine is *not* imported anywhere in `src/` or `tests/`.
2. `run_study` solves the same radiation problems with `tensorbem.free_surface_solver3d.solve_radiation_3d` on `tensorbem.geometry3d.hemisphere_hull_mesh` / `floating_sphere_hull_mesh` at the 3 (resp. 2) hull densities fixed by `HEMISPHERE_LEVELS` / `SPHERE_LEVELS` -- the counts below are read from the study, not typed into this paragraph.
3. `compare` takes the finest TensorBEM level, forms (TensorBEM - capytaine)/capytaine for every (nu, dof, added mass|damping) cell and applies the band fixed before the run: **5 %**.
4. `convergence_vs_capytaine` repeats the comparison for *every* mesh level so the panel-method convergence toward the foreign reference is visible.

## 3. Results

### 3.1 Hemisphere, finest TensorBEM level vs the two references

| nu | dof | quantity | Hulme 1982 | capytaine | TensorBEM | dev. vs capytaine |
|---:|---|---|---:|---:|---:|---:|
| 0.5 | heave | added mass | 1258.26 | 1274.64 | 1256.81 | -1.40% |
| 0.5 | heave | damping | 1611.93 | 1619.61 | 1611.29 | -0.51% |
| 0.5 | surge | added mass | 1382.29 | 1414.78 | 1375.88 | -2.75% |
| 0.5 | surge | damping | 469.04 | 483.26 | 464.41 | -3.90% |
| 1.0 | heave | added mass | 919.80 | 934.83 | 918.07 | -1.79% |
| 1.0 | heave | damping | 1670.20 | 1668.21 | 1674.94 | +0.40% |
| 1.0 | surge | added mass | 1232.13 | 1254.89 | 1228.08 | -2.14% |
| 1.0 | surge | damping | 2376.39 | 2431.08 | 2354.95 | -3.13% |
| 2.0 | heave | added mass | 833.77 | 848.40 | 829.35 | -2.25% |
| 2.0 | heave | damping | 980.27 | 944.00 | 990.95 | +4.97% |
| 2.0 | surge | added mass | 535.14 | 547.04 | 536.36 | -1.95% |
| 2.0 | surge | damping | 3255.41 | 3286.11 | 3236.59 | -1.51% |

### 3.2 Floating sphere (heave)

| nu | quantity | capytaine | TensorBEM | dev. |
|---:|---|---:|---:|---:|
| 0.5 | added mass | 1157.38 | 1129.40 | -2.42% |
| 0.5 | damping | 275.27 | 267.33 | -2.88% |
| 1.0 | added mass | 1229.71 | 1200.52 | -2.37% |
| 1.0 | damping | - | - | skipped (skipped: geometrically vanishing damping) |
| 2.0 | added mass | 1284.03 | 1252.54 | -2.45% |
| 2.0 | damping | 417.30 | 401.01 | -3.90% |

### 3.3 Convergence toward the foreign reference (worst cell per level)

| case | panels | worst dev. vs capytaine |
|---|---:|---:|
| hemisphere_r1 | 64 | 22.10 % |
| hemisphere_r1 | 256 | 8.51 % |
| hemisphere_r1 | 1024 | 4.97 % |
| sphere_r1_z-0.5 | 270 | 8.75 % |
| sphere_r1_z-0.5 | 1026 | 3.90 % |

### 3.4 Negative control: wall-sided barge

* `barge_L2_B1_T0.5` -- wall-sided barge with sharp bilge/waterline corners never leaves the spurious irregular-frequency regime: even at 576 panels the heave added mass and damping change sign between frequencies, so no converged reference exists to compare against. capytaine's own finest level (576 panels) still returns nu=0.500: heave A33=1359.5, B33=-609.3, nu=1.000: heave A33=1989.8, B33=-1398.4, nu=2.000: heave A33=-1795.2, B33=252.4. No converged reference exists, so the case is excluded instead of being compared against noise.

### 3.5 Who owns the residual gap? Three-way error decomposition

Code-to-code agreement alone cannot say *who* is off.  The hemisphere has an independent semi-analytical solution (Hulme 1982), so every cell is scored twice: each code against Hulme.  The numbers below are rendered from `study['analytic_decomposition']`.

| nu | dof | quantity | Hulme 1982 | capytaine (dev. vs Hulme) | TensorBEM (dev. vs Hulme) | code-to-code |
|---:|---|---|---:|---:|---:|---:|
| 0.5 | heave | added mass | 1258.26 | 1274.64 (+1.30 %) | 1256.81 (-0.12 %) | -1.40 % |
| 0.5 | heave | damping | 1611.93 | 1619.61 (+0.48 %) | 1611.29 (-0.04 %) | -0.51 % |
| 0.5 | surge | added mass | 1382.29 | 1414.78 (+2.35 %) | 1375.88 (-0.46 %) | -2.75 % |
| 0.5 | surge | damping | 469.04 | 483.26 (+3.03 %) | 464.41 (-0.99 %) | -3.90 % |
| 1.0 | heave | added mass | 919.80 | 934.83 (+1.63 %) | 918.07 (-0.19 %) | -1.79 % |
| 1.0 | heave | damping | 1670.20 | 1668.21 (-0.12 %) | 1674.94 (+0.28 %) | +0.40 % |
| 1.0 | surge | added mass | 1232.13 | 1254.89 (+1.85 %) | 1228.08 (-0.33 %) | -2.14 % |
| 1.0 | surge | damping | 2376.39 | 2431.08 (+2.30 %) | 2354.95 (-0.90 %) | -3.13 % |
| 2.0 | heave | added mass | 833.77 | 848.40 (+1.75 %) | 829.35 (-0.53 %) | -2.25 % |
| 2.0 | heave | damping | 980.27 | 944.00 (-3.70 %) | 990.95 (+1.09 %) | +4.97 % |
| 2.0 | surge | added mass | 535.14 | 547.04 (+2.22 %) | 536.36 (+0.23 %) | -1.95 % |
| 2.0 | surge | damping | 3255.41 | 3286.11 (+0.94 %) | 3236.59 (-0.58 %) | -1.51 % |

* worst TensorBEM cell against Hulme: +1.09 % (2.0 heave damping);
* worst capytaine cell against Hulme: -3.70 % (2.0 heave damping);
* reading: where TensorBEM sits much closer to Hulme than capytaine does, the code-to-code residual belongs to the foreign code's discretisation, not to this solver.  The sphere case has no analytic axis and stays code-to-code only.

### 3.6 Superseded records (history is not deleted)

| level sets | verdict | worst deviation | failures |
|---|---|---:|---|
| hemisphere_r1: 64/256 panels, sphere_r1_z-0.5: 270 panels | **FAIL** | 8.75 % | hemisphere_r1 nu=0.50 surge damping: -6.70%; hemisphere_r1 nu=1.00 surge damping: -5.70%; hemisphere_r1 nu=2.00 heave damping: +8.51%; sphere_r1_z-0.5 nu=0.50 heave damping: -8.75% |

* why they are kept: a re-run at the declared mesh levels supersedes a verdict, it does not delete it.  The superseded record above is the number that was published before PASS was measured, and it stays in `results.json` under `superseded_records`.

## 4. Figures

* `docs/fig_capytaine_crosscheck_frequency.png` -- added mass and damping vs frequency, TensorBEM vs capytaine vs Hulme.
* `docs/fig_capytaine_crosscheck_convergence.png` -- worst deviation from capytaine vs panel count.

## 5. Caveats, and what this does *not* show

* Two codes agreeing does not prove either is right (shared modelling assumptions: linearised potential flow, deep water, rigid body, no viscosity).  The hemisphere column also carries Hulme's analytic values, which is what makes the agreement meaningful there; the sphere case is code-to-code only.
* Mesh families differ (TensorBEM: geodesic octahedral subdivision, triangles; capytaine: lat-long spherical quads clipped at z = 0), so the comparison is made at converged/finest levels, not panel-by-panel.
* The floating-sphere heave *damping* is skipped at nu = 1 where it vanishes geometrically; comparing a relative error against ~0 is meaningless grading, not a pass.
* Sharp-corner geometries remain untreated (see 3.4): this is a limitation of both codes, not evidence of agreement.

## 6. Error verdict

* `hemisphere_r1`: worst deviation 4.97 % over 12 cells -> **PASS** (band 5 %).
* `sphere_r1_z-0.5`: worst deviation 3.90 % over 5 cells -> **PASS** (band 5 %).

* against the analytic truth (Hulme 1982) the worst TensorBEM cell is +1.09 %, the worst capytaine cell is -3.70 %.
* the code-to-code band is met with a margin of only 0.03 percentage points: this pair of low-order panel methods is *at* its resolution limit here, and the verdict would flip on a finer foreign mesh or a slightly different band.  The claim made is therefore the narrow one -- the declared finest meshes agree inside the pre-registered band -- not that either code is converged to 1 %.

**Is the key quantity within 3 %?** two questions, two answers:

1. *this solver vs the analytic truth* -- worst TensorBEM cell against Hulme is +1.09 % -> **within 3 %**.
2. *code-to-code* -- worst cell is 4.97 % -> **not within 3 %**; that quantity is bounded by the foreign code's own error against the same analytic solution (-3.70 % for capytaine), so it is graded against the 5 % band fixed before the run, and the measured numbers are reported as they are, including the cells that miss.

