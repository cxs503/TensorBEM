# Cross-code verification of the 3-D free-surface BEM solver against capytaine

*Anchor **FEAT-OSS-INVENTORY-BEM** (deep-dive wave: the Tier-1 repository `capytaine` stops being a catalogue entry and becomes an executable reference).  Every number in this document is rendered from `benchmarks/verified/capytaine_crosscheck/results.json` by `tensorbem.benchmarks.capytaine_crosscheck.render_document`; `tests/test_capytaine_crosscheck.py` re-runs the study and fails if code, artifacts and prose disagree.*

**Verdict: FAIL** -- worst deviation from the independent code 8.75 % against a fixed 5 % band.

## 1. Case, geometry and reference

| item | value |
|---|---|
| body A | surface-piercing hemisphere, R = 1.0 m, displaced volume 2/3*pi*R^3 = 2.0944 m^3 |
| body B | floating sphere R = 1.0 m, centre z = -0.5 m, waterline radius sqrt(3)/2 = 0.8660 m |
| fluid | rho = 1025.0 kg/m^3, g = 9.80665 m/s^2, infinite depth |
| frequencies | nu = 0.5, nu = 1.0, nu = 2.0 (nu = omega^2/g, m^-1) |
| independent code | capytaine 3.1.dev0 (commit `c0516db5b0df`), loongarch64, Python 3.11.6 |
| independent code meshes | finest capytaine level: 1296 panels (hemisphere), 1120 panels (sphere) |
| TensorBEM meshes | hemisphere 64 / 256 panels; sphere 270 panels |
| third axis | Hulme (1982) semi-analytical multipole solution for the hemisphere (`tensorbem.analytic.hulme_hemisphere_radiation`) |

## 2. Method

1. `scripts/capytaine_crosscheck_ref.py` builds capytaine from source, mesh-converges it on the two smooth bodies and freezes the values, the panel counts and the provenance (version, git commit, platform) into `capytaine_ref.json`.  capytaine is *not* imported anywhere in `src/` or `tests/`.
2. `run_study` solves the same radiation problems with `tensorbem.free_surface_solver3d.solve_radiation_3d` on `tensorbem.geometry3d.hemisphere_hull_mesh` / `floating_sphere_hull_mesh` at three (resp. two) hull densities.
3. `compare` takes the finest TensorBEM level, forms (TensorBEM - capytaine)/capytaine for every (nu, dof, added mass|damping) cell and applies the band fixed before the run: **5 %**.
4. `convergence_vs_capytaine` repeats the comparison for *every* mesh level so the panel-method convergence toward the foreign reference is visible.

## 3. Results

### 3.1 Hemisphere, finest TensorBEM level vs the two references

| nu | dof | quantity | Hulme 1982 | capytaine | TensorBEM | dev. vs capytaine |
|---:|---|---|---:|---:|---:|---:|
| 0.5 | heave | added mass | 1258.26 | 1274.64 | 1252.61 | -1.73% |
| 0.5 | heave | damping | 1611.93 | 1619.61 | 1609.79 | -0.61% |
| 0.5 | surge | added mass | 1382.29 | 1414.78 | 1356.92 | -4.09% |
| 0.5 | surge | damping | 469.04 | 483.26 | 450.87 | -6.70% |
| 1.0 | heave | added mass | 919.80 | 934.83 | 913.04 | -2.33% |
| 1.0 | heave | damping | 1670.20 | 1668.21 | 1689.78 | +1.29% |
| 1.0 | surge | added mass | 1232.13 | 1254.89 | 1216.07 | -3.09% |
| 1.0 | surge | damping | 2376.39 | 2431.08 | 2292.57 | -5.70% |
| 2.0 | heave | added mass | 833.77 | 848.40 | 816.45 | -3.77% |
| 2.0 | heave | damping | 980.27 | 944.00 | 1024.29 | +8.51% |
| 2.0 | surge | added mass | 535.14 | 547.04 | 539.72 | -1.34% |
| 2.0 | surge | damping | 3255.41 | 3286.11 | 3182.40 | -3.16% |

### 3.2 Floating sphere (heave)

| nu | quantity | capytaine | TensorBEM | dev. |
|---:|---|---:|---:|---:|
| 0.5 | added mass | 1157.38 | 1117.05 | -3.48% |
| 0.5 | damping | 275.27 | 251.18 | -8.75% |
| 1.0 | added mass | 1229.71 | 1189.02 | -3.31% |
| 1.0 | damping | - | - | skipped (skipped: geometrically vanishing damping) |
| 2.0 | added mass | 1284.03 | 1234.08 | -3.89% |
| 2.0 | damping | 417.30 | 423.49 | +1.49% |

### 3.3 Convergence toward the foreign reference (worst cell per level)

| case | panels | worst dev. vs capytaine |
|---|---:|---:|
| hemisphere_r1 | 64 | 22.10 % |
| hemisphere_r1 | 256 | 8.51 % |
| sphere_r1_z-0.5 | 270 | 8.75 % |

### 3.4 Negative control: wall-sided barge

* `barge_L2_B1_T0.5` -- wall-sided barge with sharp bilge/waterline corners never leaves the spurious irregular-frequency regime: even at 576 panels the heave added mass and damping change sign between frequencies, so no converged reference exists to compare against. capytaine's own finest level (576 panels) still returns nu=0.500: heave A33=1359.5, B33=-609.3, nu=1.000: heave A33=1989.8, B33=-1398.4, nu=2.000: heave A33=-1795.2, B33=252.4. No converged reference exists, so the case is excluded instead of being compared against noise.

## 4. Figures

* `docs/fig_capytaine_crosscheck_frequency.png` -- added mass and damping vs frequency, TensorBEM vs capytaine vs Hulme.
* `docs/fig_capytaine_crosscheck_convergence.png` -- worst deviation from capytaine vs panel count.

## 5. Caveats, and what this does *not* show

* Two codes agreeing does not prove either is right (shared modelling assumptions: linearised potential flow, deep water, rigid body, no viscosity).  The hemisphere column also carries Hulme's analytic values, which is what makes the agreement meaningful there; the sphere case is code-to-code only.
* Mesh families differ (TensorBEM: geodesic octahedral subdivision, triangles; capytaine: lat-long spherical quads clipped at z = 0), so the comparison is made at converged/finest levels, not panel-by-panel.
* The floating-sphere heave *damping* is skipped at nu = 1 where it vanishes geometrically; comparing a relative error against ~0 is meaningless grading, not a pass.
* Sharp-corner geometries remain untreated (see 3.4): this is a limitation of both codes, not evidence of agreement.

## 6. Error verdict

* `hemisphere_r1`: worst deviation 8.51 % over 12 cells -> **FAIL** (band 5 %).
* `sphere_r1_z-0.5`: worst deviation 8.75 % over 5 cells -> **FAIL** (band 5 %).

**Is the key quantity within 3 %?** No -- the worst cell is 8.75 %, and the acceptance band applied here is 5 % because each code's own level-to-level self-convergence at the finest mesh is of the order of 2-4 %; quoting 3 % would pretend to a resolution the panel methods do not have. The 3 % statement is therefore reported honestly as *not* met, with the measured numbers above.

