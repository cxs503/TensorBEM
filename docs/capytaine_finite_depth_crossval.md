# Finite-depth cross-validation against capytaine
**Scope**: code-to-code verification of `tensorbem.finite_depth_solver3d` in water of *finite* depth against capytaine's finite-depth Green function (the existing `docs/capytaine_crosscheck.md` is infinite depth only).
**Plan**: `tensorbem.benchmarks.finite_depth_crossval_plan` — submerged sphere `a = 1.0 m`, centre `z = -1.5 m`, grid `h/a in [3.0, 5.0, 10.0, 20.0]` x `k0 a in [0.5, 1.0, 2.0]`, `nu` back-computed per cell from `k0 tanh(k0 h) = nu` so `k0 a` is exact at every depth.
**Reference**: `scripts/capytaine_finite_depth_ref.py` (capytaine 3.1.dev0, `finite_depth_method=newer`, ladder [[12, 24], [24, 48], [48, 96]] panels, generated 2026-11-14T07:49:35+00:00).
**Execution status**: `PARTIAL: 1 of 12 planned cells`; every number below is written by the study driver from the frozen artifacts, none is transcribed.
## 1. Headline
| quantity | worst extrapolated deviation | pre-registered limit | owner limit |
|---|---:|---:|---:|
| heave added mass | +12.75 % | 3.00 % | 15 % |
| surge added mass | +7.52 % | 3.00 % | 15 % |
| heave damping | +23.85 % | 5.00 % | 15 % |
| surge damping | +22.74 % | 5.00 % | 15 % |

## 2. Gates
| gate | measured | limit | verdict |
|---|---:|---:|---|
| `added_mass_agreement` | 0.1275 | 0.03 | FAIL |
| `damping_agreement` | 0.2385 | 0.05 | FAIL |
| `negative_control_separates_depth` | 0.0701 | 0.02 | PASS |
| `reference_ladder_in_asymptotic_range` | 0.2412 | 0.3 | FAIL |
| `tensorbem_ladder_in_asymptotic_range` | no | 0.3 | FAIL |
| `linear_solve_residuals` | 0.0000 | 1e-08 | PASS |
| `reference_is_finite_depth` | yes | True | PASS |

## 3. Per-cell comparison (Richardson-extrapolated)
| `h/a` | `k0 a` | `k0 h` | A33 TB | A33 capytaine | delta | B33 TB | B33 capytaine | delta | A11 TB | A11 capytaine | delta | B11 TB | B11 capytaine | delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.50 | 1.50 | 2617.3245 | 2999.9168 | -12.75 % | 564.2107 | 740.8990 | -23.85 % | 2276.4836 | 2461.5931 | -7.52 % | 616.0365 | 797.4058 | -22.74 % |

Observed convergence orders (same three-level ladder):
| `h/a` | `k0 a` | p (A33, TB) | p (A33, capytaine) | p (B33, TB) | p (B33, capytaine) |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.50 | n/a | n/a | n/a | n/a |

## 4. Negative control (the plan must see depth)
| `h/a` | `k0 a` | `k0 h` | (A33 finite − A33 deep)/A33 deep |
|---:|---:|---:|---:|
| 3 | 0.50 | 1.50 | +7.01 % |

## 5. Notes, limitations
* Both codes discretise the sphere as an inscribed polyhedron, so the same geometric bias (panel area deficit ~1-2 % at the coarse levels) enters both sides; the extrapolation removes it to first order, and the reported order says how much of the residual is discretisation.
* capytaine's two finite-depth formulations (`newer` vs `legacy`) agree to round-off on this geometry, so they do **not** provide an independent second opinion; the reference uncertainty is the panel ladder, which is why the ladder is reported per cell instead of a single tolerance.
* Submerged closed body only: no waterline, so no irregular-frequency treatment is exercised here; the surface-piercing finite-depth case remains out of scope (`docs/finite_depth_solver3d.md`).
* Damping involves the radiated-wave amplitude and converges more slowly than added mass on both sides; the per-cell orders above make that visible rather than hiding it behind one number.
* Measured order available on 1/4 of the judged code/cell entries; where the reference ladder is not yet asymptotic (`n/a` order) the comparison uses the finest reference level, so that entry's benchmark is the least certain number in this study.
