# Finite-depth cross-validation against capytaine
**Scope**: code-to-code verification of `tensorbem.finite_depth_solver3d` in water of *finite* depth against capytaine's finite-depth Green function (the existing `docs/capytaine_crosscheck.md` is infinite depth only).
**Plan**: `tensorbem.benchmarks.finite_depth_crossval_plan` — submerged sphere `a = 1.0 m`, centre `z = -1.5 m`, grid `h/a in [3.0, 5.0, 10.0, 20.0]` x `k0 a in [0.5, 1.0, 2.0]`, `nu` back-computed per cell from `k0 tanh(k0 h) = nu` so `k0 a` is exact at every depth.
**Reference**: `scripts/capytaine_finite_depth_ref.py` (capytaine 3.1.dev0, `finite_depth_method=newer`, ladder [[12, 24], [24, 48], [48, 96]] panels, generated 2026-11-15T12:55:54+00:00).
**Execution status**: `PARTIAL: 11 covered cells missing rungs ['0', '1', '2']; 4 confirmation cells missing the 3-subdivision rung`; every number below is written by the study driver from the frozen artifacts, none is transcribed.
## 1. Headline
| quantity | worst deviation on the pre-registered basis | pre-registered limit | owner limit |
|---|---:|---:|---:|
| heave added mass | +9.01 % | 3.00 % | 15 % |
| surge added mass | +6.17 % | 3.00 % | 15 % |
| heave damping | +22.05 % | 5.00 % | 15 % |
| surge damping | +21.70 % | 5.00 % | 15 % |

## 2. Gates
| gate | measured | limit | verdict |
|---|---:|---:|---|
| `added_mass_agreement` | 0.0901 | 0.03 | FAIL |
| `damping_agreement` | 0.2205 | 0.05 | FAIL |
| `negative_control_separates_depth` | 0.1705 | 0.02 | PASS |
| `reference_ladder_in_asymptotic_range` | 0.2412 | 0.3 | FAIL |
| `reference_truncation_within_share` | 0.0181 | 0.016666666666666666 | FAIL |
| `tensorbem_ladder_in_asymptotic_range` | 0.6141 | 0.3 | PASS |
| `extrapolation_self_consistent` | no | 1.0 | FAIL |
| `linear_solve_residuals` | 0.0000 | 1e-08 | PASS |
| `reference_is_finite_depth` | yes | True | PASS |

## 3. Per-cell comparison (Richardson-extrapolated, gated by the pre-registered order floor)
| `h/a` | `k0 a` | `k0 h` | A33 TB | A33 capytaine | delta | B33 TB | B33 capytaine | delta | A11 TB | A11 capytaine | delta | B11 TB | B11 capytaine | delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.50 | 1.50 | 2998.9644 | 2999.9145 | -0.03 % | 765.9767 | 740.8984 | +3.38 % | 2545.3650 | 2566.4945 | -0.82 % | 814.3843 | 797.4057 | +2.13 % |
| 3 | 1.00 | 3.00 | 2129.3379 | 2340.1447 | -9.01 % | 2447.0447 | 3139.2583 | -22.05 % | 1945.0879 | 2072.9373 | -6.17 % | 1370.1531 | 1749.9681 | -21.70 % |

Observed convergence orders (same three-level ladder):
| `h/a` | `k0 a` | p (A33, TB) | p (A33, capytaine) | p (B33, TB) | p (B33, capytaine) |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.50 | 0.74 | n/a | 0.61 | n/a |
| 3 | 1.00 | n/a | n/a | n/a | n/a |

## 3b. Extrapolation uncertainty (cells with the confirmation rung)
The extrapolant is recomputed after dropping the coarsest rung; the spread between the two is the measured uncertainty of the extrapolated value.  A cell listed as `n/a` has only three rungs, so its extrapolation uncertainty is unknown (not zero).

| `h/a` | `k0 a` | band A33 | band B33 | band A11 | band B11 | admissible (A / B) |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.50 | n/a | n/a | n/a | n/a | 1.00 % / 1.67 % |
| 3 | 1.00 | n/a | n/a | n/a | n/a | 1.00 % / 1.67 % |

## 4. Negative control (the plan must see depth)
| `h/a` | `k0 a` | `k0 h` | (A33 finite − A33 deep)/A33 deep |
|---:|---:|---:|---:|
| 3 | 0.50 | 1.50 | +17.05 % |
| 3 | 1.00 | 3.00 | n/a |

## 5. Notes, limitations
* Both codes discretise the sphere as an inscribed polyhedron, so the same geometric bias (panel area deficit ~1-2 % at the coarse levels) enters both sides; the extrapolation removes it to first order, and the reported order says how much of the residual is discretisation.
* capytaine's two finite-depth formulations (`newer` vs `legacy`) agree to round-off on this geometry, so they do **not** provide an independent second opinion; the reference uncertainty is the panel ladder, which is why the ladder is reported per cell instead of a single tolerance.
* Submerged closed body only: no waterline, so no irregular-frequency treatment is exercised here; the surface-piercing finite-depth case remains out of scope (`docs/finite_depth_solver3d.md`).
* Damping involves the radiated-wave amplitude and converges more slowly than added mass on both sides; the per-cell orders above make that visible rather than hiding it behind one number.
* The extrapolation band (section 3b) is a *self-consistency* measure, not a truth bound: capytaine and TensorBEM share the inscribed-polyhedron geometry bias, so a common geometric error would sit inside every band and stay invisible.  What the band rules out is a discretisation error of the TensorBEM ladder being mistaken for a solver disagreement.
* Measured order available on 2/8 of the judged code/cell entries; where the reference ladder is not yet asymptotic (`n/a` order) the comparison uses the finest reference level, so that entry's benchmark is the least certain number in this study.
