# Finite-depth cross-validation against capytaine
**Scope**: code-to-code verification of `tensorbem.finite_depth_solver3d` in water of *finite* depth against capytaine's finite-depth Green function (the existing `docs/capytaine_crosscheck.md` is infinite depth only).
**Plan**: `tensorbem.benchmarks.finite_depth_crossval_plan` — submerged sphere `a = 1.0 m`, centre `z = -1.5 m`, grid `h/a in [3.0, 5.0, 10.0, 20.0]` x `k0 a in [0.5, 1.0, 2.0]`, `nu` back-computed per cell from `k0 tanh(k0 h) = nu` so `k0 a` is exact at every depth.
**Reference**: `scripts/capytaine_finite_depth_ref.py` (capytaine 3.1.dev0, `finite_depth_method=newer`, ladder [[12, 24], [24, 48], [48, 96]] panels, generated 2026-11-15T12:55:54+00:00).
**Execution status**: `COMPLETE`; **confirmation rung**: `PARTIAL: 4 confirmation cells missing the 3-subdivision rung`; every number below is written by the study driver from the frozen artifacts, none is transcribed.
## 1. Headline
| quantity | worst deviation on the pre-registered basis | pre-registered limit | owner limit |
|---|---:|---:|---:|
| heave added mass | +1.40 % | 3.00 % | 15 % |
| surge added mass | +1.66 % | 3.00 % | 15 % |
| heave damping | +3.38 % | 5.00 % | 15 % |
| surge damping | +2.15 % | 5.00 % | 15 % |

## 2. Gates
| gate | measured | limit | verdict |
|---|---:|---:|---|
| `added_mass_agreement` | 0.0166 | 0.03 | PASS |
| `damping_agreement` | 0.0338 | 0.05 | PASS |
| `negative_control_separates_depth` | 0.1705 | 0.02 | PASS |
| `reference_ladder_in_asymptotic_range` | 0.1514 | 0.3 | FAIL |
| `reference_truncation_within_share` | 0.0181 | 0.016666666666666666 | FAIL |
| `tensorbem_ladder_in_asymptotic_range` | 0.6141 | 0.3 | PASS |
| `extrapolation_self_consistent` | no | 1.0 | FAIL |
| `linear_solve_residuals` | 0.0000 | 1e-08 | PASS |
| `reference_is_finite_depth` | yes | True | PASS |

## 3. Per-cell comparison (Richardson-extrapolated, gated by the pre-registered order floor)
| `h/a` | `k0 a` | `k0 h` | A33 TB | A33 capytaine | delta | B33 TB | B33 capytaine | delta | A11 TB | A11 capytaine | delta | B11 TB | B11 capytaine | delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.50 | 5.00 | 2733.5883 | 2741.4605 | -0.29 % | 1623.5940 | 1574.0209 | +3.15 % | 2445.0619 | 2467.3597 | -0.90 % | 730.0098 | 714.6230 | +2.15 % |
| 10 | 1.00 | 10.00 | 1858.5975 | 1880.0892 | -1.14 % | 3245.3228 | 3171.4868 | +2.33 % | 2056.9732 | 2026.0488 | +1.53 % | 1667.0465 | 1634.4818 | +1.99 % |
| 10 | 2.00 | 20.00 | 1473.5782 | 1494.4701 | -1.40 % | 1175.2121 | 1169.5830 | +0.48 % | 1797.5163 | 1780.0647 | +0.98 % | 707.1435 | 702.8489 | +0.61 % |
| 20 | 0.50 | 10.00 | 2726.2311 | 2734.2938 | -0.29 % | 1622.2785 | 1572.9159 | +3.14 % | 2444.9920 | 2467.2795 | -0.90 % | 730.3967 | 715.0001 | +2.15 % |
| 20 | 1.00 | 20.00 | 1855.3249 | 1876.8638 | -1.15 % | 3241.7111 | 3168.0778 | +2.32 % | 2056.9594 | 2026.0308 | +1.53 % | 1667.0435 | 1634.4797 | +1.99 % |
| 20 | 2.00 | 40.00 | 1471.0620 | 1491.9869 | -1.40 % | 1174.1911 | 1168.5741 | +0.48 % | 1797.4616 | 1780.0000 | +0.98 % | 707.1389 | 702.8457 | +0.61 % |
| 3 | 0.50 | 1.50 | 2998.9655 | 2999.9145 | -0.03 % | 765.9752 | 740.8984 | +3.38 % | 2545.3639 | 2566.4945 | -0.82 % | 814.3840 | 797.4057 | +2.13 % |
| 3 | 1.00 | 3.00 | 2321.8795 | 2340.1447 | -0.78 % | 3222.8631 | 3139.2583 | +2.66 % | 2107.3107 | 2072.9373 | +1.66 % | 1784.1545 | 1749.9681 | +1.95 % |
| 3 | 2.00 | 6.00 | 1781.5513 | 1803.1635 | -1.20 % | 1253.0462 | 1246.6395 | +0.51 % | 1871.1397 | 1849.1385 | +1.19 % | 708.9705 | 705.1239 | +0.55 % |
| 5 | 0.50 | 2.50 | 2808.2377 | 2814.3930 | -0.22 % | 1452.7411 | 1407.1126 | +3.24 % | 2441.8558 | 2464.2097 | -0.91 % | 719.2826 | 704.2333 | +2.14 % |
| 5 | 1.00 | 5.00 | 1902.7369 | 1924.2958 | -1.12 % | 3284.0978 | 3208.2848 | +2.36 % | 2057.5103 | 2026.5767 | +1.53 % | 1668.2282 | 1635.6660 | +1.99 % |
| 5 | 2.00 | 10.00 | 1501.6624 | 1522.8851 | -1.39 % | 1185.6893 | 1179.8967 | +0.49 % | 1800.2323 | 1782.6462 | +0.99 % | 707.1329 | 702.8565 | +0.61 % |

Observed convergence orders (same three-level ladder):
| `h/a` | `k0 a` | p (A33, TB) | p (A33, capytaine) | p (B33, TB) | p (B33, capytaine) |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.50 | 0.77 | n/a | 0.62 | n/a |
| 10 | 1.00 | 1.00 | 0.17 | 0.66 | n/a |
| 10 | 2.00 | 1.18 | 0.28 | 0.79 | n/a |
| 20 | 0.50 | 0.77 | n/a | 0.62 | n/a |
| 20 | 1.00 | 1.01 | 0.17 | 0.66 | n/a |
| 20 | 2.00 | 1.18 | 0.28 | 0.79 | n/a |
| 3 | 0.50 | 0.74 | n/a | 0.61 | n/a |
| 3 | 1.00 | 0.86 | n/a | 0.64 | n/a |
| 3 | 2.00 | 0.93 | 0.15 | 0.78 | n/a |
| 5 | 0.50 | 0.76 | n/a | 0.62 | n/a |
| 5 | 1.00 | 0.98 | 0.15 | 0.66 | n/a |
| 5 | 2.00 | 1.14 | 0.27 | 0.79 | n/a |

## 3b. Extrapolation uncertainty (cells with the confirmation rung)
The extrapolant is recomputed after dropping the coarsest rung; the spread between the two is the measured uncertainty of the extrapolated value.  A cell listed as `n/a` has only three rungs, so its extrapolation uncertainty is unknown (not zero).

| `h/a` | `k0 a` | band A33 | band B33 | band A11 | band B11 | admissible (A / B) |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.50 | n/a | n/a | n/a | n/a | 1.00 % / 1.67 % |
| 10 | 1.00 | n/a | n/a | n/a | n/a | 1.00 % / 1.67 % |
| 10 | 2.00 | n/a | n/a | n/a | n/a | 1.00 % / 1.67 % |
| 20 | 0.50 | n/a | n/a | n/a | n/a | 1.00 % / 1.67 % |
| 20 | 1.00 | n/a | n/a | n/a | n/a | 1.00 % / 1.67 % |
| 20 | 2.00 | n/a | n/a | n/a | n/a | 1.00 % / 1.67 % |
| 3 | 0.50 | n/a | n/a | n/a | n/a | 1.00 % / 1.67 % |
| 3 | 1.00 | n/a | n/a | n/a | n/a | 1.00 % / 1.67 % |
| 3 | 2.00 | n/a | n/a | n/a | n/a | 1.00 % / 1.67 % |
| 5 | 0.50 | n/a | n/a | n/a | n/a | 1.00 % / 1.67 % |
| 5 | 1.00 | n/a | n/a | n/a | n/a | 1.00 % / 1.67 % |
| 5 | 2.00 | n/a | n/a | n/a | n/a | 1.00 % / 1.67 % |

## 4. Negative control (the plan must see depth)
| `h/a` | `k0 a` | `k0 h` | (A33 finite − A33 deep)/A33 deep |
|---:|---:|---:|---:|
| 10 | 0.50 | 5.00 | +8.72 % |
| 10 | 1.00 | 10.00 | +4.07 % |
| 10 | 2.00 | 20.00 | +2.32 % |
| 20 | 0.50 | 10.00 | +8.44 % |
| 20 | 1.00 | 20.00 | +3.90 % |
| 20 | 2.00 | 40.00 | +2.15 % |
| 3 | 0.50 | 1.50 | +17.05 % |
| 3 | 1.00 | 3.00 | +27.88 % |
| 3 | 2.00 | 6.00 | +22.39 % |
| 5 | 0.50 | 2.50 | +11.24 % |
| 5 | 1.00 | 5.00 | +6.41 % |
| 5 | 2.00 | 10.00 | +4.17 % |

## 5. Notes, limitations
* Both codes discretise the sphere as an inscribed polyhedron, so the same geometric bias (panel area deficit ~1-2 % at the coarse levels) enters both sides; the extrapolation removes it to first order, and the reported order says how much of the residual is discretisation.
* capytaine's two finite-depth formulations (`newer` vs `legacy`) agree to round-off on this geometry, so they do **not** provide an independent second opinion; the reference uncertainty is the panel ladder, which is why the ladder is reported per cell instead of a single tolerance.
* Submerged closed body only: no waterline, so no irregular-frequency treatment is exercised here; the surface-piercing finite-depth case remains out of scope (`docs/finite_depth_solver3d.md`).
* Damping involves the radiated-wave amplitude and converges more slowly than added mass on both sides; the per-cell orders above make that visible rather than hiding it behind one number.
* The extrapolation band (section 3b) is a *self-consistency* measure, not a truth bound: capytaine and TensorBEM share the inscribed-polyhedron geometry bias, so a common geometric error would sit inside every band and stay invisible.  What the band rules out is a discretisation error of the TensorBEM ladder being mistaken for a solver disagreement.
* Measured order available on 19/48 of the judged code/cell entries; where the reference ladder is not yet asymptotic (`n/a` order) the comparison uses the finest reference level, so that entry's benchmark is the least certain number in this study.
