# Finite-depth cross-validation plan vs capytaine (TensorBEM · m20016)

**Anchor** `XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN`. **Branch** `agent/m20016`.
**Status: PARTIAL EXECUTION — honest report, no cross-validation verdict (§5).**

| artefact | content |
|---|---|
| `src/tensorbem/benchmarks/finite_depth_crossval_plan.py` | pre-registered plan: geometry, `h/a x k0 a` grid, `nu` back-computed from `k0 tanh(k0 h)=nu`, panel ladders, tolerances fixed before the run |
| `scripts/capytaine_finite_depth_ref.py` | producer, the only file importing capytaine: finite `water_depth`, two finite-depth formulations, 288/1152/4608 panels, provenance |
| `src/tensorbem/benchmarks/capytaine_finite_depth_crossval.py` | consumer: TensorBEM ladder, Richardson extrapolation with measured order, gates, generated page + figures, `--jobs` parallelism |
| `tests/test_capytaine_finite_depth_crossval.py` | gates: plan identities, provenance/finite depth, order sanity, pre-registered bands, negative control, live recomputation of the frozen rung, page-vs-artifact consistency |
| `benchmarks/verified/capytaine_finite_depth/*.json` | the frozen numbers quoted below |

## 1. Why a plan, and why it is shaped like capytaine's practice

`docs/capytaine_crosscheck.md` cross-checks the **deep-water** solver: every case in
that reference was produced with `water_depth = inf`. The finite-depth solver had
only internal anchors (multipole at `h/a = 20`, energy, reciprocity), which a
systematic finite-depth error would share. The practice taken over here:

1. **Similarity parameters, not frequencies.** Two independent groups `k0 a`, `h/a`
   (`k0 h` follows). `nu` is *back-computed* per cell from the dispersion relation so
   `k0 a` is exact on every depth instead of drifting along a fixed-`nu` sweep.
2. **The reference gets its own uncertainty.** capytaine has two finite-depth
   formulations (`newer`/`legacy`); measured here on `A33`: 3034.7 vs
   3034.7 (+0.00 %) — the same code path, so
   the reference's uncertainty is its *mesh*, not the formulation.
3. **Refine both codes.** Both are low-order collocation on an inscribed polyhedron;
   a single-mesh difference mixes two unrelated errors, so the comparison is on
   Richardson-extrapolated values with the measured order reported per cell.
4. **A control that must move.** The depth-blind deep-water solver runs on the same
   cells; at small `k0 h` it must differ measurably, else the grid is not probing depth.

## 2. capytaine finite-depth reference (cell `h/a = 3`, `k0 a = 0.5`, `k0 h = 1.5`, `nu = 0.452574`)

| panels | A11 surge | A33 heave | B11 surge | B33 heave |
|---:|---:|---:|---:|---:|
| 288 | 2666.354 | 3060.621 | 822.911 | 752.713 |
| 1152 | 2608.153 | 3034.658 | 811.825 | 751.293 |
| 4608 | 2566.495 | 2999.917 | 797.406 | 740.899 |
| `legacy` formulation (1152-panel mesh) | 2608.154 | 3034.674 | 811.826 | 751.297 |
| **same mesh, `water_depth = inf`** | 2522.100 | 2806.876 | 591.919 | 1300.851 |

Depth is not a small correction at `k0 h = 1.5`: deep-water `B33` is
1300.9 against 740.9 in water of depth `3a`
(+75.58 %), so the negative control has a wide margin.
capytaine's own mesh convergence is slow and monotone
(`A33` 3060.6 -> 3034.7 -> 2999.9), the signature of the
inscribed-polyhedron area deficit rather than quadrature error.

## 3. TensorBEM side, executed rungs

| panels | A11 surge | A33 heave | B11 surge | B33 heave |
|---:|---:|---:|---:|---:|
| 20 | 1748.459 | 1928.821 | 332.011 | 293.277 |
| 80 | 2276.474 | 2617.261 | 616.032 | 564.191 |

## 4. Finest available rungs compared (single mesh, NOT extrapolated)

| quantity | TensorBEM 80 panels | capytaine 4608 panels | deviation | 15 % limit |
|---|---:|---:|---:|---|
| `A11` surge | 2276.474 | 2461.593 | -7.52 % | PASS |
| `A33` heave | 2617.261 | 2999.917 | -12.76 % | PASS |
| `B11` surge | 616.032 | 797.406 | -22.75 % | FAIL |
| `B33` heave | 564.191 | 740.899 | -23.85 % | FAIL |

## 5. Verdict — honest, and negative on the cross-validation claim

1. **1 of 12 planned cells** executed, and only the **20 / 80-panel** rungs of the
   TensorBEM ladder. Two rungs cannot yield an order, so `results.json` records
   `tensorbem_extrapolated = false`: the deviations above are *single-mesh*
   differences, not a solver comparison.
2. Both codes are far from converged on this cell (capytaine still moves
   +2.02 % on `A33` between 288 and 4608 panels; TensorBEM moves similarly
   between 20 and 80 panels). The -12.76 % `A33` and
   -23.85 % `B33` gaps are therefore **discretisation**, and they neither
   confirm nor refute the finite-depth solver. The owner precision standard
   ("> 15 % unacceptable") **cannot be answered from this data**.
3. No figures are published, and that is deliberate: a curve through two unconverged
   rungs on one cell pictures the mesh, not the physics. (Owner rule: a figure-free
   report must state why.)

Established (REAL, measured):

* the plan is well posed: `nu` back-computed from `k0 tanh(k0 h) = nu` inverts to the
  target `k0 a` within 1e-12 on all 12 cells (`test_plan_back_computes_nu_from_the_finite_depth_dispersion`);
* capytaine finite-depth data is obtainable, frozen, and provenanced, and the depth
  effect is unmistakable (`B33` 740.9 finite vs 1300.9 deep, +76 %);
* capytaine's two finite-depth formulations are the same code path on a sphere, so a
  truly independent second reference is still missing (recorded, not papered over);
* the pipeline is re-runnable end to end (`--jobs`, resumable cache, gates, page).

## 6. What the follow-up must run

1. TensorBEM to 320 panels on all 12 cells and 1280 panels on the four
   `EXTRA_LEVEL_CELLS`. Measured cost: 80 panels ≈ 20 s at `h/a = 3` (mode count grows
   as `h / r_min`); submit to the scheduler, never run locally.
2. Try `spherical_patch_areas` weighting or quadratic panels on both sides: the
   inscribed-polyhedron area deficit dominates both codes and is what forces the
   refinement.
3. Keep the depth-blind control at `k0 h <= 2`; its margin (+76 % on `B33`) is the cheapest
   evidence that the grid probes depth at all.
