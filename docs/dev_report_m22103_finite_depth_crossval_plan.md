# Dev report m22103 — Finite-depth cross-validation plan (XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN)

**Anchor**: `XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN` — "capytaine 的 finite_depth_crossval_plan 做法".
**Domain**: TensorBEM finite-depth BEM solver (`tensorbem.finite_depth_solver3d`) vs capytaine.
**Status**: plan defined + capytaine reference complete (12/12 cells); TensorBEM ladder **in flight** on cluster (cj3911).

## 1. What the plan is

`tensorbem.benchmarks.finite_depth_crossval_plan` is the code-to-code cross-validation
of the 3-D finite-depth radiation solver against capytaine's finite-depth Green
function.  It borrows capytaine's own approach:

1. **Similarity-parameter grid** — `h/a x k0*a` (12 cells: `h/a in {3,5,10,20}` x
   `k0*a in {0.5,1.0,2.0}`), with `nu = omega^2/g` *back-computed* per cell from
   `k0 tanh(k0 h) = nu` so `k0*a` is exact at every depth (a fixed-`nu` grid would
   let `k0*a` drift along the depth ladder, mixing the two groups).
2. **Reference's own uncertainty** — capytaine's two finite-depth formulations
   (`newer`/`legacy`); their spread is a free code-internal band.
3. **Refine both, do not trust one** — a panel ladder on each side, compared by
   *Richardson-extrapolated* values (observed order + extrapolation residual), so a
   disagreement is attributed to solver vs insufficient refinement, not reported
   as "solver disagreement" when it is mesh noise.
4. **A case that must move** — the depth-blind deep-water solver as a negative
   control; at small `k0*h` the plan must separate the two.

## 2. Current state (measured, frozen artifacts)

| artifact | state |
|---|---|
| `benchmarks/verified/capytaine_finite_depth/capytaine_ref.json` | **complete** — 12/12 cells, 3-level capytaine ladder `(12,24)/(24,48)/(48,96)` + `legacy` |
| `benchmarks/verified/capytaine_finite_depth/tensor_ladder.json` | **partial** — 1/12 cells (`h/a=3, k0*a=0.5`), 2 rungs (20/80 panels) |
| `benchmarks/verified/capytaine_finite_depth/results.json` | **partial** — `execution_status: "PARTIAL: 1 of 12 planned cells"` |

The single completed cell (`h/a=3, k0*a=0.5`, `k0*h=1.5`, shallow water) at the
80-panel rung shows a **genuine, reproducible** code-to-code deviation:

| quantity | TensorBEM (80 p) | capytaine | rel. |
|---|---:|---:|---:|
| heave A33 | 2617.32 | 2999.91 | **−12.75 %** |
| heave B33 | 564.21 | 740.90 | **−23.85 %** |
| surge A11 | 2276.48 | 2461.59 | −7.52 % |
| surge B11 | 616.04 | 797.41 | −22.74 % |

**This is NOT yet a verdict.**  Both codes are low-order constant-strength panel
methods; at 80 panels the difference is dominated by two *independent*
discretisation errors.  The plan's Richardson extrapolation (needs the 320- and
1280-panel rungs) is the tool that decides whether the −12.75 %/−23.85 % is a
genuine finite-depth kernel defect or mesh noise.  The `_partial_run` guard in
`tests/test_capytaine_finite_depth_crossval.py` correctly xfails the verdict
gates until the ladder is extrapolated.

## 3. What is in flight

The full 12-cell TensorBEM ladder (3 rungs + the 4 `EXTRA_LEVEL_CELLS` at 1280
panels) was **submitted to the cluster** as job **cj3911** (4 shards, `--ladder-only
--shard {i} --shards 4`).  The level-2 (320-panel) rung is the bottleneck
(~2–3 min/cell with the eigenseries kernel), so the full grid is a cluster-scale
job, not a local one.  On completion the shards are merged
(`--merge-caches`) and the study driver re-runs to produce the extrapolated
comparison, gates, and the published page.

## 4. Next steps (for the follow-on self-start)

1. **Fetch cj3911** → merge the 4 shard caches → re-run
   `capytaine_finite_depth_crossval --write-doc` → the `_partial_run` xfails
   become real verdicts.
2. **Decide the defect**: if the extrapolated A33/B33 stay >15 % at shallow
   `k0*h`, the finite-depth kernel has a genuine defect (the plan's whole point);
   if they collapse to <3 %, the single-cell −12.75 % was discretisation.
3. **Publish** the completed page + report with the full 12-cell table and the
   convergence-order map.

## 5. Honest limitations

- The TensorBEM ladder is **1/12 cells** at the time of writing; the full-grid
  verdict is not yet available (cluster job in flight).
- The single-cell numbers are **2-rung** (not extrapolated); they are evidence of
  a *discrepancy to investigate*, not a *defect verdict*.
- capytaine is an optional dependency (built from source for loongarch64); its
  numbers are frozen into the reference, so the test suite does not need it.
