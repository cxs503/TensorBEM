# Immersed-solid (Brinkman) projection: masked Poisson vs. legacy `kappa` relaxation

Node **P218-N05**.  Domain: resistance / incompressible solver infrastructure.
Module: `src/tensorfvm/immersed/` (new).  Tests:
`tests/test_immersed_masked_projection.py`.

## 1. Problem statement

A Brinkman-penalised immersed-solid solver keeps the body at rest with a penalty
sink `kappa * H(x) * u` (`H` = solid indicator).  In the fractional-step
implementation used here historically ("legacy"), the penalty is applied *twice*:
once implicitly inside the momentum predictor and once as a
**post-projection relaxation**

    u <- u / (1 + kappa dt H(x))

The projection step has, at that moment, already made `div u = 0` in the
discrete sense.  The relaxation multiplies the faces that touch the body, so the
continuity constraint is destroyed exactly in the first layer of cells at the
body surface: the body absorbs (or emits) volume every step.  The defect is
*systematic*, so it accumulates linearly in time, and it is invisible in a run
that only checks "is the body roughly at rest".

The fix in this node is structural: make rigidity a **constraint of the
projection** instead of a penalty.  Solid cells leave the pressure system, every
face of a solid cell has a hard zero normal velocity (homogeneous Neumann for
the correction), and the outlet plane is closed with `phi = 0` outside so the
masked operator is SPD and `Q_out = Q_in` holds by construction.

## 2. What is in the module

| file | contents |
|---|---|
| `geometry.py` | mask-only geometries: backward-facing step (ER exactly 2), symmetric sudden expansion (area ratio 2), Poiseuille inlet profile, realised-expansion-ratio measurement from the mask |
| `projection.py` | masked 5-point operator (`-h^2 laplacian` on fluid cells), masked Jacobi / geometric-multigrid preconditioned CG, and a prefactorised sparse direct path (SuperLU + MMD ordering) reused every step |
| `solver.py` | staggered-MAC fractional-step solver with `projection="masked"` and `projection="legacy"`, the discrete mass ledger, wall-shear probes |
| `study.py` | run driver, wall-shear zero-crossing (reattachment), Richardson/GCI, reference values with provenance |

## 3. Mass ledger (the observability the node asked for)

Every step records

* `div_max_fluid` = `max_cells |int_cell div u| / (U H)` (per-step continuity defect),
* `div_max_solid` = the same over solid cells,
* `solid_flux` = `int_dS_solid u . n dS` (net spurious volume flux into the body),
  accumulated in time as `drift` in units of `U H^2`,
* `mass_imbalance` = `(Q_out - Q_in)/Q_in`,
* `solid_speed_max` = `max|u|` inside the body, normalised by `U`.

Because interior faces telescope, `int_dS u.n dS = sum_solid div u dV`, so the
same defect is visible either as a surface flux or as a divergence inside the
body; both are recorded.

## 4. Criteria

See `artifacts/summary.json` → `criteria` (machine readable) and `REPORT.md`
(rendered from it).  The verdicts are recomputed by
`scripts/immersed_masked_report.py`, never typed.

## 5. Reproducing

```bash
# one run (~1-100 min depending on mesh)
PYTHONPATH=src python -m tensorfvm.immersed.study \
    --case bfs --cells-per-h 40 --arm masked --t-end 40 \
    --out-dir runs/immersed
# the whole study (one process per case/mesh/arm)
scripts/immersed_masked_study.sh
# artefacts + figures + report
python scripts/immersed_masked_report.py
python scripts/immersed_write_report.py
```

## 6. Limits

* Two-dimensional, laminar, uniform Cartesian grid; the immersed surfaces are
  stair-stepped (which for these two geometries *is* body-fitted, since all walls
  lie on grid lines — that is why the mask error cannot hide the projection
  error).
* The masked arm's rigidity is exact, so `kappa` becomes a no-op; the penalty
  parameter is *not* part of the fixed scheme's input space any more.  The
  `kappa -> 0 / -> inf` negative controls exist to prove that the legacy answer
  (not the masked one) was the one contaminated by the penalty.
* Time integration is first-order (explicit projection), convection first-order
  upwind: the observed order of the *reattachment length* is therefore expected
  to be ~1, not 2.  The reported `p_obs` measures what the scheme actually
  delivers; a first-order scheme cannot fake second order.