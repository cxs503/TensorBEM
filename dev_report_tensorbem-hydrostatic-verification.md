# TensorBEM Hydrostatic Verification — Development Report

**Mission**: m2228 (self-directed, `models` seat)
**Date**: 2026-11-03
**Branch**: `agent/m2228`
**Status**: ✅ PASS (with one honest scope limit: Wigley hull not meshed)

## 1. Why this work

`tensorbem.hydrostatic` supplies displaced volume, centre of buoyancy, waterplane
area/second moments and the 6x6 restoring matrix `C` — the only restoring term of
the frequency-domain RAO system `[-w^2 (M + A) + i w B + C] xi = F`. The m273
development report recorded a **2.5 % waterplane-area error and an 8.8 %
displaced-volume error** for a hemisphere and passed them with 5 %/10 % test
tolerances, without any evidence separating *discretisation* from *implementation*
error. That is a verification debt on a quantity that drives every GM and RAO
result downstream.

## 2. What was added

| File | Change |
|---|---|
| `src/tensorbem/geometry3d.py` | `box_hull_mesh` (rectangular barge) and `spheroid_hull_mesh` (half-ellipsoid), plus `_weld_quads_to_wetted_hull` orienting helper |
| `src/tensorbem/benchmarks/hydrostatic_benchmarks.py` | closed-form references, mesh-refinement study with observed-order estimate, reproducible markdown + figure generation (`--report` / `--figure`) |
| `tests/test_hydrostatic_verification.py` | 26 verification tests |
| `docs/hydrostatic_verification.md` | generated verification report (all numbers emitted by the CLI) |
| `docs/figures/hydrostatic_convergence.png` | convergence figure |
| `README.md` | hydrostatic verification paragraph + file index entries |

The barge is the crucial choice: its panels are planar and lie on the analytic
surface, so the discrete hydrostatics must be **exact**, which makes it a
regression test for the integration code rather than for the mesh.

## 3. Results

Closed forms used (`T` draft, `a,b,c` semi-axes):

| quantity | barge `L x B x T` | half-ellipsoid `a,b,c` |
|---|---|---|
| volume | `L B T` | `2 pi a b c / 3` |
| `A_wp` | `L B` | `pi a b` |
| `I_xx = int y^2 dA` | `L B^3/12` | `pi a b^3/4` |
| `I_yy = int x^2 dA` | `B L^3/12` | `pi b a^3/4` |
| `KB` above keel | `T/2` | `5 c/8` |
| `BM_T = I_xx / V` | `B^2/(12 T)` | `3 b^2/(8 c)` |
| `BM_L = I_yy / V` | `L^2/(12 T)` | `3 a^2/(8 c)` |

* **barge** (10 x 4 x 2 m, 4 subdivisions): all 8 quantities exact, worst
  relative error **2.2e-16**.
* **half-ellipsoid** (a=5, b=2, c=2 m): errors 16 -> 64 -> 256 -> 1024 panels fall
  at observed order **1.98-2.00**; at 1024 panels volume 5.87e-3, waterplane area
  1.61e-3, `I_xx` 3.21e-3, `KB` 1.25e-3, `BM` 2.68e-3 — all **< 1 %**.
* **hemisphere (R=1)**: the 64-panel geodesic mesh reproduces the m273 numbers
  exactly (volume 8.86 %, waterplane area 2.55 %). Refining 64 -> 256 -> 1024
  drives the volume error to 8.86 % -> 2.32 % -> 0.587 % (ratios 3.82, 3.95),
  i.e. the 8.8 % is faceting error of a coarse mesh, **not** an implementation
  defect. Classical values `KB = 5R/8`, `BM = 3R/8`, `KM = R` are recovered.

## 4. Verification commands

```bash
PYTHONPATH=src python -m pytest tests/test_hydrostatic_verification.py tests/test_hydrostatic_rao.py tests/test_geometry3d.py -q
# 54 passed
PYTHONPATH=src python -m tensorbem.benchmarks.hydrostatic_benchmarks --report docs/hydrostatic_verification.md --figure docs/figures/hydrostatic_convergence.png
```

A full-suite run was started but abandoned: the host was at load1 ~650 with other
agents, and the suite had reached only 18 of ~310 tests in seven minutes. The
three touched modules were therefore regressed directly (54 passed).

## 5. Pitfalls and limits

* **Wigley parabolic hull cannot be meshed with the quad-sheet approach.** At the
  fore/stern posts `y = 0` for all `z`, so the last quad column contains a
  *corner where keel meets post*; the resulting panel set contains a zero-volume
  sliver lying in the plane `y = 0`, and its mirror image has the opposite
  winding. `WettedHullMesh` correctly rejects that as non-manifold. A Wigley
  generator needs an explicit triangle-fan closure at the posts; the half-ellipsoid
  was used instead because it shares the curved-waterline path without a
  singularity. This is a genuine gap (ship-like fine ends are unverified).
* Both reference bodies are fore-and-aft symmetric, so an asymmetric waterplane
  centroid does not exercise `compute_waterplane_properties`; waterlines needing
  more than one loop (twin hull, appendage piercings) are untested.
* The restoring matrix is linearised at zero heel/trim; large-angle stability is
  out of scope.
* Shell pitfall: `pkill -f "pytest tests/ -q"` matches the invoking shell's own
  command line and kills it; split the literal (`P='pyte''st'`).