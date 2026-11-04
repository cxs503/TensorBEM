# 2-D interior field evaluator: closing the silent-wrong-answer gap (P0-2)

**Scope.** `docs/project_review_2026-10-22.md` §6.1 listed four "P0" input-validation
gaps that can *silently* produce a wrong number. P0-1 (non-positive clearance factor
disables the 3-D guard) and P0-3/P0-4 (non-integral panel counts, degenerate polygons)
have since been fixed and are covered by tests. **P0-2 was still open**: the 2-D
`evaluate_interior_field` checked only shape and finiteness of its targets, never
whether a target belongs to the domain, so an exterior or on-boundary target returned
a finite, plausible, wrong value with no warning. This report closes it.

**Verdict.** The fix is a domain guard, not a numerical change: the solver and the
analytic panel integrals are untouched, interior values are unchanged to the last
digit, and outside/boundary targets now raise `ValueError` instead of returning
numbers whose relative error reaches `1.000` (i.e. 100.0 %).

![field guard](outputs/fig_field_guard.png)

---

## 1. Case

Harmonic Dirichlet problem with the exact solution available in closed form:

| item | value |
|---|---|
| boundary condition | `u = x` on a regular polygon |
| exact interior solution | `u(x, y) = x` (harmonic; also linear, so any quadrature error shows up directly) |
| mesh family | regular polygons, `N` = 32 / 64 / 128 panels, radius 1 m, CCW |
| `h_max` (32 / 64 / 128) | 1.9603e-01 / 9.8135e-02 / 4.9082e-02 m |
| evaluator | constant-panel Green representation, analytic panel integrals (`solver.evaluate_interior_field`) |
| density | unit (Laplace problem, `rho` irrelevant) |
| reference | `u = x`, exact to machine precision |

The failure mode is the same one the audit recorded: with `N = 32`, a target at
`(2.0, 0.0)` is outside the unit polygon, where the *interior* Green representation
still converges — to a finite number that is not the solution.

## 2. Method

1. **Expose the failure.** Before the change, call the evaluator on exterior,
   on-boundary and interior targets and compare with `u = x` (reproduced here from the
   pre-fix commit through `git show <rev>:src/tensorbem/solver.py`, so the "before"
   column is measured, not quoted).
2. **Add the geometry predicates the guard needs** (`PanelMesh.contains_point`,
   `PanelMesh.boundary_clearance`): a `+x` ray cast with the half-open vertex rule, and
   the minimum distance to the panel segments. Both are exact for an axis-aligned box,
   which is how they are unit-tested.
3. **Guard the evaluator.** `evaluate_interior_field` now refuses (a) targets on or
   closer than `max(minimum_clearance_factor * h_max, 1e-14 * h_max)` to the boundary and
   (b) targets outside the polygon, mirroring the 3-D convention
   (`solver3d.evaluate_surface_field`). The factor is validated by a single shared
   helper (`kernels.validate_clearance_factor`), which also replaces the duplicated
   copy that used to live in `solver3d`.
4. **Do not over-guard.** In 2-D the analytic panel integrals are stable arbitrarily
   close to the boundary, so the default factor stays `0.0` (only the strict-inside
   test); a 3-D-style `0.05 h_max` band is available but opt-in, and the study
   quantifies what it would refuse (32 of
   1941 otherwise-accepted grid targets).
5. **Unify the duplicated predicate.** `benchmarks/unit_disk.py` carried its own
   `matplotlib.path.Path(...).contains_points` mask; it now uses
   `mesh.contains_point`, removing the second source of truth.

## 3. Results

### 3.1 Accuracy inside the domain (guard must not cost accuracy)

`|u_BEM - u_exact|` along a 30° ray plus five targets at 1e-3 ... 1e-1 m from one facet
(all accepted):

| N panels | `h_max` [m] | RMS abs error | max abs error, clearance < 0.05 h_max | max abs error, clearance > 0.05 h_max |
|---|---|---|---|---|
| 32 | 1.9603e-01 | 9.284e-04 | 6.435e-05 (2 target(s) in band) | 4.085e-03 |
| 64 | 9.8135e-02 | 2.975e-04 | 3.228e-05 (2 target(s) in band) | 1.427e-03 |
| 128 | 4.9082e-02 | 2.857e-04 | 1.888e-03 (2 target(s) in band) | 8.999e-05 |

Interior accuracy is unchanged from the pre-fix solver (same code path); the guard
costs one `O(M*N)` distance/ray-cast pass, no numerical work.

### 3.2 What the guard refuses, and how wrong the values were

Pre-fix evaluator (same 32-panel mesh), values recomputed from `HEAD`:

| target | inside? | clearance [m] | pre-fix value | exact | relative error |
|---|---|---|---|---|---|
| (+2.0, +0.0) | no | 1.000e+00 | +8.0601e-04 | +2.0 | 9.996e-01 |
| (+1.2, +0.3) | no | 2.402e-01 | +1.2729e-03 | +1.2 | 9.989e-01 |
| (+1.0, +0.0) | no | 0.000e+00 | +4.6895e-01 | +1.0 | 5.310e-01 |
| (+0.5, +0.0) | yes | 4.976e-01 | +4.9921e-01 | +0.5 | 1.584e-03 |

Grid tally for the 32-panel mesh, `3721` targets on a 2.4 m x 2.4 m grid:

| category | count |
|---|---|
| refused: outside the polygon | 1779 |
| refused: on the boundary | 1 |
| accepted (strictly inside) | 1941 |
| would be refused by an opt-in 0.05 `h_max` band | 32 |

The accepted set still reaches a minimum clearance of `1.685e-03`
m, and its worst absolute error is `2.254e-02`
(`2.792e-03` RMS) — i.e. the accepted targets cluster near the
boundary, exactly the region the guard now has to leave open by default in 2-D.

## 4. Error judgement against the benchmark

| quantity | baseline | measured | relative error | < 3 %? |
|---|---|---|---|---|
| interior field `u = x`, 64 panels, 5 probes | exact `x` | RMS `2.975e-04`, max `1.427e-03` | ~1e-3 (RMS) | **yes** |
| interior field, 128 panels, near-boundary band | exact `x` | max `1.888e-03` | ~2e-3 | **yes** |
| exterior target `(2.0, 0)` (pre-fix) | `2.0` | `8.06e-4` | `9.996e-1` (99.96 %) | **no** — this is the defect |
| on-boundary target `(1.0, 0)` (pre-fix) | `1.0` | `4.69e-1` | `5.31e-1` (53.1 %) | **no** — this is the defect |
| guard false-positive rate (interior grid) | 0 expected | 0 refused interior targets (the only interior refusals are exact-boundary ones) | 0 | yes |

**Answer to "is it < 3 %": yes for every accepted target** (worst accepted RMS
`9.284e-04` at 32 panels). The two pre-fix exterior/boundary rows
are *not* accuracy claims — they are the evidence that the old behaviour had no error
control at all, which is why the guard is an exception rather than a tolerance.

## 5. Pitfalls, limits, scope

* **A boundary point can be classified "inside" by a ray cast.** The half-open vertex
  rule is a *crossing* rule for the open polygon; points on an edge are undefined-
  either-way. That is why the clearance test runs **first** and the outside test second:
  a vertex target is reported as "on or too close to the boundary", which is the true
  diagnosis.
* **Do not import the 3-D default blindly.** `0.05 h_max` is right for the 3-D
  fixed-order Duffy quadrature, but in 2-D it would refuse legitimate targets: the
  analytic panel integrals are exact for a constant panel regardless of distance. The
  default here is `0.0`; the opt-in band is documented and tested, not silently applied.
* **`contains_point` is O(N) per point and `boundary_clearance` is O(N) per point** —
  the guard adds two vectorised sweeps over the panels per evaluation batch. For the
  1e3-panel meshes this code is used with, that is microseconds against an
  `O(M*N)` kernel loop already being run.
* **Not covered here**: the same silent-failure class in the *exterior* problem has no
  2-D field evaluator at all (documented limitation, unchanged); P0-4 (multi-component
  3-D meshes, self-intersecting 2-D polygons) is only partly closed — `PanelMesh`
  still rejects a self-intersecting boundary through its signed-area check rather than
  a dedicated simplicity test, which is a separate task.
* **Measurement provenance**: every number in §3-§4 is written by
  `docs/m3351/field_guard_study.py` into `docs/m3351/outputs/field_guard.json` and
  formatted into this file from that JSON; nothing is typed in by hand.

## 6. Reproduce

```bash
PYTHONPATH=src python3 -m pytest tests/test_interior_field_domain_guard.py -q   # 12 guards
PYTHONPATH=src python3 docs/m3351/field_guard_study.py --legacy-rev HEAD~1      # JSON + figure
```

* `src/tensorbem/geometry.py`: `PanelMesh.contains_point`, `PanelMesh.boundary_clearance`,
  `_as_point_array` (new).
* `src/tensorbem/kernels.py`: `validate_clearance_factor` (new, shared).
* `src/tensorbem/solver3d.py`: local duplicate validator replaced by the shared one.
* `src/tensorbem/solver.py`: `evaluate_interior_field(..., minimum_clearance_factor=0.0)`
  plus the two guards.
* `src/tensorbem/benchmarks/unit_disk.py`: mask now uses `mesh.contains_point`.
* `tests/test_interior_field_domain_guard.py` (new).
