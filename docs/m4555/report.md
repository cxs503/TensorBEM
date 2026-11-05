# m4555 — PR #289 (TensorBEM) CI-red repair

## 1. What was red

PR #289 ("`m4313: matured mission deliverables`") adds one file,
`tests/test_floating_sphere_hull_mesh.py` (+197), on top of `efec0d7`. Its CI run
[42071622727](https://github.com/cxs503/TensorBEM/actions/runs/42071622727) ended
`11 failed, 1280 passed, 9 skipped, 2 xfailed, 5 errors`. The 16 red results are
**all pre-existing tree defects**, not defects of the new test: the new test passes
on the merge tree (27/27). They group into four clusters:

| cluster | red results | root cause |
|---|---:|---|
| `evaluate_interior_field` on a multi-component mesh | 7 F + 5 E | `MultiComponentMesh` never grew `boundary_clearance` / `contains_point`, which the later interior-field domain guard (`solver.py:340`) requires |
| artifact skip-gate registry | 1 | two audited gates (`test_lifting_camber.py`, `test_near_surface_three_way.py`) were never registered |
| doc-claim ratchet | 1 | 6 mission reports added without a guard, 3 of them mis-classified as *paper-level* because the historical-name regex only matched `_m####`, not filenames starting `m####_` |
| 3-D finite-depth number pins | 4 | three `free_surface_residual` pins (~1e-15, i.e. machine round-off) were compared with a 0.5 % **relative** tolerance; cross-platform BLAS reduction order moves them 1.26e-15 -> 1.48e-15 = 17 % |

## 2. What was changed

* `src/tensorbem/multicomponent.py` — `MultiComponentMesh.contains_point` (delegates
  to `fluid_contains`: for a mesh with holes "inside the domain" is the outer polygon
  *minus* the holes) and `MultiComponentMesh.boundary_clearance` (same segment
  projection as `PanelMesh.boundary_clearance`, minimised over **all** loops, so a
  target beside a hole is screened exactly like one beside the outer loop).
* `tests/test_multicomponent.py` — two regression tests: clearance equals the minimum
  over the per-loop `PanelMesh` clearances, and `evaluate_interior_field` now returns
  the analytic `log r` field on an annulus (atol 2e-2 at 64/32 panels) while still
  refusing a target in the hole ("outside the domain") and a target within
  `0.5 h_max` of a panel ("too close to the boundary").
* `src/tensorbem/benchmarks/finite_depth3d_pins.py` — `ROUNDOFF_FLOOR = 1e-12`:
  quantities whose declared and recomputed magnitudes are both at machine round-off
  are judged by **absolute** deviation (and `Row.rel_deviation` reports it), because a
  relative tolerance there measures the reduction order, not the kernel.
* `src/tensorbem/benchmarks/doc_claim_audit.py` — historical-name regex extended with
  `^m\d{3,}_`, so dated mission reports named `m####_slug.md` classify as the frozen
  snapshots they are (this is what the module docstring already defines historical to
  mean); `docs/doc_claim_audit/baseline.json` regenerated with the module's own
  `--write-baseline`.
* `tests/test_skip_gates_are_tracked.py` — the two audited gates registered with their
  committed artifacts (`docs/lifting_camber_artifact/benchmark.json`,
  `docs/m3530/near_surface_three_way.json`).

No unrelated file was touched.

## 3. Evidence

Local (loong64, `PYTHONPATH=src`), targeted set through the same tests CI runs:

```
tests/test_multicomponent.py tests/test_interior_field_domain_guard.py
tests/test_multiloop_annulus.py tests/test_skip_gates_are_tracked.py
tests/test_doc_claim_audit.py tests/test_finite_depth3d_pins.py
tests/test_capability_evidence.py tests/test_floating_sphere_hull_mesh.py
=> 115 passed
```

Before the patch the same selection reproduced the CI red (`4 failed ... 5 errors` on
`test_multicomponent.py` + `test_skip_gates_are_tracked.py`; `3 DRIFT` rows and a
0.172 max relative deviation on `test_finite_depth3d_pins.py`).

## 4. Honest limitations

* The full ~1300-test suite was not re-run end to end inside this session (it takes
  ~13 min on CI); the eight affected/adjacent modules were run instead, plus the
  new PR test. The authoritative verdict is the CI run on the pushed head.
* `m3538_propeller_openwater_uncertainty.md` moves from *paper-level unguarded* to
  *historical unguarded* as a side effect of the `^m####_` rule. It stays listed in
  `unguarded` (its numbers are still not parsed by any test); the paper-level contract
  set therefore shrinks by one document and grows by zero.
* The cluster-2/3/4 fixes are gate hygiene; only cluster 1 is a real solver-API bug.
