# 3-D assembly cost: reuse, node caching, and the heavy tier

**Date**: 2026-11-03 · **Branch**: agent/m2450 · **Solver**: TensorBEM

Closes the open item `expand_audit_tensorbem.md` §4.2.2 ("椭球基准 1280 面元未直接复现")
and §5 recommendation 2. The 1280-panel ellipsoid verification was parked in the
`heavy` cost tier of `tensorbem.benchmarks.declared_numbers` because it "costs tens of
minutes" (`declared_numbers.TIERS` docstring). It now costs **35 s on one core**.

## 1. What was actually slow

Profiled at 320 panels (`subdivisions=2`), single-threaded BLAS:

| component | time |
|---|---|
| `assemble_surface_matrices` | 0.710 s |
| `np.linalg.cond(S)` | 0.038 s |
| `np.linalg.solve(S, q)` | 0.005 s |
| whole `ellipsoid_added_mass.run(2)` | **10.64 s** |

The gap between one assembly (0.71 s) and the whole run (10.64 s) was pure redundancy,
not arithmetic:

1. `run()` assembled the dense matrices **six times** — once per translation axis inside
   `solve_laplace_surface`, and again per axis inside
   `estimate_operator_conditions`, although the matrices depend on the mesh and the
   quadrature order only, never on the boundary data.
2. `_polar_sectors` called `numpy.polynomial.legendre.leggauss(order)` once per panel to
   evaluate the self-panel integral. `leggauss` builds a Jacobi matrix and calls
   `eigvalsh` every time: 1920 rebuilds per run, 5.0 s of cProfile's 12.8 s.
3. `diagnose_near_singular_quadrature(mesh)` was recomputed per axis although it is a
   mesh property.

## 2. What changed

| file | change |
|---|---|
| `kernels3d.py` | `gauss_legendre_nodes(order)`, an `lru_cache` table returned read-only; used by `duffy_quadrature`, `duffy_quadrature_batch`, `_polar_sectors` |
| `solver3d.py` | frozen `AssembledSurface` + `assembled_surface(mesh, order)`; `solve_laplace_surface(..., assembled=None)` reuses matrices after checking the panel count, the shapes and the quadrature order |
| `conditioning.py` | `estimate_operator_conditions(..., assembled=None)` — the diagnostic needs `S` and `H`, so it no longer assembles a second copy |
| `benchmarks/ellipsoid_added_mass.py` | one assembly and one mesh diagnostic per run; `reuse_assembly` flag (`--no-reuse-assembly`) keeps the old path for A/B checks |

A mismatched assembly raises instead of being used silently
(`tests/test_assembly_reuse.py`).

## 3. Results

| case | before | after | speed-up |
|---|---|---|---|
| ellipsoid, 320 panels (`subdivisions=2`) | 10.64 s | 2.21 s | **4.82×** |
| ellipsoid, 1280 panels (`subdivisions=3`) | 16 CPU·min, unfinished (2026-11-03 report) | **35.3 s** | ≥ 27× |

Numbers are **bitwise** unchanged: added mass, relative error, normal-derivative error,
single-layer condition number and relative residual agree to 0.000e+00 relative
deviation between the reuse path and the fresh-assembly path at 320 panels
(`docs/assembly_cost/reuse_sub2.json` vs `baseline_sub2.json`).

### 3.1 The 1280-panel declared numbers, reproduced for the first time

PASS threshold in the registry is 1.5 % relative deviation.

| axis | measured error | declared | deviation | < 3 %? |
|---|---|---|---|---|
| long (x) | 4.167936e-3 | 4.1679e-3 | 8.74e-6 | **yes** (0.42 %) |
| intermediate (y) | 4.462121e-3 | 4.4621e-3 | 4.82e-6 | **yes** (0.45 %) |
| short (z) | 4.935932e-3 | 4.9359e-3 | 6.54e-6 | **yes** (0.49 %) |

Artifact: `docs/assembly_cost/ellipsoid_1280.json` (written by the benchmark, not
transcribed). This also resolves the arithmetic dispute of `dev_report_tensorbem_declared_numbers.md`
§5.2 in favour of the documentation: the 1280-panel values are reproducible, the earlier
"not reproducible" verdict came from a wrong refinement factor (320→1280 is 4×, not 16×).

## 4. Cost model and the tier classification

Assembly is O(N²), the dense condition estimate O(N³); measured constants at 320 panels
(0.71 s and 0.038 s) predict 11.4 s and 2.4 s at 1280 panels, i.e. ≈ 35 s for a full run
with reuse — the measured 35.3 s confirms the model. Consequences:

* the three `ellipsoid_added_mass_1280_*` entries can stay in `heavy` (a 35 s gate is
  still not a unit test) but are now gated by
  `tests/test_assembly_reuse.py::test_ellipsoid_heavy_tier_reproduces_within_tolerance`
  behind `TENSORBEM_HEAVY_TESTS=1`, so the tier is reachable instead of unreachable;
* thread contention matters here: the same 320-panel run took 62 s wall / 819 CPU·s with
  the default threaded BLAS and 10.6 s with `OMP_NUM_THREADS=1`. Benchmarks on a loaded
  machine should pin BLAS threads to one; otherwise the timing says nothing about the
  algorithm.

## 5. Pitfalls and scope

* **Redundant work hides as "expensive method".** The 1280-panel case looked
  intrinsically intractable; it was six assemblies where one sufficed.
* **Reuse is only safe if the inputs are checked.** `AssembledSurface` carries the panel
  count and the quadrature order precisely so that a stale matrix cannot be fed to a
  different mesh; both mismatch paths are tested.
* **Do not cache arrays a caller may mutate.** The memoised nodes are set read-only;
  a caller that needs to modify them must copy first.
* **Scope**: only the 3-D Laplace assembly path (`solver3d`, `kernels3d`,
  `conditioning`, ellipsoid benchmark) was changed and measured. The free-surface 3-D
  solver (`free_surface_solver3d`) already cached its node tables at module level and was
  not touched. No physics, quadrature order or discretisation changed.

## 6. Pre-existing breakage found while running the suite (not fixed here)

Two failures are unrelated to this change and reproduce on the unmodified tree
(`git stash` check):

1. `tests/test_hydrostatic_verification.py` fails to import:
   `ImportError: cannot import name 'box_hull_mesh' from 'tensorbem.geometry3d'`.
   Commit `caa44f0` (m2228) says "5 个文件被门禁滤除" — the hull-mesh helpers
   (`box_hull_mesh`, `spheroid_hull_mesh`, `hemisphere_hull_mesh`) were dropped while
   `benchmarks/hydrostatic_benchmarks.py`, `benchmarks/hydrostatic_verification.py` and
   the test module that import them were committed. The hydrostatic verification and RAO
   benchmarks are therefore dead code on `main`. The helpers exist in history
   (`git log -S box_hull_mesh -- src/tensorbem/geometry3d.py`, e.g. `61a31a9`), so this
   is a restoration job, not a rewrite.
2. `tests/test_doc_claims.py` fails 3 of 7 tests: the artifact it audits is absent.

Both are recorded as findings; fixing them was out of scope for this mission.
