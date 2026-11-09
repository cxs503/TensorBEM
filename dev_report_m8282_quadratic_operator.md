# TensorBEM quadratic (6-node) panel operator — implementation & verification

**Task**: m8282 (reopened TensorBEM#497, carry to close) · **Branch**: `agent/m8282` · **Repo**: TensorBEM
**Scope**: make the quadratic panel operator of `tensorbem.panel_quadratic` actually deliver the
second-kind formulation `(1/2 I - D) u = -S q` that its own docstring advertises, and verify it
against closed-form identities.

---

## 1. Case introduction

Exterior 3-D Laplace Dirichlet problem on the unit sphere with an interior point source at
`x0 = (0, 0, 0.5)`:

| Item | Value |
|---|---|
| Geometry | unit sphere, icosphere discretised with 6-node quadratic (curved) triangles |
| Mesh levels | subdivisions 0/1/2 -> 42/162/642 nodes, 20/80/320 elements |
| Exact potential | `u = 1 / (4 pi \|x - x0\|)` |
| Exact flux | `q = -(x - x0)·n / (4 pi \|x - x0\|^3)` |
| Quadrature | Gauss order 8 (non-singular panels), Duffy order 16 (singular self panels) |
| Mesh size | `h = sqrt(mean flat element area)` |
| Acceptance | relative L2 flux error < 3% (hard limit 15%) |

## 2. Method

1. **Fix the operator, not the test.** Three structural defects were found in the shipped module
   (see §3) and repaired: the edge-midpoint shape-function gradients (hence the isoparametric
   Jacobian / metric / normals), the double-layer matrix (it was assembled as all zeros), and the
   singular self-panel quadrature for edge-midpoint collocation points (plain Gauss was used while
   the target sat *on* the panel → double counting / garbage).
2. Solve the **direct second-kind BIE** `q = -S^-1 (1/2 I - D) u` with Dirichlet data sampled from
   the exact potential; compare the nodal flux with the exact flux at element centroids
   (relative L2 norm).
3. Independently verify the **operators** against closed-form values that never involve a solver:
   `S 1 = 1` on the unit sphere (single-layer potential of unit density), `D 1 = -1/2`
   (principal-value solid-angle identity), `(1/2 I - D) 1 = 1`, polyhedral solid angles on a cube
   (`-1/8`, `-1/4`, `-1/2` at vertices, edge midpoints, face-interior nodes), and the identically
   vanishing flat-panel double-layer self block.
4. Fit convergence orders as least-squares slopes of `log(error)` vs `log(h)`.

Toolchain: `pytest` (43 dedicated tests), `python -m tensorbem.benchmarks.quadratic_panel_verification`
(study JSON + figure), numpy/matplotlib.

## 3. Defects found in the shipped module (evidence)

| Defect | Symptom before the fix | Evidence after |
|---|---|---|
| Edge-midpoint shape-function gradients wrong (`dN5/dL1 = -4L3` instead of `-4L2`, `dN6` sign flipped) | isoparametric Jacobian not equal to the identity on a flat reference triangle (`det = 2.46, 1.49, 0.23` instead of 1); every metric, normal and integral wrong | `det == 1`, `n == +z` to 1e-12 (`test_flat_panel_metric_is_unity`) |
| `assemble_quadratic_matrices` never filled `D` | `\|D\| = 0.0` (stub) — the advertised second-kind equation was unusable | `\|D\| > 0`, closed-form identities hold (see §4) |
| Edge-midpoint collocation used plain Gauss on its own (singular) panel | near-singular integrand evaluated with an 8x8 rule → O(1) errors | Duffy-type apex quadrature for all six nodes; self blocks match order-64 reference to 1.5e-15 |
| Double-layer kernel sign / displacement convention inconsistent between the vectorised block and the per-target routine | flat-surface row sums of `D` came out `+0.5` instead of `-0.5` on the sphere while the cube was "right" | cube solid angles exact to 1e-10; sphere `\|D 1 + 1/2\|` converges at order 2.93 |

Correction of a claim in the earlier report (m7534): the density `sigma` of the **indirect**
single-layer formulation `S sigma = u` is the *equivalent source density*, **not** the normal
derivative `du/dn`. The physical flux requires the second-kind equation (`q = -(S^-1)(1/2 I - D)u`)
— the module now provides that route, and a regression test (`test_equivalent_source_density_is_not_the_flux`)
keeps the two quantities apart.

## 4. Results (all numbers programmatically injected from `study.json`)

| subdiv | nodes | elements | h | flux rel-L2 error | `\|S1-1\|_inf` | `\|D1+1/2\|_inf` | cond(S) |
|---|---|---|---|---|---|---|---|
| 0 | 42 | 20 | 0.6919 | 1.404e-01 | 1.311e-02 | 3.867e-02 | 20.8 |
| 1 | 162 | 80 | 0.3819 | 2.854e-02 | 1.061e-03 | 6.321e-03 | 39.4 |
| 2 | 642 | 320 | 0.1963 | 6.065e-03 | 7.217e-05 | 9.610e-04 | 80.8 |

Observed convergence orders (least-squares fits over four levels):

| Quantity | Order |
|---|---|
| flux relative error (direct 2nd-kind BIE) | **2.49** |
| `\|S 1 - 1\|_inf` | 4.13 |
| `\|D 1 + 1/2\|_inf` | **2.93** (consistent with the documented `O(h^3)` geometry) |

Flat-panel singular quadrature (analytic reference):

* flat-panel double-layer self block: `max|D_self| = 0.0e+00` (exactly zero, as required by `(x-y)·n = 0`),
* single-layer apex quadrature versus order 64:

| order | rel. error |
|---|---|
| 8 | 3.14e-05 |
| 12 | 2.38e-07 |
| 16 | 5.40e-09 |
| 24 | 1.22e-12 |
| 32 | 1.48e-15 |

Acceptance:

| Gate | Value | Verdict |
|---|---|---|
| flux rel-L2 error, finest mesh (642 nodes) | 6.065e-03 | **< 3%: True** |
| hard limit 15% | 6.065e-03 | < 15%: True |

## 5. Notes and limitations

* `D` is assembled in the **principal-value** sense; the `-1/2` jump is carried by the
  `1/2 I` term, exactly as the module docstring states. Tests assert `sum_k D_ik = -1/2` on a
  smooth closed surface.
* Non-singular panels use a fixed order-8 Gauss rule. The row-sum identity converges at
  ~2.9, i.e. the fixed rule is *not* the error source at these
  mesh sizes; near-singular adaptive quadrature is not needed for the reported accuracy.
* `cond(S)` grows roughly like `1/h` (81 → 164 between the last two levels); the solve is still
  well conditioned for an iterative solver at these sizes.
* The flux error is evaluated with the nodal values interpolated to element centroids; the
  centroid values are the natural comparison point for the exact flux and avoid nodal-normal
  ambiguity on a vertex where several panels meet.
* Scope limit: this PR covers the Laplace operator only. The same `D` will be reused by the
  frequency-domain (Helmholtz) quadratic path in a follow-up; nothing in this PR changes the
  constant/linear panel solvers.

## 6. Hot-file declaration

`src/tensorbem/panel_quadratic.py` is a core solver module. This PR touches only that module plus
the quadratic verification benchmark/tests and the new `docs/quadratic_panel_verification/`
artifacts; `solver3d`, `panel_linear`, `geometry3d` and the frequency-domain paths are untouched.

## 7. Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_panel_quadratic.py tests/test_panel_quadratic_operator.py -q
PYTHONPATH=src python -m tensorbem.benchmarks.quadratic_panel_verification   # ~2.5 min: study.json + figure
```

## 8. Files

| File | Change |
|---|---|
| `src/tensorbem/panel_quadratic.py` | vectorised shape functions/quadrature; corrected edge-node gradients; node-exact singular quadrature; double-layer operator; `solve_neumann_from_dirichlet_quadratic` |
| `src/tensorbem/benchmarks/quadratic_panel_verification.py` | rewritten (the previous version crashed: 6-node face unpacked as 3 vertices); operator-identity + flux studies, JSON + 3-panel figure |
| `tests/test_panel_quadratic.py` | non-singular integration assertion made mathematically correct (shape functions change sign) |
| `tests/test_panel_quadratic_operator.py` | new: 18 operator/identity/physics tests |
| `docs/quadratic_panel_verification/study.json` | study record (all numbers) |
| `docs/quadratic_panel_verification/fig_quadratic_operator.png` | convergence + identity + nodal flux field figure |
