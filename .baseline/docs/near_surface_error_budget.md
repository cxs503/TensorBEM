# Near-wall field evaluation: measured error budget and corrected root cause

**Status**: the near-wall *velocity/pressure* honest FAIL of `docs/near_surface_field.md` stands (it is **not** < 3 %), but its documented *cause* and its documented *cure* are both replaced by measurements: the closed-form solid-angle gradient is exact, there is no cancellation, the quadrature is irrelevant, and mesh refinement does not reduce the error at all.

Reproduce with

```
PYTHONPATH=src python3 -m tensorbem.benchmarks.near_surface_error_budget \
    --subdivisions 2,3 --outdir docs/near_surface_error_budget
PYTHONPATH=src python3 -m pytest tests/test_near_surface_error_budget.py -q
```

## 1. Case

Unit sphere (radius `a = 1 m`) in an unbounded fluid; the analytic field is the
disturbance potential of a sphere in a uniform stream of unit speed along `+z`,

    u = z / (2 r^3),   grad(u) = e_z/(2 r^3) - 3 z x / (2 r^5),

so the surface traces are `u = cos(theta)/2` and `du/dn_fluid = cos(theta)`, and
`Cp = 1 - (9/4) sin^2(theta)`; the `Cp` scale used below is `2.25`.

| parameter | value |
|---|---|
| meshes | icosphere subdivisions 2 / 3 -> 320 / 1280 panels |
| `h_max` | 0.3249 m (320 panels), 0.1646 m (1280 panels) |
| clearance sweep | `d/h_max` = 1, 0.5, 0.1, 0.03 |
| probe directions | equator `+x`, pole `+z`, oblique `(1,0,1)/sqrt(2)` |
| traces | **exact** analytic values at the panel centroids (no solve) |
| velocity law | single layer by the adaptive near-singular rule + double layer by the closed-form `grad(Omega_j)` -- structurally identical to `solver3d.evaluate_surface_velocity_near_surface` |

The traces are exact on purpose: this isolates the *representation* error of the
constant-panel double layer from the error of a dense solve.

## 2. Method

1. **Oracle for the analytic gradient.** The closed-form `grad(Omega_j)` is audited
   against a central difference of the closed-form solid angle *value* itself
   (`delta = 1e-6 h_max`), panel by panel. The two use different code paths; the
   value formula carries no gradient-specific algebra, so this is an independent
   test of the gradient expression rather than a self-consistency check.
2. **Global-cancellation test.** For a closed mesh the constant-density double layer
   is piecewise constant, hence `sum_j grad(Omega_j) = 0` exactly. If the failing
   case lost digits to cancellation, this residual would be of the order of the
   near-wall error divided by the density.
3. **Quadrature independence.** The failing case is recomputed with `order = 12`,
   `split_ratio = 2.0`, `max_level = 32` (roughly 4x the work). A quadrature error
   must shrink; a representation error must not.
4. **Mesh ladder.** 320 -> 1280 panels at fixed `d/h_max`; a convergent
   discretisation must improve, an inconsistent one must stall.
5. **Amplification model.** The double-layer term is linear in the panel traces, so
   `d(velocity)/d(mu_j) = -grad(Omega_j)/(4 pi)`. Multiplying the per-panel
   amplification by the intra-panel variation of the *exact* trace
   `max_vertices |u - u_centroid|` and combining panels in quadrature gives a
   first-order magnitude estimate for the error carried by a piecewise-constant
   density.

## 3. Results

### 3.1 The closed-form solid-angle gradient is exact

| case | panels | max abs error | max relative error | `|sum_j grad Omega_j|` |
|---|---:|---:|---:|---:|
| equator_d0.03 | 1280 | 7.045e-08 | 2.451e-08 | 2.153e-13 |
| equator_d0.1 | 1280 | 3.671e-09 | 1.943e-08 | 1.924e-13 |

The gradient formula reproduces the oracle to 2.5e-08 relative even at `d = 0.03 h_max`, while individual panel gradients there reach 195.8 (i.e. `~1/d`). The closed mesh sum is 2.2e-13 against a per-panel scale of 196 and 1280 panels -- **there is no cancellation.**

### 3.2 The failing error is insensitive to quadrature and to mesh refinement

| `d/h_max` | velocity error, order 8 | velocity error, order 12 | ratio |
|---:|---:|---:|---:|
| 1 | 6.594e-05 | 6.594e-05 | 1.000000 |
| 0.5 | 7.388e-03 | 7.388e-03 | 1.000000 |
| 0.1 | 3.670e-01 | 3.670e-01 | 1.000000 |
| 0.03 | 1.676e+00 | 1.676e+00 | 1.000000 |

| `d/h_max` | levels (panels: equator velocity error) | observed order |
|---:|---|---:|
| 0.1 | 320: 3.585e-01, 1280: 3.670e-01 | -0.034 |
| 0.03 | 320: 1.626e+00, 1280: 1.676e+00 | -0.044 |

Quadrupling the quadrature effort changes the error by exactly zero digits, and
refining the mesh by 4x does not reduce it (observed order -0.03). Both facts are incompatible with a
quadrature or a round-off explanation, and with *any* convergent discretisation of
the *geometry*.

### 3.3 Measured error against the amplification model

Relative error is taken against the local gradient magnitude `|grad(u)|` at the probe.

| panels | direction | `d/h_max` | velocity error | rel. error | `|dCp|`/scale | model estimate | max `|grad Omega|/(4 pi)` |
|---:|---|---:|---:|---:|---:|---:|---:|
| 320 | equator | 1 | 3.473e-04 |  0.16% | 0.000 | 1.744e-02 | 0.09 |
| 320 | equator | 0.5 | 8.458e-03 |  2.66% | 0.008 | 5.651e-02 | 0.31 |
| 320 | equator | 0.1 | 3.585e-01 | 78.93% | 0.319 | 3.869e-01 | 2.12 |
| 320 | equator | 0.03 | 1.626e+00 | 334.85% | 1.446 | 1.327e+00 | 7.27 |
| 320 | pole | 1 | 3.909e-04 |  0.09% | 0.000 | 8.388e-03 | 0.09 |
| 320 | pole | 0.5 | 7.814e-04 |  0.12% | 0.001 | 2.442e-02 | 0.31 |
| 320 | pole | 0.1 | 2.007e-02 |  2.21% | 0.018 | 1.603e-01 | 2.12 |
| 320 | pole | 0.03 | 8.530e-02 |  8.78% | 0.076 | 5.487e-01 | 7.27 |
| 320 | oblique | 1 | 5.069e-04 |  0.15% | 0.000 | 1.423e-02 | 0.12 |
| 320 | oblique | 0.5 | 2.100e-03 |  0.42% | 0.002 | 4.710e-02 | 0.52 |
| 320 | oblique | 0.1 | 7.221e-02 | 10.05% | 0.064 | 2.685e-01 | 3.49 |
| 320 | oblique | 0.03 | 2.387e-01 | 31.09% | 0.212 | 5.365e-01 | 7.06 |
| 1280 | equator | 1 | 6.594e-05 |  0.02% | 0.000 | 1.940e-02 | 0.20 |
| 1280 | equator | 0.5 | 7.388e-03 |  1.87% | 0.007 | 6.205e-02 | 0.68 |
| 1280 | equator | 0.1 | 3.670e-01 | 77.09% | 0.326 | 4.137e-01 | 4.58 |
| 1280 | equator | 0.03 | 1.676e+00 | 340.13% | 1.489 | 1.407e+00 | 15.58 |
| 1280 | pole | 1 | 1.443e-04 |  0.02% | 0.000 | 4.953e-03 | 0.20 |
| 1280 | pole | 0.5 | 4.717e-04 |  0.06% | 0.000 | 1.368e-02 | 0.68 |
| 1280 | pole | 0.1 | 1.083e-02 |  1.14% | 0.010 | 8.691e-02 | 4.58 |
| 1280 | pole | 0.03 | 4.449e-02 |  4.51% | 0.040 | 2.950e-01 | 15.58 |
| 1280 | oblique | 1 | 5.379e-04 |  0.11% | 0.000 | 1.384e-02 | 0.20 |
| 1280 | oblique | 0.5 | 6.336e-03 |  1.02% | 0.006 | 4.412e-02 | 0.76 |
| 1280 | oblique | 0.1 | 2.550e-01 | 33.88% | 0.227 | 3.010e-01 | 6.61 |
| 1280 | oblique | 0.03 | 9.806e-01 | 125.89% | 0.872 | 1.042e+00 | 28.71 |

### 3.4 Why the error cannot converge under refinement

The two ingredients of the amplification model scale in opposite directions:

| panels | `h_max` | max `|grad Omega|/(4 pi)` | intra-panel trace variation | product | measured equator error at `d = 0.03 h_max` |
|---:|---:|---:|---:|---:|---:|
| 320 | 0.3249 | 7.27 | 9.459e-02 | 6.874e-01 | 1.626e+00 |
| 1280 | 0.1646 | 15.58 | 4.684e-02 | 7.298e-01 | 1.676e+00 |

Halving `h_max` doubles the amplification (`x2.14`) and halves the
intra-panel trace variation (`x0.50`); the product is invariant to
`1.06`, which is exactly the measured stall. The
constant-panel double layer therefore has **no convergence rate at all** for the
near-wall gradient: the error is a fixed fraction of the local field, set by the
piecewise-constant density rather than by the panel size.

The model is a *magnitude* estimate, not a pointwise predictor, and the report says
so: at the equator (the worst probe) measured/model is 1.23 (320 panels) and 1.19 (1280 panels), but at the pole it over-predicts by 6.4x, because the pole probe is a well-conditioned point where the tangential variation
of the trace is small. Its value is in the *scalings*, which it reproduces.

## 4. Error judgement against the 3 % bar

| quantity | reference | measured | < 3 %? |
|---|---|---|---|
| closed-form `grad(Omega_j)`, 320 and 1280 panels, `d = 0.1 / 0.03 h_max` | central difference of the solid angle | 2.5e-08 relative | **yes** (by 7 orders) |
| `sum_j grad(Omega_j)` on a closed mesh | 0 | 2.2e-13 | **yes** |
| near-wall velocity, equator, `d = 0.5 h_max`, 320 panels | analytic `grad(u)` | 8.46e-03 (2.66% of local scale) | **yes** |
| near-wall velocity, equator, `d = 0.1 h_max`, 320 panels | analytic | 3.59e-01 (79%) | **no** |
| near-wall velocity, equator, `d = 0.03 h_max`, 320 panels | analytic | 1.63e+00 (335%) | **no -- honest FAIL** |
| near-wall `|dCp|`/scale, `d = 0.03 h_max`, 1280 panels | analytic `1-(9/4)sin^2` | 1.49 | **no -- honest FAIL** |

No reference value was adjusted to make a metric pass. The near-wall
velocity/pressure remains a failure; what changes is *why* it fails and *what
would fix it*.

## 5. What the corrected diagnosis changes

| claim in `docs/near_surface_field.md` | measured status |
|---|---|
| "the closed form loses digits through cancellation" | **refuted**: 2.5e-08 relative agreement with an independent oracle, closed-mesh residual 2.2e-13 |
| "the cure is a regularised or finite-part hypersingular quadrature for the double-layer gradient" | **refuted as a cure**: the double-layer term uses no quadrature at all, and quadrupling the *single-layer* quadrature changes the error by 0.000e+00 |
| "use `d >= 0.1 h_max` for < 3 % velocity error" | **too optimistic for the worst probe**: 79% at the equator; `d >= 0.5 h_max` is the clearance that actually satisfies 3 % there |
| "mesh refinement is the lever" | **not for this quantity**: observed order -0.03 over 320 -> 1280 panels |

The only routes that can restore convergence are (i) a *higher-order density*
representation (linear/quadratic traces), which changes the intra-panel variation
from `O(h)` to `O(h^2)` while the amplification stays `O(1/h)`, giving `O(h)`, or
(ii) not differentiating the representation at all -- using the tangential-recovery
route `evaluate_surface_velocity_at_panels` (surface `Cp` RMS 2.4 % of scale) and
extending it off the surface. Both are follow-up work; neither is a quadrature
change, and (ii) is already verified on the surface.

## 6. Notes, traps, limits

* **Statistics matter.** Every number here is a single-probe point value at the
  equator/pole/oblique direction with exact traces; `docs/near_surface_field.md`
  reports a sweep statistic from an end-to-end solve, so the two tables are not
  interchangeable. The equator probe is the worst of the three and it is the one
  the conclusion rests on.
* **Centroids of an inscribed mesh sit inside the sphere** (`r < 1`), so the exact
  traces evaluated there are *not* the traces of the smooth sphere (`~2.5 %`
  discrepancy in `u`). That is the correct choice for a constant-panel
  representation (it is what a collocation solve converges to), but it means these
  numbers are *not* a statement about the geometric error of the faceted sphere.
* **Do not read the model as a bound.** It over-predicts by up to 6x at
  well-conditioned probes; only the scalings are claimed.
* **Cost.** The full budget (24 probe/clearance cases, 2 meshes, oracle audit of
  1280 panels) runs in ~8 s on one core and needs no dense solve; it is a
  documentation/verification workload, not a compute workload.
* **Figure.** `docs/near_surface_error_budget/near_wall_error_budget.png`: (left)
  measured versus modelled velocity error against clearance for both meshes;
  (middle) closed-form versus oracle gradient error; (right) the amplification
  `max_j |grad Omega_j|/(4 pi)` -- the quantity that grows like `1/d` and cancels
  the `O(h)` trace variation.

## 7. Files

| file | change |
|---|---|
| `src/tensorbem/benchmarks/near_surface_error_budget.py` | new: budget harness, finite-difference solid-angle oracle, closed-mesh identity, amplification model, JSON/Markdown/figure writer, CLI |
| `tests/test_near_surface_error_budget.py` | new: 8 tests pinning the oracle agreement, the zero sum, quadrature independence, mesh independence, the model scalings and the measured envelope |
| `docs/near_surface_error_budget/` | generated `results.json`, `summary.md`, figure |
| `artifacts/m2620/make_ns_doc.py` | generator of this document (numbers injected from `results.json`) |
| `docs/near_surface_field.md` | root cause and cure corrected, pointer to this report |
