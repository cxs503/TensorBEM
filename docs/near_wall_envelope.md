# Near-wall accuracy envelope and the fixed-clearance convergence test (m6445)

**Scope**: closes the over-broad half of `docs/near_surface_error_budget.md` §5.
The measured residual there (near-wall velocity not within 3 %, `|dCp|` = 3.34 of
the `Cp` scale at `d = 0.03 h_max`) stands unchanged; what this report replaces is
the *cure statement* — "mesh refinement is **not** the lever" and "the observed
order is −0.03 over 320 → 1280 panels" — which is true only for the experiment
that was run (fixed **relative** clearance), not for the operator.

**Deliverables**

| file | content |
|---|---|
| `src/tensorbem/near_wall_envelope.py` | `NearWallEnvelope` (measured worst-probe lookup `d/h_max → relative error`, conservative step lookups, inverse, `check()`), `build_envelope`, `load_envelope`, `required_clearance` |
| `src/tensorbem/benchmarks/near_wall_envelope_study.py` | driver: fixed-clearance ladder, `d/h_max` sweep, observed orders, envelope, clearance-requirement table, Markdown + 3 figures |
| `tests/test_near_wall_envelope.py` | 12 tests: lookup conservativeness/monotonicity/refusal to extrapolate, measured convergence, measured scale invariance, committed-artefact consistency |
| `docs/near_wall_envelope/study.json` | every number quoted below (program-written) |
| `docs/near_wall_envelope/summary.md` | tables rendered from `study.json` (program-written) |
| `docs/near_wall_envelope/fig1..3*.png` | figures (program-written) |

Reproduce:

```
PYTHONPATH=src python3 -m tensorbem.benchmarks.near_wall_envelope_study
PYTHONPATH=src python3 -m pytest tests/test_near_wall_envelope.py -q
```

## 1. Case

Unit sphere (radius `a = 1 m`) in an unbounded fluid. The analytic field is the
disturbance potential of a sphere in a uniform stream along `+z`,

    u = z / (2 r^3),   grad u = e_z/(2 r^3) - 3 z x / (2 r^5),

so the surface traces used as panel densities are **exact**:
`u = cos(theta)/2`, `du/dn_fluid = cos(theta)` at the panel centroids, and
`Cp = 1 - (9/4) sin^2(theta)` with `Cp` scale 2.25. No solve is involved, which
isolates the *representation* error of the constant-panel double layer from any
solve error — the same design as the budget report.

| parameter | value |
|---|---|
| meshes | icosphere subdivisions 2 / 3 / 4 / 5 → 320 / 1280 / 5120 / 20480 panels |
| `h_max` | 0.32492 / 0.16465 / 0.08260 / 0.04134 m |
| fixed physical clearances | `d` = 0.2, 0.1, 0.05, 0.025, 0.0125 m |
| relative clearances | `d/h_max` = 3.0, 2.0, 1.2, 0.8, 0.6, 0.45, 0.3, 0.2, 0.12, 0.08, 0.05 |
| probes | equator `+x` (worst), pole `+z`, oblique `(1,0,1)/√2` |
| reported quantity | relative error of the near-wall velocity against the closed form |

## 2. Method

1. **The missing experiment.** The budget report sweeps `d/h_max ∈ {1, 0.5, 0.1, 0.03}`
   over two meshes. Because `d = (d/h_max) h_max`, refining the mesh there moves
   the probe *with* the mesh: the amplification `~1/d` doubles while the trace
   error `~h` halves, and the product is invariant. That is a real and correctly
   measured property of the representation, but it cannot say whether the
   operator converges. Here a **fixed-`d` ladder** is added: hold `d` in metres,
   refine `h_max` by 2× over four meshes, and fit the observed order of the
   velocity error.
2. **Both experiments reported.** The `d/h_max` sweep is kept and re-measured on
   three meshes, so the fixed-ratio collapse is visible next to the fixed-`d`
   convergence. Neither statement is dropped to make the other look better.
3. **Order definition.** `order_two_finest` is the least-squares slope of
   `log(error)` against `log(h_max)` over the two finest rungs (the asymptotic
   order); `order_full_ladder` uses all four rungs. Both are published separately
   because a non-asymptotic ladder reports a misleading slope.
4. **Envelope.** Per `d/h_max`, the worst error over meshes and directions is
   recorded, then closed upwards to be non-increasing
   (`err(r_i) = max_{j ≥ i} err(r_j)`), so measurement noise can only ever make
   the envelope *more* conservative. The lookup answers with the nearest measured
   ratio that is no larger than the query, and refuses (returns `inf`) below the
   measured range instead of extrapolating towards the wall.
5. **Acceptance gates** (evaluated by the driver, printed in `verdict`):
   asymptotic order at `d = 0.05 m` (equator) `≥ 1.0`; spread across meshes at
   `d/h_max = 0.3` `≤ 3×`; envelope certifies ≤ 3 % at some measured ratio.
6. **Program-written numbers.** `study.json` and `summary.md` are written by the
   driver and cross-checked by tests: `test_published_orders_match_the_published_rows`
   recomputes every published order from the published error ladder, and
   `test_published_envelope_is_consistent_with_the_sweep` rebuilds the envelope
   from the sweep and checks that no sweep sample exceeds it. No number in
   `summary.md` is typed by hand.

## 3. Results

All numbers below are read from `docs/near_wall_envelope/study.json`
(`fixed_clearance.orders[*]`, `ratio_sweep.rows[*]`, `envelope`,
`clearance_requirements`, `verdict`).

### 3.1 Fixed physical clearance: the operator converges

Equator probe (the worst direction), error vs `h_max` at fixed `d`:

| `d` [m] | coarsest (320 panels) | finest (20480 panels) | improvement | order (4 rungs) | order (2 finest) |
|---:|---:|---:|---:|---:|---:|
| 0.2000 | 1.229e-02 | 2.667e-05 | ×460.8 | 2.83 | 2.05 |
| 0.1000 | 1.072e-01 | 1.750e-04 | ×612.4 | 3.16 | 1.85 |
| 0.0500 | 4.160e-01 | 6.587e-04 | ×631.6 | 3.22 | 3.19 |
| 0.0250 | 1.116e+00 | 5.449e-03 | ×204.8 | 2.54 | 4.06 |
| 0.0125 | 2.543e+00 | 8.876e-02 | ×28.7 | 1.61 | 2.18 |

Over the whole ladder (15 direction/clearance pairs) the asymptotic order lies in
**1.85 … 4.06** (`verdict.order_min_over_ladder`,
`verdict.order_max_over_ladder`) — never at the −0.03 of the fixed-ratio
experiment. The orders above 2 are the expected super-convergence of a
*centroid* rule once `d/h_max` grows large enough that the near-singular
amplification is weak: the trace error is then `O(h^2)` and the amplification is
bounded.

### 3.2 Fixed relative clearance: the same errors collapse

Equator error at `d/h_max = 0.3` for three meshes (`verdict.scale_invariance`):

| panels | error at `d/h_max = 0.3` |
|---:|---:|
| 320 | 1.139e-01 |
| 1280 | 9.851e-02 |
| 5120 | 9.295e-02 |

spread ×1.23 over a 16× panel-count change. Refining *at fixed relative
clearance* therefore buys nothing, exactly as the budget report says. The two
tables are consistent: refining at fixed `d` **is** refining at increasing
`d/h_max`, i.e. moving up §3.2's curve, which is why §3.1 converges.

### 3.3 The operational envelope

Worst-probe envelope and the clearance it requires (`clearance_requirements`):

| target error | required `d/h_max` | measured worst error at that ratio | certified |
|---:|---:|---:|:--:|
| 10 % | 0.45 | 3.761e-02 | yes |
| 5 % | 0.45 | 3.761e-02 | yes |
| 3 % | 0.60 | 1.360e-02 | yes |
| 1 % | 0.80 | 4.011e-03 | yes |

So the practical rule for the constant-panel near-wall field is **`d ≥ 0.6 h_max`
for 3 %** and **`d ≥ 0.8 h_max` for 1 %**, on the worst probe of the sphere. The
budget report's "`d ≥ 0.5 h_max` is the clearance that actually satisfies 3 %" is
not contradicted (0.45 still measures 3.76 %, 0.5 was not sampled), but it is
tightened: the conservative certified ratio is 0.6.

`tensorbem.near_wall_envelope.NearWallEnvelope.check(d, h_max, 0.03)` answers
this question for a concrete mesh and probe; the tests pin that it never passes
a probe that the measurements do not certify.

## 4. Figures

| figure | content |
|---|---|
| `docs/near_wall_envelope/fig1_fixed_clearance_convergence.png` | error vs `h_max` at each fixed `d` (equator), log–log, with the asymptotic order in the legend and the 3 % bar |
| `docs/near_wall_envelope/fig2_ratio_collapse.png` | error vs `d/h_max` for 320/1280/5120 panels × three directions: the curves collapse, which is the scale invariance |
| `docs/near_wall_envelope/fig3_clearance_requirement.png` | worst-probe envelope with the certified clearance for 10/5/3/1 % marked |

## 5. Notes, limits

* **Both statements survive.** Fixed-ratio scale invariance is *not* refuted;
  the refuted claim is the inference from it that refinement cannot help. The
  correct statement — now in code, in the tests and here — is that refinement
  helps when the mesh is chosen so that `d ≥ ratio(h_max)`.
* **Worst-probe, not pointwise.** The envelope is a maximum over three probe
  directions at the equator/pole/oblique points of a unit sphere with exact
  traces. It is not a per-point error bound and it is not a statement about a
  ship hull: the panel-to-probe geometry (and hence the amplification `1/d`)
  differs. The pole is 10–100× better than the equator at the same `d/h_max`
  (`ratio_sweep`), so a pointwise bound would be far more expensive.
* **Centroid traces inside the faceted sphere.** Centroids of an inscribed mesh
  sit at `r < 1`, so the exact traces there are not the smooth-sphere traces
  (≈2.5 % in `u`). That is the right density for a constant-panel collocation
  solve, but these numbers are not a statement about geometric faceting error.
* **Above `d/h_max ≈ 1.2` the error floors** at 1.4e-03 (worst probe): that
  residual is the far-field discretisation, not the near-wall representation,
  and it does not vanish under this probe design. Do not read the envelope as
  "3 % is achievable everywhere"; 0.14 % is the floor of this measure.
* **Cost.** The full study is ~19 s single-core and needs no dense solve; the
  test slice is ~1 s. This is a documentation/verification workload.

## 6. Error judgement against the 3 % bar

| quantity | reference | measured | < 3 %? |
|---|---|---|---|
| near-wall velocity, equator, `d = 0.05 m`, 320 panels | closed form | 4.160e-01 | **no** (reproduces the merged honest FAIL) |
| same probe, `d = 0.05 m`, 20480 panels | closed form | 6.587e-04 | **yes** (by 45×) |
| same probe, `d/h_max = 0.6`, 320 / 1280 / 5120 panels | closed form | ≤ 1.360e-02 worst over directions | **yes** |
| fixed-ratio invariance spread, `d/h_max = 0.3`, 320→5120 panels | measurement | ×1.23 | consistent with the scale-invariance claim |
| asymptotic order, fixed `d = 0.05 m` | ≥ 1.0 (gate) | 3.19 | **yes** |

The near-wall *velocity at very small clearance* remains a failure at fixed
panel size — that honest FAIL is untouched. What changes is that it is a
*resolution* requirement (`d ≥ 0.6 h_max`), not an impossibility: at
`d = 0.05 m` the error falls from 42 % to 0.066 % as the mesh is refined, four
orders of magnitude, and the measured order never falls below 1.85 across the
ladder.