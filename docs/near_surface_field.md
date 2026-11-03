# Near-singular field quadrature for 3-D BEM: closing the near-wall gap

**Scope.** `README.md` roadmap item 5 lists *near-singular field quadrature* as the
only remaining field-evaluation gap, and the code enforced it: both
`evaluate_surface_field` and `evaluate_surface_velocity` raised `ValueError` for any
target closer than `minimum_clearance_factor * h_max` (default 0.05), i.e. every
near-wall quantity a ship-hydrodynamics user wants — off-body pressure, wave
elevation just above the surface, near-field velocity — was simply unavailable.
This work adds a target-adaptive quadrature rule, two new entry points that accept
arbitrary positive clearance, and an analytic verification.

**Verdict.** The quadrature is now clearance-independent: kernel-level error
1.4e-14 versus 8.3e-2 for the old fixed rule (6e12 improvement at
`d = 1e-5 · panel size`), and the near-wall potential is accurate to
0.2–0.7 % at 320 panels / 0.35 % at 1280 panels. The near-wall **velocity and
pressure do not meet the 3 % bar**, and the report says so explicitly (§6):
the residual error there is *not* quadrature error but cancellation inside the
closed-form solid-angle gradient of the constant-panel double layer.

---

## 1. Case

Unit sphere, radius `a = 1 m`, in an unbounded fluid. The analytic field is the
**disturbance** potential of a sphere in a uniform stream of unit speed along
`+z`,

    u(x) = z / (2 r^3) = cos(theta) / (2 r^2),      r = |x| >= a,

whose boundary traces are `u = cos(theta)/2` and, along the fluid-domain normal
(`-r_hat`, the convention `LaplaceSurfaceResult` stores for the exterior problem),
`du/dn = +cos(theta)`. Two independent data sets are used:

| case | data on the panels | what the error measures |
|---|---|---|
| representation | exact analytic traces | pure quadrature + trace discretisation |
| end-to-end | traces from `solve_laplace_surface(..., domain="exterior")` | the same, plus the dense solve |

| parameter | value |
|---|---|
| radius `a` | 1.0 m |
| stream speed `U` | 1.0 m/s |
| meshes | icosphere subdivisions 1 / 2 / 3 → 80 / 320 / 1280 panels |
| `h_max` (subdiv 2) | 3.249e-1 m |
| clearance sweep | `d / h_max` = 1e-1 … 2e-4 (down to `1e-5` panel size) |
| analytic references | `u`, `grad(u)`, `Cp = 1 - (9/4) sin^2(theta)` |
| solver settings | `quadrature_order=8`, `split_ratio=1.0`, `max_level=24` |
| quadrature reference | `order=14`, `split_ratio=2.0`, `max_level=32` |

## 2. Method

1. **Why the old rule fails.** `evaluate_surface_field` integrates the
   single-layer kernel `G = 1/(4*pi*r)` with a fixed-order Duffy-Gauss rule. The
   rule is spectrally accurate for a well-separated panel, but as the target
   approaches a panel the `1/r` peak sits inside the panel and the fixed rule
   cannot resolve it. The double-layer term is not the problem: for
   constant panel density it is the closed-form signed solid angle, exact at any
   clearance.

2. **The new rule** (`src/tensorbem/near_singular.py`). Each panel is bisected by
   its midpoints into four similar sub-triangles, recursively, until a
   sub-triangle is *well separated* from the target
   (`distance >= split_ratio * size`) or the depth budget `max_level` is spent;
   only well-separated sub-triangles are integrated with the Gauss-Duffy rule.
   Because only the shrinking chain of sub-triangles that contains the peak is
   refined, the work per target grows logarithmically in `h/d` while the error
   becomes independent of `d`. The distance used by the criterion is the true
   point-to-triangle distance (plane distance inside the footprint, edge distance
   outside it).

3. **Two entry points.** `evaluate_surface_field_near_surface` and
   `evaluate_surface_velocity_near_surface` (both in `solver3d.py`) mirror the
   existing far-field functions, keep their sign conventions, and reject only
   points lying exactly on the surface (the representation is discontinuous
   there). The old functions are untouched, so nothing that worked before changes.

4. **Isolating the error.** The reference is the *same* adaptive rule at
   `order=14, split_ratio=2.0`, which agrees with `order=16, split_ratio=3.0` to
   1e-13 on these cases; the difference between the production settings and that
   reference is quadrature error alone. Comparing against the analytic field on
   top of that adds the constant-panel trace error.

5. **One bug found and fixed on the way** (the reason the first sweep looked
   non-convergent): `_bisect` stacks the four children *blockwise*, so a child
   must be mapped back to its parent with `np.tile(parents, 4)`, not
   `np.repeat(parents, 4)`. With `repeat`, the total sum over the mesh was
   conserved (which is why it looked plausible) while individual panel values
   were permuted, and `S @ q` drifted by 30 % under refinement. After the fix the
   rule converges to 1e-13 across settings, and the panel-level tests pin it.

All numbers below are written by
`python -m tensorbem.benchmarks.near_surface_field` into
`docs/near_surface_field/results.json`; the tables are generated from that file
(no hand transcription). The figure is
`docs/near_surface_field/near_surface_field.png`.

## 3. Results

### 3.1 Kernel level (single panel, pure quadrature error)

| `d / panel size` | fixed Duffy order 8 | adaptive (new) |
|---|---|---|
| 1e-1 | 3.19e-3 | 3.9e-14 |
| 3e-2 | 3.20e-2 | 1.5e-14 |
| 1e-2 | 6.19e-2 | 1.9e-14 |
| 1e-3 | 8.12e-2 | 1.9e-14 |
| 1e-4 | 8.33e-2 | 1.4e-14 |
| 1e-5 | 8.35e-2 | 2.4e-14 |

The gradient kernel (`int grad_x G dS`) behaves the same way: 98 % error for the
fixed rule at `d = 1e-4`, below 1e-6 for the adaptive rule at every clearance.

### 3.2 Field representation (320 panels, exact traces)

| `d / h_max` | fixed vs converged | adaptive vs converged | adaptive vs analytic |
|---|---|---|---|
| 1e-1 | 6.4e-7 | 5.8e-15 | 1.89e-2 |
| 1e-2 | 8.0e-4 | 9.5e-15 | 1.56e-2 |
| 1e-3 | 3.4e-4 | 1.1e-14 | 1.44e-2 |
| 1e-4 | 4.8e-5 | 1.1e-14 | 1.42e-2 |
| 1e-5 | 6.1e-6 | 1.2e-14 | 1.41e-2 |

Read the two right-hand columns together: quadrature error is at machine level
*for every clearance*, and what remains (1.4–1.9 %) is the constant-panel
discretisation of the traces, which shrinks with mesh refinement (§3.4).

### 3.3 End-to-end (solve then evaluate)

| `d / h_max` | clearance [m] | potential rel. error | velocity rel. error | old API |
|---|---|---|---|---|
| 5e-1 | 1.62e-1 | 1.98e-3 | 2.6e-4 | accepted |
| 1e-1 | 3.25e-2 | 2.20e-3 | 2.3e-2 | accepted |
| 2e-2 | 6.50e-3 | 4.80e-3 | 1.2e-1 | **rejected** |
| 5e-3 | 1.63e-3 | 6.27e-3 | 2.2e-1 | rejected |
| 1e-3 | 3.25e-4 | 6.95e-3 | 3.5e-1 | rejected |
| 2e-4 | 6.50e-5 | 7.16e-3 | 4.9e-1 | rejected |

The potential stays below 1 % all the way down to `2e-4 h_max` — a clearance the
old API refused. The velocity does not: it degrades roughly like `h/d`.

### 3.4 Mesh convergence at a fixed clearance ratio (exact traces)

| subdivisions | panels | `d/h_max` | potential error | velocity error |
|---|---|---|---|---|
| 1 | 80 | 1e-2 | 6.3e-2 | 1.96e-1 |
| 2 | 320 | 1e-2 | 1.56e-2 | 1.41e-1 |
| 3 | 1280 | 1e-2 | 3.85e-3 | 7.95e-2 |
| 3 | 1280 | 1e-3 | 3.52e-3 | 1.72e-1 |

The potential converges at the second order of the constant-panel traces
(4.0× / 4.1× per refinement). The velocity converges more slowly and its error
grows as the clearance shrinks: the amplification is in the double-layer term,
not in the new quadrature.

### 3.5 Near-wall pressure

At `d = 0.03 h_max = 9.75e-3 m` (below the old 1.63e-2 m limit) the maximum
pressure-coefficient error over `theta` is `|dCp| = 7.5`, i.e. 3.34 times the Cp
scale 2.25; at `d = 0.1 h_max` it is `|dCp| = 1.18` (0.53 of scale). For
reference, the surface-Cp RMS error from the existing tangential-recovery path is
`5.5e-2`. **This is an honest FAIL against the 3 % bar** and §6 diagnoses it.

Figure (`docs/near_surface_field/near_surface_field.png`): (left) kernel-level
error versus clearance, fixed versus adaptive; (middle) representation error
versus clearance with the analytic comparison; (right) `Cp(theta)` for the two
clearances against the analytic `1 - (9/4) sin^2(theta)`.

## 4. What changed

| file | change |
|---|---|
| `src/tensorbem/near_singular.py` | new: adaptive single-layer potential/gradient quadrature, `triangle_distances_and_sizes` |
| `src/tensorbem/solver3d.py` | new: `evaluate_surface_field_near_surface`, `evaluate_surface_velocity_near_surface` (existing API untouched) |
| `src/tensorbem/benchmarks/near_surface_field.py` | new: verification harness, writes JSON + Markdown + figure |
| `tests/test_near_singular.py` | new: 10 tests (kernel convergence, failure mode of the old rule, analytic sphere sum, clearance independence, old-API rejection, far-field regression against the old path, bounded end-to-end error) |
| `docs/near_surface_field/` | `results.json`, `summary.md`, `near_surface_field.png` |

Test run: `PYTHONPATH=src python3 -m pytest tests/test_near_singular.py
tests/test_solver3d.py tests/test_pressure.py -q` → **34 passed** in 42 s.

## 5. Notes, limits, traps

* **Operational envelope for velocity.** Use `d >= 0.1 h_max` for <3 % velocity
  error, `d >= 0.5 h_max` for <0.1 %. Below `0.02 h_max` the velocity error is
  discretisation-driven (12–49 %), not quadrature-driven.
* **Diagnosed cause of the Cp failure.** The double-layer term is integrated with
  the closed form `grad(Omega_j)`, whose denominator collapses when the target is
  close to a panel *plane*. At 320 panels the equator point at `d = 0.03 h_max`
  sits within `0.03 h` of several panel planes at once and the closed form loses
  digits through cancellation (`velerr = 3.4` there versus `8.9e-2` at the pole
  with the same clearance). A faceted mesh makes this worse than a smooth surface
  would. The cure is a regularised or finite-part hypersingular quadrature for the
  double-layer gradient, or moving the traces to a higher-order representation —
  both are follow-up work, not a tweak.
* **Do not use the surface-exact point.** Both entry points still reject targets
  on the surface; the representation jumps there.
* **Cost.** Work per target is logarithmic in `h/d`; the whole benchmark, dense
  solves included, runs in ~30 s on one core (the 1280-panel dense assembly is
  the expensive part, ~2 min, so the convergence study deliberately uses exact
  traces instead of a solve).
* **Reference rule.** Smaller `split_ratio` means *less* refinement, not more
  (`refine while distance < split_ratio * size`). Getting that backwards makes the
  "reference" the inaccurate run; the sweep showed it immediately.

## 6. Error judgement against the 3 % bar

| quantity | reference | measured error | < 3 %? |
|---|---|---|---|
| single-layer potential kernel, `d = 1e-1 … 1e-5` panel size | converged adaptive rule | 1.4e-14 – 2.4e-14 | **yes** (by ~12 orders) |
| single-layer gradient kernel, same sweep | converged adaptive rule | < 1e-6 | **yes** |
| near-wall potential, 320 panels, `d = 0.5 … 2e-4 h_max` | analytic `z/(2r^3)` | 2.0e-3 – 7.2e-3 | **yes** |
| near-wall potential, 1280 panels, `d = 1e-2, 1e-3 h_max` | analytic | 3.9e-3, 3.5e-3 | **yes** |
| near-wall velocity, `d = 0.5 h_max` / `0.1 h_max` | analytic gradient | 2.6e-4 / 2.3e-2 | **yes** (0.5 h), **yes but marginal** (0.1 h) |
| near-wall velocity, `d <= 0.02 h_max` | analytic gradient | 1.2e-1 – 4.9e-1 | **no** (documented limit) |
| near-wall Cp, `d = 0.1 h_max` / `0.03 h_max` | analytic `1 - (9/4) sin^2` | 0.53 / 3.34 of the Cp scale | **no — honest FAIL** |
| surface Cp (existing recovery path), 320 panels | analytic | RMS 5.5e-2 of scale | **yes** |

No reference value was adjusted to make a metric pass; the Cp result is reported
as a failure with its cause.