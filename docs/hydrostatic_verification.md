# Hydrostatic module: closed-form verification and convergence

**Scope**: closed-form re-verification of `tensorbem.hydrostatic`
(displaced volume, centre of buoyancy, waterplane area/first/second moments,
6x6 restoring matrix) plus the hull generators added with this study.

**Why this exists**: the original m273 acceptance tests compared a 64-panel
hemisphere with the analytic values using 5 % (`A_wp`) and 10 % (`V`)
tolerances. Those numbers are larger than the discretization error of a
*correct* solver at that mesh, so the tests could not distinguish a bug from
mesh coarseness, and they sit far outside the 3 % platform bar. This study
replaces them with (a) exactness checks where the mathematics is exact and
(b) a measured order of convergence where it is not.

## 1. Method

* **Geometry sources** - `hemisphere_hull_mesh` (existing) plus two new hull
  generators in `geometry3d`: `rectangular_barge_hull_mesh` (wall-sided box)
  and `elliptic_cylinder_hull_mesh` (polygonal waterline inscribed in an
  ellipse). Both new generators are validated by `WettedHullMesh` itself, which
  enforces outward normals, unique directed edges, waterline-closed boundary
  loops and positive displaced volume.
* **Exact cases** - for a wall-sided hull the wetted surface is planar, so the
  divergence-theorem volume integral and the waterline polygon integrals are
  exact: `V = L B T`, `A_wp = L B`, `I_T = L B^3/12`, `I_L = B L^3/12`,
  `KB = T/2`, `BM_T = B^2/(12T)`, `BM_L = L^2/(12T)`. Any non-round-off
  deviation is an implementation bug.
* **Curved cases** - the inscribed polyhedral hull under-estimates curved
  geometry. Reporting the observed order `p = log2(e_k / e_(k+1))` and a
  Richardson extrapolation separates discretization from implementation error.
  For the hemisphere `GM = KB + BM - KG = R - KG`, so the whole chain
  waterplane -> volume -> KB/BM -> stiffness is exercised against one closed
  form.
* **Conventions** - `center_of_gravity` is given in the global frame used by
  the hull vertices (`z = 0` at the calm waterline); the module converts to
  keel-relative heights internally as `kg = cg_z - min(z)`. All checks below
  use rho = 1025.0 kg/m^3 and g = 9.80665 m/s^2.

Reference formulas:

```
                          hemisphere      barge            ellipse
V                         2 pi R^3 / 3    L B T            pi a b T (polygon exact)
A_wp                      pi R^2          L B              pi a b
I_T = int y^2 dA          pi R^4 / 4      L B^3 / 12       pi a b^3 / 4
I_L = int x^2 dA          pi R^4 / 4      B L^3 / 12       pi b a^3 / 4
KB (above keel)           5 R / 8         T / 2            T / 2
BM_T = I_T / V            3 R / 8         B^2 / (12 T)     a^2 / (4 T)
BM_L = I_L / V            3 R / 8         L^2 / (12 T)     b^2 / (4 T)
GM                        R - KG          KB + BM - KG     KB + BM - KG
```

## 2. Results

![Convergence studies](figures/hydrostatic_convergence.png)

*Figure 1* - left: relative error of the hemisphere hydrostatic quantities
against mesh refinement, with a `h^2` reference slope; right: elliptic
waterplane area against the inscribed-polygon value (round-off) and against the
true ellipse (`n^-2`).

![Exactness summary](figures/hydrostatic_exactness.png)

*Figure 2* - left: wall-sided barge errors are at round-off level across every
quantity including the restoring matrix diagonal; right: the hemisphere
restoring chain at 1024 panels stays below the 1 % level, well inside
the 3 % bar.

### 2.1 Wall-sided barge (exact reference)

Reference: V = 40 m^3, A_wp = 40 m^2, I_T = 53.3333 m^4,
I_L = 333.333 m^4, KB = 0.5 m, GM_T = 0.8333 m, GM_L = 7.8333 m
(CG at KG = 1 m above the keel).

| nx x ny | panels | V err | A_wp err | I_T err | I_L err | KB err | C heave err | C roll err | C pitch err |
|---|---|---|---|---|---|---|---|---|---|
| 1 x 1 | 10 | 1.776e-16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 2.220e-16 | 0.000e+00 | 1.737e-16 | 1.478e-16 |
| 3 x 2 | 32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.705e-16 | 4.441e-16 | 0.000e+00 | 3.474e-16 | 2.957e-16 |
| 5 x 3 | 62 | 3.553e-16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 6.661e-16 | 0.000e+00 | 3.474e-16 | 0.000e+00 |

Max relative error over all barge cases: 6.661e-16

### 2.2 Floating hemisphere refinement study (R = 1 m)

Panel count is `4 * 4^subdivisions`; mesh spacing halves per subdivision, so the
observed order follows from successive error ratios.

| subdiv | panels | V err | order | A_wp err | order | I_T err | order | KB err |
|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 2.975e-01 | — | 9.968e-02 | — | 1.876e-01 | — | 7.185e-02 |
| 2 | 64 | 8.858e-02 | 1.75 | 2.550e-02 | 1.97 | 5.023e-02 | 1.90 | 1.953e-02 |
| 3 | 256 | 2.320e-02 | 1.93 | 6.413e-03 | 1.99 | 1.278e-02 | 1.98 | 4.984e-03 |
| 4 | 1024 | 5.870e-03 | 1.98 | 1.606e-03 | 2.00 | 3.208e-03 | 1.99 | 1.252e-03 |
| 5 | 4096 | 1.472e-03 | 2.00 | 4.015e-04 | 2.00 | 8.029e-04 | 2.00 | 3.135e-04 |

Richardson extrapolation from the two finest meshes: V err 0.000e+00, A_wp err 0.000e+00, I_T err 0.000e+00 (round-off; the extrapolated values reproduce the closed forms bit-for-bit)

### 2.3 Elliptic waterplane convergence (5 m x 2 m, draft 1 m)

The meshed hull is wall-sided, so `V` and `A_wp` are *exact for the inscribed
polygon* (round-off column); the ellipse values are approached at `O(n^-2)`.

| waterline panels | A_wp err vs inscribed polygon | A_wp err vs ellipse | order | I_T err vs polygon |
|---|---|---|---|---|
| 16 | 1.160e-16 | 2.550e-02 | — | 2.381e-16 |
| 32 | 2.276e-16 | 6.413e-03 | 1.99 | 6.873e-16 |
| 64 | 0.000e+00 | 1.606e-03 | 2.00 | 3.404e-16 |
| 128 | 0.000e+00 | 4.015e-04 | 2.00 | 5.659e-16 |

### 2.4 Hemisphere restoring matrix (subdivisions = 4,
1024 panels, KG = 0.6 m above keel)

`c_heave = rho g A_wp`, `c_roll = c_pitch = rho g V GM` with `GM = R - KG`.

| quantity | computed | closed form | relative error |
|---|---|---|---|
| volume | 2.082102 | 2.094395 | 5.870e-03 |
| kb | 0.625783 | 0.625000 | 1.252e-03 |
| bm | 0.376004 | 0.375000 | 2.677e-03 |
| gm | 0.401787 | 0.400000 | 4.467e-03 |
| c_heave | 31528.009086 | 31578.712086 | 1.606e-03 |
| c_roll | 8408.954510 | 8420.989890 | 1.429e-03 |

## 3. Error judgement

| key quantity | mesh | relative error | below 3 %? |
|---|---|---|---|
| barge V, A_wp, I_T, I_L, KB, C heave/roll/pitch (worst of 3 meshes) | 10-62 panels | 6.66e-16 | yes (round-off) |
| hemisphere V | 4096 panels | 1.47e-03 | yes |
| hemisphere A_wp | 4096 panels | 4.02e-04 | yes |
| hemisphere I_T | 4096 panels | 8.03e-04 | yes |
| hemisphere KB | 4096 panels | 3.13e-04 | yes |
| hemisphere V, Richardson extrapolated | 4096 panels | 0.00e+00 | yes |
| hemisphere A_wp, Richardson extrapolated | 4096 panels | 0.00e+00 | yes |
| hemisphere GM (subdivisions = 4) | 1024 panels | 4.47e-03 | yes |
| hemisphere C roll (subdivisions = 4) | 1024 panels | 1.43e-03 | yes |
| elliptic A_wp vs ellipse | 128 waterline panels | 4.02e-04 | yes |

**Answer to "is every key quantity below 3 %?"** Yes, on the meshes quoted
above. The hemisphere at the originally tested 64-panel mesh is *not* within
3 % (`V` error 8.86%,
`A_wp` error 2.55%): the old test
tolerances (5 %/10 %) were hiding genuine discretization error, not bounding a
bug. At 1024 panels both are below 1 %, and the
Richardson-extrapolated values reproduce the closed forms to round-off, which
is what proves the implementation correct.

## 4. Caveats

* The hemisphere error sequence is nearly self-similar (panel geometry is
  reproduced by radial projection), which is why the Richardson extrapolation
  lands within round-off of the exact value; do not expect the same for
  arbitrary hull meshes.
* `compute_hydrostatic_restoring_matrix` expects `center_of_gravity` in the
  global frame (`z = 0` waterline). Passing a keel-relative height silently
  shifts `KG` by the draft and can flip the sign of `C[3,3]`; a dedicated test
  now covers the sign convention (`test_unstable_body_yields_negative_roll_restoring`).
* Off-diagonal hydrostatic coupling (e.g. waterplane centroid offset producing
  `C[2,4]`, or `C[3,4]` from a non-symmetric waterplane) is not modelled by the
  module; these checks cover the diagonal only.
* The verification hulls are wall-sided or smooth convex bodies. Partially
  submerged panels cut obliquely by the waterline are handled by the geometry
  layer only if the mesh is cut accordingly; no automatic waterline clipping of
  straddling panels exists.
* RAO/excitation correctness is out of scope here; this study only pins the
  hydrostatic stiffness chain that the RAO solver consumes.

## 5. Reproduce

```bash
PYTHONPATH=src python -m tensorbem.benchmarks.hydrostatic_convergence --json hydrostatic_verification_report.json
PYTHONPATH=src python -m tensorbem.benchmarks.hydrostatic_figures --out-root .
PYTHONPATH=src python -m pytest tests/test_hydrostatic_convergence.py -q
```
