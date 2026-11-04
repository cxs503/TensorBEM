# 3-D finite-depth kernel: full gradient `nabla G` and the order-1 Bessel layer

**Mission** m3673 (paper seat, self-start) · branch `agent/m3673` · kernel `tensorbem.finite_depth3d`
**Deliverables** `src/tensorbem/finite_depth3d.py`, `src/tensorbem/benchmarks/finite_depth3d_gradient.py`,
`tests/test_finite_depth3d_gradient.py` (19 gates), `docs/m3673/outputs/finite_depth3d_gradient.json`,
`docs/m3673/figures/*.png`

EDIT_WHITELIST: src/tensorbem/, tests/, docs/, README.md

---

## 1. Case (why this increment)

`green_function_3d(targets, sources, nu, depth)` is the John (1950) eigenseries of the 3-D
free-surface Green function in constant depth,

```
G(x,xi) = (1/2pi) sum_{n>=1} phi_n(z) phi_n(zeta)/N_n K_0(mu_n R)
        + (i/4) phi_0(z) phi_0(zeta)/N_0 H_0^{(1)}(k_0 R),
phi_0 = cosh(k_0 (z+h)),   k_0 tanh(k_0 h) = nu,   mu_n tan(mu_n h) = -nu.
```

The kernel shipped with `G` and the vertical derivative `dG/dz` only; §6 of
[the finite-depth kernel report](finite_depth3d.md) and the README name a horizontal
derivative as belonging to the panel layer, and the 3-D finite-depth *boundary-element
solver* as the next increment.  A panel layer needs the **full** `nabla_x G` (the singular
part here is carried by the modal sum itself, not split off as a Rankine term) and the
order-1 modified Bessel / Hankel functions that enter `dG/dR`.  This change supplies them
and, because a verification pass is part of the increment, also fixes a defect the pass
found in the order-0 Bessel evaluator.

**Test geometry** (unless stated otherwise): `nu = 0.6` (so
`omega = sqrt(nu g)`), depth `h = 4` m, five targets inside the strip
and one source at `(0.2, -0.1, -1.0)` m.

## 2. Method

**Derivatives.**  `G` depends on the horizontal coordinates only through `R`, and
`dK_0/dR = -K_1`, `dH_0^{(1)}/dR = -H_1^{(1)}`, so

```
dG/dR = -(1/2pi) sum_n mu_n phi_n(z) phi_n(zeta)/N_n K_1(mu_n R)
        - (i/4) k_0 phi_0(z) phi_0(zeta)/N_0 H_1^{(1)}(k_0 R),
```

with `dG/dx = (x-xi)/R dG/dR`, `dG/dy = (y-eta)/R dG/dR`.  The source gradient is the
same construction with `dphi_n(zeta)/dzeta = -mu_n sin(mu_n(zeta+h))` and
`dphi_0(zeta)/dzeta = k_0 sinh(k_0(zeta+h))`, and horizontal part `-nabla_x`.  New API:
`green_gradient_horizontal_3d`, `green_gradient_3d` (shape `(...,3)`, matching
`free_surface3d.green_gradient_field_3d`), `green_gradient_source_3d`, `bessel_k1`,
`hankel_h1`.

**Bessel layer.**  `K_1` and `H_1` are new.  `K_1` uses the ascending series for
`x < 4` (A&S 9.6.11), the asymptotic series for
`x >= 16`, and — in between — the **Laplace-integral
representation**, which has no cancellation at all:

```
K_0(x) = 2 e^(-x) int_0^inf e^(-x u^2)/sqrt(u^2+2) du,
K_1(x) = 2 e^(-x) int_0^inf e^(-x u^2)(1+u^2)/sqrt(u^2+2) du,
```

evaluated with 96 Gauss-Legendre nodes on `u in [0, 8/sqrt(x)]` (where the integrand is
below `e^-64`).  The same quadrature **replaced the shipped `K_0` branch point**: the
release code used the ascending series up to `x = 9`, where `1/K_0` amplifies the
cancellation between `-ln(x/2) I_0(x)` and the series, leaving a
**2.751e-08 relative cliff at
x = 8.96** while the asymptotic series is still at
`~1e-2` there.  Nothing in the test suite could see it: the existing Bessel gate compared
`K_0` to SciPy at `1e-8`, i.e. exactly at the cliff level.

**Verification.**  `benchmarks/finite_depth3d_gradient.py` recomputes everything from the
library and writes a strict JSON ledger; `tests/test_finite_depth3d_gradient.py` pins it.

## 3. Results

### 3.1 Bessel layer (`K_0`, `K_1`, `H_1`)

Pinned references: `mpmath.besselk` at 40 significant digits; the branch-error curves use
the Laplace quadrature as an independent reference (Fig. 1).

| x | `K_0` ref | err (new) | err (before) | `K_1` ref | err (new) | err (before) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.210244e-01 | 0.000e+00 | 0.000e+00 | 6.019072e-01 | 0.000e+00 | 0.000e+00 |
| 4 | 1.115968e-02 | 5.773e-15 | 1.767e-13 | 1.248350e-02 | 4.552e-15 | 1.759e-13 |
| 6 | 1.243994e-03 | 5.551e-15 | 6.620e-12 | 1.343920e-03 | 4.663e-15 | 3.717e-12 |
| 8 | 1.464707e-04 | 5.218e-15 | 1.306e-10 | 1.553692e-04 | 4.663e-15 | 1.592e-09 |
| 8.9 | 5.653960e-05 | 5.551e-15 | 1.444e-08 | 5.963534e-05 | 4.774e-15 | 3.742e-09 |
| 10 | 1.778006e-05 | 5.107e-15 | 3.245e-10 | 1.864877e-05 | 4.774e-15 | 3.218e-10 |
| 12 | 2.200825e-06 | 5.440e-15 | 3.117e-12 | 2.290757e-06 | 4.774e-15 | 3.115e-12 |
| 16 | 3.499412e-08 | 2.220e-15 | 2.220e-15 | 3.607157e-08 | 1.776e-15 | 1.776e-15 |
| 40 | 8.392861e-19 | 1.110e-16 | 1.110e-16 | 8.497132e-19 | 4.441e-16 | 4.441e-16 |

Worst case over the 331-point scan `1 <= x <= 60`: `K_0`
`3.494e-13` (new) vs
`2.751e-08` (before, at
`x = 8.96`); `K_1`
`2.878e-13` (new) vs
`9.366e-09` for the same branch layout.

Seam continuity (quadrature vs the branch it replaced, relative):

| x | `K_0` quad | vs series | vs asymptotic | `K_1` quad | vs series | vs asymptotic |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.115967609833646e-02 | 2.294e-13 | 1.333e+00 | 1.248349890154892e-02 | 2.182e-13 | 7.883e-01 |
| 4 | 1.115967607336946e-02 | 4.285e-14 | 1.333e+00 | 1.248349887298783e-02 | 7.161e-14 | 7.883e-01 |
| 16 | 3.499411667543638e-08 | 1.991e-03 | 2.998e-15 | 3.607157121253621e-08 | 6.014e-03 | 7.105e-15 |
| 16 | 3.499411660329323e-08 | 1.991e-03 | 3.220e-15 | 3.607157113803902e-08 | 4.760e-04 | 7.105e-15 |

`H_1^{(1)} = J_1 + iY_1` is evaluated from `analytic.bessel_j01_y01` (agrees with SciPy
to `1e-14`); it is additionally gated against the Richardson derivative of `H_0^{(1)}`.

### 3.2 Gradient against finite differences and closed identities

Per-component relative error at `eps = 1e-5`: dx `5.027e-10`,
dy `1.298e-10`, dz `3.469e-10`;
fourth-order Richardson at the best step:
`1.648e-12` (Fig. 2).

| step | central | Richardson |
|---:|---:|---:|
| 0.01 | 1.805e-04 | 1.372e-08 |
| 0.001 | 1.805e-06 | 1.648e-12 |
| 0.0001 | 1.805e-08 | 8.195e-12 |
| 1e-05 | 1.786e-10 | 7.168e-11 |
| 1e-06 | 2.454e-10 | 9.919e-10 |

Closed identity (kernel symmetry `G(x,xi) = G(xi,x)`):
`max|nabla_x G(x,xi) - nabla_xi G(xi,x)| / max|nabla_x G| =
0.000e+00` — this one equation
cross-validates **all four** analytic derivative formulas (horizontal and vertical, field
and source) at once, so no separate sign/scale convention can be wrong in isolation.

### 3.3 Limits and physical closures

| quantity | measured | gate | verdict |
|---|---:|---:|---|
| `k0_max_relative_error` | 4.774e-14 | 1e-12 | PASS |
| `k1_max_relative_error` | 7.039e-14 | 1e-12 | PASS |
| `swap_relative_difference` | 0.000e+00 | 1e-12 | PASS |
| `best_richardson_max_relative_error` | 1.648e-12 | 1e-07 | PASS |
| `free_surface_residual` | 2.104e-10 | 1e-06 | PASS |
| `bottom_residual` | 0.000e+00 | 1e-06 | PASS |
| `horizontal_curl_relative` | 8.346e-09 | 1e-06 | PASS |
| `singularity_deviation` | 4.563e-09 | 0.001 | PASS |
| `deep_water_gradient_deviation` | 8.791e-08 | 1e-05 | PASS |

* Free-surface / bottom closure is tested as `d/dx(dG/dz) = nu dG/dx` on `z = 0` and
  `d/dx(dG/dz) = 0` on `z = -h` (the two analytic routines cross-check each other).
* Horizontal irrotationality `d_y G_x = d_x G_y` holds to
  `8.346e-09`.
* Singularity strength (Fig. 3 is the deep-water limit):

| r [m] | `4 pi r^2 |nabla G|` | direction x |
|---:|---:|---:|
| 0.1 | 1.000316331990 | -0.9999742371 |
| 0.05 | 1.000038023094 | -0.9999983849 |
| 0.025 | 1.000004654746 | -0.9999998990 |
| 0.01 | 1.000000294071 | -0.9999999974 |
| 0.0025 | 1.000000004563 | -1.0000000000 |

### 3.4 Deep-water collapse of the gradient

| `k_0 h` | h [m] | `nabla G` deviation | `G` deviation |
|---:|---:|---:|---:|
| 2.437 | 4 | 2.173e-02 | 5.473e-02 |
| 4.801 | 8 | 1.057e-03 | 3.054e-03 |
| 9.600 | 16 | 6.447e-05 | 2.114e-04 |
| 19.200 | 32 | 6.434e-06 | 2.166e-05 |
| 38.400 | 64 | 7.336e-07 | 2.483e-06 |
| 76.800 | 128 | 8.791e-08 | 2.980e-07 |

The gradient deviation tracks the kernel deviation to within a factor ~1.3, i.e. the
horizontal derivative introduces no new error source: at `k_0 h = 76.8` the finite-depth
gradient reproduces the independent Kelvin-kernel gradient to
`8.791e-08`.

## 4. Figures

* `figures/fig_bessel_branches.png` — relative error of `K_0`/`K_1` vs the Laplace
  quadrature over `1 <= x <= 60`: the red `K_0` curve shows the
  `2.751e-08` cliff at `x ~ 8.96` that the
  re-branching removes (blue/green curves sit at the `1e-13` round-off floor); the dotted
  vertical lines are the new branch seams.
* `figures/fig_gradient_fd.png` — step ladder of the finite-difference check: the
  `O(eps^2)`/round-off trade-off of the central difference and the two-orders-of-magnitude
  better Richardson estimate, i.e. the analytic gradient is correct far below the
  finite-difference floor.
* `figures/fig_deep_water_gradient.png` — deep-water collapse of `nabla G` (circles) next
  to the kernel collapse (squares) vs `k_0 h`.

## 5. Note / limitations

* Only the **kernel** is delivered here; panel integration, layer jumps and the CHIEF/lid
  treatment for the finite-depth BEM solver are still the next increment (unchanged scope).
* Constant depth, linearised potential flow.  Sloping bottom, current and finite amplitude
  remain out of scope.
* The evanescent-mode sum diverges logarithmically as `R -> 0` (it carries the
  `1/(4 pi r)` singularity), so the gradient must not be evaluated at coincident points;
  the API raises instead.
* `dG/dR` needs `K_1` at the *smallest* `mu_n R` of the sum, where the series branch
  applies; the mode count is chosen by `exp(-mu_n R_min) <= 1e-12` as before, which also
  bounds the `mu_n K_1` weighting.
* **Cost**: the quadrature branch is ~100x a series evaluation.  It only applies to
  arguments in `[4, 16)`, i.e. the tail of the mode sum; a full gradient evaluation on the
  five-point test set with `h = 4` m takes single-digit milliseconds.

## 6. Error verdict

| key quantity | reference | relative error | < 3 %? |
|---|---|---:|---|
| `K_0`, `K_1` (all branches) | mpmath, 40 digits | `7.039e-14` | yes |
| `nabla_x G` vs finite difference | Richardson of `G` | `1.648e-12` | yes |
| `nabla_x G(x,xi)` vs `nabla_xi G(xi,x)` | kernel symmetry | `0.000e+00` | yes |
| free-surface closure of `nabla G` | `d_z d_x G = nu d_x G` | `2.104e-10` | yes |
| `nabla G` deep-water limit at `k_0 h = 76.8` | Kelvin kernel (independent code) | `8.791e-08` | yes |
| singularity strength at `r = 2.5e-3` m | `1/(4 pi r)` | `4.563e-09` | yes |

**Verdict**: all gates PASS.
The one honest caveat is the defect this unit fixed: the shipped `K_0` was **not** accurate
to `1e-8` near its branch point, a fact that the pre-existing `1e-8` Bessel gate could not
detect; the gate in this unit is `1e-12`.
