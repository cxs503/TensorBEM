# CASE-BEM-ICE-DISPERSION -- ice-cover mass-loading dispersion (TensorBEM)

**Case** `CASE-BEM-ICE-DISPERSION` · **repository** TensorBEM 0.5.0 · **environment** Python 3.11.6, numpy 2.4.6

## 1. Case introduction (physics, geometry, parameters)

A 1 m ice sheet floating on water is treated as a **mass load** (inertia,
no bending stiffness) -- the Weitz & Keller (1950) model of an ice-covered
sea surface.  The water column of depth `H` is inviscid and irrotational, the
bottom is rigid, and the motion is a plane periodic wave of wavenumber `k`.
The free-surface condition is a Robin condition, so the dispersion relation
is `omega^2 = g k tanh(kH) / (1 + mu k tanh(kH))` with `mu = rho_i h_i / rho_w`.
Two series are benchmarked: a deep-water series (`H = 200 m`, `kH = 4 .. 80`)
and a shallow series (`H = 2 m`, `kH = 0.04 .. 0.4`).

| quantity | symbol | value | unit |
|---|---|---|---|
| ice thickness | h_i | 1 | m |
| ice density | rho_i | 917 | kg/m^3 |
| water density | rho_w | 1025 | kg/m^3 |
| gravity | g | 9.807 | m/s^2 |
| mass-loading parameter | mu = rho_i h_i/rho_w | 0.8946 | m |
| cutoff frequency | sqrt(g/mu) | 3.311 | rad/s |
| cutoff period | 2 pi/sqrt(g/mu) | 1.898 | s |
| deep series depth | H | 200 | m |
| shallow series depth | H | 2 | m |
| panels per wavelength | - | 64 | - |
| tolerance | - | 0.05 | - |

### 1.1 Where the reference numbers come from

* The relation itself is Weitz & Keller (1950), *Some remarks on sea ice*,
  restated as eq. (3) of Fox & Squire (1994), Phil. Trans. R. Soc. A 352.
  The same relation is already pinned inside this repository by
  `tests/test_ice_mass_loading.py::test_deep_water_limit_matches_weitz_keller`
  (capability row `ice-region-hydrodynamics`).
* **Zero hand-copy.** Every number in the tables below is computed at run
  time from that closed form (`tensorbem.ice_mass_loading_dispersion.weitz_keller_*`)
  and written by the driver, never typed in.
* **Honest limit:** the original 1950 report's numeric table is not
  reachable from this (offline) environment, so `ref` here is the closed
  form evaluated programmatically rather than a transcribed table entry.
  **WITHHELD**: verification against the printed page of Weitz & Keller
  (1950).  The closed form is instead anchored by two independent limits
  that the derivation itself produces (deep water, long wave), both checked
  against the finite-depth relation at the 1 % level (C5, C6).

## 2. Method and implementation

One period cell (`0 <= x <= 2 pi/k`, `-H <= z <= 0`) of the ice-covered
basin is discretised with constant panels on all four sides.  The Laplace
boundary integral equation with free term 1/2 is solved for the tied mixed
problem (Dirichlet on the free surface, rigid bottom, periodic walls
`phi_R = phi_L`, `q_R = -q_L`).  Two numerical routes produce the free-surface
depth factor `Lambda = (dphi/dz)/phi`:

1. **driven solve** -- prescribe `phi = cos(k x)`, solve for the flux, form the
   L2 Rayleigh quotient `Lambda = sum(q_i phi_i l_i)/sum(phi_i^2 l_i)`;
2. **DtN map** -- assemble the dense Dirichlet-to-Neumann matrix of the free
   surface and evaluate the same quotient through it (independent code path).

The frequency then follows from the Robin condition alone,
`omega^2 = g Lambda / (1 + mu Lambda)`.  `tanh(kH)` appears **only** on the
reference side, so the comparison is a genuine numerical verification of the
dispersion relation rather than an algebraic identity.

Modules: `tensorbem.ice_mass_loading_dispersion` (mesh, layer matrices, ties,
driven/DtN solves, mode sweep, interior field recovery), called from
`tensorbem.ice.mass_loading_omega_bem`; driver
`tensorbem.benchmarks.ice_mass_loading_dispersion`.

## 3. Results

### 3.1 Deep-water series (H = 200 m)

| k [1/m] | kH | Lambda BEM | Lambda ref | omega ref [rad/s] | omega got [rad/s] | error | <5% | panels |
|---|---|---|---|---|---|---|---|---|
| 0.02 | 4 | 0.02001 | 0.01999 | 0.4388 | 0.4391 | +0.070% | OK | 154 |
| 0.05 | 10 | 0.05007 | 0.05 | 0.6851 | 0.6856 | +0.069% | OK | 164 |
| 0.1 | 20 | 0.1001 | 0.1 | 0.9488 | 0.9494 | +0.066% | OK | 172 |
| 0.2 | 40 | 0.2003 | 0.2 | 1.29 | 1.291 | +0.062% | OK | 182 |
| 0.4 | 80 | 0.4006 | 0.4 | 1.7 | 1.701 | +0.054% | OK | 190 |

### 3.2 Shallow-water series (H = 2 m)

| k [1/m] | kH | Lambda BEM | Lambda ref | omega ref [rad/s] | omega got [rad/s] | error | <5% | panels |
|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.04 | 0.0008057 | 0.0007996 | 0.08852 | 0.08886 | +0.385% | OK | 144 |
| 0.05 | 0.1 | 0.005006 | 0.004983 | 0.2206 | 0.2211 | +0.221% | OK | 144 |
| 0.1 | 0.2 | 0.01978 | 0.01974 | 0.4361 | 0.4366 | +0.113% | OK | 144 |
| 0.2 | 0.4 | 0.07609 | 0.07599 | 0.8353 | 0.8358 | +0.060% | OK | 144 |

### 3.3 Higher periodic modes (same cell, L = 2 pi/k)

| mode m | k_m [1/m] | panels/wavelength | Lambda BEM | Lambda ref | error | interior BV error |
|---|---|---|---|---|---|---|
| 1 | 0.1 | 64 | 0.1001 | 0.1 | +0.145% | 0.00297 |
| 2 | 0.2 | 32 | 0.2006 | 0.2 | +0.292% | 0.00509 |
| 3 | 0.3 | 21.3 | 0.3015 | 0.3 | +0.493% | 0.00654 |

### 3.4 Field-level evidence

The potential recovered from the boundary data on a 41 x 21 interior grid
differs from the exact driven mode by **0.00345**
(relative 0.557 %, `figs/fig_field.png`).

### 3.5 Figures

* `figs/fig_dispersion.png` -- ref curves (finite depth, deep-water limit,
  long-wave limit, cutoff) with the BEM points on top.
* `figs/fig_convergence.png` -- depth-factor error vs panels per wavelength.
* `figs/fig_sensitivity.png` -- ice-thickness and water-depth sensitivity.
* `figs/fig_field.png` -- BEM potential vs exact mode in the cell (field plot).

## 4. Convergence and parameter sensitivity

**deep**

| panels/wavelength | panels | Lambda BEM | Lambda ref | error | observed order |
|---|---|---|---|---|---|
| 8 | 38 | 0.1036 | 0.1 | +3.633% | n/a |
| 16 | 62 | 0.1013 | 0.1 | +1.294% | 1.49 |
| 32 | 100 | 0.1004 | 0.1 | +0.430% | 1.59 |
| 64 | 172 | 0.1001 | 0.1 | +0.145% | 1.57 |
| 128 | 310 | 0.1001 | 0.1 | +0.052% | 1.48 |

**intermediate**

| panels/wavelength | panels | Lambda BEM | Lambda ref | error | observed order |
|---|---|---|---|---|---|
| 8 | 32 | 0.0786 | 0.07616 | +3.207% | n/a |
| 16 | 48 | 0.07686 | 0.07616 | +0.918% | 1.80 |
| 32 | 80 | 0.07638 | 0.07616 | +0.293% | 1.65 |
| 64 | 144 | 0.07628 | 0.07616 | +0.152% | 0.95 |
| 128 | 276 | 0.07619 | 0.07616 | +0.042% | 1.84 |

**shallow**

| panels/wavelength | panels | Lambda BEM | Lambda ref | error | observed order |
|---|---|---|---|---|---|
| 8 | 32 | 0.0008182 | 0.0007996 | +2.327% | n/a |
| 16 | 48 | 0.0008163 | 0.0007996 | +2.086% | 0.16 |
| 32 | 80 | 0.0008112 | 0.0007996 | +1.448% | 0.53 |
| 64 | 144 | 0.0008057 | 0.0007996 | +0.771% | 0.91 |
| 128 | 272 | 0.0008017 | 0.0007996 | +0.267% | 1.53 |

**Sensitivity to ice thickness** (k = 0.10 1/m, H = 200 m)

| h_i [m] | mu [m] | omega got [rad/s] | omega ref [rad/s] | error |
|---|---|---|---|---|
| 0.5 | 0.4473 | 0.9695 | 0.9689 | +0.069% |
| 1 | 0.8946 | 0.9494 | 0.9488 | +0.066% |
| 2 | 1.789 | 0.9126 | 0.912 | +0.061% |

**Sensitivity to water depth** (k = 0.10 1/m, h_i = 1 m)

| H [m] | kH | omega got [rad/s] | omega ref [rad/s] | deep-water limit [rad/s] | error |
|---|---|---|---|---|---|
| 2 | 0.2 | 0.4366 | 0.4361 | 0.9488 | +0.113% |
| 5 | 0.5 | 0.6601 | 0.6597 | 0.9488 | +0.055% |
| 20 | 2 | 0.9335 | 0.9329 | 0.9488 | +0.063% |
| 200 | 20 | 0.9494 | 0.9488 | 0.9488 | +0.066% |

## 5. Pitfalls, limitations, scope

* The model is a **mass load**: bending stiffness (the flexural regime, valid
  for wavelengths above a few metres of ice) is not included here; the elastic
  extension lives in `tensorbem.flexural_gravity`.
* `omega` is bounded by the cutoff `sqrt(g/mu)`: for `omega` above it the
  Robin coefficient changes sign and no propagating ice wave exists
  (`tensorbem.ice.robin_coefficient` raises there).
* Constant panels converge at order ~1.5, not 2, because the depth factor is a
  flux at the panel corners of the cell; the error is still below 1.5 % at the
  default 64 panels per wavelength.
* The dense eigenproblem of the free-surface DtN map in the
  piecewise-constant basis is spectrally polluted by the panel corners
  (measured: 20 % error on mode 1), so the DtN route is evaluated as a
  quotient `Lambda = <T c, c>/<c, c>` instead of an eigenvalue solve.
* Ecosystem friction: `scipy.linalg.solve` returned a solution with relative
  residual 3.6e3 on this operator (condition number 5.4e2) in the reference
  environment while `numpy.linalg.solve` gave 6e-15; the module therefore uses
  the numpy path and reports the linear-system residual in every artifact.
* Scope: infinite ice sheet, uniform depth, normal incidence, linear theory.
  Ice edges, oblique incidence and finite ice floes are out of scope.

## 6. Error judgement against the acceptance criteria

| criterion | quantity | value | threshold | verdict |
|---|---|---|---|---|
| C1_omega_error_below_tolerance | max |omega_got/omega_ref - 1| over every dispersion row | 0.003846 | 0.05 | PASS |
| C2_routes_agree | driven solve vs assembled Dirichlet-to-Neumann map | 1.521e-14 | 1e-09 | PASS |
| C3_refinement_reduces_error | monotone under refinement: {'deep': True, 'shallow': True, 'intermediate': True}; mean observed order | 1.292 | 0.8 | PASS |
| C4_interior_field_error | max |phi_bem - phi_exact| / max |phi_exact| on the 41x21 grid | 0.005575 | 0.01 | PASS |
| C5_deep_water_anchor | BEM omega vs deep-water limit g k/(1+mu k), rows with kH >= 10 (gap is the panel discretisation error) | 0.0006858 | 0.01 | PASS |
| C6_long_wave_anchor | BEM omega vs long-wave limit sqrt(gH) k, rows with kH <= 0.1 | 0.00322 | 0.01 | PASS |

**Is the key quantity within tolerance?** The largest frequency error over
all 9 dispersion rows is **0.385 %**,
against a predeclared tolerance of 5 % (and the platform's 15 % acceptability line) -- **yes**.
All six criteria passed: **True**.

## Reproduction

```
PYTHONPATH=src python -m tensorbem.benchmarks.ice_mass_loading_dispersion --figure
PYTHONPATH=src python -m pytest tests/test_ice_mass_loading_dispersion.py -q
```

The driver writes `numbers.json`, `table.md`, this report and the four figures;
run time is ~40 s on one core (no cluster needed).  The criteria are evaluated
inside `run()` and the exit status is non-zero when any of them fails.

## Appendix A -- derivation of the mass-loading condition

Kinematic condition `deta/dt = dphi/dz`; the sheet's vertical momentum balance
with the linearised pressure `p = rho_w (-dphi/dt - g eta)` at the surface gives
`m deta^2/dt^2 = p` with `m = rho_i h_i`.  With `phi ~ exp(-i omega t)` and
`q = dphi/dz`:

```
- m omega^2 eta = i rho_w omega phi - rho_w g eta,    eta = i q / omega
=>  q (rho_w g - m omega^2) = rho_w omega^2 phi
=>  dphi/dz = Lambda phi,   Lambda = rho_w omega^2 / (rho_w g - rho_i h_i omega^2)
```

For `phi = cos(kx) cosh(k(z+H))` the flux ratio is `Lambda = k tanh(kH)`, so
`k tanh(kH) (rho_w g - rho_i h_i omega^2) = rho_w omega^2`, i.e.
`omega^2 = g k tanh(kH) / (1 + mu k tanh(kH))` with `mu = rho_i h_i/rho_w`.
Limits: `kH -> infinity` gives `omega^2 = gk/(1+mu k)`; `kH -> 0` gives
`omega -> sqrt(gH) k` with the mass loading entering only at `O((kH)^2)`.
