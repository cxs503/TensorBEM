# Wet modes and wave-excited response of a hull panel (BEM hydroelasticity)

Campaign node `IN41-N04`.  Module `tensorbem.hydroelastic`; generator
`python -m tensorbem.benchmarks.hydroelastic_verification`.

## 1. Case

A simply supported Kirchhoff panel of planform `a x b = 1 m x 0.6 m`, thickness `h = 10 mm`, steel
(`E = 2.1e+11 Pa`, `nu = 0.30`, `rho_p = 7850 kg/m^3`), mounted flush in an
infinite rigid baffle with seawater on one side
(`rho = 1025 kg/m^3`): the classical *wet panel* configuration of ship hydroelasticity (hull / bottom / tank panel).

| quantity | value |
|---|---|
| flexural rigidity `D` | 1.9231e+04 N m |
| mass per area `m''` | 7.8500e+01 kg/m^2 |
| plate mass | 47.1 kg |
| dry modes | (1,1), (2,1), (3,1) |
| dry frequencies | 92.88 Hz, 166.64 Hz, 289.57 Hz |

## 2. How it was done

Fluid: inviscid, incompressible, irrotational velocity potential with
`Laplace(phi) = 0` in the half space, `dphi/dz = 0` on the baffle and
`dphi/dz = -i omega w` on the panel.  In the zero-frequency limit the calm
free surface is a rigid wall, so the whole plane `z = 0` is Neumann and the
half-space Green function evaluated on that plane collapses to
`1/(2 pi |x - x'|)`; the generalised added mass of the plate modes is the
symmetric Galerkin single-layer integral

```
A_ij = (rho / 2 pi) * int_panel int_panel w_i(x) w_j(x') / |x - x'| dA dA'
```

evaluated by `tensorbem.hydroelastic.panel_added_mass_matrix` on a uniform
panel grid (levels 8x5, 16x10, 32x20): the self-panel term is
integrated **exactly** (the four-fold integral collapses to a one-dimensional
autocorrelation integral), separated panels use a 3x3 Gauss product rule and
panels closer than two panel sizes use a 4x4 subdivided 2x2-point rule.
Structure: exact dry modes `w_mn = sin(m pi x/a) sin(n pi y/b)`,
`omega_dry^2 = D k_mn^4 / m''`, `k_mn = pi sqrt((m/a)^2 + (n/b)^2)`.
Coupling: the generalised eigenproblem `K phi = omega^2 (M + A) phi` in the
dry-mode basis, and a partitioned displacement/pressure fixed point (structure
solve -> fluid re-linearisation -> under-relaxation, `omega_r = 0.7`).  Wave
excitation: Froude-Krylov pressure of a regular wave (deep-water wavenumber
`k = omega^2/g`); modal damping `C_ii = 2 zeta omega_wet,i (M_ii + A_ii)`.

## 3. Results

### 3.1 Kernel and quadrature against closed forms

| mesh | panels | uniform-mode added mass [kg] | rel. error vs closed form |
|---|---|---|---|
| 8x5 | 40 | 221.848628 | 2.39e-04 |
| 16x10 | 160 | 221.872655 | 1.31e-04 |
| 32x20 | 640 | 221.886555 | 6.83e-05 |

Closed form of `int int dA dA'/|x-x'|` over the panel: `221.901710` kg; in dimensionless form `A/(rho A_p^1.5) = 0.4658`.

Kernel constant.  The analytic plane-wave impedance `m_a = rho/k` (a plane wave under a rigid baffle gives `p = rho omega^2 w/k`, independent of the discretisation) is recovered as `k a -> infinity`: the excess `mu_BEM/mu_local - 1` decays as `k a^(-2.21)` (modes (3,1), (1,2), (2,2), (3,2), (4,2), (5,3), residual `3.94e-02`), i.e. `0.29%` at `k a = 50`.  If the `rho/(2 pi)` constant of the kernel were wrong by any factor, this excess would tend to a finite offset instead of vanishing.

### 3.2 Wet modes and added mass

| mode | `k_mn a` | `omega_dry` [rad/s] | `omega_wet` [rad/s] |
| `mu = A_ii/M_ii` | `omega_wet/omega_dry` | error vs extrapolated | GCI |
|---|---|---|---|---|---|---|
| (1,1) | 6.106 | 583.579 | 258.869 | 4.0751 | 0.4436 | 0.01% | 0.01% |
| (2,1) | 8.179 | 1047.009 | 597.389 | 2.0718 | 0.5706 | 0.10% | 0.12% |
| (3,1) | 10.782 | 1819.393 | 1155.123 | 1.5485 | 0.6349 | 0.21% | 0.26% |

Mesh-independent (Richardson-extrapolated) values: 258.896, 596.819, 1152.729 rad/s (observed order 2.48, 1.57, 1.53).
Local plane-wave model `omega^2 = D k^4/(m'' + rho/k)`: 329.417, 649.769, 1223.558 rad/s -- an asymptote, not a reference (see below).

Modal added-mass ratios and the local asymptote (production mesh):

| mode | `k a` | `mu` (BEM) | `mu` (local `rho/k`) | BEM/local |
|---|---|---|---|---|
| (1,1) | 3.66 | 4.0751 | 2.1384 | 1.906 |
| (2,1) | 4.91 | 2.0718 | 1.5965 | 1.298 |
| (3,1) | 6.47 | 1.5485 | 1.2111 | 1.279 |
| (1,2) | 6.56 | 1.5563 | 1.1943 | 1.303 |
| (2,2) | 7.33 | 1.2765 | 1.0692 | 1.194 |
| (3,2) | 8.45 | 1.0520 | 0.9268 | 1.135 |
| (4,2) | 9.81 | 0.8765 | 0.7982 | 1.098 |
| (5,3) | 13.33 | 0.6236 | 0.5878 | 1.061 |

The BEM added mass exceeds the local plane-wave value and the ratio decays
towards one as `k a` grows: the local impedance is the high-wavenumber
asymptote and cannot be used as the reference for the fundamental mode.

Cross-module dispersion check (`D = 0`, water depth 200 m) against
`tensorbem.ice.mass_loading_omega`:

| `k` [1/m] | this module [rad/s] | `tensorbem.ice` [rad/s] | rel. error |
|---|---|---|---|
| 0.05 | 0.698900597912 | 0.698900597912 | 0.0e+00 |
| 0.2 | 1.389870992573 | 1.389870992573 | 0.0e+00 |
| 1 | 3.018117209571 | 3.018117209571 | 0.0e+00 |
| 5 | 5.954506022779 | 5.954506022779 | 0.0e+00 |
| 20 | 8.801741047774 | 8.801741047774 | 0.0e+00 |

### 3.3 Wave-excited response (vertical and bending RAO)

| quantity | value |
|---|---|
| sweep | 1200 points, 5.18 .. 388.30 rad/s |
| coupled `omega_1` | 258.8686 rad/s |
| vertical RAO peak | 262.0836 rad/s, 7.4278e-03 m/m |
| peak position error vs `omega_1` | 1.242% |
| bending RAO peak | 262.0832 rad/s, 2.4449e+03 N m/m |
| bending peak error vs `omega_1` | 1.242% |
| fundamental modal peak | 262.0837 rad/s, error vs `omega_1` 1.2420% |
| quasi-static closed form | 5.886274e-04 m/m |
| quasi-static evaluation at `omega = 2.589e-01` rad/s | 5.886275e-04 m/m (error 6.71e-08) |

Damping sweep (structural modal ratio only; the model has no radiation damping):

| `zeta` | peak [rad/s] | peak RAO [m/m] | error vs `omega_1` |
|---|---|---|---|
| 0.005 | 258.8828 | 2.5455e-02 | 0.005% |
| 0.020 | 262.0836 | 7.4278e-03 | 1.242% |
| 0.050 | 262.0857 | 3.3918e-03 | 1.243% |

Local-operator check: the resonance of the plane-wave response at
`k = 3 1/m` sits at 61.083902 rad/s against the
dispersion root 61.083905 rad/s (rel. error 4.69e-08).

### 3.4 Partitioned fluid-structure coupling

| quantity | value |
|---|---|
| sweeps to convergence | 13 |
| converged | True |
| final `omega` residual | 3.670e-07 |
| final pressure/operator residual | 5.314e-07 |
| monolithic residual (per mode) | < 1e-14, < 1e-14, < 1e-14 |

Residual history (relative change of the eigenfrequencies and of the
added-mass operator):

| sweep | `|domega|/omega` | `|dA|/|A|` |
|---|---|---|
| 1 | inf | 1.000e+00 |
| 2 | 4.283e-01 | 3.000e-01 |
| 3 | 7.297e-02 | 9.000e-02 |
| 4 | 1.951e-02 | 2.700e-02 |
| 5 | 5.668e-03 | 8.100e-03 |
| 6 | 1.685e-03 | 2.430e-03 |
| 7 | 5.040e-04 | 7.290e-04 |
| 8 | 1.511e-04 | 2.187e-04 |
| 9 | 4.531e-05 | 6.561e-05 |
| 10 | 1.359e-05 | 1.968e-05 |
| 11 | 4.077e-06 | 5.905e-06 |
| 12 | 1.223e-06 | 1.771e-06 |
| 13 | 3.670e-07 | 5.314e-07 |

## 4. Figures

* `benchmarks/verified/bem_hydroelastic/figures/bem_hydroelastic_kernel.png` -- (a) quadrature error of the
  uniform-mode added mass against the closed form versus panel count,
  (b) BEM/local added-mass ratio versus `k a`.
* `benchmarks/verified/bem_hydroelastic/figures/bem_hydroelastic_rao.png` -- (a) vertical and bending RAO with the
  coupled eigenfrequency marked, (b) damping sweep around the coupled resonance.

## 5. Error statement

| criterion | tolerance | measured | verdict |
|---|---|---|---|
| uniform-mode kernel vs closed form | < 5e-03 | 2.39e-04 | PASS |
| kernel constant: `mu_BEM/mu_local -> 1` | < 3% | 0.29% | PASS |
| wet frequencies vs mesh-independent value | < 3% | 0.21% | PASS |
| vertical RAO peak position vs coupled eigenfrequency | < 2% | 1.24% | PASS |
| bending RAO peak position vs coupled eigenfrequency | < 2% | 1.24% | PASS |
| fundamental modal peak vs `omega_1` | < 2% | 1.24% | PASS |
| quasi-static RAO vs closed form | < 1e-04 | 6.71e-08 | PASS |
| dispersion cross-check vs `tensorbem.ice` | < 1e-12 | 0.0e+00 | PASS |
| partitioned coupling residual | < 1e-06 | 3.67e-07 (omega), 5.31e-07 (pressure) | PASS |

## 6. Limitations

* The hydrodynamic model is the **zero-frequency** one: added mass is real and
  frequency independent and there is **no radiation damping** (an incompressible
  fluid bounded by a rigid baffle radiates no acoustic energy).  The RAO damping
  is structural only, so the peaks are sharper than a physical wetted panel would
  show; section 3.3 gives the peak position for a range of `zeta` instead of
  asserting a damping model.
* Excitation is Froude-Krylov only; the scattered field of a flush panel in an
  infinite baffle is not represented.  A *floating* plate would additionally
  require diffraction (see the caveat in `hydroelastic.infinite_plate_response`).
* Kirchhoff plate: no shear deformation, no rotary inertia, no geometric
  nonlinearity, small deflections only.
* The panel is simply supported; clamped or free edges change the wet/dry ratio
  through the mode shape rather than through the kernel and are not covered.
* Absolute validation against experiment or against an independent hydroelastic
  code is **WITHHELD**: every number above is a discretisation/consistency
  statement for the stated model.

