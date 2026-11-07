# BEM verification anchor: ellipsoid wave resistance against Michell theory

**Case.** Surface-piercing spheroid (submerged half ellipsoid),
`L = 10.0 m`, `B = 1.0 m`,
`T = 0.6 m`
(`L/B = 10`, `T/L = 0.06`),
rho = 1025.0 kg/m^3, g = 9.80665 m/s^2, even keel,
no sinkage or trim.  Displaced volume `3.14159 m^3`
(`Cb = pi/6 = 0.52360`), wetted area
`13.67281 m^2` (2-D quadrature, converged to 1e-10 under
refinement).

**Method.** Two independent evaluations of Michell's thin-ship integral

    R_w = (4 rho g^2 / (pi U^2)) int_0^{pi/2} sec^3(theta) |A|^2 dtheta,
    A = int int y_x exp(k z + i alpha x) dx dz,  k = k_0 sec^2(theta),
    alpha = k_0 sec(theta),  k_0 = g/U^2,

1. **Reference** -- the depth integral is done analytically.  For fixed `x` the
   vertical extent is `z in [-c sqrt(1-(x/a)^2), 0]`, and with
   `z = -c sqrt(1-(x/a)^2) u` the `z` integral collapses onto
   `G(v) = int_0^1 e^{-vu} du/sqrt(1-u^2)`, giving
   `A = -b c int sin(theta) cos(theta) G(k c cos(theta)) e^{i alpha a sin(theta)}
   dtheta`.  `G` is evaluated from `(pi/2)(I_0 - L_0)` below `v = 15` and from its
   Watson expansion above.  The remaining one-dimensional integral is integrated
   with a uniform composite Gauss rule of six cells per oscillation period and
   the `lambda` integral is truncated at `lambda_max = 40.0`
   with `360` Gauss nodes and the truncation residual is
   *measured* against a `lambda_max = 80.0` run rather
   than extrapolated (Michell's spectrum of this hull decays slowly, see Notes).  The `theta` form of the integral is
   evaluated separately and agrees to
   `2.81e-03` relative.

2. **Panel path** -- flat centreplane panels on a cosine-clustered
   `(psi, phi)` grid with `x = a sin psi cos phi`, `z = -c sin psi sin phi`; the
   source density `y_x` is sampled at 2x2 Gauss points inside every panel and the
   panel bilinear map carries the area and the phase.  This is the Rankine
   source-sheet discretisation a thin-ship panel code assembles; the panel grid
   is the only approximation.

**Results.**

| Fr | R_w ref [N] | R_w theta-form [N] | R_w panel [N] | panel error % | GCI % | C_w ref |
|---|---|---|---|---|---|---|
| 0.20 | 75.7616 | 75.9742 | 74.9216 | +1.109 | 2.71 | 0.00276 |
| 0.30 | 266.0728 | 266.1605 | 263.7533 | +0.872 | 1.06 | 0.00430 |
| 0.40 | 469.0760 | 468.1772 | 466.3907 | +0.572 | 0.66 | 0.00427 |

Panel convergence (cosine psi spacing, error measured against the reference):

| Fr | panels (psi x phi) | panel count | R_w [N] | error % |
|---|---|---|---|---|
| 0.20 | 16x128 | 2048 | 72.9887 | -3.660 |
| 0.20 | 32x256 | 8192 | 74.1421 | -2.138 |
| 0.20 | 64x512 | 32768 | 74.9216 | -1.109 |
| 0.30 | 16x128 | 2048 | 256.5425 | -3.582 |
| 0.30 | 32x256 | 8192 | 261.3946 | -1.758 |
| 0.30 | 64x512 | 32768 | 263.7533 | -0.872 |
| 0.40 | 16x128 | 2048 | 457.8389 | -2.396 |
| 0.40 | 32x256 | 8192 | 463.6421 | -1.158 |
| 0.40 | 64x512 | 32768 | 466.3907 | -0.572 |

**Acceptance.**

| criterion | measured | limit | verdict |
|---|---|---|---|
| Fr=0.20 / wave_resistance_error | 0.0110875 | 0.05 | PASS |
| Fr=0.20 / gci | 0.0271154 | 0.03 | PASS |
| Fr=0.20 / reference_two_forms | 0.00280663 | 0.005 | PASS |
| Fr=0.20 / lambda_truncation | 0.00173162 | 0.005 | PASS |
| Fr=0.30 / wave_resistance_error | 0.00871751 | 0.05 | PASS |
| Fr=0.30 / gci | 0.0105757 | 0.03 | PASS |
| Fr=0.30 / reference_two_forms | 0.000329359 | 0.005 | PASS |
| Fr=0.30 / lambda_truncation | 3.01299e-05 | 0.005 | PASS |
| Fr=0.40 / wave_resistance_error | 0.00572479 | 0.05 | PASS |
| Fr=0.40 / gci | 0.00662831 | 0.03 | PASS |
| Fr=0.40 / reference_two_forms | 0.00191619 | 0.005 | PASS |
| Fr=0.40 / lambda_truncation | 4.98161e-05 | 0.005 | PASS |

**Notes and limitations.**

- The panel error is negative (the discretisation under-integrates the sheet)
  and decays at first order in the panel size.  That
  order is set by the two boundary layers of this hull: `y_x ~ 1/cos(psi)`
  diverges on the keel line, and the kernel `exp(k z)` decays over a depth
  `1/k` (down to `1/(k_0 lambda_max^2) ~ 1 mm` at Fr = 0.2).  Cosine clustering
  of the `psi` stations halves the error constant at the same panel count; a
  finer panel grid is what buys accuracy here, not a higher panel order.
- This anchor verifies *Michell theory against its own discretisation*.  It does
  not validate Michell theory against experiment (no experimental data set is
  shipped with this repository) and it does not include viscosity, wave
  breaking, sinkage/trim or finite thickness.  Evidence level for the absolute
  resistance values: WITHHELD.  The `C_w` column is given for continuity with
  the literature, based on the wetted area above.
