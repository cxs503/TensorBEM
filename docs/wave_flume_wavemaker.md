# CASE-WAVE-FLUME: linear wavemaker transfer functions

**Anchor**: `CASE-WAVE-FLUME` (算例池·规则波生成) — acceptance `<10%` against the
closed-form linear-wavemaker reference.

**Reference**: Biesel & Suquet (1951) / Havelock (1929) piston and flap transfer
functions, re-derived in `src/tensorbem/wavemaker.py` and *also* checked against the
long-wave mass balance (`H/S -> kh` and `H/S -> kh/2`) in the tests.

## 1. Case

Constant-depth flume, `h = 1.0 m`, `g = 9.80665`.
The end wall is a vertical face panelled from the bed to the waterline and driven
with a prescribed normal velocity; the free surface condition, the bed and the
radiation condition are carried by the finite-depth kernel.

| wavemaker | wall velocity `u(z)` | `H/S` |
|---|---|---|
| piston | `U0` | `4 sinh^2(kh) / (2kh + sinh 2kh)` |
| flap (bed hinge) | `U0 (z+h)/h` | `4 sinh(kh)[sinh(kh)+(1-cosh(kh))/(kh)] / (2kh+sinh 2kh)` |

Dimensionless depths: `kh=0.5`, `kh=1.0`, `kh=2.0`, `kh=3.0`; panel counts: `8`, `16`, `32`, `64`.

## 2. How it was done

1. Panelize the wall (`tensorbem.wavemaker.vertical_wavemaker`) and assemble the
   finite-depth single-layer normal-derivative matrix: exact Rankine row integrals
   plus Gauss quadrature of the smooth part of the kernel.
2. Solve `(A + I/2) sigma = du/dn` for the prescribed stroke velocity.
3. Read the far-field elevation off two independent quantities: the Kochin
   integral over the wall (route 1) and direct quadrature of the kernel two
   wavelengths downstream (route 2).
4. Compare with the closed-form transfer function, and with the exact
   eigenfunction field (propagating + evanescent modes) at five stations.

## 3. Results — transfer function

| wavemaker | kh | panels | BEM `H/S` | analytic `H/S` | rel. error | route disagreement |
|---|---|---|---|---|---|---|
| piston | 0.5 | 8 | 0.499257 | 0.499338 | 0.0163% | 0.0163% |
| piston | 0.5 | 16 | 0.499318 | 0.499338 | 0.0041% | 0.0041% |
| piston | 0.5 | 32 | 0.499333 | 0.499338 | 0.0010% | 0.0010% |
| piston | 0.5 | 64 | 0.499337 | 0.499338 | 0.0003% | 0.0003% |
| piston | 1 | 8 | 0.981150 | 0.981789 | 0.0651% | 0.0651% |
| piston | 1 | 16 | 0.981630 | 0.981789 | 0.0163% | 0.0163% |
| piston | 1 | 32 | 0.981749 | 0.981789 | 0.0041% | 0.0041% |
| piston | 1 | 64 | 0.981779 | 0.981789 | 0.0010% | 0.0010% |
| piston | 2 | 8 | 1.677208 | 1.681579 | 0.2599% | 0.2599% |
| piston | 2 | 16 | 1.680485 | 1.681579 | 0.0651% | 0.0651% |
| piston | 2 | 32 | 1.681305 | 1.681579 | 0.0163% | 0.0163% |
| piston | 2 | 64 | 1.681511 | 1.681579 | 0.0041% | 0.0041% |
| piston | 3 | 8 | 1.921346 | 1.932623 | 0.5835% | 0.5836% |
| piston | 3 | 16 | 1.929795 | 1.932623 | 0.1463% | 0.1464% |
| piston | 3 | 32 | 1.931916 | 1.932623 | 0.0366% | 0.0368% |
| piston | 3 | 64 | 1.932446 | 1.932623 | 0.0092% | 0.0099% |
| flap | 0.5 | 8 | 0.254623 | 0.254744 | 0.0475% | 0.0163% |
| flap | 0.5 | 16 | 0.254713 | 0.254744 | 0.0119% | 0.0041% |
| flap | 0.5 | 32 | 0.254736 | 0.254744 | 0.0030% | 0.0010% |
| flap | 0.5 | 64 | 0.254742 | 0.254744 | 0.0007% | 0.0003% |
| flap | 1 | 8 | 0.527154 | 0.528088 | 0.1768% | 0.0651% |
| flap | 1 | 16 | 0.527854 | 0.528088 | 0.0442% | 0.0163% |
| flap | 1 | 32 | 0.528029 | 0.528088 | 0.0111% | 0.0041% |
| flap | 1 | 64 | 0.528073 | 0.528088 | 0.0028% | 0.0010% |
| flap | 2 | 8 | 1.035209 | 1.041239 | 0.5791% | 0.2599% |
| flap | 2 | 16 | 1.039728 | 1.041239 | 0.1451% | 0.0651% |
| flap | 2 | 32 | 1.040861 | 1.041239 | 0.0363% | 0.0163% |
| flap | 2 | 64 | 1.041144 | 1.041239 | 0.0091% | 0.0041% |
| flap | 3 | 8 | 1.334867 | 1.349520 | 1.0858% | 0.5835% |
| flap | 3 | 16 | 1.345840 | 1.349520 | 0.2727% | 0.1463% |
| flap | 3 | 32 | 1.348599 | 1.349520 | 0.0682% | 0.0366% |
| flap | 3 | 64 | 1.349289 | 1.349520 | 0.0171% | 0.0092% |

Worst transfer error over every mesh: **1.0858%**; on the finest mesh: **0.0171%**. Worst disagreement between the two far-field routes: **0.5836%**.

## 4. Results — near field versus the exact eigenfunction series

| wavemaker | kh | station (`x/lambda`) | `\|eta\|` BEM | `\|eta\|` exact | rel. error |
|---|---|---|---|---|---|
| piston | 0.5 | 0.25 | 0.249669 | 0.249669 | 0.0000% |
| piston | 0.5 | 0.5 | 0.249669 | 0.249669 | 0.0000% |
| piston | 0.5 | 1 | 0.249669 | 0.249669 | 0.0000% |
| piston | 0.5 | 2 | 0.249669 | 0.249669 | 0.0000% |
| piston | 0.5 | 4 | 0.249669 | 0.249669 | 0.0000% |
| piston | 1 | 0.25 | 0.491162 | 0.491162 | 0.0000% |
| piston | 1 | 0.5 | 0.490895 | 0.490895 | 0.0000% |
| piston | 1 | 1 | 0.490895 | 0.490895 | 0.0000% |
| piston | 1 | 2 | 0.490895 | 0.490895 | 0.0000% |
| piston | 1 | 4 | 0.490895 | 0.490895 | 0.0000% |
| piston | 2 | 0.25 | 0.867819 | 0.867819 | 0.0000% |
| piston | 2 | 0.5 | 0.840798 | 0.840798 | 0.0000% |
| piston | 2 | 1 | 0.840789 | 0.840789 | 0.0000% |
| piston | 2 | 2 | 0.840789 | 0.840789 | 0.0000% |
| piston | 2 | 4 | 0.840789 | 0.840789 | 0.0000% |
| piston | 3 | 0.25 | 1.085925 | 1.085925 | 0.0000% |
| piston | 3 | 0.5 | 0.967025 | 0.967025 | 0.0000% |
| piston | 3 | 1 | 0.966319 | 0.966319 | 0.0000% |
| piston | 3 | 2 | 0.966312 | 0.966312 | 0.0000% |
| piston | 3 | 4 | 0.966312 | 0.966312 | 0.0000% |
| flap | 0.5 | 0.25 | 0.127370 | 0.127371 | 0.0005% |
| flap | 0.5 | 0.5 | 0.127371 | 0.127372 | 0.0005% |
| flap | 0.5 | 1 | 0.127371 | 0.127372 | 0.0005% |
| flap | 0.5 | 2 | 0.127371 | 0.127372 | 0.0005% |
| flap | 0.5 | 4 | 0.127371 | 0.127372 | 0.0005% |
| flap | 1 | 0.25 | 0.263593 | 0.263598 | 0.0017% |
| flap | 1 | 0.5 | 0.264039 | 0.264044 | 0.0017% |
| flap | 1 | 1 | 0.264039 | 0.264044 | 0.0017% |
| flap | 1 | 2 | 0.264039 | 0.264044 | 0.0017% |
| flap | 1 | 4 | 0.264039 | 0.264044 | 0.0017% |
| flap | 2 | 0.25 | 0.516037 | 0.516059 | 0.0043% |
| flap | 2 | 0.5 | 0.520594 | 0.520620 | 0.0050% |
| flap | 2 | 1 | 0.520593 | 0.520619 | 0.0050% |
| flap | 2 | 2 | 0.520593 | 0.520619 | 0.0050% |
| flap | 2 | 4 | 0.520593 | 0.520619 | 0.0050% |
| flap | 3 | 0.25 | 0.688634 | 0.688677 | 0.0062% |
| flap | 3 | 0.5 | 0.674718 | 0.674771 | 0.0079% |
| flap | 3 | 1 | 0.674707 | 0.674760 | 0.0079% |
| flap | 3 | 2 | 0.674706 | 0.674760 | 0.0079% |
| flap | 3 | 4 | 0.674706 | 0.674760 | 0.0079% |

Worst near-field error: **0.0079%**.

## 5. Scope of the operator — what this case does and does not test

For a vertical full-depth wall every cross-panel entry of the normal-derivative
matrix vanishes: all panels are co-planar, so the kernels depend on `|dx|` only and
their `x`-derivative is zero. The system is then `(1/2) I` to round-off and the panel
solution *is* the Havelock projection — so this case tests the kernel quadrature and
the propagating/evanescent decomposition, but **not** the integral-equation solve.

| face inclination (deg) | condition number | max off-diagonal | `H/S` | `H/S` / piston |
|---|---|---|---|---|
| 0 | 1.0000 | 0.000e+00 | 0.981779 | 1.000000 |
| 0.001 | 1.0000 | 3.318e-06 | 0.981780 | 0.999991 |
| 5 | 1.1401 | 1.149e-02 | 0.987283 | 1.005596 |
| 10 | 1.2994 | 2.304e-02 | 0.999767 | 1.018311 |
| 20 | 1.6912 | 4.651e-02 | 1.048268 | 1.067712 |
| 30 | 2.2194 | 7.077e-02 | 1.136129 | 1.157202 |

The inclined face is *not* co-planar, the operator becomes non-degenerate, and as the
inclination goes to zero the computed transfer returns to the analytic piston value —
which is the quantitative anchor for the non-degenerate solve (no closed form exists
for the sloped face itself).

## 6. Results — evanescent decay (flume attenuation)

| wavemaker | `x/lambda` | `x/h` | `\|eta\|` | deviation from flat |
|---|---|---|---|---|
| piston | 0.25 | 1.5708 | 0.491162 | 0.0544% |
| piston | 0.5 | 3.14159 | 0.490895 | 0.0000% |
| piston | 1 | 6.28319 | 0.490895 | 0.0000% |
| piston | 2 | 12.5664 | 0.490895 | 0.0000% |
| piston | 3 | 18.8496 | 0.490895 | 0.0000% |
| piston | 4 | 25.1327 | 0.490895 | 0.0000% |
| flap | 0.25 | 1.5708 | 0.263593 | 0.1688% |
| flap | 0.5 | 3.14159 | 0.264039 | 0.0000% |
| flap | 1 | 6.28319 | 0.264039 | 0.0000% |
| flap | 2 | 12.5664 | 0.264039 | 0.0000% |
| flap | 3 | 18.8496 | 0.264039 | 0.0000% |
| flap | 4 | 25.1327 | 0.264039 | 0.0000% |

Past one wavelength the elevation modulus is flat to round-off, i.e. only the
propagating mode survives — the frequency-domain statement of the flume's
wave-height attenuation being a wavemaker/absorption property, not propagation loss.

## 7. Error judgement

- Transfer function versus Biesel-Havelock: worst **1.0858%**, finest mesh **0.0171%** — **PASS** at the 10% anchor tolerance and **PASS** at 3%.
- Near field versus the exact eigenfunction series: worst **0.0079%** — **PASS** at 1%.
- Two far-field routes: worst **0.5836%**.

## 8. Caveats

- Linear theory only: no steepness limit, no breaking, no wavemaker backlash.
- The wall meets the waterline and the bed, which the closed-body solvers in
  `tensorbem.finite_depth_solver` refuse by construction; this module assembles the
  same two operator pieces for the open curve and proves equality with those
  routines on a submerged closed body in `tests/test_wavemaker_transfer.py`.
- Gauss nodes stay strictly inside each panel, so no kernel call ever lands on the
  waterline where the image term is `log`-singular; the invariant is asserted, not
  assumed.
