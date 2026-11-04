# Kramers-Kronig causality audit of the Ursell cylinder solutions (m2913)

**Verdict (honest, and partly negative)**: the transform itself is exact -- it reproduces causal pairs built from exponentials to 5e-5 -- but on the finite frequency band that a panel sweep can actually cover, the residual of the *analytical* reference is already of the same order as the panel residuals, so the audit **cannot resolve** the panel discretisation error (2-5 %) at the few-percent level. The panel solutions are not distinguishable from causality by this test on this band; what the audit does establish is where the limitation lives (the infinite-frequency added mass and the two truncated bands, not the transform), and that the mechanism is the one the theory predicts.

## 1. Case

Ursell's (1949) submerged circular cylinder in deep water, heave:

| quantity | value |
|---|---|
| radius $a$ | 1.0 m |
| axis submergence $f$ | 2.0 m |
| density $\rho$ | 1025.0 kg/m³ |
| gravity $g$ | 9.80665 m/s² |
| frequency band | $ka \in [0.05, 3.0]$, 32 log-spaced points |
| quadrature nodes per panel | 24 |

The reference is the multipole closed form of `tensorbem.analytic.ursell_cylinder_added_mass` (a truncated infinite system of elementary functions plus an exponential integral; it shares no code with the panel assembly).

## 2. Method

The radiation force of a body in harmonic motion is the transform of **one** causal memory kernel, so the added mass and damping of a frequency sweep are not independent: they satisfy the Kramers-Kronig relations

$$A(\omega) - A_\infty = \frac{2}{\pi}\,\mathrm{PV}\!\int_0^\infty \frac{B(\sigma)}{\sigma^2-\omega^2}\,d\sigma \qquad (1)$$

$$B(\omega) = -\frac{2}{\pi}\,\mathrm{PV}\!\int_0^\infty \frac{\sigma^2\,[A(\sigma)-A_\infty]}{\sigma^2-\omega^2}\,d\sigma \qquad (2)$$

implemented in the new module `tensorbem.causality`. The sweep is interpolated linearly and both integrals are evaluated **exactly** for that interpolant, interval by interval; the pole is handled analytically (the two logarithmic parts of the interval integral are individually singular and their sum is finite). Above the last node the analytic tail of the $1/\sigma^2$ asymptote is added; below the first node the tabulated value is held constant.

The tool was calibrated on coefficient pairs whose causality is exact before being applied to a solver: for a memory kernel that is a sum of exponentials the pair $B=\sum_j c_j a_j/(a_j^2+\omega^2)$, $A-A_\infty=-\sum_j c_j/(a_j^2+\omega^2)$ satisfies (1) and (2) identically, and `tests/test_causality.py` locks that in together with a closed-form check of the principal-value kernel $\mathrm{PV}\int_0^\infty h_a/(\sigma^2-\omega^2)d\sigma = -(\pi/2a)\,h_a(\omega)$ and a negative control (a 5 % corruption of $B$ is detected).

The analytical reference is run through the *same* transform on the *same* grid, so its residual is the transform's own floor (piecewise-linear interpolation plus the two truncated bands); panel residuals are read against that floor. $A_\infty$ is taken from the analytical high-frequency limit for every solution ($A_\infty = 2845.737$ kg/m, extrapolated from $ka = 8\ldots30$ with a $1/(ka)$ model, residual 3.41e-01 kg/m; that is 0.8837 times the displaced mass per span $\rho\pi a^2 = 3220.132$ kg/m).

## 3. Results

| solution | max error vs Ursell | KK damping residual | KK added-mass residual | KK damping residual (fitted $A_\infty$) | residual / reference floor |
|---|---|---|---|---|---|
| analytic multipole (reference) | — | 1.90e-01 | 9.71e-03 | 3.59e-01 | 1.0e+00 |
| indirect panels = 32 | 5.07e-02 | 1.83e-01 | 3.48e-02 | 3.59e-01 | 1.0e+00 |
| indirect panels = 64 | 2.82e-02 | 1.62e-01 | 2.32e-02 | 3.59e-01 | 8.0e-01 |
| indirect panels = 96 | 1.95e-02 | 1.71e-01 | 1.89e-02 | 3.59e-01 | 9.0e-01 |

Transform floor (analytical pair, same grid): damping 1.90e-01, added mass 9.71e-03.

Three-dimensional sanity check on the semi-analytical Hulme (1982) floating hemisphere ($ka \in [0.1, 4.0]$, 40 points, coefficients from the spherical-harmonic expansion, no panels):

| mode | KK damping residual | KK added-mass residual |
|---|---|---|
| heave | 1.35e-01 | 1.36e-01 |
| surge | 1.16e+00 | 4.64e-01 |

![Kramers-Kronig audit](docs/m2913/figures/kkr_causality.png)

*Left/centre: added mass and damping of the analytical reference and of the panel solutions, with the extrapolated infinite-frequency added mass. Right: relative Kramers-Kronig residual in the damping versus frequency; the black dashed curve is the transform's own floor on the same grid.*

## 4. Notes, limits of the check

1. **The check is blind to consistent errors.** A coarse mesh produces an added mass and a damping that are wrong by the same amount in the same direction, and the residual stays near the discretisation level. Causality is a constraint on the *pair*, not a substitute for an analytical reference.
2. **$A_\infty$ is an input, not an output.** Eq. (2) needs the infinite-frequency added mass. Fitting it from the upper tail of a truncated sweep biases the damping prediction (the fitted column in the table); the $1/(ka)$ tail of the submerged cylinder makes that bias of order one per cent, which is why the extrapolated analytical limit is used for every case and the fitted variant is reported separately.
3. **The sample frequency must not sit on a tabulated node**; with the pole exactly at a node the piecewise-linear model is kinked there and the cancellation of the two logarithmic singularities is incomplete. The module raises `ValueError` instead of returning that number.
4. **Low frequencies are the weak spot.** Below the first tabulated frequency the integrand is closed by holding the value constant, which is only first-order accurate; samples within the first decade of the sweep are therefore excluded by default.
5. **Zero speed, linear theory, single body** -- as the whole repository. The relations themselves are general.

## 5. Error judgement

- Transform floor (analytical causal pair): max residual in $B$ 1.90e-01, in $A$ 9.71e-03.
- Panel solution with 32 panels: causality residual in $B$ 1.83e-01 against a mesh error of 5.07e-02 -- is it below 3 %? **no**.
- Panel solution with 64 panels: causality residual in $B$ 1.62e-01 against a mesh error of 2.82e-02 -- is it below 3 %? **no**.
- Panel solution with 96 panels: causality residual in $B$ 1.71e-01 against a mesh error of 1.95e-02 -- is it below 3 %? **no**.

No experimental data enters this study, and none is needed: the criterion is structural (the two coefficients must come from the same causal response), not a comparison with a measurement. The 3 % yardstick is applied to the residuals above and the numbers, not the adjectives, decide.

## 6. Reproduction

```bash
PYTHONPATH=src python3 -m tensorbem.benchmarks.kkr_causality \
    --points 32 --ka-min 0.05 --ka-max 3.0 \
    --panels 32 64 96 --output docs/m2913/outputs/kkr_causality/sweep.json
PYTHONPATH=src python3 docs/m2913/kkr_causality_report.py
PYTHONPATH=src python3 -m pytest tests/test_causality.py -q
```
