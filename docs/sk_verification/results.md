# SK iteration verification (`tensorbem.sk_fit`)

Rendered programmatically by `python -m tensorbem.benchmarks.sk_fit_verification`.

## 1. Fit accuracy: SK iteration vs the pre-fix fixed-pole solve

| case | n_poles | method | max rel err B | rms rel err B | n_iter | converged |
|---|---|---|---|---|---|---|
| exact_model_class | 3 | fixed_pole | 4.0434e-01 | 1.6726e-01 | 1 | True |
| exact_model_class | 3 | sk | 3.8604e-02 | 1.7900e-02 | 4 | False |
| hulme_hemisphere | 12 | fixed_pole | 1.6836e-01 | 3.2872e-02 | 1 | True |
| hulme_hemisphere | 12 | sk | 1.5471e-02 | 3.1799e-03 | 4 | False |

## 2. Memory-kernel consequences

Case A (synthetic, exact model class) -- closed-form kernel `sum c_k exp(-a_k t)` against the exact piecewise-linear quadrature of the damping table (tau <= 8.001 s, 382 samples):

* analytic identity, max rel err = 3.6939e-01
* fixed-pole fit kernel, max rel err = 7.7296e-02
* SK fit kernel, max rel err = 1.3499e-02

Case B (Hulme hemisphere heave) -- closed-form kernel of the fitted model against the quadrature kernel of the table:

* fixed-pole fit, max rel err = 1.1897e+02
* SK fit, max rel err = 5.4403e-03
