### Single-panel kernel integral (pure quadrature error)

| d / size | fixed-order error | adaptive error |
|---|---|---|
| 1e-01 | 3.191e-03 | 3.922e-14 |
| 3e-02 | 3.203e-02 | 1.505e-14 |
| 1e-02 | 6.191e-02 | 1.875e-14 |
| 3e-03 | 7.655e-02 | 2.224e-14 |
| 1e-03 | 8.115e-02 | 1.886e-14 |
| 3e-04 | 8.280e-02 | 1.846e-14 |
| 1e-04 | 8.327e-02 | 1.429e-14 |
| 1e-05 | 8.349e-02 | 2.369e-14 |

### Green representation with exact traces

| d / h_max | fixed vs converged | adaptive vs converged | adaptive vs analytic |
|---|---|---|---|
| 1e-01 | 6.363e-07 | 5.791e-15 | 1.894e-02 |
| 3e-02 | 8.744e-06 | 3.455e-15 | 1.707e-02 |
| 1e-02 | 8.019e-04 | 9.536e-15 | 1.564e-02 |
| 3e-03 | 7.155e-04 | 1.095e-14 | 1.476e-02 |
| 1e-03 | 3.406e-04 | 1.127e-14 | 1.439e-02 |
| 3e-04 | 1.264e-04 | 1.149e-14 | 1.423e-02 |
| 1e-04 | 4.846e-05 | 1.115e-14 | 1.416e-02 |
| 1e-05 | 6.091e-06 | 1.160e-14 | 1.413e-02 |

### End-to-end solve, near-surface evaluation

| d / h_max | clearance | field rel. error | velocity rel. error | old API rejects |
|---|---|---|---|---|
| 5.0e-01 | 1.625e-01 | 1.975e-03 | 2.629e-04 | False |
| 1.0e-01 | 3.249e-02 | 2.202e-03 | 2.315e-02 | False |
| 2.0e-02 | 6.498e-03 | 4.796e-03 | 1.169e-01 | True |
| 5.0e-03 | 1.625e-03 | 6.267e-03 | 2.225e-01 | True |
| 1.0e-03 | 3.249e-04 | 6.953e-03 | 3.532e-01 | True |
| 2.0e-04 | 6.498e-05 | 7.158e-03 | 4.863e-01 | True |

### Mesh convergence at fixed clearance ratios (exact traces)

| subdivisions | panels | d / h_max | field error | velocity error |
|---|---|---|---|---|
| 1 | 80 | 1e-02 | 6.314e-02 | 1.962e-01 |
| 1 | 80 | 1e-03 | 5.916e-02 | 5.150e-01 |
| 2 | 320 | 1e-02 | 1.564e-02 | 1.407e-01 |
| 2 | 320 | 1e-03 | 1.439e-02 | 3.198e-01 |
| 3 | 1280 | 1e-02 | 3.854e-03 | 7.954e-02 |
| 3 | 1280 | 1e-03 | 3.520e-03 | 1.719e-01 |

Pressure (near-wall): max |dCp| = 7.515e+00 (3.340e+00 of the Cp scale 2.25) at clearance 9.748e-03 (old API limit 1.625e-02); surface-Cp RMS error 5.475e-02.
Pressure (far): max |dCp| = 1.184e+00 (5.264e-01 of the Cp scale 2.25) at clearance 3.249e-02 (old API limit 1.625e-02); surface-Cp RMS error 5.475e-02.
