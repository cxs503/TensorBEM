### Generated tables

Domain: annulus $r_{\mathrm{in}} = 0.5$, $r_{\mathrm{out}} = 1$; ladder $N = 16, 32, 64, 128$ panels per loop; 18 interior probes per case. All values written by `python -m tensorbem.benchmarks.multicomponent_annulus`.

### Panel refinement

| case | N=16 | N=32 | N=64 | N=128 | order (last pair) |
|---|---|---|---|---|---|
| log flux $L_2$ | 9.673e-03 | 2.641e-03 | 6.896e-04 | 1.761e-04 | 1.97 |
| log probe RMSE | 3.236e-03 | 7.987e-04 | 1.999e-04 | 5.008e-05 | 2.00 |
| power1 flux $L_2$ | 3.745e-02 | 9.584e-03 | 2.414e-03 | 6.049e-04 | 2.00 |
| power1 probe RMSE | 1.263e-02 | 2.289e-03 | 2.010e-04 | 5.417e-05 | 1.89 |
| power2 flux $L_2$ | 3.651e-02 | 8.755e-03 | 2.129e-03 | 5.240e-04 | 2.02 |
| power2 probe RMSE | 4.618e-02 | 5.208e-03 | 1.168e-03 | 2.703e-04 | 2.11 |

### Per-loop flux identity (log case, finest mesh)

| loop | panel solution $\oint q\,ds$ | analytic | relative to $2\pi A$ |
|---|---:|---:|---:|
| outer | 6.283673 | 6.284447 | 1.232e-04 |
| hole | -6.283207 | -6.284447 | 1.974e-04 |
| total | 4.659e-04 | 0.000e+00 | net flux / $\oint|q|\,ds$ = 3.71e-05 |

### Mesh conditioning and residuals (finest mesh per case)

| case | condition number | relative residual $L_2$ |
|---|---:|---:|
| log | 271.06 | 4.81e-16 |
| power1 | 271.06 | 4.69e-16 |
| power2 | 271.06 | 6.18e-16 |

### Negative control: hole orientation

| configuration | flux $L_2$ error | condition number |
|---|---:|---:|
| hole reversed (correct) | 6.896e-04 | 135.71 |
| hole not reversed (control) | 1.413e+00 | 135.71 |

Error ratio control/correct = **2049x** at N = 64 panels per loop. The reversed mesh still solves its own system to round-off (relative residual 3.7e-16), so the wrong answer is not detectable from the residual alone. The conditioning does not flag it either: the single-layer matrix $S$ is orientation independent, only the double-layer $H$ changes sign on the hole, so $\kappa = 135.71$ in both cases. That is why the orientation check is enforced in `MultiComponentMesh` instead of being left to the caller.

### Pointwise field error map (log case)

$N$ = 64 panels per loop, 908 grid points inside the fluid, max $|u_{panel} - u_{exact}|$ = 3.632e-04, RMSE = 2.030e-04.

