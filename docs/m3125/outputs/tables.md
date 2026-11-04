### T1. Unit-rescaling test (the decisive measurement)

| path | measured response `sm^a sl^b st^c` | force requires | homogeneous? |
|---|---|---|---|
| kochin | `sm^1 sl^-1 st^2` = kg m^-1 s^2 | `sm sl st^-2` = kg m s^-2 | **NO** |
| pressure | `sm^1 sl^1 st^-2` = kg m s^-2 | `sm sl st^-2` = kg m s^-2 | yes |

unit change used: mass x5, length x2, time x3; Kochin ratio = 22.5 vs required 1.11111 (ratio of ratios = 20.25 = $g^2$-scaled), pressure ratio = 1.11111.

### T2. Geometric-similarity test (why no convergence study could see it)

| path | $R_w$ base [N] | $R_w$ at $s=2$, $U\to U\sqrt2$ [N] | measured ratio | required $s^3$ | rel. error |
|---|---|---|---|---|---|
| kochin | 0.0983411 | 0.786728 | 8.0000000000 | 8.0 | 4.44e-16 |
| pressure | -0.0203003 | -0.162402 | 8.0000000000 | 8.0 | 6.00e-15 |

Both paths pass: geometric similarity is blind to the defect.

### T3. Froude ladder ($\rho=1025$, $g=9.80665$, $L=2$, $B=0.2$, $T=0.125$, $S_{wet}=0.5949$ m$^2$)

| $Fr$ | $U$ [m/s] | $R_w$ Kochin [N] | $R_w$ pressure [N] | pressure/Kochin | $C_w$ Kochin (doc) | $C_w$ pressure |
|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.8857 | 2.0601e-02 | -4.9616e-03 | -0.241 | 8.613e-05 | -2.074e-05 |
| 0.25 | 1.1072 | 2.8774e-02 | -1.1411e-02 | -0.397 | 7.699e-05 | -3.053e-05 |
| 0.30 | 1.3286 | 9.8341e-02 | -2.0300e-02 | -0.206 | 1.827e-04 | -3.772e-05 |
| 0.35 | 1.5500 | 3.7499e-02 | -2.9020e-02 | -0.774 | 5.119e-05 | -3.962e-05 |
| 0.40 | 1.7715 | 2.1059e-01 | -4.1521e-02 | -0.197 | 2.201e-04 | -4.340e-05 |

### T4. Dimensionally forced correction band for $C_w$ (wetted area)

| $Fr$ | $C_w$ as implemented | $g^2/C$, $C=4\pi$ | $C=2\pi$ | $C=\pi$ | $C=4$ | $C=2$ | $C=1$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 8.613e-05 | 6.592e-04 | 1.318e-03 | 2.637e-03 | 2.071e-03 | 4.142e-03 | 8.283e-03 |
| 0.25 | 7.699e-05 | 5.892e-04 | 1.178e-03 | 2.357e-03 | 1.851e-03 | 3.702e-03 | 7.404e-03 |
| 0.30 | 1.827e-04 | 1.398e-03 | 2.797e-03 | 5.594e-03 | 4.393e-03 | 8.787e-03 | 1.757e-02 |
| 0.35 | 5.119e-05 | 3.918e-04 | 7.835e-04 | 1.567e-03 | 1.231e-03 | 2.462e-03 | 4.923e-03 |
| 0.40 | 2.201e-04 | 1.685e-03 | 3.369e-03 | 6.738e-03 | 5.292e-03 | 1.058e-02 | 2.117e-02 |

### T5. Pressure-path mesh ladder ($Fr=0.3$)

| $n_x\times n_z$ | panels | $R_w$ pressure [N] | $R_w$ Kochin [N] | pressure/Kochin |
|---|---:|---:|---:|---:|
| 8$\times$4 | 128 | -4.6375e-02 | 9.8341e-02 | -0.4716 |
| 16$\times$8 | 512 | -2.0300e-02 | 9.8341e-02 | -0.2064 |
| 24$\times$12 | 1152 | -8.0619e-03 | 9.8341e-02 | -0.0820 |
| 32$\times$16 | 2048 | -7.1713e-03 | 9.8341e-02 | -0.0729 |
| 48$\times$24 | 4608 | -5.7314e-03 | 9.8341e-02 | -0.0583 |
