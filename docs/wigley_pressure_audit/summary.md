| quantity | value |
|---|---|
| hull | Wigley $L=2.0$, $B=0.2$, $T=0.125$ m |
| density / gravity | $\rho=1025.0$ kg/m3, $g=9.80665$ m/s2 |
| reference | Kochin far-field path (verified to 2.6e-5 vs converged quadrature) |

### Cut-off sensitivity ($Fr=0.30$, mesh [16, 8])

| $k_{max}/k_0$ | $R_W$ pressure [N] | ratio to Kochin |
|---:|---:|---:|
| 5 | -2.0281e-02 | -0.2062 |
| 10 | -3.6925e-02 | -0.3755 |
| 20 | -1.5807e-02 | -0.1607 |
| 40 | -2.1258e-02 | -0.2162 |
| 80 | -4.2790e-02 | -0.4351 |
| 160 | -5.2744e-02 | -0.5363 |

Kochin reference $R_W=9.8341e-02$ N; total spread over the cut-off sweep 37.6 % of the Kochin value.

### Quadrature-node convergence at the legacy cut-off ($Fr=0.30$)

| nodes | $R_W$ pressure [N] | ratio to Kochin |
|---:|---:|---:|
| 32 | -1.1652e-02 | -0.1185 |
| 64 | -4.3751e-02 | -0.4449 |
| 128 | -2.0300e-02 | -0.2064 |
| 256 | -2.0300e-02 | -0.2064 |
| 512 | -2.0300e-02 | -0.2064 |

The sampled integral is *converged* (identical to seven digits from 128 nodes on): the defect is not quadrature error, it is the integrand.

### Mesh ladder ($Fr=0.30$)

| $n_x\times n_z$ | panels | $h_{mean}$ [m] | $R_W$ pressure [N] | ratio | rel. difference to Kochin |
|---|---:|---:|---:|---:|---:|
| 8x4 | 128 | 0.1805 | -4.6375e-02 | -0.4716 | 1.472 |
| 16x8 | 512 | 0.0903 | -2.0300e-02 | -0.2064 | 1.206 |
| 32x16 | 2048 | 0.0451 | -7.1713e-03 | -0.0729 | 1.073 |
| 64x32 | 8192 | 0.0226 | -2.7121e-03 | -0.0276 | 1.028 |

Refining the mesh drives the pressure path to $R_W\to 0$, i.e. the relative difference tends to 1, not to 0.

### Froude sweep (mesh [16, 8])

| $Fr$ | $R_W$ Kochin [N] | $R_W$ pressure [N] | ratio | rel. difference |
|---:|---:|---:|---:|---:|
| 0.20 | 2.0601e-02 | -4.9616e-03 | -0.2408 | 1.241 |
| 0.25 | 2.8774e-02 | -1.1411e-02 | -0.3966 | 1.397 |
| 0.30 | 9.8341e-02 | -2.0300e-02 | -0.2064 | 1.206 |
| 0.35 | 3.7499e-02 | -2.9020e-02 | -0.7739 | 1.774 |
| 0.40 | 2.1059e-01 | -4.1521e-02 | -0.1972 | 1.197 |

Ratio spans -0.7739 ... -0.1972 (spread 0.5767): no single sign or constant factor maps one path onto the other.

### Free-surface residual of the legacy field

| $Fr$ | residual, standard operator | residual, documented operator | nodes on dispersion curve | min $|\lambda-1|$ |
|---:|---:|---:|---:|---:|
| 0.20 | 0.753 | 1.000 | 0 | 1.03e-03 |
| 0.25 | 0.832 | 1.000 | 0 | 1.67e-03 |
| 0.30 | 0.880 | 1.000 | 0 | 2.44e-03 |
| 0.35 | 0.909 | 1.000 | 0 | 3.36e-03 |
| 0.40 | 0.930 | 1.000 | 0 | 4.41e-03 |

### Machine verdict

| test | result |
|---|---|
| `field_off_dispersion_curve` | True |
| `is_formulation_error_not_thin_ship_limit` | True |
| `no_constant_factor` | True |
| `pressure_path_cutoff_dependent` | True |
| `pressure_path_mesh_nonconvergent` | True |
| `pressure_path_negative_everywhere` | True |
| `quadrature_nodes_on_dispersion_curve` | 0 |
