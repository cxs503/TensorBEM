# CASE-BEM-CUMMINS-WIGLEY

Auto-rendered by `tensorbem.benchmarks.case_bem_cummins_wigley`; every number below is read from `case_result.json` produced by the same run.

## 1. Case setup

| item | value |
| --- | --- |
| hull form | mathematical Wigley hull y = (B/2)(1-(2x/L)^2)(1-(z/T)^2) |
| L / B / T [m] | 3.0 / 0.3 / 0.1875 |
| Cb (analytic) | 0.4444 |
| displaced volume (analytic) [m^3] | 0.075 |
| frequency band [rad/s] | 0.15 .. 7.999999999999999 |
| frequency nodes | 13 (log-spaced) |
| meshes | coarse: 12x4 |
| CHIEF irregular-frequency suppression | auto |

Criterion source: tensor_case_pool id=145 (tol_policy <10%).

![fig_kk_closure.png](fig_kk_closure.png)

## 2. Gates

| gate | criterion | threshold | passed |
| --- | --- | --- | --- |
| C1 | geometry precondition (Cb, L/B, B/T, displaced volume) | Cb within 1% of 4/9; volume within 2%; L/B=10; B/T=1.6 | PASS |
| C2 | panel resolution at the top of the band (coarse mesh) | k*sqrt(A_panel,max) <= 1 | PASS |
| C3 | Kramers-Kronig closure of the added mass (6 DOF) | max over samples and DOF |A_BEM-A_KK|/max|A_BEM| <= 10% | FAIL |
| C4 | Cummins round trip B->K->B (heave) | max pointwise relative error <= 2% | PASS |
| C5 | Cummins free decay vs frequency-domain damping ratio (heave) | |zeta_decay - zeta_fd|/zeta_fd <= 5% | FAIL |
| C6a | negative control: zero damping gives zero decay (heave) | max|K| <= 1e-12, zeta_decay <= 1e-3, late/early amplitude ratio >= 0.98 | PASS |
| C6b | negative control: corrupted damping shape must fail the KK gate | max over samples and DOF |A_BEM-A_KK|/max|A_BEM| <= 10% | PASS |

## 3. Reference vs got (KK closure, per DOF)

Reference = direct BEM added mass at the same sample frequency; got = KK reconstruction from the damping sweep with a power-law tail. Errors are normalised by `max|A|` of the DOF.

| DOF | max rel. residual | worst sample omega | A_BEM (ref) | A_KK (got) | abs. error | A_inf (closure) | tail exponent p | tail valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| surge | 10.08% | 6.778 | 0.591519 | 0.365698 | 0.226 | 0.808237 | -0.392 | no |
| sway | 16.11% | 6.778 | 115.815 | 84.7096 | 31.1 | 100.539 | -2.4 | no |
| heave | 3.53% | 6.778 | 31.6283 | 26.68 | 4.95 | 55.7518 | 0.66 | no |
| roll | 9.66% | 6.778 | 0.487415 | 0.428906 | 0.0585 | 0.447383 | -2.92 | no |
| pitch | 6.16% | 6.778 | 10.8414 | 8.76772 | 2.07 | 16.2951 | 0.142 | no |
| yaw | 14.52% | 6.778 | 99.1775 | 79.0555 | 20.1 | 77.3313 | -4.13 | no |

### C4: Cummins round trip B->K->B (heave)

Threshold: max pointwise relative error <= 2% -> **PASS**

| quantity | value |
| --- | --- |
| dof | 2 |
| dof_name | heave |
| damping_relative_error_max | 0.002074 |
| added_mass_relative_error_max | 0.226 |
| kernel_causal | yes |
| kernel_even_parity_relative_error | 0 |
| kernel_tau_max | 25 |
| kernel_dt | 0.01 |

### C5: Cummins free decay vs frequency-domain damping ratio (heave)

Threshold: |zeta_decay - zeta_fd|/zeta_fd <= 5% -> **FAIL**

| quantity | value |
| --- | --- |
| dof | 2 |
| dof_name | heave |
| zeta_decay | 0.00178 |
| zeta_frequency_domain | n/a |
| relative_error | n/a |
| damped_period_s | 0.7712 |
| natural_frequency_rad_s | 8.147 |
| added_mass_at_natural | n/a |
| damping_at_natural | n/a |
| restoring_n_per_m | 5989 |
| displaced_mass_kg | 76.88 |
| added_mass_inf_used | 55.75 |
| added_mass_inf_source | KK closure (least-squares offset) |
| n_peaks | 51 |
| dt | 0.01 |
| duration | 40 |

### C6a: negative control: zero damping gives zero decay (heave)

Threshold: max|K| <= 1e-12, zeta_decay <= 1e-3, late/early amplitude ratio >= 0.98 -> **PASS**

| quantity | value |
| --- | --- |
| kernel_max_abs | 0 |
| zeta_zero_damping | -9.916e-07 |
| amplitude_ratio_late_over_early | 1 |
| n_peaks | 35 |

### C6b: negative control: corrupted damping shape must fail the KK gate

Threshold: max over samples and DOF |A_BEM-A_KK|/max|A_BEM| <= 10% -> **PASS**

| quantity | value |
| --- | --- |
| mesh | coarse |
| worst_dof | surge |
| worst_relative_residual | 0.269 |
| tail_fit_model | B(s) = C s^-p on the upper quartile of the sweep |
| added_mass_inf_estimator | least-squares closure (one free parameter) |

## 5. Verdict

5 / 7 gates passed (FAILED: C3, C5).

Case tolerance (KK closure) = 10%; coarse-mesh closure and the corrupted damping control are reported alongside so the verdict cannot be read as a self-fulfilling gate.

## 6. Reproduction

```
PYTHONPATH=src python -m tensorbem.benchmarks.case_bem_cummins_wigley --out benchmarks/verified/casepool_bem_cummins_wigley --jobs 1
```

