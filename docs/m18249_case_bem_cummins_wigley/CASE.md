# CASE-BEM-CUMMINS-WIGLEY

Auto-rendered by `tensorbem.benchmarks.case_bem_cummins_wigley`; every number below is read from `case_result.json` produced by the same run.

## 1. Case setup

| item | value |
| --- | --- |
| hull form | mathematical Wigley hull y = (B/2)(1-(2x/L)^2)(1-(z/T)^2) |
| L / B / T [m] | 3.0 / 0.3 / 0.1875 |
| Cb (analytic) | 0.4444 |
| displaced volume (analytic) [m^3] | 0.075 |
| frequency band [rad/s] | 0.2 .. 12.0 |
| frequency nodes | 41 (log-spaced) |
| meshes | coarse: 12x4 |
| CHIEF irregular-frequency suppression | auto |

Criterion source: tensor_case_pool id=145 (tol_policy <10%).

## 2. Gates

| gate | criterion | threshold | passed |
| --- | --- | --- | --- |
| C1 | geometry precondition (Cb, L/B, B/T, displaced volume) | Cb within 1% of 4/9; volume within 2%; L/B=10; B/T=1.6 | PASS |
| C2 | panel resolution at the top of the band (coarse mesh) | k*sqrt(A_panel,max) <= 1 | FAIL |
| C3 | Kramers-Kronig closure of the added mass (6 DOF) | max over samples and DOF |A_BEM-A_KK|/max|A_BEM| <= 10% | PASS |
| C4 | Cummins round trip B->K->B (heave) | max pointwise relative error <= 2% | PASS |
| C5 | Cummins free decay vs frequency-domain damping ratio (heave) | |zeta_decay - zeta_fd|/zeta_fd <= 5% | PASS |
| C6a | negative control: zero damping gives zero decay (heave) | max|K| <= 1e-12, zeta_decay <= 1e-3, late/early amplitude ratio >= 0.98 | PASS |
| C6b | negative control: corrupted damping shape must fail the KK gate | max over samples and DOF |A_BEM-A_KK|/max|A_BEM| <= 10% | PASS |

## 3. Reference vs got (KK closure, per DOF)

Reference = direct BEM added mass at the same sample frequency; got = KK reconstruction from the damping sweep with a power-law tail. Errors are normalised by `max|A|` of the DOF.

| DOF | max rel. residual | worst sample omega | A_BEM (ref) | A_KK (got) | abs. error | A_inf (closure) | tail exponent p | tail valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| surge | 2.58% | 9.291 | 0.213591 | 0.155665 | 0.0579 | 0.581411 | 0.247 | no |
| sway | 2.70% | 9.291 | 42.4138 | 37.1876 | 5.23 | 77.3058 | -0.302 | no |
| heave | 0.28% | 9.291 | 29.8466 | 29.451 | 0.396 | 50.3771 | 1.45 | no |
| roll | 0.75% | 9.291 | 0.312267 | 0.307611 | 0.00466 | 0.40623 | -0.124 | no |
| pitch | 0.50% | 9.291 | 8.69769 | 8.5296 | 0.168 | 14.6392 | 1.57 | no |
| yaw | 3.40% | 9.291 | 33.3682 | 28.2868 | 5.08 | 58.2088 | -0.842 | no |

### C4: Cummins round trip B->K->B (heave)

Threshold: max pointwise relative error <= 2% -> **PASS**

| quantity | value |
| --- | --- |
| dof | 2 |
| dof_name | heave |
| damping_relative_error_max | 0.000867 |
| added_mass_relative_error_max | 0.1081 |
| kernel_causal | yes |
| kernel_even_parity_relative_error | 0 |
| kernel_tau_max | 25 |
| kernel_dt | 0.01 |

### C5: Cummins free decay vs frequency-domain damping ratio (heave)

Threshold: |zeta_decay - zeta_fd|/zeta_fd <= 5% -> **PASS**

| quantity | value |
| --- | --- |
| dof | 2 |
| dof_name | heave |
| zeta_decay | 0.1335 |
| zeta_frequency_domain | 0.1296 |
| relative_error | 0.02932 |
| natural_frequency_in_band | yes |
| status | decided |
| damped_period_s | 0.81 |
| natural_frequency_rad_s | 7.827 |
| added_mass_at_natural | 28.55 |
| damping_at_natural | 214 |
| restoring_n_per_m | 5989 |
| displaced_mass_kg | 76.88 |
| added_mass_inf_used | 50.38 |
| added_mass_inf_source | KK closure (least-squares offset) |
| n_peaks | 4 |
| peak_amplitude_floor | 0.01 |
| dt | 0.01 |
| duration | 40 |

### C6a: negative control: zero damping gives zero decay (heave)

Threshold: max|K| <= 1e-12, zeta_decay <= 1e-3, late/early amplitude ratio >= 0.98 -> **PASS**

| quantity | value |
| --- | --- |
| kernel_max_abs | 0 |
| zeta_zero_damping | 1.677e-06 |
| amplitude_ratio_late_over_early | 1 |
| n_peaks | 36 |

### C6b: negative control: corrupted damping shape must fail the KK gate

Threshold: max over samples and DOF |A_BEM-A_KK|/max|A_BEM| <= 10% -> **PASS**

| quantity | value |
| --- | --- |
| mesh | coarse |
| worst_dof | surge |
| worst_relative_residual | 0.2053 |
| tail_fit_model | B(s) = C s^-p on the upper quartile of the sweep |
| added_mass_inf_estimator | least-squares closure (one free parameter) |

## 5. Verdict

6 / 7 gates passed (FAILED: C2).

Case tolerance (KK closure) = 10%; coarse-mesh closure and the corrupted damping control are reported alongside so the verdict cannot be read as a self-fulfilling gate.

## 6. Reproduction

```
PYTHONPATH=src python -m tensorbem.benchmarks.case_bem_cummins_wigley --out benchmarks/verified/casepool_bem_cummins_wigley --jobs 1
```

