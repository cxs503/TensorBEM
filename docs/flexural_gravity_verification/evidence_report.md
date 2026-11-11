# Flexural-gravity evidence bundle (m14560)

Generated from `run/verify_run.json` (2026-11-11T08:20:25Z, git `b76bfbab11a7`).  Every number below is emitted by `scripts/verify_flexural_gravity_evidence.py`.

## 1. Parameters

| parameter | value |
|---|---|
| h_ice | [0.5, 1.0, 2.0] |
| E | 5000000000.0 |
| nu | 0.334 |
| rho_w | 1025.0 |
| rho_i | 917.0 |
| depth | 200.0 |
| g | 9.80665 |

## 2. Acceptance criteria

| criterion | pass | key number |
|---|---|---|
| E1_dispersion_solver | True | max rel error 1.872e-12 |
| E1b_independent_model | True | max rel error vs FD 1.023e-04 |
| E2_long_wave_limit | True | max rel error 3.733e-07 |
| E3_short_wave_limit | True | max rel error 1.715e-08 |
| E4_energy_conservation | True | max abs(R^2+T^2-1) = 4.441e-16 |
| E5_wavelength_behavior | True | 6 long-wave points shorter |
| E6_wedge_ice | True | 9/9 wedge < abrupt |
| E7_zero_rigidity | True | max rel error 0.000e+00 |
| E8_parameter_sensitivity | True | monotonic eq2=True eq3=True |

## 3. Benchmark: independent finite-difference depth solve

max relative error in $\omega^2$: **1.023e-04** (threshold 1e-03); observed FD order 1.98.

| h_ice | k | closed form omega^2 | FD omega^2 | rel error |
|---|---|---|---|---|
| 0.5 | 0.01 | 9.413837512e-02 | 9.413836616e-02 | 9.526e-08 |
| 0.5 | 0.05 | 4.970876029e-01 | 4.970864653e-01 | 2.289e-06 |
| 0.5 | 0.2 | 1.859922300e+01 | 1.859858706e+01 | 3.419e-05 |
| 0.5 | 1.0 | 3.952352051e+04 | 3.951947841e+04 | 1.023e-04 |
| 0.5 | 5.0 | 5.522146471e+07 | 5.521893914e+07 | 4.574e-05 |
| 0.5 | 20.0 | 1.840057783e+10 | 1.840030397e+10 | 1.488e-05 |
| 1.0 | 0.01 | 9.377416226e-02 | 9.377415336e-02 | 9.485e-08 |
| 1.0 | 0.05 | 6.061993156e-01 | 6.061979579e-01 | 2.240e-06 |
| 1.0 | 0.2 | 1.258568982e+02 | 1.258529215e+02 | 3.160e-05 |
| 1.0 | 1.0 | 2.415009159e+05 | 2.414820482e+05 | 7.813e-05 |
| 1.0 | 5.0 | 2.612437294e+08 | 2.612366637e+08 | 2.705e-05 |
| 1.0 | 20.0 | 7.749812167e+10 | 7.749751444e+10 | 7.835e-06 |
| 2.0 | 0.01 | 9.328264426e-02 | 9.328263549e-02 | 9.405e-08 |
| 2.0 | 0.05 | 1.500002349e+00 | 1.499999128e+00 | 2.148e-06 |
| 2.0 | 0.2 | 8.640690214e+02 | 8.640453167e+02 | 2.743e-05 |
| 2.0 | 1.0 | 1.312307753e+06 | 1.312238110e+06 | 5.307e-05 |
| 2.0 | 5.0 | 1.150036118e+09 | 1.150019002e+09 | 1.488e-05 |
| 2.0 | 20.0 | 3.184195467e+11 | 3.184182653e+11 | 4.024e-06 |

### FD depth-factor convergence

| dz | nodes | g0 FD | g0 exact | rel error |
|---|---|---|---|---|
| 0.05 | 4000 | 0.999093310623 | 1.000000000000 | 9.067e-04 |
| 0.025 | 8000 | 0.999769503785 | 1.000000000000 | 2.305e-04 |
| 0.0125 | 16000 | 0.999941892814 | 1.000000000000 | 5.811e-05 |
| 0.00625 | 32000 | 0.999985412488 | 1.000000000000 | 1.459e-05 |

## 4. External anchor (in-vacuo Euler-Bernoulli plate)

Reference: Euler-Bernoulli in-vacuo plate: omega^2 = D k^4 / (rho_i h) (Timoshenko & Woinowsky-Krieger, Theory of Plates and Shells, 2nd ed., 1959)  
Limit: eq.(3) with k -> infinity; error is O(rho_w / (rho_i h k)).

| h_ice | k | plate omega^2 | eq3 omega^2 | rel error |
|---|---|---|---|---|
| 0.5 | 10.0 | 1.278584315e+09 | 1.044974900e+09 | 1.827e-01 |
| 0.5 | 100.0 | 1.278584315e+13 | 1.250625938e+13 | 2.187e-02 |
| 0.5 | 1000.0 | 1.278584315e+17 | 1.275732351e+17 | 2.231e-03 |
| 1.0 | 10.0 | 5.114337261e+09 | 4.600144462e+09 | 1.005e-01 |
| 1.0 | 100.0 | 5.114337261e+13 | 5.057802393e+13 | 1.105e-02 |
| 1.0 | 1000.0 | 5.114337261e+17 | 5.108626964e+17 | 1.117e-03 |
| 2.0 | 10.0 | 2.045734904e+10 | 1.937453042e+10 | 5.293e-02 |
| 2.0 | 100.0 | 2.045734904e+14 | 2.034365089e+14 | 5.558e-03 |
| 2.0 | 1000.0 | 2.045734904e+18 | 2.044592207e+18 | 5.586e-04 |

## 5. Integration and reproducibility

Integration callers:

- `src/tensorbem/hydroelastic.py:flexural_gravity_omega delegates to tensorbem.flexural_gravity.plate_fluid_omega (plate/fluid main solver)`
- `src/tensorbem/benchmarks/hydroelastic_verification.py:uses the same flexural-gravity dispersion for the hydroelastic benchmark`

Evidence artifacts:

- `docs/flexural_gravity_verification/run/verify_run.json`
- `docs/flexural_gravity_verification/run/pytest_flexural_gravity.log`
- `docs/flexural_gravity_verification/benchmark.json`
- `scripts/verify_flexural_gravity_evidence.py`
- `docs/flexural_gravity_verification/external_anchor.json`

Reproduce:

```bash
python scripts/verify_flexural_gravity_evidence.py && python -m pytest -q tests/test_flexural_gravity.py tests/test_flexural_gravity_mass_loading.py tests/test_flexural_gravity_evidence.py tests/test_hydroelastic.py
```

## 6. Limitations (honest scope)

- The finite-difference check verifies the *depth factor* `k tanh(kH)` and the
  linearised ice/water free-surface condition; it does not independently
  re-derive Kirchhoff plate theory (`D = E h^3/(12(1-nu^2))`), which is a
  modelling assumption anchored to the textbook plate limit above.
- Ice-edge reflection/transmission uses group-velocity admittance matching,
  which conserves energy by construction; a full Wiener-Hopf / four-condition
  edge solution with evanescent modes is still not implemented.
- `hydroelastic.flexural_gravity_omega` now delegates to the shared core, so
  the cross-module comparison is a regression pin, not an independent check.
