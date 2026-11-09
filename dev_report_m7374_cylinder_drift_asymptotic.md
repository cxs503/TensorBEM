# m7374 — cylinder mean drift force: exact two-formula verification and the `(ka)^3` law

**Node**: `p41w3-n01-bem-diffraction-drift-maturity` (TensorBEM). **Branch**:
`agent/m7374`. **Status**: PASS on the second-order drift half of the brief;
Wigley first-order RAO vs experiment declared **not delivered** (see §5).

## Delivered

| file | what |
|---|---|
| `src/tensorbem/cylinder_drift.py` | exact MacCamy-Fuchs reference field (pure-NumPy Bessel, machine-accurate to `1e-11` vs published values) + near-field (wall pressure + waterline) and far-field (reduced Maruo kernel) drift formulas + long-wave closed form + energy balance + Richardson helper |
| `src/tensorbem/benchmarks/cylinder_drift_asymptotic.py` | driver: 8-point `ka` scan, 8-point long-wave scan, energy balance at 9 `(ka, R)`, wall anchor, repository-convention audit, JSON + 2 PNG |
| `tests/test_cylinder_drift_asymptotic.py` | 29 tests, all green |
| `docs/cylinder_drift_asymptotic.md` | paper-style report (6 elements, tables injected from the artifact) |
| `artifacts/cylinder_drift_asymptotic/{study.json, fig_long_wave_law.png, fig_near_far_energy.png}` | evidence |

## Why this node (not a repeat of m5242)

`dev_report_m5242_second_order_drift.md` left one gate as an **honest FAIL**: the
cylinder `(ka)^3` law was not evaluable because the 3-D Maruo far-field route
integrated the un-reduced interference flux, which oscillates in `kR` (spread
~100 %). That FAIL is untouched. This node replaces the *route*, not the verdict:
the exact MF field is used twice (near field, far field), so no radiation radius
is ever introduced.

## Numbers (all from `artifacts/cylinder_drift_asymptotic/study.json`)

| gate | target | measured | pass |
|---|---|---|---|
| near field vs far field, 8 `ka` in `[0.2, 2.0]` | `< 5 %` | `2.8e-15` | yes |
| far-field radiation-radius dependence | `< 3 %` | reduced kernel radius-free; exact-field flux residual `<= 9.6e-15` over `R = 5,10,20` | yes |
| `(ka)^3` slope (fit `ka in [0.02, 0.30]`) | `3 +- 0.2` | `2.977` | yes |
| `(ka)^3` coefficient vs closed form `5 pi^2/16 = 3.08425` | `< 3 %` | `1.2e-4` at `ka = 0.02` | yes |
| Kochin energy residual (fixed, non-absorbing body) | `< 1 %` | `<= 9.6e-15` | yes |
| wall anchor `rho g A^2/2` (fixes the `1/4` and the sign) | exact | `0.0` | yes |
| pytest | `>= 15` | 29 passed | yes |

The closed form `F_x = (5 pi^2/16) rho g A^2 a (ka)^3` is derived in the doc from
`S ~ i kappa(2 cos theta - 1)`, `kappa = pi (ka)^2/4`, and
`Int (2cos theta - 1)^2 (1 - cos theta) dtheta = 10 pi`.

## Findings worth propagating

1. **The waterline term is a leading-order term, not a correction.** At `ka = 1`
   the wall-pressure term is `-4.3e2` and the waterline term `+1.1e3` in the same
   units; dropping it reverses the sign of the drift force.
2. **Repository convention audit (reported, not edited).**
   `tensorbem.maruo_drift.maruo_mean_drift_force`, fed with this module's Kochin
   function in the `free_surface_solver3d` normalisation, returns exactly `2 pi`
   times the calibrated value (ratio `6.2831816` vs `2 pi = 6.2831853`). The two
   modules are each self-consistent under their own `K` normalisation; mixing them
   carries a `2 pi` bias. `maruo_drift.py` is left untouched (its tests pin its own
   convention) and the audit is recorded for whoever unifies them.
3. **Validity band of the asymptotic law.** The exact solution departs from
   `5 pi^2/16` by `3.0 %` at `ka = 0.2` and by a factor 39 at `ka = 2`; the law is
   a `ka <~ 0.1` statement, not a general one.

## Limitations / not delivered

* Deep water only; finite depth needs the `tanh(kh)` depth integral and is out of
  scope here.
* Fixed bottom-mounted cylinder: no forward speed, no radiation modes, no
  second-order potential.
* **Wigley III wave-exciting force and heave/pitch RAO vs towing-tank experiment
  (brief item (a)) is not delivered.** No digitised experimental data set is
  carried in this repository; typing experimental numbers from memory would be
  fabrication, so the comparison is not claimed. The `ka`-scan drift work (brief
  item (b)) and the energy identity (brief item (c)) are delivered.
* `energy_balance` refuses `k R > 35`: the ascending Bessel series used for the
  exact field loses accuracy there. The far-field drift route does not need it.

## Pre-existing failures (not caused here, verified against `HEAD`)

`tests/test_capability_evidence.py::{test_uncapability_baseline_has_no_untested_module_without_a_reason, test_published_block_matches_the_tree}`
and `tests/test_doc_claim_audit.py::{test_ratchet_against_baseline, test_committed_artifacts_are_current}`
fail identically on a clean `HEAD` worktree (baseline drift from the `m6656`
`kcs_hull` row and two unowned benchmark modules). Left untouched.

## Reproduce

```
PYTHONPATH=src python3 -m tensorbem.benchmarks.cylinder_drift_asymptotic \
    --json artifacts/cylinder_drift_asymptotic/study.json \
    --figures artifacts/cylinder_drift_asymptotic
PYTHONPATH=src python3 -m pytest tests/test_cylinder_drift_asymptotic.py -q
```