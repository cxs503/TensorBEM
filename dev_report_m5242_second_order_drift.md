# m5242 — second-order mean wave drift force (TensorBEM)

**Delivered**: `src/tensorbem/drift_force.py` (three independent routes + closed-form
anchor + Richardson helper), `src/tensorbem/benchmarks/drift_force_study.py` (one-shot
JSON + 4 figures), `tests/test_drift_force.py` (12 tests), `docs/drift_force_report.md`,
`artifacts/drift_force/` (study.json + 4 PNG).

## Verdict (numbers from `artifacts/drift_force/study.json`)

| gate | target | measured | pass |
|---|---|---|---|
| near field vs anchor `rho g a^2/2` | < 3 % | 0.019 % (N=1600) | yes |
| near field, textbook surface+waterline form | < 3 % | 9.6e-8 | yes |
| far field (travelling-wave + radiation stress) | < 3 % | ~1e-15 | yes |
| panel Richardson extrapolation residual | < 2 % | 8.3e-5 | yes |
| cylinder `(ka)^3` asymptotic slope | 3 ± 0.2 | not evaluable | **no (honest FAIL)** |

The 3-D Maruo far-field flux route is **not converged in the radiation radius**
(the interference channel oscillates with kR, relative spread ~100 %); for a fore-aft
symmetric dipole `int |K|^2 cos(theta) dtheta` vanishes identically, so the whole
force sits in the interference channel and needs the analytically reduced Maruo
`(1+cos theta)` kernel. This is pinned by
`tests/test_drift_force.py::test_maruo_far_field_is_not_converged_in_radius` and must be
replaced, not relaxed, when the reduction lands.

## Bug found and fixed during the work

Integrating `q = -rho phi_t - (1/2) rho |grad phi|^2` up to the instantaneous surface
returns `3 rho g a^2/2` (exactly 3x the anchor) unless the still-water compensation
`-(1/2) rho g zeta^2` is applied — `q = p + rho g z` double counts `rho g <zeta^2>/2`
over the moving wetted surface. Without it the near-field error is O(1), not O(h).

## Reproduce

```
PYTHONPATH=src python3 -m tensorbem.benchmarks.drift_force_study \
    --json artifacts/drift_force/study.json --figures artifacts/drift_force
PYTHONPATH=src python3 -m pytest tests/test_drift_force.py -q
```
