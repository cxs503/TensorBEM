# Verified anchors: thin-ship wave resistance and Wigley hydrostatics

This directory holds the machine-checkable acceptance records of the
Michell/thin-ship anchors.  Each subdirectory contains

| file | content |
|---|---|
| `results.json` | every number of the study, produced programmatically (thresholds, measured values, verdicts, artifact paths) |
| `report.md` | the paper-level summary: case, method, tables, limitations, acceptance table |
| `figures/<study>.png` | convergence and comparison figures |

Regenerate with

```
python -m tensorbem.benchmarks.thin_ship_verification            # all three studies
python -m tensorbem.benchmarks.thin_ship_verification --study ellipsoid_wave
```

and verify with

```
pytest tests/test_thin_ship_michell.py tests/test_thin_ship_anchor_artifacts.py
```

| study | physics | reference | acceptance (passed) |
|---|---|---|---|
| `bem_wigley_hydrostatics` | Wigley III parabolic hull, `L/B = 10`, `B/T = 1.6`: displaced volume, `Cb`, `KB`, waterplane area and second moments, wetted area | closed forms `V = 4LBT/9`, `Cb = 4/9`, `KB = 5T/8`, `A_wp = 2LB/3`; wetted area by 2-D Gauss quadrature | `Cb` < 0.5 %, `V` < 1 %, `S_w` < 2 %, GCI < 3 % (measured: 0.03 %, 0.03 %, 0.003 %, GCI 1e-4) |
| `bem_ellipsoid_wave` | surface-piercing spheroid, `L/B = 10`, `T/L = 0.06`, `Fr = 0.2/0.3/0.4`: wave resistance | Michell integral with the depth integral done analytically, `lambda` form, `lambda_max = 40`, `n_lam = 360`, cross-checked against the `theta` form | panel error < 5 % and GCI < 3 % at each Froude number (measured: 1.1/0.9/0.6 %, GCI 2.7/1.1/0.7 %) |
| `bem_multihull_interference` | two Wigley III hulls in tandem, clear gaps `L/2, L, 2L`, `Fr = 0.2/0.3/0.4`: `sigma = R_pair/(2 R_single)` | far-field superposition `A_pair = A(1 + exp(i alpha d))`; panel path sums both sheets | `|sigma_panel - sigma_ref| <= 0.02` (measured 0.0013), GCI(`sigma`) < 3 % (measured 0.023 %), `sigma(0) = 2` exact, `sigma(d -> inf) -> 1` |

## What these anchors do and do not establish

They verify that the *panel discretisation* of the Rankine centreplane source
sheet reproduces the same integral evaluated without that discretisation, and
they quantify the panel count needed for a given accuracy (first order in the
panel size for the spheroid, where `y_x` is singular on the keel line).

They do **not** validate Michell theory against experiment.  Michell's is a
slender-body approximation -- no viscosity, no wave breaking, no sinkage or
trim, no finite thickness -- and no experimental data set ships with this
repository, so the evidence level for the absolute resistance values is
`WITHHELD`.  The interference factor of the tandem study is a ratio and is
therefore independent of the constant in front of Michell's integral; its
physics (coherent vs incoherent wave systems) is what the numbers test.

## Known open item

`tensorbem.wigley.michell_wave_resistance` (the older, Wigley-only path) uses a
`lambda` weight of `lambda^-2 / sqrt(lambda^2 - 1)` and no `g^2` factor, whereas
the standard form used here is `lambda^2 / sqrt(lambda^2 - 1)` with
`(4 rho g^2 / (pi U^2))` in front and the two different wavenumbers
`k = k_0 lambda^2` (depth decay) and `alpha = k_0 lambda` (streamwise phase).
The older path is therefore dimensionally inconsistent and returns a value
about an order of magnitude below the standard form (12.3 N versus 144.1 N for
the Wigley III form at `Fr = 0.30`, wetted area 14.88 m^2).  Fixing it would
change pinned numbers in `tests/test_wigley_verification.py`,
`docs/wigley_wave_resistance.md` and the declared-number ledger, so it is
recorded here as a finding rather than patched inside a verification mission.