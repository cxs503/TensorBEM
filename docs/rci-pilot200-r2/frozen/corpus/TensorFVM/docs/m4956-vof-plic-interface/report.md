# PLIC interface transport — minimal verification surface (two-phase VOF)

Module under test: `src/tensorfvm/vof_plic.py`. Reproduce:

```bash
python -m tensorfvm.verification.vof_plic_interface --output <dir>
pytest tests/test_vof_plic_interface.py -q
```

## 1. What is measured and why

The shipped algebraic transport (`two_phase_vof.VOFTransport`) is flux-form and conservative but has no interface reconstruction; its own artefacts record Zalesak peak retention 0.19/0.31/0.52 and relative L1 1.19–1.77 at 32²/64²/128², i.e. the disk is smeared over many cells. A two-phase VOF capability cannot be validated on momentum alone, so this node measures the interface transport itself with a geometric PLIC reconstruction.

## 2. Method

* Reconstruction: Youngs 3×3 normal + plane constant from the exact cut-area inverse (`cut_area`/`plane_constant`, bisection to ~1e-13).
* Transport: directional (Lie) split, flux = exact reconstructed liquid volume in the swept strip of the upwind cell; sweep order alternated each step; periodic boundaries (the rigid-rotation field is divergence-free, exactly and discretely).
* Reference: first-order donor cell with identical velocity field and identical periodic BCs (`donor_advect_periodic`), plus the shipped `VOFTransport(scheme='upwind')` for provenance.
* Geometry provenance: Zalesak geometry is imported from `verification/vof_rotation.py::zalesak_disk` (domain 1, centre (0.5, 0.75), R = 0.15, slot half-width 0.05) — same口径 as the plan-13 P2 fluid leg.

## 3. Results — 1-D step interface

| N | scheme | L1 | LinF | 10–90 % width [cells] | peak retention | worst per-step drift |
|---|---|---|---|---|---|---|
| 32 | PLIC | 2.5000e-02 | 0.400 | 1 | 1.000 | 0.00e+00 |
| 32 | PLIC 1-D | 2.5000e-02 | 0.400 | 1 | 1.000 | 0.00e+00 |
| 32 | donor cell | 1.2032e-01 | 0.489 | 6 | 0.999 | 0.00e+00 |
| 64 | PLIC | 6.2500e-03 | 0.200 | 1 | 1.000 | 0.00e+00 |
| 64 | PLIC 1-D | 6.2500e-03 | 0.200 | 1 | 1.000 | 0.00e+00 |
| 64 | donor cell | 8.4262e-02 | 0.461 | 9 | 1.000 | 0.00e+00 |
| 128 | PLIC | 6.2500e-03 | 0.400 | 1 | 1.000 | 0.00e+00 |
| 128 | PLIC 1-D | 6.2500e-03 | 0.400 | 1 | 1.000 | 0.00e+00 |
| 128 | donor cell | 5.9850e-02 | 0.489 | 13 | 1.000 | 0.00e+00 |

## 4. Results — Zalesak slotted disk (one revolution)

| N | scheme | area error | L1 | LinF | peak retention | distorted fraction | worst per-step drift |
|---|---|---|---|---|---|---|---|
| 64 | PLIC | 0.000 % | 6.103e-03 | 1.000 | 1.000 | 0.018 | 6.61e-16 |
| 64 | donor cell | 0.000 % | 6.321e-02 | 0.837 | 0.332 | 0.230 | 3.30e-16 |
| 128 | PLIC | 0.000 % | 2.617e-03 | 1.000 | 1.000 | 0.007 | 4.82e-16 |
| 128 | donor cell | 0.000 % | 5.768e-02 | 0.816 | 0.443 | 0.194 | 3.21e-16 |

## 5. Criteria

* **C1_interface_sharpness: PASS** — PLIC 10-90 % interface width <= 2 cells at every resolution
  * `{"plic_width_cells": {"32": 1.0, "64": 1.0, "128": 1.0}, "donor_width_cells": {"32": 6.0, "64": 9.0, "128": 13.0}, "plic_l1": {"32": 0.025000000000005407, "64": 0.006250000000004863, "128": 0.00625000000000514}, "donor_l1": {"32": 0.1203224636785955, "64": 0.084261719880677, "128": 0.05985004873823433}, "pass": true, "criterion": "PLIC 10-90 % interface width <= 2 cells at every resolution"}`
* **C2_zalesak_area_error: PASS** — area (volume) error < 5 % after one revolution
  * `{"plic_area_error": {"64": 8.757852324484956e-15, "128": 6.422985182012205e-16}, "donor_area_error": {"64": 0.0, "128": 0.0}, "pass": true, "criterion": "area (volume) error < 5 % after one revolution"}`
* **C3_mass_conservation: PASS** — worst per-step relative volume change < 1e-6
  * `{"plic_worst_step_relative_drift": {"one_d_worst_step_drift": 0.0, "zalesak_worst_step_drift": 6.609699867535812e-16}, "pass": true, "criterion": "worst per-step relative volume change < 1e-6"}`
* **C4_negative_evidence: PASS** — every invalid input (alpha outside [0,1], dt <= 0, CFL > 1, negative directional speed) raises ValueError
  * `{"pass": true, "criterion": "every invalid input (alpha outside [0,1], dt <= 0, CFL > 1, negative directional speed) raises ValueError"}`
* **C5_peak_retention: PASS** — peak retention reported (no criterion in the task book; PLIC must beat donor cell at every resolution)
  * `{"plic_peak_retention": {"64": 1.0, "128": 1.0}, "donor_peak_retention": {"64": 0.33153779290029567, "128": 0.443283443378241}, "criterion": "peak retention reported (no criterion in the task book; PLIC must beat donor cell at every resolution)", "pass": true}`

## 6. Negative evidence

| case | rejected | message |
|---|---|---|
| alpha_above_one | True | alpha has 8 entries above 1 (max 2.5); volume fraction must lie in [0, 1] |
| alpha_below_zero | True | alpha has 6 entries below 0 (min -0.25); volume fraction must lie in [0, 1] |
| alpha_nan | True | alpha contains 1 non-finite entries |
| dt_non_positive | True | dt must be > 0, got 0 |
| dt_negative | True | dt must be > 0, got -0.001 |
| cfl_above_one | True | directional CFL 10 exceeds the geometric-VOF limit 1; reduce dt or split the step |
| negative_directional_speed | True | directional speed must be >= 0, got -0.5; use direction=-1 for a leftward sweep instead of a negative rate |
| negative_velocity_accepted | False | signed back-flow accepted by design |

## 7. Limitations / roadmap

* Surface tension (CSF) and phase change are **out of scope** for this node; they are declared roadmap items: CSF is audited separately (`docs/two-phase-csf-balance`, `docs/two-phase-surface-tension-audit`), phase change has no module yet.
* PLIC is 2-D on a uniform Cartesian grid; a 3-D general polygon-clipping reconstruction and an unstructured-mesh variant are not implemented.
* Dimensional splitting is Lie (two sweeps, alternating order), so it is formally first order in time; a Strang split or an unsplit geometric flux is the next accuracy step.
* Boundedness is a property of the geometric flux; the residual ~1e-14 excursions come from float roundoff in the plane-constant inversion.

