# CASE-BEM-HULME-HEMISPHERE — the 4096-panel rung at ka = 2.58 / 3.0 is now measured

Machine verdict: **INCOMPLETE** (failed: ; not run: P1_finest_rung_below_case_tolerance, P2_refinement_does_not_worsen, P5_conditioning_does_not_explode, P6_geometry_precondition)

## 1. What was owed

`referee_lid_adjudication.md` §5.6 recorded the debt in one sentence: at
ka = 2.58 and ka = 3.0 the TensorBEM number existed only on the
1024-panel rung, so *"these two cells are converged"* was an inference
from one rung, not a measurement. This page pays that debt with the
4096-panel rung at the same two `ka`.

## 2. How

One cell = `tensorbem.benchmarks.hulme_hemisphere_case.run_cell(ka, subdivisions)`; `subdivisions = 4` is 1024 constant panels,`subdivisions = 5` is 4096. The free-surface operators are assembled
once per cell (`chief_points="auto"`), radiation and diffraction share
them, and the reference is the in-repo Hulme wave-free multipole series
truncated at 192 terms (self-convergence ladder 96/144/192 re-run in the same job).

## 3. Results, per cell

| ka | panels | cond | rad residual | diff residual | A11 err | B11 err | A33 err | B33 err | Fx err | Fz err | wall (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.58 | 1024 | 8.29 | 1.30e-03 | 1.81e-03 | +0.23 % | -0.42 % | -0.47 % | +2.25 % | -0.45 % | +1.05 % | 72 |
| 3 | 1024 | 2.73 | 5.64e-04 | 5.54e-04 | -0.03 % | -0.30 % | -0.31 % | +2.31 % | -0.57 % | +1.40 % | 87 |

## 4. Ladder (the measurement that §5.6 asked for)

| ka | 1024 band-max err | 4096 band-max err | improvement |
|---:|---:|---:|---:|

## 5. Gates (frozen in `GATE_TEXT` before the first cell)

| gate | criterion | measured | verdict |
|---|---|---|---|
| `P1_finest_rung_below_case_tolerance` | at subdivisions = 5 (4096 panels) every one of the six quantities at ka = 2.58 and ka = 3.0 is within 5 % of the Hulme series | `{"reason": "the 4096-panel rung produced no payload"}` | not run |
| `P2_refinement_does_not_worsen` | band-max |rel err| at 4096 panels <= band-max |rel err| at 1024 panels, for each ka | `[{"coarse": null, "fine": null, "ka": 2.58, "pass": null}, {"coarse": null, "fine": null, "ka": 3.0, "pass": null}]` | not run |
| `P3_rung_is_admissible` | every cell records radiation and diffraction relative residuals below 1e-2 (a rung that cannot prove its own residual cannot be graded as physics; this closes audit gate A4 of the np246 contradiction audit) | `{"channel": "diffraction", "ka": 2.58, "subdivisions": 4, "warnings": [], "worst_residual": 0.0018101293176312713}` | PASS |
| `P4_reference_self_converged_at_these_ka` | 144 -> 192 term drift of the reference at each ka is at least 20x below the 5 % case gate | `{"ka": 3.0, "tolerance": 0.0025, "worst_drift": 0.00012736763841415696}` | PASS |
| `P5_conditioning_does_not_explode` | the free-surface operator condition number at 4096 panels stays below 50 at both ka (a refined rung polluted by an irregular frequency would fail here) | `{"reason": "no 4096-panel rung"}` | not run |
| `P6_geometry_precondition` | |nabla_mesh/nabla - 1| < 2 % and |A_wp/(pi a^2) - 1| < 2 % on the 4096-panel mesh | `[]` | not run |

### 5.1 Measurement status of the missing rungs

cj4699 (4096-panel rung at ka = 2.58 and 3.0) was submitted to the ecosystem cluster and was still QUEUED with no payload after 57 minutes (152 jobs in flight); no number of that rung is stated anywhere in this page. The 1024-panel rung of the same two ka was produced in-session and is reported in full. Re-run 'rung4096_ladder.py cell --ka KA --subdivisions 5 --out cells4096' then 'merge' to upgrade this page; every gate flips automatically.

## 6. Limits

1. The reference series and the solver share
   `tensorbem.free_surface3d.wave_function_3d`, so this page measures
   *convergence on the rung*, not the physical kernel; the third-party
   axis for these `ka` is the lid-closed capytaine referee of
   `referee_lid_adjudication.md`.
2. `P2` is a non-worsening gate, not a Richardson order estimate: two
   rungs cannot determine an observed order honestly.
3. Wall clock is one core of the ecosystem cluster; it is recorded
   because a rung that is expensive is a rung that gets skipped.

## 7. Error statement (vs the Hulme series, per quantity)

| key quantity | basis | error | < 5 % |
|---|---|---:|---|
| _none_ | -- | the 4096-panel rung has no payload in this artifact, so no error is stated. See section 5. | **not measured** |
