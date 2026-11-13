# CASE-BEM-HULME-HEMISPHERE — floating hemisphere, 3-D frequency-domain BEM

Case-pool entry `BEM-HULME-HEMISPHERE` (id 138) of repo `TensorBEM`, tolerance 5 % per cell, unchanged since the case was opened. A hemisphere of radius a = 1 m floats with its flat face on the mean free surface in infinite-depth water; it is forced to oscillate in surge and heave and is excited by unit-amplitude head-on waves. Six quantities are scored: added mass `A11`, `A33`, damping `B11`, `B33` and the two exciting-force amplitudes `|Fx|`, `|Fz|` together with their phases.

**Machine verdict: PASS** — worst relative error on the case mesh (1024 constant panels, subdivisions 4) is +1.15 % on `B33` at ka = 2; on the coarse mesh (256 panels) it is +4.76 %. All four pre-registered gates pass.

## Contents

| path | what it is |
|---|---|
| `results.json` | every number of the run, written by the case driver |
| `reference.json` | the reference series as stored by the run |
| `report.md` | the paper-level report: method, tables, convergence, verdict |
| `evidence.json` | machine acceptance record (levels, impls, callers, tests) |
| `run_case.py` | driver: shard fan-out + merge, exits non-zero on FAIL |
| `independent_axis.py` | third-party (capytaine) cross-check producer |
| `capytaine_excitation.json` | frozen capytaine values, 1600 panels reference level |
| `evidence/` | the machine-readable bundle built by `evidence/build_evidence.py` |
| `README.md` | this file, rendered from the bundle (do not edit by hand) |

## Evidence bundle

| file | content |
|---|---|
| `evidence/reference.json` | reference series per ka, truncation ladder (48/96/144 terms), self-convergence margin, provenance gaps |
| `evidence/benchmark.json` | all rows with re-derived per-cell errors, the four gates, geometry precondition, verdict |
| `evidence/independent.json` | capytaine axis over all six quantities with its own four gates |
| `evidence/anchor_registry.json` | provenance record; states why this anchor is **not** eligible for `benchmarks/anchors/external/index.json` |
| `evidence/artifacts_sha256.json` | sha256 of every artifact the numbers come from, so hand edits are detectable |

## Pre-registered gates

| gate | criterion | measured | verdict |
|---|---|---|---|
| `G1_case_mesh_band_error_below_5pct` | \|rel err\| < 5% for every quantity and every ka on the case mesh | measured = 1.1480e-02 | PASS |
| `G2_geometry_precondition` | \|nabla_mesh/nabla-1\| < 2% and \|A_wp/(pi a^2)-1\| < 2% on the case mesh | volume_error = 5.8696e-03; waterplane_error = 1.6056e-03 | PASS |
| `G3_refinement_does_not_worsen` | band-max error at 1024 panels <= band-max error at 256 panels | coarse = 4.7551e-02; case = 1.1480e-02 | PASS |
| `G4_reference_self_converged` | reference 96->144 term drift at least 20x below the 5 % gate | margin = 154.7x | PASS |

## Computed vs reference — case mesh

| ka | A11 | B11 | A33 | B33 | Fx | Fz | max |
|---|---|---|---|---|---|---|---|
| 0.2 | -0.39 % | -0.96 % | -0.16 % | -0.20 % | -0.52 % | -0.10 % | 0.96 % |
| 0.5 | -0.46 % | -0.99 % | -0.11 % | -0.03 % | -0.54 % | -0.01 % | 0.99 % |
| 0.8 | -0.42 % | -0.96 % | -0.13 % | +0.16 % | -0.53 % | +0.08 % | 0.96 % |
| 1 | -0.33 % | -0.90 % | -0.18 % | +0.30 % | -0.51 % | +0.13 % | 0.90 % |
| 1.5 | -0.01 % | -0.73 % | -0.34 % | +0.69 % | -0.44 % | +0.25 % | 0.73 % |
| 2 | +0.23 % | -0.58 % | -0.53 % | +1.15 % | -0.40 % | +0.49 % | 1.15 % |

Full absolute values, phases, condition numbers and the coarse-mesh twin of this table are in `evidence/benchmark.json` and `report.md`.

## Third-party axis (capytaine)

capytaine 3.1.dev0 solves the same body with 1600 panels. Worst deviation of capytaine from the reference series, per quantity:

| quantity | worst ka | deviation |
|---|---|---|
| `A11` | 0.2 | +2.18 % |
| `B11` | 0.5 | +2.78 % |
| `A33` | 1.5 | +1.67 % |
| `B33` | 2 | -3.36 % |
| `Fx` | 0.5 | +0.57 % |
| `Fz` | 2 | -2.33 % |

Gates I1–I4 pass; the worst cell over all six quantities is +3.36 % (`B33` at ka = 2).

Post-hoc observation, reported because it is honest and *not* a gate: with capytaine as the denominator, the solver's worst cell is +4.67 % (`B33` at ka = 2), i.e. the top of the band is where the 5 % criterion is tightest.

Context only: the ka -> 0 even-mode limit of the same quantity is 0.8310 (antecedent in-repo study, not an independent code); the lowest band point ka = 0.2 sits -4.47 % below it, as expected for a band that stops at ka = 0.2.

## Honest limits

1. **No paper digits.** NOT AVAILABLE (publisher paywall, no copy in the sandbox): no digit of Hulme's printed table is transcribed or hand-copied anywhere in this repository. Every reference number in results.json is computed at run time by the wave-free multipole series (recomputed and cross-checked by evidence.py), so 'ref' here means 'Hulme's semi-analytical method, re-derived and self-converged', not 'Hulme's printed digits' -- see docs/bem_hulme_hemisphere_case.md sections 1 and 6.
2. **Shared kernel.** the reference series and the solver share tensorbem.free_surface3d.wave_function_3d, so agreement alone cannot catch a defect in that kernel; the independent axes are the third-party BEM crosscheck (capytaine, docs/capytaine_crosscheck.md) and the ka->0 rigid-lid even-mode series.
3. **Case-level anchor only.** the reference lives in tensorbem.analytic.hulme_hemisphere_radiation (inside src/) and shares tensorbem.free_surface3d.wave_function_3d with the solver, so it cannot be registered as an external anchor; it is a LITERATURE_CLOSED_FORM re-derivation registered at case level. The external-anchor gate rule `reference_method.independent_of_repo_package` is therefore not satisfied, and the gate's own markers ['tensorbem.'] flag the implementation path.
4. **Band edge.** heave damping at ka = 2 is the constraining cell; the band is not widened to hide discretisation error, instead the mesh ladder in `report.md` shows the error falling under refinement.

## Reproduce

```
python benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py --out benchmarks/verified/casepool_bem_hulme_hemisphere --parallel 8
python benchmarks/verified/casepool_bem_hulme_hemisphere/evidence/build_evidence.py
pytest tests/test_casepool_hulme_hemisphere_results.py tests/test_casepool_hulme_hemisphere_evidence_bundle.py
```

## Related documents

- `docs/bem_hulme_hemisphere_case.md` — the full bilingual case page
- `docs/case_bem_hulme_independent_axis.md` — the capytaine study in prose
- `benchmarks/anchors/external/index.json` — the external-anchor ledger this case is deliberately **not** in
