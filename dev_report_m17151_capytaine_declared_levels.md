# m17151 — capytaine crosscheck: run at the declared mesh levels, and attribute the residual gap

*Anchor in the ledger: `KPI-TOOLS-5` was already carried to 10/5 REAL tool classes by
m15147 / m16760 (`tensor_reports` 14265, 14312, 15755), so this ticket took the
next-best verification-debt item inside TensorBEM (see the finish summary for the
suggested replacement ledger entry).*

## 1. Defect found (drift between declared and recorded mesh levels)

`tensorbem/benchmarks/capytaine_crosscheck.py` declares

```
HEMISPHERE_LEVELS = (2, 3, 4)   # 64 / 256 / 1024 panels
SPHERE_LEVELS     = (2, 3)      # 270 / 1026 panels
```

but the committed artifact `benchmarks/verified/capytaine_crosscheck/results.json`
carried hemisphere levels **(64, 256)** and sphere level **(270)** only, i.e. the
declared finest meshes were never exercised, the sphere had no second level (so its
"convergence" table was a single row), and `render_document` *typed* the sentence
"at three (resp. two) hull densities" into the prose.  Prose, declaration and
artifact disagreed and no test bound them, so the recorded verdict `FAIL`
(worst 8.75 %) described a mesh set the module does not claim.

## 2. Fix

* `render_document` reads the level counts from the study; the hard-typed sentence
  is gone.
* `decompose_against_analytic` (new, pure) scores every hemisphere cell against
  Hulme (1982) for *both* codes, so the code-to-code gap can be attributed; the
  result is stored in the artifact (`analytic_decomposition`) and rendered as
  section 3.5 of `docs/capytaine_crosscheck.md`.
* `carry_superseded_records` (new): if a re-run changes the verdict, the previous
  record (level sets, verdict, worst deviation, failures) is moved into
  `superseded_records` instead of being overwritten — the honest FAIL stays in the
  audit trail, and the document lists it (section 3.6).
* Two new gates: `test_artifact_uses_the_declared_mesh_levels` (artifact level sets
  must equal the declared constants, ≥ 2 levels per case) and
  `test_analytic_decomposition_is_consistent` (section 3.5 is the arithmetic of the
  recorded numbers, not prose).  `test_cross_code_fail_is_not_overwritten` now binds
  to the preserved record.

## 3. Result — study re-run at the declared levels (≈ 500 s, single core)

| case | panels | worst dev. vs capytaine |
|---|---:|---:|
| hemisphere | 64 | 22.10 % |
| hemisphere | 256 | 8.51 % |
| hemisphere | **1024** | **4.97 %** |
| sphere | 270 | 8.75 % |
| sphere | **1026** | **3.90 %** |

Verdict flips `FAIL` → **PASS** (4.97 % vs the 5 % band pre-registered before the
run) — margin 0.03 pp, which the document states explicitly: this pair is at its
resolution limit, the claim is only that the declared meshes agree inside the band.

Three-way decomposition (vs Hulme 1982, hemisphere, 12 cells): worst **TensorBEM**
cell **+1.09 %**, worst **capytaine** cell **−3.70 %** (both at nu = 2, heave
damping).  The residual code-to-code gap is therefore dominated by the foreign
code's discretisation error, not by this solver — the analysis that the previous
FAIL could not make.

## 4. Verification

```
PYTHONPATH=src python3 -m pytest tests/test_capytaine_crosscheck.py \
    tests/test_capytaine_attribution.py -q   # 31 passed
```

Pre-existing, unrelated red found while screening:
`tests/test_rigid_lid_hemisphere_attribution.py::test_all_criteria_pass_and_summary_is_rendered`
fails on a missing `benchmarks/verified/bem_rigid_lid_hemisphere/figure.png`
(artifact-not-committed class, cf. report 16109); untouched here.