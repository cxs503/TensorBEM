# Audit: CASE-BEM-BARGE-HYDROSTATIC (m17208) — close-out verification

*Audit mission m17240, verdict_by `board-audit`. Subject: m17208 "CASE-BEM-BARGE-HYDROSTATIC: Wall-Sided
Barge Hydrodynamic Cross-Validation" (TensorBEM, status `succeeded`, 2026-11-13 02:25Z).
This is a close-out check of an existing delivery, not a re-run of the research. Level judgement is
functional substance (L0-L5 rubric), the benchmark registry is the only judge.*

## 1. Verdict

**L4** — the case is a real, reproducible benchmark validation (real 3D free-surface BEM solve +
independent capytaine reference on identical panels + per-cell error table + one-command repro),
**but the validation does not pass the module's own pre-declared 5 % band** and the m17208 report does
not say so (§6 finding F3). Registry tolerance for this case is `<10 %` (case pool
`BEM-BARGE-HYDROSTATIC`); the physically correct *bare* route meets it at the finest mesh
(max 5.70 %), the *lid* diagnostic route does not (17.86 %). Not L5: the anchor is a second panel
code, not an experiment or an analytic solution.

## 2. Three-piece check on m17208 (artifact / numbers / repro)

| piece | claimed | verified |
|---|---|---|
| run_artifacts | `benchmarks/verified/barge_crosscheck/{results.json,capytaine_ref.json}` | present, 36 comparison cells + 3 mesh levels + sealed-box root-cause record + provenance metadata (`tool=capytaine 3.1.dev0`, loongarch64, python 3.11.6, 2026-11-12T16:38:08Z) |
| ref vs got | report §4 table (12 rows at 640 panels) | 12/12 rows reproduce the artifact to <1e-3 absolute (<5e-4 on the printed percentage) — no hand-copied number found |
| repro_cmd | `PYTHONPATH=src python -m tensorbem.benchmarks.barge_crosscheck --write`; `pytest tests/test_barge_crosscheck.py -v` | tests: **9 passed** (1.8 s, audit run). Reference producer `scripts/barge_crosscheck_ref.py --out /tmp/ref_recheck.json` re-run by this audit: **56/56 numeric keys identical to the committed reference, max rel diff 0.0** (capytaine 3.1.dev0, same machine class). TensorBEM side: 160-panel level re-solved from current sources, **max rel diff 8.9e-16** vs `results.json` (23 s); 40-panel level is re-derived inside the test ledger at <1e-10. 640-panel level submitted to the cluster (`cj1463`) — not finished at hand-off, marked **not_run** below. |

`tests/test_barge_crosscheck.py` does assert invariants rather than merely executing code: the coarse
level is re-solved and compared to the committed cells at 1e-10 (`test_coarsest_level_reproduces_committed_numbers`),
the verdict is recomputed from the cells (`test_verdict_follows_from_the_cells`), the prose must equal
`render_document(study)` (`test_document_matches_the_artifact`), and the old negative control is
re-measured (`test_old_negative_control_was_a_geometry_defect`: waterplane face survives
`immersed_part()`, volume 2.0 m³ for a T = 0.5 case, negative heave damping).

## 3. Registry check (the only judge)

`SELECT case_name, metric, tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-BARGE-HYDROSTATIC'`
returned **0 rows**, and the case-pool row (`tensor_case_pool.id=144`, `case_code=BEM-BARGE-HYDROSTATIC`,
`repo=TensorBEM`, `ref_source=capytaine 交叉校验（同三角剖分）`, `tol_policy=<10%`, `tier=core`,
`status=dispatched`, `linked_anchor=` **empty**) had never been promoted to a benchmark case. So m17208 was
`succeeded` with `server_quality_audit = {gate_valid:false, key_present:false, gate_missing_warning:true}`:
there was no machine judge to consult. This audit registers the missing row (see §7) using the case-pool
tolerance; the registered metric is the bare route at 640 panels, with the lid-route and 5 %-band failures
written into the metric text rather than dropped.

## 4. Numbers, as measured from the artifact (no transcription)

| mesh (panels) | worst \|err\| bare | worst \|err\| lid | verdict at 5 % |
|---:|---:|---:|---|
| 40 | 24.398 % | 43.255 % | FAIL |
| 160 | 12.540 % | 25.545 % | FAIL |
| 640 | 5.702 % | 17.860 % | FAIL (4/24 finest cells) |

At 640 panels: added mass max \|err\| bare 1.839 %, lid 4.670 %; damping max \|err\| bare 5.702 % (ν = 2),
lid 17.860 %. Band summary over all 36 cells: 14 cells > 5 %, 8 cells > 10 % (all damping, 5 of them lid).
The convergence sequence 24.40 → 12.54 → 5.70 % (bare, worst cell) is monotone, i.e. the residual is
discretisation, not a frozen bias.

## 5. Level rationale

* L2 (independent function): `src/tensorbem/benchmarks/barge_crosscheck.py` builds the wetted-surface mesh
  (`geometry3d.box_hull_mesh`), solves heave radiation (`free_surface_solver3d.solve_radiation_3d`),
  optionally with an interior lid (`interior_panels.waterplane_lid_points`), and writes ref/got cells — input →
  compute → output. Real body, no stub.
* L3 (integration): the solver entry `solve_radiation_3d` is exported from `tensorbem/__init__.py:97`
  and called by five other benchmark drivers (`assembly_scaling`, `capytaine_attribution`,
  `capytaine_crosscheck`, `cross_solver_regression`, `barge_crosscheck`).
* L4 (benchmark): complete case + independent reference + per-cell error analysis + reproducible command,
  verified above by an independent re-run of both sides.
* L5: not met — the anchor is another numerical code under the same linearised potential-flow model
  (the document itself states agreement "does not certify the physics"), no experimental or analytic value.

## 6. Findings (severity ranked)

* **F1 (medium, delivery hygiene).** m17208 produced **no new computation and no code**: its workspace diff is
  `M docs/fig_barge_crosscheck_convergence.png`, `?? docs/fig_barge_crosscheck_error.png`,
  `?? docs/report_m17208_barge_crosscheck.md`; the solver, tests and both JSON artifacts predate the
  mission (`git log` → `9a7470c Merge #718 (m16571)`). `tensor_dev_missions.result.head_commit` is `null`,
  so the report and the error figure were **never committed**. This audit copies them into
  `artifacts/m17208_audit/` (`report…md` sha256 `6e02efcb…faaa`, `fig…error.png` sha256 `bc93c42e…1fc`).
* **F2 (medium, governance).** No `tensor_benchmark_cases` row existed for the anchor and the case-pool row's
  `linked_anchor` was empty → the gate had nothing to judge (§3). Registered by this audit.
* **F3 (medium, tolerance moved after the fact).** The module fixes its band **before** the run
  (`TOLERANCE = 0.05`, echoed in `results.json.comparison.tolerance` and in the committed
  `docs/barge_crosscheck.md` header: *"Verdict: FAIL — worst deviation 43.26 %"*). The m17208 report judges
  against the looser 10 % case-pool tolerance and never mentions the 5 % band or the artifact's own
  `FAIL` verdict; a reader of the report alone would conclude the case passes. Both statements are true
  simultaneously (bare/finest < 10 %, all-mesh/5 % FAIL); the omission is the defect, not the number.
* **F4 (low, report discipline).** The report has no explicit answer to the owner's "是否 < 3 %" question
  (element ⑥); the closest sentence is §5.2 "低频 … <3 %", which leaves the high-frequency 5.70 % /
  lid 17.86 % unjudged.
* **F5 (low).** `repro_cmd` in the report is anchored at `/root/tp-agent/work/m17208` (workspace path) instead
  of a repo-relative command; the two commands themselves work from any checkout with `PYTHONPATH=src`.
* **F6 (info, open physics).** lid route carries a frequency-dependent bias (+10.89 %, +10.85 %, −17.86 % at
  640 panels) and bare damping loses accuracy at ν = 2 (5.70 %, growing to 24.40 % at 40 panels). The lid is a
  diagnostic device, not the case geometry, but the bias is unexplained in the artifact and is the natural
  next node.

## 7. Checks run / not run

Run: registry query; artifact schema/cell count; report-vs-artifact number trace (12/12); pytest
`tests/test_barge_crosscheck.py` (9 passed); independent capytaine reference regeneration (56/56 keys,
diff 0.0); independent TensorBEM 160-panel re-solve (diff 8.9e-16); `_solve` timing probe showing one
640-panel solve exceeds the local 120 s tool budget (hence the cluster submission).

Not run: 640-panel TensorBEM re-solve — submitted as cluster job `cj1463`
(`PYTHONPATH=src python3 artifacts/m17208_audit/repro_finest.py`, writes `artifacts/m17208_audit/repro_finest.json`);
still queued at hand-off, so this cell is **unknown/not_run**, not passed. No convergence study beyond the three
committed meshes; no analytic (e.g. strip/Ursel) barge reference; no frequency sweep beyond ν ∈ {0.5, 1, 2}.

## 8. Reproduction

```bash
cd <TensorBEM checkout>
PYTHONPATH=src python3 -m pytest tests/test_barge_crosscheck.py -q          # 9 passed
python3 scripts/barge_crosscheck_ref.py --out /tmp/ref_recheck.json        # 56/56 keys == committed ref
PYTHONPATH=src python3 artifacts/m17208_audit/repro_finest.py              # 640-panel re-solve vs results.json
```
## 9. Artifact locations

Working-tree copies used for the gate checks: `artifacts/m17208_audit/` (git-ignored in this repo).
Durable repo copies of the same bytes: `docs/audit_m17208/` (`report_m17208_barge_crosscheck.md`,
`fig_barge_crosscheck_error.png`, `repro_finest.py`, `audit_error_vs_panels.png`).
The 640-panel reproduction script writes `repro_finest.json` next to itself when the cluster job completes.
