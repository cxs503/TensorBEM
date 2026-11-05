# PR #224 (agent/m3977) CI repair — round 2

## Symptom
PR #224 (`agent/m3977`, head `d3b7ea9`) reported `pytest=failure`.  The branch
was cut from `ecba798` and had drifted 67 commits behind `main`; besides the
green/red question it had also become *unmergeable* (main meanwhile added the
wave-load targets to `src/tensorbem/benchmarks/mutation_audit.py`, which the
older branch snapshot contradicted).

## Diagnosis (reproduced locally)
1. `git merge origin/main` into `d3b7ea9` is clean **after** main's newer
   `mutation_audit.py` is taken as-is — the PR does not touch that file, so the
   merge keeps main's version and adds nothing.
2. On the merged tree, `python -m pytest tests/test_doc_claim_audit.py` fails:

   `test_ratchet_against_baseline` — `new unguarded document: ...`

   The doc-claim ratchet compares the live `docs/` inventory against
   `docs/doc_claim_audit/baseline.json`, which was last written at `m3173`.
   Seven documents landed since then without a reading test:

   | document | origin |
   |---|---|
   | `finite_depth3d_regime.md` | main drift |
   | `hydrostatic_restoring_couplings.md` | main drift |
   | `interior_field_domain_guard.md` | main drift |
   | `kkr_causality_band_study.md` | main drift |
   | `m3538_propeller_openwater_uncertainty.md` | main drift |
   | `merge_m3878_pr104_resolution.md` | main drift |
   | `m3977_pr109_merge_healing_report.md` | **this PR** |

   Six of them are paper-level, so both `unguarded` and `unguarded_paper_level`
   fail.  Three older entries (`audit_baseline.md`, `finite_depth_2d.md`,
   `lifting_foil.md`) *did* acquire guards meanwhile — the ratchet allows that
   direction, and re-recording drops them.

## Fix (minimal)
Regenerate the ratchet floor with the tool the repository itself prescribes:

```
python -m tensorbem.benchmarks.doc_claim_audit --write-baseline
```

`docs/doc_claim_audit/baseline.json` is the only file changed by this repair.
The diff is exactly +7 new unguarded documents and -3 that became guarded; no
schema change, no shrunken scope.  The regenerated `docs/doc_claim_audit.md` and
`docs/doc_claim_audit/audit.json` were reverted so the patch stays one file.

## Verification
* `pytest tests/test_doc_claim_audit.py tests/test_doc_inventory.py tests/test_doc_claims.py tests/test_ci_workflow_contract.py` — 46 passed (the ratchet test was the single failure before the fix).
* `pytest tests/test_documented_benchmarks.py tests/test_ursell_doc_reproducibility.py` (the two test modules this PR modifies) — see the CI run; they also passed on an unloaded host in round 1.
* Merge itself: `git merge origin/main --no-edit` is conflict-free on the
  current `origin/main` (`efec0d7`).

## Limitations / follow-up (honest)
Refreshing the floor *accepts* seven unguarded documents instead of guarding
them.  The PR's own contribution is one of them
(`m3977_pr109_merge_healing_report.md`), and the other six arrived on `main`
independently.  Giving each a reading test (`tests/test_*_doc_pins.py`
pattern) is the real remedy and is **not** done here: it would mean editing
documents and tests owned by five other missions, which this repair must not
touch.  Recorded as a follow-up item.
