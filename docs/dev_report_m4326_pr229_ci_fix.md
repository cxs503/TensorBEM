# m4326 — PR #229 (`agent/m4008`) CI red: reproduction, root cause, fix

## 1. Where the red actually was

The branch `agent/m4008` (head `2e39173`) was cut from an old main (`ecba798`).
GitHub evaluates pull requests on the **merge** of head into current `main`, so
the red was reproduced locally the same way: `git merge origin/main`
(`5364459`) into a worktree branched from `agent/m4008` — the merge is textually
clean, no conflicts, so the failure was purely semantic.

Reproduced failures (3, all in the audit/gate layer, none in the branch's own
new mathematics):

| # | test | failure |
|---|---|---|
| 1 | `tests/test_capability_evidence.py::test_uncapability_baseline_has_no_untested_module_without_a_reason` | `unexplained == ['tensorbem.benchmarks.finite_depth3d_verification']` |
| 2 | `tests/test_capability_evidence.py::test_published_block_matches_the_tree` | `docs/audit_baseline.md` §4 drifted from the tree |
| 3 | `tests/test_doc_claim_audit.py::test_ratchet_against_baseline` | 11 "new unguarded document" entries (e.g. `finite_depth3d_regime.md`) |

All three are main-side drift that arrived through the merge: main added
`src/tensorbem/benchmarks/finite_depth3d_verification.py` and a batch of new
documents after `agent/m4008` forked, without updating the two published
baselines those gates compare against. The branch's own deliverables
(`docs/m2966/*`, `src/tensorbem/benchmarks/rankine_primitive_referee.py`,
`tests/test_rankine_primitive_referee.py`, the per-mesh MFS claims) pass:
`tests/test_rankine_primitive_referee.py tests/test_doc_claims.py
tests/test_ursell_doc_reproducibility.py` → 34 passed.

## 2. Fix (minimal, three files + two regenerated baselines)

1. `src/tensorbem/capability_evidence.py` — one entry in `UNTESTED_MODULES`
   for `tensorbem.benchmarks.finite_depth3d_verification`, stating what the
   module is (report/figure harness writing
   `artifacts/finite_depth3d/verification.json` + a figure for
   `docs/finite_depth3d_green_function.md`) and why no test drives it (its
   purpose *is* to regenerate those artifacts — same class as the already
   listed `hydrostatic_figures`). This is the documented mechanism: the gate
   fails only *unexplained* unreferenced modules.
2. `docs/audit_baseline.md` — §4 regenerated with
   `python -m tensorbem.capability_evidence --write`.
3. `docs/doc_claim_audit/{baseline.json,audit.json}`, `docs/doc_claim_audit.md` —
   regenerated with `python -m tensorbem.benchmarks.doc_claim_audit
   --write-baseline`: documents=64, pinned=26, artifact=21, unguarded=17
   (paper-level 10), number coverage 83.8 %. Regenerating the ratchet baseline
   *records* the pre-existing unguarded documents instead of leaving the repo
   red; it does not guard them, so 17 documents remain on the debt list.

Nothing in the branch's own deliverables was touched.

## 3. Verification

* Reproduced: `tests/test_capability_evidence.py tests/test_capability_evidence.py::test_published_block_matches_the_tree tests/test_doc_claim_audit.py` on the merged tree → 3 failed, 31 passed.
* After the fix: `tests/test_capability_evidence.py tests/test_doc_claim_audit.py` → **20 passed**.
* Whole-suite run on the merged tree (`pytest -q -n 8`), post-fix: first 16 % of
  the collection (previously the band holding all three failures) is clean;
  the full local run was not carried to completion because the host load
  average was ≈4·10³ (shared build box), so CI remains the authoritative gate.

## 4. Honest scope note

`--write-baseline` accepts rather than removes documentation debt: the 17
unguarded documents are now the recorded floor, and any *new* unguarded
document still fails the ratchet. Turning those 17 into guarded documents is
separate work.