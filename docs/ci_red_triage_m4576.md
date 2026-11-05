# CI red triage: PR #278 (branch `agent/m4287`) — inherited red, not the PR's own content

Mission m4576 (round 2). Repo: TensorBEM. PR #278 = one commit on top of the
2026-11-05-era `main` (merge-base `5364459`, PR #173) that adds a frozen
`.baseline/` archive snapshot (80 files, +23 075 lines). Net content of the PR
after merge: **zero** — `.baseline/` is byte-identical to what `main` already
carries (merged earlier as `db57840`).

## 1. What CI actually failed on

Run `42067971596`, job `127387378624` (`pytest`): steps 1–22 (install + all 17
benchmark/evidence generators) **all succeeded**; only step 23 `Run test suite`
failed. The failure list, read from the job log:

| module | failure |
|---|---|
| `test_finite_depth3d_pins.py` | 3 pinned rows DRIFT (table 3.2 `free_surface_residual`, h=2/4/10), worst 1.72e-01; plus `test_mutated_document_is_caught` (4 rows), `test_cli_artifact_roundtrip` |
| `test_doc_claim_audit.py` | `test_ratchet_against_baseline`: 11 "new unguarded document" + 5 "paper-level lost its guard" |
| `test_capability_evidence.py` | `tensorbem.benchmarks.finite_depth3d_verification` unregistered vs `docs/audit_baseline.md` §4 |
| `test_skip_gates_are_tracked.py` | unregistered ARTIFACT gates: `test_lifting_camber.py`, `test_near_surface_three_way.py` |
| `test_multicomponent.py` | 9 × `AttributeError: 'MultiComponentMesh' object has no attribute 'boundary_clearance'` |

## 2. A/B experiment: is `.baseline/` the culprit?

The snapshot is a doc/pin copy from a *different* tree state, so the obvious
hypothesis was that doc/pin/evidence scanners pick up the stale copies. Tested
directly on the PR tree (`git archive` of `agent/m4287`):

| tree | result of the 5 affected modules |
|---|---|
| PR tree **with** `.baseline/` | 11 failed, 5 errors, 44 passed |
| same tree, `.baseline/` removed | **11 failed, 5 errors, 44 passed** (identical set) |

So `.baseline/` is **CI-neutral**: the PR's own contribution is not the cause.
The remaining redness is inherited from the stale base — those modules were
already red at `5364459` and were repaired by later `main` PRs (e.g.
`multicomponent` → `multiloop` refactor, `finite_depth3d_verification`
registration).

## 3. What is still red on current `main` (the real fix target)

Running the same modules on the current `main` tip (`a261e23`) shows two of the
failures are **live defects on `main`**, not history:

* `test_skip_gates_are_tracked.py::test_gate_registry_matches_the_test_suite` —
  the AST scanner finds two live `ARTIFACT` skip gates that the registry never
  audited. Both artifacts are committed and tracked
  (`docs/lifting_camber_artifact/benchmark.json`,
  `docs/m3530/near_surface_three_way.json`), so the gates can fire and must be
  registered; until then a genuinely dead gate could hide among them.
* `test_doc_claim_audit.py::test_ratchet_against_baseline` — 9 documents added
  by recent merged PRs are missing from `docs/doc_claim_audit/baseline.json`
  (6 unguarded, 3 paper-level).

Any branch, including PR #278, stays red as long as these two are red.

## 4. Fix applied (minimal, additive)

1. `tests/test_skip_gates_are_tracked.py`: registered the two audited,
   committed-artifact gates in `ARTIFACT_GATES`.
2. `docs/doc_claim_audit/baseline.json`: regenerated with
   `python -m tensorbem.benchmarks.doc_claim_audit --write-baseline`; the diff
   is **additive only** (+9 names, 0 removed), so the ratchet floor is
   preserved, not lowered.

Verification (local, `python -m pytest`):

| selection | before | after |
|---|---|---|
| `tests/test_doc_claim_audit.py` + `tests/test_skip_gates_are_tracked.py` | 2 failed, 22 passed | **24 passed** |
| `tests/test_capability_evidence.py` + `tests/test_documented_benchmarks.py` | pass | pass |

## 5. Note on PR #278 itself

Because the PR's only added tree is already on `main`, re-basing it onto the
current `main` makes its diff empty; with §4 merged, that branch's `pytest`
step is green (same tree as `main`). Nothing in the PR needs to be reverted and
no unrelated file was touched.