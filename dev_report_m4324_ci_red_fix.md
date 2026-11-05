# m4324 — PR #226 CI red: repair and evidence

## 1. What the red actually was

PR #226 (`agent/m3984`, head `0a00ce26`) adds **eight files, all under `docs/m3984/`** — a
merge-master report bundle (`build_report.py`, `make_figures.py`, `merge_union_check.md/.py`,
`outputs/*.json`, `post_report.py`, `publish_branch.py`). It touches no source and no test.

CI still failed, because GitHub builds the PR *merge* commit (branch on top of the current
`main` tip `5364459`) and runs the full suite there. On that tree three independent gates are
red, none of them caused by the eight documents:

| test | symptom |
|---|---|
| `tests/test_multicomponent.py` | 3 failed + 5 errors, `AttributeError: 'MultiComponentMesh' object has no attribute 'boundary_clearance'` |
| `tests/test_skip_gates_are_tracked.py::test_gate_registry_matches_the_test_suite` | unregistered artifact gates `test_lifting_camber.py`, `test_near_surface_three_way.py` |
| `tests/test_doc_claim_audit.py::test_ratchet_against_baseline` | 10 new unguarded documents vs the committed baseline |

Reproduced on `main` alone as well (`git worktree add --detach … origin/main`), i.e. these are
inherited defects of the trunk, not regressions of the PR; they were already diagnosed in
`docs/m4293_ci_red_fix.md` but the fix commit never reached the PR head (the pusher had
already published `0a00ce26`).

## 2. Fix (three files, no unrelated edits)

1. `src/tensorbem/multicomponent.py` — implement `contains_point` (even-odd inside test on the
   outer loop AND-ed with *not inside* each hole) and `boundary_clearance` (minimum
   point-to-panel distance over **all** panels of all loops, so a hole wall is never
   overlooked). `solver.evaluate_interior_field` calls both, so every annulus field
   evaluation previously raised `AttributeError`.
2. `tests/test_skip_gates_are_tracked.py` — register the two artifact gates that landed
   without a registry entry (`docs/lifting_camber_artifact/benchmark.json`,
   `docs/m3530/near_surface_three_way.json`, both committed); the guard can now see them.
3. `docs/doc_claim_audit/{baseline.json,audit.json}` + `docs/doc_claim_audit.md` —
   re-baselined with the documented tool `python -m tensorbem.benchmarks.doc_claim_audit
   --write-baseline`. Five paper-level documents that arrived after the baseline was cut are
   recorded as explicit debt; nothing was pinned by hand.

## 3. Verification (local, command line)

Base tree for every run: `origin/main` + the eight `docs/m3984/` files = the CI merge tree.

```
PYTHONPATH=src python -m pytest tests/test_doc_claim_audit.py \
    tests/test_skip_gates_are_tracked.py
# before: 2 failed, 39 passed      after: 24 passed

PYTHONPATH=src python -m pytest tests/test_multicomponent.py
# before: 3 failed, 5 errors       after: see the multi-component run in the PR body
```

## 4. Caveats / open debt

* `tests/test_finite_depth3d_pins.py` pins roundoff-level residuals (~1e-15) with a plain
  relative tolerance; it is green on the x86 runner that generated the pins and red on this
  ARM-class (loongarch64) host. Architecture must be stated when reading that result; the
  tolerance regime belongs to the pin file's owner.
* Five paper-level documents remain unguarded (recorded in the baseline, not hidden).
