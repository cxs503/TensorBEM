# m4317 — PR #209 CI red: stale capability-evidence baseline after a new module

## 1. What was red

PR #209 (`agent/m3878`, head `e690827`) adds nine files to the m3878 line, two of
which are new Python modules:

* `src/tensorbem/finite_depth3d_quadrature.py` (new kernel/quadrature module)
* `src/tensorbem/benchmarks/finite_depth3d_verification.py` (new report driver)

The CI job `pytest` failed on that head.  The failing node ids are in
`tests/test_capability_evidence.py`:

```
FAILED tests/test_capability_evidence.py::test_uncapability_baseline_has_no_untested_module_without_a_reason
FAILED tests/test_capability_evidence.py::test_published_block_matches_the_tree
```

Both raise the same diagnostic:

```
AssertionError: docs/audit_baseline.md section 4 no longer matches the tree;
regenerate it with `python -m tensorbem.capability_evidence --write`
```

## 2. Root cause

`src/tensorbem/capability_evidence.py` is a self-describing registry: it walks
the package, grades every node (REAL / MOCK / STUB / WITHHELD), and publishes the
result into section 4 of `docs/audit_baseline.md`.  Two invariants are enforced
by tests:

1. **published == tree** — the committed section 4 must be byte-identical to what
   a fresh `--write` produces; a module added without regenerating the document
   breaks the pin.
2. **no unexplained untested module** — every module that no test file references
   must carry an entry in `UNTESTED_MODULES` with a reason.  The new report
   driver `tensorbem.benchmarks.finite_depth3d_verification` is not referenced by
   any test, so it needs an entry *and* a regenerated document.

So the red is not a numerical failure inside the new kernel: it is the
capability-evidence ratchet firing on a tree that grew two modules.

## 3. Fix (two files, five lines)

* `src/tensorbem/capability_evidence.py` — added one `UNTESTED_MODULES` entry for
  `tensorbem.benchmarks.finite_depth3d_verification`, stating what it is (a
  report driver that only assembles/prints the finite-depth 3-D checks) and where
  the physics it reports on is actually pinned
  (`tests/test_finite_depth3d.py`, `tests/test_finite_depth3d_quadrature.py`).
* `docs/audit_baseline.md` — regenerated with
  `python -m tensorbem.capability_evidence --write`, i.e. the document is again a
  program output, not a hand edit.

The new kernel module `tensorbem.finite_depth3d_quadrature` needed no entry: it
is referenced by `tests/test_finite_depth3d_quadrature.py`.

`finite_depth3d_quadrature.py` and its tests are untouched by this commit — the
m3878 numerics (including the two deliberately `xfail`-pinned regimes, the
axis-singularity and the deep-water limit) are carried unchanged.

## 4. Verification

```
# before (head e690827, PYTHONPATH=src so the stale editable install of another
# worktree cannot shadow this tree)
python -m pytest tests/test_capability_evidence.py -q
2 failed, 9 passed        # both failures quoted above

# after
python -m tensorbem.capability_evidence --write
python -m pytest tests/test_capability_evidence.py -q
11 passed

# the PR's own numerics unchanged
python -m pytest tests/test_finite_depth3d.py tests/test_finite_depth3d_quadrature.py -q
46 passed, 2 xfailed
```

Note on the local environment: the machine carries an editable install of
`tensorbem` pointing at a *different* worktree (`/root/tp-agent/work/m4292/src`),
so `import tensorbem` silently loads the wrong tree unless `PYTHONPATH=src` is
set.  CI installs the checked-out tree (`pip install -e .`), so this is a local
reproduction artefact only — but it explains why a naive local `pytest` can look
green (or red) for the wrong reason.

## 5. Limits / open items

* The `xfail` pins inherited from the m3878 merge stay in place: the on-axis
  kernel singularity and the deep-water limit cross-check are still open
  physics debts (`docs/finite_depth3d_offaxis_diagnosis.md`), not fixed here.
* This commit deliberately does not regenerate any other report: `--write` output
  is deterministic, and a broader regeneration would add unrelated churn to a CI
  repair.