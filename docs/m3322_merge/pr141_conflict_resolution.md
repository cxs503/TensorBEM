# PR #141 (agent/m3322): conflict resolution record

Mission m4039 (healing round). Branch `agent/m3322` fell behind `main`; the PR
showed `mergeable=false`. This record states the conflict set, the resolution
per file, and the test evidence, so the merge can be audited without trusting a
narrative.

- Branch tip before the healing merge: `eb06f6c` (`chore: re-run CI against
  current main`)
- `main` merged in: `e8c0aef` (`Merge pull request #138 from cxs503/agent/m3288`)
- Merge base: `4f5b73e`
- Healing merge commit: created on top of `eb06f6c`, tree pushed back to
  `refs/heads/agent/m3322` (SHA recorded in the mission report), so PR #141 is
  re-based on the current `main` by a true merge, not by a rebase or a
  mechanical overlay.

## Conflict set

`git merge --no-commit origin/main` on `eb06f6c` produced exactly **one**
conflicted path:

| path | kind | sides |
|---|---|---|
| `docs/wigley_wave_resistance.md` | content conflict, 2 hunks | `agent/m3322` (m3322 formulation diagnosis) × `main` (m3100 integrand audit) |

Everything else auto-merged: the branch's `docs/wigley_pressure_path_diagnosis/`
tree, `src/tensorbem/benchmarks/wigley_pressure_path_diagnosis.py` and
`tests/test_wigley_pressure_path_diagnosis.py` are disjoint from main's
`docs/m3100/`, `src/tensorbem/capability_evidence.py` and the m3288/m3173/m3203
work, so no file was dropped or overwritten on either side (`git diff --stat`
of the merge against each parent shows both sides' additions present; see
"Content preservation" below).

## Why these two hunks conflicted

Both sides independently rewrote the same two passages of the Wigley document,
each from its own measured audit:

- **main / m3100** — `docs/m3100/wigley_pressure_path_audit.md`: the pressure
  integral's integrand is *odd* in `x` for a fore-aft symmetric hull, so the
  integral vanishes identically by parity (measured `1e-14` relative to
  `Rw_Kochin`); the non-zero number the routine returns is centroid aliasing
  that decays like `h`; ladder at `Fr = 0.30` 0.472 / 0.206 / 0.0729 / 0.0276 /
  0.0059 for 128 … 32768 panels, sign always negative.
- **branch / m3322** — `docs/wigley_pressure_path_diagnosis/wigley_pressure_path_diagnosis.md`:
  the potential used is a function of `(x, z)` only, so `dphi/dy = 0`
  identically where the hull condition requires a transverse flux
  (area-weighted mean `1.12 m/s` at `Fr = 0.30`, relative residual `1.0`), and
  the same potential violates `phi_z = phi_xx/k0` with a residual `4.02x` its own
  `phi_z` scale; `B/L` sweeps (32x) leave the ratio scale invariant
  (spread `5.2e-15`), and the `k`-integral is converged (`1.3e-13`).

The two ladders coincide on the shared meshes (0.472 / 0.206 / 0.0729), i.e. the
two audits measured the same residual in the same code — which is exactly why a
mechanical "take ours" or "take theirs" would have silently deleted a whole
audit from the document.

## Resolution per file (semantic union, no silent drop)

`docs/wigley_wave_resistance.md` — both hunks replaced by a union that:

1. keeps **all five** measured facts, each labelled with the audit that produced
   it (parity, centroid aliasing, h-decay ladder ← m3100; `B/L` invariance,
   converged `k`-integral, `dphi/dy = 0` and free-surface residuals ← m3322);
2. keeps both remedies lists (quadratic near-field term; Dawson/Rankine panels;
   full Havelock Green function with the radiation-condition pole);
3. states the one genuine disagreement of detail — m3100 sees the ratio change
   sign along its `B/L` ladder, m3322 measures exact `B/L` invariance at a fixed
   mesh — and marks the reconciling reading (sign is set by the mesh aliasing
   pattern) explicitly as an *interpretation*, not as a measurement;
4. corrects two base-text sentences that still blamed the thin-ship
   approximation ("evaluates Michell's potential on the actual hull surface…",
   "asymptotic value may not match… due to thin-ship"), so the merged document
   does not assert the refuted cause and its refutation at the same time. These
   are base-text edits, not deletions of either side.

The resolution is reproducible: `runs/resolve_wigley_doc_conflict.py` (in the
mission worktree) substitutes the two conflict blocks with the union fragments
and aborts if any conflict marker survives.

## Content preservation

- Branch-side deliverables present in the merged tree: the four new
  `docs/wigley_pressure_path_diagnosis/` files, the benchmark module and the
  test module (unchanged blobs — auto-merge).
- Main-side deliverables present: `docs/m3100/*`, `src/tensorbem/capability_evidence.py`,
  `docs/doc_claim_audit/*`, `docs/finite_depth3d_*`, `docs/lifting_camber*`.
- Document gates from **both** sides pass on the merged text:
  `tests/test_wigley_pressure_path_audit.py::test_document...` requires
  `parity`/`centroid` + `identically zero`/"no resistance at first order";
  `tests/test_wigley_pressure_path_diagnosis.py::test_main_document_no_longer_asserts_the_thin_ship_cause`
  requires the `wigley_pressure_path_diagnosis` cross-reference and
  "scale invariant", and forbids the old "…do not agree within 5%** because"
  attribution. Both are green (see below).

## Test evidence

Environment: this worktree uses a stale editable install that resolves to
another worktree, so every run below is forced onto the local sources with
`PYTHONPATH=<tree>/src` (`python -c "import tensorbem; print(tensorbem.__file__)"`
verified). Command shape:

```
PYTHONPATH=<tree>/src python -m pytest -q -p no:cacheprovider <core set>
```

Core set (13 files, the touched area plus every document/number gate in it):
`test_wigley_pressure_path_audit.py`, `test_wigley_verification.py`,
`test_wigley_wave_resistance.py`, `test_wigley_hydrostatics.py`,
`test_wigley_normalization_audit.py`, `test_doc_claims.py`,
`test_declared_numbers.py`, `test_ci_workflow_contract.py`,
`test_capability_evidence.py`, `test_benchmark_cli_contract.py`,
`test_documented_benchmarks.py`, `test_milestone_claims.py`,
`test_three_dimensional_doc_pins.py`.

| tree | tests | result |
|---|---|---|
| `main` @ `e8c0aef` (baseline) | 154 collected | **151 passed, 3 skipped, 0 failed** (791 s) |
| merged branch (core set + the branch's `test_wigley_pressure_path_diagnosis.py`) | 166 collected | **163 passed, 3 skipped, 0 failed** (725 s) |

The 3 skips are the `TENSORBEM_HEAVY_TESTS` multi-minute refinement studies,
skipped identically on both sides. The merged tree is therefore at or above the
`main` baseline (163 = 151 + the branch's 12 diagnosis tests), with no failure
introduced.

## Limits of this record

- The full repository suite (1010 tests) was started on `main` but did not
  finish inside this mission's budget in this shared, heavily loaded host; the
  core set above is the verified comparison. No claim is made about the other
  ~850 tests beyond "not run here".
- The thin-ship/`B/L` sign question is reconciled by an interpretation, not by a
  new measurement; a single sweep with both parameterisations (`B` alone vs
  `B` and `T` at fixed `B/T`) at a fixed mesh would settle it and is not done
  here.