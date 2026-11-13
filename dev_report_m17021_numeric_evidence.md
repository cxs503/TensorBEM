# m17021 — Numeric evidence layer: criterion + artifact + independent regrade

**Mission**: m17021 (self-start, model seat) — anchor `CAP-LEDGER-EVIDENCE`
**Repo**: TensorBEM · branch `agent/m17021`
**Date**: 2026-11-12

## 1. Why this slice of the anchor

The ledger anchor asks for 21 capability entries that each carry 判据 (criterion),
工件 (artifact) and 独立复核 (independent re-verification). The platform-level
21-row table lives in `TensorPlatform/config/capabilities.json` — outside this
worktree (and outside this mission's `EDIT_WHITELIST`). What *is* in scope, and
what was genuinely missing, is the evidence discipline itself: this repository's
capability table (`docs/audit_baseline.md` §4, `tensorbem.capability_evidence`)
grades a capability from **structure** — module exists, a test node references
it, no stub body, no RNG marker — and publishes one line of prose as the claim.

Structure alone cannot catch the failure the ledger exists to prevent: a
capability can be structurally `REAL` while the number it produces misses its own
tolerance. Running the new layer against the committed tree immediately produced
such a case (KP505, §3).

Delivered here: a **numeric evidence layer** — registry of criteria, pinned
artifacts, error recomputed from the raw pair, fail-closed grading, and a
structural↔numeric cross-check that publishes the disagreement instead of hiding
it.

## 2. Method

| piece | file |
|---|---|
| grader, CLI (`--check` / `--write`), cross-layer report | `src/tensorbem/numeric_evidence.py` |
| criterion registry (tolerance + literature source + artifact + selectors) | `docs/numeric_evidence_registry.json` |
| generated evidence matrix | `docs/numeric_evidence_matrix.md` |
| gates incl. 11 negative controls | `tests/test_numeric_evidence.py` |

Grading inputs per item: tolerance (`tolerance_pct`), a literature/analytic
citation, the artifact path, a pinned `sha256`, and JSON selectors for the raw
computed and reference values (plain JSON pointer, or name-matched array lookup
so an array reorder cannot silently retarget the record).

Grading rules (fail-closed):

1. artifact absent → `MISSING`; digest ≠ pinned → `STALE`;
2. both numbers resolved → error **recomputed** as `|c − r| / |r| · 100`;
3. the artifact's own reported error, when present, is cross-checked against the
   recomputed one (`reported_error_ratio` for dimensionless fields,
   `reported_error_pct` for percent fields); disagreement → `WITHHELD` — the
   artifact's self-report is never trusted over its own numbers;
4. error ≤ tolerance → `REAL`, else `WITHHELD` with the numbers in the reason;
5. a `WITHHELD` row must be registered with `expected_level` + `gap_note`, else
   the gate goes red: an honest FAIL cannot be introduced *or deleted* silently.

## 3. Results

Eight criteria over four capabilities, all re-derived from committed artifacts:

| item | capability | tolerance | recomputed | level |
|---|---|---|---|---|
| barge displaced volume vs analytic L·B·T | hydrostatics | 1e-6% | 0% | REAL |
| barge pitch restoring c55 vs ρgI_yy | hydrostatics | 1e-6% | 0% | REAL |
| hemisphere volume, 4096 panels vs 2πR³/3 | hydrostatics | 0.2% | 0.147% | REAL |
| hemisphere GM_T vs closed form | hydrostatics | 0.2% | 0.119% | REAL |
| box roll restoring c44 vs ρgI_xx | rao | 1e-6% | 0% | REAL |
| Wigley Cb vs 4/9 | wigley3-hullform | 0.02% | 0.010% | REAL |
| KP505 K_T at design J=0.8 | propeller-open-water | 3% | **29.13%** | WITHHELD (registered gap) |
| KP505 η₀ worst point J=0.9 | propeller-open-water | 5% | **24.38%** | WITHHELD (registered gap) |

Gate output: `{"REAL": 6, "WITHHELD": 2, "MISSING": 0, "STALE": 0, "n_items": 8,
"n_capabilities": 4}`, `--check` exit 0.

**Structural/numeric disagreement (the layer's first real catch):**
`propeller-open-water` is graded `REAL` by the structural audit, yet both of its
committed criteria fail (29.13% vs 3%, 24.38% vs 5%). The row is now published as
a registered gap with the numbers and the cause (the published KP505 particulars
fix only six scalars; the chord/radial-pitch reconstruction is the residual), so
the structural `REAL` can no longer be read as "the number meets its bar".

**Second catch (unit defect in an artifact):** `docs/hydrostatic_verification.json`
stores `volume_relative_error` as a dimensionless ratio (0.00147) while the
sibling fields are percent-scaled. The cross-check channel flagging a 100×
disagreement is what surfaced it; the registry now declares the intended scale
per item, so this class of mismatch can never read as agreement.

## 4. Verification

```
PYTHONPATH=src python -m tensorbem.numeric_evidence --check      # exit 0
PYTHONPATH=src python -m pytest tests/test_numeric_evidence.py -q # 18 passed
```

Negative controls (each in a synthetic repo under `tmp_path`, not on the real
tree): within-tolerance → REAL; overrun → WITHHELD; tightening the tolerance
flips a green item; artifact error field contradicting its own numbers → WITHHELD;
agreeing error field → verdict unchanged; tampered bytes → STALE; missing
artifact → MISSING; unpublished capability → WITHHELD; unregistered gap → gate
red; registered REAL slipping to WITHHELD → gate red; document drift → gate red.

Regression surface: `tensorbem.capability_evidence` gained one row
(`numeric-evidence`, level REAL) and the published block in
`docs/audit_baseline.md` was regenerated, so the structural gates stay green.

## 5. Caveats

* Coverage is deliberately partial: 8 criteria / 4 capabilities of the 78 rows in
  the structural matrix. The remaining capabilities need their numbers extracted
  into artifacts before they can be bound; that is the follow-up queue, not a
  claim of completeness.
* `tests/test_capability_coverage.py` has three pre-existing failures on `HEAD`
  (`tensorbem.shape_gradient`, `tensorbem.steady_kelvin` unowned) — reproduced on
  an untouched `git archive HEAD` checkout, therefore not caused by this change
  and not touched here.
* Tolerances are declared from the governing document per item (analytic identity
  → 1e-6%; convergence claim → 0.2%; published benchmark criterion → its own
  threshold). No tolerance was chosen to make a row pass: the two KP505 rows fail
  under their own published thresholds.

## 6. Next

1. Bind the remaining high-value capabilities (`haskind-2d-excitation`,
   `maruo-far-field-drift`, `finite-depth-3d`, `kcs-class-hull-anchors`) — each
   already has a committed JSON with computed/reference pairs.
2. Feed `numeric_evidence.audit()` into the platform-side ledger so the 21-entry
   chain can consume per-repo numeric records instead of prose.
3. Report the live defect `audit_capabilities` hits: it reads
   `/root/ship-performance-platform-incoming/.../platform_harness/capabilities.json`,
   which is now `capabilities.json.bak`, so the tool raises `FileNotFoundError`
   for every caller — the platform ledger cannot even be read today.