# m4614 · doc-claim ratchet: mission-prefixed reports are work products, not contracts

Status: fix + regression tests, gate green on current `main`.
Scope: `src/tensorbem/benchmarks/doc_claim_audit.py`, `docs/doc_claim_audit/baseline.json`,
`tests/test_doc_claim_audit.py`, regenerated `docs/doc_claim_audit.md` /
`docs/doc_claim_audit/audit.json` artifacts.

## 1. The defect

`doc_claim_audit` splits every number-bearing markdown file into two orthogonal
axes:

* **tier** — `PINNED` / `ARTIFACT` / `UNGUARDED` (does anything read the numbers?);
* **paper level** — `True` for a contract with a reader, `False` for a dated
  snapshot of one run (dev reports, audits, reviews, incident logs).

The ratchet in `check()` uses both halves: the `UNGUARDED` set may never grow,
and a *paper-level* document may never leave the guard set. The second half is
the strict one — it is what forces a new contract document to be pinned or
generated before it lands.

Membership in "historical" was decided by `HISTORICAL_RE`, which knows the
spellings

| spelling | example |
|---|---|
| `dev_report` | `dev_report_tensorbem_declared_numbers.md` |
| `report_m\d` | `report_m96.md` |
| `_m\d{3,}` | `m3977_pr109_merge_healing_report.md`? — **no** |
| `audit` / `review` / `incident` | `tensorbem_health_audit_m1621.md` |

Mission reports have migrated to *mission id first* (`m4290_ci_fix_report.md`,
`m4283_pr186_ci_triage.md`, `m3977_pr109_merge_healing_report.md`). That
spelling matches none of the patterns: the id is not behind a prefix and there
is no leading underscore. The result was an accounting error, not a numerical
one — three frozen snapshots of past runs were counted as paper-level contracts
that "lost their guard":

```
unguarded_paper_level: 13   <- of which 3 are dated mission reports
```

Every consequence of that error is wrong in the expensive direction:

* `check()` cannot distinguish "a contract regressed" from "the baseline still
  lists an old report", so the only way to keep the gate green was to keep
  hand-editing the paper-level half of the baseline for work products that are
  *supposed* to be frozen;
* the audit's headline metric (`unguarded_paper_level`, and the "paper-level
  UNGUARDED" table in `docs/doc_claim_audit.md`) over-reported risk, which is
  the failure mode a ratchet is least able to tolerate.

## 2. Fix

`HISTORICAL_RE` is a substring match, so extending it is unsafe: a document that
merely *mentions* `m4290_` inside a name would flip. The fix is a separate,
anchored pattern plus an explicit exemption list:

```python
MISSION_RE = re.compile(r"^m\d{3,}[_-]")          # id leads the name: m4290_..., m3538-...
PAPER_LEVEL_EXTRA = frozenset({"m3538_propeller_openwater_uncertainty.md"})

def is_historical(name):
    if name in PAPER_LEVEL_EXTRA:   # contracts that only look like work products
        return False
    if name in HISTORICAL_EXTRA:    # existing English/Chinese singles
        return True
    return bool(HISTORICAL_RE.search(name) or MISSION_RE.search(name))
```

Design points:

1. **Anchored, separator-required.** `^m\d{3,}[_-]` needs the id at the start
   *and* a separator after the digits, so `mid3_report.md` stays paper-level.
2. **Freezing is not exempting.** A frozen report still reaches the ratchet
   through the `UNGUARDED` half: it must be registered in
   `baseline["unguarded"]`. The mission rule only moves a document out of the
   *paper-level* half.
3. **Exemptions are explicit.** `m3538_propeller_openwater_uncertainty.md` is a
   mission-prefixed name that is nevertheless a reader-facing benchmark note, so
   it keeps the paper-level gate. Making that a named frozenset (instead of a
   regex exception) means `test_paper_level_exemptions_survive_the_mission_prefix_rule`
   can assert the two invariants together.
4. **A naming trap is now documented.** Tier assignment is a plain filename
   match, so quoting a report's *file name* in a test docstring silently marks
   that report `PINNED` ("tier laundering"). The module docstring says so, and
   the new tests use synthetic `m9999_*.md` names to avoid it.

## 3. Baseline accounting

`baseline.json` is regenerated (`--write-baseline`) so both halves describe the
same classification the code now computes. Only the paper-level half moves:

| key | before | after |
|---|---:|---:|
| `unguarded` | 22 | 22 (unchanged — frozen reports stay registered) |
| `unguarded_paper_level` | 13 | 10 |
| removed from paper level | `m3977_pr109_merge_healing_report.md`, `m4283_pr186_ci_triage.md`, `m4290_ci_fix_report.md` | |

Fleet-wide audit after the change (regenerated programmatically, no hand-copied
digits — `docs/doc_claim_audit/audit.json` is the source):

| metric | value |
|---|---:|
| audited documents | 70 |
| `PINNED` | 26 |
| `ARTIFACT` | 22 |
| `UNGUARDED` | 22 |
| of which paper-level | 10 |
| numeric claim tokens | 7744 |
| guarded tokens | 6411 |
| numeric coverage | 82.8 % |

## 4. Tests

`tests/test_doc_claim_audit.py` gains three tests (12 total, all green):

* `test_mission_prefixed_reports_are_historical` — positive cases plus the
  negative control `mid3_report.md`; uses synthetic names so it cannot pin a real
  document.
* `test_paper_level_exemptions_survive_the_mission_prefix_rule` — every name in
  `PAPER_LEVEL_EXTRA` must match `MISSION_RE` (so the exemption is meaningful)
  and must *not* be historical.
* `test_historical_documents_are_still_registered_in_the_baseline` — the negative
  control for point 2 above: the set of unguarded *and* historical documents must
  be a subset of the registered baseline, so "freeze it" can never become
  "forget it".

Reproduce:

```bash
export PYTHONPATH=src
python -m tensorbem.benchmarks.doc_claim_audit --write-baseline --check
python -m pytest tests/test_doc_claim_audit.py -q
```

Honest notes / limitations:

* The classification is a heuristic over file names and test sources; it cannot
  see a number that a test pins through a different document, nor a report whose
  numbers are correct but superseded. It answers "is this file read?", not "is
  this file right?".
* `PAPER_LEVEL_EXTRA` is a manual list by design. Adding a name is a policy
  decision and should be reviewed, not bulk-generated.
* This change does not reduce the 22 unguarded documents; it only stops three of
  them from being miscounted. Guarding them is the ratchet's next round of work.