# Remediation spec -- FEAT-TK-OSS-LANDING (from the m21572 re-read audit)

Scope: TensorKnowledge (`knowledge/oss-lbm/**`, `schemas/*.json`, `tools/validate_*`).
This file is a hand-off: the audit was run read-only from a TensorBEM worktree, so no
TensorKnowledge file was modified. Ordered by blocking severity.

## R1 -- make "全部 oss-cache 实读定位逐字引文" actually checkable  (blocking)

135 of 197 evidence records cite a repository that is
**not** present in the local oss-cache. Missing (from the locator census):

```
lbm-principles-practice/code   openlb            MF-LBM
fingering_dynamics             ThermalMultiphaseOpenLB
ludwig-cf/ludwig               lettuce           XLB
lbmpy
```

Actions
1. Fetch each of the 9 repositories into the shared oss-cache **at the commit sha named in the
   evidence**, not at HEAD (`git init && git fetch --depth 1 origin <sha>`).
2. Add a cache manifest (`knowledge/oss-lbm/provenance/cache_manifest.json`: repo -> sha ->
   fetch date -> bytes) so an auditor can tell "absent" from "changed".
3. Re-run `audit_oss_lbm_claims.py` until `verdict.C4_unverifiable_records == 0`.
   Only then can the ledger read the 逐字 clause as *checked*, today it is *unknown*.

## R2 -- put the GAP anchor under contract  (blocking)

`schemas/claim-record.schema.json` has `additionalProperties: false` and no `gap_anchor`
property, so the 27 claims that carry `gap_anchor`/`gap_anchors` **fail their own schema** while
satisfying the acceptance criterion. Either:
* add to claim-record.schema.json
  ```json
  "gap_anchor":  {"type": "string", "pattern": "^GAP-[A-Z0-9][A-Z0-9-]*$"},
  "gap_anchors": {"type": "array", "items": {"type": "string", "pattern": "^GAP-[A-Z0-9][A-Z0-9-]*$"},
                   "minItems": 1, "uniqueItems": true}
  ```
  and require at least one of them (anyOf), or
* forbid the fields in the schema and normalise every anchor into `notes` with a fixed
  `GAP anchor: <slug>` prefix that the validator greps.
Pick one; today the corpus mixes both and the validator cannot be green either way.

## R3 -- evidence records: required blocks and id grammar  (blocking)

* 30 records have no `source` block at all; 22 have a
  `source` without the required `source_type`/`citation` pair (structured dialect
  `source.repo/commit/branch`). Normalise to `source_type: "other"` + `citation`, or extend the
  schema -- but do not leave records that the schema rejects.
* 34 records in the `thermalmulti` module use 4-segment ids
  (`tke:oss-lbm:thermalmulti:<slug>`, `tkc:oss-lbm:thermalmulti:<slug>`); the schema allows 3.
  Fold the module into the slug (`tke:oss-lbm:thermalmulti-latent-heat-src`) and update every
  `claim_refs`/`evidence_refs` pointer in the same commit.

## R4 -- locator normalisation  (blocking for the 逐字 gate, clerical effort)

| defect | records | fix |
|---|---|---|
| not machine-resolvable as written (multi-target `A + B`, `path:Section Heading`, empty value) | 35 | one locator per evidence record; `path` relative to the repo root; line window or symbol anchor |
| pinned to a branch (`@master`) instead of a sha | 9 | replace with the resolved sha |
| structured dialect (`locator.repo/commit/path/line_start`) | 20 | emit `locator.value = "<repo>@<sha> <path>:<l1>-<l2>"` (generator change) |
| path resolved only by unique suffix (not root-relative) | 5 | prefix the package directory (`analysis/x.py` -> `pylbm/analysis/x.py`) |
| quote not literal (quote-mark style / re-rendered) | 7 | copy the literal source text; keep computed summaries in `notes` |

## R5 -- independent assessment is missing, not passed  (process)

`assessment_status` is `unassessed` on 0/197 evidence records and no claim
carries an assessor. "无自批" is therefore vacuous: a second agent must review the claims and
set `assessment_status`, otherwise the criterion says nothing. Recommended: rotate producer /
assessor agent ids and assert `captured_by != assessed_by` in the validator (the audit tool
already does this in C6).

## R6 -- pin the A2 definition  (ledger wording)

`可微/框架` is ambiguous and the gate flips on the reading:

* differentiable only (JAX/PyTorch/autodiff): **9** claims < 15 -> not met
* system is a framework (xlb/lettuce/lbmpy/ludwig/pylbm/sailfish/openlb): **47** claims >= 15 -> met

Owner action: write the definition into the ledger `acceptance` text; until then the audit
reports A2 as WITHHELD rather than pass.

## R7 -- markdown dialect

12 `CUDAPF-*` claims live as one markdown file each with the quotation inline; the JSON
contract cannot see any evidence for them. Either convert to claim+evidence JSON, or declare the
markdown dialect in the schema and teach the validator to read it.

## Verification of the remediation

```bash
python3 docs/knowledge-audit/audit_oss_lbm_claims.py \
    --knowledge-root <TensorKnowledge> --oss-cache <oss-cache> --out /tmp/recheck
# close the anchor only when: verdict.green == true
#                            and verdict.C4_unverifiable_records == 0
```
