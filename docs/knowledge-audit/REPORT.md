# OSS-LBM claim/evidence landing -- independent re-read audit (FEAT-TK-OSS-LANDING)

**Anchor**: FEAT-TK-OSS-LANDING -- *LBM 开源知识按四契约落库 TensorKnowledge（实读复认非搬运）*
**Acceptance under test**: `schema 全过 + 全部 oss-cache 实读定位逐字引文 + 自带校验器全绿 +
无自批 + 每条映射回 GAP 锚点`, plus `A1 >= 20` multiphase/MCMP and `A2 >= 15`
differentiable/framework claims.

**Disposition: REQUEST CHANGES -- the anchor cannot be closed as "green" on the current
artefacts.**  No claim was found to be *substantively* misquoted; the blockers are contract
and coverage defects, not fabrication.  Details and the exact reproduction command follow.

> Independence limitation (state it up front): this audit was produced by a different agent
> session than the claim producers, but it uses **the same local oss-cache** as the producers
> and **cannot** reach the 9 source repositories that are not cached locally. For those records
> the verdict is *unknown*, which this report never counts as a pass.

## 1. Case -- what is being audited

The TensorKnowledge tree holds `knowledge/oss-lbm/` with **162 claim records**
and **197 evidence records** spread over the OSS LBM projects that the LBM
domain is meant to mine (pyLBM, Maarten-vd-Sande/lbm free-surface, ludwig, openlb,
ThermalMultiphaseOpenLB, MF-LBM, fingering_dynamics, lettuce, XLB, lbmpy, lbm-principles-practice
code, CudaPhaseField). Two dialects coexist: JSON claim/evidence bundles (the four-contract
form) and 12 one-file-per-claim markdown records (`CUDAPF-*`).

The acceptance criterion is unusually strict because it forbids *搬运*: a claim only counts if
a reviewer can **re-open the cited source at the cited commit and read the quoted text there**.
That is a mechanical property, so it can be audited mechanically -- which is what this work does.

## 2. Method

`docs/knowledge-audit/audit_oss_lbm_claims.py` (stdlib only, `--knowledge-root`/`--oss-cache`
configurable, no domain content hardcoded) re-reads every landed record and tries to falsify
seven checks:

* **C1 schema** -- validate each claim/evidence record against `schemas/claim-record.schema.json`
  and `schemas/evidence-record.schema.json` with `jsonschema`.
* **C2 closure** -- `claim.evidence_refs` must resolve, `evidence.claim_refs` must resolve back;
  neither orphans nor dangling references may exist.
* **C3 locator** -- the locator must be a machine-resolvable
  `<repo>@<commit> <path>:<lines|symbol>`; a branch/tag, an empty `value`, a structured
  `repo/path/line_start` dialect, or a human multi-target locator is flagged with its own
  severity.
* **C4 verbatim** -- for every evidence record whose source repo *is* in the oss-cache, the file
  is read with `git show <commit>:<path>` at the **cited commit** (never the working tree), the
  cited line window is cut, and every quoted fragment (the quote is split on `...`/`…` so
  elisions are allowed) must appear after whitespace normalisation. Falling back through:
  window -> whole file (locator drift) -> quote-mark normalisation (cosmetic) ->
  numeric cross-check (re-rendered quote); anything else is a falsification.
  Records whose repository is not cached are counted **unverifiable**, never passed.
* **C5 gap anchor** -- each claim must map to a `GAP-LBM-*` anchor (field or notes).
* **C6 self-approval** -- producer must not equal assessor; assessment coverage is counted.
* **C7 tally** -- A1/A2 counts under three explicitly stated rules, because "可微/框架" admits
  more than one reading.

Verification of the verifier: `python3 audit_oss_lbm_claims.py --self-test` builds a throwaway
git repo plus fabricated claim/evidence bundles (a genuine quote, a drift case, a paraphrase,
an empty locator, a branch-pinned locator, a dangling ref, a claim without a GAP anchor) and
asserts that each defect is caught and that acceptance can never go green on that fixture
(`SELF-TEST PASS`). The checks are therefore asserted as invariants, not merely executed.

## 3. Results

### 3.1 Overall verdict

| field | value |
|---|---|
| claims discovered | 162 |
| evidence records discovered | 197 |
| acceptance_met | False |
| C4 records actually re-read | 18 |
| C4 records unverifiable (source not cached) | 135 |

Blocking findings recorded by the tool:

* C1 schema: 113 record(s) fail schema
* C2 closure: 0 dangling, 12 claim(s) with no evidence
* C5 gap anchor: 38 claim(s) unmapped
* C7 A2 is WITHHELD, not met: differentiable-only claims = 9 (< gate 15) while framework-inclusive claims = 47 (>= gate); the ledger wording '可微/框架' is ambiguous, so the gate cannot be declared met

### 3.2 C4 -- 逐字引文 re-read (the hardest gate)

| outcome | records |
|---|---|
| verbatim quote re-read OK | 16 |
| OK but cited line window wrong (locator drift) | 2 |
| differs only in quote-mark style | 1 |
| re-rendered quote, all numbers present | 6 |
| falsified (fragment absent) | 0 |
| locator not machine-resolvable as written | 35 |
| locator uses branch/tag, not a commit sha | 9 |
| structured locator dialect (schema-invalid but resolvable) | 20 |
| path resolved only by unique suffix (not root-relative) | 5 |
| negative-search locator (pattern/scope) | 2 |
| in-repository artefact verified against the knowledge tree | 4 |
| **unverifiable: source repo absent from local oss-cache** | **135** |

Only **18 of 197** records
(9.1 %) could be re-read against a local
source at all; **135**
(68.5 %) point at repositories that are
**not** in `/root/oss-cache`, so the acceptance phrase "全部 oss-cache 实读定位逐字引文" is
*satisfied neither as pass nor as fail -- it is unknown* for two thirds of the corpus.

| repo key | records | repo absent | quote ok | drift | cosmetic | re-rendered | falsified | unparsable |
|---|---|---|---|---|---|---|---|---|
| `lbm-principles-practice/code` | 36 | 36 | 0 | 0 | 0 | 0 | 0 | 0 |
| `?` | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 30 |
| `fingering_dynamics` | 23 | 23 | 0 | 0 | 0 | 0 | 0 | 0 |
| `ludwig-cf/ludwig` | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| `ThermalMultiphaseOpenLB` | 17 | 17 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Maarten-vd-Sande/lbm` | 16 | 0 | 16 | 0 | 0 | 0 | 0 | 0 |
| `openlb` | 13 | 13 | 0 | 0 | 0 | 0 | 0 | 0 |
| `MF-LBM` | 10 | 5 | 0 | 0 | 0 | 0 | 0 | 5 |
| `lbm-principles-practice` | 9 | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| `pylbm` | 5 | 0 | 0 | 2 | 1 | 2 | 0 | 0 |
| `lettuce` | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| `xlb` | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| `lbmpy` | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| `runs/oss-lbm-kruger-8676/validation_summary.json#port_chapter5` | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| `runs/oss-lbm-kruger-8676/validation_summary.json#taylor_green` | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| `runs/oss-lbm-kruger-8676/validation_summary.json#young_laplace` | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| `runs/oss-lbm-kruger-8676/validation_summary.json#ib_lbm_poiseuille` | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |

### 3.3 C1 -- schema compliance

| failure class | records |
|---|---|
| 'source' block missing (schema-required) | 30 |
| source.source_type missing | 22 |
| gap_anchor / gap_anchors rejected by schema (additionalProperties) | 27 |
| id not matching ^tkc:[a-z0-9-]*:[a-z0-9-]*$ (4-segment thermalmulti ids) | 34 |
| **total** | **113** |

Two of these classes are structural, not clerical:

1. `gap_anchor` / `gap_anchors` -- the very field that carries the required
   "每条映射回 GAP 锚点" mapping is **rejected** by `claim-record.schema.json`
   (`additionalProperties: false`). Producers had to smuggle anchors into `notes`, so
   124/162 claims do carry an anchor string but only as free text; the
   machine-readable contract and the acceptance criterion disagree.
2. evidence records written in the structured dialect (`locator.repo/commit/path/line_start`)
   omit `source.source_type` and `locator.value`, both schema-required -- 22 + 30 records.

### 3.4 C5 / C6 -- GAP mapping and independent review

* GAP anchors: **124/162** claims mapped; **38** unmapped
  (e.g. `tkc:oss-lbm:pylbm-equiv-eq`, `tkc:oss-lbm:pylbm-stability`, `tkc:oss-lbm:pylbm-bfl-bounce-back`, `tkc:oss-lbm:pylbm-linear-bb-family`).
  Anchor distribution: `GAP-LBM-MCMP-POISEUILLE` x27, `GAP-LBM-THERMAL-MULTIPHASE` x17, `GAP-LBM-BENCH-METHODOLOGY` x16, `GAP-LBM-INDUSTRIAL-ENG` x13, `GAP-LBM-FREESURFACE` x12, `GAP-LBM-PHASEFIELD-3PH` x12.
* Self-approval: **0** record(s) where producer == assessor.
  Independent assessment coverage: **0/197** evidence
  records carry any assessment (`assessment_status`), i.e. every landed record is still
  `unassessed`. "无自批" is trivially true only because **nobody has reviewed anything yet** --
  the criterion is vacuously satisfied, which the ledger should not read as a positive result.

### 3.5 C7 -- A1/A2 tally vs the 20 / 15 gates

| rule | count | gate | met? |
|---|---|---|---|
| A1 multiphase/MCMP (keyword rule, 31 keywords) | 85 | 20 | yes |
| A2 broad keyword rule | 63 | 15 | yes |
| A2 system in framework set | 47 | 15 | yes |
| A2 differentiable only (JAX/Torch/autodiff) | 9 | 15 | **no -- WITHHELD** |

A1 is comfortably met under any reading. A2 is **not** resolvable by measurement alone: the
corpus contains 47 claims about framework systems but only
9 about genuinely differentiable ones (XLB/lettuce/autodiff).
The ledger must pin the definition before the gate can be declared met.

### 3.6 Figures

* `fig_oss_lbm_claim_audit.png` -- C4 coverage per source repository, and the outcome split of
  the 197 evidence records.
* `fig_oss_lbm_claim_tally.png` -- the A1/A2 tally under the four candidate rules.

## 4. Notes, limitations, scope

1. **Unverifiable != pass.** 135 of 197 evidence records cite a repository that is not in the
   local oss-cache. Confirming them requires fetching those 9 repositories
   (lbm-principles-practice/code, openlb, MF-LBM, fingering_dynamics, ThermalMultiphaseOpenLB,
   ludwig, lettuce, XLB, lbmpy) at the pinned commits -- that is the single highest-value
   remediation and the reason this anchor should stay `in_progress`.
2. **12 markdown claims (`CUDAPF-*`)** carry their quotations inside the `.md` file and have no
   `evidence_refs`; C2 lists them as "claims without evidence" because the JSON contract cannot
   see inline evidence. That is a *dialect* finding, not a fabrication finding -- the report
   does not count them as defects, but the four-contract goal is not met for them either.
3. **Unpinned provenance**: 9 records pin a branch (`@master`)
   instead of a commit sha; 35 locators are not
   machine-resolvable as written (multi-target `A + B` locators, section-title `path:Some Heading`
   forms, empty `value`).
4. The audit is **read-only** with respect to TensorKnowledge; findings are reported, not patched.
   Remediation is scoped to a TensorKnowledge-domain 单 (see the filed mission).
5. Chekcs not run: no attempt was made to re-derive the *scientific* content of the claims
   (e.g. whether pylbm's equivalent-equation expansion is really second order); this audit
   verifies the evidence chain, not the physics.

## 5. Error judgement (vs the owner's 3 % / 15 % line)

This is a provenance audit, so the "error" metric is a **quote-fidelity rate**, not a physics
residual. Against the acceptance wording "全部 ... 逐字引文":

* falsified quotes: **0 / 197** = 
  0.00 % -- **< 3 %**, and the single
  case is a quote-mark style divergence, so the *substantive* misquote rate is 0 %.
* records whose acceptance status is *established* (re-read and verbatim): 
  **9.1 %** -- so **the "全部" clause is
  NOT met (< 3 % cannot even be evaluated for 68.5 % of the corpus)**.

Answer to the explicit question: **no, the acceptance gate < 3 % is not established** -- not
because the quotes are wrong, but because 68.5 % of them cannot be checked without fetching the
missing repositories. That part is an honest FAIL / WITHHELD, not a pass.

## 6. Reproduction

```bash
cd docs/knowledge-audit
python3 audit_oss_lbm_claims.py --self-test                 # verifier invariants
python3 audit_oss_lbm_claims.py --out .                     # -> audit.json + audit.md
python3 make_figures_and_report.py                          # -> figures + this report
```
Artifacts: `audit.json` (machine verdict), `audit.md` (generated summary), this file, two PNGs.
Generated at run time.
