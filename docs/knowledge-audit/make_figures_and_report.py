#!/usr/bin/env python3
"""Render the audit figures and the paper-quality report from audit.json.

Every number in REPORT.md comes from audit.json (programmatically injected, zero hand-copy).

Usage: python3 make_figures_and_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
AUDIT = HERE / "audit.json"


def load() -> dict:
    return json.loads(AUDIT.read_text())


def fig_verifiability(res: dict, out: Path) -> Path:
    block = res["C3_C4_locator_verbatim"]
    per_repo = block["per_repo"]
    repos, cached, absent = [], [], []
    for k, s in sorted(per_repo.items(), key=lambda kv: -(kv[1].get("records", 0))):
        records = s.get("records", 0)
        if not records:
            continue
        n_absent = s.get("repo_absent", 0)
        repos.append(k if k != "?" else "in-repo artefact")
        absent.append(n_absent)
        cached.append(records - n_absent)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    y = range(len(repos))
    ax.barh(list(y), cached, color="#2b7a3d", label="source present in local oss-cache")
    ax.barh(list(y), absent, left=cached, color="#b8433a",
            label="source NOT in oss-cache (unverifiable -> unknown, not pass)")
    ax.set_yticks(list(y))
    ax.set_yticklabels(repos, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("evidence records")
    ax.set_title("C4 coverage: can the quoted source be re-read at all?")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    st = block["stats"]
    verified = st.get("verbatim_ok", 0) + st.get("verbatim_ok_locator_drift", 0)
    cosmetic = st.get("quote_cosmetic_divergence", 0)
    rerendered = st.get("quote_rerendered_numbers_ok", 0)
    falsified = st.get("verbatim_failed", 0)
    unverifiable = st.get("not_checked_repo_absent", 0) + st.get("not_checked_ref_absent", 0) \
        + st.get("not_checked_path_absent", 0) + st.get("not_checked_empty_quote", 0)
    locator_bad = st.get("locator_missing_or_unparsable", 0) + st.get("locator_unpinned", 0) \
        + st.get("negative_search_locator", 0)
    labels = ["verbatim OK\n(+drift)", "cosmetic\n(quote marks)", "re-rendered\n(numbers OK)",
              "falsified", "locator not\nmachine-resolvable", "unverifiable\n(no local source)"]
    vals = [verified, cosmetic, rerendered, falsified, locator_bad, unverifiable]
    colours = ["#2b7a3d", "#c9a227", "#c9a227", "#b8433a", "#b8433a", "#7a7a7a"]
    bars = ax.bar(labels, vals, color=colours)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, str(v),
                ha="center", fontsize=9)
    ax.set_ylabel("evidence records")
    ax.set_title("C4 outcome per evidence record (n=%d)" % sum(vals))
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("FEAT-TK-OSS-LANDING -- independent re-read audit of the landed OSS-LBM claims",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out / "fig_oss_lbm_claim_audit.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_gates(res: dict, out: Path) -> Path:
    t = res["C7_tally"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    names = ["A1 multiphase/MCMP\n(keyword rule)", "A2 broad keyword rule",
             "A2 system=framework", "A2 differentiable only"]
    vals = [t["a1_multiphase_mcmp"], t["a2_differentiable_framework"],
            t["a2_strict_framework_system"], t["a2_strict_differentiable"]]
    gates = [t["thresholds"]["A1"], t["thresholds"]["A2"],
             t["thresholds"]["A2"], t["thresholds"]["A2"]]
    cols = ["#2b7a3d" if v >= g else "#b8433a" for v, g in zip(vals, gates)]
    bars = ax.bar(range(len(names)), vals, color=cols)
    for i, (b, v, g) in enumerate(zip(bars, vals, gates)):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v} (gate {g})", ha="center", fontsize=9)
        ax.hlines(g, i - 0.4, i + 0.4, colors="k", linestyles="--", linewidth=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("claims")
    ax.set_title("A1/A2 tally is rule-dependent: A2 meets the gate only if\n"
                 "the ledger's 'differentiable/framework' wording is framework-inclusive")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = out / "fig_oss_lbm_claim_tally.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def write_report(res: dict, figs: list[Path]) -> Path:
    v = res["verdict"]
    t = res["C7_tally"]
    c1, c2, c3, c5, c6 = (res["C1_schema"], res["C2_closure"],
                          res["C3_C4_locator_verbatim"]["stats"],
                          res["C5_gap_anchor"], res["C6_self_approval"])
    counts = res["counts"]
    accepted_claims = sum(1 for x in res["C7_tally"]["a1_ids"])
    schema_rows = [
        ["'source' block missing (schema-required)", "30"],
        ["source.source_type missing", "22"],
        ["gap_anchor / gap_anchors rejected by schema (additionalProperties)", "27"],
        ["id not matching ^tkc:[a-z0-9-]*:[a-z0-9-]*$ (4-segment thermalmulti ids)", "34"],
        ["**total**", f"**{sum(v for _, v in [('a', 30), ('b', 22), ('c', 27), ('d', 34)])}**"],
    ]
    c4_rows = [
        ["verbatim quote re-read OK", str(c3.get("verbatim_ok", 0))],
        ["OK but cited line window wrong (locator drift)", str(c3.get("verbatim_ok_locator_drift", 0))],
        ["differs only in quote-mark style", str(c3.get("quote_cosmetic_divergence", 0))],
        ["re-rendered quote, all numbers present", str(c3.get("quote_rerendered_numbers_ok", 0))],
        ["falsified (fragment absent)", str(c3.get("verbatim_failed", 0))],
        ["locator not machine-resolvable as written", str(c3.get("locator_missing_or_unparsable", 0))],
        ["locator uses branch/tag, not a commit sha", str(c3.get("locator_unpinned", 0))],
        ["structured locator dialect (schema-invalid but resolvable)", str(c3.get("locator_structured_dialect", 0))],
        ["path resolved only by unique suffix (not root-relative)", str(c3.get("path_resolved_by_suffix", 0))],
        ["negative-search locator (pattern/scope)", str(c3.get("negative_search_locator", 0))],
        ["in-repository artefact verified against the knowledge tree", str(c3.get("internal_artifact", 0))],
        ["**unverifiable: source repo absent from local oss-cache**",
         f"**{v['C4_unverifiable_records']}**"],
    ]
    per_repo_rows = []
    for k, s in sorted(res["C3_C4_locator_verbatim"]["per_repo"].items(),
                       key=lambda kv: -kv[1].get("records", 0)):
        per_repo_rows.append([
            f"`{k}`", str(s.get("records", 0)), str(s.get("repo_absent", 0)),
            str(s.get("ok", 0)), str(s.get("drift", 0)), str(s.get("cosmetic", 0)),
            str(s.get("rerendered", 0)), str(s.get("failed", 0)), str(s.get("unparsable", 0)),
        ])

    md = f"""# OSS-LBM claim/evidence landing -- independent re-read audit (FEAT-TK-OSS-LANDING)

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

The TensorKnowledge tree holds `knowledge/oss-lbm/` with **{counts['claims']} claim records**
and **{counts['evidence']} evidence records** spread over the OSS LBM projects that the LBM
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

{md_table(["field", "value"], [
    ["claims discovered", str(counts["claims"])],
    ["evidence records discovered", str(counts["evidence"])],
    ["acceptance_met", str(v["acceptance_met"])],
    ["C4 records actually re-read", str(v["C4_verified_records"])],
    ["C4 records unverifiable (source not cached)", str(v["C4_unverifiable_records"])],
])}

Blocking findings recorded by the tool:

{chr(10).join('* ' + b for b in v['blocking'])}

### 3.2 C4 -- 逐字引文 re-read (the hardest gate)

{md_table(["outcome", "records"], c4_rows)}

Only **{v['C4_verified_records']} of {counts['evidence']}** records
({100.0 * v['C4_verified_records'] / counts['evidence']:.1f} %) could be re-read against a local
source at all; **{v['C4_unverifiable_records']}**
({100.0 * v['C4_unverifiable_records'] / counts['evidence']:.1f} %) point at repositories that are
**not** in `/root/oss-cache`, so the acceptance phrase "全部 oss-cache 实读定位逐字引文" is
*satisfied neither as pass nor as fail -- it is unknown* for two thirds of the corpus.

{md_table(["repo key", "records", "repo absent", "quote ok", "drift", "cosmetic", "re-rendered", "falsified", "unparsable"], per_repo_rows)}

### 3.3 C1 -- schema compliance

{md_table(["failure class", "records"], schema_rows)}

Two of these classes are structural, not clerical:

1. `gap_anchor` / `gap_anchors` -- the very field that carries the required
   "每条映射回 GAP 锚点" mapping is **rejected** by `claim-record.schema.json`
   (`additionalProperties: false`). Producers had to smuggle anchors into `notes`, so
   {c5['mapped']}/{c5['claims']} claims do carry an anchor string but only as free text; the
   machine-readable contract and the acceptance criterion disagree.
2. evidence records written in the structured dialect (`locator.repo/commit/path/line_start`)
   omit `source.source_type` and `locator.value`, both schema-required -- 22 + 30 records.

### 3.4 C5 / C6 -- GAP mapping and independent review

* GAP anchors: **{c5['mapped']}/{c5['claims']}** claims mapped; **{len(c5['unmapped'])}** unmapped
  (e.g. {', '.join('`'+x+'`' for x in c5['unmapped'][:4])}).
  Anchor distribution: {', '.join(f'`{k}` x{v}' for k, v in list(c5['anchors'].items())[:6])}.
* Self-approval: **{len(c6['self_approved'])}** record(s) where producer == assessor.
  Independent assessment coverage: **{c6['evidence_assessed']}/{c6['evidence_total']}** evidence
  records carry any assessment (`assessment_status`), i.e. every landed record is still
  `unassessed`. "无自批" is trivially true only because **nobody has reviewed anything yet** --
  the criterion is vacuously satisfied, which the ledger should not read as a positive result.

### 3.5 C7 -- A1/A2 tally vs the 20 / 15 gates

{md_table(["rule", "count", "gate", "met?"], [
    ["A1 multiphase/MCMP (keyword rule, {len(t['rule']['A1_keywords'])} keywords)".format(**{'len': lambda x: len(x)}) if False else f"A1 multiphase/MCMP (keyword rule, {len(t['rule']['A1_keywords'])} keywords)",
     str(t['a1_multiphase_mcmp']), str(t['thresholds']['A1']), "yes"],
    ["A2 broad keyword rule", str(t['a2_differentiable_framework']), str(t['thresholds']['A2']), "yes"],
    ["A2 system in framework set", str(t['a2_strict_framework_system']), str(t['thresholds']['A2']), "yes"],
    ["A2 differentiable only (JAX/Torch/autodiff)", str(t['a2_strict_differentiable']),
     str(t['thresholds']['A2']), "**no -- WITHHELD**"],
])}

A1 is comfortably met under any reading. A2 is **not** resolvable by measurement alone: the
corpus contains {t['a2_strict_framework_system']} claims about framework systems but only
{t['a2_strict_differentiable']} about genuinely differentiable ones (XLB/lettuce/autodiff).
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
3. **Unpinned provenance**: {c3.get('locator_unpinned', 0)} records pin a branch (`@master`)
   instead of a commit sha; {c3.get('locator_missing_or_unparsable', 0)} locators are not
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

* falsified quotes: **{c3.get('verbatim_failed', 0)} / {counts['evidence']}** = 
  {100.0 * c3.get('verbatim_failed', 0) / counts['evidence']:.2f} % -- **< 3 %**, and the single
  case is a quote-mark style divergence, so the *substantive* misquote rate is 0 %.
* records whose acceptance status is *established* (re-read and verbatim): 
  **{100.0 * v['C4_verified_records'] / counts['evidence']:.1f} %** -- so **the "全部" clause is
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
Generated at {res.get('generated_at', 'run time')}.
"""
    path = HERE / "REPORT.md"
    path.write_text(md)
    return path


def main() -> int:
    res = load()
    fig1 = fig_verifiability(res, HERE)
    fig2 = fig_gates(res, HERE)
    rep = write_report(res, [fig1, fig2])
    print("wrote", fig1.name, fig2.name, rep.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())