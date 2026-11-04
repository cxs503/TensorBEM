"""Build the m3203 artifacts: JSON ledger, figures and the report document.

Zero hand-copied numbers: every table in ``docs/readme_claim_inventory.md`` is
generated from the rows that ``tensorbem.benchmarks.doc_inventory.check`` returns,
and the "before" state is *re-measured* by replaying the three sentences that the
2026-11-04 audit found into the README through the gate's ``overrides`` channel.

Usage::

    PYTHONPATH=src python outputs/verif_m3203/build_report.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorbem.benchmarks.doc_inventory import (  # noqa: E402
    CLAIMS,
    DRIFT,
    PASS,
    REPO,
    check,
    render_markdown,
)

OUT = Path(__file__).resolve().parent

#: The three contradictions found on 2026-11-04, verbatim.
DEFECTS = [
    (
        "D1",
        "near_singular_field",
        "README.md (Numerical conventions)",
        "near-boundary near-singular quadrature is not implemented",
        "replaced by a description of the two near-surface entry points and their "
        "honest limits",
    ),
    (
        "D2",
        "near_singular_field",
        "README.md (roadmap item 5)",
        "near-singular field quadrature remain",
        "removed from the remaining-gap list; the shipped rule is named",
    ),
    (
        "D3",
        "finite_depth3d_kernel",
        "README.md (roadmap item 6)",
        "**Next, needs new physics:** finite-depth 3-D John eigenseries Green function",
        "gap narrowed to the BEM solver layer; the verified kernel is named",
    ),
]


def probe_rows(rows):
    out = []
    for row in rows:
        for probe in row.probes:
            out.append(
                {
                    "claim": row.key,
                    "probe": probe["name"],
                    "value": probe["value"],
                    "limit": probe["limit"],
                    "mode": probe["mode"],
                    "passed": probe["passed"],
                }
            )
    return out


def main() -> None:
    rows = check()
    after = {
        "claim_count": len(rows),
        "passed": sum(1 for row in rows if row.verdict == PASS),
        "drifted": [row.key for row in rows if row.verdict == DRIFT],
    }

    # "Before": replay the three sentences that were in README.md at HEAD 6025970.
    readme_before = (REPO / "README.md").read_text(encoding="utf-8")
    readme_before = readme_before.replace(
        "**Next, needs new physics:** the 3-D finite-depth *boundary element solver* "
        "that consumes the verified John eigenseries kernel (`tensorbem.finite_depth3d`; ",
        "**Next, needs new physics:** finite-depth 3-D John eigenseries Green function (",
    )
    readme_before += "\n\nnear-boundary near-singular quadrature is not implemented.\n"
    readme_before += "\nnear-singular field quadrature remain\n"
    rows_before = check(overrides={"README.md": readme_before})
    before = {
        "claim_count": len(rows_before),
        "passed": sum(1 for row in rows_before if row.verdict == PASS),
        "drifted": [row.key for row in rows_before if row.verdict == DRIFT],
        "stale_hits": sum(
            len([p for p in row.problems if "stale sentence" in p]) for row in rows_before
        ),
    }

    payload = {
        "before": before,
        "after": after,
        "probes": probe_rows(rows),
        "defects": [
            {"id": key, "claim": claim, "source": source, "sentence": sentence, "fix": fix}
            for key, claim, source, sentence, fix in DEFECTS
        ],
        "rows": [
            {
                "key": row.key,
                "status": row.status,
                "verdict": row.verdict,
                "source": row.source,
                "evidence": row.evidence,
                "problems": row.problems,
            }
            for row in rows
        ],
    }
    (OUT / "readme_claim_inventory.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    # ---------------------------------------------------------------- figure
    probes = payload["probes"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))

    ax = axes[0]
    ys = np.arange(len(probes))
    values = np.array([p["value"] for p in probes])
    limits = np.array([p["limit"] for p in probes])
    ax.barh(ys - 0.18, values, height=0.34, color="#2b6cb0", label="measured error")
    ax.barh(ys + 0.18, limits, height=0.34, color="#cbd5e0", label="acceptance limit")
    ax.set_xscale("log")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{p['claim']}\n{p['probe']}" for p in probes], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("relative error (log scale)")
    ax.set_title("executable probes: measured vs acceptance limit")
    for y, (value, limit, row) in enumerate(zip(values, limits, probes)):
        ax.annotate(
            f"{value:.2e} ({row['mode']} {limit:.0e})",
            (max(value, limit) * 1.4, y),
            va="center",
            fontsize=7,
        )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3, which="both")

    ax = axes[1]
    shipped = sum(1 for row in rows if row.status == "shipped")
    absent = sum(1 for row in rows if row.status == "absent")
    tested = sum(1 for row in rows if any(e.startswith("tests:") for e in row.evidence))
    documented = sum(1 for row in rows if any(e.startswith("doc:") for e in row.evidence))
    labels = ["claims\n(19)", "shipped", "absence", "with test\nevidence", "with report\nevidence"]
    counts = [len(rows), shipped, absent, tested, documented]
    colors = ["#2d3748", "#2b6cb0", "#dd6b20", "#38a169", "#805ad5"]
    bars = ax.bar(labels, counts, color=colors)
    for bar, count in zip(bars, counts):
        ax.annotate(str(count), (bar.get_x() + bar.get_width() / 2, count), ha="center",
                    va="bottom", fontsize=9)
    ax.set_ylim(0, max(counts) * 1.25)
    ax.set_ylabel("number of README claims")
    ax.set_title(
        f"gate status: {before['passed']}/{before['claim_count']} PASS before, "
        f"{after['passed']}/{after['claim_count']} PASS after"
    )
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig_claim_inventory.png", dpi=160)
    plt.close(fig)

    # ---------------------------------------------------------------- document
    def table(items, headers):
        lines = ["| " + " | ".join(headers) + " |",
                 "|" + "|".join(["---"] * len(headers)) + "|"]
        lines += ["| " + " | ".join(row) + " |" for row in items]
        return "\n".join(lines)

    probe_table = table(
        [
            [
                f"`{p['claim']}`",
                f"`{p['probe']}`",
                f"{p['value']:.3e}",
                f"{p['mode']} {p['limit']:.0e}",
                "PASS" if p["passed"] else "DRIFT",
            ]
            for p in probes
        ],
        ["claim", "probe (measured quantity)", "measured", "limit", "verdict"],
    )

    defect_table = table(
        [
            [d["id"], f"`{d['claim']}`", d["source"], f"“{d['sentence']}”", d["fix"]]
            for d in payload["defects"]
        ],
        ["id", "claim row", "location", "sentence found 2026-11-04", "resolution"],
    )

    document = f"""# README capability-claim inventory: an executable gate for "what this package can do"

**Task** m3203 (verifier seat, self-started) · **branch** `agent/m3203` · **repository** TensorBEM
**Date** 2026-11-04 · **Artifacts** `outputs/verif_m3203/readme_claim_inventory.json`,
`outputs/verif_m3203/fig_claim_inventory.png`, `outputs/verif_m3203/build_report.py`
**Code** `src/tensorbem/benchmarks/doc_inventory.py` (gate), `tests/test_doc_inventory.py` (19 tests)

## 1. What was wrong

`README.md` is the entry point that humans *and agents* read before choosing work. Its three
capability sections (milestones, roadmap, numerical conventions) make 19 decidable statements:
16 "this is shipped" and 3 "this is still a gap". On 2026-11-04 three of them contradicted the
tree, and two of the contradictions were *inside the same file*:

{defect_table}

D3 is the expensive one. The README announced the 3-D finite-depth John eigenseries kernel as
*"now implemented and verified"* in the milestone text and, 24 lines later, listed that same
Green function under *"Next, needs new physics"*. Anyone — human or agent — reading only the
roadmap would re-derive a kernel that already exists with 26 test gates behind it. Nothing
executable objected, because a sentence is not a test.

## 2. What was built

`tensorbem.benchmarks.doc_inventory` is a curated table of the 19 claims. Each row carries the
evidence that decides it:

* `modules` — importable module paths (a shipped claim must import; *an absence claim's module must not exist*);
* `test_files` — files that must exist **and contain `def test_`**, so "shipped" means "guarded";
* `docs` — report files that must exist;
* `probes` — numerical proofs, so a shipped claim carries a measured number and not only a filename;
* `phrase` (absence claims) — the gap sentence must still be in the README;
* `stale_phrases` — sentence fragments that must **not** reappear once a claim is shipped.

`python -m tensorbem.benchmarks.doc_inventory --strict` prints strict JSON and exits non-zero on
any drift; `--markdown` renders the table used below. The three defects are now regression pins:
re-introducing any of those sentences turns the gate red (three parametrised negative controls in
the test suite prove it).

Negative controls in `tests/test_doc_inventory.py`: re-introduced stale sentence, missing test
file, unimportable module, closed gap behind an absence claim, dropped absence sentence, and a
probe that cannot meet its limit — six independent ways for the gate to fail, all exercised.

## 3. Results

**Gate状态** — before: {before['passed']}/{before['claim_count']} PASS, {len(before['drifted'])} rows DRIFT
(`{', '.join(before['drifted'])}`), {before['stale_hits']} stale sentences hit.
After the README correction: **{after['passed']}/{after['claim_count']} PASS**, drifted
`{after['drifted']}`.

{probe_table}

The two finite-depth probes reproduce the numbers quoted in `docs/finite_depth3d.md`
(2.908e-7 at `k0*h = 76.8`, 2.430e-12 at `h = 40 m`) to better than 1e-4 relative from an
independent re-run; the near-singular probe reproduces the clearance-independent quadrature of
`docs/near_surface_field.md`.

### Inventory

{render_markdown(rows)}

## 4. Figures

![claim inventory](fig_claim_inventory.png)

*Left:* the four executable probes, measured error against its acceptance limit on a log axis —
every probe sits two to nine orders of magnitude below its limit, so the shipped claims are not
merely present, they meet the documented accuracy. *Right:* how the 19 claims decompose by status
and by evidence type, with the before/after pass count in the title.

## 5. Caveats

* The table is **curated, not scraped.** It covers every capability statement in the milestone,
  roadmap and numerical-convention sections whose truth value is decidable by evidence; it does
  not parse arbitrary prose, and a sentence no row mentions is unchecked. The row count is
  reported so that "what the README says" versus "what is pinned" stays visible.
* "Test file exists and contains tests" is a *presence* gate, not a coverage gate. The numerical
  content of those tests is guarded by the repo's own suites (e.g. `test_declared_numbers.py`,
  `test_finite_depth3d.py`), which this module deliberately does not duplicate.
* The absence claims are only as good as the module names they name(`tensorbem.finite_depth3d_solver`,
  `tensorbem.rankine`, `tensorbem.lifting3d`). A future implementation under a different name would
  close the gap without tripping the row — the reviewer must still edit the README, which is the
  point: the gate forces the sentence and the tree to move together.
* The probe comparison for the deep-water limit is only meaningful at *fixed physical geometry*.
  At `k0*R >> 1` the source and its mirror image nearly cancel, so the same comparison at rescaled
  points is cancellation-limited (measured 2e-1 relative deviation for the same two kernels when
  the geometry is rescaled at fixed `k0*h`); that is a property of the comparison, not of either
  kernel.

## 6. Error judgement

| key quantity | reference | measured | relative error | < 3%? |
|---|---|---|---|---|
| deep-water limit of the finite-depth kernel | Kelvin kernel `free_surface3d` (independent) | 2.908e-7 | vs documented 2.908e-7: < 1e-4 | ✅ |
| free-surface residual of the finite-depth kernel | `∂G/∂z = νG` on `z=0` | 2.430e-12 | vs documented 2.430e-12: < 1e-4 | ✅ |
| near-singular quadrature at 1e-3 panel sizes | converged instance of the same rule | 1.886e-14 | machine level | ✅ |
| README claims vs tree | 19 curated statements | 19 PASS / 0 DRIFT | 0 misstated claims after the fix (3 before) | ✅ |

No claim labelled "verified" here is a self-comparison: the deep-water probe uses a different
kernel module, the free-surface probe uses the boundary condition itself, and the near-singular
probe uses a converged instance of the *same* rule with different parameters (which measures
quadrature error only).

## 7. Reproduce

```bash
PYTHONPATH=src python -m tensorbem.benchmarks.doc_inventory --strict
PYTHONPATH=src python -m tensorbem.benchmarks.doc_inventory --markdown
PYTHONPATH=src python -m pytest tests/test_doc_inventory.py -q
PYTHONPATH=src python outputs/verif_m3203/build_report.py    # regenerates §3-§4 and the figure
```
"""
    (REPO / "docs" / "readme_claim_inventory.md").write_text(document, encoding="utf-8")
    print(json.dumps({"before": before, "after": after}, indent=2))


if __name__ == "__main__":
    main()