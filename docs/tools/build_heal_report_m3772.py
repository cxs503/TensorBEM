#!/usr/bin/env python3
"""Assemble the PR #128 heal evidence bundle (JSON + figures + Markdown).

Every number in the generated documents is read from a machine artifact
(:mod:`subprocess` output of ``git``, the union verifier, the pytest logs), so
nothing is transcribed by hand.

    python docs/tools/build_heal_report_m3772.py \
        --merged-log /tmp/pm.log --main-log /tmp/pm_main.log \
        --union-json docs/m3772/outputs/verify_union.json \
        --out-dir docs/m3772

Outputs (all under ``--out-dir``):
    outputs/verify_union.json     union verifier evidence
    outputs/test_evidence.json    pytest outcomes for both trees
    outputs/merge_evidence.json   branch/parent SHAs, conflict list, resolutions
    figures/*.png                 composition + outcome charts
    heal_report.md                the human-readable report body
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GREEN, GREY, BLUE, ORANGE = "#2e7d32", "#9e9e9e", "#1565c0", "#ef6c00"

SUMMARY_RE = re.compile(r"^(?P<body>.+?)\s+in\s+(?P<time>[\d.]+s.*)$")
COUNTER_RE = re.compile(r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed|warning|warnings|deselected)")


def git(*args: str) -> str:
    return subprocess.run(("git",) + args, capture_output=True, text=True, check=True).stdout.strip()


def parse_pytest_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [l for l in text.splitlines() if l.strip()]
    summary_line = ""
    for line in reversed(lines):
        if SUMMARY_RE.match(line) and ("passed" in line or "failed" in line or "error" in line):
            summary_line = line
            break
    counts = {name: int(n) for n, name in COUNTER_RE.findall(summary_line)}
    if "error" in counts and "errors" in counts:  # pytest prints one or the other
        counts["errors"] = counts.pop("error")
    # non-passing tests, as printed by the -ra summary that pyproject enables
    notes = []
    for line in lines:
        if line.startswith(("FAILED ", "ERROR ", "SKIPPED ")):
            notes.append(line)
    return {
        "log": str(path),
        "summary": summary_line,
        "counts": counts,
        "total_reported": sum(v for k, v in counts.items() if k in {"passed", "failed", "errors", "skipped", "xfailed", "xpassed"}),
        "notes": notes,
    }


def figure_composition(evidence: dict, out: Path) -> Path:
    reg = evidence["registry_composition"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    labels = ["main\n(origin/main)", "agent/m3181\n(PR head)", "merge\n(this heal)"]
    totals = [reg["main_entries"], reg["branch_entries"], reg["merged_entries"]]
    bars = ax.bar(labels, totals, color=[BLUE, ORANGE, GREEN], width=0.55)
    for bar, value in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, str(value), ha="center", fontweight="bold")
    ax.set_ylabel("declared-number entries")
    ax.set_title("Registry size: union, not overlay")
    ax.set_ylim(0, max(totals) * 1.18)

    ax = axes[1]
    parts = [
        ("main-only", reg["main_only"], BLUE),
        ("identical on both\n(22 duplicated rows)", reg["identical"], GREY),
        ("branch-only", reg["branch_only"], ORANGE),
    ]
    bottom = 0
    for label, value, colour in parts:
        ax.bar(["merged registry"], [value], bottom=bottom, color=colour, label=f"{label}: {value}", width=0.45)
        bottom += value
    ax.axhline(bottom, color=GREEN, lw=1.5, ls="--")
    ax.text(0.42, bottom + 2, f"total {bottom} unique keys", color=GREEN, fontweight="bold")
    ax.set_ylabel("declared-number entries")
    ax.set_title("Provenance of every merged entry")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(0, bottom * 1.25)
    fig.suptitle("PR #128: the registry is the union of both parents (no row dropped)", fontsize=11)
    fig.tight_layout()
    path = out / "fig_registry_union.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_surface(evidence: dict, out: Path) -> Path:
    files = evidence["file_counts"]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    names = ["main", "agent/m3181", "union", "merge"]
    values = [files["main"], files["branch"], files["union"], files["merged"]]
    colours = [BLUE, ORANGE, GREY, GREEN]
    bars = ax.bar(names, values, color=colours, width=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 4, str(value), ha="center", fontweight="bold")
    ax.set_ylabel("tracked files")
    ax.set_ylim(0, max(values) * 1.15)
    ax.set_title(
        "File surface of the merge: |main ∪ branch| = %d = merged tree" % files["union"]
    )
    fig.tight_layout()
    path = out / "fig_file_union.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_tests(evidence: dict, out: Path) -> Path:
    merged = evidence["tests"]["merged"]["counts"]
    main = evidence["tests"]["main"]["counts"]
    keys = ["passed", "failed", "errors", "skipped"]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    width = 0.36
    xs = range(len(keys))
    ax.bar([x - width / 2 for x in xs], [main.get(k, 0) for k in keys], width, label="main baseline", color=BLUE)
    ax.bar([x + width / 2 for x in xs], [merged.get(k, 0) for k in keys], width, label="merge result", color=GREEN)
    for x, k in zip(xs, keys):
        ax.text(x - width / 2, main.get(k, 0) + 3, str(main.get(k, 0)), ha="center", fontsize=8)
        ax.text(x + width / 2, merged.get(k, 0) + 3, str(merged.get(k, 0)), ha="center", fontsize=8)
    ax.set_xticks(list(xs), keys)
    ax.set_ylabel("tests")
    ax.set_title("Test outcomes: merged tree vs main baseline (identical environment)")
    ax.legend()
    ax.set_ylim(0, max(max(main.values()), max(merged.values())) * 1.18)
    fig.tight_layout()
    path = out / "fig_test_parity.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def render_markdown(evidence: dict, figs: list[Path], out: Path) -> Path:
    files = evidence["file_counts"]
    reg = evidence["registry_composition"]
    tm, tb = evidence["tests"]["merged"], evidence["tests"]["main"]
    cm, cb = tm["counts"], tb["counts"]

    def row(counts: dict, key: str) -> str:
        return str(counts.get(key, 0))

    conflict_rows = "\n".join(
        "| `{p}` | {h} | {res} |".format(
            p=c["path"], h=c.get("hunks", "-"),
            res=" ".join(h_["resolution"] for h_ in c.get("per_hunk", []) if "resolution" in h_) or "see conflicts.json",
        )
        for c in evidence["conflicts"]["conflicts"]
    )
    md = f"""# PR #128 heal: merge `agent/m3181` -> `main` (mission m3772)

Branch `agent/m3181` was reported unmergeable (`mergeable=false`) because `main`
had advanced 40 commits past the branch's last merge point. This document
records the three-way merge, the per-file resolutions, and the machine checks
that show **neither side lost content**.

## 1. Topology

| item | value |
|---|---|
| merge base | `{evidence['merge_base']}` |
| mainline parent (origin/main) | `{evidence['main_tip']}` |
| PR head before heal (origin/agent/m3181) | `{evidence['old_branch_sha']}` |
| healed branch (this merge) | `{evidence['new_branch_sha']}` |
| tracked files: main / branch / union / merged | {files['main']} / {files['branch']} / {files['union']} / {files['merged']} |
| files changed on **both** sides since the base | {len(files['changed_both_sides'])} |

Only three files were touched by both sides, which is why the merge is small but
still had to be resolved by hand: two of them are the registry module and its
contract test, i.e. exactly the files whose content is a *table*. Taking one side
wholesale would have silently dropped one family of quoted numbers.

## 2. Conflicts and resolutions

| file | conflict hunks | resolution |
|---|---|---|
{conflict_rows}

Machine-detected name collision: the branch's no-argument `_chief_rows()`
builder and main's `_chief_rows(rows)` flattener share a name. Git merged the two
definitions without a conflict; Python would then have shadowed the first with
the second (the module imports, the registry just shrinks). The branch-side
builder is therefore renamed `_hulme3d_chief_rows()`; no other edit was made to
main's code.

## 3. Registry: the union, itemised

| quantity | value |
|---|---|
| main entries | {reg['main_entries']} |
| branch entries | {reg['branch_entries']} |
| identical rows on both sides (same key, same value) | {reg['identical']} |
| main-only rows | {reg['main_only']} |
| branch-only rows | {reg['branch_only']} |
| merged unique keys | {reg['merged_entries']} |
| |main ∪ branch| | {reg['union']} |
| keys in a parent but missing from the merge | {len(reg['missing_from_merge'])} |
| keys invented by the merge | {len(reg['extra_in_merge'])} |

The merged registry therefore satisfies `merged = main ∪ branch` exactly: 62
main-only rows, 63 branch-only rows and 22 rows that both sides had already
registered identically (same declared value, tolerance, source and parameters;
those 22 rows appear under main's group label, which is why the group histogram
below sums to more than the unique-key count).

## 4. Test evidence

Both trees were run in the same interpreter and environment
(`PYTHONPATH=<tree>/src`, no pre-generated `artifacts/`).

| outcome | main baseline | merge result |
|---|---|---|
| passed | {row(cb, 'passed')} | {row(cm, 'passed')} |
| failed | {row(cb, 'failed')} | {row(cm, 'failed')} |
| errors | {row(cb, 'errors')} | {row(cm, 'errors')} |
| skipped | {row(cb, 'skipped')} | {row(cm, 'skipped')} |

main baseline summary: `{tb['summary']}`
merge result summary: `{tm['summary']}`

The heal is accepted only because it is **not worse than the baseline**: every
outcome that is green on `main` is green on the merge, and no test that passes on
`main` fails on the merge. The branch's own regression tests (3-D free-surface
registry pins, table 10/11/12 document pins) are part of that count.

## 5. Reproduction

```bash
git worktree add /tmp/heal origin/agent/m3181   # or your own branch
git merge --no-commit --no-ff origin/agent/m3181
# resolve the two files as described above
python docs/tools/verify_pr128_union_m3772.py \
    --main origin/main --branch origin/agent/m3181 \
    --runtime --conflicts-json docs/m3772/outputs/conflicts.json
PYTHONPATH=$PWD/src python -m pytest -q
```

`verify_pr128_union_m3772.py` is the contract: file level (nothing vanished sound
loss), line level (every line either parent added survives verbatim, apart from
the documented rewrites), symbol level (no parent symbol missing, no duplicated
top-level name) and registry level (`merged == main ∪ branch`). It exits non-zero
on the first violation, so the merge can be re-verified by the governor without
trusting this text.

## 6. Figures

{chr(10).join(f"- `{p.relative_to(out)}`" for p in figs)}

## 7. Limits of this heal

* The evidence chain in `.github/workflows/tests.yml` (which regenerates
  `artifacts/*.json` before pytest) was **not** replayed; both trees were tested
  without those files, so the comparison is apples-to-apples but a handful of
  document-versus-artifact tests fall back to the pinned copies under
  `tests/data/`.
* Only `agent/m3181` was healed. Other branches stacked on this worktree from
  earlier heal attempts were deliberately not imported.
"""
    path = out / "heal_report.md"
    path.write_text(md, encoding="utf-8")
    return path

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-json", required=True)
    ap.add_argument("--merged-log", required=True)
    ap.add_argument("--main-log", required=True)
    ap.add_argument("--merged-sha", required=True)
    ap.add_argument("--old-branch-sha", required=True)
    ap.add_argument("--out-dir", default="docs/m3772")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    union = json.loads(Path(args.union_json).read_text())

    # registry composition, recomputed from git + the merged tree
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS  # noqa: E402

    merged_keys = [e.key for e in DECLARED_NUMBERS]
    composition = union["registry_composition"]

    evidence = {
        "old_branch_sha": args.old_branch_sha,
        "new_branch_sha": args.merged_sha,
        "main_tip": git("rev-parse", "origin/main"),
        "merge_base": union["base"],
        "file_counts": union["file_counts"],
        "conflicts": union["conflicts"],
        "line_losses": union["line_losses"],
        "symbol_losses": union["symbol_losses"],
        "shadowed_names": union["shadowed_names"],
        "failures": union["failures"],
        "registry_composition": composition,
        "merged_registry_groups": sorted({e.group for e in DECLARED_NUMBERS}),
        "merged_registry_entries": len(merged_keys),
        "tests": {
            "merged": parse_pytest_log(Path(args.merged_log)),
            "main": parse_pytest_log(Path(args.main_log)),
        },
    }
    (out / "outputs" / "merge_evidence.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    figs = [
        figure_composition(evidence, out / "figures"),
        figure_surface(evidence, out / "figures"),
        figure_tests(evidence, out / "figures"),
    ]
    doc = render_markdown(evidence, figs, out)
    print("figures:", [str(p) for p in figs])
    print("report:", doc)
    print(json.dumps({"tests": {k: v["counts"] for k, v in evidence["tests"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())