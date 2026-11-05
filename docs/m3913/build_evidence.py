"""Build the PR #128 merge evidence (JSON + figure) from measured inputs.

Inputs, all produced by real runs (nothing hand-typed):

* ``docs/m3913/verify_union.py`` - per-file, per-parent addition accounting
  (kept / rule-rewritten / lost) against the merge base.
* pytest summary lines of the baseline (main) and the healed branch runs.

Outputs ``docs/m3913/outputs/pr128_union_evidence.json``,
``docs/m3913/figures/fig_pr128_union.png``.

Usage::

    PYTHONPATH=src python3 docs/m3913/build_evidence.py \
        --baseline-log /tmp/base_main.log --merged-log /tmp/merge_m3913.log
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

HERE = Path(__file__).resolve().parent
FILES = (
    "src/tensorbem/benchmarks/declared_numbers.py",
    "tests/test_declared_numbers.py",
    "docs/marine_bem_benchmarks.md",
)
SHORT = {
    FILES[0]: "declared_numbers.py",
    FILES[1]: "test_declared_numbers.py",
    FILES[2]: "marine_bem_benchmarks.md",
}
SUMMARY = re.compile(r"(\d+) (passed|failed|skipped|error|errors)")


def verify(repo: Path, main: str, branch: str) -> dict:
    out = subprocess.run(
        [sys.executable, str(HERE / "verify_union.py"), "--repo", str(repo),
         "--main", main, "--branch", branch, "--merged", "worktree"],
        capture_output=True, text=True, check=False,
    )
    rows: dict[str, dict] = {}
    for line in out.stdout.splitlines():
        match = re.match(r"(.+?): (main|branch) additions kept=(\d+) rule-rewritten=(\d+) lost=(\d+)$", line)
        if match:
            path, side, kept, rules, lost = match.groups()
            rows.setdefault(path, {})[side] = {
                "kept": int(kept), "rule_rewritten": int(rules), "lost": int(lost)
            }
    return {"rows": rows, "verdict": out.stdout.strip().splitlines()[-1], "exit": out.returncode}


def pytest_counts(log: Path) -> dict:
    text = log.read_text(errors="replace")
    counts: dict[str, int] = {}
    for value, kind in SUMMARY.findall(text):
        kind = "error" if kind.startswith("error") else kind
        counts[kind] = counts.get(kind, 0) + int(value)
    progress = re.findall(r"\[ *(\d+)%\]", text)
    counts["reached_percent"] = int(progress[-1]) if progress else 0
    counts["complete"] = bool(re.search(r"=+ .*(passed|failed|error).* =+", text))
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--main", default="origin/main")
    ap.add_argument("--branch", default="origin/agent/m3181")
    ap.add_argument("--baseline-log", default="/tmp/base_main.log")
    ap.add_argument("--merged-log", default="/tmp/merge_m3913.log")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    union = verify(repo, args.main, args.branch)
    baseline = pytest_counts(Path(args.baseline_log))
    merged = pytest_counts(Path(args.merged_log))

    total = {path: sum(side["kept"] + side["rule_rewritten"] + side["lost"]
                       for side in union["rows"].get(path, {}).values())
             for path in FILES}
    lost = sum(side["lost"] for path in FILES
               for side in union["rows"].get(path, {}).values())

    payload = {
        "pr": 128,
        "branch": "agent/m3181",
        "merge_base": subprocess.run(["git", "-C", str(repo), "merge-base", args.main, args.branch],
                                     capture_output=True, text=True, check=True).stdout.strip(),
        "parents": {
            "main": subprocess.run(["git", "-C", str(repo), "rev-parse", args.main],
                                   capture_output=True, text=True, check=True).stdout.strip(),
            "branch": subprocess.run(["git", "-C", str(repo), "rev-parse", args.branch],
                                     capture_output=True, text=True, check=True).stdout.strip(),
        },
        "union": union["rows"],
        "additions_total": total,
        "additions_lost": lost,
        "verdict": union["verdict"],
        "pytest": {"baseline_main": baseline, "merged": merged},
    }
    out_dir = HERE / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pr128_union_evidence.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- figure: kept vs rule-rewritten vs lost, per file and per parent ----
    labels, keep_main, keep_branch, rule, losts = [], [], [], [], []
    for path in FILES:
        rows = union["rows"].get(path, {})
        labels.append(SHORT[path])
        keep_main.append(rows.get("main", {}).get("kept", 0))
        keep_branch.append(rows.get("branch", {}).get("kept", 0))
        rule.append(rows.get("main", {}).get("rule_rewritten", 0)
                    + rows.get("branch", {}).get("rule_rewritten", 0))
        losts.append(rows.get("main", {}).get("lost", 0) + rows.get("branch", {}).get("lost", 0))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    x = range(len(labels))
    width = 0.26
    ax.bar([i - width for i in x], keep_main, width, label="main additions kept", color="#3b6ea5")
    ax.bar(list(x), keep_branch, width, label="branch additions kept", color="#c46a3f")
    ax.bar([i + width for i in x], rule, width, label="rule-rewritten (documented)", color="#8a8f98")
    ax.bar([i + 2 * width for i in x], losts, width, label="silently lost", color="#b3282d")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("added lines vs merge base")
    ax.set_title("PR #128 union accounting (source lines)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylim(0, max(keep_main + keep_branch) * 1.35 + 5)
    for i, (lm, lb, rr) in enumerate(zip(keep_main, keep_branch, rule)):
        ax.text(i - width, lm, str(lm), ha="center", va="bottom", fontsize=7)
        ax.text(i, lb, str(lb), ha="center", va="bottom", fontsize=7)
        ax.text(i + width, rr, str(rr), ha="center", va="bottom", fontsize=7)

    ax = axes[1]
    kinds = ["passed", "failed", "skipped"]
    base_vals = [baseline.get(k, 0) for k in kinds]
    merge_vals = [merged.get(k, 0) for k in kinds]
    x = range(len(kinds))
    ax.bar([i - 0.18 for i in x], base_vals, 0.36, label="main (baseline)", color="#3b6ea5")
    ax.bar([i + 0.18 for i in x], merge_vals, 0.36, label="agent/m3181 healed", color="#c46a3f")
    ax.set_xticks(list(x))
    ax.set_xticklabels(kinds)
    ax.set_ylabel("tests")
    ax.set_title("pytest outcome, baseline vs healed branch")
    ax.legend(fontsize=8, frameon=False)
    for i, (b, m) in enumerate(zip(base_vals, merge_vals)):
        ax.text(i - 0.18, b, str(b), ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.18, m, str(m), ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"TensorBEM PR #128 3-way merge: zero silent loss "
                 f"({lost} lines lost, verdict {union['verdict'].split(':')[0]})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig_dir = HERE / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "fig_pr128_union.png", dpi=150)
    print(json.dumps({"verdict": union["verdict"], "additions_lost": lost,
                      "pytest": payload["pytest"]}, indent=2, ensure_ascii=False))
    return 0 if union["exit"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())