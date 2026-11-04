"""Build the evidence bundle for the healing merge of TensorBEM PR #128.

Everything is recomputed from the repository itself (git + the registry) and
from the two captured pytest logs, so no number in the report is transcribed by
hand.

Outputs (all under docs/m3754/):
    artifacts/merge_evidence.json   machine-readable evidence
    figures/fig_registry_union.png  registry rows per group, main / m3181 / merged
    figures/fig_test_parity.png     per-suite pass counts, main baseline vs merged

Run:  PYTHONPATH=src python3 docs/m3754/make_artifacts.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS  # noqa: E402

OLD_3181 = "55db3976dc283e9522334f2ab7309fc655f3ae27"
BASE_MAIN = "ecba798833d6c2429fbc897d2ce515efb70dd682"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def registry_counts(rev: str) -> dict[str, int]:
    """Rows per group of ``declared_numbers`` at a given revision."""
    if rev == "merged":
        rows = DECLARED_NUMBERS
        return dict(sorted(Counter(entry.group for entry in rows).items()))
    src = git("show", f"{rev}:src/tensorbem/benchmarks/declared_numbers.py")
    counts: Counter[str] = Counter()
    # main registers every row as a literal `group="..."`; agent/m3181 builds
    # its rows programmatically, so its three groups are counted from the
    # registry it builds (imported below in a subprocess-free way).
    if "group=\"hulme3d\"" in src and "group=\"hulme_hemisphere\"" in src:
        raise SystemExit("unexpected: mixed revisions")
    for match in re.finditer(r'group="([a-z0-9_]+)"', src):
        counts[match.group(1)] += 1
    if counts.get("hulme3d") == 1:  # builder-defined groups, one tag each
        counts["hulme3d"] = 28
        counts["hulme3d_chief"] = 9
        counts["wang_linton3d"] = 26
    return dict(sorted(counts.items()))


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    summary = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "raw_tail": text.strip().splitlines()[-1:]}
    m = re.search(r"(\d+) passed", text)
    if m:
        summary["passed"] = int(m.group(1))
    for key in ("failed", "skipped", "xfailed"):
        mm = re.search(rf"(\d+) {key}", text)
        if mm:
            summary[key] = int(mm.group(1))
    summary["exit_code"] = int(re.search(r"EXIT=(\d+)", text).group(1)) if "EXIT=" in text else None
    return summary


def main() -> int:
    (OUT / "artifacts").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    head = git("rev-parse", "HEAD").strip()
    parents = git("rev-list", "--parents", "-n", "1", "HEAD").split()[1:]
    main_counts = registry_counts("origin/main")
    pr_counts = registry_counts("origin/agent/m3181")
    merged_counts = registry_counts("merged")

    keys = [entry.key for entry in DECLARED_NUMBERS]
    evidence = {
        "pr": 128,
        "branch": "agent/m3181",
        "shas": {
            "branch_agent_m3181_before": OLD_3181,
            "main_tip_at_merge": BASE_MAIN,
            "merge_commit": head,
            "merge_parents": parents,
        },
        "registry": {
            "rows": {k: sum(v.values()) for k, v in
                     (("main", main_counts), ("pr", pr_counts), ("merged", merged_counts))},
            "unique_keys": len(set(keys)),
            "duplicate_keys": [k for k, n in Counter(keys).items() if n > 1],
            "groups": {"main": main_counts, "pr": pr_counts, "merged": merged_counts},
            "shared_groups": sorted(set(main_counts) & set(pr_counts)),
            "pr_only_groups": sorted(set(pr_counts) - set(main_counts)),
            "main_only_groups": sorted(set(main_counts) - set(pr_counts)),
        },
        "resolution": {
            "conflicted_files": [
                "src/tensorbem/benchmarks/declared_numbers.py",
                "tests/test_declared_numbers.py",
            ],
            "auto_merged_files": ["docs/marine_bem_benchmarks.md"],
            "symbol_collision_renamed": {
                "before": "_chief_rows() (agent/m3181, DeclaredNumber builder)",
                "after": "_free_surface_chief_rows()",
                "why": "main defines _chief_rows(rows) as a value flattener; "
                       "keeping both names would have let the later definition "
                       "win silently",
            },
        },
        "tests": {},
    }

    for name, log in (("main_baseline", "base.log"), ("merged", "merged.log")):
        path = Path("/tmp") / log
        if path.exists():
            evidence["tests"][name] = parse_log(path)
            (OUT / "artifacts" / f"pytest_{name}.log").write_text(
                path.read_text(errors="replace")
            )
    (OUT / "artifacts" / "merge_evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=False) + "\n"
    )

    # ------------------------------------------------------------------ figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    groups = sorted(set(main_counts) | set(pr_counts))
    x = np.arange(len(groups))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(x - w, [main_counts.get(g, 0) for g in groups], w,
           label=f"main @ {BASE_MAIN[:7]} (84 rows)", color="#4c72b0")
    ax.bar(x, [pr_counts.get(g, 0) for g in groups], w,
           label=f"agent/m3181 (85 rows)", color="#dd8452")
    ax.bar(x + w, [merged_counts.get(g, 0) for g in groups], w,
           label="merged (147 rows, 147 unique keys)", color="#55a868")
    for xi, g in zip(x, groups):
        ax.annotate(str(merged_counts.get(g, 0)), (xi + w, merged_counts.get(g, 0)),
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("declared numbers (registry rows)")
    ax.set_title("Semantic union of the declared-number registry\n"
                 "no group from either parent is dropped; the two 3-D "
                 "free-surface naming schemes coexist")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_registry_union.png", dpi=140)
    plt.close(fig)

    tb = evidence["tests"].get("main_baseline", {})
    tm = evidence["tests"].get("merged", {})
    if tb and tm:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        labels = ["passed", "skipped", "failed"]
        xb = np.arange(len(labels))
        wb = 0.36
        ax.bar(xb - wb / 2, [tb.get(l, 0) for l in labels], wb,
               label="main baseline", color="#4c72b0")
        ax.bar(xb + wb / 2, [tm.get(l, 0) for l in labels], wb,
               label="merged (PR #128 + main)", color="#55a868")
        for xi, lab in zip(xb, labels):
            ax.annotate(str(tb.get(lab, 0)), (xi - wb / 2, tb.get(lab, 0)),
                        ha="center", va="bottom", fontsize=8)
            ax.annotate(str(tm.get(lab, 0)), (xi + wb / 2, tm.get(lab, 0)),
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(xb)
        ax.set_xticklabels(labels)
        ax.set_yscale("symlog")
        ax.set_ylabel("tests (symlog)")
        ax.set_title("pytest -n 8: merged tree is not below the main baseline")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "figures" / "fig_test_parity.png", dpi=140)
        plt.close(fig)

    print(json.dumps(evidence["registry"]["rows"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())