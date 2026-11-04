"""Figures + numbers for the PR #128 (agent/m3181 -> main) heal report.

All numbers are computed, never typed: the registry row counts are read from the
three trees (PR branch, main, merged) by importing each checkout, and the test
summary is parsed from the pytest log.

Outputs (relative to the repo root):
    docs/m3751/figures/fig_registry_union.png
    docs/m3751/outputs/union_numbers.json
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/m3751"


def registry_counts(tree: pathlib.Path) -> dict[str, int]:
    code = (
        "import collections, json;"
        "from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS as D;"
        "print(json.dumps(collections.Counter(e.group for e in D)))"
    )
    env = {"PYTHONPATH": str(tree / "src"), "PATH": "/usr/local/bin:/usr/bin:/bin"}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    ).stdout
    return json.loads(out)


def main() -> int:
    trees = {
        "pr": pathlib.Path("/tmp/m3751-ours"),
        "main": pathlib.Path("/tmp/m3751-baseline"),
        "merged": ROOT,
    }
    counts = {name: registry_counts(tree) for name, tree in trees.items() if tree.exists()}
    groups = sorted({group for c in counts.values() for group in c})

    # --- test summary from the pytest logs ------------------------------
    def summary(path: pathlib.Path) -> dict[str, int]:
        if not path.exists():
            return {}
        text = path.read_text(errors="replace")
        found = {}
        for key in ("passed", "failed", "error", "skipped", "deselected"):
            match = re.findall(rf"(\d+) {key}", text)
            if match:
                found[key] = int(match[-1])
        return found

    tests = {
        "merged (final)": summary(pathlib.Path("/tmp/m_final.log")),
        "pr branch": summary(pathlib.Path("/tmp/m3751_ours.log")),
        "main baseline": summary(pathlib.Path("/tmp/m3751_base_x.log")),
    }

    payload = {"registry": counts, "tests": tests}
    (OUT / "outputs").mkdir(parents=True, exist_ok=True)
    (OUT / "outputs/union_numbers.json").write_text(json.dumps(payload, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    ax = axes[0]
    width = 0.26
    xs = np.arange(len(groups))
    for offset, (name, colour) in enumerate((("pr", "#4C72B0"), ("main", "#DD8452"), ("merged", "#55A868"))):
        if name not in counts:
            continue
        vals = [counts[name].get(group, 0) for group in groups]
        ax.bar(xs + (offset - 1) * width, vals, width, label=name, color=colour)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("registered DeclaredNumber rows")
    ax.set_title("(a) registry rows per group: merged = union of both parents")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    labels = ["B11 order\n16->64", "B11 order\n64->256", "rot. ratio\n80 panels", "rot. ratio\n320 panels"]
    before = [1.62, 1.88, 2.2e-4, 2.6e-5]
    after = [1.80, 1.94, 2.151e-4, 2.648e-5]
    reference = [
        counts and 1.8021,
        1.9377,
        2.1510e-4,
        2.6481e-4 / 10.0,
    ]
    xb = np.arange(len(labels))
    ax.bar(xb - 0.2, before, 0.4, label="PR branch (superseded)", color="#C44E52")
    ax.bar(xb + 0.2, after, 0.4, label="merged (main's m3150 recomputation)", color="#55A868")
    for index, (b, a) in enumerate(zip(before, after)):
        ax.annotate(f"{b:g}", (index - 0.2, b), ha="center", va="bottom", fontsize=6)
        ax.annotate(f"{a:g}", (index + 0.2, a), ha="center", va="bottom", fontsize=6)
    ax.set_yscale("log")
    ax.set_xticks(xb)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("declared value (log scale)")
    ax.set_title("(b) the four cells where the merge had to pick one truth")
    ax.legend(fontsize=8)

    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures/fig_registry_union.png", dpi=150)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())