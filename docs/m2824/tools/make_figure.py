"""Figure for the gap-closure report: mutation score before vs after the backfill."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
BEFORE = REPO / "docs/m2582/outputs/mutation_audit_after/mutation_report.json"
AFTER = REPO / "docs/m2824/outputs/mutation_audit_after/mutation_report.json"
REPLAY = REPO / "docs/m2824/outputs/mutant_replay.json"
OUT = REPO / "docs/m2824/outputs/mutation_gap_closure.png"


def scores(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    table = {}
    for module in payload["modules"]:
        name = module["module"] if isinstance(module, dict) else module
        if isinstance(module, dict):
            table[name] = module["score"]
    return table


def main() -> None:
    before = scores(BEFORE)
    after = scores(AFTER)
    modules = sorted(after)
    short = [m.split(".")[-1] for m in modules]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))

    ax = axes[0]
    x = np.arange(len(modules))
    w = 0.38
    ax.bar(x - w / 2, [before[m] for m in modules], w, label="before (m2582)", color="#b0b7c3")
    ax.bar(x + w / 2, [after[m] for m in modules], w, label="after (this work)", color="#2f6fbf")
    for xi, m in zip(x, modules):
        ax.text(xi - w / 2, before[m] + 0.015, f"{before[m]:.3f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, after[m] + 0.015, f"{after[m]:.3f}", ha="center", fontsize=8, weight="bold")
    ax.set_xticks(x, short)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("mutation score (killed / total)")
    ax.set_title("Seeded 8-mutant sweep per module (seed 20261103)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    replay = json.loads(REPLAY.read_text())
    labels = [row["id"].replace("hydrostatic ", "hyd ").replace("free_surface", "fs") for row in replay["mutants"]]
    pre = [1.0 if row.get("pre_suite_green") else 0.0 for row in replay["mutants"]]
    post = [0.0 if not row.get("post_suite_green") else 1.0 for row in replay["mutants"]]
    y = np.arange(len(labels))
    ax.barh(y - 0.19, pre, 0.34, color="#c9ccd1", label="pre-existing suite: green (mutant lives)")
    ax.barh(y + 0.19, post, 0.34, color="#c0392b", label="with backfill: red (mutant killed)")
    ax.set_yticks(y, [l[:46] for l in labels], fontsize=7)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("suite status under the mutant (1 = green)")
    ax.set_title(f"Replay of the audited survivors: {replay['killed']}/{replay['total']} killed")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()