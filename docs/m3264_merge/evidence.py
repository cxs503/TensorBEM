"""Recompute the 3-D free-surface rows and write the merge evidence of PR #135.

Writes ``docs/m3264_merge/free_surface_rows.json`` (declared / measured /
deviation for every row of tables 10-12) and the two figures used by the merge
report. Run from the repository root::

    PYTHONPATH=src python3 docs/m3264_merge/evidence.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS, check  # noqa: E402

GROUPS = ("hulme_chief", "hulme_hemisphere", "submerged_sphere")
OUT = Path(__file__).resolve().parent


def main() -> int:
    report = check(tiers=("standard",))
    rows = [r for r in report["checks"] if r["group"] in GROUPS]
    (OUT / "free_surface_rows.json").write_text(
        json.dumps(
            {
                "summary": report["summary"],
                "registry_size": len(DECLARED_NUMBERS),
                "free_surface_rows": len(rows),
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )

    values = [r for r in rows if r["comparison"] == "value"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    for group, marker in zip(GROUPS, ("o", "s", "^")):
        sel = [r for r in values if r["group"] == group]
        ax.plot(
            [r["declared"] for r in sel],
            [r["measured"] for r in sel],
            marker,
            markersize=4,
            linestyle="none",
            label=f"{group} ({len(sel)})",
        )
    lims = [1e-6, 1e3]
    ax.plot(lims, lims, "k-", linewidth=0.8, label="measured = declared")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("declared value from docs/marine_bem_benchmarks.md")
    ax.set_ylabel("recomputed by the merged registry")
    ax.set_title("Tables 10-12: 69 cells, declared vs recomputed")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    rel = [
        abs(r["relative_deviation"])
        for r in values
        if "rotational_coupling" not in r["key"] and "kappa" not in r["key"]
    ]
    ax.hist(rel, bins=18, color="#3b6ea5", edgecolor="black", linewidth=0.4)
    ax.axvline(5.0e-3, color="crimson", linestyle="--", label="1 % guard (5e-3 rel.)")
    ax.set_xlabel("|measured - declared| / declared  (four-figure cells)")
    ax.set_ylabel("cells")
    ax.set_title(f"worst four-figure deviation {max(rel):.2e}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_merge_registry.png", dpi=140)
    print("summary:", report["summary"])
    print("worst four-figure rel. dev:", max(rel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())