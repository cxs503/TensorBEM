"""Figures for the PR #133 -> main union merge (mission m3934).

Figure 1 is structural (registry composition of the two merged families) and
needs no solver run. Figure 2 recomputes the coarse row of every 3-D
free-surface family through the merged registry and plots the deviation of the
measured value from the number the documentation quotes, against the registry
tolerance, for both flattening families at once -- that is the point of the
merge: one solve feeds both families and neither side's numbers moved.

Run: PYTHONPATH=src python3 docs/m3934/make_figures.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorbem.benchmarks.declared_numbers import (
    DECLARED_NUMBERS,
    DEFAULT_TOLERANCE,
    FREE_SURFACE_3D_TRANSCRIPT_KEYS,
    check,
)

OUT = Path(__file__).with_name("figures")
OUT.mkdir(exist_ok=True)
DATA = Path(__file__).with_name("figure_data.json")

GROUPS_3D = ("hulme_hemisphere", "hulme_chief", "submerged_sphere", "submerged_sphere_wave")


def family_of(entry) -> str:
    return "PR #133 transcript" if entry.key in FREE_SURFACE_3D_TRANSCRIPT_KEYS else "main"


def figure_registry_composition() -> dict:
    rows = [e for e in DECLARED_NUMBERS if e.group in GROUPS_3D]
    groups = [g for g in GROUPS_3D if any(e.group == g for e in rows)]
    counts = {fam: [sum(1 for e in rows if e.group == g and family_of(e) == fam) for g in groups]
              for fam in ("main", "PR #133 transcript")}
    tiers = {fam: Counter(e.tier for e in rows if family_of(e) == fam)
             for fam in ("main", "PR #133 transcript")}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    left = [0] * len(groups)
    colours = {"main": "#4C72B0", "PR #133 transcript": "#DD8452"}
    for fam, values in counts.items():
        ax.barh(groups, values, left=left, label=fam, color=colours[fam])
        left = [a + b for a, b in zip(left, values)]
    for idx, total in enumerate(left):
        ax.text(total + 0.5, idx, str(total), va="center", fontsize=9)
    ax.set_xlabel("registered numbers (DECLARED_NUMBERS rows)")
    ax.set_title("3-D free-surface registry after the union")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()

    ax = axes[1]
    tiers_order = ("fast", "standard", "heavy")
    bottoms = [0, 0]
    fams = ("main", "PR #133 transcript")
    for tier, colour in zip(tiers_order, ("#55A868", "#C44E52", "#8172B2")):
        values = [tiers[fam].get(tier, 0) for fam in fams]
        ax.bar(fams, values, bottom=bottoms, label=tier, color=colour)
        bottoms = [b + v for b, v in zip(bottoms, values)]
    for idx, total in enumerate(bottoms):
        ax.text(idx, total + 0.5, str(total), ha="center", fontsize=9)
    ax.set_ylabel("rows")
    ax.set_title("tier policy per family (kept as each side set it)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "fig_registry_union.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"counts": counts, "tiers": {f: dict(t) for f, t in tiers.items()}, "figure": path.name}


def figure_recompute() -> dict:
    coarse = [
        e
        for e in DECLARED_NUMBERS
        if e.group in GROUPS_3D
        and len(e.parameters.get("subdivisions", ()) or [1]) == 1
        and e.tier in ("fast", "standard")
    ]
    report = check(entries=coarse)
    rows = report["checks"]
    deviations = [(row["key"], abs(row["measured"] - row["declared"]) / abs(row["declared"])
                  if row["declared"] else abs(row["measured"]), row["verdict"],
                  "PR #133 transcript" if row["key"] in FREE_SURFACE_3D_TRANSCRIPT_KEYS else "main")
                 for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for fam, colour, marker in (("main", "#4C72B0", "o"), ("PR #133 transcript", "#DD8452", "s")):
        pts = [(i, d) for i, (k, d, v, f) in enumerate(deviations) if f == fam]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=26, color=colour,
                   marker=marker, label=fam)
    ax.axhline(DEFAULT_TOLERANCE, color="crimson", linestyle="--", linewidth=1.2,
               label=f"registry tolerance {DEFAULT_TOLERANCE:g}")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1e-1)
    ax.set_xlabel("registered coarse row (index)")
    ax.set_ylabel("|measured - declared| / |declared|")
    ax.set_title(f"Recompute of {len(deviations)} coarse 3-D free-surface rows\n"
                 f"({report['summary']['verdict']}, worst {max(d for _, d, _, _ in deviations):.2e})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = OUT / "fig_recompute_deviations.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {
        "rows": [{"key": k, "rel_deviation": d, "verdict": v, "family": f} for k, d, v, f in deviations],
        "summary": report["summary"],
        "figure": path.name,
    }


def main() -> int:
    data = {"registry": figure_registry_composition(), "recompute": figure_recompute()}
    DATA.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(json.dumps(data["recompute"]["summary"], indent=2))
    print("figures ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
