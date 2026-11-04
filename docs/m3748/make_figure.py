#!/usr/bin/env python3
"""Figure for the #128 healing report: declared vs measured parity for the 125
3-D free-surface registry rows re-verified on the merged tree, plus the
deviation budget of the worst rows.

usage: make_figure.py <registry_audit.json> <out.png>
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GROUPS = {
    "hulme_hemisphere": ("#1f77b4", "o", "main: hemisphere (tbl 10)"),
    "submerged_sphere": ("#d62728", "s", "main: submerged sphere (tbl 12)"),
    "hulme_chief": ("#9467bd", "^", "main: irregular freq (tbl 11)"),
    "hulme3d": ("#2ca02c", "D", "m3181: hulme3d"),
    "hulme3d_chief": ("#8c564b", "v", "m3181: hulme3d_chief"),
    "wang_linton3d": ("#ff7f0e", "P", "m3181: wang_linton3d"),
}


def main(src: str, out: str) -> int:
    report = json.load(open(src))
    rows = [r for r in report["checks"] if abs(r["declared"]) > 1e-12]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.4))

    lo, hi = 1e-4, 1.0
    ax.plot([lo, hi], [lo, hi], "k-", lw=1.0, label="measured = declared")
    ax.plot([lo, hi], [lo * 1.03, hi * 1.03], "k--", lw=0.7, alpha=0.6)
    ax.plot([lo, hi], [lo / 1.03, hi / 1.03], "k--", lw=0.7, alpha=0.6,
            label="+/-3 % band")
    for group, (colour, marker, label) in GROUPS.items():
        xs = [abs(r["declared"]) for r in rows if r["group"] == group]
        ys = [abs(r["measured"]) for r in rows if r["group"] == group]
        if xs:
            ax.scatter(xs, ys, s=34, c=colour, marker=marker, alpha=0.85,
                       label=f"{label} ({len(xs)})", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("declared value (registry quote)")
    ax.set_ylabel("measured value (merged tree, recomputed)")
    ax.set_title("125 registry rows re-verified after the #128 merge")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)

    worst = sorted(rows, key=lambda r: -abs(r["relative_deviation"]) / r["tolerance"])[:16]
    labels = [r["key"] for r in worst][::-1]
    frac = [abs(r["relative_deviation"]) / r["tolerance"] for r in worst][::-1]
    colours = [GROUPS[r["group"]][0] for r in worst][::-1]
    ax2.barh(range(len(labels)), frac, color=colours)
    ax2.axvline(1.0, color="k", ls="--", lw=1.0, label="tolerance limit")
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=6.5)
    ax2.set_xlabel("|measured - declared| / tolerance")
    ax2.set_title("Worst 16 rows: deviation budget (all < 1 = PASS)")
    ax2.grid(True, axis="x", ls=":", alpha=0.4)
    ax2.legend(fontsize=8)

    summary = report["summary"]
    ax2.text(0.98, 0.02,
             f"{summary['passed']}/{summary['checked']} PASS, "
             f"{summary['drifted']} drift, worst "
             f"{max(frac):.3f}x tol",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=8,
             bbox={"fc": "white", "ec": "0.6", "alpha": 0.9})

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))