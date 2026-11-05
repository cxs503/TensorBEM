#!/usr/bin/env python3
"""Figures for the m3985 merge report (PR #162 -> main).

Every plotted number is read from the JSON that the registry produced itself
(``runs/declared_numbers_merged.json`` for the merged tree and the same file
recomputed on the ``main`` baseline), so nothing in the report is transcribed by
hand.

    python3 docs/m3985/make_figures.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MERGED = ROOT / "runs" / "declared_numbers_merged.json"
BASE = Path("/tmp/declared_numbers_base.json")
OUT = HERE / "figures"


def load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def fig_registry_union(merged: dict, base: dict) -> None:
    groups = sorted({row["group"] for row in merged["checks"]})
    base_counts = collections.Counter(row["group"] for row in base["checks"])
    merged_counts = collections.Counter(row["group"] for row in merged["checks"])

    x = np.arange(len(groups))
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - width / 2, [base_counts[g] for g in groups], width, label="main baseline")
    ax.bar(x + width / 2, [merged_counts[g] for g in groups], width, label="merged (m3985)")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=35, ha="right")
    ax.set_ylabel("registered declared numbers (tier=standard)")
    ax.set_title(
        "Registry union: every family from both sides survives the three-way merge\n"
        f"(main {base['summary']['checked']} rows -> merged {merged['summary']['checked']} rows, "
        f"verdict {merged['summary']['verdict']})"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_registry_union.png", dpi=130)
    plt.close(fig)


def _series(rows, prefix: str) -> tuple[list[int], list[float]]:
    points = []
    for row in rows:
        key = row["key"]
        if key.startswith(prefix) and key.endswith(tuple(f"_{n}" for n in (16, 20, 32, 64, 80, 128, 320))):
            points.append((int(key.rsplit("_", 1)[1]), row["measured"]))
    points.sort()
    return [p[0] for p in points], [p[1] for p in points]


def fig_rigid_body(merged: dict) -> None:
    rows = [row for row in merged["checks"] if row["group"] == "rigid_body"]
    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(11, 4.6))

    n2, e2 = _series(rows, "rigid_body_2d_roll_order_")
    if n2:
        ax2.loglog(n2, e2, "o-", label="2-D ellipse: observed order of $A_{66}$")
    _, sym2 = _series(rows, "rigid_body_2d_symmetry_")
    if sym2:
        ax2.axhline(1e-11, color="grey", ls="--", lw=1,
                    label=r"declared antisymmetry bound $10^{-11}$")
    ax2.set_xlabel("panels N")
    ax2.set_ylabel("observed order / residual")
    ax2.set_title("2-D ellipse: second-order convergence\nand round-off symmetry")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)

    n3, e3 = _series(rows, "rigid_body_3d_roll_x_")
    n3t, e3t = _series(rows, "rigid_body_3d_max_translation_")
    if n3:
        ax3.loglog(n3, e3, "s-", label=r"3-D ellipsoid: rel. error of $A_{44}$ (roll)")
    if n3t:
        ax3.loglog(n3t, e3t, "^-", label="3-D ellipsoid: max rel. error, translation")
    _, od = _series(rows, "rigid_body_3d_yaw_z_order_")
    if n3 and od:
        for n, order in zip(n3, od):
            ax3.annotate(f"p={order:.2f}", (n, e3[n3.index(n)]), fontsize=7,
                         textcoords="offset points", xytext=(3, -10))
    ax3.set_xlabel("panels N")
    ax3.set_ylabel("relative error vs Lamb (1932)")
    ax3.set_title("3-D ellipsoid: rotational and\ntranslational Neumann tensors")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(fontsize=8)

    fig.suptitle("Rigid-body Neumann tensors recomputed from the merged tree "
                 f"({len(rows)} rows, verdict {merged['summary']['verdict']})")
    fig.tight_layout()
    fig.savefig(OUT / "fig_rigid_body_measurements.png", dpi=130)
    plt.close(fig)


def fig_deviation(merged: dict) -> None:
    rows = [row for row in merged["checks"] if row["comparison"] == "value"]
    dev = np.array([abs(row["relative_deviation"]) for row in rows], dtype=float)
    order = np.argsort(dev)[::-1]
    dev = dev[order]
    worst = [(rows[i]["key"], dev[k]) for k, i in enumerate(order[:5])]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.semilogy(np.arange(1, dev.size + 1), np.maximum(dev, 1e-12), ".-")
    ax.axhline(0.015, color="tab:red", ls="--", lw=1,
               label="typical declared tolerance (1.5 %)")
    ax.annotate("\n".join(f"{k}: {v:.2e}" for k, v in worst),
                xy=(0.35, 0.55), xycoords="axes fraction", fontsize=7,
                bbox=dict(fc="white", ec="0.7"))
    ax.set_xlabel("quoted number, sorted by deviation")
    ax.set_ylabel("|measured - declared| / declared")
    ax.set_title(
        "Registry integrity after the merge: "
        f"{rows.__len__()} value rows, worst deviation {dev[0]:.2e}, "
        f"{merged['summary']['drifted']} drift"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_deviation_audit.png", dpi=130)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    merged = load(MERGED)
    base = load(BASE)
    fig_registry_union(merged, base)
    fig_rigid_body(merged)
    fig_deviation(merged)
    print("wrote:", *(p.name for p in sorted(OUT.glob("*.png"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())