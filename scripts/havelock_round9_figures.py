#!/usr/bin/env python3
"""Round-9 figures: rendered directly from ``adjudicate.json`` (zero hand-copied numbers).

    PYTHONPATH=src python3 scripts/havelock_round9_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "case_bem_submerged_sphere_havelock_round9"
ADJ = OUT / "adjudicate.json"
MEASURES = ("flat", "patch", "uniform")
COLORS = {"flat": "#b03030", "patch": "#2b6cb0", "uniform": "#808080"}


def fig_tiers(adj: dict) -> Path:
    tiers = list(adj["all_tier_verdicts"])
    x = np.arange(len(tiers))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for k, m in enumerate(MEASURES):
        vals = [adj["all_tier_verdicts"][t][m]["band_max_rel_error"] * 100 for t in tiers]
        ax.bar(x + (k - 1) * w, vals, w, label=m, color=COLORS[m])
    ax.axhline(5.0, color="k", ls="--", lw=1.2, label="case tolerance 5%")
    ax.axhline(3.0, color="k", ls=":", lw=1.0, label="3% standard")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("band-max relative error [%]")
    ax.set_title("Havelock submerged sphere: surface-measure intervention, all tiers\n"
                 "(recovered run, re-derived verdicts; 5% gate applied per item)")
    ax.legend(fontsize=8, ncol=5)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = OUT / "fig1_tiers_bandmax.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_case_mesh_cells(adj: dict) -> Path:
    """Per-cell errors, taken from the adjudicated item lists (no re-derivation)."""
    items = {m: {(round(float(it["ka"]), 6), it["quantity"]): it["rel_error"]
                 for it in adj["case_mesh_verdicts"][m]["items"]}
             for m in MEASURES}
    keys = sorted(items["flat"])
    cells = adj["patch_vs_uniform_control"]["cells"]
    gap = {(round(float(c["ka"]), 6), c["quantity"]): c["rel_gap"] for c in cells}
    labels = [f"ka={ka:g}\n{c}" for ka, c in keys]
    x = np.arange(len(keys))
    w = 0.26
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.2]})
    for k, m in enumerate(MEASURES):
        vals = [items[m][key] * 100 for key in keys]
        ax.bar(x + (k - 1) * w, vals, w, label=m, color=COLORS[m])
    ax.axhline(5.0, color="k", ls="--", lw=1.2, label="case tolerance 5%")
    ax.axhline(3.0, color="k", ls=":", lw=1.0, label="3% standard")
    ax.set_yscale("log")
    ax.set_ylabel("relative error vs Havelock multipole [%]")
    ax.set_title("Case mesh (1280 panels, f/a = 2.0), declared band ka = 0.1/1.0/3.0\n"
                 "flat chords vs exact spherical-patch areas vs scalar uniform control")
    ax.legend(fontsize=8, ncol=5)
    ax.grid(alpha=0.3, axis="y")

    gaps = np.array([gap[key] for key in keys])
    ax2.bar(x, gaps, color=["#2b6cb0" if g < 0 else "#b03030" for g in gaps])
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel("(patch-uniform)/uniform", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = OUT / "fig2_case_mesh_cells.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_prediction(adj: dict) -> Path:
    ra = adj["registered_prediction_RA4"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    pred = ra["predicted_patch_band_max"] * 100
    meas = ra["measured_patch_band_max"] * 100
    flat = ra["committed_flat_band_max"] * 100
    ax.axhspan(pred / 2.0, pred * 2.0, color="#2b6cb0", alpha=0.12,
               label="R-A4 factor-2 acceptance band")
    ax.scatter([0], [flat], color=COLORS["flat"], s=90, zorder=3, label="flat measure (gated table)")
    ax.scatter([1], [pred], marker="X", color="#6b46c1", s=110, zorder=3,
               label="round-6 registered prediction")
    ax.scatter([1], [meas], color=COLORS["patch"], s=110, zorder=3,
               label="measured (recovered run)")
    ax.annotate(f"{meas/pred:.2f}x", (1, meas), textcoords="offset points", xytext=(12, -4),
                fontsize=10, fontweight="bold")
    ax.axhline(5.0, color="k", ls="--", lw=1.0, label="case tolerance 5%")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["case mesh, flat", "case mesh, exact measure"])
    ax.set_ylabel("band-max relative error [%]")
    ax.set_title(f"Registered prediction vs measurement\nverdict: {ra['verdict']}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = OUT / "fig3_registered_prediction.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> int:
    adj = json.loads(ADJ.read_text())
    for fn in (fig_tiers, fig_case_mesh_cells, fig_prediction):
        print(fn(adj))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())