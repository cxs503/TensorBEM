#!/usr/bin/env python3
"""Generate the paper-exp figures for the propeller-openwater-bem mission (m27906).

Figures are drawn from the committed artefacts (never hand-typed):
  * benchmarks/anchors/kp505_openwater.json            -- anchor transcription
  * benchmarks/kp505_openwater_bem/verdict.json        -- BEM study
  * benchmarks/verified/propeller_openwater/kp505_gci_ladder_gci.csv -- GCI
  * benchmarks/kp505_openwater_bem/sensitivity.csv     -- sensitivity

Output PNGs are written to docs/figures/ (worktree path).
Run:  python -m tools.make_kp505_openwater_figures
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
ANCHOR = REPO / "benchmarks" / "anchors" / "kp505_openwater.json"
BEM_VERDICT = REPO / "benchmarks" / "kp505_openwater_bem" / "verdict.json"
GCI_CSV = REPO / "benchmarks" / "verified" / "propeller_openwater" / "kp505_gci_ladder_gci.csv"
SENS = REPO / "benchmarks" / "kp505_openwater_bem" / "sensitivity.csv"
OUT_DIR = REPO / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def fig1_openwater_curves(art: dict) -> str:
    """Anchor open-water curves: KT(J), 10KQ(J), eta0(J) with peak + design marked."""
    rows = art["open_water_table"]
    J = [r["J"] for r in rows]
    KT = [r["KT"] for r in rows]
    KQ10 = [r["10KQ"] for r in rows]
    eta = [r["eta0"] for r in rows]
    peak = art["peak_efficiency"]
    dp = art["design_point"]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    ax1.plot(J, KT, "o-", color="#1f77b4", lw=1.6, label=r"$K_T$ (anchor)")
    ax1.plot(J, KQ10, "s-", color="#ff7f0e", lw=1.6, label=r"$10K_Q$ (anchor)")
    ax1.set_xlabel(r"Advance ratio $J = V_a/(nD)$")
    ax1.set_ylabel(r"$K_T$, $10K_Q$")
    ax1.set_ylim(-0.05, 0.75)

    ax2 = ax1.twinx()
    ax2.plot(J, eta, "^-", color="#2ca02c", lw=1.6, label=r"$\eta_0$ (anchor)")
    ax2.set_ylabel(r"$\eta_0$")
    ax2.set_ylim(0.0, 0.8)

    ax1.axvline(dp["J"], color="gray", ls=":", lw=1)
    ax1.axvline(peak["J"], color="gray", ls="--", lw=1)
    ax1.annotate(
        "design J = 0.70\n$\\eta_0$ = %.4f" % dp["eta0"],
        xy=(dp["J"], dp["eta0"]),
        xytext=(0.42, 0.680),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.7),
    )
    ax1.annotate(
        "peak J = 0.80\n$\\eta_0$ = %.4f" % peak["eta0"],
        xy=(peak["J"], peak["eta0"]),
        xytext=(0.62, 0.72),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.7),
    )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, framealpha=0.9)
    ax1.set_title("KP505 open-water anchor (Table 2, p.10, MARINE 2023)")
    fig.tight_layout()
    p = OUT_DIR / "fig1_kp505_openwater_anchor.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def fig2_bem_vs_anchor(vd: dict) -> str:
    """BEM vs anchor deviations: dKT, d10KQ, deta0 vs J (all 10 points)."""
    pts = vd["points"]
    J = [p["J"] for p in pts]
    dkt = [p["dKT_pct"] for p in pts]
    dkq = [p["d10KQ_pct"] for p in pts]
    deta = [p["deta0_pct"] for p in pts]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(J, dkt, "o-", color="#1f77b4", lw=1.6, label=r"$\Delta K_T$ (%)")
    ax.plot(J, dkq, "s-", color="#ff7f0e", lw=1.6, label=r"$\Delta 10K_Q$ (%)")
    ax.plot(J, deta, "^-", color="#2ca02c", lw=1.6, label=r"$\Delta \eta_0$ (%)")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(-3, 3, color="green", alpha=0.08, label="|d| <= 3% band")
    ax.axhspan(-5, 5, color="orange", alpha=0.06, label="|d| <= 5% band")
    ax.set_xlabel(r"Advance ratio $J$")
    ax.set_ylabel("Deviation vs anchor (%)")
    ax.set_ylim(-80, 40)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Vortex-lattice BEM vs KP505 anchor (all 10 points, honest FAIL)")
    fig.tight_layout()
    p = OUT_DIR / "fig2_bem_vs_anchor.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def fig3_gci_sensitivity() -> str:
    """GCI ladder (KT vs J) and sensitivity of the design point."""
    gci_rows = []
    if GCI_CSV.exists():
        with open(GCI_CSV) as fh:
            for row in csv.DictReader(fh):
                gci_rows.append(row)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    # left: GCI on KT vs J (spanwise/radial ladder)
    ax = axes[0]
    if gci_rows:
        js, gcis = [], []
        for row in gci_rows:
            try:
                axis = row.get("axis", "")
                j = float(row.get("J", ""))
                q = row.get("quantity", "")
                gci = float(row.get("gci_fine_pct_fine_triplet", ""))
            except (ValueError, TypeError):
                continue
            if q == "KT" and axis == "spanwise":
                js.append(j)
                gcis.append(gci)
        if js:
            ax.plot(js, gcis, "o-", color="#1f77b4", lw=1.5)
            ax.set_xlabel(r"Advance ratio $J$")
            ax.set_ylabel("GCI on $K_T$ (%)")
            ax.set_title("Radial-ladder GCI on $K_T$ (r = 2)")
        else:
            axes[0].text(0.5, 0.5, "no spanwise KT rows", ha="center")
    else:
        axes[0].text(0.5, 0.5, "GCI ladder CSV not found", ha="center")

    # right: sensitivity of dKT at J=0.7 (from sensitivity.csv)
    sens_rows = []
    with open(SENS) as fh:
        for row in csv.DictReader(fh):
            sens_rows.append(row)
    ax = axes[1]
    labels = []
    vals = []
    for row in sens_rows:
        try:
            j = float(row["J"])
            var = row["variant"]
            dkt = float(row["dKT_pct"])
        except (KeyError, ValueError):
            continue
        if abs(j - 0.7) < 1e-9 and var != "baseline":
            labels.append(var)
            vals.append(dkt)
    if labels:
        ax.barh(labels, vals, color="#d62728")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(r"$\Delta K_T$ at J=0.7 (%)")
        ax.set_title("Sensitivity of $K_T$ (J = 0.7)")
    else:
        axes[1].text(0.5, 0.5, "sensitivity CSV not found", ha="center")

    fig.tight_layout()
    p = OUT_DIR / "fig3_gci_sensitivity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def main() -> None:
    art = json.loads(ANCHOR.read_text())
    vd = json.loads(BEM_VERDICT.read_text())
    p1 = fig1_openwater_curves(art)
    p2 = fig2_bem_vs_anchor(vd)
    p3 = fig3_gci_sensitivity()
    print("figures written:")
    for p in (p1, p2, p3):
        print("  ", p)


if __name__ == "__main__":
    main()
