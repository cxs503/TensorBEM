#!/usr/bin/env python3
"""Generate figures for CASE-BEM-CUMMINS-WIGLEY Round 7 report.

Produces:
- fig_kk_closure.png: KK closure residuals (C3 vs C3a) per DOF
- fig_anchor_consistency.png: A_inf fitted vs anchor per DOF
- fig_gate_summary.png: All 13 gate pass/fail summary
"""
from __future__ import annotations

import json
import base64
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACT = Path("benchmarks/verified/casepool_bem_cummins_wigley/case_result.json")
OUTDIR = Path("artifacts/case_bem_cummins_wigley_round7")
DOF_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def load_data():
    with open(ARTIFACT) as f:
        return json.load(f)


def fig_kk_closure(data):
    """KK closure residuals: C3 (fitted) vs C3a (anchored) per DOF."""
    c3 = data["gates"]["C3"]["values"]
    c3a = data["gates"]["C3a"]["values"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(DOF_NAMES))
    width = 0.35

    c3_vals = [d["max_relative_residual"] for d in c3["per_dof"]]
    c3a_vals = [d["max_relative_residual"] for d in c3a["per_dof"]]

    bars1 = ax.bar(x - width/2, [v*100 for v in c3_vals], width, label="C3 (fitted A_inf)", color="#4C72B0")
    bars2 = ax.bar(x + width/2, [v*100 for v in c3a_vals], width, label="C3a (anchored A_inf, 0 free params)", color="#DD8452")

    ax.axhline(y=10, color="red", linestyle="--", linewidth=1, label="10% tolerance")
    ax.set_xticks(x)
    ax.set_xticklabels(DOF_NAMES)
    ax.set_ylabel("Max relative residual (%)")
    ax.set_title("KK Closure: Fitted vs Anchored A_inf (medium mesh)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = OUTDIR / "fig_kk_closure.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_anchor_consistency(data):
    """A_inf fitted vs anchor per DOF."""
    c7 = data["gates"]["C7"]["values"]["per_dof"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(DOF_NAMES))
    width = 0.35

    fitted = [d["added_mass_inf_fitted"] for d in c7]
    anchor = [d["added_mass_inf_anchor"] for d in c7]

    bars1 = ax.bar(x - width/2, fitted, width, label="Fitted A_inf (KK closure)", color="#4C72B0")
    bars2 = ax.bar(x + width/2, anchor, width, label="Anchor A_inf (image method)", color="#55A868")

    ax.set_xticks(x)
    ax.set_xticklabels(DOF_NAMES)
    ax.set_ylabel("A_inf (kg)")
    ax.set_title("A_inf: Fitted vs Physical Anchor (medium mesh)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add relative error labels
    for i, d in enumerate(c7):
        rel = d["relative_to_anchor"] * 100
        ax.text(x[i] + width/2, max(fitted[i], anchor[i]) + max(fitted[i], anchor[i])*0.02,
                f"Δ={rel:.1f}%", ha="center", va="bottom", fontsize=7, color="red")

    fig.tight_layout()
    path = OUTDIR / "fig_anchor_consistency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_gate_summary(data):
    """All 13 gates pass/fail summary."""
    gates = data["gates"]
    gate_ids = ["C1", "C0", "C0_coarse", "C2", "C3", "C3a", "C7",
                "C3_coarse", "C3a_coarse", "C4", "C5", "C6a", "C6b"]
    gate_titles = [
        "C1: Geometry", "C0: Solver (med)", "C0: Solver (coarse)",
        "C2: Panel res", "C3: KK (fitted)", "C3a: KK (anchored)",
        "C7: A_inf anchor", "C3: KK coarse", "C3a: KK coarse",
        "C4: Round trip", "C5: Free decay", "C6a: Neg ctrl (B=0)",
        "C6b: Neg ctrl (corrupt)"
    ]
    passed = [gates[g]["passed"] for g in gate_ids]

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#55A868" if p else "#C44E52" for p in passed]
    bars = ax.barh(range(len(gate_ids)), [1]*len(gate_ids), color=colors, edgecolor="white")

    ax.set_yticks(range(len(gate_ids)))
    ax.set_yticklabels(gate_titles, fontsize=9)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Status")
    ax.set_title(f"CASE-BEM-CUMMINS-WIGLEY: {sum(passed)}/{len(passed)} gates passed")
    ax.set_xticks([])

    for i, (g, p) in enumerate(zip(gate_ids, passed)):
        ax.text(0.5, i, "PASS" if p else "FAIL", ha="center", va="center",
                fontweight="bold", color="white", fontsize=10)

    fig.tight_layout()
    path = OUTDIR / "fig_gate_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def png_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    p1 = fig_kk_closure(data)
    p2 = fig_anchor_consistency(data)
    p3 = fig_gate_summary(data)

    print(f"Figures written to {OUTDIR}:")
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  {p3}")

    # Print base64 for report embedding
    for p in [p1, p2, p3]:
        print(f"\n{p.name} base64 (first 80 chars): {png_base64(p)[:80]}...")


if __name__ == "__main__":
    main()
