#!/usr/bin/env python3
"""Figures for the m17312 board audit (data straight from audit_summary.json + artifact)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / "runs/m17312_audit"
SUBJECT = json.loads((AUDIT / "source_m17275/artifacts/case_bem_ice_sphere_heave.json").read_text())
SUMMARY = json.loads((AUDIT / "audit_summary.json").read_text())
FIG = REPO / "docs/board_audit/m17312"
FIG.mkdir(parents=True, exist_ok=True)
TOL = SUMMARY["criterion"]["case_tolerance"]


def figures() -> list[Path]:
    out = []
    rows = {r["label"]: r for r in SUBJECT["rows"]}
    ladder = [rows[f"ladder__{n}"] for n in (168, 352, 816, 1472)]
    panels = [r["panels"] for r in ladder]
    e_lamb = [100 * r["A33_relative_error_vs_lamb"] for r in ladder]
    e_exact = [100 * r["A33_relative_error_vs_exact"] for r in ladder]
    offset = 100 * SUMMARY["decomposition"]["physics_offset_exact_vs_lamb"]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(panels, e_lamb, "o-", label="|A33 − Lamb| / Lamb")
    ax.loglog(panels, e_exact, "s--", label="|A33 − multipole| / multipole")
    ax.axhline(100 * TOL, color="crimson", lw=1.2, ls=":", label="case tolerance 5 %")
    ax.axhline(offset, color="grey", lw=1.0, ls="-.", label=f"physics floor at d=3R ({offset:.2f} %)")
    ax.set_xlabel("panels"), ax.set_ylabel("relative error [%]")
    ax.set_title("m17275 mesh ladder: Lamb-referenced error is non-monotone", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    p = FIG / "fig1_ladder_error.png"
    fig.tight_layout(), fig.savefig(p, dpi=150), plt.close(fig)
    out.append(p)

    d = SUMMARY["decomposition"]
    vals = [100 * d["physics_offset_exact_vs_lamb"],
            100 * d["discretisation_undershoot_on_lamb_scale"],
            100 * d["net_error_vs_lamb"]]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    bars = ax.bar(["physics offset\n(exact vs Lamb)", "discretisation undershoot\n(exact vs BEM, Lamb scale)",
                   "net error\n(BEM vs Lamb)"], vals,
                  color=["#4C72B0", "#DD8452", "#55A868"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.12 if v > 0 else -0.28), f"{v:+.2f} %",
                ha="center", fontsize=9)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("contribution to the primary-point error [%]")
    ax.set_title(f"1.63 % agreement = cancellation of +2.96 % and −4.58 % "
                 f"(ratio {d['cancellation_ratio']:.2f})", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    p = FIG / "fig2_error_decomposition.png"
    fig.tight_layout(), fig.savefig(p, dpi=150), plt.close(fig)
    out.append(p)

    ratios = [2.0, 3.0, 5.0, 8.0, 12.0]
    med = [rows[f"depth__{r:g}R__med"] for r in ratios]
    fine = [rows[f"depth__{r:g}R__fine"] for r in ratios]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ratios, [100 * r["A33_relative_error_vs_lamb"] for r in med], "o-", label="BEM 352 panels vs Lamb")
    ax.plot(ratios, [100 * r["A33_relative_error_vs_lamb"] for r in fine], "s-", label="BEM 816 panels vs Lamb")
    ax.plot(ratios, [100 * r["exact_offset_vs_lamb"] for r in med], "k^--", label="exact multipole offset vs Lamb")
    ax.axhline(100 * TOL, color="crimson", lw=1.2, ls=":", label="case tolerance 5 %")
    ax.set_xlabel("submergence d/R"), ax.set_ylabel("relative error [%]")
    ax.set_title("submergence sweep: the BEM error floor is resolution, not submergence", fontsize=10)
    ax.grid(alpha=0.3), ax.legend(fontsize=8)
    p = FIG / "fig3_submergence.png"
    fig.tight_layout(), fig.savefig(p, dpi=150), plt.close(fig)
    out.append(p)

    g = SUMMARY["gate_weakness"]["sensitivity_spread_pct"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    keys = ["ice_thickness_816", "freq_09m", "freq_25m"]
    labels = ["ice thickness\n0–2.5 m", "frequency\n0.3–0.8 rad/s (0.9 m)", "frequency\n0.3–0.8 rad/s (2.5 m)"]
    b = ax.bar(labels, [g[k] for k in keys], color="#8172B3")
    for bar, k in zip(b, keys):
        ax.text(bar.get_x() + bar.get_width() / 2, g[k] + 0.03, f"{g[k]:.2f} %", ha="center", fontsize=9)
    ax.axhline(1.0, color="crimson", lw=1.0, ls=":", label="1 % (\"fraction of a percent\" claim)")
    ax.set_ylabel("full spread of A33 [%]"), ax.set_ylim(0, max(g.values()) * 1.35)
    ax.set_title("sensitivity: frequency spread reaches 1.5 %, not <1 %", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3), ax.legend(fontsize=8)
    p = FIG / "fig4_sensitivity.png"
    fig.tight_layout(), fig.savefig(p, dpi=150), plt.close(fig)
    out.append(p)
    return out


if __name__ == "__main__":
    for path in figures():
        print(path.relative_to(REPO), path.stat().st_size)