#!/usr/bin/env python3
"""Figure for the m22460 tail-budget certificate (four panels, artifact-driven).

Panels
------
(a) per DOF: closed-form budget, probe-measured omission, and the C3a residual --
    all normalised by max|A| of the same DOF, so the three quantities are
    directly comparable;
(b) measured omission against the C3a residual, with the y = x line;
(c) sensitivity of the certification to a synthetic resonance at 17 rad/s
    (peak / damping at the band top), for the three certified DOF;
(d) tail exponent per DOF on the committed coarse and medium meshes.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAYLOAD = json.loads(
    (ROOT / "docs" / "case_bem_cummins_wigley_tail_certificate.json").read_text()
)
OUT = ROOT / "docs" / "fig_m22460_cummins_tail_certificate.png"
NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def main() -> None:
    per_dof = PAYLOAD["per_dof"]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))

    # (a) budget vs measured omission vs residual
    ax = axes[0, 0]
    x = np.arange(len(NAMES))
    width = 0.27
    budget = [per_dof[n]["budget_relative_max"] or np.nan for n in NAMES]
    omission = [per_dof[n]["omission_lower_relative_max"] for n in NAMES]
    residual = [per_dof[n]["c3a_residual"] for n in NAMES]
    ax.bar(x - width, budget, width, label=r"power-tail budget $\beta$", color="#4c72b0")
    ax.bar(x, omission, width, label="probe-measured omission (lower bound)", color="#dd8452")
    ax.bar(x + width, residual, width, label="C3a residual", color="#8c8c8c")
    for index, name in enumerate(NAMES):
        if not np.isfinite(budget[index]):
            ax.annotate(
                "no budget\n(uncertified)",
                (index - width, 0.004),
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
                color="#4c72b0",
            )
    ax.axhline(PAYLOAD["tolerance"], ls="--", lw=1.0, color="crimson")
    ax.annotate("case tolerance 10%", (len(NAMES) - 0.6, PAYLOAD["tolerance"]),
                ha="right", va="bottom", fontsize=8, color="crimson")
    ax.set_xticks(x, NAMES)
    ax.set_ylabel(r"fraction of $\max|A|$")
    ax.set_title("(a) truncation budget vs the tail actually measured\n"
                 "all three normalised by the same scale")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, 0.105)

    # (b) omission vs residual
    ax = axes[0, 1]
    for name in NAMES:
        row = per_dof[name]
        certified = row["verdict"] == "certified"
        ax.scatter(
            row["c3a_residual"], row["omission_lower_relative_max"],
            s=70, marker="o" if certified else "s",
            color="#55a868" if certified else "#c44e52",
            label=("certified" if certified else "uncertified") if name in ("heave", "surge") else None,
            zorder=3,
        )
        ax.annotate(name, (row["c3a_residual"], row["omission_lower_relative_max"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    lim = [0.0, 0.10]
    ax.plot(lim, lim, ls=":", color="k", lw=1.0)
    ax.annotate(r"omission $=$ residual", (0.055, 0.058), rotation=38, fontsize=8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"C3a residual $|\Delta A|/\max|A|$")
    ax.set_ylabel("probe-measured omission (lower bound)")
    ax.set_title("(b) the omitted band is the same size as the residual\n"
                 "so C3a cannot separate causality from truncation")
    ax.legend(fontsize=8, loc="upper left")

    # (c) resonance sensitivity for the certified DOF
    ax = axes[1, 0]
    colors = {"heave": "#4c72b0", "roll": "#dd8452", "pitch": "#55a868"}
    for name, block in PAYLOAD["sensitivity"].items():
        rows = block["rows"]
        ratio = [row["peak_over_damping_at_band_top"] for row in rows]
        ax.plot(ratio, [row["measured_over_budget"] for row in rows],
                marker="o", color=colors.get(name, "k"), label=name)
    ax.axhline(1.0, ls="--", color="k", lw=1.0)
    ax.annotate("budget exactly conservative", (30, 1.05), ha="right", fontsize=8)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel(r"unsolved resonance peak at 17 rad/s / $B(\omega_{max})$")
    ax.set_ylabel("measured omission / budget")
    ax.set_title("(c) how large an unsolved resonance the certification survives")
    ax.legend(fontsize=8)

    # (d) tail exponent, coarse vs medium
    ax = axes[1, 1]
    medium_p = [per_dof[n]["tail_exponent"] for n in NAMES]
    coarse_p = [per_dof[n]["coarse_mesh"]["tail_exponent"] for n in NAMES]
    ax.bar(x - width / 1.5, coarse_p, 0.5, label="coarse (192 panels)", color="#937860")
    ax.bar(x + width / 1.5, medium_p, 0.5, label="medium (480 panels)", color="#8172b2")
    ax.axhline(2.05, ls="--", color="crimson", lw=1.0)
    ax.annotate("admissibility of a bounded tail (p > 2.05)", (0.0, 2.12), fontsize=8, color="crimson")
    ax.set_xticks(x, NAMES)
    ax.set_ylabel(r"measured tail exponent $p$")
    ax.set_title("(d) the certified set {heave, roll, pitch} is mesh-robust")
    ax.legend(fontsize=8)

    fig.suptitle(
        "CASE-BEM-CUMMINS-WIGLEY: auditing the truncation budget behind gate C3a "
        "(mission m22460)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()