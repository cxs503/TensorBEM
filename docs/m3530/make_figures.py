"""Figures for the m3530 three-way separation study (reads the JSON, no hand data).

Run: python docs/m3530/make_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
PAYLOAD = json.loads((HERE / "near_surface_three_way.json").read_text(encoding="utf-8"))

CRUX = (1.05, 1.0)
GATE = 3.0e-2


def crux_row():
    return next(
        row for row in PAYLOAD["rows"]
        if (row["depth_ratio"], row["ka"]) == CRUX
    )


def figure_separation() -> Path:
    row = crux_row()
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    axis = axes[0]
    rungs = sorted(row["panels"].items(), key=lambda item: int(item[0]))
    spacing = [rung["panel_spacing"] for _, rung in rungs]
    for quantity, marker, colour in (
        ("added_mass", "o", "tab:blue"),
        ("damping", "s", "tab:red"),
    ):
        errors = [rung[f"{quantity}_error"] for _, rung in rungs]
        axis.loglog(spacing, errors, marker + "-", color=colour,
                    label=f"panel, {quantity.replace('_', ' ')}")
    mfs_spacing = rungs[-1][1]["panel_spacing"]
    for quantity, marker, colour in (
        ("added_mass", "o", "tab:blue"),
        ("damping", "s", "tab:red"),
    ):
        axis.scatter([mfs_spacing * 1.15], [row["mfs"]["48"][f"{quantity}_error"]],
                     marker=marker, facecolors="none", edgecolors=colour, s=90,
                     label=f"MFS 48 sources, {quantity.replace('_', ' ')}")
    axis.axhline(GATE, color="k", linestyle="--", linewidth=1.0)
    axis.annotate("3 % gate", xy=(spacing[-1], GATE), xytext=(-70, 4),
                  textcoords="offset points", fontsize=9)
    first, last = rungs[0][1], rungs[-1][1]
    slope = math.log(first["added_mass_error"] / last["added_mass_error"]) / math.log(
        first["panel_spacing"] / last["panel_spacing"]
    )
    axis.set_xlabel("panel spacing $h = 2\\pi a / N$  [m]")
    axis.set_ylabel("relative error vs Ursell multipole reference")
    axis.set_title(
        f"a) $f/a={CRUX[0]}$, $ka={CRUX[1]}$: panel error decays ~$h^{{{slope:.2f}}}$,"
        " MFS stays a floor below"
    )
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8, loc="lower left")

    axis = axes[1]
    depths = sorted({row["depth_ratio"] for row in PAYLOAD["rows"]})
    for ka, marker, linestyle in ((0.6, "o", "-"), (1.0, "s", "--")):
        values = []
        for depth in depths:
            row = next(r for r in PAYLOAD["rows"]
                       if r["depth_ratio"] == depth and r["ka"] == ka)
            worst = max(row["panels"]["256"]["added_mass_error"],
                        row["panels"]["256"]["damping_error"])
            values.append(worst)
        axis.semilogy(depths, values, marker + linestyle, label=f"panel 256, $ka={ka}$")
        mfs_values = [
            max(next(r for r in PAYLOAD["rows"]
                     if r["depth_ratio"] == depth and r["ka"] == ka)["mfs"]["48"]
                ["added_mass_error"],
                next(r for r in PAYLOAD["rows"]
                     if r["depth_ratio"] == depth and r["ka"] == ka)["mfs"]["48"]
                ["damping_error"])
            for depth in depths
        ]
        axis.semilogy(depths, mfs_values, marker=marker, linestyle=":",
                      color=axis.lines[-1].get_color(),
                      label=f"MFS 48, $ka={ka}$")
    axis.axhline(GATE, color="k", linestyle="--", linewidth=1.0)
    axis.set_xlabel("submergence ratio $f/a$")
    axis.set_ylabel("worst of added-mass / damping relative error")
    axis.set_title("b) bar is the panel method everywhere; MFS is fine near the surface")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)

    figure.tight_layout()
    out = HERE / "fig_separation.png"
    figure.savefig(out, dpi=100)
    plt.close(figure)
    return out


if __name__ == "__main__":
    for produced in (figure_separation(),):
        print(f"wrote {produced} ({produced.stat().st_size} bytes)")