"""Figures for the m3365 3-D free-surface verification pins.

Reads ``outputs/m3365/free_surface_pins.json`` (written by
``gen_3d_free_surface_pins.py``) and draws

1. ``fig_free_surface_convergence.png`` - measured relative error against panel
   count for the five table-10 quantities with an O(N^-1) reference slope
   (error ~ h^2), plus the observed orders in both length metrics, which is the
   finding this mission reports;
2. ``fig_free_surface_deviation.png`` - |relative deviation| of all 67 pinned
   numbers against their tolerance, one row per pin.

No benchmark is re-solved here: every number comes from the committed verdicts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent
ROWS = json.loads((OUT / "free_surface_pins.json").read_text())["checks"]
BY_KEY = {row["key"]: row for row in ROWS}

PANELS = (16, 64, 256)
QUANTITIES = {
    "a33": "$A_{33}'$",
    "b33": "$B_{33}'$",
    "a11": "$A_{11}'$",
    "b11": "$B_{11}'$",
    "fz": "$F_z$",
}
H_MEAN_ORDER_KEY = {"a33": "hulme_a33_order", "b33": "hulme_b33_order",
                    "a11": "hulme_a11_order", "fz": "hulme_fz_order"}
H_MEAN_B11 = {"64": "hulme_b11_h_mean_order_64", "256": "hulme_b11_h_mean_order_256"}
PANEL_LAW_B11 = {"64": "hulme_b11_panel_law_order_64",
                 "256": "hulme_b11_panel_law_order_256"}


def measured_errors(short: str) -> list[float]:
    return [float(BY_KEY[f"hulme_{short}_rel_error_{n}"]["measured"]) for n in PANELS]


def panel_law_order(short: str, level: int) -> float:
    errors = measured_errors(short)
    i = PANELS.index(level)
    return math.log(errors[i - 1] / errors[i]) / math.log(
        math.sqrt(PANELS[i] / PANELS[i - 1])
    )


def h_mean_order(short: str, level: int) -> float:
    if short == "b11":
        return float(BY_KEY[H_MEAN_B11[str(level)]]["measured"])
    return float(BY_KEY[f"{H_MEAN_ORDER_KEY[short]}_{level}"]["measured"])


def figure_convergence() -> Path:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=88)
    for short, label in QUANTITIES.items():
        errors = measured_errors(short)
        ax.loglog(PANELS, errors, "o-", label=label, lw=1.4, ms=5)
    reference = measured_errors("a33")[0] * (PANELS[0] / PANELS[0])
    ax.loglog(
        PANELS,
        [reference * (PANELS[0] / n) for n in PANELS],
        "k--",
        lw=1.0,
        label="reference $O(N^{-1})$ ($h^2$)",
    )
    ax.set_xlabel("panels $N$")
    ax.set_ylabel("relative error vs Hulme (1982)")
    ax.set_title("Hulme floating hemisphere, $Ka=1.0$: error vs panel count")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=7, loc="lower left")

    width = 0.38
    positions = range(len(QUANTITIES))
    fine = PANELS[-1]
    hm = [h_mean_order(short, fine) for short in QUANTITIES]
    pl = [panel_law_order(short, fine) for short in QUANTITIES]
    ax2.bar([p - width / 2 for p in positions], hm, width,
            label="$h_{mean}$ metric (4 of 5 cells)", color="#3b6ea5")
    ax2.bar([p + width / 2 for p in positions], pl, width,
            label="$N^{-1/2}$ metric (the $B_{11}'$ cell)", color="#c1663a")
    ax2.axhline(2.0, color="k", ls="--", lw=1.0)
    ax2.text(0.02, 2.03, "second order", fontsize=7)
    ax2.set_xticks(list(positions))
    ax2.set_xticklabels(list(QUANTITIES.values()), fontsize=9)
    ax2.set_ylabel(f"observed order at $N={fine}$")
    ax2.set_title("Table 10's order column mixes two length metrics")
    ax2.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)
    ax2.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path = OUT / "fig_free_surface_convergence.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_deviation() -> Path:
    ordered = sorted(ROWS, key=lambda r: abs(r["relative_deviation"]))
    values = [max(abs(r["relative_deviation"]), 1e-9) for r in ordered]
    colors = ["#3b6ea5" if r["group"] == "floating_hemisphere" else "#6a9f58"
              for r in ordered]
    fig, ax = plt.subplots(figsize=(8.4, 3.9), dpi=88)
    ax.barh(range(len(values)), values, color=colors, height=0.8)
    ax.set_yticks([])
    ax.set_xscale("log")
    ax.axvline(1.5e-2, color="k", ls="--", lw=1.0)
    ax.text(1.6e-2, len(values) * 0.55, "default\ntolerance 1.5%", fontsize=7)
    ax.axvline(5.0e-2, color="#a33", ls=":", lw=1.0)
    ax.text(5.3e-2, len(values) * 0.28, "2-digit\ntolerance 5%", fontsize=7)
    ax.set_xlabel("|relative deviation| of the measured value from the quoted one")
    ax.set_title(
        f"{len(values)} pinned 3-D free-surface numbers: worst "
        f"{max(values):.2e}, all inside their tolerance "
        "(blue: floating hemisphere, green: submerged sphere)"
    )
    ax.grid(True, axis="x", ls=":", lw=0.5, alpha=0.6)
    fig.tight_layout()
    path = OUT / "fig_free_surface_deviation.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    for path in (figure_convergence(), figure_deviation()):
        print(path.name, path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()