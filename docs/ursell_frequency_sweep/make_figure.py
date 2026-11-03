"""Regenerate the sweep figure from ``results.json`` (no hand-drawn numbers).

    PYTHONPATH=src python3 docs/ursell_frequency_sweep/make_figure.py

Panel (a): added-mass relative error against the closed-form multipole reference
versus ``ka`` at the finest panel count, one curve per submergence ratio, with
the 3 % acceptance threshold drawn.
Panel (b): the same error versus panel count at two points on a log-log scale,
with a first-order slope drawn for comparison.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECORD = HERE / "results.json"
FIGURE = REPO / "docs" / "figures" / "ursell_frequency_sweep.png"


def main() -> int:
    record = json.loads(RECORD.read_text())
    rows = record["rows"]
    settings = record["settings"]
    finest = max(settings["panel_counts"])
    ratios = [value for value in settings["a_over_f_values"]]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    colours = plt.cm.viridis([i / max(len(ratios) - 1, 1) for i in range(len(ratios))])
    for colour, ratio in zip(colours, ratios):
        points = sorted(
            (
                row
                for row in rows
                if row["a_over_f"] == ratio and row["panels"] == finest
            ),
            key=lambda row: row["ka"],
        )
        left.semilogy(
            [row["ka"] for row in points],
            [row["error_added_mass"] for row in points],
            "o-",
            color=colour,
            label=f"a/f = {ratio:g}",
        )
        for row in points:
            left.annotate(
                f"{row['error_added_mass'] * 100:.2f}%",
                (row["ka"], row["error_added_mass"]),
                textcoords="offset points",
                xytext=(0, 6),
                fontsize=6.5,
                color=colour,
            )
    left.axhline(3.0e-2, color="crimson", linestyle="--", linewidth=1.0)
    left.annotate(
        "3 % acceptance threshold",
        (settings["ka_values"][0], 3.0e-2),
        textcoords="offset points",
        xytext=(2, 4),
        fontsize=7,
        color="crimson",
    )
    left.set_xlabel(r"non-dimensional frequency $ka$")
    left.set_ylabel("relative error in added mass")
    left.set_title(f"Ursell panel solution, {int(finest)} panels")
    left.grid(True, which="both", alpha=0.3)
    left.legend(fontsize=8)

    highlight = [
        (row["ka"], row["a_over_f"])
        for row in sorted(rows, key=lambda item: -item["error_added_mass"])
        if row["panels"] == finest
    ][:2]
    for (ka, ratio), colour in zip(highlight, ("tab:blue", "tab:orange")):
        points = sorted(
            (
                row
                for row in rows
                if row["a_over_f"] == ratio and row["ka"] == ka
            ),
            key=lambda row: row["panels"],
        )
        right.loglog(
            [row["panels"] for row in points],
            [row["error_added_mass"] for row in points],
            "o-",
            color=colour,
            label=f"ka={ka:g}, a/f={ratio:g}",
        )
        panels = [row["panels"] for row in points]
        anchor = points[0]["error_added_mass"] * panels[0]
        right.loglog(
            panels,
            [anchor / value for value in panels],
            ":",
            color=colour,
            linewidth=1.0,
            label=None,
        )
    right.loglog([], [], "k:", label=r"first order, $N\epsilon$ = const")
    right.set_xlabel("panels on the cylinder")
    right.set_ylabel("relative error in added mass")
    right.set_title("Convergence with panel refinement")
    right.grid(True, which="both", alpha=0.3)
    right.legend(fontsize=8)

    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=160)
    print(f"wrote {FIGURE} ({FIGURE.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())