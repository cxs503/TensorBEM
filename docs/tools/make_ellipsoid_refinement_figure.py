"""Figure for the ellipsoid refinement study (``ellipsoid_refinement.json``).

Two panels:

* left  -- discretisation errors (faceted volume, and added mass along each
  axis) against the mean panel size on log-log axes, with reference slopes;
* right -- added mass against ``1/N`` for each axis, straight-line fits and the
  extrapolated intercepts, plus Lamb's analytic value as a star. A star sitting
  on the intercept is the graphical proof that the reference is right.

Usage::

    python docs/tools/make_ellipsoid_refinement_figure.py \
        --input outputs/ellipsoid_heavy/ellipsoid_refinement.json \
        --output outputs/ellipsoid_heavy/ellipsoid_refinement.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

AXES = ("x", "y", "z")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    meshes = report["meshes"]
    h = [row["h_mean"] for row in meshes]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # --- left: error decay -------------------------------------------------
    volume_errors = [row["volume_relative_error"] for row in meshes]
    left.loglog(h, volume_errors, "o-", label="enclosed-volume error", color="0.35")
    colours = {"x": "tab:blue", "y": "tab:orange", "z": "tab:green"}
    for name in AXES:
        levels = report["axes"][name]["levels"]
        errors = [row["added_mass_relative_error"] for row in levels]
        order = report["axes"][name].get("error_decay_order")
        label = f"added mass {name}"
        if order is not None:
            label += f" (order {order:.2f})"
        left.loglog(h, errors, "s-", label=label, color=colours[name])
    # reference slopes anchored on the finest volume error
    anchor_h, anchor_e = h[-1], volume_errors[-1]
    for order, style in ((1.0, ":"), (2.0, "--")):
        left.loglog(
            [h[0], h[-1]],
            [anchor_e * (h[0] / anchor_h) ** order, anchor_e],
            style,
            color="0.6",
            linewidth=1.0,
            label=f"slope {order:g}",
        )
    left.set_xlabel("mean panel size $h$ (m)")
    left.set_ylabel("relative error")
    left.set_title("discretisation error vs mesh size")
    left.grid(True, which="both", alpha=0.3)
    left.legend(fontsize=7)

    # --- right: extrapolation ---------------------------------------------
    for name in AXES:
        entry = report["axes"][name]
        levels = entry["levels"]
        inverse_n = [1.0 / row["panels"] for row in levels]
        masses = [row["added_mass"] for row in levels]
        right.plot(
            inverse_n,
            masses,
            "o",
            color=colours[name],
            label=f"{name} panels",
        )
        if "extrapolation" in entry:
            limit = entry["extrapolation"]["extrapolated_added_mass"]
            right.plot([0.0, inverse_n[-1]], [limit, masses[-1]], "-", color=colours[name])
            right.plot([0.0], [limit], "*", markersize=13, color=colours[name])
        right.axhline(
            levels[-1]["exact_added_mass"],
            color=colours[name],
            linestyle=":",
            linewidth=1.0,
        )
    right.set_xlabel("$1/N$ (panel count)")
    right.set_ylabel("added mass (kg)")
    right.set_title("extrapolation to zero mesh size\n(stars = limit, dotted = Lamb)")
    right.grid(True, alpha=0.3)
    right.legend(fontsize=7)

    fig.suptitle(
        "translating ellipsoid (1.5, 1.0, 0.75) m: "
        + "/".join(str(row["panels"]) for row in meshes)
        + " panels",
        fontsize=11,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())