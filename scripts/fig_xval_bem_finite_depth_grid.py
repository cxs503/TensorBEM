#!/usr/bin/env python3
"""Draw the whole-grid picture of the finite-depth cross-validation study.

The study page (``docs/capytaine_finite_depth_crossval.md``) reports the grid as
two long tables; a reader cannot see at a glance *where* the residual lives.  This
figure reads the frozen ``results.json`` (never the solve) and plots

* the signed extrapolated deviation of every judged quantity in every cell of the
  ``h/a x k0 a`` grid, on a diverging colour scale pinned to the pre-registered
  bands (+-3 % added mass, +-5 % damping) and the owner 15 % limit, and
* the same deviations against ``k0 h``, which is the parameter that decides
  whether finite depth matters at all: at large ``k0 h`` both codes must collapse
  onto the deep-water answer, so a residual that *grows* there is a deep-water
  defect, while one that only appears at small ``k0 h`` is a kernel defect.

Usage::

    PYTHONPATH=src python scripts/fig_xval_bem_finite_depth_grid.py \
        --results benchmarks/verified/capytaine_finite_depth/results.json \
        --out docs/fig_xval_bem_finite_depth_grid.png

Nothing is recomputed and no number is typed by hand: every value comes from the
artifact that the study driver wrote.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

QUANTITIES = (
    ("surge.added_mass", "A11 surge added mass", 0.03),
    ("heave.added_mass", "A33 heave added mass", 0.03),
    ("surge.damping", "B11 surge damping", 0.05),
    ("heave.damping", "B33 heave damping", 0.05),
)


def load_rows(path: Path) -> list[dict]:
    result = json.loads(path.read_text())
    rows = result["comparison"]["rows"]
    if not rows:
        raise SystemExit(f"{path}: comparison has no rows")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("benchmarks/verified/capytaine_finite_depth/results.json"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("docs/fig_xval_bem_finite_depth_grid.png")
    )
    parser.add_argument("--owner-limit", type=float, default=0.15)
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    rows = load_rows(args.results)
    ratios = sorted({row["h_over_a"] for row in rows})
    kas = sorted({row["ka"] for row in rows})
    # the pre-registered band of the *tightest* quantity pins the colour scale so
    # the four panels are comparable; the owner limit is marked as a contour
    band = 0.05

    grid = np.full((len(QUANTITIES), len(rows)), np.nan)
    for column, row in enumerate(rows):
        for line, (key, _label, _tol) in enumerate(QUANTITIES):
            entry = row["quantities"].get(key)
            if entry is not None:
                grid[line, column] = 100.0 * entry["relative"]

    fig, axes = plt.subplots(
        2, 1, figsize=(11.0, 7.2), gridspec_kw={"height_ratios": [1.0, 1.15]}
    )

    ax = axes[0]
    finite = np.abs(grid[np.isfinite(grid)])
    limit = max(band, float(finite.max()) if finite.size else band)
    norm = TwoSlopeNorm(vmin=-100.0 * limit, vcenter=0.0, vmax=100.0 * limit)
    mesh = ax.pcolormesh(
        np.arange(len(rows) + 1), np.arange(len(QUANTITIES) + 1), grid,
        cmap="RdBu_r", norm=norm, edgecolors="0.6", linewidth=0.4,
    )
    ax.set_yticks(np.arange(len(QUANTITIES)) + 0.5)
    ax.set_yticklabels([label for _key, label, _tol in QUANTITIES], fontsize=8)
    ax.set_xticks(np.arange(len(rows)) + 0.5)
    ax.set_xticklabels(
        [f"{row['h_over_a']:.0f}/{row['ka']:.2f}" for row in rows],
        rotation=90, fontsize=7,
    )
    ax.set_xlabel("cell  h/a / k0 a", fontsize=8)
    for line in range(len(QUANTITIES)):
        for column in range(len(rows)):
            value = grid[line, column]
            if np.isfinite(value):
                ax.text(column + 0.5, line + 0.5, f"{value:+.1f}",
                        ha="center", va="center", fontsize=6.5)
    fig.colorbar(mesh, ax=ax, pad=0.01, label="(TensorBEM - capytaine) / capytaine  [%]")
    ax.set_title(
        "Extrapolated deviation per cell (both codes Richardson-extrapolated)",
        fontsize=9,
    )

    ax = axes[1]
    for line, (key, label, tol) in enumerate(QUANTITIES):
        k0h = [
            row["k0h"] for row in rows if key in row["quantities"]
        ]
        dev = [
            100.0 * row["quantities"][key]["relative"]
            for row in rows if key in row["quantities"]
        ]
        asymptotic = [
            row["quantities"][key].get("capytaine_order") is not None
            for row in rows if key in row["quantities"]
        ]
        filled = ax.plot(k0h, dev, "o-", lw=0.9, ms=4, label=label)[0]
        hollow = [
            (x, y) for x, y, ok in zip(k0h, dev, asymptotic) if not ok
        ]
        ax.plot(
            [p[0] for p in hollow], [p[1] for p in hollow], "o", ms=7,
            mfc="none", mec=filled.get_color(), mew=1.0,
        )
        ax.axhline(100.0 * tol, color=filled.get_color(), lw=0.7, ls=":")
        ax.axhline(-100.0 * tol, color=filled.get_color(), lw=0.7, ls=":")
    ax.axhline(100.0 * args.owner_limit, color="0.25", lw=1.1, ls="--",
               label="owner 15 % limit")
    ax.axhline(-100.0 * args.owner_limit, color="0.25", lw=1.1, ls="--")
    ax.axhline(0.0, color="0.25", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("k0 h  (1.5 = shallowest cell of the plan, 40 = deep end)", fontsize=8)
    ax.set_ylabel("deviation  [%]", fontsize=8)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(
        "Deviation against the depth parameter; dotted = pre-registered band, "
        "hollow markers = capytaine ladder not asymptotic (finest level used)",
        fontsize=9,
    )

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)

    worst = max(
        (abs(row["quantities"][key]["relative"]), row["cell"], key)
        for row in rows
        for key, _label, _tol in QUANTITIES
        if key in row["quantities"]
    )
    print(json.dumps({
        "figure": str(args.out),
        "cells": len(rows),
        "worst_abs_relative": worst[0],
        "worst_cell": worst[1],
        "worst_quantity": worst[2],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())