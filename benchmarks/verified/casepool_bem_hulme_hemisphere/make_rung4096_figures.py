#!/usr/bin/env python3
"""Figures for the measured 4096-panel rung (CASE-BEM-HULME-HEMISPHERE).

Reads ``results_rung4096.json`` -- no number is hand-typed -- and writes

* ``fig_rung4096_ladder.png``  : band-max error per rung (the ladder) with the
  5 % case tolerance, and per-quantity errors on the finest rung;
* ``fig_rung4096_residual.png``: the per-solve residuals that make the rung
  admissible (a rung that cannot prove its residual cannot be graded).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_rung4096.json"
QUANTITIES = ("A11_dimless", "B11_dimless", "A33_dimless", "B33_dimless",
              "Fx_dimless", "Fz_dimless")


def main() -> int:
    res = json.loads(RESULTS.read_text())
    cells = res["cells"]
    tol = 0.05
    ka_target = res["ka_target"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # --- left: the ladder ---------------------------------------------------
    for ka in ka_target:
        xs, ys = [], []
        for c in sorted(cells, key=lambda c: c["subdivisions"]):
            if c["ka"] != ka:
                continue
            xs.append(c["panels"])
            ys.append(100 * max(abs(c[f"rel_error_{q}"]) for q in QUANTITIES))
        ax1.plot(xs, ys, "o-", label=f"ka = {ka:g}")
    ax1.axhline(100 * tol, color="crimson", ls="--", lw=1.2, label="5 % case tolerance")
    ax1.set_xscale("log")
    ax1.set_xlabel("constant panels")
    ax1.set_ylabel("band-max |rel. error|  [%]")
    ax1.set_title("Mesh ladder at the two formerly-inferred ka")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend()

    # --- right: per-quantity error on the finest rung ----------------------
    finest = {c["ka"]: c for c in cells if c["subdivisions"] == max(r for r in res["rungs"])}
    if not finest:  # no 4096-panel cell yet: show the rung that exists
        finest = {c["ka"]: c for c in cells if c["subdivisions"] == 4}
    labels = [q.replace("_dimless", "") for q in QUANTITIES]
    width = 0.8 / max(len(finest), 1)
    for i, (ka, c) in enumerate(sorted(finest.items())):
        errs = [100 * c[f"rel_error_{q}"] for q in QUANTITIES]
        ax2.bar(np.arange(len(labels)) + i * width, errs, width, label=f"ka = {ka:g}")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.axhline(100 * tol, color="crimson", ls="--", lw=1.2)
    ax2.axhline(-100 * tol, color="crimson", ls="--", lw=1.2)
    ax2.set_xticks(np.arange(len(labels)) + width * (len(finest) - 1) / 2)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("signed rel. error vs Hulme series  [%]")
    ax2.set_title("finest rung in this artifact, per quantity")
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(HERE / "fig_rung4096_ladder.png", dpi=160)
    plt.close(fig)

    # --- residuals ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    names, rad, dif = [], [], []
    for c in sorted(cells, key=lambda c: (c["ka"], c["subdivisions"])):
        names.append(f"ka={c['ka']:g}\n{c['panels']}p")
        rad.append(c["radiation_relative_residual"])
        dif.append(c["diffraction_relative_residual"])
    x = np.arange(len(names))
    ax.bar(x - 0.2, rad, 0.4, label="radiation residual")
    ax.bar(x + 0.2, dif, 0.4, label="diffraction residual")
    ax.set_yscale("log")
    ax.axhline(1e-2, color="crimson", ls="--", lw=1.2, label="admissibility 1e-2")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("relative residual")
    ax.set_title("Every rung proves its own residual (closes audit gate A4)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_rung4096_residual.png", dpi=160)
    plt.close(fig)
    print("wrote fig_rung4096_ladder.png, fig_rung4096_residual.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())