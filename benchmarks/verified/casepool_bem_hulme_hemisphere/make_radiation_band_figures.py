#!/usr/bin/env python3
"""Figures for the independent (capytaine) radiation-band axis.

Reads the frozen artifact ``radiation_capytaine_band_xval.json`` (produced by
``radiation_capytaine_band_xval.py``) and the reference series, and draws

* ``fig_radiation_band_coefficients.png`` -- A33 / B33 against ka: the Hulme
  series (dense line), capytaine at 1024 panels (squares) and the TensorBEM
  1024-panel cells (triangles).  The accepted band ``ka <= 2`` is shaded, so the
  figure shows at a glance where the two panel codes depart from the series;
* ``fig_radiation_band_error.png`` -- |relative error| of capytaine (solid) and
  TensorBEM (dashed) for all four magnitudes against ka, log scale, with the 5 %
  gate drawn as a line.

Numbers come from the artifact only; nothing is typed in by hand.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "radiation_capytaine_band_xval.json"
FIG_COEFF = HERE / "fig_radiation_band_coefficients.png"
FIG_ERROR = HERE / "fig_radiation_band_error.png"

MAGNITUDES = ("A11", "B11", "A33", "B33")
LABELS = {"A11": "A11/(rho nabla)", "B11": "B11/(rho omega nabla)",
          "A33": "A33/(rho nabla)", "B33": "B33/(rho omega nabla)"}


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    study = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    fine = str(max(study["constants"]["sub_levels"]))
    band_edge = max(r["ka"] for r in study["rows"] if r["in_band"])

    from tensorbem.benchmarks.hulme_hemisphere_case import reference as hulme_ref

    ka_dense = np.linspace(0.3, 4.2, 79)
    dense = {m: [] for m in MAGNITUDES}
    for ka in ka_dense:
        vals = hulme_ref(float(ka))
        for m in MAGNITUDES:
            dense[m].append(vals[f"{m}_dimless"])

    ka_rows = np.array([r["ka"] for r in study["rows"]])
    cpt = {m: np.array([r["capytaine"][fine][m] for r in study["rows"]]) for m in MAGNITUDES}
    tbm_rows = [r["attribution"]["tensorbem_cell"] for r in study["rows"]]

    # ---- figure 1: coefficients -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    for ax, key in zip(axes, ("A33", "B33")):
        ax.axvspan(0.3, band_edge, color="0.90", zorder=0,
                   label=f"accepted band ka <= {band_edge:g}")
        ax.plot(ka_dense, dense[key], "-", color="black", lw=1.6,
                label="Hulme (1982) series, 144 terms")
        ax.plot(ka_rows, cpt[key], "s", ms=7, mfc="none", mew=1.6, color="tab:blue",
                label="capytaine, 1024 panels")
        tb_x = [r["ka"] for r, c in zip(study["rows"], tbm_rows) if c]
        tb_y = [c["values"][key] for c in tbm_rows if c]
        ax.plot(tb_x, tb_y, "^", ms=7, mfc="none", mew=1.6, color="tab:red",
                label="TensorBEM Kelvin+CHIEF, 1024 panels")
        ax.set_xlabel("ka = k a")
        ax.set_ylabel(LABELS[key])
        ax.grid(alpha=0.3)
        ax.set_title(key.replace("33", "") + "33 vs ka")
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Floating hemisphere, radiation coefficients: series vs two independent panel codes")
    fig.tight_layout()
    fig.savefig(FIG_COEFF, dpi=160)
    plt.close(fig)

    # ---- figure 2: relative error -----------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axvspan(0.3, band_edge, color="0.90", zorder=0)
    ax.axhline(5.0, color="black", ls="--", lw=1.0, label="5 % gate")
    colors = {"A11": "tab:blue", "B11": "tab:green", "A33": "tab:red", "B33": "tab:purple"}
    for m in MAGNITUDES:
        ref = np.array([r["reference"][m] for r in study["rows"]])
        ax.semilogy(ka_rows, np.abs(100.0 * (cpt[m] / ref - 1.0)), "-o", ms=4,
                    color=colors[m], label=f"capytaine {m}")
        xs, ys = [], []
        for r in study["rows"]:
            cell = r["attribution"]["tensorbem_cell"]
            if cell:
                xs.append(r["ka"])
                ys.append(abs(100.0 * (cell["values"][m] / r["reference"][m] - 1.0)))
        if xs:
            ax.semilogy(xs, ys, "--^", ms=4, mfc="none", color=colors[m],
                        alpha=0.85, label=f"TensorBEM {m}")
    ax.set_xlabel("ka = k a")
    ax.set_ylabel("|relative error| vs Hulme series  [%]")
    ax.set_title("Capytaine stays inside the 5 % gate where TensorBEM leaves it")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_ERROR, dpi=160)
    plt.close(fig)

    print(f"wrote {FIG_COEFF.name}, {FIG_ERROR.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())