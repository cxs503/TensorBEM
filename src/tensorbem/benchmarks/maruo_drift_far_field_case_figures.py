"""Figures for CASE-BEM-MARUO-DRIFT (matplotlib, study JSON in, PNG out).

Kept separate from :mod:`tensorbem.benchmarks.maruo_drift_far_field_case` so that
the study can run on a compute node without matplotlib installed; the JSON it
writes is the only input here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = ["plot_study"]


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_study(study: dict[str, Any], outdir: Path | str) -> list[Path]:
    """Write the three case figures and return their paths."""
    plt = _mpl()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # ---- figure 1: BEM vs closed form, panel convergence -------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ka_values = sorted({cell["ka"] for cell in study["cells"]})
    colours = plt.cm.viridis(np.linspace(0.05, 0.85, len(ka_values)))
    for colour, ka in zip(colours, ka_values):
        cells = sorted(
            [c for c in study["cells"] if c["ka"] == ka],
            key=lambda c: c["panel_count"],
        )
        panels = [c["panel_count"] for c in cells]
        err = [100.0 * c["rel_error"] for c in cells]
        axes[0].plot(panels, err, "o-", color=colour, label=f"ka = {ka:g}")
        axes[1].plot(panels, [c["bem_force_coefficient"] for c in cells], "o-",
                     color=colour, label=f"ka = {ka:g}")
    axes[0].axhline(5.0, color="crimson", ls="--", lw=1.0, label="5 % case tolerance")
    axes[0].axhline(3.0, color="darkorange", ls=":", lw=1.0, label="3 % report line")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("panels (triangles)")
    axes[0].set_ylabel("|F_BEM - F_ref| / F_ref  [%]")
    axes[0].set_title("Maruo far-field drift: BEM vs closed form")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("panels (triangles)")
    axes[1].set_ylabel(r"$F_x/(\rho g A^2 a)$")
    axes[1].set_title("dimensionless drift force (converging to the closed form)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = out / "fig_bem_vs_closed_form.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)

    # ---- figure 2: coefficient vs ka, closed form curve + BEM + asymptote --
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ka_grid = np.logspace(-2.0, math.log10(3.0), 200)
    coeff = [study_ref_coefficient(study, float(x)) for x in ka_grid]
    ax.loglog(ka_grid, coeff, "k-", label="closed form (modal Maruo integral)")
    asymptote = [math.pi**2 * 5.0 / 16.0 * x**3 for x in ka_grid]
    ax.loglog(ka_grid, asymptote, "b--", lw=1.0,
              label=r"long-wave law $(5\pi^2/16)(ka)^3$")
    production = study["parameters"]["production_resolution"]
    prod_cells = [c for c in study["cells"] if c.get("resolution") == production]
    ax.loglog([c["ka"] for c in prod_cells],
              [c["bem_force_coefficient"] for c in prod_cells], "ro",
              label=f"BEM, {production} mesh")
    ax.set_xlabel("ka")
    ax.set_ylabel(r"$F_x/(\rho g A^2 a)$")
    ax.set_title("mean drift force coefficient, fixed vertical cylinder")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = out / "fig_coefficient_vs_ka.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)

    # ---- figure 3: convergence fit + quadrature sensitivity ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for colour, ka in zip(colours, ka_values):
        cells = sorted([c for c in study["cells"] if c["ka"] == ka],
                       key=lambda c: c["panel_count"])
        if len(cells) >= 2:
            h = [c["panel_count"] for c in cells]
            e = [c["rel_error"] for c in cells]
            axes[0].loglog(h, e, "o-", color=colour, label=f"ka = {ka:g}")
            slope = (math.log(e[-1]) - math.log(e[0])) / (math.log(h[-1]) - math.log(h[0]))
            axes[0].annotate(f"slope {slope:+.2f}", (h[-1], e[-1]),
                             textcoords="offset points", xytext=(-52, 6),
                             fontsize=7, color=colour)
    axes[0].axhline(0.05, color="crimson", ls="--", lw=1.0)
    axes[0].set_xlabel("panels (triangles)")
    axes[0].set_ylabel("relative error")
    axes[0].set_title("observed panel convergence of the far-field drift force")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    for colour, ka in zip(colours, ka_values):
        rows = study["quadrature_sensitivity"][repr_ka(study, ka)]["rows"]
        n_theta = [r["n_theta"] for r in rows]
        values = [r["integral_weighted"] for r in rows]
        axes[1].semilogx(n_theta, values, "o-", color=colour, label=f"ka = {ka:g}")
    axes[1].set_xlabel("Maruo kernel quadrature points n_theta")
    axes[1].set_ylabel(r"$\int |K|^2 (1-\cos\theta)\,d\theta$")
    axes[1].set_title("azimuthal quadrature sensitivity (exact field)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = out / "fig_convergence_and_quadrature.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)
    return paths


def repr_ka(study: dict[str, Any], ka: float) -> str:
    """Key of ``quadrature_sensitivity`` for ``ka`` (JSON round-trip friendly)."""
    keys = list(study["quadrature_sensitivity"])
    for key in keys:
        if abs(float(key) - float(ka)) <= 1e-12:
            return key
    raise KeyError(ka)


def study_ref_coefficient(study: dict[str, Any], ka: float) -> float:
    """Closed-form coefficient at ``ka`` taken from the study reference rows."""
    rows = study["reference"]
    for row in rows:
        if abs(row["ka"] - ka) <= 1e-12:
            return float(row["force_coefficient"])
    # outside the tabulated grid: rebuild the closed form from the modal formula
    from tensorbem.benchmarks.maruo_drift_far_field_case import closed_form_case

    pars = study["parameters"]
    return float(closed_form_case(ka, density=pars["density"],
                                  gravity=pars["gravity"])["force_coefficient"])


if __name__ == "__main__":  # pragma: no cover
    import sys

    src = Path(sys.argv[1])
    data = json.loads(src.read_text())
    for p in plot_study(data, sys.argv[2] if len(sys.argv) > 2 else src.parent):
        print(p)