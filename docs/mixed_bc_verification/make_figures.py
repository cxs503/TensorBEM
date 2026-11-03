"""Figures for the mixed Dirichlet/Neumann verification study.

Reads ``docs/mixed_bc_verification/results.json`` (written by
``python -m tensorbem.benchmarks.mixed_bc_verification --outdir ...``) and writes
three PNGs next to it:

* ``fig_convergence.png``   error vs panels per loop/side with an O(h^2) slope,
* ``fig_annulus_field.png`` reconstructed potential and its error in the annulus
  (mixed outer-Dirichlet / inner-Neumann data), i.e. the multiply connected field,
* ``fig_reference_length.png`` the logarithmic-capacity degeneracy at L = outer radius.

Run: python docs/mixed_bc_verification/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorbem.benchmarks.mixed_bc_verification import (
    FIELD_POINTS,
    INNER_RADIUS,
    OUTER_RADIUS,
    REFERENCE_LENGTH,
    annulus_panels,
    cached_matrices,
    radial_flux,
    radial_potential,
)
from tensorbem.mixed_bc import evaluate_field, solve_mixed

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.json"

CASE_LABELS = {
    "annulus_dirichlet": "annulus, all Dirichlet (2 loops)",
    "annulus_mixed_outer_dirichlet": "annulus, outer D / inner N",
    "annulus_mixed_hole_dirichlet": "annulus, hole D / outer N",
    "annulus_pure_neumann": "annulus, pure Neumann",
    "square_mixed": "square, mixed (90 deg corners)",
    "circle_mixed": "circle, mixed (smooth junctions)",
}


def plot_convergence(results: dict) -> None:
    resolutions = results["setup"]["panels"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(results["cases"])))
    for color, (name, case) in zip(colors, results["cases"].items()):
        errors = np.array(case["headline"]["errors"])
        panel_counts = np.array([case[str(n)]["panel_count"] for n in resolutions], dtype=float)
        ax.loglog(panel_counts, errors, "o-", color=color, label=CASE_LABELS[name])
    reference = np.array([case[str(resolutions[0])]["panel_count"] for case in results["cases"].values()], dtype=float)
    anchor = results["cases"]["annulus_dirichlet"]["headline"]["errors"][0]
    panel_counts = np.array(
        [results["cases"]["annulus_dirichlet"][str(n)]["panel_count"] for n in resolutions],
        dtype=float,
    )
    ax.loglog(
        panel_counts,
        anchor * (panel_counts[0] / panel_counts) ** 2,
        "k--",
        lw=1.0,
        label=r"reference slope $h^{2}$",
    )
    ax.set_xlabel("panels on the boundary (total)")
    ax.set_ylabel("relative error of the headline trace")
    ax.set_title("Mixed / multiply connected 2-D BEM convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_convergence.png", dpi=150)
    plt.close(fig)


def plot_annulus_field(per_loop: int = 48) -> None:
    panels = annulus_panels(per_loop)
    exact_u = radial_potential(panels.centers)
    exact_q = radial_flux(panels)
    inner = np.where(panels.loop == 1)[0]
    outer = np.where(panels.loop == 0)[0]
    result = solve_mixed(
        panels,
        dirichlet={int(i): float(exact_u[i]) for i in outer},
        neumann={int(i): float(exact_q[i]) for i in inner},
        reference_length=REFERENCE_LENGTH,
        matrices=cached_matrices(panels, REFERENCE_LENGTH),
    )
    grid = np.linspace(-OUTER_RADIUS, OUTER_RADIUS, 72)
    x, y = np.meshgrid(grid, grid)
    points = np.column_stack((x.ravel(), y.ravel()))
    radius = np.hypot(points[:, 0], points[:, 1])
    inside = (radius < OUTER_RADIUS * 0.985) & (radius > INNER_RADIUS * 1.015)
    values = np.full(len(points), np.nan)
    values[inside] = evaluate_field(
        panels, result.boundary_values, result.normal_derivative,
        points[inside], reference_length=REFERENCE_LENGTH,
    )
    exact = radial_potential(points)
    error = np.abs(values - exact)
    z = values.reshape(x.shape)
    e = error.reshape(x.shape)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    mesh = axes[0].pcolormesh(x, y, z, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=axes[0], label="u (reconstructed)")
    axes[0].plot(panels.centers[panels.loop == 0, 0], panels.centers[panels.loop == 0, 1],
                 "w.", ms=3, label="Dirichlet panels")
    axes[0].plot(panels.centers[panels.loop == 1, 0], panels.centers[panels.loop == 1, 1],
                 "r.", ms=3, label="Neumann panels")
    axes[0].set_title("annulus field, outer Dirichlet / inner Neumann")
    axes[0].set_aspect("equal")
    axes[0].legend(loc="upper right", fontsize=7)
    with np.errstate(invalid="ignore"):
        floor = np.nanmax(e) * 1.0e-3
        mesh2 = axes[1].pcolormesh(x, y, np.log10(e + floor), shading="auto", cmap="magma")
    fig.colorbar(mesh2, ax=axes[1], label=r"$\log_{10}|u_h-u|$")
    axes[1].set_title("reconstruction error (analytic reference)")
    axes[1].set_aspect("equal")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(HERE / "fig_annulus_field.png", dpi=150)
    plt.close(fig)
    probes = evaluate_field(
        panels, result.boundary_values, result.normal_derivative,
        FIELD_POINTS, reference_length=REFERENCE_LENGTH,
    )
    print("probe points (x, y, u_num, u_exact):")
    for point, got, want in zip(FIELD_POINTS, probes, radial_potential(FIELD_POINTS)):
        print(f"  ({point[0]:+.2f}, {point[1]:+.2f})  {got:.6f}  {want:.6f}")


def plot_reference_length(results: dict) -> None:
    rows = results["reference_length_sweep"]
    lengths = [row["reference_length"] for row in rows]
    errors = [row["flux_rel_l2_all"] for row in rows]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    bars = ax.bar([str(value) for value in lengths], errors, color="#3b6ea5")
    for bar, row in zip(bars, rows):
        if abs(row["reference_length"] - OUTER_RADIUS) < 1.0e-12:
            bar.set_color("#c0392b")
    ax.set_yscale("log")
    ax.set_xlabel("reference length L (m)")
    ax.set_ylabel("flux relative L2 error")
    ax.set_title("annulus Dirichlet, 64 panels/loop:\nL = outer radius is degenerate")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "fig_reference_length.png", dpi=150)
    plt.close(fig)


def main() -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    plot_convergence(results)
    plot_annulus_field()
    plot_reference_length(results)
    print(f"wrote figures to {HERE}")


if __name__ == "__main__":
    main()