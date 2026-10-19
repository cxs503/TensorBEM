"""Manufactured solution on a polygonal approximation of the unit disk.

Exact harmonic field: u(r,theta)=r*cos(theta)=x.
Exact outward normal derivative on the unit circle: q=cos(theta).
Because the numerical boundary is polygonal, this is a geometry/discretization
benchmark rather than a claim of exact circle geometry.
Run: python -m tensorbem.benchmarks.unit_disk --panels 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tensorbem import regular_polygon, solve_laplace_dirichlet
from tensorbem.solver import evaluate_interior_field


def run(panel_count: int) -> dict[str, float | int | str]:
    mesh = regular_polygon(panel_count)
    # For u=x, prescribe its value at panel midpoints.
    u = mesh.centers[:, 0].copy()
    result = solve_laplace_dirichlet(mesh, u)
    # Circle normal is approximately the radial direction at panel midpoints.
    theta = np.arctan2(mesh.centers[:, 1], mesh.centers[:, 0])
    q_exact_circle = np.cos(theta)
    q_error = result.normal_derivative - q_exact_circle
    return {
        "benchmark": "unit_disk_linear_harmonic_field",
        "panels": panel_count,
        "perimeter": mesh.perimeter,
        "residual_l2": result.residual_l2,
        "single_layer_condition_number": result.condition_number,
        "normal_derivative_rmse_vs_circle_reference": float(
            np.sqrt(np.mean(q_error**2))
        ),
        "normal_derivative_max_abs_error_vs_circle_reference": float(
            np.max(np.abs(q_error))
        ),
        "note": "Circle reference is approximate for a polygonal boundary; use convergence trends, not a single value, as the verification criterion.",
    }


def plot_field_comparison(
    panel_count: int, output_path: str | Path, grid_size: int = 121
) -> None:
    """Save BEM, analytical, and absolute-error potential cloudplots."""
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3")

    import matplotlib.pyplot as plt
    from matplotlib.path import Path as PolygonPath

    mesh = regular_polygon(panel_count)
    result = solve_laplace_dirichlet(mesh, mesh.centers[:, 0])
    coordinates = np.linspace(-1.05, 1.05, grid_size)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    inside = PolygonPath(mesh.vertices).contains_points(grid_points)

    numerical = np.full(len(grid_points), np.nan)
    numerical[inside] = evaluate_interior_field(mesh, result, grid_points[inside])
    exact = grid_x.ravel()
    error = np.abs(numerical - exact)
    numerical = numerical.reshape(grid_x.shape)
    exact = exact.reshape(grid_x.shape)
    error = error.reshape(grid_x.shape)

    field_levels = np.linspace(-1.0, 1.0, 31)
    error_max = max(float(np.nanmax(error)), np.finfo(float).eps)
    error_levels = np.linspace(0.0, error_max, 21)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, values, title, levels in (
        (axes[0], numerical, "BEM potential", field_levels),
        (axes[1], exact, "Analytical potential", field_levels),
        (axes[2], error, "Absolute error", error_levels),
    ):
        image = axis.contourf(
            grid_x, grid_y, values, levels=levels, cmap="viridis"
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")
        fig.colorbar(image, ax=axis)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=int, default=64)
    parser.add_argument(
        "--plot",
        type=Path,
        help="save BEM, analytical, and absolute-error potential cloudplots",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.panels), indent=2, sort_keys=True))
    if args.plot:
        plot_field_comparison(args.panels, args.plot)


if __name__ == "__main__":
    main()
