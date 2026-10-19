"""Analytical verification on smooth, polygonal, and re-entrant 2-D domains.

Run: python -m tensorbem.benchmarks.analytical_suite --panels 16 32 64 128
References are evaluated on the actual straight-panel boundary, not a circle.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.geometry import PanelMesh, regular_polygon
from tensorbem.solver import evaluate_interior_field, solve_laplace_dirichlet


CASES = ("disk_cubic", "ellipse_quadratic", "square_separable", "l_shape_singular")


def _validate_panel_count(panel_count: int) -> None:
    if (
        isinstance(panel_count, (bool, np.bool_))
        or not isinstance(panel_count, (int, np.integer))
        or panel_count < 8
        or panel_count % 8
    ):
        raise ValueError("panel counts must be integers >= 8 and divisible by 8")


def _polygon(vertices: list[list[float]], panel_count: int) -> PanelMesh:
    corners = np.asarray(vertices, dtype=float)
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    counts = np.rint(panel_count * lengths / np.sum(lengths)).astype(int)
    return PanelMesh(np.concatenate([
        start + np.arange(count)[:, None] / count * edge
        for start, edge, count in zip(corners, edges, counts)
    ]))


def make_mesh(case: str, panel_count: int) -> PanelMesh:
    """Construct CCW meshes; square and L-shaped boundaries are geometrically exact."""
    _validate_panel_count(panel_count)
    if case in ("disk_cubic", "ellipse_quadratic"):
        mesh = regular_polygon(panel_count)
        if case == "ellipse_quadratic":
            return PanelMesh(mesh.vertices * np.array([1.5, 0.75]))
        return mesh
    if case == "square_separable":
        return _polygon([[0, 0], [1, 0], [1, 1], [0, 1]], panel_count)
    if case == "l_shape_singular":
        return _polygon(
            [[0, 0], [1, 0], [1, 1], [-1, 1], [-1, -1], [0, -1]],
            panel_count,
        )
    raise ValueError(f"unknown analytical case: {case}")


def exact_field(case: str, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return potential and Cartesian gradient of each analytical harmonic field.

    The L-shaped field uses 0 <= theta <= 3*pi/2, with a singular gradient at
    the origin. Its evaluation points must exclude that re-entrant corner.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points must be a finite array of shape (M, 2)")
    x, y = points.T
    if case == "disk_cubic":
        return x**3 - 3 * x * y**2, np.column_stack((3 * (x**2 - y**2), -6 * x * y))
    if case == "ellipse_quadratic":
        return x**2 - y**2, np.column_stack((2 * x, -2 * y))
    if case == "square_separable":
        sx, cx = np.sin(np.pi * x), np.cos(np.pi * x)
        sy = np.sinh(np.pi * y) / np.sinh(np.pi)
        cy = np.cosh(np.pi * y) / np.sinh(np.pi)
        return sx * sy, np.pi * np.column_stack((cx * sy, sx * cy))
    if case == "l_shape_singular":
        radius = np.hypot(x, y)
        if np.any(radius == 0):
            raise ValueError("the singular gradient is undefined at the origin")
        theta = np.mod(np.arctan2(y, x), 2 * np.pi)
        alpha = 2.0 / 3.0
        value = radius**alpha * np.sin(alpha * theta)
        scale = alpha * radius ** (alpha - 1)
        gradient = scale[:, None] * np.column_stack((
            np.sin((alpha - 1) * theta), np.cos((alpha - 1) * theta),
        ))
        return value, gradient
    raise ValueError(f"unknown analytical case: {case}")


def interior_points(case: str) -> np.ndarray:
    """Fixed probe sets, independent of mesh size, kept away from boundaries."""
    coordinates = np.linspace(-0.6, 0.6, 9)
    x, y = np.meshgrid(coordinates, coordinates)
    points = np.column_stack((x.ravel(), y.ravel()))
    if case == "disk_cubic":
        return points[np.linalg.norm(points, axis=1) <= 0.6]
    if case == "ellipse_quadratic":
        return points * np.array([1.5, 0.75])
    if case == "square_separable":
        x, y = np.meshgrid(np.linspace(0.15, 0.85, 9), np.linspace(0.15, 0.85, 9))
        return np.column_stack((x.ravel(), y.ravel()))
    if case == "l_shape_singular":
        # Omit both the excluded quadrant and the two re-entrant boundary rays.
        return points[(points[:, 1] > 0) | (points[:, 0] < 0)]
    raise ValueError(f"unknown analytical case: {case}")


def run_case(case: str, panel_count: int) -> dict[str, Any]:
    """Solve one case and report length-weighted boundary and fixed-probe errors."""
    mesh = make_mesh(case, panel_count)
    boundary, gradient = exact_field(case, mesh.centers)
    exact_flux = np.sum(gradient * mesh.normals, axis=1)
    result = solve_laplace_dirichlet(mesh, boundary)
    flux_error = result.normal_derivative - exact_flux
    flux_reference_norm = float(np.sqrt(np.sum(mesh.lengths * exact_flux**2)))
    points = interior_points(case)
    field_exact, _ = exact_field(case, points)
    numerical = evaluate_interior_field(mesh, result, points)
    field_error = numerical - field_exact
    rhs = (0.5 * np.eye(panel_count) + result.double_layer_matrix) @ boundary
    flux_scale = float(np.sum(mesh.lengths * np.abs(exact_flux)))
    return {
        "benchmark": case,
        "panels": mesh.panel_count,
        "h_max": float(np.max(mesh.lengths)),
        "perimeter": mesh.perimeter,
        "normal_derivative_relative_l2_error": float(
            np.sqrt(np.sum(mesh.lengths * flux_error**2)) / flux_reference_norm
        ),
        "normal_derivative_max_abs_error": float(np.max(np.abs(flux_error))),
        "field_rmse": float(np.sqrt(np.mean(field_error**2))),
        "field_max_abs_error": float(np.max(np.abs(field_error))),
        "net_flux": float(np.dot(mesh.lengths, result.normal_derivative)),
        "normalized_flux_imbalance": float(
            abs(np.dot(mesh.lengths, result.normal_derivative)) / flux_scale
        ),
        "reference_midpoint_net_flux": float(np.dot(mesh.lengths, exact_flux)),
        "residual_l2": result.residual_l2,
        "relative_residual_l2": float(result.residual_l2 / np.linalg.norm(rhs)),
        "single_layer_condition_number": result.condition_number,
        "probe_points": points.tolist(),
        "probe_exact": field_exact.tolist(),
        "probe_numerical": numerical.tolist(),
    }


def run_study(panel_counts: list[int]) -> dict[str, Any]:
    """Return reproducible data and observed orders, without hiding failed trends."""
    if len(panel_counts) < 2:
        raise ValueError("a convergence study requires at least two panel counts")
    for n in panel_counts:
        _validate_panel_count(n)
    if len(set(panel_counts)) != len(panel_counts):
        raise ValueError("panel counts must be unique")
    counts = sorted(int(n) for n in panel_counts)
    cases = {}
    for case in CASES:
        rows = [run_case(case, n) for n in counts]
        for index, row in enumerate(rows):
            for metric in ("normal_derivative_relative_l2_error", "field_rmse"):
                order = None
                if index:
                    previous = rows[index - 1]
                    if previous[metric] > 0 and row[metric] > 0:
                        order = math.log(previous[metric] / row[metric]) / math.log(
                            previous["h_max"] / row["h_max"]
                        )
                row[metric + "_order"] = order
        cases[case] = {
            "results": rows,
            "checks": {
                metric + "_decreases": all(
                    later[metric] < earlier[metric]
                    for earlier, later in zip(rows, rows[1:])
                )
                for metric in ("normal_derivative_relative_l2_error", "field_rmse")
            },
        }
    return {
        "schema_version": 1,
        "study": "tensorbem_analytical_verification",
        "panel_counts": counts,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "scope": "Bounded 2-D Laplace Dirichlet problems, constant straight panels, "
        "analytical panel integration. Not exterior-flow or experimental validation.",
        "cases": cases,
    }


def plot_convergence(study: dict[str, Any], output: Path) -> None:
    """Save publication-oriented error curves using the optional plotting extra."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for case, data in study["cases"].items():
        rows = data["results"]
        for axis, metric in zip(
            axes, ("normal_derivative_relative_l2_error", "field_rmse")
        ):
            axis.loglog(
                [row["h_max"] for row in rows],
                [row[metric] for row in rows],
                "o-",
                label=case,
            )
    for axis, title in zip(axes, ("Boundary flux relative L2 error", "Interior probe RMSE")):
        axis.set_xlabel("Maximum panel length")
        axis.set_ylabel(title)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize="small")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--output", type=Path, help="also save the JSON report")
    parser.add_argument("--plot", type=Path, help="save convergence curves (plotting extra)")
    args = parser.parse_args()
    try:
        study = run_study(args.panels)
    except ValueError as error:
        parser.error(str(error))
    rendered = json.dumps(study, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.plot:
        plot_convergence(study, args.plot)


if __name__ == "__main__":
    main()
