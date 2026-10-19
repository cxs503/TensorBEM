"""Reproducible three-dimensional verification and refinement study.

Cases
-----
1. ``interior_linear_harmonic``: u = x on a flat-faceted sphere. The linear
   field is harmonic on the actual polyhedron, so this isolates panel,
   quadrature, and solver errors from geometric modeling error.
2. ``exterior_translating_sphere``: the translating-sphere added-mass
   benchmark with a faceted geometry.
3. ``exterior_constant_dirichlet``: u = 1 exterior problem; the total flux
   converges to the sphere capacitance reference 4*pi*a.

Run:
    python -m tensorbem.benchmarks.three_dimensional_study --levels 0 1 2 3
"""
from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.benchmarks.sphere_added_mass import run as run_sphere_added_mass
from tensorbem.geometry3d import icosphere
from tensorbem.solver3d import evaluate_surface_field, solve_laplace_surface

CASES = (
    "interior_linear_harmonic",
    "exterior_translating_sphere",
    "exterior_constant_dirichlet",
)
INTERIOR_METRICS = ("normal_derivative_relative_l2_error", "field_rmse")
ADDED_MASS_METRICS = ("added_mass_relative_error", "normal_derivative_relative_l2_error")
CONSTANT_METRICS = ("total_flux_relative_error",)


def _validate_levels(levels: list[int]) -> list[int]:
    if not levels or len(levels) < 2:
        raise ValueError("a convergence study requires at least two subdivision levels")
    checked = []
    for level in levels:
        if isinstance(level, (bool, np.bool_)) or not isinstance(level, (int, np.integer)):
            raise ValueError("subdivision levels must be non-negative integers")
        if level < 0:
            raise ValueError("subdivision levels must be non-negative integers")
        checked.append(int(level))
    if len(set(checked)) != len(checked):
        raise ValueError("subdivision levels must be unique")
    return sorted(checked)


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(math.isfinite(float(value)))
    return True


def _validate_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")


def interior_probe_points(radius: float = 1.0) -> np.ndarray:
    """Fixed interior probes, independent of the surface mesh."""
    coordinates = np.linspace(-0.4, 0.4, 5)
    grid_x, grid_y, grid_z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    points = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))
    points = points[np.linalg.norm(points, axis=1) <= 0.5]
    return radius * points


def _flux_metrics(computed: np.ndarray, exact: np.ndarray, areas: np.ndarray) -> dict[str, float]:
    error = computed - exact
    return {
        "normal_derivative_relative_l2_error": float(
            math.sqrt(float(np.sum(areas * error**2)) / float(np.sum(areas * exact**2)))
        ),
        "normal_derivative_rmse": float(np.sqrt(np.mean(error**2))),
        "normal_derivative_max_abs_error": float(np.max(np.abs(error))),
    }


def run_interior_case(subdivisions: int, radius: float = 1.0) -> dict[str, Any]:
    """Linear harmonic field u=x on the actual polyhedral domain."""
    mesh = icosphere(subdivisions, radius=radius)
    boundary = mesh.centroids[:, 0].copy()
    result = solve_laplace_surface(mesh, boundary, domain="interior")
    exact_flux = mesh.normals[:, 0]
    probes = interior_probe_points(radius)
    numerical = evaluate_surface_field(mesh, result, probes, domain="interior")
    exact_probe = probes[:, 0]
    field_error = numerical - exact_probe
    row = {
        "benchmark": "interior_linear_harmonic",
        "subdivisions": int(subdivisions),
        "panels": mesh.panel_count,
        "vertices": mesh.vertex_count,
        "h_max": mesh.h_max,
        "h_mean": mesh.h_mean,
        "surface_area": mesh.surface_area,
        "volume": mesh.volume,
        "residual_l2": result.residual_l2,
        "relative_residual_l2": result.relative_residual_l2,
        "single_layer_condition_number": result.condition_number,
        "double_layer_row_sum_error": result.double_layer_row_sum_error,
        "probe_exact": exact_probe.tolist(),
        "probe_numerical": numerical.tolist(),
    }
    row.update(_flux_metrics(result.normal_derivative, exact_flux, mesh.areas))
    row["field_rmse"] = float(np.sqrt(np.mean(field_error**2)))
    row["field_max_abs_error"] = float(np.max(np.abs(field_error)))
    return row


def run_constant_dirichlet_case(subdivisions: int, radius: float = 1.0) -> dict[str, Any]:
    """Exterior constant Dirichlet problem; total flux reference 4*pi*a."""
    mesh = icosphere(subdivisions, radius=radius)
    boundary = np.ones(mesh.panel_count)
    result = solve_laplace_surface(mesh, boundary, domain="exterior")
    total_flux = float(np.sum(result.normal_derivative * mesh.areas))
    exact_total_flux = 4.0 * math.pi * radius
    uniform_flux = 1.0 / radius
    return {
        "benchmark": "exterior_constant_dirichlet",
        "subdivisions": int(subdivisions),
        "panels": mesh.panel_count,
        "vertices": mesh.vertex_count,
        "h_max": mesh.h_max,
        "h_mean": mesh.h_mean,
        "surface_area": mesh.surface_area,
        "volume": mesh.volume,
        "total_flux": total_flux,
        "exact_total_flux": exact_total_flux,
        "total_flux_relative_error": abs(total_flux - exact_total_flux)
        / exact_total_flux,
        "normal_derivative_mean": float(np.mean(result.normal_derivative)),
        "normal_derivative_rmse_vs_uniform": float(
            np.sqrt(np.mean((result.normal_derivative - uniform_flux) ** 2))
        ),
        "residual_l2": result.residual_l2,
        "relative_residual_l2": result.relative_residual_l2,
        "single_layer_condition_number": result.condition_number,
        "double_layer_row_sum_error": result.double_layer_row_sum_error,
    }


def _add_orders(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> dict[str, bool]:
    checks = {}
    for metric in metrics:
        for index, row in enumerate(rows):
            if index == 0:
                row[metric + "_order"] = None
                continue
            previous = rows[index - 1]
            coarse = float(previous[metric])
            fine = float(row[metric])
            if coarse > 0.0 and fine > 0.0:
                row[metric + "_order"] = math.log(coarse / fine) / math.log(
                    previous["h_max"] / row["h_max"]
                )
            else:
                row[metric + "_order"] = None
        checks[metric + "_decreases"] = all(
            float(later[metric]) < float(earlier[metric])
            for earlier, later in zip(rows, rows[1:])
        )
    return checks


def run_study(
    levels: list[int],
    radius: float = 1.0,
    density: float = 1025.0,
    speed: float = 1.0,
) -> dict[str, Any]:
    """Return JSON-serializable three-dimensional verification evidence."""
    counts = _validate_levels(levels)
    _validate_positive("radius", radius)
    _validate_positive("density", density)
    _validate_positive("speed", speed)

    interior = [run_interior_case(level, radius) for level in counts]
    added_mass = [
        run_sphere_added_mass(level, radius=radius, speed=speed, density=density)
        for level in counts
    ]
    constant = [run_constant_dirichlet_case(level, radius) for level in counts]

    checks: dict[str, bool] = {
        "all_outputs_finite": all(
            _all_finite(row) for rows in (interior, added_mass, constant) for row in rows
        ),
    }
    interior_checks = _add_orders(interior, INTERIOR_METRICS)
    added_mass_checks = _add_orders(added_mass, ADDED_MASS_METRICS)
    constant_checks = _add_orders(constant, CONSTANT_METRICS)
    checks.update(interior_checks)
    checks.update(added_mass_checks)
    checks.update(constant_checks)
    row_sum_errors = [
        float(row["double_layer_row_sum_error"]) for row in interior + constant
    ]
    row_sum_errors += [float(row["double_layer_row_sum_error"]) for row in added_mass]
    checks["double_layer_row_sum_error_below_1e-10"] = max(row_sum_errors) < 1.0e-10

    return {
        "schema_version": 1,
        "study": "tensorbem_three_dimensional_verification",
        "levels": counts,
        "radius": radius,
        "density": density,
        "speed": speed,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "scope": (
            "Constant triangular panels on flat-faceted icospheres, centroid "
            "collocation, c=1/2, analytic double-layer solid angles, Duffy "
            "single-layer quadrature, and dense direct solves. The exterior "
            "sphere added-mass case is a verification benchmark; it does not "
            "include free-surface, wave, viscosity, or general ship-hull physics. "
            "The interior linear case is exact on the actual polyhedron."
        ),
        "interior_probe_points": interior_probe_points(radius).tolist(),
        "cases": {
            "interior_linear_harmonic": {
                "results": interior,
                "checks": interior_checks,
            },
            "exterior_translating_sphere": {
                "results": added_mass,
                "checks": added_mass_checks,
            },
            "exterior_constant_dirichlet": {
                "results": constant,
                "checks": constant_checks,
            },
        },
        "checks": checks,
    }


def plot_study(study: dict[str, Any], output: Path) -> None:
    """Save optional convergence curves (requires matplotlib)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    panels = (
        (
            axes[0],
            study["cases"]["interior_linear_harmonic"]["results"],
            ("normal_derivative_relative_l2_error", "field_rmse"),
            ("Interior linear: flux error", "Interior linear: field RMSE"),
            "Interior linear harmonic",
        ),
        (
            axes[1],
            study["cases"]["exterior_translating_sphere"]["results"],
            ("added_mass_relative_error", "normal_derivative_relative_l2_error"),
            ("Added-mass relative error", "Flux relative L2 error"),
            "Exterior translating sphere",
        ),
        (
            axes[2],
            study["cases"]["exterior_constant_dirichlet"]["results"],
            ("total_flux_relative_error",),
            ("Total-flux relative error",),
            "Exterior constant Dirichlet",
        ),
    )
    for axis, rows, metrics, labels, title in panels:
        for metric, label in zip(metrics, labels):
            axis.loglog(
                [float(row["h_max"]) for row in rows],
                [float(row[metric]) for row in rows],
                "o-",
                label=label,
            )
        axis.set_title(title)
        axis.set_xlabel("Maximum panel length")
        axis.set_ylabel("Error")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize="small")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=1025.0)
    parser.add_argument("--output", type=Path, help="also save the JSON report")
    parser.add_argument("--plot", type=Path, help="save convergence curves (plotting extra)")
    args = parser.parse_args()
    try:
        study = run_study(args.levels, args.radius, args.density, args.speed)
    except ValueError as error:
        parser.error(str(error))
    rendered = json.dumps(study, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.plot:
        plot_study(study, args.plot)


if __name__ == "__main__":
    main()
