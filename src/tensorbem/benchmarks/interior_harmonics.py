"""Interior solid harmonics of degree 1, 2 and 3 on a faceted sphere.

Every polynomial is represented exactly on a polyhedral domain, so a harmonic
polynomial of degree ``d`` gives an exact solution of the *discrete* interior
Dirichlet problem's continuous target: the only errors are the constant-panel
trace, the panel quadrature and the flat faceting of the sphere. This is the
3-D analogue of the 2-D analytical suite's cubic and quadratic cases, and it
extends the existing 3-D verification, which only used the linear field
``u = x``.

The harmonics are real solid harmonics in Cartesian coordinates, registered in
:data:`tensorbem.analytic.SOLID_HARMONICS`. Degree 1 is ``x, y, z``; degree 2
is ``x**2-y**2, 2xy, 2z**2-x**2-y**2``; degree 3 includes the two cubic
harmonics in ``x, y`` and two mixed ones. Higher degree means stronger angular
variation, which is exactly what a constant-panel discretisation struggles
with, so the observed orders are the informative output.

Run: python -m tensorbem.benchmarks.interior_harmonics --subdivisions 1 2 3
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.analytic import (
    solid_harmonic,
    solid_harmonic_degree,
    solid_harmonic_names,
)
from tensorbem.geometry3d import icosphere
from tensorbem.solver3d import evaluate_surface_field, solve_laplace_surface


def interior_probe_points(radius: float = 1.0) -> np.ndarray:
    """Fixed interior probes, independent of the surface mesh."""
    coordinates = np.linspace(-0.4, 0.4, 5)
    grid_x, grid_y, grid_z = np.meshgrid(
        coordinates, coordinates, coordinates, indexing="ij"
    )
    points = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))
    points = points[np.linalg.norm(points, axis=1) <= 0.5]
    return radius * points


def run_case(
    name: str,
    subdivisions: int,
    radius: float = 1.0,
    probes: np.ndarray | None = None,
) -> dict[str, Any]:
    """Solve one harmonic case and report flux and interior-field errors."""
    mesh = icosphere(subdivisions, radius=radius)
    probes = interior_probe_points(radius) if probes is None else probes

    boundary, boundary_gradient = solid_harmonic(name, mesh.centroids)
    exact_flux = np.sum(boundary_gradient * mesh.normals, axis=1)
    result = solve_laplace_surface(mesh, boundary, domain="interior")

    flux_error = result.normal_derivative - exact_flux
    reference = float(np.sqrt(np.sum(mesh.areas * exact_flux**2)))
    numerical, _ = solid_harmonic(name, probes)
    field = evaluate_surface_field(mesh, result, probes, domain="interior")
    field_error = field - numerical

    degree = solid_harmonic_degree(name)
    return {
        "benchmark": "interior_solid_harmonic",
        "harmonic": name,
        "degree": int(degree),
        "subdivisions": int(subdivisions),
        "panels": mesh.panel_count,
        "vertices": mesh.vertex_count,
        "h_max": mesh.h_max,
        "h_mean": mesh.h_mean,
        "normal_derivative_relative_l2_error": float(
            np.sqrt(np.sum(mesh.areas * flux_error**2)) / reference
        ),
        "normal_derivative_rmse": float(np.sqrt(np.mean(flux_error**2))),
        "normal_derivative_max_abs_error": float(np.max(np.abs(flux_error))),
        "field_rmse": float(np.sqrt(np.mean(field_error**2))),
        "field_max_abs_error": float(np.max(np.abs(field_error))),
        "residual_l2": result.residual_l2,
        "relative_residual_l2": result.relative_residual_l2,
        "single_layer_condition_number": result.condition_number,
        "double_layer_row_sum_error": result.double_layer_row_sum_error,
    }


def run(
    subdivisions: Sequence[int],
    degrees: Sequence[int] = (1, 2, 3),
    radius: float = 1.0,
) -> dict[str, Any]:
    """Run every requested harmonic at every requested mesh level."""
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be a positive finite number")
    levels = []
    for level in subdivisions:
        if isinstance(level, (bool, np.bool_)) or not isinstance(level, (int, np.integer)):
            raise ValueError("subdivision levels must be non-negative integers")
        if level < 0:
            raise ValueError("subdivision levels must be non-negative integers")
        levels.append(int(level))
    if not levels:
        raise ValueError("at least one subdivision level is required")
    if len(set(levels)) != len(levels):
        raise ValueError("subdivision levels must be unique")
    levels = sorted(levels)

    selected = [
        name for degree in degrees for name in solid_harmonic_names(int(degree))
    ]
    probes = interior_probe_points(radius)
    cases: dict[str, Any] = {}
    for name in selected:
        rows = [run_case(name, level, radius, probes) for level in levels]
        for index, row in enumerate(rows):
            for metric in ("normal_derivative_relative_l2_error", "field_rmse"):
                order = None
                if index:
                    previous = rows[index - 1]
                    if previous[metric] > 0.0 and row[metric] > 0.0:
                        order = math.log(previous[metric] / row[metric]) / math.log(
                            previous["h_max"] / row["h_max"]
                        )
                row[metric + "_order"] = order
        cases[name] = {
            "results": rows,
            "checks": {
                "normal_derivative_relative_l2_error_decreases": all(
                    later["normal_derivative_relative_l2_error"]
                    < earlier["normal_derivative_relative_l2_error"]
                    for earlier, later in zip(rows, rows[1:])
                ),
                "field_rmse_decreases": all(
                    later["field_rmse"] < earlier["field_rmse"]
                    for earlier, later in zip(rows, rows[1:])
                ),
            },
        }

    return {
        "schema_version": 1,
        "study": "tensorbem_interior_solid_harmonics",
        "levels": levels,
        "degrees": [int(value) for value in degrees],
        "radius": radius,
        "probe_count": int(len(probes)),
        "cases": cases,
        "scope": (
            "Bounded 3-D Laplace Dirichlet problems on a flat-faceted icosphere, "
            "constant triangular panels, centroid collocation. The harmonics are "
            "exact on the polyhedron, so the errors are discretisation errors. No "
            "free-surface, wave or ship-hull physics is included."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivisions", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--degrees", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--output", type=Path, help="also write machine-readable JSON")
    args = parser.parse_args()
    try:
        study = run(args.subdivisions, args.degrees, args.radius)
    except ValueError as error:
        parser.error(str(error))
    rendered = json.dumps(study, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
