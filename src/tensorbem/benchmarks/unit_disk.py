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
import numpy as np

from tensorbem import regular_polygon, solve_laplace_dirichlet


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(run(args.panels), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
