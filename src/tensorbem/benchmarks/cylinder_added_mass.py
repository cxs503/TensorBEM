"""Potential-flow benchmark for a translating circular cylinder.

The exact exterior potential is phi = -U*a**2*cos(theta)/r. On the body,
phi = -U*a*cos(theta), and the normal derivative with the fluid-domain normal
(pointing into the body) is q = -U*cos(theta). The corresponding added mass
per unit span is rho*pi*a**2.

The current solver is a 2-D Laplace prototype on a bounded polygon. For this
first benchmark, its interior Dirichlet-to-Neumann trace is compared with the
exact circular-cylinder trace; this verifies boundary values and integrated
added mass, not a general exterior-domain formulation.
Run: python -m tensorbem.benchmarks.cylinder_added_mass --panels 64
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from tensorbem import regular_polygon, solve_laplace_dirichlet


def run(
    panel_count: int,
    radius: float = 1.0,
    speed: float = 1.0,
    density: float = 1025.0,
) -> dict[str, float | int | str]:
    """Run the circular-cylinder translating-body benchmark."""
    for name, value in (("radius", radius), ("speed", speed), ("density", density)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")

    mesh = regular_polygon(panel_count, radius=radius)
    theta = np.arctan2(mesh.centers[:, 1], mesh.centers[:, 0])
    boundary_potential = -speed * radius * np.cos(theta)
    result = solve_laplace_dirichlet(mesh, boundary_potential)

    normal_angle = np.arctan2(mesh.normals[:, 1], mesh.normals[:, 0])
    exact_normal_derivative = -speed * np.cos(normal_angle)
    flux_error = result.normal_derivative - exact_normal_derivative
    added_mass = float(
        density
        / speed**2
        * np.sum(boundary_potential * result.normal_derivative * mesh.lengths)
    )
    exact_added_mass = density * np.pi * radius**2

    return {
        "benchmark": "two_dimensional_translating_circular_cylinder",
        "panels": panel_count,
        "radius": radius,
        "speed": speed,
        "fluid_density": density,
        "added_mass_per_unit_span": added_mass,
        "exact_added_mass_per_unit_span": exact_added_mass,
        "added_mass_relative_error": abs(added_mass - exact_added_mass)
        / exact_added_mass,
        "added_mass_method": "boundary_energy_integral_using_computed_flux",
        "normal_derivative_rmse": float(np.sqrt(np.mean(flux_error**2))),
        "residual_l2": result.residual_l2,
        "single_layer_condition_number": result.condition_number,
        "note": (
            "The polygon approximates the circular body. Added mass uses the "
            "computed boundary flux in the boundary energy integral. For this "
            "first angular mode, the bounded-domain flux trace matches the "
            "exterior fluid-normal trace; this does not validate a general "
            "exterior-domain solver."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=int, default=64)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=1025.0)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.panels, args.radius, args.speed, args.density),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
