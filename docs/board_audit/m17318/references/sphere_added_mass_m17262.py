"""Potential-flow benchmark: added mass of a translating sphere.

For a sphere of radius ``a`` translating with speed ``U`` in an unbounded,
inviscid, incompressible fluid, the exact exterior potential is

    phi(r, theta) = -(1/2) * U * a**3 * cos(theta) / r**2,   r >= a,

with boundary trace ``phi = -(1/2) * U * a * cos(theta)`` and fluid-domain
normal derivative ``q = -U * cos(theta)``. The added mass is
``0.5 * rho * (4/3) * pi * a**3``.

The numerical surface is a flat-faceted icosphere. Boundary values are
prescribed at panel centroids using the angular coordinate of each centroid,
so no artificial radial-depth bias is introduced. The solution uses the
exterior Dirichlet formulation ``S q = (1/2 I - H) u`` and the added mass is
the boundary energy integral with the computed flux. This is a verification
case, not a free-surface or ship-hydrodynamics validation.

Run: python -m tensorbem.benchmarks.sphere_added_mass --subdivisions 2
Ladder: python -m tensorbem.benchmarks.sphere_added_mass --ladder 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.conditioning import estimate_operator_conditions
from tensorbem.geometry3d import icosphere
from tensorbem.solver3d import solve_laplace_surface


def run(
    subdivisions: int = 2,
    radius: float = 1.0,
    speed: float = 1.0,
    density: float = 1025.0,
) -> dict[str, float | int | str]:
    """Run the translating-sphere added-mass benchmark."""
    for name, value in (("radius", radius), ("speed", speed), ("density", density)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")

    mesh = icosphere(subdivisions, radius=radius)
    centroids = mesh.centroids
    radial_distance = np.linalg.norm(centroids, axis=1)
    cos_theta = centroids[:, 0] / radial_distance
    boundary_potential = -0.5 * speed * radius * cos_theta
    result = solve_laplace_surface(mesh, boundary_potential, domain="exterior")
    exact_normal_derivative = -speed * cos_theta
    flux_error = result.normal_derivative - exact_normal_derivative
    weighted_flux_error = math.sqrt(
        float(np.sum(mesh.areas * flux_error**2))
        / float(np.sum(mesh.areas * exact_normal_derivative**2))
    )
    added_mass = float(
        density
        / speed**2
        * np.sum(boundary_potential * result.normal_derivative * mesh.areas)
    )
    exact_added_mass = 0.5 * density * 4.0 / 3.0 * math.pi * radius**3
    quality = mesh.quality

    # Compute conditioning diagnostics
    conditioning = estimate_operator_conditions(
        mesh, boundary_potential, domain="exterior"
    )

    return {
        "benchmark": "three_dimensional_translating_sphere",
        "subdivisions": int(subdivisions),
        "panels": mesh.panel_count,
        "vertices": mesh.vertex_count,
        "radius": radius,
        "speed": speed,
        "fluid_density": density,
        "added_mass": added_mass,
        "exact_added_mass": exact_added_mass,
        "added_mass_relative_error": abs(added_mass - exact_added_mass)
        / exact_added_mass,
        "added_mass_method": "boundary_energy_integral_using_computed_flux",
        "normal_derivative_rmse": float(np.sqrt(np.mean(flux_error**2))),
        "normal_derivative_max_abs_error": float(np.max(np.abs(flux_error))),
        "normal_derivative_relative_l2_error": weighted_flux_error,
        "h_max": mesh.h_max,
        "h_mean": mesh.h_mean,
        "surface_area": mesh.surface_area,
        "exact_surface_area": 4.0 * math.pi * radius**2,
        "surface_area_ratio": mesh.surface_area / (4.0 * math.pi * radius**2),
        "volume": mesh.volume,
        "exact_volume": 4.0 / 3.0 * math.pi * radius**3,
        "volume_ratio": mesh.volume / (4.0 / 3.0 * math.pi * radius**3),
        "min_edge_length": quality["min_edge_length"],
        "max_edge_length": quality["max_edge_length"],
        "max_aspect_ratio": quality["max_aspect_ratio"],
        "residual_l2": result.residual_l2,
        "relative_residual_l2": result.relative_residual_l2,
        "single_layer_condition_number": result.condition_number,
        "double_layer_row_sum_error": result.double_layer_row_sum_error,
        "note": (
            "Flat triangular panels approximate the sphere. Boundary values use "
            "the sphere solution at the centroid angular coordinate. The flux is "
            "the exterior fluid-normal derivative; the added mass uses the "
            "computational flux. Results should be assessed by refinement, not as "
            "an exact-sphere calculation. No free-surface, wave, or ship-hull "
            "physics is included."
        ),
    }


def run_refinement(
    subdivisions: Sequence[int] = (1, 2, 3, 4),
    radius: float = 1.0,
    speed: float = 1.0,
    density: float = 1025.0,
    tolerance: float = 0.03,
) -> dict[str, Any]:
    """Run the icosphere refinement ladder and compare against the closed form.

    Reference: ``m_A = 0.5 * rho * V`` (Lamb 1932, *Hydrodynamics*, 6th ed.,
    sec. 6.10; Newman 1977 eq. 4.132 sphere limit), machine-evaluated and
    cross-checked against ``tensorbem.analytic.ellipsoid_added_mass`` at the
    sphere point. Convergence orders use base 2 because each icosphere
    subdivision halves the edge length (panel count x4).
    """
    subs = sorted({int(s) for s in subdivisions})
    if not subs or any(s < 0 for s in subs):
        raise ValueError("subdivisions must be non-negative integers")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a positive finite number")

    levels = [run(s, radius, speed, density) for s in subs]
    exact = 0.5 * density * 4.0 / 3.0 * math.pi * radius**3
    from tensorbem.analytic import ellipsoid_added_mass

    analytic = float(ellipsoid_added_mass([radius, radius, radius], density)[0])

    rows: list[dict[str, Any]] = []
    for row in levels:
        got = float(row["added_mass"])
        rel_error = abs(got - exact) / exact
        rows.append(
            {
                "subdivisions": int(row["subdivisions"]),
                "panels": int(row["panels"]),
                "ref": exact,
                "got": got,
                "rel_error": rel_error,
                "pass": bool(rel_error < tolerance),
                "h_max": float(row["h_max"]),
                "normal_derivative_relative_l2_error": float(
                    row["normal_derivative_relative_l2_error"]
                ),
                "single_layer_condition_number": float(
                    row["single_layer_condition_number"]
                ),
            }
        )

    orders = [
        math.log2(earlier["rel_error"] / later["rel_error"])
        for earlier, later in zip(rows, rows[1:])
    ]

    # Negative control: radius 2a -> added mass scales by exactly 8 (a**3)
    # and the relative error is unchanged (linearity/scaling invariance).
    neg = run(subs[0], radius=2.0 * radius, speed=speed, density=density)
    neg_scale = float(neg["added_mass"]) / float(levels[0]["added_mass"])

    return {
        "benchmark": "three_dimensional_translating_sphere_refinement",
        "anchor_slug": "CASE-BEM-SPHERE-ADDED-MASS",
        "reference": {
            "closed_form": "m_A = 0.5*rho*V = (2/3)*pi*rho*a^3",
            "ref": exact,
            "analytic_module_cross_check": analytic,
            "analytic_module_rel_diff": abs(analytic - exact) / exact,
            "source": (
                "Lamb (1932) Hydrodynamics 6th ed. sec. 6.10; Newman (1977) "
                "eq. 4.132 sphere limit; closed form, machine-evaluated"
            ),
        },
        "tolerance": tolerance,
        "levels": rows,
        "convergence_orders_log2": orders,
        "negative_control_radius_2x": {
            "scale_expected": 8.0,
            "scale_measured": neg_scale,
            "rel_error_unchanged": bool(
                abs(
                    float(neg["added_mass_relative_error"])
                    - float(levels[0]["added_mass_relative_error"])
                )
                < 1e-12
            ),
        },
        "worst_rel_error": max(r["rel_error"] for r in rows),
        "finest_rel_error": float(rows[-1]["rel_error"]),
        "all_pass": all(r["pass"] for r in rows),
        "first_passing_level": next(
            (r["panels"] for r in rows if r["pass"]), None
        ),
        "scope": (
            "Exterior Dirichlet BEM with the exact analytic trace imposed; "
            "verifies the exterior single-layer operator and the boundary "
            "energy integral, not a Neumann moving-body BVP. No free surface."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivisions", type=int, default=2)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=1025.0)
    parser.add_argument(
        "--ladder",
        type=int,
        nargs="+",
        default=None,
        metavar="SUB",
        help="run the refinement ladder over these subdivision levels and "
        "emit the ref-vs-got benchmark instead of a single level",
    )
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--output", type=Path, help="also write machine-readable JSON")
    args = parser.parse_args()
    try:
        if args.ladder is not None:
            result = run_refinement(
                args.ladder,
                radius=args.radius,
                speed=args.speed,
                density=args.density,
                tolerance=args.tolerance,
            )
        else:
            result = run(args.subdivisions, args.radius, args.speed, args.density)
    except ValueError as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
