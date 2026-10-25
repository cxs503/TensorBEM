"""Added mass of a translating ellipsoid (3-D exterior benchmark).

For an ellipsoid with semi-axes ``a1, a2, a3`` and volume ``V`` in an
unbounded, inviscid, incompressible fluid, the added mass for translation
along principal axis ``i`` is

    m_A,i = rho * V * alpha_i / (2 - alpha_i),

with Lamb's shape factors ``alpha_i`` supplied by
:mod:`tensorbem.analytic`. For a sphere ``alpha_i = 2/3`` and this is the
classical ``rho*V/2``; a slender spheroid has a small axial and a large
transverse added mass, which the sphere case cannot show.

The same structural simplification as for the sphere applies: the exterior
radiation potential of a *translating ellipsoid* has a linear Dirichlet trace,
``u = -U * k * x_axis`` with ``k = alpha_i/(2 - alpha_i)``, and the exact
fluid-domain normal derivative is ``q_e = -U * n_axis``. Neither needs a
pointwise exterior solution, so the benchmark measures the discretisation and
the exterior integral equation rather than the accuracy of a classical
series evaluation.

The surface is a flat-faceted ellipsoid built by scaling an icosphere. The
boundary values are prescribed at the radial projection of each panel centroid
onto the ellipsoid, which removes the artificial ``O(h**2)`` "depth" bias of
evaluating the trace at a centroid that lies *inside* the true surface. This is
the same convention as the sphere benchmark, which uses the angular coordinate
of the centroid.

Run: python -m tensorbem.benchmarks.ellipsoid_added_mass --subdivisions 2
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tensorbem.analytic import (
    ellipsoid_added_mass,
    ellipsoid_shape_factors,
    ellipsoid_translation_trace_scale,
    ellipsoid_volume,
)
from tensorbem.conditioning import estimate_operator_conditions
from tensorbem.geometry3d import ellipsoid_mesh, project_to_ellipsoid
from tensorbem.mesh_quality import diagnose_mesh_quality
from tensorbem.quadrature_diagnostics import diagnose_near_singular_quadrature
from tensorbem.solver3d import solve_laplace_surface

_AXIS_NAMES = ("x", "y", "z")


def _validate_semi_axes(semi_axes: tuple[float, float, float]) -> np.ndarray:
    axes = np.asarray(semi_axes, dtype=float)
    if axes.shape != (3,):
        raise ValueError("semi_axes must have three entries")
    if not np.isfinite(axes).all() or np.any(axes <= 0.0):
        raise ValueError("semi_axes must all be positive finite numbers")
    return axes


def run(
    subdivisions: int,
    semi_axes: tuple[float, float, float] = (1.5, 1.0, 0.75),
    speed: float = 1.0,
    density: float = 1025.0,
) -> dict[str, object]:
    """Run the translating-ellipsoid benchmark along all three axes."""
    for name, value in (("speed", speed), ("density", density)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")
    axes = _validate_semi_axes(semi_axes)

    mesh = ellipsoid_mesh(subdivisions, axes)
    centroids = mesh.centroids
    projected = project_to_ellipsoid(centroids, axes)
    exact_added_mass = ellipsoid_added_mass(axes, density)
    shape_factors = ellipsoid_shape_factors(axes)
    exact_volume = ellipsoid_volume(axes)
    quality = mesh.quality

    # Compute mesh quality diagnostics
    mesh_quality = diagnose_mesh_quality(mesh)

    rows: list[dict[str, object]] = []
    for axis, axis_name in enumerate(_AXIS_NAMES):
        trace_scale = ellipsoid_translation_trace_scale(axes, axis)
        boundary = -speed * trace_scale * projected[:, axis].copy()
        result = solve_laplace_surface(mesh, boundary, domain="exterior")

        # Compute conditioning diagnostics
        conditioning = estimate_operator_conditions(
            mesh, boundary, domain="exterior"
        )

        # Compute near-singular quadrature diagnostics
        quadrature_diag = diagnose_near_singular_quadrature(mesh)

        exact_flux = -speed * mesh.normals[:, axis]
        flux_error = result.normal_derivative - exact_flux
        reference = float(np.sqrt(np.sum(mesh.areas * exact_flux**2)))
        added_mass = float(
            density
            / speed**2
            * np.sum(boundary * result.normal_derivative * mesh.areas)
        )
        rows.append(
            {
                "benchmark": f"three_dimensional_translating_ellipsoid_{axis_name}",
                "motion_axis": axis_name,
                "semi_axes": [float(value) for value in axes],
                "subdivisions": int(subdivisions),
                "panels": mesh.panel_count,
                "vertices": mesh.vertex_count,
                "shape_factor": float(shape_factors[axis]),
                "boundary_trace_scale": trace_scale,
                "added_mass": added_mass,
                "exact_added_mass": float(exact_added_mass[axis]),
                "added_mass_relative_error": abs(
                    added_mass - float(exact_added_mass[axis])
                )
                / float(exact_added_mass[axis]),
                "normal_derivative_relative_l2_error": float(
                    np.sqrt(np.sum(mesh.areas * flux_error**2)) / reference
                ),
                "normal_derivative_rmse": float(np.sqrt(np.mean(flux_error**2))),
                "normal_derivative_max_abs_error": float(
                    np.max(np.abs(flux_error))
                ),
                "residual_l2": result.residual_l2,
                "relative_residual_l2": result.relative_residual_l2,
                "single_layer_condition_number": result.condition_number,
                "double_layer_row_sum_error": result.double_layer_row_sum_error,
                "single_layer_condition": conditioning.single_layer_condition,
                "double_layer_condition": conditioning.double_layer_condition,
                "near_singular_ratio": quadrature_diag.near_singular_fraction,
            }
        )

    return {
        "benchmark": "three_dimensional_translating_ellipsoid",
        "subdivisions": int(subdivisions),
        "panels": mesh.panel_count,
        "vertices": mesh.vertex_count,
        "semi_axes": [float(value) for value in axes],
        "aspect_ratios": [
            float(axes[0] / axes[1]),
            float(axes[1] / axes[2]),
        ],
        "speed": speed,
        "fluid_density": density,
        "shape_factors": [float(value) for value in shape_factors],
        "shape_factor_sum": float(np.sum(shape_factors)),
        "h_max": mesh.h_max,
        "h_mean": mesh.h_mean,
        "surface_area": mesh.surface_area,
        "volume": mesh.volume,
        "exact_volume": exact_volume,
        "volume_ratio": mesh.volume / exact_volume,
        "max_aspect_ratio": quality["max_aspect_ratio"],
        "min_angle_degrees": quality["min_angle_degrees"],
        "mesh_quality": asdict(mesh_quality),
        "added_mass_method": "boundary_energy_integral_using_computed_flux",
        "axes": rows,
        "note": (
            "Flat triangular panels approximate the ellipsoid, so the reference is "
            "the true ellipsoid: assess refinement trends, not a single value. The "
            "Dirichlet trace is the exact linear surface potential of the "
            "translating ellipsoid and the flux is the exterior fluid-normal "
            "derivative; the added mass uses the computed flux. No free-surface, "
            "wave, viscosity or ship-hull physics is included."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivisions", type=int, default=2)
    parser.add_argument(
        "--semi-axes", nargs=3, type=float, default=[1.5, 1.0, 0.75]
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=1025.0)
    parser.add_argument("--output", type=Path, help="also write machine-readable JSON")
    args = parser.parse_args()
    try:
        result = run(
            args.subdivisions,
            tuple(args.semi_axes),
            args.speed,
            args.density,
        )
    except ValueError as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
