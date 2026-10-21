"""Potential-flow benchmark: the velocity field of a translating sphere.

For a sphere of radius ``a`` translating with speed ``U`` along ``+x`` in an
unbounded, inviscid, incompressible fluid, the exact exterior potential is

    phi(r, theta) = -(1/2) * U * a**3 * cos(theta) / r**2,   r >= a,

with boundary trace ``phi = -(1/2) * U * a * cos(theta)`` and fluid-domain
normal derivative ``q = -U * cos(theta)``.  Differentiating gives the exact
exterior velocity in spherical coordinates with the polar axis along ``+x``:

    v_r    = U * a**3 * cos(theta) / r**3
    v_theta = (1/2) * U * a**3 * sin(theta) / r**3

which is the analytic reference used here.

This benchmark exercises the *gradient* of the Green representation, which is
the quantity a potential-flow code needs for Kelvin sources, Bernoulli
pressure, and force integration.  The single-layer gradient kernel
``-(x - y)/(4*pi*|x - y|**3)`` is integrable and is evaluated with the same
fixed-order Duffy rule as the scalar potential.  The double-layer gradient is
hypersingular and is not integrable by ordinary quadrature; it is evaluated
from the closed-form gradient of the panel's signed solid angle, which is
exact for constant panels.

The surface is a flat-faceted icosphere, and the boundary values are
prescribed at panel centroids using the angular coordinate of each centroid,
so no artificial radial-depth bias is introduced.  This is a verification
case, not a free-surface or ship-hydrodynamics validation.

Two surface velocities are reported.  The *recovered* one builds the
velocity from the exact identity ``grad(phi)|_Gamma = grad_tan(u) + n*q``,
where the tangential derivative of the boundary values is recovered by a
superconvergent least-squares patch fit; it converges at second order in
the panel size.  The *representation* one differentiates the Green formula
with the piecewise-constant traces of the discretisation; the potential is
then accurate to ``O(h)`` but the gradient amplifies that error to ``O(1)``,
so it is reported as a diagnostic only.

Run: python -m tensorbem.benchmarks.sphere_velocity --subdivisions 2
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.geometry3d import icosphere
from tensorbem.solver3d import (
    evaluate_representation_surface_velocity_at_panels,
    evaluate_surface_velocity,
    evaluate_surface_velocity_at_panels,
    solve_laplace_surface,
)

DEFAULT_PROBE_RADII = (1.5, 2.5, 4.0)


def analytic_sphere_velocity(
    points: np.ndarray, radius: float = 1.0, speed: float = 1.0
) -> np.ndarray:
    """Exact exterior velocity of a sphere translating along ``+x``.

    Spherical coordinates use the polar axis along ``+x``:

    * ``cos(theta) = x / r``, ``sin(theta) = sqrt(y**2 + z**2) / r``
    * ``e_r  = ( cos theta, sin theta cos phi, sin theta sin phi)``
    * ``e_th = (-sin theta, cos theta cos phi, cos theta sin phi)``

    Returns an array of shape ``(M, 3)``.
    """
    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("points must have shape (M, 3)")
    r = np.linalg.norm(x, axis=1)
    if np.any(r <= 0.0):
        raise ValueError("points must not contain the origin")
    perpendicular = np.linalg.norm(x[:, 1:], axis=1)
    safe = np.where(perpendicular > 0.0, perpendicular, 1.0)
    cos_theta = x[:, 0] / r
    sin_theta = perpendicular / r
    cos_phi = np.where(perpendicular > 0.0, x[:, 1] / safe, 1.0)
    sin_phi = np.where(perpendicular > 0.0, x[:, 2] / safe, 0.0)
    v_r = speed * radius**3 * cos_theta / r**3
    v_theta = 0.5 * speed * radius**3 * sin_theta / r**3
    e_r = np.stack([cos_theta, sin_theta * cos_phi, sin_theta * sin_phi], axis=1)
    e_th = np.stack([-sin_theta, cos_theta * cos_phi, cos_theta * sin_phi], axis=1)
    return v_r[:, None] * e_r + v_theta[:, None] * e_th


def velocity_probe_points(
    radius: float = 1.0, probe_radii: Sequence[float] = DEFAULT_PROBE_RADII, count: int = 12
) -> np.ndarray:
    """Fixed exterior probes on spheres of the requested radii.

    The probes do not move with the surface mesh, so a refinement study
    measures the discretisation error rather than a changing sample set.
    """
    if count < 2:
        raise ValueError("count must be at least 2")
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    points = []
    for probe_radius in probe_radii:
        if not math.isfinite(probe_radius) or probe_radius <= radius:
            raise ValueError("probe radii must be finite and larger than the sphere radius")
        for index in range(count):
            y = 1.0 - (index / (count - 1.0)) * 2.0
            ring = math.sqrt(max(0.0, 1.0 - y * y))
            phi = math.radians(golden_angle * index)
            points.append(
                [
                    probe_radius * math.cos(phi) * ring,
                    probe_radius * math.sin(phi) * ring,
                    probe_radius * y,
                ]
            )
    return np.asarray(points, dtype=float)


def _probe_radii(
    radius: float, probe_radii: Sequence[float], count: int
) -> tuple[np.ndarray, list[float]]:
    """Return the probe points and the radius each probe belongs to."""
    points = velocity_probe_points(radius, probe_radii, count)
    radii = [float(value) for value in probe_radii for _ in range(count)]
    return points, radii


def run(
    subdivisions: int = 2,
    radius: float = 1.0,
    speed: float = 1.0,
    probe_radii: Sequence[float] = DEFAULT_PROBE_RADII,
    probe_count: int = 12,
) -> dict[str, Any]:
    """Run the translating-sphere velocity benchmark."""
    for name, value in (("radius", radius), ("speed", speed)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")
    if not probe_radii:
        raise ValueError("at least one probe radius is required")
    for probe_radius in probe_radii:
        if not math.isfinite(probe_radius) or probe_radius <= radius:
            raise ValueError("probe radii must be finite and larger than the sphere radius")

    mesh = icosphere(subdivisions, radius=radius)
    centroids = mesh.centroids
    radial_distance = np.linalg.norm(centroids, axis=1)
    cos_theta = centroids[:, 0] / radial_distance
    boundary_potential = -0.5 * speed * radius * cos_theta
    result = solve_laplace_surface(mesh, boundary_potential, domain="exterior")

    points, radii = _probe_radii(radius, probe_radii, probe_count)
    numerical = evaluate_surface_velocity(mesh, result, points, domain="exterior")
    exact = analytic_sphere_velocity(points, radius=radius, speed=speed)
    error = numerical - exact
    relative_l2 = float(
        math.sqrt(float(np.sum(error**2)) / float(np.sum(exact**2)))
    )
    per_radius = []
    for probe_radius in probe_radii:
        mask = np.asarray([value == probe_radius for value in radii])
        subset = error[mask]
        subset_exact = exact[mask]
        per_radius.append(
            {
                "probe_radius": float(probe_radius),
                "relative_l2_error": float(
                    math.sqrt(
                        float(np.sum(subset**2)) / float(np.sum(subset_exact**2))
                    )
                ),
                "max_abs_error": float(np.max(np.abs(subset))),
            }
        )

    # The surface velocity on the panel centroids.  The recovered path builds
    # the velocity from the exact identity grad(phi) = grad_tan(u) + n*q and
    # converges at second order; the representation path differentiates the
    # Green formula with the piecewise-constant traces and is reported as a
    # diagnostic of that amplification.
    surface_velocity = evaluate_surface_velocity_at_panels(mesh, result)
    surface_exact = analytic_sphere_velocity(centroids, radius=radius, speed=speed)
    surface_error = surface_velocity - surface_exact
    representation_velocity = evaluate_representation_surface_velocity_at_panels(
        mesh, result
    )
    representation_error = representation_velocity - surface_exact

    return {
        "benchmark": "three_dimensional_translating_sphere_velocity",
        "subdivisions": int(subdivisions),
        "panels": mesh.panel_count,
        "vertices": mesh.vertex_count,
        "radius": radius,
        "speed": speed,
        "probe_radii": [float(value) for value in probe_radii],
        "probe_count": int(probe_count),
        "velocity_relative_l2_error": relative_l2,
        "velocity_max_abs_error": float(np.max(np.abs(error))),
        "velocity_rmse": float(np.sqrt(np.mean(error**2))),
        "per_probe_radius": per_radius,
        "probe_exact": exact.tolist(),
        "probe_numerical": numerical.tolist(),
        "surface_velocity_relative_l2_error": float(
            math.sqrt(
                float(np.sum(surface_error**2)) / float(np.sum(surface_exact**2))
            )
        ),
        "surface_velocity_max_abs_error": float(np.max(np.abs(surface_error))),
        "representation_surface_velocity_relative_l2_error": float(
            math.sqrt(
                float(np.sum(representation_error**2))
                / float(np.sum(surface_exact**2))
            )
        ),
        "representation_surface_velocity_max_abs_error": float(
            np.max(np.abs(representation_error))
        ),
        "h_max": mesh.h_max,
        "h_mean": mesh.h_mean,
        "surface_area": mesh.surface_area,
        "exact_surface_area": 4.0 * math.pi * radius**2,
        "residual_l2": result.residual_l2,
        "relative_residual_l2": result.relative_residual_l2,
        "single_layer_condition_number": result.condition_number,
        "double_layer_row_sum_error": result.double_layer_row_sum_error,
        "note": (
            "Flat triangular panels approximate the sphere. Boundary values use "
            "the sphere solution at the centroid angular coordinate. The velocity "
            "is the exact exterior field of the translating sphere. The "
            "surface_velocity_* numbers use the recovered surface velocity, "
            "grad(phi) = grad_tan(u) + n*q, and are an accuracy claim. The "
            "representation_surface_velocity_* numbers differentiate the Green "
            "formula with the piecewise-constant traces and are reported as a "
            "diagnostic: the gradient amplifies the O(h) trace error to O(1). "
            "No free-surface, wave, viscosity, or ship-hull physics is "
            "included."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivisions", type=int, default=2)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--probe-radii", nargs="+", type=float, default=list(DEFAULT_PROBE_RADII)
    )
    parser.add_argument("--probe-count", type=int, default=12)
    parser.add_argument("--output", type=Path, help="also write machine-readable JSON")
    args = parser.parse_args()
    try:
        result = run(
            args.subdivisions,
            args.radius,
            args.speed,
            args.probe_radii,
            args.probe_count,
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
