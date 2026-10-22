"""Added mass of a translating elliptic cylinder (2-D exterior benchmark).

For an elliptic cylinder of semi-axes ``a`` along ``x`` and ``b`` along ``y`` in
an unbounded, inviscid, incompressible fluid, the added mass per unit span is

    m_A,x = rho * pi * b**2,   m_A,y = rho * pi * a**2.

Note that each value uses the semi-axis *perpendicular* to the motion, so an
elongated ellipse has a small axial and a large transverse added mass.

The exterior radiation potential of a translating elliptic cylinder has the
exact Dirichlet trace ``u = -U * k * x_axis`` with ``k = b/a`` for motion along
``x`` and ``k = a/b`` along ``y``. That linear trace is what makes this a clean
benchmark: the boundary data are known in closed form and need no pointwise
exterior solution, while the flux still comes from the boundary-element solve.
The fluid-domain normal derivative is ``q_e = -U * n_axis`` exactly.

This case exercises the 2-D *exterior* Dirichlet formulation, which the
cylinder benchmark did not: for a circle the first harmonic mode is the same
linear field inside and out, so the bounded-domain solver reproduces the
exterior flux *in the continuum*. An ellipse breaks that coincidence - the
exact flux is not the interior harmonic extension of the trace - so this
benchmark is the first test of the exterior operator itself.

Run: python -m tensorbem.benchmarks.ellipse_added_mass --panels 64
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorbem.analytic import (
    ellipse_added_mass,
    ellipse_translation_trace_scale,
)
from tensorbem.geometry import ellipse_mesh
from tensorbem.solver import solve_laplace_exterior_dirichlet

_AXES = {"x": 0, "y": 1}


def run(
    panel_count: int,
    semi_major: float = 1.5,
    semi_minor: float = 0.75,
    speed: float = 1.0,
    density: float = 1025.0,
    axis: str = "x",
) -> dict[str, float | int | str]:
    """Run the translating-ellipse added-mass benchmark for one motion axis."""
    for name, value in (("speed", speed), ("density", density)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")
    if axis not in _AXES:
        raise ValueError("axis must be 'x' or 'y'")

    index = _AXES[axis]
    mesh = ellipse_mesh(panel_count, semi_major, semi_minor)
    trace_scale = ellipse_translation_trace_scale(semi_major, semi_minor, axis)
    boundary_potential = -speed * trace_scale * mesh.centers[:, index].copy()
    result = solve_laplace_exterior_dirichlet(mesh, boundary_potential)

    exact_flux = -speed * mesh.normals[:, index]
    flux_error = result.normal_derivative - exact_flux
    flux_reference = float(np.sqrt(np.sum(mesh.lengths * exact_flux**2)))
    added_mass = float(
        density
        / speed**2
        * np.sum(boundary_potential * result.normal_derivative * mesh.lengths)
    )
    exact_added_mass = ellipse_added_mass(semi_major, semi_minor, density)[index]
    exact_area = math.pi * semi_major * semi_minor

    return {
        "benchmark": f"two_dimensional_translating_ellipse_{axis}",
        "panels": mesh.panel_count,
        "semi_major": semi_major,
        "semi_minor": semi_minor,
        "aspect_ratio": semi_major / semi_minor,
        "speed": speed,
        "fluid_density": density,
        "motion_axis": axis,
        "boundary_trace_scale": trace_scale,
        "added_mass_per_unit_span": added_mass,
        "exact_added_mass_per_unit_span": exact_added_mass,
        "added_mass_relative_error": abs(added_mass - exact_added_mass)
        / exact_added_mass,
        "added_mass_method": "boundary_energy_integral_using_computed_flux",
        "normal_derivative_relative_l2_error": float(
            np.sqrt(np.sum(mesh.lengths * flux_error**2)) / flux_reference
        ),
        "normal_derivative_rmse": float(np.sqrt(np.mean(flux_error**2))),
        "normal_derivative_max_abs_error": float(np.max(np.abs(flux_error))),
        "h_max": float(np.max(mesh.lengths)),
        "perimeter": mesh.perimeter,
        "area": mesh.area,
        "exact_area": exact_area,
        "area_ratio": mesh.area / exact_area,
        "residual_l2": result.residual_l2,
        "relative_residual_l2": result.relative_residual_l2,
        "single_layer_condition_number": result.condition_number,
        "note": (
            "Inscribed polygon approximating the ellipse, so the reference is the "
            "true ellipse: assess refinement trends, not a single value. The flux "
            "comes from the exterior 2-D Dirichlet solve with the fluid-domain "
            "normal, and the added mass uses that computed flux."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=int, default=64)
    parser.add_argument("--semi-major", type=float, default=1.5)
    parser.add_argument("--semi-minor", type=float, default=0.75)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=1025.0)
    parser.add_argument("--axis", choices=sorted(_AXES), default="x")
    parser.add_argument("--output", type=Path, help="also write machine-readable JSON")
    args = parser.parse_args()
    try:
        result = run(
            args.panels,
            args.semi_major,
            args.semi_minor,
            args.speed,
            args.density,
            args.axis,
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
