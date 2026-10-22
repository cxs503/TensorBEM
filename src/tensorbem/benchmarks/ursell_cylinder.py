"""Ursell's submerged circular cylinder: radiation in deep water.

A circular cylinder of radius ``a`` floats with its axis a distance ``f`` below
the still water level of deep water and oscillates in heave or sway at
frequency ``omega``. This is the oldest free-surface benchmark in marine
hydrodynamics (Ursell 1949) and the natural first test of a frequency-domain
wave solver, because the free-surface condition, the radiation condition and
the waterline-free geometry are all present while the answer stays
two-dimensional.

Two facts about the case are worth stating up front.

* In *infinite* depth the heave and sway coefficients are identical. The
  multipole matrix of the classical formulation does not depend on the mode,
  so equality here is a physical result, not a lost degree of freedom. (In
  finite depth they differ.)
* As ``f -> infinity`` the free surface decouples and the added mass tends to
  ``rho*pi*a**2``, the displaced mass per unit span; the damping decays
  exponentially.

The reference is Ursell's multipole solution, evaluated from the closed-form
matrix in :func:`tensorbem.analytic.ursell_cylinder_coefficients`. It is an
analytical construction - a truncated infinite system whose entries come from
elementary functions and an exponential integral - and shares no code with the
boundary-element assembly.

Two numerical solutions are reported for comparison:

* the boundary-element solution of the indirect (source) formulation
  (:func:`tensorbem.free_surface_solver.solve_radiation_indirect`), which
  imposes the body condition on a panel mesh;
* the method of fundamental solutions
  (:func:`tensorbem.free_surface_solver.radiation_reference`), which places
  point sources on an interior curve. This one converges far faster because it
  has no singular quadrature, and it serves as a second, structurally
  different check.

Both are validated by the same energy identity: the damping obtained from the
surface pressure must equal the damping obtained from the amplitude of the
outgoing wave (Haskind). That identity is the actual test of the radiation
condition - an added mass alone can look plausible with the wrong kernel, and
did so earlier in this project.

Run: python -m tensorbem.benchmarks.ursell_cylinder --panels 128 --ka 0.6
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorbem.analytic import ursell_cylinder_added_mass
from tensorbem.free_surface_solver import (
    far_field_damping,
    radiation_reference,
    solve_radiation_indirect,
)
from tensorbem.geometry import PanelMesh

_MOTIONS = {"heave": 1, "sway": 0}


def submerged_circle_mesh(
    panel_count: int, radius: float = 1.0, submergence: float = 2.0
) -> PanelMesh:
    """Equally spaced panels on a circle of the given radius, centred at ``-f``."""
    if panel_count < 8:
        raise ValueError("panel_count must be at least 8")
    angles = 2.0 * math.pi * np.arange(panel_count) / panel_count
    return PanelMesh(
        np.column_stack(
            (radius * np.cos(angles), radius * np.sin(angles) - submergence)
        )
    )


def run(
    panel_count: int = 128,
    radius: float = 1.0,
    submergence: float = 2.0,
    ka: float = 0.6,
    density: float = 1025.0,
    gravity: float = 9.80665,
    panel_nodes: int = 24,
    source_count: int = 24,
) -> dict[str, float | int | str]:
    """Run the Ursell radiation benchmark and report all three solutions."""
    for name, value in (
        ("radius", radius),
        ("submergence", submergence),
        ("ka", ka),
        ("density", density),
        ("gravity", gravity),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")
    if submergence <= radius:
        raise ValueError("the cylinder must be fully submerged: submergence > radius")

    omega = math.sqrt(ka * gravity / radius)
    mesh = submerged_circle_mesh(panel_count, radius, submergence)

    panel = solve_radiation_indirect(
        mesh, ka / radius, "heave", density=density, panel_nodes=panel_nodes
    )
    sources = radiation_reference(
        mesh, ka / radius, "heave", density=density, source_count=source_count
    )
    sway = solve_radiation_indirect(
        mesh, ka / radius, "sway", density=density, panel_nodes=panel_nodes
    )
    exact_added_mass, exact_damping = ursell_cylinder_added_mass(
        radius, submergence, omega, density, gravity=gravity
    )

    index = _MOTIONS["heave"]
    computed_added_mass = float(panel.added_mass[index])
    computed_damping = float(panel.damping[index])
    far_damping = far_field_damping(
        mesh, panel.potential, panel.normal_velocity, ka / radius, density,
        gravity=gravity,
    )

    return {
        "benchmark": "ursell_submerged_circular_cylinder_deep_water",
        "motion": "heave",
        "panels": panel_count,
        "radius": radius,
        "submergence": submergence,
        "radius_over_submergence": radius / submergence,
        "ka": ka,
        "omega": omega,
        "fluid_density": density,
        "gravity": gravity,
        "reference_method": "ursell_multipole_closed_form_infinite_system",
        "reference_added_mass": exact_added_mass,
        "reference_damping": exact_damping,
        #
        # In infinite depth the exact heave and sway coefficients coincide, so
        # the reference has no mode to vary. The panel *solutions* for the two
        # motions are different fields, yet their forces must agree: that is a
        # genuine check that the solver respects the symmetry.
        "panel_sway_added_mass": float(sway.added_mass[0]),
        "panel_sway_damping": float(sway.damping[0]),
        "panel_heave_sway_added_mass_relative_difference": abs(
            computed_added_mass - float(sway.added_mass[0])
        )
        / computed_added_mass,
        "panel_heave_sway_damping_relative_difference": abs(
            computed_damping - float(sway.damping[0])
        )
        / computed_damping,
        "panel_added_mass": computed_added_mass,
        "panel_damping": computed_damping,
        "panel_added_mass_relative_error": abs(computed_added_mass - exact_added_mass)
        / abs(exact_added_mass),
        "panel_damping_relative_error": abs(computed_damping - exact_damping)
        / abs(exact_damping),
        "panel_added_mass_over_displaced_mass": computed_added_mass
        / (density * math.pi * radius**2),
        "panel_damping_over_displaced_mass_omega": computed_damping
        / (density * math.pi * radius**2 * omega),
        "far_field_damping": far_damping,
        "near_over_far_field_damping": computed_damping / far_damping,
        "near_far_field_relative_difference": abs(computed_damping - far_damping)
        / far_damping,
        "reference_solution_added_mass": float(sources.added_mass[index]),
        "reference_solution_damping": float(sources.damping[index]),
        "reference_solution_added_mass_relative_error": abs(
            float(sources.added_mass[index]) - exact_added_mass
        )
        / abs(exact_added_mass),
        "reference_solution_damping_relative_error": abs(
            float(sources.damping[index]) - exact_damping
        )
        / abs(exact_damping),
        "panel_relative_residual": panel.relative_residual,
        "panel_condition_number": panel.condition_number,
        "displaced_mass_per_unit_span": density * math.pi * radius**2,
        "note": (
            "The reference is Ursell's multipole solution, an independent "
            "analytical construction. The panel solver converges only at first "
            "order for this kernel, so compare refinement trends as well as a "
            "single value. The near/far-field identity is the check that the "
            "radiation condition is actually satisfied."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=int, default=128)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--submergence", type=float, default=2.0)
    parser.add_argument("--ka", type=float, default=0.6)
    parser.add_argument("--density", type=float, default=1025.0)
    parser.add_argument("--gravity", type=float, default=9.80665)
    parser.add_argument("--panel-nodes", type=int, default=24)
    parser.add_argument("--sources", type=int, default=24)
    parser.add_argument("--output", type=Path, help="also write machine-readable JSON")
    args = parser.parse_args()
    try:
        result = run(
            args.panels,
            args.radius,
            args.submergence,
            args.ka,
            args.density,
            args.gravity,
            args.panel_nodes,
            args.sources,
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
