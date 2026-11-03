"""End-to-end finite-depth radiation of a submerged circular cylinder.

This benchmark is the end-to-end counterpart of
:mod:`tensorbem.benchmarks.finite_depth_kernel`. That module validates the
finite-depth Green function point by point; this one drives the *boundary
element solver* with it and asks whether the hydrodynamic coefficients come
out right.

Geometry and reference
----------------------
Ursell's (1949) submerged circular cylinder, the same case as
:mod:`tensorbem.benchmarks.ursell_cylinder`: radius ``a = 1 m``, axis at
``z = -2 m``, non-dimensional frequency ``nu*a = 0.6`` (``nu = omega^2/g``),
``rho = 1025 kg/m^3``. The deep-water reference values, ``A = 2545.1672 kg/m``
and ``B = 2718.1540 kg/(m*s)``, are Ursell's multipole solution and are quoted
here only to check the ``depth -> inf`` limit; they are not an input to any
finite-depth number below.

What is checked, and why each check is independent
--------------------------------------------------
1. **Mode normalisation.** Feed the *same* deep-water potential to
   :func:`tensorbem.free_surface_solver.far_field_damping` with ``depth=None``
   and with ``depth = 50, 200, 800``. If the finite-depth energy identity is
   normalised correctly the three must reproduce the deep-water value to
   machine precision, since the same field is being measured. A wrong constant
   or a wrong ``c_g`` factor shows up here as a depth-independent offset. No
   BEM solve is involved, so this check cannot be masked by discretisation.
2. **Deep-water limit.** The finite-depth panel solution is compared with the
   deep-water panel solution at the same mesh (the depth effect only) and
   with Ursell's reference (the absolute accuracy). The method of fundamental
   solutions (:func:`tensorbem.free_surface_solver.radiation_reference`), which
   shares no panels and no matrix entries with the panel method, gives the
   finite-depth coefficients to ``~10^-3`` so the limit can be measured
   without waiting for panel convergence.
3. **Near-field/far-field energy identity.** The damping from the pressure
   integral must equal the damping from the outgoing-wave amplitude. At a fixed
   mesh this residual is dominated by the representation error of piecewise
   constant sources, so the benchmark reports the finite-depth residual next to
   the deep-water one on the same mesh: the identity itself is depth
   independent, and that is the claim being tested.
4. **Panel convergence.** Refining the body discretisation must reduce the
   distance to the independent MFS reference at fixed depth.

Run ``python -m tensorbem.benchmarks.finite_depth_radiation`` for the table.
"""

from __future__ import annotations

import json
import math

import numpy as np

from ..free_surface_solver import (
    far_field_damping,
    radiation_reference,
    solve_radiation_indirect,
)
from .ursell_cylinder import submerged_circle_mesh

NU = 0.6
DENSITY = 1025.0
GRAVITY = 9.80665

#: Ursell (1949) multipole reference, deep water, heave, per unit span.
REFERENCE_ADDED_MASS = 2545.1672
REFERENCE_DAMPING = 2718.1540

DEFAULT_DEPTHS = (8.0, 16.0, 32.0)


_SOLUTIONS: dict = {}


def reset_cache() -> None:
    """Drop the memoised solves (each finite-depth solve costs ~15 s)."""
    _SOLUTIONS.clear()


def _heave(nu: float, panels: int, depth: float | None):
    """Heave solve on the Ursell mesh, memoised on (nu, panels, depth)."""
    key = (nu, panels, depth)
    if key not in _SOLUTIONS:
        mesh = submerged_circle_mesh(panels)
        _SOLUTIONS[key] = (
            mesh,
            solve_radiation_indirect(mesh, nu, "heave", density=DENSITY, depth=depth),
        )
    return _SOLUTIONS[key]


def mode_normalisation(nu: float = NU, panels: int = 64, h_deep: float = 800.0) -> list[dict]:
    """Same potential, four water depths: the far-field constant is depth free."""
    mesh, deep = _heave(nu, panels, None)
    far_deep = far_field_damping(
        mesh, deep.potential, deep.normal_velocity, nu, density=DENSITY
    )
    rows = []
    for depth in (50.0, 200.0, h_deep):
        far_finite = far_field_damping(
            mesh, deep.potential, deep.normal_velocity, nu, density=DENSITY, depth=depth
        )
        rows.append(
            {
                "depth": depth,
                "nu_times_depth": nu * depth,
                "far_field_damping": far_finite,
                "far_field_damping_deep_water": far_deep,
                "relative_deviation": abs(far_finite / far_deep - 1.0),
            }
        )
    return rows


def deep_water_limit(
    nu: float = NU, panels: int = 64, depths=DEFAULT_DEPTHS
) -> list[dict]:
    """Leading the depth to infinity must recover the deep-water solution.

    ``panel_*`` are the panel-method coefficients, ``mfs_*`` the method of
    fundamental solutions on the same body mesh (independent representation),
    and ``reference_*`` Ursell's closed-form values.
    """
    mesh, deep = _heave(nu, panels, None)
    deep_added_mass = float(deep.added_mass[1])
    deep_damping = float(deep.damping[1])
    rows = []
    for depth in depths:
        finite = solve_radiation_indirect(
            mesh, nu, "heave", density=DENSITY, depth=depth
        )
        mfs = radiation_reference(mesh, nu, "heave", density=DENSITY, depth=depth)
        added_mass = float(finite.added_mass[1])
        damping = float(finite.damping[1])
        mfs_added_mass = float(mfs.added_mass[1])
        mfs_damping = float(mfs.damping[1])
        rows.append(
            {
                "depth": depth,
                "nu_times_depth": nu * depth,
                "panel_added_mass": added_mass,
                "panel_damping": damping,
                "panel_added_mass_vs_deep_water": abs(added_mass / deep_added_mass - 1.0),
                "panel_damping_vs_deep_water": abs(damping / deep_damping - 1.0),
                "mfs_added_mass": mfs_added_mass,
                "mfs_damping": mfs_damping,
                "mfs_added_mass_vs_ursell": abs(
                    mfs_added_mass / REFERENCE_ADDED_MASS - 1.0
                ),
                "mfs_damping_vs_ursell": abs(mfs_damping / REFERENCE_DAMPING - 1.0),
            }
        )
    return rows


def energy_identity(
    nu: float = NU, panels: int = 64, depths=DEFAULT_DEPTHS
) -> list[dict]:
    """Near-field pressure damping vs far-field wave-amplitude damping."""
    mesh, deep = _heave(nu, panels, None)
    rows = []
    for depth in (None,) + tuple(depths):
        if depth is None:
            solution = deep
            label = "deep water"
        else:
            solution = solve_radiation_indirect(
                mesh, nu, "heave", density=DENSITY, depth=depth
            )
            label = f"h = {depth:g} m"
        near = float(solution.damping[1])
        far = far_field_damping(
            mesh, solution.potential, solution.normal_velocity, nu,
            density=DENSITY, depth=depth,
        )
        mfs = radiation_reference(mesh, nu, "heave", density=DENSITY, depth=depth)
        mfs_near = float(mfs.damping[1])
        mfs_far = far_field_damping(
            mesh, mfs.potential, mfs.normal_velocity, nu, density=DENSITY, depth=depth
        )
        rows.append(
            {
                "case": label,
                "depth": depth,
                "near_field_damping": near,
                "far_field_damping": far,
                "panel_relative_residual": abs(far - near) / near,
                "mfs_relative_residual": abs(mfs_far - mfs_near) / mfs_near,
                "condition_number": solution.condition_number,
                "relative_residual": solution.relative_residual,
            }
        )
    return rows


def panel_convergence(
    nu: float = NU, depth: float = 16.0, panels=(32, 64, 128)
) -> list[dict]:
    """Distance to the MFS reference at fixed depth, versus body resolution."""
    rows = []
    for count in panels:
        mesh, solution = _heave(nu, count, depth)
        mfs = radiation_reference(mesh, nu, "heave", density=DENSITY, depth=depth)
        added_mass = float(solution.added_mass[1])
        damping = float(solution.damping[1])
        mfs_added_mass = float(mfs.added_mass[1])
        mfs_damping = float(mfs.damping[1])
        rows.append(
            {
                "panels": count,
                "depth": depth,
                "panel_added_mass": added_mass,
                "panel_damping": damping,
                "mfs_added_mass": mfs_added_mass,
                "mfs_damping": mfs_damping,
                "added_mass_relative_error": abs(added_mass - mfs_added_mass)
                / mfs_added_mass,
                "damping_relative_error": abs(damping - mfs_damping) / mfs_damping,
                "condition_number": solution.condition_number,
            }
        )
    return rows


def build_record(
    nu: float = NU, panels: int = 64, depths=DEFAULT_DEPTHS, panels_sweep=(32, 64)
) -> dict:
    """Machine-generated record; nothing in the documentation is hand-copied."""
    return {
        "case": {
            "body": "submerged circular cylinder, heave",
            "radius": 1.0,
            "axis_submergence": 2.0,
            "nu": nu,
            "nu_times_radius": nu * 1.0,
            "density": DENSITY,
            "gravity": GRAVITY,
            "omega": math.sqrt(nu * GRAVITY),
            "panels": panels,
            "depths": list(depths),
        },
        "deep_water_reference": {
            "added_mass": REFERENCE_ADDED_MASS,
            "damping": REFERENCE_DAMPING,
            "source": "Ursell (1949) multipole solution, deep water, per unit span",
        },
        "mode_normalisation": mode_normalisation(nu=nu, panels=panels),
        "deep_water_limit": deep_water_limit(nu=nu, panels=panels, depths=depths),
        "energy_identity": energy_identity(nu=nu, panels=panels, depths=depths),
        "panel_convergence": panel_convergence(
            nu=nu, depth=depths[1], panels=panels_sweep
        ),
    }


def _markdown(record: dict) -> str:
    case = record["case"]
    lines = [
        f"nu = {case['nu']:g}, a = {case['radius']:g} m, submergence "
        f"{case['axis_submergence']:g} m, {case['panels']} panels",
        "",
        "| h [m] | nu*h | A/MFS [kg/m] | err A vs Ursell | B/MFS [kg/(m s)] | err B vs Ursell |",
        "|---|---|---|---|---|---|",
    ]
    for row in record["deep_water_limit"]:
        lines.append(
            f"| {row['depth']:g} | {row['nu_times_depth']:.2f} | "
            f"{row['mfs_added_mass']:.3f} | {row['mfs_added_mass_vs_ursell']:.2e} | "
            f"{row['mfs_damping']:.3f} | {row['mfs_damping_vs_ursell']:.2e} |"
        )
    lines += [
        "",
        "| case | near-field B | far-field B | panel residual | MFS residual |",
        "|---|---|---|---|---|",
    ]
    for row in record["energy_identity"]:
        lines.append(
            f"| {row['case']} | {row['near_field_damping']:.3f} | "
            f"{row['far_field_damping']:.3f} | {row['panel_relative_residual']:.2e} | "
            f"{row['mfs_relative_residual']:.2e} |"
        )
    lines += [
        "",
        "| panels | A [kg/m] | B [kg/(m s)] | err A vs MFS | err B vs MFS |",
        "|---|---|---|---|---|",
    ]
    for row in record["panel_convergence"]:
        lines.append(
            f"| {row['panels']} | {row['panel_added_mass']:.3f} | "
            f"{row['panel_damping']:.3f} | {row['added_mass_relative_error']:.2e} | "
            f"{row['damping_relative_error']:.2e} |"
        )
    return "\n".join(lines)


def main() -> None:
    record = build_record()
    print(json.dumps(record, indent=2, sort_keys=True))
    print()
    print(_markdown(record))


if __name__ == "__main__":
    main()