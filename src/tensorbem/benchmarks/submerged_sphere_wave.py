"""Wang (1986) / Linton (1991) 3-D submerged-sphere wave radiation & diffraction.

Verifies the 3-D deep-water Kelvin Green-function BEM solver
(:mod:`tensorbem.free_surface_solver3d`) on a closed geodesic sphere of radius
``a`` centered at ``(0, 0, -f)`` against the closed-form Thorne/Wang/Linton
multipole reference (:func:`tensorbem.analytic.submerged_sphere_wave_multipole`):

1. **Mesh refinement (``Ka = 0.8, f/a = 1.5``)**: ``h``-convergence across
   icosphere subdivisions (``20, 80, 320`` panels) for surge/heave added mass,
   radiation damping, and complex diffraction forces ``(F_x, F_z)``, alongside
   Kochin far-field damping conservation, 3-D Haskind reciprocity, and 6-DOF
   rotational decoupling about the sphere center.
2. **Frequency sweep (``Ka`` from ``0.2`` to ``2.0``)**: compares BEM against
   exact multipole solutions across the full free-surface wave-making band.

Run:
    python -m tensorbem.benchmarks.submerged_sphere_wave --subdivisions 0 1 2
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    far_field_damping_matrix_3d,
    haskind_excitation_force_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import submerged_sphere_mesh


def _observed_orders(
    h_values: Sequence[float], errors: Sequence[float]
) -> list[float | None]:
    orders: list[float | None] = [None]
    for idx in range(1, len(errors)):
        h0, h1 = h_values[idx - 1], h_values[idx]
        e0, e1 = errors[idx - 1], errors[idx]
        if e0 <= 0.0 or e1 <= 0.0 or h0 <= h1:
            orders.append(None)
        else:
            orders.append(float(math.log(e0 / e1) / math.log(h0 / h1)))
    return orders


def _complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def run_refinement_study(
    subdivisions: Sequence[int] = (0, 1, 2),
    ka: float = 0.8,
    radius: float = 1.0,
    submergence: float = 1.5,
    amplitude: float = 1.0,
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> dict[str, Any]:
    """Convergence of 3-D submerged-sphere radiation and diffraction."""
    if not subdivisions:
        raise ValueError("at least one subdivision level is required")
    ref = submerged_sphere_wave_multipole(
        ka,
        radius=radius,
        submergence=submergence,
        amplitude=amplitude,
        density=density,
        gravity=gravity,
    )
    nu = ka / radius
    exact_vol = ref.displaced_volume
    sphere_center = (0.0, 0.0, -float(submergence))

    levels: list[dict[str, Any]] = []
    h_list: list[float] = []
    err_a33: list[float] = []
    err_b33: list[float] = []
    err_a11: list[float] = []
    err_fz: list[float] = []

    for sub in subdivisions:
        mesh = submerged_sphere_mesh(int(sub), radius=radius, submergence=submergence)
        ops = assemble_free_surface_operators_3d(mesh, nu)
        rad = solve_radiation_3d(
            mesh,
            nu,
            center=sphere_center,
            density=density,
            gravity=gravity,
            operators=ops,
        )
        diff = solve_diffraction_3d(
            mesh,
            nu,
            amplitude=amplitude,
            center=sphere_center,
            density=density,
            gravity=gravity,
            operators=ops,
        )
        b_far = far_field_damping_matrix_3d(
            mesh, rad, density=density, gravity=gravity
        )
        hask = haskind_excitation_force_3d(
            mesh,
            rad,
            amplitude=amplitude,
            density=density,
            gravity=gravity,
        )

        a11_dim = float(rad.added_mass[0, 0] / (density * exact_vol))
        b11_dim = float(rad.damping[0, 0] / (density * ref.omega * exact_vol))
        a33_dim = float(rad.added_mass[2, 2] / (density * exact_vol))
        b33_dim = float(rad.damping[2, 2] / (density * ref.omega * exact_vol))

        fx_bem = complex(diff.excitation_force[0])
        fz_bem = complex(diff.excitation_force[2])
        fx_hask = complex(hask["boundary_haskind"][0])
        fz_hask = complex(hask["boundary_haskind"][2])

        e_a33 = abs(a33_dim - ref.heave_added_mass_dimless) / ref.heave_added_mass_dimless
        e_b33 = abs(b33_dim - ref.heave_damping_dimless) / ref.heave_damping_dimless
        e_a11 = abs(a11_dim - ref.surge_added_mass_dimless) / ref.surge_added_mass_dimless
        e_fz = abs(fz_bem - ref.excitation_force_z) / abs(ref.excitation_force_z)
        e_fx = abs(fx_bem - ref.excitation_force_x) / abs(ref.excitation_force_x)

        h_list.append(mesh.h_mean)
        err_a33.append(e_a33)
        err_b33.append(e_b33)
        err_a11.append(e_a11)
        err_fz.append(e_fz)

        levels.append(
            {
                "subdivisions": int(sub),
                "panel_count": mesh.panel_count,
                "h_mean": mesh.h_mean,
                "displaced_volume": mesh.volume,
                "surge_added_mass_dimless": a11_dim,
                "surge_damping_dimless": b11_dim,
                "heave_added_mass_dimless": a33_dim,
                "heave_damping_dimless": b33_dim,
                "surge_added_mass_rel_error": e_a11,
                "heave_added_mass_rel_error": e_a33,
                "heave_damping_rel_error": e_b33,
                "excitation_force_x_rel_error": e_fx,
                "excitation_force_z_rel_error": e_fz,
                "rotational_added_mass_max_ratio": float(
                    np.max(np.abs(rad.added_mass[3:, 3:])) / max(rad.added_mass[2, 2], 1.0e-30)
                ),
                "far_field_damping_rel_diff_heave": float(
                    abs(rad.damping[2, 2] - b_far[2, 2]) / max(abs(rad.damping[2, 2]), 1.0e-30)
                ),
                "haskind_reciprocity_rel_diff_x": float(
                    abs(fx_bem - fx_hask) / max(abs(fx_bem), 1.0e-30)
                ),
                "haskind_reciprocity_rel_diff_z": float(
                    abs(fz_bem - fz_hask) / max(abs(fz_bem), 1.0e-30)
                ),
                "condition_number": rad.condition_number,
                "relative_residual": rad.relative_residual,
            }
        )

    orders_a33 = _observed_orders(h_list, err_a33)
    orders_b33 = _observed_orders(h_list, err_b33)
    orders_a11 = _observed_orders(h_list, err_a11)
    orders_fz = _observed_orders(h_list, err_fz)
    for idx, row in enumerate(levels):
        row["observed_order_a33"] = orders_a33[idx]
        row["observed_order_b33"] = orders_b33[idx]
        row["observed_order_a11"] = orders_a11[idx]
        row["observed_order_fz"] = orders_fz[idx]

    return {
        "ka": ka,
        "submergence_ratio": submergence / radius,
        "exact_reference": {
            "surge_added_mass_dimless": ref.surge_added_mass_dimless,
            "surge_damping_dimless": ref.surge_damping_dimless,
            "heave_added_mass_dimless": ref.heave_added_mass_dimless,
            "heave_damping_dimless": ref.heave_damping_dimless,
            "excitation_force_x": _complex_pair(ref.excitation_force_x),
            "excitation_force_z": _complex_pair(ref.excitation_force_z),
            "haskind_error_x": ref.haskind_error_x,
            "haskind_error_z": ref.haskind_error_z,
        },
        "levels": levels,
    }


def run_frequency_sweep(
    subdivision: int = 2,
    ka_values: Sequence[float] = (0.2, 0.5, 0.8, 1.0, 1.5, 2.0),
    radius: float = 1.0,
    submergence: float = 1.5,
    amplitude: float = 1.0,
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> list[dict[str, Any]]:
    """Frequency sweep across submerged-sphere radiation and diffraction."""
    mesh = submerged_sphere_mesh(int(subdivision), radius=radius, submergence=submergence)
    center = (0.0, 0.0, -float(submergence))
    rows: list[dict[str, Any]] = []
    for ka in ka_values:
        nu = float(ka) / radius
        ref = submerged_sphere_wave_multipole(
            float(ka),
            radius=radius,
            submergence=submergence,
            amplitude=amplitude,
            density=density,
            gravity=gravity,
        )
        ops = assemble_free_surface_operators_3d(mesh, nu)
        rad = solve_radiation_3d(
            mesh, nu, modes=(0, 2), center=center, density=density, gravity=gravity, operators=ops
        )
        diff = solve_diffraction_3d(
            mesh,
            nu,
            amplitude=amplitude,
            center=center,
            density=density,
            gravity=gravity,
            operators=ops,
        )
        vol = ref.displaced_volume
        a11 = float(rad.added_mass[0, 0] / (density * vol))
        b11 = float(rad.damping[0, 0] / (density * ref.omega * vol))
        a33 = float(rad.added_mass[1, 1] / (density * vol))
        b33 = float(rad.damping[1, 1] / (density * ref.omega * vol))
        rows.append(
            {
                "ka": float(ka),
                "bem_a11": a11,
                "exact_a11": ref.surge_added_mass_dimless,
                "bem_b11": b11,
                "exact_b11": ref.surge_damping_dimless,
                "bem_a33": a33,
                "exact_a33": ref.heave_added_mass_dimless,
                "bem_b33": b33,
                "exact_b33": ref.heave_damping_dimless,
                "force_x_rel_error": float(
                    abs(diff.excitation_force[0] - ref.excitation_force_x)
                    / abs(ref.excitation_force_x)
                ),
                "force_z_rel_error": float(
                    abs(diff.excitation_force[2] - ref.excitation_force_z)
                    / abs(ref.excitation_force_z)
                ),
                "condition_number": rad.condition_number,
            }
        )
    return rows


def run(
    subdivisions: Sequence[int] = (0, 1, 2),
    ka: float = 0.8,
    radius: float = 1.0,
    submergence: float = 1.5,
    amplitude: float = 1.0,
    density: float = 1025.0,
    gravity: float = 9.80665,
    *,
    quick: bool = False,
) -> dict[str, Any]:
    """Execute the full submerged-sphere 3-D wave verification suite."""
    subs = (0, 1) if quick else tuple(int(s) for s in subdivisions)
    fine_sub = max(subs)
    sweep_kas = (0.4, 0.8, 1.5) if quick else (0.2, 0.5, 0.8, 1.0, 1.5, 2.0)

    return {
        "benchmark": "submerged_sphere_3d_free_surface_wave",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "refinement": run_refinement_study(
            subdivisions=subs,
            ka=ka,
            radius=radius,
            submergence=submergence,
            amplitude=amplitude,
            density=density,
            gravity=gravity,
        ),
        "frequency_sweep": run_frequency_sweep(
            subdivision=fine_sub,
            ka_values=sweep_kas,
            radius=radius,
            submergence=submergence,
            amplitude=amplitude,
            density=density,
            gravity=gravity,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivisions", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--ka", type=float, default=0.8)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--submergence", type=float, default=1.5)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=1025.0)
    parser.add_argument("--gravity", type=float, default=9.80665)
    parser.add_argument("--quick", action="store_true", help="fast CI smoke run")
    parser.add_argument("--output", type=Path, help="also write machine-readable JSON")
    args = parser.parse_args()
    try:
        report = run(
            args.subdivisions,
            args.ka,
            args.radius,
            args.submergence,
            args.amplitude,
            args.density,
            args.gravity,
            quick=args.quick,
        )
    except ValueError as error:
        parser.error(str(error))
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
