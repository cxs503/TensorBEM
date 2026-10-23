"""Hulme (1982) 3-D floating-hemisphere radiation, diffraction, and CHIEF study.

Verifies the 3-D deep-water Kelvin Green-function BEM solver
(:mod:`tensorbem.free_surface_solver3d`) on a surface-piercing hemisphere of
radius ``a`` against Hulme's (1982) semi-analytical spherical wave-free
multipole reference (:func:`tensorbem.analytic.hulme_hemisphere_radiation`):

1. **Mesh refinement (``Ka = 1.0``)**: ``h``-convergence across recursive
   octahedral hull subdivisions (``16, 64, 256`` panels) for surge/heave added
   mass, radiation damping, regular-wave diffraction forces, Kochin far-field
   damping conservation, and 3-D Haskind reciprocity.
2. **Frequency sweep (``Ka`` from ``0.3`` to ``3.0``)**: captures the peak in
   radiation damping and high-frequency decay across surge and heave.
3. **3-D irregular-frequency suppression (``Ka ~ 2.58``)**: demonstrates how
   the unregularized surface-piercing BIE exhibits a sharp condition-number
   spike and unphysical *negative* heave damping ``B_33 < 0`` near the first
   interior waterplane resonance, whereas 5 interior 3-D CHIEF null-field
   equations restore smooth conditioning and accurate positive damping without
   waterplane lid panels.

Run:
    python -m tensorbem.benchmarks.floating_hemisphere --subdivisions 1 2 3
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    far_field_damping_matrix_3d,
    haskind_excitation_force_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh


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
    subdivisions: Sequence[int] = (1, 2, 3),
    ka: float = 1.0,
    radius: float = 1.0,
    amplitude: float = 1.0,
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> dict[str, Any]:
    """Convergence of 3-D floating-hemisphere radiation and diffraction."""
    if not subdivisions:
        raise ValueError("at least one subdivision level is required")
    ref = hulme_hemisphere_radiation(
        ka, radius=radius, amplitude=amplitude, density=density, gravity=gravity
    )
    nu = ka / radius
    exact_vol = ref.displaced_volume

    levels: list[dict[str, Any]] = []
    h_list: list[float] = []
    err_a33: list[float] = []
    err_b33: list[float] = []
    err_a11: list[float] = []
    err_fz: list[float] = []

    for sub in subdivisions:
        mesh = hemisphere_hull_mesh(int(sub), radius=radius)
        ops = assemble_free_surface_operators_3d(mesh, nu, chief_points="auto")
        rad = solve_radiation_3d(
            mesh,
            nu,
            modes=(0, 2),
            density=density,
            gravity=gravity,
            operators=ops,
        )
        diff = solve_diffraction_3d(
            mesh,
            nu,
            amplitude=amplitude,
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
        a33_dim = float(rad.added_mass[1, 1] / (density * exact_vol))
        b33_dim = float(rad.damping[1, 1] / (density * ref.omega * exact_vol))

        fx_bem = complex(diff.excitation_force[0])
        fz_bem = complex(diff.excitation_force[2])
        fx_hask = complex(hask["boundary_haskind"][0])
        fz_hask = complex(hask["boundary_haskind"][1])

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
                "waterplane_area": mesh.waterplane_area,
                "surge_added_mass_dimless": a11_dim,
                "surge_damping_dimless": b11_dim,
                "heave_added_mass_dimless": a33_dim,
                "heave_damping_dimless": b33_dim,
                "surge_added_mass_rel_error": e_a11,
                "heave_added_mass_rel_error": e_a33,
                "heave_damping_rel_error": e_b33,
                "excitation_force_x_rel_error": e_fx,
                "excitation_force_z_rel_error": e_fz,
                "far_field_damping_rel_diff_heave": float(
                    abs(rad.damping[1, 1] - b_far[1, 1]) / max(abs(rad.damping[1, 1]), 1.0e-30)
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
        "exact_reference": {
            "surge_added_mass_dimless": ref.surge_added_mass_dimless,
            "surge_damping_dimless": ref.surge_damping_dimless,
            "heave_added_mass_dimless": ref.heave_added_mass_dimless,
            "heave_damping_dimless": ref.heave_damping_dimless,
            "excitation_force_x": _complex_pair(ref.excitation_force_x),
            "excitation_force_z": _complex_pair(ref.excitation_force_z),
        },
        "levels": levels,
    }


def run_frequency_sweep(
    subdivision: int = 3,
    ka_values: Sequence[float] = (0.3, 0.6, 1.0, 1.5, 2.0, 3.0),
    radius: float = 1.0,
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> list[dict[str, Any]]:
    """Frequency sweep across Hulme's wave-radiation regime."""
    mesh = hemisphere_hull_mesh(int(subdivision), radius=radius)
    rows: list[dict[str, Any]] = []
    for ka in ka_values:
        ref = hulme_hemisphere_radiation(
            float(ka), radius=radius, density=density, gravity=gravity
        )
        rad = solve_radiation_3d(
            mesh,
            float(ka) / radius,
            modes=(0, 2),
            density=density,
            gravity=gravity,
            chief_points="auto",
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
                "condition_number": rad.condition_number,
            }
        )
    return rows


def run_irregular_frequency_study(
    subdivision: int = 3,
    ka_values: Sequence[float] = (2.52, 2.58, 2.64),
    radius: float = 1.0,
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> list[dict[str, Any]]:
    """Compare unregularized BIE vs 3-D interior CHIEF near ``Ka ~ 2.58``."""
    mesh = hemisphere_hull_mesh(int(subdivision), radius=radius)
    rows: list[dict[str, Any]] = []
    for ka in ka_values:
        nu = float(ka) / radius
        ref = hulme_hemisphere_radiation(
            float(ka), radius=radius, density=density, gravity=gravity
        )
        vol = ref.displaced_volume
        no_chief = solve_radiation_3d(
            mesh, nu, modes=(2,), density=density, gravity=gravity, chief_points=None
        )
        with_chief = solve_radiation_3d(
            mesh, nu, modes=(2,), density=density, gravity=gravity, chief_points="auto"
        )
        b33_no = float(no_chief.damping[0, 0] / (density * ref.omega * vol))
        b33_ch = float(with_chief.damping[0, 0] / (density * ref.omega * vol))
        a33_no = float(no_chief.added_mass[0, 0] / (density * vol))
        a33_ch = float(with_chief.added_mass[0, 0] / (density * vol))
        rows.append(
            {
                "ka": float(ka),
                "exact_a33": ref.heave_added_mass_dimless,
                "exact_b33": ref.heave_damping_dimless,
                "unregularized": {
                    "condition_number": no_chief.condition_number,
                    "a33_dimless": a33_no,
                    "b33_dimless": b33_no,
                    "b33_rel_error": abs(b33_no - ref.heave_damping_dimless)
                    / ref.heave_damping_dimless,
                    "passivity_violated": bool(b33_no < 0.0),
                },
                "chief_regularized": {
                    "chief_point_count": int(with_chief.chief_points_used.shape[0]),
                    "condition_number": with_chief.condition_number,
                    "a33_dimless": a33_ch,
                    "b33_dimless": b33_ch,
                    "b33_rel_error": abs(b33_ch - ref.heave_damping_dimless)
                    / ref.heave_damping_dimless,
                    "passivity_violated": bool(b33_ch < 0.0),
                },
            }
        )
    return rows


def run(
    subdivisions: Sequence[int] = (1, 2, 3),
    ka: float = 1.0,
    radius: float = 1.0,
    amplitude: float = 1.0,
    density: float = 1025.0,
    gravity: float = 9.80665,
    *,
    quick: bool = False,
) -> dict[str, Any]:
    """Execute the full Hulme (1982) floating-hemisphere verification suite."""
    subs = (1, 2) if quick else tuple(int(s) for s in subdivisions)
    fine_sub = max(subs)
    sweep_kas = (0.5, 1.0, 2.0) if quick else (0.3, 0.6, 1.0, 1.5, 2.0, 3.0)
    irreg_sub = 2 if quick else max(fine_sub, 3)
    irreg_kas = (2.58,) if quick else (2.52, 2.58, 2.64)

    return {
        "benchmark": "hulme_floating_hemisphere_3d_free_surface",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "refinement": run_refinement_study(
            subdivisions=subs,
            ka=ka,
            radius=radius,
            amplitude=amplitude,
            density=density,
            gravity=gravity,
        ),
        "frequency_sweep": run_frequency_sweep(
            subdivision=fine_sub,
            ka_values=sweep_kas,
            radius=radius,
            density=density,
            gravity=gravity,
        ),
        "irregular_frequency_study": run_irregular_frequency_study(
            subdivision=irreg_sub,
            ka_values=irreg_kas,
            radius=radius,
            density=density,
            gravity=gravity,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivisions", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--ka", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=1.0)
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
