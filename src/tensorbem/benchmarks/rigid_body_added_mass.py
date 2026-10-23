"""Exterior Neumann rigid-body added-mass and rotational-inertia benchmark.

Existing no-free-surface benchmarks in TensorBEM prescribe the analytical
Dirichlet trace ``phi|_Gamma`` and solve the first-kind integral equation
``S q_e = (1/2 I - H) phi``. For general hulls and rotational modes ``phi|_Gamma``
is unknown a priori; the physical boundary condition is the rigid-body
no-penetration *Neumann* condition ``d(phi_j)/dn_body = n_j``, solved via the
second-kind Fredholm equation

    (1/2 I - H) phi_j = S (-n_j).

This benchmark verifies both the 2-D ``(3, 3)`` rigid-body added-mass tensor
``(surge_x, heave_y, roll_xy)`` of an elliptic cylinder and the 3-D ``(6, 6)``
rigid-body added-mass tensor ``(surge, sway, heave, roll, pitch, yaw)`` of a
triaxial ellipsoid against Lamb's (1932, secs. 71, 114-115) closed-form
translational added masses and rotational added moments of inertia.

Run:
    python -m tensorbem.benchmarks.rigid_body_added_mass --panels 16 32 64 128 --subdivisions 0 1 2
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.analytic import (
    ellipse_added_mass_matrix,
    ellipsoid_added_mass_matrix,
    ellipsoid_shape_factors,
    ellipsoid_volume,
)
from tensorbem.geometry import ellipse_mesh
from tensorbem.geometry3d import ellipsoid_mesh
from tensorbem.solver import compute_added_mass_matrix_2d
from tensorbem.solver3d import compute_added_mass_matrix_3d

_MODES_2D = ("surge_x", "heave_y", "roll_xy")
_MODES_3D = ("surge_x", "sway_y", "heave_z", "roll_x", "pitch_y", "yaw_z")


def _observed_orders(h_values: Sequence[float], errors: Sequence[float]) -> list[float | None]:
    orders: list[float | None] = [None]
    for index in range(1, len(errors)):
        h_prev, h_curr = h_values[index - 1], h_values[index]
        e_prev, e_curr = errors[index - 1], errors[index]
        if e_prev <= 0.0 or e_curr <= 0.0 or h_prev <= h_curr:
            orders.append(None)
        else:
            orders.append(float(math.log(e_prev / e_curr) / math.log(h_prev / h_curr)))
    return orders


def run_ellipse_neumann_study(
    panel_counts: Sequence[int] = (16, 32, 64, 128),
    semi_major: float = 1.5,
    semi_minor: float = 0.75,
    density: float = 1025.0,
) -> dict[str, Any]:
    """Convergence of the 2-D ``(3, 3)`` added-mass tensor on an ellipse."""
    if not panel_counts:
        raise ValueError("at least one panel count is required")
    exact_matrix = ellipse_added_mass_matrix(semi_major, semi_minor, density)
    exact_diag = np.diag(exact_matrix)
    if np.any(exact_diag <= 0.0):
        raise ValueError("rotational inertia benchmark requires semi_major != semi_minor")

    levels: list[dict[str, Any]] = []
    h_list: list[float] = []
    per_mode_errors: dict[str, list[float]] = {mode: [] for mode in _MODES_2D}

    for count in panel_counts:
        mesh = ellipse_mesh(int(count), semi_major, semi_minor)
        solved = compute_added_mass_matrix_2d(mesh, density=density)
        diag = np.diag(solved.added_mass_matrix)
        off_diag = solved.added_mass_matrix - np.diag(diag)
        rel_errors = np.abs(diag - exact_diag) / exact_diag
        h_max = float(np.max(mesh.lengths))
        h_list.append(h_max)
        for idx, mode in enumerate(_MODES_2D):
            per_mode_errors[mode].append(float(rel_errors[idx]))

        levels.append(
            {
                "panels": mesh.panel_count,
                "h_max": h_max,
                "area_ratio": mesh.area / (math.pi * semi_major * semi_minor),
                "condition_number": solved.condition_number,
                "max_relative_residual_l2": solved.max_relative_residual_l2,
                "symmetry_error": solved.symmetry_error,
                "max_off_diagonal_abs": float(np.max(np.abs(off_diag))),
                "computed_diagonal": [float(v) for v in diag],
                "exact_diagonal": [float(v) for v in exact_diag],
                "diagonal_relative_errors": {
                    mode: float(rel_errors[idx]) for idx, mode in enumerate(_MODES_2D)
                },
            }
        )

    return {
        "case": "two_dimensional_ellipse_rigid_body_neumann",
        "semi_major": float(semi_major),
        "semi_minor": float(semi_minor),
        "fluid_density": float(density),
        "modes": list(_MODES_2D),
        "exact_diagonal": [float(v) for v in exact_diag],
        "levels": levels,
        "observed_orders": {
            mode: _observed_orders(h_list, per_mode_errors[mode]) for mode in _MODES_2D
        },
    }


def run_ellipsoid_neumann_study(
    subdivisions: Sequence[int] = (0, 1, 2),
    semi_axes: tuple[float, float, float] = (1.5, 1.0, 0.75),
    density: float = 1025.0,
) -> dict[str, Any]:
    """Convergence of the 3-D ``(6, 6)`` added-mass tensor on a triaxial ellipsoid."""
    if not subdivisions:
        raise ValueError("at least one subdivision level is required")
    axes = np.asarray(semi_axes, dtype=float)
    exact_matrix = ellipsoid_added_mass_matrix(axes, density)
    exact_diag = np.diag(exact_matrix)
    if np.any(exact_diag <= 0.0):
        raise ValueError(
            "6-DOF rotational inertia benchmark requires three distinct semi-axes"
        )
    shape_factors = ellipsoid_shape_factors(axes)
    exact_vol = ellipsoid_volume(axes)

    levels: list[dict[str, Any]] = []
    h_list: list[float] = []
    per_mode_errors: dict[str, list[float]] = {mode: [] for mode in _MODES_3D}

    for sub in subdivisions:
        mesh = ellipsoid_mesh(int(sub), axes)
        solved = compute_added_mass_matrix_3d(mesh, density=density)
        diag = np.diag(solved.added_mass_matrix)
        off_diag = solved.added_mass_matrix - np.diag(diag)
        rel_errors = np.abs(diag - exact_diag) / exact_diag
        h_list.append(mesh.h_max)
        for idx, mode in enumerate(_MODES_3D):
            per_mode_errors[mode].append(float(rel_errors[idx]))

        levels.append(
            {
                "subdivisions": int(sub),
                "panels": mesh.panel_count,
                "vertices": mesh.vertex_count,
                "h_max": mesh.h_max,
                "volume_ratio": mesh.volume / exact_vol,
                "condition_number": solved.condition_number,
                "max_relative_residual_l2": solved.max_relative_residual_l2,
                "double_layer_row_sum_error": solved.double_layer_row_sum_error,
                "symmetry_error": solved.symmetry_error,
                "max_off_diagonal_abs": float(np.max(np.abs(off_diag))),
                "computed_diagonal": [float(v) for v in diag],
                "exact_diagonal": [float(v) for v in exact_diag],
                "diagonal_relative_errors": {
                    mode: float(rel_errors[idx]) for idx, mode in enumerate(_MODES_3D)
                },
            }
        )

    return {
        "case": "three_dimensional_ellipsoid_rigid_body_neumann",
        "semi_axes": [float(v) for v in axes],
        "shape_factors": [float(v) for v in shape_factors],
        "fluid_density": float(density),
        "modes": list(_MODES_3D),
        "exact_diagonal": [float(v) for v in exact_diag],
        "levels": levels,
        "observed_orders": {
            mode: _observed_orders(h_list, per_mode_errors[mode]) for mode in _MODES_3D
        },
    }


def run(
    panel_counts: Sequence[int] = (16, 32, 64, 128),
    subdivisions: Sequence[int] = (0, 1, 2),
    density: float = 1025.0,
) -> dict[str, Any]:
    """Run both 2-D and 3-D Neumann rigid-body added-mass convergence studies."""
    ellipse_report = run_ellipse_neumann_study(
        panel_counts=panel_counts, density=density
    )
    ellipsoid_report = run_ellipsoid_neumann_study(
        subdivisions=subdivisions, density=density
    )
    return {
        "benchmark": "rigid_body_neumann_added_mass_and_rotational_inertia",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "ellipse_2d": ellipse_report,
        "ellipsoid_3d": ellipsoid_report,
        "note": (
            "Solved from the rigid-body no-penetration Neumann boundary condition "
            "without prescribing an analytical surface potential. The second-kind "
            "operator (1/2 I - H) maintains O(1) conditioning across refinement. "
            "No free-surface or wave effects are included."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--subdivisions", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--density", type=float, default=1025.0)
    parser.add_argument("--output", type=Path, help="also write machine-readable JSON")
    args = parser.parse_args()
    try:
        report = run(args.panels, args.subdivisions, args.density)
    except ValueError as error:
        parser.error(str(error))
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
