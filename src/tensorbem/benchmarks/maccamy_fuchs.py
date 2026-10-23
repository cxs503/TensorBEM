"""MacCamy-Fuchs (1954) vertical-cylinder wave diffraction and CHIEF study.

For a bottom-mounted, surface-piercing vertical cylinder of uniform horizontal
cross-section in water depth ``h``, linear Airy wave diffraction separates
vertically and reduces exactly to the 2-D exterior Helmholtz boundary-value
problem solved by :mod:`tensorbem.helmholtz`.

Three aspects of marine diffraction BEM are verified against the closed-form
Bessel/Hankel solution of MacCamy & Fuchs (1954):

1. **Mesh refinement (``ka = 1.0, h/a = 3.0``)**: second-order ``O(h**2)``
   convergence of the total horizontal wave force ``F_x``, phase lag, seabed
   overturning moment ``M_y``, and surface potential ``varphi(a, theta)``.
2. **Diffraction regime sweep (``ka`` from ``0.1`` to ``3.0``)**: captures the
   transition of the effective inertia coefficient ``C_M(ka)`` from the
   long-wave Morison value ``2.0`` down into the short-wave scattering regime.
3. **Irregular-frequency suppression**: evaluates the solver at the discrete
   interior Dirichlet resonances near ``J_0(ka) = 0`` and ``J_1(ka) = 0``,
   comparing the unaugmented boundary integral equation against Schenck's
   (1968) CHIEF interior null-field augmentation.

Run:
    python -m tensorbem.benchmarks.maccamy_fuchs --panels 16 32 64 128
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
    BESSEL_J0_FIRST_ZERO,
    BESSEL_J1_FIRST_ZERO,
    maccamy_fuchs_diffraction,
    maccamy_fuchs_surface_potential,
)
from tensorbem.geometry import regular_polygon
from tensorbem.helmholtz import solve_vertical_cylinder_diffraction


def _observed_orders(h_values: Sequence[float], errors: Sequence[float]) -> list[float | None]:
    orders: list[float | None] = [None]
    for idx in range(1, len(errors)):
        h0, h1 = h_values[idx - 1], h_values[idx]
        e0, e1 = errors[idx - 1], errors[idx]
        if e0 <= 0.0 or e1 <= 0.0 or h0 <= h1:
            orders.append(None)
        else:
            orders.append(float(math.log(e0 / e1) / math.log(h0 / h1)))
    return orders


def run_refinement_study(
    panel_counts: Sequence[int] = (16, 32, 64, 128),
    ka: float = 1.0,
    radius: float = 1.0,
    depth: float = 3.0,
    amplitude: float = 1.0,
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> dict[str, Any]:
    """Multi-resolution convergence of force, moment and surface potential."""
    if not panel_counts:
        raise ValueError("at least one panel count is required")
    exact = maccamy_fuchs_diffraction(
        ka, radius=radius, depth=depth, amplitude=amplitude, density=density, gravity=gravity
    )
    exact_fx = complex(exact["force_x"])
    exact_my = complex(exact["moment_about_seabed"]) if exact["moment_about_seabed"] is not None else None
    wavenumber = ka / radius

    levels: list[dict[str, Any]] = []
    h_list: list[float] = []
    force_errors: list[float] = []
    potential_errors: list[float] = []

    for count in panel_counts:
        mesh = regular_polygon(int(count), radius=radius)
        solved = solve_vertical_cylinder_diffraction(
            mesh,
            wavenumber,
            depth=depth,
            amplitude=amplitude,
            density=density,
            gravity=gravity,
            chief_points="auto",
        )
        angles = np.arctan2(mesh.centers[:, 1], mesh.centers[:, 0])
        exact_phi = maccamy_fuchs_surface_potential(ka, angles)
        phi_diff = solved.helmholtz.total_potential - exact_phi
        phi_rel_l2 = float(
            math.sqrt(
                float(np.sum(mesh.lengths * np.abs(phi_diff) ** 2))
                / float(np.sum(mesh.lengths * np.abs(exact_phi) ** 2))
            )
        )
        fx_num = complex(solved.force[0])
        fx_rel_err = float(abs(fx_num - exact_fx) / abs(exact_fx))
        my_rel_err: float | None = None
        if exact_my is not None and solved.moment_about_seabed is not None:
            my_rel_err = float(
                abs(complex(solved.moment_about_seabed[1]) - exact_my) / abs(exact_my)
            )

        h_max = float(np.max(mesh.lengths))
        h_list.append(h_max)
        force_errors.append(fx_rel_err)
        potential_errors.append(phi_rel_l2)
        levels.append(
            {
                "panels": mesh.panel_count,
                "h_max": h_max,
                "force_x_magnitude": float(abs(fx_num)),
                "exact_force_x_magnitude": float(abs(exact_fx)),
                "force_x_relative_error": fx_rel_err,
                "moment_y_relative_error": my_rel_err,
                "surface_potential_relative_l2_error": phi_rel_l2,
                "condition_number": solved.helmholtz.condition_number,
                "boundary_relative_residual": solved.helmholtz.boundary_relative_residual,
                "chief_relative_residual": solved.helmholtz.chief_relative_residual,
            }
        )

    return {
        "ka": float(ka),
        "radius": float(radius),
        "depth": float(depth),
        "levels": levels,
        "observed_orders": {
            "force_x_relative_error": _observed_orders(h_list, force_errors),
            "surface_potential_relative_l2_error": _observed_orders(
                h_list, potential_errors
            ),
        },
    }


def run_frequency_sweep(
    ka_values: Sequence[float] = (0.1, 0.5, 1.0, 1.5, 2.0, 3.0),
    panel_count: int = 64,
    radius: float = 1.0,
    depth: float = 3.0,
    amplitude: float = 1.0,
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> list[dict[str, float]]:
    """Sweep ``ka`` from the long-wave Morison limit into the diffraction regime."""
    mesh = regular_polygon(int(panel_count), radius=radius)
    area = math.pi * radius * radius
    rows: list[dict[str, float]] = []
    for ka in ka_values:
        k = float(ka) / radius
        exact = maccamy_fuchs_diffraction(
            float(ka),
            radius=radius,
            depth=depth,
            amplitude=amplitude,
            density=density,
            gravity=gravity,
        )
        solved = solve_vertical_cylinder_diffraction(
            mesh,
            k,
            depth=depth,
            amplitude=amplitude,
            density=density,
            gravity=gravity,
            chief_points="auto",
        )
        exact_fx = complex(exact["force_x"])
        num_fx = complex(solved.force[0])
        tanh_kh = math.tanh(k * depth) if not math.isinf(depth) else 1.0
        fk_scale = density * gravity * amplitude * area * tanh_kh
        rows.append(
            {
                "ka": float(ka),
                "computed_force_magnitude": float(abs(num_fx)),
                "exact_force_magnitude": float(abs(exact_fx)),
                "force_relative_error": float(abs(num_fx - exact_fx) / abs(exact_fx)),
                "computed_effective_cm": float(abs(num_fx) / fk_scale),
                "exact_effective_cm": float(exact["effective_inertia_coefficient"]),
                "exact_phase_lag_degrees": math.degrees(
                    float(exact["phase_lag_radians"])
                ),
                "condition_number": solved.helmholtz.condition_number,
            }
        )
    return rows


def run_irregular_frequency_check(
    panel_count: int = 64,
    radius: float = 1.0,
    depth: float = 3.0,
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> list[dict[str, Any]]:
    """Compare unaugmented BIE vs CHIEF at the discrete ``J_0`` and ``J_1`` resonances."""
    mesh = regular_polygon(int(panel_count), radius=radius)
    effective_radius_ratio = math.sqrt(mesh.area / (math.pi * radius * radius))
    cases = (
        ("j01_axisymmetric_resonance", BESSEL_J0_FIRST_ZERO / effective_radius_ratio),
        ("j11_dipole_resonance", BESSEL_J1_FIRST_ZERO / effective_radius_ratio),
    )
    results: list[dict[str, Any]] = []
    for name, ka in cases:
        k = ka / radius
        exact = maccamy_fuchs_diffraction(
            ka, radius=radius, depth=depth, density=density, gravity=gravity
        )
        exact_fx = complex(exact["force_x"])
        unaugmented = solve_vertical_cylinder_diffraction(
            mesh, k, depth=depth, density=density, gravity=gravity, chief_points=None
        )
        stabilised = solve_vertical_cylinder_diffraction(
            mesh, k, depth=depth, density=density, gravity=gravity, chief_points="auto"
        )
        results.append(
            {
                "resonance": name,
                "ka": float(ka),
                "unaugmented_condition_number": unaugmented.helmholtz.condition_number,
                "chief_condition_number": stabilised.helmholtz.condition_number,
                "unaugmented_force_relative_error": float(
                    abs(complex(unaugmented.force[0]) - exact_fx) / abs(exact_fx)
                ),
                "chief_force_relative_error": float(
                    abs(complex(stabilised.force[0]) - exact_fx) / abs(exact_fx)
                ),
            }
        )
    return results


def run(
    panel_counts: Sequence[int] = (16, 32, 64, 128),
    ka: float = 1.0,
    radius: float = 1.0,
    depth: float = 3.0,
    amplitude: float = 1.0,
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> dict[str, Any]:
    """Run refinement, ``ka`` sweep and irregular-frequency verification."""
    for name, val in (
        ("ka", ka),
        ("radius", radius),
        ("depth", depth),
        ("amplitude", amplitude),
        ("density", density),
        ("gravity", gravity),
    ):
        if math.isnan(val) or val <= 0.0:
            raise ValueError(f"{name} must be positive")
    refinement = run_refinement_study(
        panel_counts=panel_counts,
        ka=ka,
        radius=radius,
        depth=depth,
        amplitude=amplitude,
        density=density,
        gravity=gravity,
    )
    sweep_panels = max(int(c) for c in panel_counts)
    sweep = run_frequency_sweep(
        panel_count=min(sweep_panels, 64),
        radius=radius,
        depth=depth,
        amplitude=amplitude,
        density=density,
        gravity=gravity,
    )
    irregular = run_irregular_frequency_check(
        panel_count=min(sweep_panels, 64),
        radius=radius,
        depth=depth,
        density=density,
        gravity=gravity,
    )
    return {
        "benchmark": "maccamy_fuchs_vertical_cylinder_wave_diffraction",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "refinement": refinement,
        "frequency_sweep": sweep,
        "irregular_frequency_study": irregular,
        "note": (
            "Depth-uniform bottom-mounted vertical cylinder solved via exact "
            "vertical eigenmode separation + 2-D exterior Helmholtz BEM with "
            "Schenck CHIEF interior null-field regularisation. For general "
            "3-D submerged and surface-piercing bodies using the 3-D Kelvin "
            "free-surface Green function, see "
            "tensorbem.benchmarks.floating_hemisphere and "
            "tensorbem.benchmarks.submerged_sphere_wave."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--ka", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--depth", type=float, default=3.0)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=1025.0)
    parser.add_argument("--gravity", type=float, default=9.80665)
    parser.add_argument("--output", type=Path, help="also write machine-readable JSON")
    args = parser.parse_args()
    try:
        report = run(
            args.panels,
            args.ka,
            args.radius,
            args.depth,
            args.amplitude,
            args.density,
            args.gravity,
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
