"""Two-dimensional deep-water wave diffraction: Dean's theorem and Haskind reciprocity.

This benchmark extends the two-dimensional deep-water free-surface layer from
forced radiation (Ursell 1949) to incident-wave scattering and wave-excitation
loads:

1. **Submerged circular cylinder (Dean 1948; Ursell 1950; Ogilvie 1963).**
   A regular deep-water Airy wave of unit amplitude passes over a fixed circular
   cylinder of radius ``a`` at axis submergence ``f > a``. Three exact results
   are verified simultaneously against the independent multipole solution
   :func:`tensorbem.analytic.dean_cylinder_diffraction` and an interior-source
   MFS solution:
   * *Zero reflection*: ``|R| = 0`` identically at all frequencies and depths;
   * *Unit transmission & phase shift*: ``|T| = 1``, ``T = exp(i*delta)``;
   * *Circularly polarised wave load*: ``F_z = -i * F_x`` (equal horizontal and
     vertical load amplitudes with a 90-degree phase lag).
2. **Haskind-Newman reciprocity.** Wave exciting forces obtained from integrating
   the diffraction pressure over the body are cross-checked against
   :func:`tensorbem.free_surface_solver.haskind_excitation_force`, which uses
   only the calm-water sway/heave radiation solutions.
3. **Submerged elliptic cylinder.** Breaking circular symmetry produces a
   non-zero reflection coefficient ``|R| > 0`` while still obeying wave-energy
   conservation ``|R|**2 + |T|**2 = 1`` and Haskind reciprocity.

Run:
    python -m tensorbem.benchmarks.dean_diffraction --panels 128 --ka 0.6
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.analytic import dean_cylinder_diffraction
from tensorbem.benchmarks.ursell_cylinder import submerged_circle_mesh
from tensorbem.free_surface_solver import (
    diffraction_reference,
    haskind_excitation_force,
    solve_diffraction_indirect,
    solve_radiation_indirect,
)
from tensorbem.geometry import PanelMesh


def submerged_ellipse_mesh(
    panel_count: int,
    semi_major: float = 1.4,
    semi_minor: float = 0.7,
    submergence: float = 2.0,
) -> PanelMesh:
    """Inscribed counter-clockwise ellipse centred at ``(0, -submergence)``."""
    if panel_count < 8:
        raise ValueError("panel_count must be at least 8")
    if semi_major <= 0.0 or semi_minor <= 0.0 or submergence <= semi_minor:
        raise ValueError("submerged ellipse requires submergence > semi_minor > 0")
    angles = 2.0 * math.pi * np.arange(panel_count) / panel_count
    vertices = np.column_stack(
        (semi_major * np.cos(angles), semi_minor * np.sin(angles) - submergence)
    )
    return PanelMesh(vertices)


def _complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def run(
    panel_count: int = 128,
    radius: float = 1.0,
    submergence: float = 2.0,
    ka: float = 0.6,
    amplitude: float = 1.0,
    density: float = 1025.0,
    gravity: float = 9.80665,
    panel_nodes: int = 24,
    source_count: int = 32,
) -> dict[str, Any]:
    """Run the Dean-Ursell-Ogilvie circle diffraction and ellipse scattering study."""
    for name, value in (
        ("radius", radius),
        ("submergence", submergence),
        ("ka", ka),
        ("amplitude", amplitude),
        ("density", density),
        ("gravity", gravity),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")
    if submergence <= radius:
        raise ValueError("the cylinder must be fully submerged: submergence > radius")

    nu = ka / radius
    circle = submerged_circle_mesh(panel_count, radius, submergence)
    exact = dean_cylinder_diffraction(
        radius, submergence, ka, amplitude, density, gravity=gravity
    )
    panel = solve_diffraction_indirect(
        circle,
        nu,
        amplitude,
        density=density,
        gravity=gravity,
        panel_nodes=panel_nodes,
    )
    mfs = diffraction_reference(
        circle,
        nu,
        amplitude,
        source_count=source_count,
        density=density,
        gravity=gravity,
    )
    rad_sway = solve_radiation_indirect(
        circle, nu, "sway", density=density, panel_nodes=panel_nodes
    )
    rad_heave = solve_radiation_indirect(
        circle, nu, "heave", density=density, panel_nodes=panel_nodes
    )
    haskind_x = haskind_excitation_force(
        circle, rad_sway, nu, amplitude, density=density, gravity=gravity
    )
    haskind_z = haskind_excitation_force(
        circle, rad_heave, nu, amplitude, density=density, gravity=gravity
    )

    exact_t = complex(exact["transmission"])
    exact_fx = complex(exact["force_x"])
    exact_fz = complex(exact["force_z"])
    fx_panel = complex(panel.excitation_force[0])
    fz_panel = complex(panel.excitation_force[1])

    # Non-circular submerged ellipse: non-zero reflection + energy + Haskind check.
    ellipse_a = 1.4 * radius
    ellipse_b = 0.7 * radius
    ellipse_f = max(submergence, 2.0 * ellipse_b)
    ellipse = submerged_ellipse_mesh(panel_count, ellipse_a, ellipse_b, ellipse_f)
    ellipse_diff = solve_diffraction_indirect(
        ellipse,
        nu,
        amplitude,
        density=density,
        gravity=gravity,
        panel_nodes=panel_nodes,
    )
    ellipse_rad_sway = solve_radiation_indirect(
        ellipse, nu, "sway", density=density, panel_nodes=panel_nodes
    )
    ellipse_rad_heave = solve_radiation_indirect(
        ellipse, nu, "heave", density=density, panel_nodes=panel_nodes
    )
    ellipse_haskind_x = haskind_excitation_force(
        ellipse, ellipse_rad_sway, nu, amplitude, density=density, gravity=gravity
    )
    ellipse_haskind_z = haskind_excitation_force(
        ellipse, ellipse_rad_heave, nu, amplitude, density=density, gravity=gravity
    )

    return {
        "benchmark": "dean_submerged_cylinder_wave_diffraction_deep_water",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "panels": int(panel_count),
        "radius": float(radius),
        "submergence": float(submergence),
        "ka": float(ka),
        "wavenumber": float(nu),
        "wave_amplitude": float(amplitude),
        "fluid_density": float(density),
        "gravity": float(gravity),
        "circular_cylinder": {
            "reference_method": "ursell_ogilvie_multipole_closed_form",
            "exact_reflection_modulus": 0.0,
            "exact_transmission_real_imag": _complex_pair(exact_t),
            "exact_transmission_phase_degrees": math.degrees(
                float(exact["transmission_phase_radians"])
            ),
            "exact_force_x_real_imag": _complex_pair(exact_fx),
            "exact_force_z_real_imag": _complex_pair(exact_fz),
            "panel_reflection_modulus": float(abs(panel.reflection_coefficient)),
            "panel_transmission_modulus": float(abs(panel.transmission_coefficient)),
            "panel_transmission_phase_degrees": math.degrees(
                math.atan2(
                    panel.transmission_coefficient.imag,
                    panel.transmission_coefficient.real,
                )
            ),
            "panel_transmission_error": float(
                abs(panel.transmission_coefficient - exact_t)
            ),
            "panel_energy_balance_error": panel.energy_balance_error,
            "panel_force_x_real_imag": _complex_pair(fx_panel),
            "panel_force_z_real_imag": _complex_pair(fz_panel),
            "panel_force_x_relative_error": float(
                abs(fx_panel - exact_fx) / abs(exact_fx)
            ),
            "panel_force_z_relative_error": float(
                abs(fz_panel - exact_fz) / abs(exact_fz)
            ),
            "panel_vertical_over_horizontal_force_modulus": float(
                abs(fz_panel) / abs(fx_panel)
            ),
            "panel_circular_polarisation_error": float(
                abs(fz_panel - (-1.0j * fx_panel)) / abs(fx_panel)
            ),
            "haskind_force_x_real_imag": _complex_pair(haskind_x),
            "haskind_force_z_real_imag": _complex_pair(haskind_z),
            "haskind_diffraction_force_x_relative_difference": float(
                abs(fx_panel - haskind_x) / abs(fx_panel)
            ),
            "haskind_diffraction_force_z_relative_difference": float(
                abs(fz_panel - haskind_z) / abs(fz_panel)
            ),
            "mfs_reflection_modulus": float(abs(mfs.reflection_coefficient)),
            "mfs_transmission_error": float(abs(mfs.transmission_coefficient - exact_t)),
            "mfs_force_x_relative_error": float(
                abs(mfs.excitation_force[0] - exact_fx) / abs(exact_fx)
            ),
            "panel_condition_number": panel.condition_number,
            "panel_relative_residual": panel.relative_residual,
        },
        "elliptic_cylinder": {
            "semi_major": float(ellipse_a),
            "semi_minor": float(ellipse_b),
            "submergence": float(ellipse_f),
            "reflection_modulus": float(abs(ellipse_diff.reflection_coefficient)),
            "transmission_modulus": float(abs(ellipse_diff.transmission_coefficient)),
            "energy_balance_error": ellipse_diff.energy_balance_error,
            "haskind_diffraction_force_x_relative_difference": float(
                abs(ellipse_diff.excitation_force[0] - ellipse_haskind_x)
                / abs(ellipse_diff.excitation_force[0])
            ),
            "haskind_diffraction_force_z_relative_difference": float(
                abs(ellipse_diff.excitation_force[1] - ellipse_haskind_z)
                / abs(ellipse_diff.excitation_force[1])
            ),
            "condition_number": ellipse_diff.condition_number,
            "relative_residual": ellipse_diff.relative_residual,
        },
        "note": (
            "Two-dimensional deep-water linear diffraction around fully submerged "
            "cylinders. Dean's zero-reflection theorem holds for the circular "
            "cross-section and breaks for the ellipse, while |R|**2 + |T|**2 = 1 "
            "and Haskind reciprocity hold for both."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=int, default=128)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--submergence", type=float, default=2.0)
    parser.add_argument("--ka", type=float, default=0.6)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=1025.0)
    parser.add_argument("--gravity", type=float, default=9.80665)
    parser.add_argument("--panel-nodes", type=int, default=24)
    parser.add_argument("--sources", type=int, default=32)
    parser.add_argument("--output", type=Path, help="also write machine-readable JSON")
    args = parser.parse_args()
    try:
        result = run(
            args.panels,
            args.radius,
            args.submergence,
            args.ka,
            args.amplitude,
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
