"""Verification benchmark for finite-depth wave diffraction and Haskind reciprocity.

Physics
-------
A circular cylinder (radius ``a = 1 m``, axis submergence ``f = 2 m``) in a
strip of constant depth ``h``, loaded by a regular Airy wave of unit elevation
amplitude travelling towards ``+x``.  The heave exciting force per unit span and
the reflection/transmission coefficients are the reported quantities.

Four independent checks are run on every case:

1. **Haskind-Newman reciprocity** -- the exciting force from radiation data
   alone (:func:`tensorbem.finite_depth_diffraction.haskind_excitation_force`)
   against the pressure integral of the diffraction solve
   (:func:`...solve_diffraction_indirect`).  Different unknowns (radiation
   densities vs diffraction densities), same answer.
2. **Independent discretisation** -- the method of fundamental solutions
   (:func:`...diffraction_reference`) on the same kernel: no panel density, no
   singular quadrature, no shared matrix entries.
3. **Energy flux balance** -- ``|R|^2 + |T|^2 = 1`` for a fixed body; the group
   velocity is the same in both directions, so elevation ratios are flux ratios.
4. **Deep-water collapse** -- at ``k0 h`` large the finite-depth answer must
   reproduce the *separately implemented* deep-water solver
   (:mod:`tensorbem.free_surface_solver`).

Run: ``python -m tensorbem.benchmarks.finite_depth_diffraction --output out.json``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tensorbem import free_surface_solver as deep_water
from tensorbem.benchmarks.ursell_cylinder import submerged_circle_mesh
from tensorbem.finite_depth import finite_depth_wavenumber
from tensorbem.finite_depth_diffraction import (
    diffraction_reference,
    haskind_excitation_force,
    incident_wave,
    solve_diffraction_indirect,
)
from tensorbem.finite_depth_solver import solve_radiation_indirect

_GRAVITY = 9.80665
_DENSITY = 1025.0
_RADIUS = 1.0
_SUBMERGENCE = 2.0


def _case(panel_count: int, ka: float, depth: float) -> dict[str, object]:
    nu = ka / _RADIUS
    mesh = submerged_circle_mesh(panel_count, _RADIUS, _SUBMERGENCE)
    radiation = solve_radiation_indirect(mesh, nu, depth, "heave")
    direct = solve_diffraction_indirect(mesh, nu, depth)
    reference = diffraction_reference(mesh, nu, depth)
    haskind = haskind_excitation_force(mesh, radiation, nu, depth)
    x_direct = complex(direct.excitation_force[1])
    x_reference = complex(reference.excitation_force[1])
    return {
        "panels": panel_count,
        "ka": ka,
        "depth": depth,
        "depth_over_radius": depth / _RADIUS,
        "bottom_clearance_over_radius": (depth - _SUBMERGENCE - _RADIUS) / _RADIUS,
        "k0_times_depth": float(finite_depth_wavenumber(nu, depth) * depth),
        "excitation_direct_real": x_direct.real,
        "excitation_direct_imag": x_direct.imag,
        "excitation_direct_abs": abs(x_direct),
        "excitation_mfs_real": x_reference.real,
        "excitation_mfs_imag": x_reference.imag,
        "haskind_reciprocity_rel": float(abs(haskind - x_direct) / abs(x_direct)),
        "mfs_reference_rel": float(abs(x_reference - x_direct) / abs(x_direct)),
        "reflection_abs": abs(direct.reflection_coefficient),
        "transmission_abs": abs(direct.transmission_coefficient),
        "energy_balance_error_panel": direct.energy_balance_error,
        "energy_balance_error_mfs": reference.energy_balance_error,
        "area_condition_number": direct.condition_number,
        "boundary_relative_residual": direct.relative_residual,
    }


def _deep_water_collapse(panel_count: int, ka: float, depth: float) -> dict[str, object]:
    nu = ka / _RADIUS
    mesh = submerged_circle_mesh(panel_count, _RADIUS, _SUBMERGENCE)
    finite = solve_diffraction_indirect(mesh, nu, depth)
    deep = deep_water.solve_diffraction_indirect(mesh, nu)
    x_finite = complex(finite.excitation_force[1])
    x_deep = complex(deep.excitation_force[1])
    return {
        "panels": panel_count,
        "ka": ka,
        "depth": depth,
        "k0_times_depth": float(finite_depth_wavenumber(nu, depth) * depth),
        "excitation_finite_depth_abs": abs(x_finite),
        "excitation_deep_water_abs": abs(x_deep),
        "excitation_relative_difference": float(abs(x_finite - x_deep) / abs(x_deep)),
        "reflection_finite_depth": abs(finite.reflection_coefficient),
        "reflection_deep_water": abs(deep.reflection_coefficient),
        "reflection_relative_difference": float(
            abs(abs(finite.reflection_coefficient) - abs(deep.reflection_coefficient))
            / abs(deep.reflection_coefficient)
        ),
    }


def _incident_wave_checks(ka: float, depth: float) -> dict[str, object]:
    """Free-surface and bottom conditions of the incident potential, by finite differences."""
    nu = ka / _RADIUS
    k0 = finite_depth_wavenumber(nu, depth)
    unit = np.array([[0.0, 1.0]])
    step = 1.0e-6
    z0 = -1.0e-4
    probe = np.array([[0.0, z0]])
    derivative = (
        incident_wave(np.array([[0.0, z0 + step]]), unit, nu, depth)[0]
        - incident_wave(np.array([[0.0, z0 - step]]), unit, nu, depth)[0]
    ) / (2.0 * step)
    phi0 = complex(np.ravel(incident_wave(probe, unit, nu, depth)[0])[0])
    free_surface_error = float(abs(complex(np.ravel(derivative)[0]) - nu * phi0) / abs(nu * phi0))
    bottom = np.array([[0.0, -depth + 1.0e-3]])
    phi_bottom, dphi_bottom = incident_wave(bottom, unit, nu, depth)
    bottom_error = float(abs(complex(np.ravel(dphi_bottom)[0])) / (k0 * abs(complex(np.ravel(phi_bottom)[0]))))
    # The depth-dependent decay rate must be reproduced pointwise for all z.
    z = np.linspace(-depth + 1.0e-3, -1.0e-3, 41)
    points = np.column_stack((np.zeros_like(z), z))
    normals = np.tile([0.0, 1.0], (len(z), 1))
    phi, dphi_dn = incident_wave(points, normals, nu, depth)
    identity = dphi_dn - phi * k0 * np.tanh(k0 * (z + depth))
    analytic_consistency = float(np.max(np.abs(identity)) / np.max(np.abs(dphi_dn)))
    return {
        "ka": ka,
        "depth": depth,
        "free_surface_condition_relative_error": free_surface_error,
        "bottom_condition_relative_error": bottom_error,
        "normal_derivative_identity_relative_error": analytic_consistency,
    }


def run(
    panels: int = 64,
    ka_values: tuple[float, ...] = (0.2, 0.6, 1.0),
    depth_ratios: tuple[float, ...] = (4.0, 8.0),
    *,
    convergence_panels: tuple[int, ...] = (32, 64),
    deep_depth: float = 60.0,
) -> dict[str, object]:
    sweep = [
        _case(panels, ka, depth_ratio * _RADIUS)
        for ka in ka_values
        for depth_ratio in depth_ratios
    ]
    convergence = [
        _case(count, 0.6, 4.0 * _RADIUS) for count in convergence_panels
    ]
    collapse = _deep_water_collapse(panels, 0.6, deep_depth * _RADIUS)
    waves = [_incident_wave_checks(ka, 4.0 * _RADIUS) for ka in ka_values]
    return {
        "case": {
            "geometry": "circular cylinder, per unit span",
            "radius_m": _RADIUS,
            "axis_submergence_m": _SUBMERGENCE,
            "density_kg_m3": _DENSITY,
            "gravity_m_s2": _GRAVITY,
            "reported_quantity": "heave exciting force amplitude per unit span, unit incident elevation",
        },
        "sweep": sweep,
        "panel_convergence": convergence,
        "deep_water_collapse": collapse,
        "incident_wave_conditions": waves,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    payload = run(args.panels)
    text = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())