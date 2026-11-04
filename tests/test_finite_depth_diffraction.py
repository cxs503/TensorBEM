"""Tests for finite-depth wave diffraction and Haskind-Newman reciprocity.

The gates are the four independent checks described in
:mod:`tensorbem.benchmarks.finite_depth_diffraction`, run on a coarse mesh so
that the suite stays fast:

* incident potential satisfies the finite-depth free-surface and bottom
  conditions (analytic identities, exact to rounding);
* deep-water limit of the incident potential equals the deep-water solver's;
* Haskind reciprocity between radiation and diffraction datasets;
* agreement with the fundamental-solution reference;
* energy flux balance ``|R|^2 + |T|^2 = 1``;
* deep-water collapse against the (separately verified) deep-water solver.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

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

GRAVITY = 9.80665
RADIUS = 1.0
SUBMERGENCE = 2.0
DEPTH = 4.0
KA = 0.6


@pytest.fixture(scope="module")
def mesh():
    return submerged_circle_mesh(32, RADIUS, SUBMERGENCE)


@pytest.fixture(scope="module")
def diffraction(mesh):
    return solve_diffraction_indirect(mesh, KA, DEPTH)


@pytest.fixture(scope="module")
def radiation(mesh):
    return solve_radiation_indirect(mesh, KA, DEPTH, "heave")


def test_incident_wave_satisfies_free_surface_and_bottom_conditions():
    nu = KA / RADIUS
    depth = DEPTH
    k0 = finite_depth_wavenumber(nu, depth)
    z = np.linspace(-depth + 1.0e-3, -1.0e-3, 21)
    points = np.column_stack((np.zeros_like(z), z))
    normals = np.tile([0.0, 1.0], (len(z), 1))
    phi, dphi_dn = incident_wave(points, normals, nu, depth)
    # d(phi)/dz = k0 tanh(k0 (z + h)) phi, which is nu*phi at z = 0 and 0 at z = -h.
    identity = dphi_dn - phi * k0 * np.tanh(k0 * (z + depth))
    assert np.max(np.abs(identity)) <= 1.0e-12 * np.max(np.abs(dphi_dn))
    # Free-surface condition at z = 0, approached from below: d(phi)/dz = nu*phi.
    # The condition holds only on the free surface, so the probe sits at z0 < 0
    # and the residual it leaves is the O(k0**2 * z0) offset of the probe itself.
    z0 = -1.0e-4
    probe = np.array([[0.0, z0]])
    unit = np.array([[0.0, 1.0]])
    step = 1.0e-6
    plus = np.array([[0.0, z0 + step]])
    minus = np.array([[0.0, z0 - step]])
    derivative = (incident_wave(plus, unit, nu, depth)[0] - incident_wave(minus, unit, nu, depth)[0]) / (2.0 * step)
    phi0 = incident_wave(probe, unit, nu, depth)[0]
    assert abs(derivative - nu * phi0) <= 1.0e-3 * abs(nu * phi0)
    # Bottom impermeability: the vertical decay rate k0*tanh(k0 (z + h)) vanishes at z = -h.
    bottom = np.array([[0.0, -depth + 1.0e-3]])
    phi_bottom, dphi_bottom = incident_wave(bottom, unit, nu, depth)
    assert abs(dphi_bottom[0]) <= 1.0e-3 * k0 * abs(phi_bottom[0])


def test_incident_wave_reduces_to_deep_water():
    """At k0*h = 60 the finite-depth incident wave must match the deep-water one."""
    nu = KA / RADIUS
    depth = 100.0
    k0 = finite_depth_wavenumber(nu, depth)
    assert k0 * depth > 40.0
    z = np.linspace(-3.0, -0.05, 17)
    points = np.column_stack((np.linspace(-2.0, 2.0, 17), z))
    normals = np.column_stack((np.cos(np.linspace(0, 1, 17)), np.sin(np.linspace(0, 1, 17))))
    phi, dphi_dn = incident_wave(points, normals, nu, depth)
    phi_deep, dphi_deep = deep_water.incident_airy_wave(points, normals, k0)
    scale = np.max(np.abs(phi_deep))
    assert np.max(np.abs(phi - phi_deep)) <= 1.0e-9 * scale
    assert np.max(np.abs(dphi_dn - dphi_deep)) <= 1.0e-9 * np.max(np.abs(dphi_deep))


def test_haskind_reciprocity(mesh, radiation, diffraction):
    """Radiation-only Haskind force equals the diffraction pressure integral."""
    haskind = haskind_excitation_force(mesh, radiation, KA, DEPTH)
    direct = complex(diffraction.excitation_force[1])
    assert abs(direct) > 0.0
    assert abs(haskind - direct) <= 1.0e-3 * abs(direct)


def test_haskind_reciprocity_sway(mesh):
    sway = solve_radiation_indirect(mesh, KA, DEPTH, "sway")
    solved = solve_diffraction_indirect(mesh, KA, DEPTH)
    haskind = haskind_excitation_force(mesh, sway, KA, DEPTH)
    direct = complex(solved.excitation_force[0])
    assert abs(haskind - direct) <= 1.0e-3 * abs(direct)


def test_fundamental_solution_reference(mesh, diffraction):
    reference = diffraction_reference(mesh, KA, DEPTH)
    direct = complex(diffraction.excitation_force[1])
    expected = complex(reference.excitation_force[1])
    assert abs(expected - direct) <= 5.0e-2 * abs(direct)


def test_energy_flux_balance(diffraction, mesh):
    assert abs(diffraction.energy_balance_error) <= 1.0e-3
    reference = diffraction_reference(mesh, KA, DEPTH)
    assert abs(reference.energy_balance_error) <= 1.0e-8
    # A deeply submerged cylinder reflects very little and transmits almost all.
    assert abs(diffraction.reflection_coefficient) < 0.1
    assert abs(abs(diffraction.transmission_coefficient) - 1.0) < 0.01


def test_wave_direction_symmetry(mesh, diffraction):
    """Reversing the wave direction mirrors the force for a symmetric body."""
    reversed_wave = solve_diffraction_indirect(mesh, KA, DEPTH, direction=-1)
    direct = diffraction.excitation_force
    mirrored = reversed_wave.excitation_force
    assert abs(direct[1] - mirrored[1]) <= 1.0e-8 * abs(direct[1])
    assert abs(direct[0] + mirrored[0]) <= 1.0e-8 * abs(direct[0])


def test_deep_water_collapse(mesh):
    """k0*h large: the finite-depth solve must reproduce the deep-water solver."""
    depth = 60.0
    ka = KA
    nu = ka / RADIUS
    assert finite_depth_wavenumber(nu, depth) * depth > 30.0
    finite = solve_diffraction_indirect(mesh, nu, depth)
    deep = deep_water.solve_diffraction_indirect(mesh, nu)
    finite_force = complex(finite.excitation_force[1])
    deep_force = complex(deep.excitation_force[1])
    assert abs(finite_force - deep_force) <= 2.0e-2 * abs(deep_force)


def test_blockage_changes_the_excitation(mesh):
    """Shallower water must change the exciting force by a resolved amount."""
    shallow = solve_diffraction_indirect(mesh, KA, 3.2)
    deep = solve_diffraction_indirect(mesh, KA, 20.0)
    shallow_force = abs(shallow.excitation_force[1])
    deep_force = abs(deep.excitation_force[1])
    assert abs(shallow_force - deep_force) / deep_force > 1.0e-3


def test_input_validation(mesh):
    with pytest.raises(ValueError):
        incident_wave(np.array([[0.0, 0.0]]), np.array([[0.0, 1.0]]), KA, DEPTH)
    with pytest.raises(ValueError):
        solve_diffraction_indirect(mesh, KA, 2.9)  # body would cross the sea bed
    solved = solve_radiation_indirect(mesh, KA, DEPTH, "heave")
    with pytest.raises(TypeError):
        haskind_excitation_force(mesh, solved.potential, KA, DEPTH)
    with pytest.raises(ValueError):
        solve_diffraction_indirect(mesh, KA, DEPTH, direction=2)