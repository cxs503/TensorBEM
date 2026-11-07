"""Verification of the ice-covered 3-D radiation/diffraction/RAO chain.

The checks are structural rather than reference-value based, because no
published reference exists in this repository for a hull in a mass-loading ice
field:

* the deep-water ice wavenumber must equal the Robin coefficient ``K`` and the
  finite-depth solver in the deep-water limit;
* the ``mu -> 0`` limit must reproduce the open-water added mass, damping and
  excitation force (catches bookkeeping errors in the damping rescaling);
* the excitation force must be invariant to the solver's internal
  ``omega = sqrt(g nu)`` bookkeeping (a gravity rescaling of the kernel must
  scale the force by exactly that factor, because the incident potential
  carries the compensating ``1/omega``);
* the validity envelope (deep water, propagating branch) and the input
  validation must reject rather than fabricate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.free_surface_solver3d import assemble_free_surface_operators_3d, solve_diffraction_3d
from tensorbem.hydrostatic import compute_hydrostatic_restoring_matrix
from tensorbem.ice import IceCover, mass_loading_wavenumber, robin_coefficient
from tensorbem.ice_rao import (
    compute_ice_rao_spectrum,
    ice_wavenumber_deep_water,
    no_ice_limit_check,
    solve_diffraction_3d_ice,
    solve_radiation_3d_ice,
    solve_radiation_3d_open_water,
)
from tensorbem.pc6_hull import IceClassHull, hull_mass_properties, ice_class_hull_mesh, validate_hull_geometry

ICE = IceCover(thickness=0.9, water_depth=200.0)


def _small_hull():
    hull = IceClassHull(length=120.0, breadth=24.0, draft=9.0)
    mesh = ice_class_hull_mesh(hull, n_sections=13, n_side=4, n_bottom=3)
    check = validate_hull_geometry(hull, mesh, cb_tolerance=0.02)
    return hull, mesh, check


def test_deep_water_ice_wavenumber_identity():
    """``k_i = omega^2/(g - mu omega^2)`` and consistency with the depth solver."""
    for period in (4.0, 6.0, 10.0):
        omega = 2.0 * math.pi / period
        k_deep = ice_wavenumber_deep_water(omega, ICE)
        assert k_deep == pytest.approx(robin_coefficient(omega, ICE), rel=1e-14)
        k_general = mass_loading_wavenumber(omega, ICE).wavenumber
        assert k_deep == pytest.approx(k_general, rel=1e-3)
        assert k_deep > omega * omega / ICE.gravity  # ice shortens the waves


def test_cutoff_frequency_rejected():
    with pytest.raises(ValueError, match="cut-off"):
        ice_wavenumber_deep_water(ICE.cutoff_omega * 1.01, ICE)


def test_shallow_water_envelope_rejected():
    """The deep-water ice kernel is withheld outside ``k H >= 3``."""
    shallow = IceCover(thickness=0.9, water_depth=10.0)
    mesh = _small_hull()[1]
    with pytest.raises(ValueError, match="validity envelope"):
        solve_radiation_3d_ice(mesh, 2.0 * math.pi / 10.0, shallow)


def test_negative_wave_amplitude_rejected():
    mesh = _small_hull()[1]
    with pytest.raises(ValueError, match="wave amplitude"):
        solve_diffraction_3d_ice(mesh, 1.0, ICE, amplitude=-1.0)


def test_zero_and_negative_frequency_rejected():
    mesh = _small_hull()[1]
    with pytest.raises(ValueError, match="omega"):
        solve_radiation_3d_ice(mesh, 0.0, ICE)
    with pytest.raises(ValueError, match="omega"):
        solve_radiation_3d_ice(mesh, -0.5, ICE)


def test_excitation_force_is_bookkeeping_invariant():
    """A gravity rescaling of the kernel must scale the force by the same factor."""
    _, mesh, _ = _small_hull()
    omega = 2.0 * math.pi / 8.0
    k_ice = ice_wavenumber_deep_water(omega, ICE)
    operators = assemble_free_surface_operators_3d(mesh, k_ice)
    g = ICE.gravity
    f1 = solve_diffraction_3d(mesh, k_ice, amplitude=1.0, operators=operators, gravity=g)
    f2 = solve_diffraction_3d(
        mesh, k_ice, amplitude=1.0, operators=operators, gravity=1.5 * g
    )
    ratio = f2.excitation_force / f1.excitation_force
    assert np.allclose(ratio.real, 1.5, rtol=1e-9, atol=1e-9)
    assert np.allclose(ratio.imag, 0.0, atol=1e-9)


def test_no_ice_limit_reproduces_open_water():
    """``mu -> 0`` must give back the open-water radiation problem."""
    _, mesh, _ = _small_hull()
    check = no_ice_limit_check(
        mesh, 2.0 * math.pi / 8.0, water_depth=1.0e4, chief_points="auto"
    )
    assert check["wavenumber_relative_difference"] < 1.0e-8
    assert check["added_mass_relative_difference"] < 1.0e-6
    assert check["damping_relative_difference"] < 1.0e-4


def test_radiation_damping_in_ice_is_physical():
    """Added mass and damping in ice must be finite and damping non-negative."""
    _, mesh, check = _small_hull()
    cg = np.array([0.0, 0.0, -0.5 * 9.0])
    result = solve_radiation_3d_ice(mesh, 2.0 * math.pi / 8.0, ICE, center=cg)
    assert np.all(np.isfinite(result.added_mass))
    # the added-mass matrix is symmetric only up to the discretisation error of
    # the collocation solve (measured 1.1e-3 relative on this mesh)
    assert np.allclose(
        result.added_mass, result.added_mass.T, atol=5e-3 * np.abs(result.added_mass).max()
    )
    assert result.damping[2, 2] >= 0.0
    assert result.depth_parameter >= 3.0
    assert result.relative_residual < 1.0e-6


def test_rao_spectrum_evanescent_frequency_is_flagged_not_fabricated():
    """Above cut-off the spectrum must flag the frequency and report no response."""
    _, mesh, check = _small_hull()
    props = hull_mass_properties(IceClassHull(), check)
    spectrum = compute_ice_rao_spectrum(
        mesh,
        [ICE.cutoff_omega * 1.2],
        wave_angle=math.pi,
        mass=props["mass"],
        center_of_gravity=props["center_of_gravity"],
        ice=ICE,
        inertia=props["inertia_about_origin"],
    )
    assert spectrum.propagating.tolist() == [False]
    assert np.all(spectrum.rao_ice == 0.0)
    assert np.all(np.isnan(spectrum.added_mass_ice))
    assert spectrum.decay_rate[0] > 0.0


def test_rao_spectrum_satisfies_the_full_impedance_balance():
    """``Z xi = F_exc`` must hold on the whole 6x6 coupled system, not the heave row.

    The six-DOF response is coupled (heave does not decouple from pitch), so the
    check is the residual of the full impedance equation, which is what the
    solver actually inverted.
    """
    from tensorbem.hydrostatic import compute_mass_matrix

    _, mesh, check = _small_hull()
    props = hull_mass_properties(IceClassHull(), check)
    omega = 2.0 * math.pi / 8.0
    spectrum = compute_ice_rao_spectrum(
        mesh,
        [omega],
        wave_angle=math.pi,
        mass=props["mass"],
        center_of_gravity=props["center_of_gravity"],
        ice=ICE,
        inertia=props["inertia_about_origin"],
    )
    assert spectrum.propagating.tolist() == [True]
    mass_matrix = compute_mass_matrix(
        props["mass"], props["center_of_gravity"], props["inertia_about_origin"]
    )
    impedance = (
        -(omega ** 2) * (mass_matrix + spectrum.added_mass_ice[0])
        + 1j * omega * spectrum.damping_ice[0]
        + spectrum.restoring_matrix
    )
    residual = impedance @ spectrum.rao_ice[0] - spectrum.excitation_ice[0]
    reference = np.linalg.norm(spectrum.excitation_ice[0])
    assert np.linalg.norm(residual) <= 1e-9 * reference
    assert spectrum.restoring_matrix[2, 2] > 0.0
    assert abs(spectrum.rao_ice[0, 2]) > 0.0
    assert abs(spectrum.rao_ice[0, 4]) > 0.0  # pitch is driven in head seas


def test_added_mass_matrix_report_keys():
    """The reporting helper must expose all 36 entries with mode names."""
    from tensorbem.ice_rao import added_mass_matrix_report

    matrix = np.arange(36, dtype=float).reshape(6, 6)
    report = added_mass_matrix_report(matrix)
    assert len(report) == 36
    assert report["A33_heave_heave"] == 14.0
    assert report["A11_surge_surge"] == 0.0
    with pytest.raises(ValueError, match="6x6"):
        added_mass_matrix_report(np.zeros((5, 5)))
