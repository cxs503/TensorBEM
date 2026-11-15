"""The production RAO default inertia route must be a property of the *body*.

``tensorbem.rao.compute_rao_at_frequency`` and
``tensorbem.ice_rao.compute_ice_rao_spectrum`` accept ``inertia=None``; they then
must build the rigid-body mass matrix from the hull geometry
(``tensorbem.hydrostatic.mass_matrix_from_hull``).  Two defects are pinned here:

1. the *approximate* route (equivalent half-ellipsoid fitted to the displaced
   volume and the waterplane second moments) used to be the default.  It is
   exact for an elliptic waterplane but 8-20 % low on the rotational block of a
   box-like hull -- i.e. outside the 3 % acceptance of the case-pool anchor
   ``CASE-BEM-RIGID-BODY-MASS`` -- and the RAO inherits that bias;
2. the wetted-mesh inertia was not scaled by body mass, so a body that is not
   neutrally buoyant got the second moments of the displaced *water*.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.geometry3d import rectangular_barge_hull_mesh, spheroid_hull_mesh
from tensorbem.hydrostatic import (
    compute_hydrostatic_restoring_matrix,
    compute_mass_matrix,
    mass_matrix_from_hull,
    mass_matrix_from_hull_geometry,
    mass_matrix_from_wetted_hull,
)
from tensorbem.rao import compute_rao_at_frequency

DENSITY = 1025.0
LENGTH, BEAM, DRAFT = 20.0, 8.0, 3.0
BARGE_MASS = DENSITY * LENGTH * BEAM * DRAFT
BARGE_CG = (0.0, 0.0, -DRAFT / 2.0)


def barge():
    return rectangular_barge_hull_mesh(LENGTH, BEAM, DRAFT, 10, 8)


def box_inertia_cg():
    """Solid-box second moments about the centre of gravity (closed form)."""
    return BARGE_MASS * np.array(
        [
            (BEAM * BEAM + DRAFT * DRAFT) / 12.0,
            (LENGTH * LENGTH + DRAFT * DRAFT) / 12.0,
            (LENGTH * LENGTH + BEAM * BEAM) / 12.0,
        ]
    )


def hemisphere_inertia_cg(mass, radius=1.0):
    """Lower half of a solid sphere: ``I_xx = I_yy = (2/5 - 9/64) m a^2``."""
    lever = 3.0 * radius / 8.0
    return mass * np.array(
        [
            2.0 * radius * radius / 5.0 - lever * lever,
            2.0 * radius * radius / 5.0 - lever * lever,
            2.0 * radius * radius / 5.0,
        ]
    )


def test_barge_default_matrix_is_the_closed_form_block():
    """A polyhedral hull: the mesh route must reproduce the closed form exactly."""
    matrix = mass_matrix_from_hull(barge(), BARGE_MASS, BARGE_CG)
    reference = compute_mass_matrix(
        BARGE_MASS, BARGE_CG, np.diag(box_inertia_cg()), inertia_about="cg"
    )
    scale = np.max(np.abs(reference))
    assert np.max(np.abs(matrix - reference)) / scale < 1e-12
    # and it is *not* the same as the approximation that used to be the default
    props = compute_hydrostatic_restoring_matrix(barge(), BARGE_MASS, BARGE_CG)
    with pytest.warns(RuntimeWarning, match="not elliptical"):
        legacy = mass_matrix_from_hull_geometry(props, BARGE_MASS, BARGE_CG)
    assert np.max(np.abs(legacy - reference)) / scale > 0.05


def test_barge_rao_default_reproduces_the_closed_form_inertia():
    """``inertia=None`` and the exact box tensor must give the same RAO."""
    hull = barge()
    kwargs = dict(hull=hull, omega=1.0, wave_angle=0.0, mass=BARGE_MASS,
                  center_of_gravity=BARGE_CG)
    default = np.asarray(compute_rao_at_frequency(inertia=None, **kwargs).rao_amplitude)
    explicit = np.asarray(
        compute_rao_at_frequency(inertia=np.diag(box_inertia_cg()), **kwargs).rao_amplitude
    )
    assert np.allclose(default, explicit, rtol=1e-10, atol=0.0)


def test_legacy_route_bias_exceeds_the_case_pool_tolerance():
    """Negative control: the old default is outside 3 % on every rotational entry."""
    props = compute_hydrostatic_restoring_matrix(barge(), BARGE_MASS, BARGE_CG)
    with pytest.warns(RuntimeWarning, match="not elliptical"):
        legacy = mass_matrix_from_hull_geometry(props, BARGE_MASS, BARGE_CG)
    exact = compute_mass_matrix(
        BARGE_MASS, BARGE_CG, np.diag(box_inertia_cg()), inertia_about="cg"
    )
    error = np.diag(legacy)[3:] / np.diag(exact)[3:] - 1.0
    assert np.all(error < -0.03), error            # low, and by more than the tolerance
    # ... and the bias is visible in the response, not only in the matrix
    hull = barge()
    kwargs = dict(hull=hull, omega=1.0, wave_angle=0.0, mass=BARGE_MASS,
                  center_of_gravity=BARGE_CG)
    r_legacy = np.asarray(
        compute_rao_at_frequency(inertia=_inertia_from_matrix(legacy, BARGE_CG),
                                 **kwargs).rao_amplitude
    )
    r_exact = np.asarray(
        compute_rao_at_frequency(inertia=np.diag(box_inertia_cg()), **kwargs).rao_amplitude
    )
    pitch = abs(r_legacy[4] - r_exact[4]) / abs(r_exact[4])
    assert pitch > 0.03, pitch


def _inertia_from_matrix(matrix, cg):
    """Invert the parallel-axis term to recover ``I_g`` from the 6x6 block."""
    c = np.asarray(cg, dtype=float)
    s = np.array([[0.0, -c[2], c[1]], [c[2], 0.0, -c[0]], [-c[1], c[0], 0.0]])
    return matrix[3:, 3:] - BARGE_MASS * (s.T @ s)


def test_mesh_route_converges_to_the_closed_hemisphere():
    """Curved hull: second-order convergence to the analytic half-sphere inertia."""
    mass = 1000.0
    cg = (0.0, 0.0, -3.0 / 8.0)
    reference = compute_mass_matrix(
        mass, cg, np.diag(hemisphere_inertia_cg(mass)), inertia_about="cg"
    )
    scale = np.max(np.abs(reference))
    errors = []
    for subdivisions in (2, 3, 4):
        hull = spheroid_hull_mesh((1.0, 1.0, 1.0), subdivisions=subdivisions)
        matrix = mass_matrix_from_hull(hull, mass, cg)
        errors.append(float(np.max(np.abs(matrix - reference)) / scale))
    assert errors[0] > errors[1] > errors[2]
    slope = math.log(errors[0] / errors[2]) / math.log(4.0)
    assert slope > 1.8, (errors, slope)
    assert errors[2] < 5e-3, errors


def test_wetted_hull_inertia_scales_with_body_mass():
    """The rotational block is a property of the body, not of the displaced water."""
    hull = spheroid_hull_mesh((1.0, 1.0, 1.0), subdivisions=2)
    cg = (0.0, 0.0, -3.0 / 8.0)
    light = mass_matrix_from_wetted_hull(hull, 1000.0, cg)
    heavy = mass_matrix_from_wetted_hull(hull, 2000.0, cg)
    assert np.allclose(heavy[3:, 3:], 2.0 * light[3:, 3:], rtol=1e-12, atol=0.0)
    assert np.allclose(heavy[:3, :3], 2.0 * light[:3, :3], rtol=1e-12, atol=0.0)
    # scaling matters: the mesh (fluid) mass is not the body mass here
    assert abs(2000.0 / (DENSITY * 2.0943951) - 1.0) > 0.05


def test_unusable_mesh_falls_back_with_a_warning(monkeypatch):
    """If the panels cannot be integrated the approximate route runs -- loudly."""
    hull = barge()
    props = compute_hydrostatic_restoring_matrix(hull, BARGE_MASS, BARGE_CG)

    def boom(*args, **kwargs):
        raise ValueError("open hull: topology guard")

    monkeypatch.setattr("tensorbem.hydrostatic.mass_matrix_from_wetted_hull", boom)
    with pytest.warns(RuntimeWarning, match="mesh route is unusable"):
        with pytest.warns(RuntimeWarning, match="not elliptical"):
            matrix = mass_matrix_from_hull(
                hull, BARGE_MASS, BARGE_CG, hydrostatic_props=props
            )
    fallback = mass_matrix_from_hull_geometry(props, BARGE_MASS, BARGE_CG)
    assert np.allclose(matrix, fallback, rtol=1e-12, atol=0.0)


def test_forced_mesh_source_does_not_silently_fall_back(monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("open hull")

    monkeypatch.setattr("tensorbem.hydrostatic.mass_matrix_from_wetted_hull", boom)
    with pytest.raises(ValueError, match="open hull"):
        mass_matrix_from_hull(barge(), BARGE_MASS, BARGE_CG, source="mesh")
    with pytest.raises(ValueError, match="source must be"):
        mass_matrix_from_hull(barge(), BARGE_MASS, BARGE_CG, source="guess")


def test_ice_rao_default_matches_an_explicit_mesh_tensor():
    """The ice chain shares the route: same default, same parity statement."""
    from tensorbem.ice import IceCover
    from tensorbem.ice_rao import compute_ice_rao_spectrum
    from tensorbem.rigid_body_mass import wetted_hull_mass_properties

    hull = spheroid_hull_mesh((1.0, 1.0, 1.0), subdivisions=2)
    mass = 2000.0
    cg = (0.0, 0.0, -3.0 / 8.0)
    props = wetted_hull_mass_properties(hull.vertices, hull.faces)
    tensor = np.asarray(props["inertia_cg"]) * (mass / float(props["mass"]))
    ice = IceCover()
    common = dict(wave_angle=math.pi, mass=mass, center_of_gravity=cg, ice=ice)
    omega = [1.0]
    default = compute_ice_rao_spectrum(hull, omega, inertia=None, **common)
    explicit = compute_ice_rao_spectrum(hull, omega, inertia=tensor, **common)
    assert np.allclose(default.rao_ice, explicit.rao_ice, rtol=1e-10, atol=1e-12)
