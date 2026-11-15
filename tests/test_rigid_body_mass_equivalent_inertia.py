"""Pins for the geometry-anchored rigid-body inertia default.

Subject: the rigid-body mass matrix's rotational block when the caller supplies
no inertia tensor (``CASE-BEM-RIGID-BODY-MASS``).  The pre-audit default
``I_cg = m*(0.4*|r_g|)^2 I3`` is a function of the distance from the coordinate
origin to the CG, i.e. not a property of the body; these tests pin the defect
and the replacement.
"""

import numpy as np
import pytest

from tensorbem.geometry3d import (
    WettedHullMesh,
    hemisphere_hull_mesh,
    rectangular_barge_hull_mesh,
    spheroid_hull_mesh,
)
from tensorbem.hydrostatic import (
    compute_hydrostatic_restoring_matrix,
    compute_mass_matrix,
    mass_matrix_from_hull_geometry,
)
from tensorbem.rigid_body_mass import (
    equivalent_half_ellipsoid,
    equivalent_inertia_from_hydrostatics,
    equivalent_inertia_from_hydrostatics as _derived,
)

MASS = 1.0e3
SEMI = (10.0, 2.0, 1.0)


def closed_form_inertia_cg(semi_axes, mass):
    """CG inertia of the lower half of a solid ellipsoid (roll, pitch, yaw)."""
    a, b, c = (float(v) for v in semi_axes)
    lever = 3.0 * c / 8.0
    return np.array(
        [
            mass * (b * b + c * c) / 5.0 - mass * lever ** 2,
            mass * (a * a + c * c) / 5.0 - mass * lever ** 2,
            mass * (a * a + b * b) / 5.0,
        ]
    )


def derived(hull, mass=MASS, **kwargs):
    props = compute_hydrostatic_restoring_matrix(hull, mass, (0.0, 0.0, -0.1))
    return equivalent_inertia_from_hydrostatics(
        props.displaced_volume,
        props.waterplane_area,
        props.waterplane_moments,
        props.waterplane_center,
        mass=mass,
        **kwargs,
    )


def test_waterplane_moments_are_central():
    """The reported waterplane moments must not move when the hull is moved.

    This is the discriminating experiment for the convention: origin-referred
    moments would differ by ``A*y_c^2 = 50.3`` and ``A*x_c^2 = 153.9`` m^4 for a
    ``(7, -4)`` m shift, central ones cannot differ at all.
    """
    base = compute_hydrostatic_restoring_matrix(hemisphere_hull_mesh(4, 1.0), MASS, (0.0, 0.0, -0.1))
    moved = compute_hydrostatic_restoring_matrix(
        hemisphere_hull_mesh(4, 1.0, center_xy=(7.0, -4.0)), MASS, (7.0, -4.0, -0.1)
    )
    assert moved.waterplane_area == pytest.approx(base.waterplane_area, rel=0.0)
    assert np.allclose(
        np.asarray(moved.waterplane_moments)[:2],
        np.asarray(base.waterplane_moments)[:2],
        rtol=1e-9,
        atol=0.0,
    )


def test_equivalent_axes_recover_the_spheroid():
    """The equivalent solid must be the hull itself on the spheroid family."""
    result = derived(spheroid_hull_mesh(SEMI, subdivisions=4))
    assert np.allclose(result["semi_axes"], np.array(SEMI), rtol=1e-2)
    assert result["waterplane_area_ratio"] == pytest.approx(1.0, abs=1e-3)


def test_derived_inertia_converges_to_the_closed_form():
    """Error must fall with panelisation and be inside 3 % at the finest level."""
    truth = closed_form_inertia_cg(SEMI, MASS)
    worst = []
    for sub in (2, 3, 4):
        got = np.diag(derived(spheroid_hull_mesh(SEMI, subdivisions=sub))["inertia_cg"])
        worst.append(float(np.max(np.abs(got / truth - 1.0))))
    assert worst[-1] < 0.03
    assert worst[-1] < worst[1] < worst[0], worst
    assert worst[-1] < 5.0e-3


def test_derived_inertia_is_invariant_under_translation():
    """Same body, origin moved: the CG-referred inertia must not change."""
    hull = spheroid_hull_mesh(SEMI, subdivisions=3)
    base = derived(hull)
    cg = np.asarray(base["centroid"], dtype=float)
    shift = np.array([1000.0, -500.0, 0.0])  # z stays 0: wetted hulls live in z <= 0
    hull_shift = WettedHullMesh(hull.vertices + shift, hull.faces)
    moved = derived(
        hull_shift,
        center_of_gravity=tuple(cg + shift),
        reference_point=tuple(shift),
    )
    scale = np.max(np.abs(np.diag(base["inertia_cg"])))
    deviation = np.max(np.abs(np.diag(moved["inertia_cg"]) - np.diag(base["inertia_cg"])))
    assert deviation / scale < 1e-6


def test_legacy_default_is_origin_dependent_and_warns():
    """Regression pin of the defect: the old fallback is not a body property."""
    with pytest.warns(RuntimeWarning, match="inertia=None"):
        near = compute_mass_matrix(MASS, (0.5, 0.0, 0.0))
    with pytest.warns(RuntimeWarning, match="inertia=None"):
        far = compute_mass_matrix(MASS, (50.5, 0.0, 0.0))
    # the same body's roll inertia would change by (50.5/0.5)^2 = 10201
    assert far[3, 3] / near[3, 3] > 1.0e3
    truth = closed_form_inertia_cg(SEMI, MASS)
    assert np.max(np.abs(np.diag(near)[3:] / truth - 1.0)) > 0.9


def test_explicit_radius_of_gyration_is_invariant_and_physical():
    """An explicit ``k`` is origin-invariant and reproduces ``I_cg = m k^2 I3``."""
    k = 2.5
    for cg in ((0.5, 0.0, 0.0), (50.5, -12.0, 3.0)):
        matrix = compute_mass_matrix(MASS, cg, radius_of_gyration=k)
        c = np.asarray(cg, dtype=float)
        s = np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]])
        about_origin = MASS * k * k * np.eye(3) + MASS * (s.T @ s)
        assert np.allclose(matrix[3:, 3:], about_origin, rtol=1e-12, atol=0.0)


def test_hull_geometry_assembly_matches_the_explicit_tensor():
    """The RAO-path assembly must agree with an explicit closed-form tensor."""
    hull = spheroid_hull_mesh(SEMI, subdivisions=4)
    cg = (0.0, 0.0, -3.0 * SEMI[2] / 8.0)
    props = compute_hydrostatic_restoring_matrix(hull, MASS, cg)
    explicit = compute_mass_matrix(
        MASS, cg, np.diag(closed_form_inertia_cg(SEMI, MASS)), inertia_about="cg"
    )
    derived_matrix = mass_matrix_from_hull_geometry(props, MASS, cg)
    scale = np.max(np.abs(explicit))
    assert np.max(np.abs(derived_matrix - explicit)) / scale < 5.0e-3


def test_rao_omitting_inertia_uses_the_wetted_mesh_tensor():
    """``inertia=None`` must reproduce the wetted-mesh tensor, not an approximation.

    The production RAO path used to take the equivalent-half-ellipsoid
    reconstruction of the waterplane moments; it now integrates the wetted mesh
    itself (``hydrostatic.mass_matrix_from_hull``).  The invariant is exact:
    passing that same tensor explicitly must reproduce the default RAO to
    round-off.  The closed-form (smooth) hemisphere tensor is then a
    *panelisation* reference, so it is compared with a declared band.
    """
    from tensorbem.hydrostatic import mass_matrix_from_wetted_hull
    from tensorbem.rao import compute_rao_at_frequency
    from tensorbem.rigid_body_mass import wetted_hull_mass_properties

    hull = spheroid_hull_mesh((1.0, 1.0, 1.0), subdivisions=3)
    mass = 1000.0
    cg = (0.0, 0.0, -3.0 / 8.0)
    props = wetted_hull_mass_properties(hull.vertices, hull.faces)
    # the body's mass is not the displaced-water mass, so the specific second
    # moments are scaled by mass / (density * V) inside the assembly
    tensor = np.asarray(props["inertia_cg"]) * (mass / float(props["mass"]))
    matrix = mass_matrix_from_wetted_hull(hull, mass, cg)
    assert np.allclose(matrix[3:, 3:] - mass * _skew_outer(cg), tensor, rtol=1e-12, atol=0.0)

    kwargs = dict(hull=hull, omega=1.2, wave_angle=0.0, mass=mass, center_of_gravity=cg)
    default = compute_rao_at_frequency(inertia=None, **kwargs)
    explicit = compute_rao_at_frequency(inertia=tensor, **kwargs)
    r_default = np.asarray(default.rao_amplitude)
    r_explicit = np.asarray(explicit.rao_amplitude)
    assert np.allclose(r_default, r_explicit, rtol=1e-9, atol=0.0)

    # closed-form smooth hemisphere: agreement at panelisation level (subdiv=3)
    closed = compute_rao_at_frequency(
        inertia=np.diag(closed_form_inertia_cg((1.0, 1.0, 1.0), mass)), **kwargs
    )
    r_closed = np.abs(np.asarray(closed.rao_amplitude))
    mask = r_closed > 1e-3
    assert np.allclose(r_default[mask], r_closed[mask], rtol=2e-2, atol=0.0)


def _skew_outer(cg):
    """``S(r)^T S(r)`` -- the parallel-axis term that the assembly adds back."""
    c = np.asarray(cg, dtype=float)
    s = np.array([[0.0, -c[2], c[1]], [c[2], 0.0, -c[0]], [-c[1], c[0], 0.0]])
    return s.T @ s


def test_barge_is_a_declared_approximation():
    """Box-like hull: the fallback is approximate and says so, with a bounded band."""
    length, beam, draft = 20.0, 8.0, 3.0
    mass = 1025.0 * length * beam * draft
    hull = rectangular_barge_hull_mesh(length, beam, draft, 10, 8)
    props = compute_hydrostatic_restoring_matrix(hull, mass, (0.0, 0.0, -draft / 2.0))
    with pytest.warns(RuntimeWarning, match="not elliptical"):
        mass_matrix_from_hull_geometry(props, mass, (0.0, 0.0, -draft / 2.0))
    result = equivalent_inertia_from_hydrostatics(
        props.displaced_volume,
        props.waterplane_area,
        props.waterplane_moments,
        props.waterplane_center,
        mass=mass,
    )
    assert result["waterplane_area_ratio"] == pytest.approx(np.pi / 3.0, rel=0.02)
    box = mass * np.array(
        [(beam ** 2 + draft ** 2) / 12.0, (length ** 2 + draft ** 2) / 12.0,
         (length ** 2 + beam ** 2) / 12.0]
    )
    error = np.diag(result["inertia_cg"]) / box - 1.0
    assert np.all(error < 0.0) and np.all(error > -0.25), error


def test_degenerate_hydrostatics_are_rejected():
    with pytest.raises(ValueError):
        equivalent_half_ellipsoid(10.0, 0.0, (1.0, 1.0, 0.0))
    with pytest.raises(ValueError):
        equivalent_half_ellipsoid(10.0, 4.0, (-1.0, 1.0, 0.0))
    with pytest.raises(ValueError):
        equivalent_half_ellipsoid(-1.0, 4.0, (1.0, 1.0, 0.0))
    with pytest.raises(ValueError):
        compute_mass_matrix(MASS, (1.0, 0.0, 0.0), radius_of_gyration=-1.0)