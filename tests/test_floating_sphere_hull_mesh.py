"""Verification of :func:`tensorbem.geometry3d.floating_sphere_hull_mesh`.

The builder was merged (PR #183) without any test of its own, so nothing in the
suite pinned the properties a surface-piercing wetted mesh has to have:

* every vertex stays below the calm waterplane and inside the analytic sphere,
* the open boundary is the waterline circle ``sqrt(R**2 - z0**2)`` (inscribed
  polygon: the faceted panels cut the circle on chords, so the waterline radius
  approaches the analytic value from below as the panels shrink),
* the enclosed volume converges to the analytic spherical-cap value
  ``4/3*pi*R**3 - pi*h**2*(3R - h)/3`` with ``h = R + z0`` the cap above water,
* the wetted area converges to the spherical zone ``2*pi*R*(R - z0)``,
* the panel normals point out of the body (into the fluid).

All references below are closed-form, so the checks compare against exact values
rather than against a previously stored number.  For ``R = 1``, ``z0 = -0.5``
the exact answers are ``V = 3.5342917352885173`` and ``A = 9.42477796076938``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.geometry3d import floating_sphere_hull_mesh

RADIUS = 1.0
CENTER_Z = -0.5  # sphere pierces the waterplane, 75 % of its height submerged


def _analytic_submerged_volume(radius: float, z0: float) -> float:
    """Volume of the sphere that lies below ``z = 0`` (a spherical-cap cut)."""
    height_above = radius + z0  # z0 < 0 -> cap above the waterplane
    cap_above = math.pi * height_above**2 * (3.0 * radius - height_above) / 3.0
    return 4.0 / 3.0 * math.pi * radius**3 - cap_above


def _analytic_waterline_radius(radius: float, z0: float) -> float:
    return math.sqrt(radius**2 - z0**2)


def _analytic_wetted_area(radius: float, z0: float) -> float:
    """Area of the spherical zone below the waterplane: ``2*pi*R*(R - z0)``."""
    return 2.0 * math.pi * radius * (radius - z0)


def _boundary_vertices(mesh) -> np.ndarray:
    """Boundary half-edges of the open wetted surface (each used once)."""
    return np.unique(mesh.waterline_edges)


def test_vertices_stay_below_the_waterplane_and_inside_the_sphere() -> None:
    mesh = floating_sphere_hull_mesh(2, radius=RADIUS, center=(0.0, 0.0, CENTER_Z))

    assert np.all(mesh.vertices[:, 2] <= 0.0)
    # The facets are inscribed in the sphere: no vertex may leave it, and the
    # excursion inwards is bounded by the panel sagitta (~1.5 % at level 2).
    radii = np.linalg.norm(mesh.vertices - np.array([0.0, 0.0, CENTER_Z]), axis=1)
    assert np.all(radii <= RADIUS + 1.0e-12)
    assert float(radii.min()) >= 0.98 * RADIUS


def test_waterline_is_an_inscribed_polygon_of_the_analytic_circle() -> None:
    mesh = floating_sphere_hull_mesh(3, radius=RADIUS, center=(0.0, 0.0, CENTER_Z))

    boundary = _boundary_vertices(mesh)
    on_waterline = np.flatnonzero(np.abs(mesh.vertices[:, 2]) <= 1.0e-12)
    # The clipping boundary is exactly the set of waterplane vertices.
    assert set(boundary.tolist()) == set(on_waterline.tolist())
    assert len(boundary) >= 16

    exact = _analytic_waterline_radius(RADIUS, CENTER_Z)  # sqrt(0.75)
    rho = np.linalg.norm(mesh.vertices[boundary, :2], axis=1)
    # Inscribed: never outside the true circle, and within 1 % of it at level 3
    # (0.8622 <= rho <= 0.86603 for the measured level-3 mesh).
    assert np.all(rho <= exact + 1.0e-12)
    assert float(rho.min()) >= 0.99 * exact


def test_waterline_radius_converges_to_the_analytic_circle() -> None:
    exact = _analytic_waterline_radius(RADIUS, CENTER_Z)
    deficits = []
    for subdivisions in (1, 2, 3):
        mesh = floating_sphere_hull_mesh(
            subdivisions, radius=RADIUS, center=(0.0, 0.0, CENTER_Z)
        )
        rho = np.linalg.norm(mesh.vertices[_boundary_vertices(mesh), :2], axis=1)
        # The mean radius of the inscribed polygon is the fair measure: the
        # maximum is the tangent vertex, which sits on the circle exactly.
        deficits.append(1.0 - float(rho.mean()) / exact)
    assert all(later < former for former, later in zip(deficits, deficits[1:])), deficits
    assert deficits[-1] < 5.0e-3, deficits


def test_a_translated_centre_moves_the_waterline_with_it() -> None:
    center = (0.35, -0.2, -0.4)
    mesh = floating_sphere_hull_mesh(2, radius=RADIUS, center=center)

    boundary = _boundary_vertices(mesh)
    rho = np.linalg.norm(mesh.vertices[boundary, :2] - np.array(center[:2]), axis=1)
    exact = _analytic_waterline_radius(RADIUS, center[2])
    assert np.all(rho <= exact + 1.0e-12)
    assert float(rho.min()) >= 0.98 * exact
    # The cap follows the centre: nothing is left at the origin.
    centred = np.linalg.norm(mesh.vertices[:, :2] - np.array(center[:2]), axis=1)
    assert float(centred.max()) <= 1.0 + 1.0e-12


def test_panel_normals_point_out_of_the_body() -> None:
    center = np.array([0.0, 0.0, CENTER_Z])
    mesh = floating_sphere_hull_mesh(2, radius=RADIUS, center=tuple(center))

    v0 = mesh.vertices[mesh.faces[:, 0]]
    v1 = mesh.vertices[mesh.faces[:, 1]]
    v2 = mesh.vertices[mesh.faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    centroids = (v0 + v1 + v2) / 3.0
    # Outward normals: positive component along the radius; a reversed winding
    # would flip the sign and make the sum negative.
    dots = np.einsum("ij,ij->i", normals, centroids - center)
    assert np.all(dots > 0.0), f"{int(np.sum(dots <= 0.0))} panels face inwards"


@pytest.mark.parametrize("subdivisions", [1, 2, 3, 4])
def test_displaced_volume_converges_to_the_spherical_cap(subdivisions: int) -> None:
    mesh = floating_sphere_hull_mesh(
        subdivisions, radius=RADIUS, center=(0.0, 0.0, CENTER_Z)
    )

    exact = _analytic_submerged_volume(RADIUS, CENTER_Z)  # 3.5342917352885173
    error = abs(mesh.volume - exact) / exact
    # A facet-inscribed polyhedron under-estimates the volume and converges at
    # roughly first order: 11.3 %, 3.0 %, 0.77 %, 0.19 % for levels 1-4.
    expected_error = {1: 0.12, 2: 0.035, 3: 0.009, 4: 0.003}[subdivisions]
    assert error < expected_error, (subdivisions, error, exact)


def test_volume_error_decreases_with_refinement() -> None:
    exact = _analytic_submerged_volume(RADIUS, CENTER_Z)
    errors = []
    for subdivisions in (1, 2, 3, 4):
        mesh = floating_sphere_hull_mesh(
            subdivisions, radius=RADIUS, center=(0.0, 0.0, CENTER_Z)
        )
        errors.append(abs(mesh.volume - exact) / exact)
    assert all(later < former for former, later in zip(errors, errors[1:])), errors
    assert errors[-1] < 0.005, errors


def test_wetted_area_converges_to_the_spherical_zone() -> None:
    exact = _analytic_wetted_area(RADIUS, CENTER_Z)  # 9.42477796076938
    errors = []
    for subdivisions in (1, 2, 3):
        mesh = floating_sphere_hull_mesh(
            subdivisions, radius=RADIUS, center=(0.0, 0.0, CENTER_Z)
        )
        errors.append(abs(mesh.surface_area - exact) / exact)
    assert all(later < former for former, later in zip(errors, errors[1:])), errors
    assert errors[-1] < 0.01, errors


def test_displacement_coefficient_matches_the_analytic_cap() -> None:
    """Cb = V / (4/3 pi R^3); z0 = -0.5 gives 0.84375 exactly."""
    mesh = floating_sphere_hull_mesh(4, radius=RADIUS, center=(0.0, 0.0, CENTER_Z))
    sphere = 4.0 / 3.0 * math.pi * RADIUS**3
    exact_ratio = _analytic_submerged_volume(RADIUS, CENTER_Z) / sphere
    assert abs(exact_ratio - 0.84375) < 1.0e-15
    assert abs(mesh.volume / sphere - exact_ratio) / exact_ratio < 0.005


@pytest.mark.parametrize("subdivisions", [True, False, -1, 1.5, "2", None])
def test_subdivisions_must_be_a_non_negative_integer(subdivisions) -> None:
    with pytest.raises(ValueError):
        floating_sphere_hull_mesh(subdivisions)  # type: ignore[arg-type]


@pytest.mark.parametrize("radius", [0.0, -1.0, float("nan"), float("inf")])
def test_radius_must_be_positive_and_finite(radius: float) -> None:
    with pytest.raises(ValueError):
        floating_sphere_hull_mesh(1, radius=radius)


@pytest.mark.parametrize(
    "center",
    [
        (0.0, 0.0, 0.0),  # tangent to the waterplane, no waterline
        (0.0, 0.0, -1.0),  # fully submerged, no waterline
        (0.0, 0.0, 0.5),  # above the waterplane
        (0.0, 0.0),
        (0.0, 0.0, float("nan")),
    ],
)
def test_center_must_pierce_the_waterplane(center) -> None:
    with pytest.raises(ValueError):
        floating_sphere_hull_mesh(1, radius=RADIUS, center=center)