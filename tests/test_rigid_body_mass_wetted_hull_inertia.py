"""Tests: geometry-exact rigid-body inertia from a wetted hull mesh.

Anchor: ``CASE-BEM-RIGID-BODY-MASS``.  Covers the closure of a ``WettedHullMesh``
with the calm-waterplane lid (:func:`tensorbem.rigid_body_mass.wetted_hull_solid`),
the mass properties derived from it
(:func:`tensorbem.rigid_body_mass.wetted_hull_mass_properties`) and the assembled
mass matrix entry
(:func:`tensorbem.hydrostatic.mass_matrix_from_wetted_hull`), against closed-form
and independent references.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.benchmarks.rigid_body_mass_hull_inertia import (
    box_reference,
    half_ellipsoid_reference,
    polygon_prism_reference,
    run,
)
from tensorbem.geometry3d import (
    elliptic_cylinder_hull_mesh,
    rectangular_barge_hull_mesh,
    spheroid_hull_mesh,
)
from tensorbem.hydrostatic import compute_mass_matrix, mass_matrix_from_wetted_hull
from tensorbem.rigid_body_mass import (
    mesh_topology_report,
    wetted_hull_mass_properties,
    wetted_hull_solid,
)

DENSITY = 1025.0


def _barge():
    return rectangular_barge_hull_mesh(20.0, 8.0, 3.0)


def _spheroid(subdivisions: int = 4):
    return spheroid_hull_mesh((60.0, 12.0, 8.0), subdivisions=subdivisions)


def _cylinder(panels: int = 32):
    return elliptic_cylinder_hull_mesh(15.0, 5.0, 2.5, panels=panels)


class TestWaterplaneClosure:
    def test_barge_closes_into_a_watertight_solid(self) -> None:
        hull = _barge()
        solid = wetted_hull_solid(hull.vertices, hull.faces)
        assert solid["topology"]["is_closed"] is True
        assert solid["topology"]["open_edges"] == 0
        assert solid["topology"]["nonmanifold_edges"] == 0
        assert solid["topology"]["orientation_mismatches"] == 0
        assert solid["waterline_loops"] == 1
        # The barge waterplane is a rectangle: four boundary half-edges.
        assert solid["lid_panels"] == 4
        assert solid["wetted_faces"] == int(np.asarray(hull.faces).shape[0])

    def test_lid_does_not_change_the_displaced_volume(self) -> None:
        for hull in (_barge(), _spheroid(3), _cylinder(16)):
            solid = wetted_hull_solid(hull.vertices, hull.faces)
            assert solid["volume_mismatch"] < 1.0e-12
            assert solid["solid_volume"] == pytest.approx(solid["wetted_volume"], rel=1.0e-12)
            assert solid["solid_volume"] > 0.0

    def test_barge_volume_and_waterplane_area_are_exact(self) -> None:
        hull = _barge()
        solid = wetted_hull_solid(hull.vertices, hull.faces)
        assert solid["solid_volume"] == pytest.approx(20.0 * 8.0 * 3.0, rel=1.0e-14)
        props = wetted_hull_mass_properties(hull.vertices, hull.faces, DENSITY)
        assert props["waterplane_lid_area"] == pytest.approx(20.0 * 8.0, rel=1.0e-14)

    def test_curved_hull_closes_with_a_fanned_lid(self) -> None:
        hull = _spheroid(2)
        solid = wetted_hull_solid(hull.vertices, hull.faces)
        assert solid["topology"]["is_closed"] is True
        assert solid["lid_panels"] == int(np.asarray(hull.waterline_edges).shape[0])
        assert solid["waterline_loops"] == 1

    def test_rejects_an_already_closed_surface(self) -> None:
        """Closing a closed solid would add a second lid, so it is refused."""
        hull = _barge()
        solid = wetted_hull_solid(hull.vertices, hull.faces)
        with pytest.raises(ValueError, match="no open waterline boundary"):
            wetted_hull_solid(solid["vertices"], solid["faces"])

    def test_rejects_a_boundary_off_the_waterplane(self) -> None:
        hull = _barge()
        vertices = np.array(hull.vertices, dtype=float)
        waterline = np.where(np.abs(vertices[:, 2]) <= 1.0e-12)[0]
        assert waterline.size > 0
        vertices[waterline[0], 2] = -0.05
        with pytest.raises(ValueError, match="must lie on the calm waterplane"):
            wetted_hull_solid(vertices, hull.faces)

    def test_rejects_a_flipped_panel(self) -> None:
        """The pre-guard quadrature silently integrated corrupted meshes."""
        hull = _barge()
        faces = np.array(hull.faces, dtype=np.int64)
        faces[0] = faces[0][::-1]
        with pytest.raises(ValueError):
            wetted_hull_solid(hull.vertices, faces)

    def test_rejects_degenerate_input(self) -> None:
        hull = _barge()
        with pytest.raises(ValueError, match="at least one panel"):
            wetted_hull_solid(hull.vertices, np.empty((0, 3), dtype=np.int64))
        with pytest.raises(ValueError, match="out of range"):
            wetted_hull_solid(hull.vertices, np.array([[0, 1, 10_000]], dtype=np.int64))


class TestGeometryInertia:
    def test_box_matches_the_cuboid_formula_to_round_off(self) -> None:
        hull = _barge()
        props = wetted_hull_mass_properties(hull.vertices, hull.faces, DENSITY)
        reference = box_reference(20.0, 8.0, 3.0, DENSITY)
        got = np.diag(np.asarray(props["inertia_cg"], dtype=float))
        assert got == pytest.approx(np.asarray(reference["inertia_cg"]), rel=1.0e-12)
        assert np.asarray(props["centroid"]) == pytest.approx(
            np.asarray(reference["centroid"]), abs=1.0e-12
        )
        assert float(props["mass"]) == pytest.approx(reference["mass"], rel=1.0e-14)

    def test_half_ellipsoid_within_tolerance_at_the_accepted_level(self) -> None:
        hull = _spheroid(4)
        props = wetted_hull_mass_properties(hull.vertices, hull.faces, DENSITY)
        reference = half_ellipsoid_reference(60.0, 12.0, 8.0, DENSITY)
        got = np.diag(np.asarray(props["inertia_cg"], dtype=float))
        rel = np.abs(got - np.asarray(reference["inertia_cg"])) / np.asarray(
            reference["inertia_cg"]
        )
        assert float(np.max(rel)) < 0.03

    def test_half_ellipsoid_error_converges_quadratically(self) -> None:
        reference = np.asarray(
            half_ellipsoid_reference(60.0, 12.0, 8.0, DENSITY)["inertia_cg"]
        )
        errors = []
        for subdivisions in (2, 3, 4, 5):
            hull = _spheroid(subdivisions)
            props = wetted_hull_mass_properties(hull.vertices, hull.faces, DENSITY)
            got = np.diag(np.asarray(props["inertia_cg"], dtype=float))
            errors.append(float(np.max(np.abs(got - reference) / reference)))
        assert all(b < a for a, b in zip(errors, errors[1:]))
        slope = float(np.polyfit(np.log([2.0**-s for s in (2, 3, 4, 5)]), np.log(errors), 1)[0])
        assert 1.8 < slope < 2.2

    def test_prismatic_hull_matches_independent_2d_polygon_route(self) -> None:
        hull = _cylinder(32)
        props = wetted_hull_mass_properties(hull.vertices, hull.faces, DENSITY)
        reference = polygon_prism_reference(15.0, 5.0, 2.5, 32, DENSITY)
        got = np.diag(np.asarray(props["inertia_cg"], dtype=float))
        scale = np.max(np.abs(reference["inertia_cg"]))
        assert np.max(np.abs(got - np.asarray(reference["inertia_cg"]))) / scale < 1.0e-12

    def test_inertia_is_a_body_property_not_a_frame_property(self) -> None:
        hull = _barge()
        base = wetted_hull_mass_properties(hull.vertices, hull.faces, DENSITY)
        shift = np.array([7.0, -4.0, 0.0])
        moved = wetted_hull_mass_properties(
            np.asarray(hull.vertices, dtype=float) + shift, hull.faces, DENSITY
        )
        reference = np.asarray(base["inertia_cg"], dtype=float)
        assert np.asarray(moved["inertia_cg"]) == pytest.approx(reference, rel=1.0e-12)
        assert np.asarray(moved["centroid"]) - shift == pytest.approx(
            np.asarray(base["centroid"]), abs=1.0e-12
        )

    def test_closed_solid_is_watertight_by_the_shipped_guard(self) -> None:
        hull = _spheroid(3)
        solid = wetted_hull_solid(hull.vertices, hull.faces)
        report = mesh_topology_report(solid["vertices"], solid["faces"])
        assert report["is_closed"] is True
        assert report["degenerate_faces"] == 0


class TestMassMatrixEntry:
    def test_defaults_are_buoyancy_and_the_volume_centroid(self) -> None:
        hull = _barge()
        props = wetted_hull_mass_properties(hull.vertices, hull.faces, DENSITY)
        matrix = mass_matrix_from_wetted_hull(hull, density=DENSITY)
        mass = DENSITY * 20.0 * 8.0 * 3.0
        assert np.asarray(matrix[:3, :3]) == pytest.approx(mass * np.eye(3), rel=1.0e-12)
        assert float(props["mass"]) == pytest.approx(mass, rel=1.0e-14)
        assert np.min(np.linalg.eigvalsh(matrix)) > 0.0
        assert np.allclose(matrix, matrix.T, atol=1.0e-9 * np.max(np.abs(matrix)))

    def test_explicit_inertia_passes_through(self) -> None:
        hull = _barge()
        inertia = np.diag([1.0e6, 2.0e6, 3.0e6])
        matrix = mass_matrix_from_wetted_hull(
            hull, mass=1.0e6, center_of_gravity=(0.0, 0.0, -1.0), inertia=inertia, density=DENSITY
        )
        expected = compute_mass_matrix(1.0e6, (0.0, 0.0, -1.0), inertia, inertia_about="cg")
        assert np.asarray(matrix) == pytest.approx(np.asarray(expected), rel=1.0e-14)

    def test_kinetic_energy_identity(self) -> None:
        hull = _cylinder(32)
        props = wetted_hull_mass_properties(hull.vertices, hull.faces, DENSITY)
        matrix = mass_matrix_from_wetted_hull(hull, density=DENSITY)
        mass = float(props["mass"])
        cg = np.asarray(props["centroid"], dtype=float)
        inertia = np.asarray(props["inertia_cg"], dtype=float)
        rng = np.random.default_rng(7)
        for _ in range(8):
            v, w = rng.normal(size=3), rng.normal(size=3)
            twist = np.concatenate((v, w))
            lhs = 0.5 * float(twist @ matrix @ twist)
            v_cg = v + np.cross(w, cg)
            rhs = 0.5 * mass * float(v_cg @ v_cg) + 0.5 * float(w @ inertia @ w)
            assert abs(lhs - rhs) / abs(rhs) < 1.0e-12

    def test_rejects_a_non_positive_mass(self) -> None:
        hull = _barge()
        with pytest.raises(ValueError, match="positive finite"):
            mass_matrix_from_wetted_hull(hull, mass=0.0, density=DENSITY)


class TestBenchmarkVerdict:
    def test_all_checks_pass(self) -> None:
        payload = run(density=DENSITY)
        failing = [name for name, value in payload["checks"].items() if not value]
        assert failing == []
        assert payload["verdict"] == "PASS"
        # The improvement must be visible in the payload, not just asserted in prose.
        assert payload["improvement"]["box_previous_route_worst_rel"] > payload["tolerance"]
        assert payload["improvement"]["box_geometry_route_worst_rel"] < 1.0e-12
        accepted = payload["accepted"]
        for entry in accepted.values():
            assert entry["worst_rel"] < payload["tolerance"]

    def test_reported_numbers_are_finite_and_json_safe(self) -> None:
        import json

        payload = run(density=DENSITY)
        text = json.dumps(payload, allow_nan=False)
        assert "Infinity" not in text


def test_half_ellipsoid_reference_is_physically_consistent() -> None:
    """The reference itself: parallel-axis transfer to the centroid must hold."""
    a, b, c, density = 60.0, 12.0, 8.0, DENSITY
    reference = half_ellipsoid_reference(a, b, c, density)
    mass = reference["mass"]
    offset = abs(reference["centroid"][2])
    assert offset == pytest.approx(3.0 * c / 8.0, rel=1.0e-14)
    axial = mass * (a * a + b * b) / 5.0
    assert reference["inertia_cg"][2] == pytest.approx(axial, rel=1.0e-14)
    # Transverse entries are the full-body values minus m d^2.
    assert reference["inertia_cg"][0] == pytest.approx(
        mass * (b * b + c * c) / 5.0 - mass * offset**2, rel=1.0e-14
    )
    assert math.isclose(reference["volume"], 2.0 / 3.0 * math.pi * a * b * c, rel_tol=1e-14)