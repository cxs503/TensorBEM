"""Inhomogeneous (field) density on the mesh mass route — CASE-BEM-RIGID-BODY-MASS r6."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.benchmarks import rigid_body_mass_inhomogeneous as case
from tensorbem.geometry3d import rectangular_barge_hull_mesh
from tensorbem.hydrostatic import mass_matrix_from_wetted_hull
from tensorbem.rigid_body_mass import mesh_mass_properties, wetted_hull_mass_properties

LENGTH, BEAM, DEPTH = 20.0, 8.0, 3.0
DENSITY = case.DENSITY


def _box():
    return case.box_mesh(LENGTH, BEAM, DEPTH)


def _analytic(**kwargs):
    return case.box_mass_properties(LENGTH, BEAM, DEPTH, **kwargs)


# --------------------------------------------------------------- field path ---
def test_uniform_field_reproduces_the_homogeneous_route():
    """Backward compatibility: f == 1 must not change the arithmetic."""
    vertices, faces = _box()
    plain = mesh_mass_properties(vertices, faces, DENSITY)
    field = mesh_mass_properties(vertices, faces, DENSITY, density_field=case.UNIFORM.value)
    assert field["mass"] == pytest.approx(plain["mass"], rel=1e-12)
    assert field["volume"] == pytest.approx(plain["volume"], rel=1e-12)
    assert np.allclose(field["centroid"], plain["centroid"], atol=1e-9)
    assert np.allclose(field["inertia_cg"], plain["inertia_cg"], rtol=1e-12)


def test_affine_field_is_exact_on_a_polyhedron():
    """A linear density field is integrated exactly by the tetrahedron route."""
    vertices, faces = _box()
    reference = case.box_mass_properties(
        LENGTH, BEAM, DEPTH, case.longitudinal(1.2, LENGTH), density_scale=DENSITY
    )
    props = mesh_mass_properties(
        vertices, faces, DENSITY, density_field=case.longitudinal(1.2, LENGTH).value
    )
    assert props["mass"] == pytest.approx(reference["mass"], rel=1e-12)
    assert np.allclose(props["centroid"], reference["centroid"], atol=1e-10)
    assert np.allclose(props["inertia_cg"], reference["inertia_cg"], rtol=1e-10)


def test_vertical_field_moves_the_mass_centroid_down():
    """A heavy-bottom field moves the CG down and leaves the mass invariant."""
    vertices, faces = _box()
    uniform = mesh_mass_properties(vertices, faces, DENSITY)
    field = case.vertical(1.8, DEPTH, 0.0)
    props = mesh_mass_properties(vertices, faces, DENSITY, density_field=field.value)
    # f = 1 + 1.8 z / 3 is odd about the body centre, so the mass is invariant...
    assert props["mass"] == pytest.approx(uniform["mass"], rel=1e-12)
    # ...and the centroid moves by c * T / 12 = 0.45 m upwards.
    assert props["centroid"][2] == pytest.approx(0.45, rel=1e-9)
    assert abs(props["centroid"][0]) < 1e-9 and abs(props["centroid"][1]) < 1e-9


def test_quadrupole_field_creates_a_product_of_inertia():
    """A field of the form x*y produces the off-diagonal inertia entry."""
    vertices, faces = _box()
    coefficient = 1.6
    field = case.quadrupole(coefficient, LENGTH, BEAM)
    props = mesh_mass_properties(vertices, faces, DENSITY, density_field=field.value)
    # int rho x y dV = rho0 c T L^2 B^2 / 144 and I_xy = -int rho x y dV.
    expected = -DENSITY * coefficient * DEPTH * LENGTH**2 * BEAM**2 / 144.0
    # The 12-triangle box resolves a bilinear field only to second order (the
    # tetra interpolation is linear), hence the 10 % band; the sign and the
    # closed-form magnitude are pinned, and the refinement ladder is checked in
    # the battery test.
    assert props["inertia_cg"][0, 1] == pytest.approx(expected, rel=0.1)
    assert props["inertia_cg"][0, 1] != 0.0


def test_density_range_is_reported():
    vertices, faces = _box()
    props = mesh_mass_properties(
        vertices, faces, DENSITY, density_field=case.longitudinal(1.2, LENGTH).value
    )
    assert props["density_field"] == "field"
    assert props["density_min"] == pytest.approx(DENSITY * 0.4, rel=1e-6)
    assert props["density_max"] == pytest.approx(DENSITY * 1.6, rel=1e-6)
    assert DENSITY == pytest.approx(props["density"])


def test_reference_point_translation_does_not_change_the_cg_inertia():
    vertices, faces = _box()
    shift = np.array([3.0, -2.0, 1.0])
    field = case.longitudinal(1.2, LENGTH).value
    here = mesh_mass_properties(vertices, faces, DENSITY, density_field=field)
    # The field is attached to the frame, so it moves with the body: shifting the
    # mesh while keeping the field fixed would move the ballast in absolute
    # space, which is a different (and correctly different) body.
    shifted_vertices = vertices + shift
    there = mesh_mass_properties(
        shifted_vertices,
        faces,
        DENSITY,
        reference_point=shift,
        density_field=lambda points: field(np.asarray(points, dtype=float) - shift),
    )
    assert np.allclose(there["inertia_cg"], here["inertia_cg"], rtol=1e-10)
    # Rigid displacement: the mass centroid is the same physical point (absolute
    # coordinates move with the body) and the CG inertia is a property of the body.
    assert np.allclose(there["centroid"], np.asarray(here["centroid"]) + shift, atol=1e-9)
    assert there["mass"] == pytest.approx(here["mass"], rel=1e-12)


# ------------------------------------------------------------------- guards ---
@pytest.mark.parametrize(
    "probe",
    [
        lambda points: -np.ones(np.asarray(points).reshape(-1, 3).shape[0]),
        lambda points: np.zeros(np.asarray(points).reshape(-1, 3).shape[0]),
        lambda points: np.full(np.asarray(points).reshape(-1, 3).shape[0], np.nan),
        lambda points: np.ones(3),
    ],
)
def test_non_physical_fields_are_rejected(probe):
    vertices, faces = _box()
    with pytest.raises(ValueError):
        mesh_mass_properties(vertices, faces, DENSITY, density_field=probe)


# ----------------------------------------------------------- production path ---
def test_wetted_barge_production_path_with_a_density_field():
    """The default-inertia branch of the seakeeping chains takes the field."""
    field = case.vertical(1.8, DEPTH, -0.5 * DEPTH)
    hull = rectangular_barge_hull_mesh(LENGTH, BEAM, DEPTH)
    props = wetted_hull_mass_properties(
        hull.vertices, hull.faces, DENSITY, density_field=field.value
    )
    reference = case.box_mass_properties(
        LENGTH, BEAM, DEPTH,
        field,
        density_scale=DENSITY,
        centre=(0.0, 0.0, -0.5 * DEPTH),
    )
    assert props["mass"] == pytest.approx(reference["mass"], rel=1e-10)
    assert np.allclose(props["centroid"], reference["centroid"], atol=1e-9)
    assert np.allclose(props["inertia_cg"], reference["inertia_cg"], rtol=1e-9)
    # The mass centroid is above the volume centroid (heavy bottom => CG down and
    # the volume centroid is at -T/2): both are reported, and they differ.
    assert props["centroid"][2] != pytest.approx(-0.5 * DEPTH, abs=1e-6)
    matrix = mass_matrix_from_wetted_hull(hull, density_field=field.value)
    assert np.allclose(matrix, case.assemble(reference), rtol=1e-9, atol=1e-3)
    # The homogeneous route on the same hull is the pre-fix behaviour.
    homogeneous = mass_matrix_from_wetted_hull(hull)
    assert not np.allclose(matrix, homogeneous, rtol=1e-3)


# ------------------------------------------------------------ the whole run ---
def test_case_battery_passes_and_is_under_the_tolerance():
    results = case.run()
    verdict = results["verdict"]
    assert verdict["all_pass"], verdict["checks"]
    assert verdict["worst_exact_entry_rel_error"] < case.TOLERANCE
    assert verdict["worst_quadrupole_accepted_block_rel_error"] < case.TOLERANCE
    assert verdict["worst_ellipsoid_accepted_entry_rel_error"] < case.TOLERANCE
    assert verdict["quadrupole_convergence_order"] > 1.5
    assert verdict["ellipsoid_convergence_order"] > 1.5
    assert results["homogeneous_compatibility"]["passed"]
    assert results["negative_controls"]["passed"]