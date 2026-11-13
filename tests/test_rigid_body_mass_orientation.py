"""Tests for the orientation-covariance extension of CASE-BEM-RIGID-BODY-MASS.

Two groups:

* ``reference_point`` regression -- :func:`tensorbem.rigid_body_mass.
  mesh_mass_properties` must take its second moments about the *supplied*
  reference point.  Before this change the tensor was integrated about the
  coordinate origin, so translating the mesh and the reference point together
  moved the reported ``inertia_cg`` by 60 % on the case body.
* orientation covariance -- the rotated body's 6x6 must satisfy
  ``M(R.b) = A M(b) A^T`` with ``A = blockdiag(R, R)``, the mesh route must
  reproduce the closed form, and an independent code (capytaine, when
  installed) must agree on the block conventions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.benchmarks.rigid_body_mass_orientation import (
    DEFAULT_CG_OFFSET,
    DEFAULT_DENSITY,
    DEFAULT_ROTATION,
    SEMI_AXES,
    external_capytaine,
    legacy_convention,
    mesh_level,
    rotated_panel_mesh,
    rotated_reference,
    rotation_matrix,
    twist_congruence,
)
from tensorbem.geometry3d import ellipsoid_mesh
from tensorbem.hydrostatic import compute_mass_matrix
from tensorbem.rigid_body_mass import (
    energy_residual,
    mesh_mass_properties,
    rigid_body_mass_matrix,
    solid_ellipsoid_mass_properties,
)


# --------------------------------------------------------------------------- #
# reference_point handling of the mesh quadrature
# --------------------------------------------------------------------------- #
def test_reference_point_translation_invariance():
    """Same physical body, two frames: every mass property is invariant."""
    mesh = ellipsoid_mesh(2, np.asarray(SEMI_AXES, dtype=float))
    base = mesh_mass_properties(mesh.vertices, mesh.faces, DEFAULT_DENSITY)
    shift = np.array([7.0, -4.0, -20.0])
    moved = mesh_mass_properties(
        mesh.vertices + shift, mesh.faces, DEFAULT_DENSITY, reference_point=shift
    )
    for key in ("inertia_cg", "inertia_reference"):
        a = np.asarray(base[key], dtype=float)
        b = np.asarray(moved[key], dtype=float)
        assert np.max(np.abs(a - b)) / np.max(np.abs(a)) < 1.0e-12, key
    assert abs(base["volume"] - moved["volume"]) / base["volume"] < 1.0e-12
    offset_centroid = np.asarray(moved["centroid"], dtype=float) - shift
    assert np.max(np.abs(offset_centroid - np.asarray(base["centroid"], dtype=float))) < 1.0e-9


def test_reference_point_moments_are_about_that_point():
    """``inertia_reference`` must equal the parallel-axis transfer to ``p0``."""
    mesh = ellipsoid_mesh(2, np.asarray(SEMI_AXES, dtype=float))
    shift = np.array([0.0, 0.0, -20.0])
    props = mesh_mass_properties(
        mesh.vertices + shift, mesh.faces, DEFAULT_DENSITY, reference_point=shift
    )
    mass = float(props["mass"])
    r = np.asarray(props["centroid"], dtype=float) - shift  # CG relative to p0
    expected = np.asarray(props["inertia_cg"], dtype=float) + mass * (
        float(r @ r) * np.eye(3) - np.outer(r, r)
    )
    got = np.asarray(props["inertia_reference"], dtype=float)
    assert np.max(np.abs(got - expected)) / np.max(np.abs(got)) < 1.0e-12


# --------------------------------------------------------------------------- #
# orientation covariance
# --------------------------------------------------------------------------- #
def test_rotation_matrix_is_a_proper_rotation():
    rot = rotation_matrix(*DEFAULT_ROTATION)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1.0e-15)
    assert math.isclose(float(np.linalg.det(rot)), 1.0, abs_tol=1.0e-15)


def test_rotation_congruence_holds_to_roundoff():
    """M(R.b) = A M(b) A^T for several unrelated orientations."""
    body = solid_ellipsoid_mass_properties(SEMI_AXES, DEFAULT_DENSITY)
    m = float(body["mass"])
    inertia_body = np.asarray(body["inertia_cg"], dtype=float)
    matrix_body = rigid_body_mass_matrix(
        m, DEFAULT_CG_OFFSET, inertia_body, inertia_about="cg"
    )
    for angles in [
        DEFAULT_ROTATION,
        (math.radians(-35.0), math.radians(10.0), math.radians(-12.0)),
        (math.radians(70.0), math.radians(-20.0), math.radians(40.0)),
    ]:
        rot = rotation_matrix(*angles)
        rotated = rotated_reference(rot)
        congruence = twist_congruence(rot)
        rebuild = congruence @ matrix_body @ congruence.T
        scale = float(np.max(np.abs(rotated["matrix_rotated"])))
        assert float(np.max(np.abs(np.asarray(rotated["matrix_rotated"]) - rebuild))) / scale < 1.0e-12
        assert rotated["congruence_residual_rel"] < 1.0e-12


def test_rotation_activates_products_of_inertia_and_cross_entries():
    """The rotated body must have a genuinely full inertia tensor."""
    rot = rotation_matrix(*DEFAULT_ROTATION)
    ref = rotated_reference(rot)
    assert ref["max_product_over_max_diagonal"] > 0.05
    matrix = np.asarray(ref["matrix_rotated"], dtype=float)
    scale = float(np.max(np.abs(matrix)))
    # roll-pitch entry is exactly zero for the axis-aligned body and O(10 %) here
    assert abs(matrix[3, 4]) / scale > 0.1
    aligned = np.asarray(
        rigid_body_mass_matrix(
            ref["mass"],
            DEFAULT_CG_OFFSET,
            ref["inertia_body"],
            inertia_about="cg",
        ),
        dtype=float,
    )
    assert abs(aligned[3, 4]) / scale < 1.0e-9


def test_rotated_mesh_reproduces_closed_form_within_criterion():
    rot = rotation_matrix(*DEFAULT_ROTATION)
    ref = rotated_reference(rot)
    level = mesh_level(rot, 3, ref)
    assert level["within_tolerance"], level["worst_entry"]
    assert level["max_entry_rel_error"] < 0.03
    assert level["energy_identity_error"] < 1.0e-12
    # the cross-inertia entries are reproduced, not merely the diagonal
    assert level["max_nonzero_entry_rel_error"] < 0.03


def test_legacy_convention_is_detected():
    """The pre-fix assembly is symmetric yet O(1) wrong: the criteria notice."""
    rot = rotation_matrix(*DEFAULT_ROTATION)
    control = legacy_convention(rotated_reference(rot))
    assert control["symmetric"] is True
    assert control["detects"]["entrywise_over_3pct"] is True
    assert control["max_entry_rel_error"] > 1.0


# --------------------------------------------------------------------------- #
# solver-path convention
# --------------------------------------------------------------------------- #
def test_platform_mass_matrix_expects_cg_inertia():
    """``compute_mass_matrix`` adds the parallel-axis term itself."""
    body = solid_ellipsoid_mass_properties(SEMI_AXES, DEFAULT_DENSITY)
    m = float(body["mass"])
    inertia_cg = np.asarray(body["inertia_cg"], dtype=float)
    cg = np.asarray(DEFAULT_CG_OFFSET, dtype=float)
    got = compute_mass_matrix(m, cg, inertia=inertia_cg)
    ref = rigid_body_mass_matrix(m, cg, inertia_cg, inertia_about="cg")
    scale = float(np.max(np.abs(ref)))
    assert float(np.max(np.abs(got - ref))) / scale < 1.0e-14
    assert energy_residual(got, m, inertia_cg, cg) < 1.0e-12


# --------------------------------------------------------------------------- #
# external anchor
# --------------------------------------------------------------------------- #
def test_capytaine_agrees_on_conventions():
    """Independent code (capytaine) reproduces mass/coupling and the rotation block."""
    rot = rotation_matrix(*DEFAULT_ROTATION)
    ref = rotated_reference(rot)
    ext = external_capytaine(rot, 3, ref)
    if not ext["available"]:
        pytest.skip(f"capytaine not installed ({ext['import_error']})")
    assert ext["within_tolerance"], ext["max_entry_rel_error_vs_closed_form"]
    block_errors = ext["block_errors_external_vs_tensor_bem"]
    assert block_errors["translation_translation"] < 1.0e-9
    assert block_errors["translation_rotation"] < 1.0e-9
    # capytaine's rotation block uses face-centroid quadrature: converge-level
    assert block_errors["rotation_rotation"] < 0.01