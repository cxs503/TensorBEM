"""Rigid-body mass matrix: closed block form, mesh mass properties, integration.

The block form under test is Fossen (2011) eq. (3.44),

    M_RB = [[ m I3, -m S(r_g) ], [ m S(r_g), I_g + m S(r_g)^T S(r_g) ]],

and the physics it must satisfy is the rigid-body kinetic energy written at the
CG, ``0.5 x^T M_RB x = 0.5 m |v + w x r_g|^2 + 0.5 w I_g w``.  The energy
identity is the arbiter: it is frame independent, so it catches sign slips in
the off-diagonal blocks and a double-counted parallel-axis term, neither of which
breaks symmetry (mission m17324 found exactly that in
``hydrostatic.compute_mass_matrix``).
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.benchmarks.rigid_body_mass_case import run as run_case
from tensorbem.geometry3d import ellipsoid_mesh
from tensorbem.hydrostatic import compute_mass_matrix
from tensorbem.rigid_body_mass import (
    block_form_residual,
    energy_residual,
    inertia_about_centroid,
    inertia_about_point,
    mesh_mass_properties,
    rigid_body_mass_matrix,
    skew_matrix,
    solid_ellipsoid_mass_properties,
)

MASS = 8200.0
CG = np.array([0.6, -0.4, -1.3])
INERTIA_CG = np.array(
    [
        [9.0e3, 0.0, 0.0],
        [0.0, 1.1e4, 0.0],
        [0.0, 0.0, 1.3e4],
    ]
)


class TestBlockForm:
    def test_blocks_are_the_closed_form(self):
        matrix = rigid_body_mass_matrix(MASS, CG, INERTIA_CG, inertia_about="cg")
        s = skew_matrix(CG)
        assert np.allclose(matrix[:3, :3], MASS * np.eye(3), rtol=0.0, atol=0.0)
        assert np.allclose(matrix[:3, 3:], -MASS * s, rtol=0.0, atol=1e-9)
        assert np.allclose(matrix[3:, :3], MASS * s, rtol=0.0, atol=1e-9)
        assert np.allclose(
            matrix[3:, 3:], INERTIA_CG + MASS * (s.T @ s), rtol=0.0, atol=1e-6
        )

    def test_coupling_blocks_are_not_the_transpose_convention(self):
        """A sign slip in both off-diagonal blocks keeps symmetry but flips physics."""
        matrix = rigid_body_mass_matrix(MASS, CG, INERTIA_CG, inertia_about="cg")
        slipped = matrix.copy()
        slipped[:3, 3:] = -slipped[:3, 3:]
        slipped[3:, :3] = -slipped[3:, :3]
        assert np.allclose(slipped, slipped.T)  # symmetry cannot see it
        assert energy_residual(slipped, MASS, INERTIA_CG, CG) > 0.1
        assert energy_residual(matrix, MASS, INERTIA_CG, CG) < 1e-13

    def test_symmetric_and_positive_definite(self):
        matrix = rigid_body_mass_matrix(MASS, CG, INERTIA_CG, inertia_about="cg")
        assert np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-9)
        eigenvalues = np.linalg.eigvalsh(matrix)
        # A point mass at r_g has a singular block form (a twist with
        # w x r_g = -v carries no kinetic energy); only the finite inertia makes
        # the matrix positive definite, and it must stay above the point-mass
        # null direction.
        assert np.all(eigenvalues > 0.0)
        point_mass = rigid_body_mass_matrix(MASS, CG, np.zeros((3, 3)))
        assert np.min(np.linalg.eigvalsh(point_mass)) == pytest.approx(0.0, abs=1e-6)
        assert eigenvalues[0] > np.min(np.linalg.eigvalsh(point_mass))

    def test_cg_at_reference_point_decouples(self):
        matrix = rigid_body_mass_matrix(MASS, (0.0, 0.0, 0.0), INERTIA_CG)
        assert np.allclose(matrix[:3, 3:], 0.0)
        assert np.allclose(matrix[3:, :3], 0.0)

    def test_origin_and_cg_conventions_agree(self):
        inertia_origin = inertia_about_point(INERTIA_CG, MASS, CG)
        from_origin = rigid_body_mass_matrix(
            MASS, CG, inertia_origin, inertia_about="origin"
        )
        from_cg = rigid_body_mass_matrix(MASS, CG, INERTIA_CG, inertia_about="cg")
        assert np.allclose(from_origin, from_cg, rtol=1e-12, atol=1e-6)
        back = inertia_about_centroid(inertia_origin, MASS, CG)
        assert np.allclose(back, INERTIA_CG, rtol=1e-12, atol=1e-6)

    def test_invalid_inputs_are_rejected(self):
        with pytest.raises(ValueError):
            rigid_body_mass_matrix(0.0, CG, INERTIA_CG)
        with pytest.raises(ValueError):
            rigid_body_mass_matrix(MASS, CG, np.eye(3) + np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
        with pytest.raises(ValueError):
            rigid_body_mass_matrix(MASS, CG, INERTIA_CG, inertia_about="nonsense")


class TestMeshMassProperties:
    """Signed-tetrahedron quadrature of a closed BEM mesh vs closed forms."""

    @pytest.mark.parametrize("subdivisions", [2, 3])
    def test_ellipsoid_mass_properties_converge(self, subdivisions):
        axes = (60.0, 12.0, 8.0)
        reference = solid_ellipsoid_mass_properties(axes, 1025.0)
        mesh = ellipsoid_mesh(subdivisions, np.asarray(axes))
        props = mesh_mass_properties(mesh.vertices, mesh.faces, 1025.0)
        assert props["volume"] == pytest.approx(reference["volume"], rel=0.04)
        assert np.allclose(
            props["centroid"], np.zeros(3), atol=1e-9
        )
        inertia = np.diag(np.asarray(props["inertia_cg"]))
        expected = np.diag(np.asarray(reference["inertia_cg"]))
        assert np.all(np.abs(inertia - expected) / expected < 0.06)

    def test_finer_mesh_is_more_accurate(self):
        axes = (60.0, 12.0, 8.0)
        reference = solid_ellipsoid_mass_properties(axes, 1025.0)
        errors = []
        for subdivisions in (1, 2, 3):
            mesh = ellipsoid_mesh(subdivisions, np.asarray(axes))
            props = mesh_mass_properties(mesh.vertices, mesh.faces, 1025.0)
            errors.append(
                abs(props["volume"] - reference["volume"]) / reference["volume"]
            )
        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] < 0.01

    def test_sphere_limit_matches_two_fifths_formula(self):
        axes = (3.0, 3.0, 3.0)
        mesh = ellipsoid_mesh(4, np.asarray(axes))
        props = mesh_mass_properties(mesh.vertices, mesh.faces, 1.0)
        analytic = 0.4 * props["mass"] * 9.0
        assert props["inertia_cg"][0, 0] == pytest.approx(analytic, rel=5e-3)

    def test_translation_moves_only_the_coupling_blocks(self):
        axes = (60.0, 12.0, 8.0)
        mesh = ellipsoid_mesh(3, np.asarray(axes))
        offset = np.array([5.0, 0.0, -3.0])
        base = mesh_mass_properties(mesh.vertices, mesh.faces, 1025.0)
        moved = mesh_mass_properties(mesh.vertices + offset, mesh.faces, 1025.0)
        assert np.allclose(np.asarray(moved["centroid"]), offset, atol=1e-9)
        assert np.allclose(np.asarray(moved["inertia_cg"]), np.asarray(base["inertia_cg"]), rtol=1e-9, atol=1e-3)
        m1 = rigid_body_mass_matrix(
            base["mass"], (0.0, 0.0, 0.0), base["inertia_cg"], inertia_about="cg"
        )
        m2 = rigid_body_mass_matrix(
            moved["mass"], moved["centroid"], moved["inertia_cg"], inertia_about="cg"
        )
        assert np.allclose(m1[:3, :3], m2[:3, :3], rtol=0.0)
        assert np.max(np.abs(m2[3:, 3:] - m1[3:, 3:])) > 0.0


class TestPlatformIntegration:
    """``hydrostatic.compute_mass_matrix`` is the assembly used by rao/ice_rao."""

    def test_platform_assembly_matches_closed_form(self):
        inertia_origin = inertia_about_point(INERTIA_CG, MASS, CG)
        matrix = compute_mass_matrix(MASS, CG, inertia_origin, inertia_about="origin")
        residual = block_form_residual(matrix, MASS, INERTIA_CG, CG)
        assert residual["max_relative_error"] < 1e-12
        assert residual["block_relative_error"]["translation_rotation"] < 1e-12

    def test_platform_assembly_satisfies_kinetic_energy(self):
        inertia_origin = inertia_about_point(INERTIA_CG, MASS, CG)
        matrix = compute_mass_matrix(MASS, CG, inertia_origin, inertia_about="origin")
        assert energy_residual(matrix, MASS, INERTIA_CG, CG) < 1e-12

    def test_cg_convention_keyword(self):
        inertia_origin = inertia_about_point(INERTIA_CG, MASS, CG)
        from_origin = compute_mass_matrix(
            MASS, CG, inertia_origin, inertia_about="origin"
        )
        from_cg = compute_mass_matrix(MASS, CG, INERTIA_CG, inertia_about="cg")
        assert np.allclose(from_origin, from_cg, rtol=1e-12, atol=1e-6)

    def test_default_gyration_path_is_positive_definite(self):
        matrix = compute_mass_matrix(MASS, CG)
        assert np.all(np.linalg.eigvalsh(matrix) > 0.0)


class TestCaseVerdict:
    def test_case_passes_all_checks(self):
        report = run_case(subdivisions=(2, 3))
        verdict = report["verdict"]
        assert verdict["all_pass"], verdict["checks"]
        assert verdict["accepted_errors"][
            str(report["parameters"]["accepted_from_subdivision"])
        ] < report["parameters"]["tolerance"]

    def test_case_uses_the_same_platform_assembly(self):
        report = run_case(subdivisions=(3,))
        platform = report["platform_assembly"]
        assert platform["energy_identity_error"] < 1e-12
        assert platform["block_form_residual"] < 1e-12