"""Tests for the Wigley-hull rigid-body mass-matrix case (CASE-BEM-RIGID-BODY-MASS).

The tests assert invariants, not merely that code runs:

* the closed-form mass properties (route R) are re-derived by an independent
  Gauss-Legendre quadrature of the definitions (route Q) to round-off -- this
  fails if any of the reduction coefficients in the module docstring is wrong;
* the parabolic Wigley hull has ``Cb = 4/9`` and ``V = 4 L B T / 9`` exactly;
* the hand-written closed block form equals
  :func:`tensorbem.rigid_body_mass.rigid_body_mass_matrix` (the assembler is
  checked, not reused);
* the closed mesh (wetted surface + waterplane fan) is watertight -- every
  directed half-edge appears exactly once and its reverse exactly once;
* the platform assembly path ``hydrostatic.compute_mass_matrix`` reproduces the
  closed-form 6x6 matrix entry-by-entry within the 3 % case-pool criterion at
  the accepted mesh level, and the discretisation error is O(h^2) down the
  ladder;
* the kinetic-energy identity holds at round-off on the mesh properties, and
  the pre-m17324 transposed coupling block breaks it (negative control with
  discriminating power).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.benchmarks.case_rigid_body_mass import (
    kinetic_energy_error,
    relative_error_matrix,
)
from tensorbem.benchmarks.case_rigid_body_mass_wigley import (
    HULL,
    TOLERANCE,
    WigleyHull,
    closed_form_mass_properties,
    hand_block_matrix,
    mesh_route,
    quadrature_mass_properties,
    run,
    solid_mesh,
    transfer_inertia,
)
from tensorbem.hydrostatic import compute_mass_matrix
from tensorbem.rigid_body_mass import mesh_mass_properties, rigid_body_mass_matrix


def test_volume_and_block_coefficient_are_the_wigley_closed_forms() -> None:
    props = closed_form_mass_properties(HULL)
    L, B, T = HULL.length, HULL.beam, HULL.draft
    assert props["volume"] == pytest.approx(4.0 * L * B * T / 9.0, rel=1e-15)
    assert props["block_coefficient"] == pytest.approx(4.0 / 9.0, rel=1e-15)
    assert props["centroid"][2] == pytest.approx(-3.0 * T / 8.0, rel=1e-15)
    assert props["centroid"][0] == 0.0


def test_closed_form_is_reproduced_by_independent_quadrature() -> None:
    """Route Q integrates the definitions; disagreement means a wrong coefficient."""
    closed = closed_form_mass_properties(HULL)
    quad = quadrature_mass_properties(HULL, order=24)
    assert quad["volume"] == pytest.approx(closed["volume"], rel=1e-12)
    assert np.allclose(quad["centroid"], closed["centroid"], rtol=0.0, atol=1e-12)
    for key in ("inertia_cg", "inertia_at_design_cg"):
        scale = float(np.max(np.abs(closed[key])))
        assert np.allclose(quad[key], closed[key], rtol=1e-12, atol=1e-10 * scale)


def test_second_moment_reduction_terms() -> None:
    """``int x^2 = L^2 V/20``, ``int y^2 = 48 B^2 V/1225``, ``int z^2 = T^2 V/5``."""
    props = closed_form_mass_properties(HULL)
    L, B, T, V = HULL.length, HULL.beam, HULL.draft, props["volume"]
    moments = props["second_moments"]
    assert moments["int_x2_dV"] == pytest.approx(L * L * V / 20.0, rel=1e-14)
    assert moments["int_y2_dV"] == pytest.approx(48.0 * B * B * V / 1225.0, rel=1e-14)
    assert moments["int_z2_dV"] == pytest.approx(T * T * V / 5.0, rel=1e-14)
    # parallel-axis shift between z = 0 (waterplane) and the centroid z_B
    shift = V * (3.0 * T / 8.0) ** 2
    assert (
        moments["int_z2_dV"] - moments["int_z2_dV_about_centroid"]
    ) == pytest.approx(shift, rel=1e-13)


def test_inertia_is_positive_definite_and_realisable() -> None:
    inertia = closed_form_mass_properties(HULL)["inertia_cg"]
    eigenvalues = np.linalg.eigvalsh(inertia)
    assert np.all(eigenvalues > 0.0)
    # physically realisable: for a rigid body I1 + I2 >= I3 on every permutation
    i1, i2, i3 = sorted(eigenvalues)
    assert i1 + i2 >= i3 * (1.0 - 1e-12)


def test_transfer_inertia_is_invertible_and_symmetric() -> None:
    props = closed_form_mass_properties(HULL)
    inertia = props["inertia_cg"]
    mass = props["mass"]
    there = transfer_inertia(inertia, mass, props["centroid"], HULL.cg)
    # the parallel-axis term depends on |d|^2 only, so returning to the centroid
    # means *subtracting* it (transfer_inertia shifts one way only)
    back = there - transfer_inertia(
        np.zeros((3, 3)), mass, props["centroid"], HULL.cg
    )
    assert np.allclose(back, inertia, rtol=1e-12, atol=1e-6 * np.max(np.abs(inertia)))
    assert np.allclose(there, there.T, rtol=0.0, atol=1e-6 * np.max(np.abs(there)))
    # the transfer to the design CG creates a product of inertia I_xz = -m d_x d_z
    dx = HULL.cg[0] - props["centroid"][0]
    dz = HULL.cg[2] - props["centroid"][2]
    assert there[0, 2] == pytest.approx(-mass * dx * dz, rel=1e-12)


def test_hand_block_form_matches_the_assembler() -> None:
    props = closed_form_mass_properties(HULL)
    ref = hand_block_matrix(
        props["mass"], HULL.cg, props["inertia_at_design_cg"]
    )
    got = rigid_body_mass_matrix(
        props["mass"], HULL.cg, props["inertia_at_design_cg"], inertia_about="cg"
    )
    scale = float(np.max(np.abs(ref)))
    assert np.allclose(got, ref, rtol=1e-15, atol=1e-12 * scale)
    assert np.allclose(ref, ref.T, rtol=0.0, atol=1e-12 * scale)
    # the coupling block is m * S(r_g): linear in the CG offset, with the
    # documented sign (upper-right = -m S)
    x, y, z = HULL.cg
    # surge-pitch = m z_g, sway-roll = -m z_g  (upper-right block = -m S)
    assert ref[0, 4] == pytest.approx(props["mass"] * z, rel=1e-15)
    assert ref[1, 3] == pytest.approx(-props["mass"] * z, rel=1e-15)
    assert ref[3, 1] == pytest.approx(-props["mass"] * z, rel=1e-15)


@pytest.mark.parametrize("n_sections,n_levels", [(6, 3), (24, 12)])
def test_closed_mesh_is_watertight(n_sections: int, n_levels: int) -> None:
    vertices, faces, meta = solid_mesh(HULL, n_sections, n_levels)
    counts: dict[tuple[int, int], int] = {}
    for triangle in faces:
        for a, b in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            counts[(int(a), int(b))] = counts.get((int(a), int(b)), 0) + 1
    assert all(count == 1 for count in counts.values()), "duplicated half-edge"
    missing = [edge for edge in counts if (edge[1], edge[0]) not in counts]
    assert not missing, "surface is not closed"
    props = mesh_mass_properties(vertices, faces, HULL.density)
    assert props["volume"] > 0.0
    assert meta["lid_panels"] > 0 and meta["wetted_panels"] > 0


def test_platform_assembly_reproduces_the_closed_form_within_3pct() -> None:
    closed = closed_form_mass_properties(HULL)
    ref = hand_block_matrix(
        closed["mass"], HULL.cg, closed["inertia_at_design_cg"]
    )
    route = mesh_route(HULL, 24, 12)
    errors = relative_error_matrix(ref, route["matrix"])
    assert errors.max() < TOLERANCE, f"max per-entry error {errors.max():.4%}"
    # the geometry gate the ecosystem requires: Cb / L / B before any tolerance
    assert abs(route["block_coefficient"] - 4.0 / 9.0) / (4.0 / 9.0) < TOLERANCE
    assert abs(route["bbox"]["length"] - HULL.length) / HULL.length < 1e-12
    assert abs(route["bbox"]["beam"] - HULL.beam) / HULL.beam < TOLERANCE


def test_mesh_route_energy_identity_is_roundoff() -> None:
    route = mesh_route(HULL, 48, 24)
    assert route["energy_identity_error"] < 1e-12
    # the platform call must be the same matrix as the assembler's
    direct = compute_mass_matrix(
        route["mass"], HULL.cg, route["inertia_at_design_cg"], inertia_about="cg"
    )
    assert np.allclose(route["matrix"], direct, rtol=0.0, atol=0.0)


def test_coarse_mesh_fails_the_criterion_so_the_case_discriminates() -> None:
    """A 6x3 grid is not enough: the case must not be trivially passed."""
    closed = closed_form_mass_properties(HULL)
    ref = hand_block_matrix(
        closed["mass"], HULL.cg, closed["inertia_at_design_cg"]
    )
    coarse = mesh_route(HULL, 6, 3)
    assert relative_error_matrix(ref, coarse["matrix"]).max() > TOLERANCE


def test_convergence_is_second_order() -> None:
    from tensorbem.benchmarks.case_rigid_body_mass_wigley import run as run_case

    report = run_case(ladder=[(6, 3), (12, 6), (24, 12)])
    errors = [level["max_entry_rel_error"] for level in report["levels"]]
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    # h halves between rungs (n_sections doubles): O(h^2) means ratio ~ 4
    assert all(3.0 < ratio < 5.0 for ratio in ratios), ratios


def test_negative_control_and_verdict() -> None:
    report = run(ladder=[(12, 6), (24, 12)])
    assert report["verdict"]["all_pass"], report["verdict"]["checks"]
    assert report["negative_control"]["detected"]
    assert report["negative_control"]["max_entry_rel_error"] > TOLERANCE
    assert report["negative_control"]["energy_identity_error"] > 0.1
    assert math.isfinite(report["routes"]["A_assembler"]["rel_residual_vs_hand_block"])
    assert report["routes"]["A_assembler"]["rel_residual_vs_hand_block"] < 1e-15
    assert report["reference"]["block_coefficient"] == pytest.approx(4.0 / 9.0)


def test_reference_matrix_energy_identity_uses_the_design_cg() -> None:
    closed = closed_form_mass_properties(HULL)
    ref = hand_block_matrix(
        closed["mass"], HULL.cg, closed["inertia_at_design_cg"]
    )
    assert (
        kinetic_energy_error(
            ref, closed["mass"], HULL.cg, closed["inertia_at_design_cg"]
        )
        < 1e-12
    )


def test_invalid_hull_is_rejected() -> None:
    with pytest.raises(ValueError):
        WigleyHull(length=-1.0)
    with pytest.raises(ValueError):
        WigleyHull(density=0.0)