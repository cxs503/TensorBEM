"""Gates for CASE-BEM-RIGID-BODY-MASS (the 6x6 rigid-body mass matrix).

The unit under test is :func:`tensorbem.hydrostatic.compute_mass_matrix`, the
function the seakeeping solver assembles into its impedance.  Three independent
statements are gated here:

1. the shipped matrix equals the closed-form block form of Fossen (2011,
   section 3.3.1, eq. 3.51) entry by entry, for a body whose mass properties are
   derived twice (analytic composite route and mesh/divergence-theorem route);
2. the physical kinetic-energy identity ``0.5 x^T M x = 0.5 m |v_g|^2 + 0.5 w^T
   I_g w`` holds -- an identity that a *transposed* coupling block violates,
   although such a block is still symmetric and positive definite (the reason
   the earlier symmetry-only tests could not see the defect);
3. the negative control (the pre-fix ``-m S^T`` upper-right block) is rejected
   by the same criterion, so a green run means the benchmark discriminates;
4. the main solver (``tensorbem.rao``) really consumes this matrix: injecting
   the reference matrix at that call site reproduces its RAO to round-off.

Run: ``python -m pytest tests/test_case_rigid_body_mass.py -q``
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tensorbem.benchmarks import case_rigid_body_mass as module
from tensorbem.hydrostatic import compute_mass_matrix

TOL = module.TOLERANCE


@pytest.fixture(scope="module")
def study() -> dict:
    """One execution of the case (RAO caller check included)."""
    return module.run((1, 2, 4), with_rao=True)


# --------------------------------------------------------------------------
# reference definition: the block form itself
# --------------------------------------------------------------------------
def test_reference_upper_right_block_is_minus_m_skew() -> None:
    """M[0:3, 3:6] must be -m*S(r_g), not its transpose (Fossen eq. 3.51)."""
    mass, r_g = 1000.0, (1.2, -0.7, 3.1)
    inertia = np.diag([2.0e3, 3.0e3, 4.0e3])
    ref = module.closed_form_mass_matrix(mass, r_g, inertia)
    s = module.skew(r_g)
    assert np.array_equal(ref[:3, 3:], -mass * s)
    assert np.array_equal(ref[3:, :3], -mass * s.T)
    assert np.array_equal(ref, ref.T)
    # the transposed variant is symmetric too: symmetry alone cannot see it
    control = module.legacy_coupling_matrix(mass, r_g, inertia)
    assert np.array_equal(control, control.T)
    assert np.abs(control - ref).max() == pytest.approx(2.0 * mass * abs(s).max(), rel=1e-12)


def test_coupling_sign_is_pinned_by_the_velocity_field() -> None:
    """v_g = v_o + w x r_g gives M[0,4] = +m*z_g for a body with r_g = (0,0,z_g)."""
    mass, z_g = 2.1e7, -2.8148148148148148
    matrix = compute_mass_matrix(mass, np.array([0.0, 0.0, z_g]), np.diag([1.0, 2.0, 3.0]))
    assert matrix[0, 4] == pytest.approx(mass * z_g, rel=1e-12)  # surge - pitch
    assert matrix[4, 0] == pytest.approx(mass * z_g, rel=1e-12)
    assert matrix[1, 3] == pytest.approx(-mass * z_g, rel=1e-12)  # sway - roll


# --------------------------------------------------------------------------
# the shipped matrix against the reference
# --------------------------------------------------------------------------
def test_all_36_entries_within_three_percent(study: dict) -> None:
    for level in study["levels"]:
        errors = np.asarray(level["rel_error_matrix"])
        assert errors.shape == (6, 6)
        assert errors.max() < TOL, (level["subdivisions"], errors.max())
        assert np.isfinite(errors).all()
    assert study["passed"] is True


def test_mesh_route_reproduces_the_analytic_mass_properties(study: dict) -> None:
    """Geometry/mass-property check before any tolerance judgement."""
    for level in study["levels"]:
        g = level["geometry_agreement"]
        assert g["volume_rel_error"] < 1e-12
        assert g["mass_rel_error"] < 1e-12
        assert g["cg_abs_error"] < 1e-10
        assert g["inertia_cg_rel_error"] < 1e-12
        # and the library consumes the *mesh* mass properties just as well
        assert level["max_rel_error_from_mesh_properties"] < TOL


def test_case_body_exercises_the_coupling_block(study: dict) -> None:
    geom = study["primary"]["analytic"]
    g = study["geometry"]
    r_g = np.asarray(g["center_of_gravity"])
    assert np.min(np.abs(r_g)) > 0.1, "the CG must be off all three axes"
    inertia = np.asarray(geom["inertia_about_cg"])
    off_diagonal = inertia - np.diag(np.diag(inertia))
    assert np.abs(off_diagonal).max() > 1e6, "the products of inertia must be non-zero"
    assert study["primary"]["coupling_block_norm"] > 1e7
    # equilibrium draft implied by the mass model stays inside the box
    assert 0.0 < g["equilibrium_draft"] <= 8.0 + 1e-12


# --------------------------------------------------------------------------
# the identity that actually constrains the sign
# --------------------------------------------------------------------------
def test_kinetic_energy_identity_holds(study: dict) -> None:
    assert study["primary"]["kinetic_energy_error"] < 1e-12 * 1.0
    for level in study["levels"]:
        assert level["kinetic_energy_error"] < 1e-12


def test_kinetic_energy_identity_rejects_the_transposed_block() -> None:
    """Negative control: the pre-fix block violates the physical identity."""
    mass, r_g = 1000.0, (1.2, -0.7, 3.1)
    inertia = np.diag([2.0e3, 3.0e3, 4.0e3])
    good = module.kinetic_energy_error(
        module.closed_form_mass_matrix(mass, r_g, inertia), mass, r_g, inertia
    )
    bad = module.kinetic_energy_error(
        module.legacy_coupling_matrix(mass, r_g, inertia), mass, r_g, inertia
    )
    assert good < 1e-12
    assert bad > 1e-2, bad


def test_negative_control_fails_the_benchmark(study: dict) -> None:
    assert study["control_passed"] is False
    assert study["primary"]["control_max_rel_error"] > TOL
    assert study["primary"]["control_max_rel_error"] == pytest.approx(2.0, rel=0.05)


def test_error_ladder_is_flat_in_subdivision(study: dict) -> None:
    """Exactness for polyhedral bodies: refinement changes nothing."""
    errors = [level["max_rel_error"] for level in study["levels"]]
    assert max(errors) < 1e-12
    panels = [level["panels"] for level in study["levels"]]
    assert panels == sorted(panels) and len(set(panels)) == len(panels)


def test_payload_ladder_scales_the_coupling_block(study: dict) -> None:
    rows = study["payload_ladder"]
    norms = [row["coupling_block_norm"] for row in rows]
    assert all(b > a for a, b in zip(norms, norms[1:])), "coupling must grow with x_g"
    assert all(row["passed"] for row in rows)
    assert not any(row["control_passed"] for row in rows)


# --------------------------------------------------------------------------
# integration caller: the seakeeping solver
# --------------------------------------------------------------------------
def test_rao_solver_consumes_the_reference_matrix(study: dict) -> None:
    rao = study["rao_integration"]
    # injecting the reference block form reproduces the shipped RAO exactly
    assert rao["replay_with_injected_ref_matrix"] < 1e-9
    # and the pre-fix block would have moved the delivered RAO numbers
    assert rao["max_rao_change"] > 0.03
    assert math.isfinite(rao["max_rao_change"])


def test_in_repo_closed_form_check_is_not_a_tautology() -> None:
    """The benchmark's own block-form comparison must use the literature sign."""
    from tensorbem.benchmarks import hydrostatic_rao_verification as verification

    out = verification.run_mass_matrix_check()
    assert out["max_abs_difference"] == pytest.approx(0.0, abs=0.0)
    # the reference it builds must differ from the transposed convention
    control = module.legacy_coupling_matrix(
        out["mass"],
        np.asarray(out["center_of_gravity"]),
        np.asarray(out["inertia_tensor_about_cg"]),
    )
    reference = module.closed_form_mass_matrix(
        out["mass"],
        np.asarray(out["center_of_gravity"]),
        np.asarray(out["inertia_tensor_about_cg"]),
    )
    assert np.abs(control - reference).max() > 0.0


# --------------------------------------------------------------------------
# entry point / artifacts
# --------------------------------------------------------------------------
def test_entry_point_writes_artifacts(tmp_path) -> None:
    code = module.main(["--out", str(tmp_path), "--subdivisions", "2", "--no-rao"])
    assert code == 0
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["anchor"] == "CASE-BEM-RIGID-BODY-MASS"
    assert results["passed"] is True
    assert results["primary"]["max_rel_error"] < TOL
    table = (tmp_path / "report.txt").read_text()
    assert "Fossen (2011)" in table, "the reference source must be printed"
    assert (tmp_path / "rigid_body_mass_error.png").stat().st_size > 10_000

# --------------------------------------------------------------------------
# the platform referee row (tensor_benchmark_cases id=57)
# --------------------------------------------------------------------------
def test_referee_ellipsoid_ladder_reproduces_the_registered_numbers(study: dict) -> None:
    """The row publishes 0.2161 % volume, 0.3552 % entry, 5.50 % at 320 panels."""
    ladder = {row["panels"]: row for row in study["ellipsoid"]["ladder"]}
    assert set(ladder) == {20, 80, 320, 1280, 5120}
    assert ladder[5120]["volume_rel_error"] == pytest.approx(0.002161, rel=0.02)
    assert ladder[5120]["max_rel_error"] == pytest.approx(0.003552, rel=0.02)
    assert ladder[320]["max_rel_error"] == pytest.approx(0.0550, rel=0.02)
    # monotone refinement, and the 3 % criterion met from 1280 panels up
    errors = [ladder[n]["max_rel_error"] for n in (20, 80, 320, 1280, 5120)]
    assert errors == sorted(errors, reverse=True)
    assert ladder[1280]["passed"] and ladder[5120]["passed"]
    assert not ladder[320]["passed"], "the 320-panel rung is a disclosed sub-criterion FAIL"
    # and the pre-fix coupling block fails at every panel count
    assert not any(row["control_passed"] for row in study["ellipsoid"]["ladder"])
