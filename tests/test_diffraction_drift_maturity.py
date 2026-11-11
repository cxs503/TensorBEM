"""Tests for :mod:`tensorbem.diffraction_drift_maturity` and its study driver.

The suite is split in two layers:

* **convention / algebra** tests (cheap, no BEM solve): the closed-form Wigley
  reference integrals, the 2-DOF heave/pitch motion solve, the exact
  MacCamy-Fuchs near/far-field pair, the ``(ka)^3`` law, the Kochin amplitude
  convention that bridges :mod:`tensorbem.cylinder_drift` and
  :mod:`tensorbem.drift_far_field3d`, and the mesh builders.
* **solver** tests (a few seconds each, deliberately tiny meshes): the
  three-route exciting force, the near/far-field damping identity and the
  optical theorem on a real ``solve_diffraction_3d`` run, plus the BEM drift
  force of a discretised cylinder against the exact value.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import cylinder_drift as cd
from tensorbem import diffraction_drift_maturity as m
from tensorbem.benchmarks import bem_diffraction_drift_maturity as driver
from tensorbem.drift_far_field3d import (
    maruo_far_field_drift,
    optical_theorem_residual,
    radius_scan,
)


# ---------------------------------------------------------------------------
# 1. closed-form Wigley reference
# ---------------------------------------------------------------------------
def test_wigley_reference_block_coefficient_is_four_ninths():
    ref = m.wigley_reference()
    assert ref["Cb"] == pytest.approx(4.0 / 9.0, rel=1e-12)
    assert ref["volume"] == pytest.approx(4.0 * ref["length"] * ref["beam"] * ref["draft"] / 9.0)


def test_wigley_reference_waterplane_area():
    ref = m.wigley_reference()
    assert ref["waterplane_area"] == pytest.approx(2.0 * ref["length"] * ref["beam"] / 3.0)


def test_wigley_reference_pitch_inertia_frame_convention():
    """Origin frame: ``I55 = rho T B L^3/45``; CG frame: ``I55 - m z_G^2``."""
    ref = m.wigley_reference()
    origin = ref["length"] ** 2 / 20.0
    cg = origin - ref["z_g"] ** 2
    assert ref["pitch_radius_of_gyration"] == pytest.approx(math.sqrt(origin), rel=1e-12)
    assert ref["pitch_radius_of_gyration_cg"] == pytest.approx(math.sqrt(cg), rel=1e-12)
    assert ref["mass_block"][1, 1] == pytest.approx(ref["mass"] * origin, rel=1e-12)
    assert ref["pitch_inertia_about_origin"] - ref["pitch_inertia_about_cg"] == pytest.approx(
        ref["mass"] * ref["z_g"] ** 2, rel=1e-12)


def test_pitch_restoring_uses_the_longitudinal_metacentric_height():
    """``C55 = rho g nabla GM_L``; ``wigley_hydrostatic_reference`` labels C44/C55 swapped.

    The roll/pitch convention of ``compute_hydrostatic_restoring_matrix`` (pinned
    by ``tests/test_wigley_hydrostatics.py``) puts ``GM_T`` on the roll entry and
    ``GM_L`` on the pitch entry, while ``wigley_hydrostatic_reference`` returns
    ``rho g nabla GM_L`` under the key ``C44``.  Using its ``C55`` key for pitch
    understates the restoring by ``GM_L/GM_T ~ 56`` and inflates the pitch RAO by
    the same factor (measured before the fix: 63.5 instead of 0.99).
    """
    ref = m.wigley_reference()
    hydro = ref["hydrostatic_reference"]
    gamma = ref["density"] * ref["gravity"] * ref["volume"]
    assert ref["restoring_block"][1, 1] == pytest.approx(gamma * hydro["GM_L"], rel=1e-12)
    assert ref["restoring_block"][1, 1] != pytest.approx(gamma * hydro["C55"], rel=1e-3)
    assert hydro["GM_L"] / hydro["GM_T"] > 10.0


def test_wigley_reference_mass_and_restoring_positive():
    ref = m.wigley_reference()
    assert ref["mass"] == pytest.approx(ref["density"] * ref["volume"])
    assert ref["mass_block"][0, 0] > 0.0
    assert ref["mass_block"][1, 1] > 0.0
    assert ref["restoring_block"][0, 0] > 0.0
    assert ref["restoring_block"][1, 1] > 0.0
    # no inertia coupling for x_G = y_G = 0
    assert ref["mass_block"][0, 1] == 0.0


# ---------------------------------------------------------------------------
# 2. geometry precondition gate
# ---------------------------------------------------------------------------
def test_geometry_gate_reports_measured_errors():
    ref = m.wigley_reference()
    mesh = m.wigley_mesh(8, 4)
    gate = m.geometry_gate(mesh, ref)
    assert gate["panel_count"] == mesh.panel_count
    assert 0.0 < gate["Cb_rel_error"] < 0.05
    assert 0.0 < gate["waterplane_rel_error"] < 0.05


def test_geometry_gate_passes_on_the_study_mesh_with_default_tolerance():
    """The 32x16 study mesh (2048 panels) meets the documented 0.2 % tolerance."""
    ref = m.wigley_reference()
    mesh = m.wigley_mesh(32, 16)
    gate = m.geometry_gate(mesh, ref)
    assert mesh.panel_count == 2048
    assert gate["Cb_rel_error"] < 2.0e-3
    assert gate["passed"] is True


def test_geometry_gate_passes_with_loose_tolerance_and_fails_with_tight():
    ref = m.wigley_reference()
    mesh = m.wigley_mesh(8, 4)
    loose = m.geometry_gate(mesh, ref, cb_tolerance=0.05, volume_tolerance=0.05)
    tight = m.geometry_gate(mesh, ref, cb_tolerance=1.0e-6, volume_tolerance=1.0e-6)
    assert loose["passed"] is True
    assert tight["passed"] is False


# ---------------------------------------------------------------------------
# 3. 2-DOF heave/pitch motion solve
# ---------------------------------------------------------------------------
def test_rao_2dof_matches_direct_linear_solve():
    omega = 1.7
    added = np.array([[3.0, -0.4], [-0.4, 5.0]])
    damping = np.array([[0.5, 0.1], [0.1, 0.9]])
    mass = np.array([[10.0, 0.0], [0.0, 4.0]])
    restoring = np.array([[900.0, 0.0], [0.0, 120.0]])
    excitation = np.array([7.0 - 1.0j, -2.0 + 3.0j])
    out = m.rao_2dof(omega, added, damping, mass, restoring, excitation)
    matrix = (
        -omega**2 * (mass + added) + 1.0j * omega * damping + restoring
    )
    expected = np.linalg.solve(matrix, excitation)
    assert np.allclose(out["displacement"], expected)
    assert out["system_residual"] < 1.0e-12


def test_rao_2dof_single_dof_limit():
    """Zero coupling -> the two modes must decouple into ``F / (C - w^2 M)``."""
    omega = 0.8
    added = np.zeros((2, 2))
    damping = np.zeros((2, 2))
    mass = np.diag([2.0, 5.0])
    restoring = np.diag([3.0, 4.0])
    excitation = np.array([1.0 + 0.0j, 2.0 + 0.0j])
    out = m.rao_2dof(omega, added, damping, mass, restoring, excitation)
    heave = 1.0 / (3.0 - omega**2 * 2.0)
    pitch = 2.0 / (4.0 - omega**2 * 5.0)
    assert out["displacement"][0].real == pytest.approx(heave, rel=1e-12)
    assert out["displacement"][1].real == pytest.approx(pitch, rel=1e-12)


def test_rao_2dof_zero_excitation_is_zero():
    out = m.rao_2dof(
        1.0, np.zeros((2, 2)), np.zeros((2, 2)), np.eye(2) * 3.0,
        np.diag([5.0, 7.0]), np.zeros(2, dtype=complex),
    )
    assert out["heave_rao"] == 0.0
    assert out["pitch_rao_per_ka"] == 0.0


def test_rao_2dof_low_frequency_heave_limit_is_one():
    """Static limit ``xi_3 = F_3 / C_33`` with ``F_3 -> rho g A_wp A`` gives 1."""
    ref = m.wigley_reference()
    c33 = ref["restoring_block"][0, 0]
    excitation = np.array([c33 + 0.0j, 0.0 + 0.0j])
    out = m.rao_2dof(
        1.0e-4, np.zeros((2, 2)), np.zeros((2, 2)), ref["mass_block"],
        ref["restoring_block"], excitation,
    )
    assert out["heave_rao"] == pytest.approx(1.0, rel=1e-6)


def test_rao_2dof_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        m.rao_2dof(1.0, np.zeros((3, 3)), np.zeros((2, 2)), np.eye(2),
                   np.eye(2), np.zeros(2, dtype=complex))
    with pytest.raises(ValueError):
        m.rao_2dof(1.0, np.zeros((2, 2)), np.zeros((2, 2)), np.eye(2),
                   np.eye(2), np.zeros(3, dtype=complex))


def test_rao_2dof_rejects_nonpositive_omega():
    with pytest.raises(ValueError):
        m.rao_2dof(0.0, np.zeros((2, 2)), np.zeros((2, 2)), np.eye(2),
                   np.eye(2), np.ones(2))


# ---------------------------------------------------------------------------
# 4. verdict helper: not-evaluable is never a pass
# ---------------------------------------------------------------------------
def test_gate_not_evaluable_is_not_a_pass():
    verdict = m.gate("G", "t", "thr", None, None).as_dict()
    assert verdict["passed"] is False
    assert verdict["evaluable"] is False
    assert verdict["value"] is None
    good = m.gate("G", "t", "thr", 0.5, True).as_dict()
    assert good["passed"] is True and good["evaluable"] is True


# ---------------------------------------------------------------------------
# 5. exact cylinder reference (MacCamy-Fuchs)
# ---------------------------------------------------------------------------
def test_cylinder_exact_near_and_far_field_agree():
    gate = m.cylinder_asymptotic_gate()
    assert gate["max_near_vs_far_rel_diff"] < 1.0e-9


def test_cylinder_long_wave_slope():
    gate = m.cylinder_asymptotic_gate()
    assert abs(gate["law_slope"] - 3.0) <= 0.2


def test_cylinder_long_wave_coefficient():
    gate = m.cylinder_asymptotic_gate()
    assert gate["closed_form_coefficient"] == pytest.approx(5.0 * math.pi**2 / 16.0)
    assert gate["coefficient_error_at_ka_min"] < 0.03


# ---------------------------------------------------------------------------
# 6. Kochin amplitude convention between cylinder_drift and drift_far_field3d
# ---------------------------------------------------------------------------
def test_maruo_reduced_kernel_amplitude_convention():
    """``F_exact = (g A^2 / k) F_maruo(K_unit)`` for the unit-potential Kochin.

    ``cylinder_drift.kochin_3d`` is built from the *dimensionless* scattering
    amplitude, i.e. it omits the potential amplitude factor ``g A / omega`` of
    :func:`tensorbem.free_surface_solver3d.incident_airy_wave_3d`.  Since that
    factor enters the force quadratically, feeding the unit-potential Kochin to
    :func:`tensorbem.drift_far_field3d.maruo_far_field_drift` understates the
    drift force by ``(g / omega)^2 = g / k``.  Derived in
    ``docs/diffraction_drift_maturity.md`` §5; checked here so no downstream
    user has to rediscover it.
    """
    gravity = cd.DEFAULT_GRAVITY
    for ka in (0.5, 1.0, 2.0):
        k = ka
        theta = np.linspace(0.0, 2.0 * math.pi, 1440, endpoint=False)
        kochin_unit = cd.kochin_3d(ka, theta)
        maruo = maruo_far_field_drift(kochin_unit, theta, k, n_theta=1440)["force_x"]
        exact = cd.far_field_cylinder_drift(ka)["mean_drift_force_x"]
        assert exact == pytest.approx(gravity / k * maruo, rel=1e-9)


def test_optical_theorem_uses_imaginary_forward_amplitude_for_this_convention():
    """The 2-D scattering convention satisfies the identity with ``Im K(0)``.

    ``optical_theorem_residual`` states the identity with ``Re K(0)``, which is
    the phase convention of the BEM Kochin function.  A Kochin built from the
    scattering amplitude of :func:`tensorbem.cylinder_drift.kochin_2d` carries a
    ``-i`` and must be contracted with ``Im K(0)``; using ``Re`` there gives a
    large residual for a field that is exact.
    """
    gravity = cd.DEFAULT_GRAVITY
    ka = 1.0
    k = ka
    n_theta = 2880
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    amplitude = 1.0
    omega = math.sqrt(gravity * k)
    kochin_physical = (gravity * amplitude / omega) * cd.kochin_3d(ka, theta)
    integral = float(np.sum(np.abs(kochin_physical) ** 2) * (2.0 * math.pi / n_theta))
    forward = float(np.imag(kochin_physical[0]))
    predicted = (4.0 * math.pi / k) * (gravity * amplitude / omega) * forward
    assert abs(integral - predicted) / abs(integral) < 1.0e-6
    # the Re-based residual of the module is NOT small for this convention
    wrong = optical_theorem_residual(kochin_physical, theta, k, radius=1.0)
    assert wrong["residual_relative"] > 0.05


def test_radius_scan_reports_non_convergence_for_finite_radius_route():
    """The finite-radius radiation-stress route keeps an O(1) residual (m9286)."""
    theta = np.linspace(0.0, 2.0 * math.pi, 1440, endpoint=False)
    kochin = cd.kochin_3d(1.0, theta)
    scan = radius_scan(kochin, theta, 1.0, radii=(5.0, 10.0, 20.0))
    assert len(scan["rows"]) == 3
    assert scan["spread_relative"] > 0.5  # documented non-convergence, not a gate
    assert scan["richardson_evaluable"] is False


# ---------------------------------------------------------------------------
# 7. cylinder mesh builder
# ---------------------------------------------------------------------------
def test_vertical_cylinder_mesh_waterplane_is_the_inscribed_polygon():
    n_theta = 48
    mesh = m.vertical_cylinder_hull_mesh(1.0, 8.0, n_theta=n_theta, n_levels=3)
    polygon = 0.5 * n_theta * math.sin(2.0 * math.pi / n_theta)
    assert mesh.waterplane_area == pytest.approx(polygon, rel=1e-12)


def test_vertical_cylinder_mesh_volume():
    n_theta = 48
    mesh = m.vertical_cylinder_hull_mesh(1.0, 8.0, n_theta=n_theta, n_levels=3)
    polygon = 0.5 * n_theta * math.sin(2.0 * math.pi / n_theta)
    assert mesh.volume == pytest.approx(polygon * 8.0, rel=1e-12)


def test_vertical_cylinder_mesh_panel_count_and_grading():
    mesh = m.vertical_cylinder_hull_mesh(1.0, 8.0, n_theta=32, n_levels=4)
    assert mesh.panel_count == 32 * (2 * 4 + 1)
    # rings are clustered towards the free surface
    z = np.unique(np.round(mesh.vertices[:, 2], 12))
    gaps = -np.diff(z[z <= 0.0])
    assert gaps[0] < gaps[-1]


def test_vertical_cylinder_mesh_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        m.vertical_cylinder_hull_mesh(0.0, 8.0)
    with pytest.raises(ValueError):
        m.vertical_cylinder_hull_mesh(1.0, 0.0)
    with pytest.raises(ValueError):
        m.vertical_cylinder_hull_mesh(1.0, 8.0, n_theta=4, n_levels=2)


# ---------------------------------------------------------------------------
# 8. solver-level checks (tiny meshes)
# ---------------------------------------------------------------------------
def test_wigley_head_seas_case_three_routes_agree():
    ref = m.wigley_reference()
    mesh = m.wigley_mesh(8, 4)
    nu = 2.4**2 / ref["gravity"]
    case = m.wigley_head_seas_case(nu, ref, mesh, n_azimuth=48)
    routes = case["excitation_routes"]
    assert max(routes["rel_near_vs_boundary"]) < 0.05
    assert max(routes["rel_near_vs_kochin"]) < 0.05
    assert case["energy"]["damping_rel_diff"] < 0.05
    assert case["energy"]["optical_theorem"]["residual_relative"] < 0.02
    assert case["rao"]["system_residual"] < 1.0e-10
    assert 0.0 < case["rao"]["heave_rao"] < 3.0


def test_wigley_heave_rao_long_wave_limit_is_near_one():
    ref = m.wigley_reference()
    mesh = m.wigley_mesh(8, 4)
    nu = 1.2**2 / ref["gravity"]
    case = m.wigley_head_seas_case(nu, ref, mesh, n_azimuth=48)
    assert case["rao"]["heave_rao"] == pytest.approx(1.0, abs=0.10)
    # long waves: the hull follows the wave slope, so pitch/(kA) -> 1 as well
    assert case["rao"]["pitch_rao_per_ka"] == pytest.approx(1.0, abs=0.20)


def test_cylinder_bem_drift_matches_exact_within_coarse_mesh_tolerance():
    """Deliberately tiny mesh: 264 panels give ~16 % error on the drift force.

    The study itself runs 912 panels (48 azimuth rings x 9 graded levels) and
    lands at ~5 %; this test only pins the *direction* and the order of
    magnitude so a regression that breaks the far-field coupling is caught
    without spending minutes of CPU (see ``docs/diffraction_drift_maturity.md``
    for the panel-convergence table).
    """
    case = m.cylinder_bem_case(1.0, panels=24, n_levels=5, n_azimuth=64,
                               radii=(5.0, 10.0, 20.0))
    assert case["panel_count"] == 24 * (2 * 5 + 1)
    assert case["exact_near_vs_far_rel_error"] < 1.0e-9
    assert 0.0 < case["bem_vs_exact_rel_error"] < 0.25
    assert case["optical_theorem"]["residual_relative"] < 0.10

# ---------------------------------------------------------------------------
# 9. study driver: artifact merge and gate recomputation
# ---------------------------------------------------------------------------
def test_driver_merge_recomputes_gates_from_committed_shards():
    """The committed per-frequency shards must merge into a consistent study.

    This doubles as the capability-evidence reference for the driver module
    (``tensorbem.capability_evidence`` treats a module as covered when a test
    references it) and as a regression pin on ``runs/m14571/study.json``.
    """
    import json
    from pathlib import Path

    shards = sorted(Path("runs/m14571").glob("shard_w*.json"))
    assert len(shards) == 6, f"expected six committed frequency shards, got {shards}"
    study = driver.merge_shards([str(p) for p in shards])
    assert len(study["wigley"]["rows"]) == 6
    omegas = [row["omega"] for row in study["wigley"]["rows"]]
    assert omegas == sorted(omegas)
    by_gate = {v["gate"]: v for v in study["verdicts"]}
    assert by_gate["G2_excitation"]["passed"] is True
    assert by_gate["G4_damping_energy"]["passed"] is True
    assert by_gate["G5_optical"]["passed"] is True
    # the experiment comparison is honestly not evaluable, never a silent pass
    assert by_gate["G10_wigley_vs_experiment"]["evaluable"] is False
    # strict JSON payload (no NaN / complex leakage)
    payload = json.dumps(driver._jsonable(study))
    assert "NaN" not in payload and "Infinity" not in payload
    assert json.loads(payload)["all_passed"] is True
