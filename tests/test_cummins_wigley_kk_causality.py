"""Tests for the Cummins / Kramers-Kronig Wigley III case (anchor CASE-BEM-CUMMINS-WIGLEY).

The suite is deliberately cheap: it never runs a panel solve except the one
coarse heave/roll pair that pins the rotation-centre convention (audit finding
F1), which is the failure the round-1 attempt died of.  Everything else is
tested on an *exact* causal pair, where the correct answer is known in closed
form, so a red test means the gate is wrong -- not the physics.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import time_domain as td
from tensorbem.benchmarks import cummins_wigley_kk_causality as case


def exact_pair(a: float = 1.2, c: float = 2.0, omegas=None) -> tuple:
    """Exact causal (A, B) pair of the exponential kernel ``K = c exp(-a t)``.

    ``B = c a / (a^2 + w^2)`` and ``A - A_inf = -c / (a^2 + w^2)`` satisfy both
    Kramers-Kronig relations identically, so any residual reported for this
    sweep is a defect of the gate, not of the physics.  The default grid spans
    five decades because the principal-value integral then has no band edge
    worth arguing about; :func:`test_kramers_kronig_residual_concentrates_at_the_band_edge`
    shows what happens when it does.
    """
    omegas = np.asarray(
        omegas if omegas is not None else np.geomspace(0.01, 12.5, 96), dtype=float
    )
    b = c * a / (a * a + omegas**2)
    a_added = 5.0 - c / (a * a + omegas**2)
    return omegas, a_added, b


def sweep_of(omegas, added, damping) -> dict:
    """Wrap a diagonal (A, B) pair as a synthetic one-DOF-per-mode sweep."""
    om = np.asarray(omegas, dtype=float)
    out = {
        "level": "analytic",
        "rotation_center": list(case.ROTATION_CENTER),
        "omega": om.tolist(),
        "added_mass": np.tile(np.eye(6), (om.size, 1, 1)).tolist(),
        "damping": np.zeros((om.size, 6, 6)).tolist(),
    }
    for i in range(om.size):
        out["added_mass"][i][2][2] = float(added[i])
        out["damping"][i][2][2] = float(damping[i])
    return out


def test_geometry_gate_passes_before_any_number():
    gate = case.geometry_gate()
    assert gate["passed"]
    assert gate["Cb_abs_error"] <= 1e-9
    assert abs(gate["L_over_B"] - 10.0) <= 1e-12
    assert abs(gate["B_over_T"] - 1.6) <= 1e-12
    assert gate["rotation_center"] == [0.0, 0.0, -case.WIGLEY3_T / 2.0]


def test_rotation_center_convention_is_pinned(monkeypatch):
    """Audit F1: roll/pitch must be taken about mid-draft, not the waterline.

    Two independent pins, because this is the failure the round-1 attempt died
    of:

    * *forwarding* -- :func:`case.solve_point` must hand the solver
      ``center = ROTATION_CENTER`` (checked on the recorded kwargs, so a
      refactor that drops the argument is red without paying for a solve);
    * *kinematics* -- the generalised normals ``(n, (x - center) x n)`` are
      identical for surge/sway/heave/yaw under either centre and differ for
      roll/pitch, which is why only those two move.
    """
    recorded = {}
    real = case.solve_radiation_3d

    def spy(mesh, nu, **kwargs):
        recorded.update(kwargs)
        return real(mesh, nu, **kwargs)

    monkeypatch.setattr(case, "solve_radiation_3d", spy)
    payload = case.solve_point("coarse", 1.0)
    assert payload["rotation_center"] == [0.0, 0.0, -case.WIGLEY3_T / 2.0]
    assert recorded["center"] == case.ROTATION_CENTER

    from tensorbem.free_surface_solver3d import rigid_body_normals_3d

    mesh = case.wetted_mesh("coarse")
    fixed = rigid_body_normals_3d(mesh, center=case.ROTATION_CENTER)
    waterline = rigid_body_normals_3d(mesh, center=(0.0, 0.0, 0.0))
    for dof in (0, 1, 2, 5):  # surge, sway, heave, yaw: no moment arm
        assert np.allclose(fixed[:, dof], waterline[:, dof], atol=1e-12, rtol=0)
    for dof in (3, 4):  # roll, pitch: lever arm z - center_z enters
        assert not np.allclose(fixed[:, dof], waterline[:, dof], atol=1e-9, rtol=0)


def test_exact_causal_pair_passes_both_kramers_kronig_and_round_trip():
    omegas, added, damping = exact_pair()
    study = case.analysis_gates(sweep_of(omegas, added, damping), dof=2)
    assert study["added_mass_inf"] == pytest.approx(5.0, abs=0.02)
    assert study["G2"]["rel_err"] < 1e-10
    assert study["G3"]["rel_err"] < 1e-2, study["G3"]["rel_err"]
    assert study["G5"]["rel_err"] < case.GATES["G5_round_trip"]["tol"], study["G5"]["rel_err"]


def test_kramers_kronig_gate_bites_when_causality_is_broken():
    """Negative control: halving ``A - A_inf`` must push G3 far above tolerance."""
    omegas, added, damping = exact_pair()
    broken = 5.0 + 0.5 * (added - 5.0)
    study = case.analysis_gates(sweep_of(omegas, broken, damping), dof=2)
    assert study["G3"]["rel_err"] > case.GATES["G3_kramers_kronig"]["tol"] * 3.0


def test_kramers_kronig_residual_concentrates_at_the_band_edge():
    """The causality residual is a property of the *band*, not of the pair.

    Same exact pair, two lower band edges.  The half-line integral is truncated
    at ``omega_min`` and the tabulated value is held constant below it, so the
    error is largest at the first midpoint and grows as ``omega_min`` rises --
    the mechanism behind audit finding F2.
    """
    wide = case.analysis_gates(
        sweep_of(*exact_pair(omegas=np.geomspace(0.01, 12.5, 96))), dof=2)
    narrow = case.analysis_gates(
        sweep_of(*exact_pair(omegas=np.geomspace(0.30, 12.5, 96))), dof=2)
    assert wide["G3"]["rel_err"] < narrow["G3"]["rel_err"]
    assert int(np.argmax(narrow["G3"]["pointwise_rel_err"])) == 0


def test_band_truncation_is_what_breaks_the_round_1_gates():
    """Audit F2, as an executable statement.

    With the round-1 band ``[0.5, 6] rad/s`` the transform band stops below the
    resonance the case has to resolve (:data:`case.LEGACY_GRID`), while the
    round-2 grid reaches past it.  Compared on the same causal pair.
    """
    legacy = np.asarray(case.LEGACY_GRID, dtype=float)
    wide = np.asarray(case.OMEGA_GRID, dtype=float)
    narrow_study = case.analysis_gates(
        sweep_of(*exact_pair(a=0.35, c=3.0, omegas=legacy)), dof=2)
    wide_study = case.analysis_gates(
        sweep_of(*exact_pair(a=0.35, c=3.0, omegas=wide)), dof=2)
    assert wide_study["G3"]["rel_err"] < narrow_study["G3"]["rel_err"]
    assert wide_study["G5"]["band"][1] > narrow_study["G5"]["band"][1]
    assert legacy[-1] < 0.6 * wide[-1]


def test_gate_verdict_is_machine_decided():
    distinct = case._verdict("X", 0.05, 0.10)
    assert distinct["passed"]
    assert not case._verdict("X", 0.11, 0.10)["passed"]
    assert not case._verdict("X", float("nan"), 0.10)["passed"]
    assert not case._verdict("X", None, 0.10)["passed"]


def test_sweep_from_points_refuses_to_interpolate():
    points = {
        1.0: {"level": "coarse", "omega": 1.0, "added_mass": np.eye(6).tolist(),
              "damping": np.zeros((6, 6)).tolist(), "relative_residual": 1e-4,
              "rotation_center": list(case.ROTATION_CENTER), "density": 1025.0,
              "gravity": 9.80665, "wall_seconds": 1.0},
    }
    almost = dict(points[1.0], omega=1.0000001)
    points[1.0] = almost  # still keyed and rounded to 1.0
    sweep = case.sweep_from_points(points, [1.0])
    assert sweep["omega"] == [1.0]
    with pytest.raises(KeyError):
        case.sweep_from_points(points, [1.0, 2.0])


def test_heave_resonance_helper_reports_absence_honestly():
    """A band without a resonance must say so instead of returning a number."""
    omega = np.linspace(0.5, 2.0, 9)
    sweep = {
        "level": "analytic",
        "rotation_center": list(case.ROTATION_CENTER),
        "omega": omega.tolist(),
        "added_mass": np.tile(np.eye(6), (omega.size, 1, 1)).tolist(),
        "damping": np.zeros((omega.size, 6, 6)).tolist(),
    }
    for i, w in enumerate(omega):
        sweep["added_mass"][i][2][2] = 100.0
        sweep["damping"][i][2][2] = 1.0
    band = case.heave_resonance_band(sweep)
    # C33 > w^2 (m + A) over the whole band, so no crossing exists
    assert band["inside_band"] is False
    assert "error" in band