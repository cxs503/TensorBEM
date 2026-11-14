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

# ---------------------------------------------------------------------------
# Round 4: the closure band (extended tails + a local resonance cluster) and
# the second Kramers-Kronig direction.  These tests never run a panel solve:
# they pin the *grid design* and the *estimator's* behaviour on analytic data,
# which is what decides whether G3/G4 can answer at all.
# ---------------------------------------------------------------------------

RESONANCE_MEDIUM = 7.378504598323515  # round-2 medium level, 400 quads


def test_closure_grid_spans_both_kramers_kronig_tails():
    """The closure band must contain the round-2 nodes and both omitted tails."""
    grid = case.closure_grid()
    assert list(grid) == sorted(grid)
    assert grid[0] == 0.10 and grid[-1] == 20.0
    # every predeclared node survives, so the narrow-band A/B control is a
    # strict subset of the same run (no re-solve, no interpolation)
    for w in case.OMEGA_GRID:
        assert round(w, 9) in {round(v, 9) for v in grid}
    for w in case.EXTENDED_GRID:
        assert round(w, 9) in {round(v, 9) for v in grid}
    assert 1.0 in grid  # G1's reference frequency


def test_closure_grid_cluster_resolves_the_half_power_width():
    """G4 needs >= 5 samples per half-power width (2*zeta*omega_n ~ 0.074 rad/s)."""
    zeta_declared = 0.005  # round-2 free-decay estimate, the design assumption
    width = 2.0 * zeta_declared * RESONANCE_MEDIUM
    grid = np.asarray(case.closure_grid(RESONANCE_MEDIUM), dtype=float)
    inside = grid[np.abs(grid - RESONANCE_MEDIUM) <= case.RESONANCE_HALF_WIDTH + 1e-9]
    assert inside.size >= 40
    assert np.max(np.diff(inside)) <= 2.0 * case.RESONANCE_STEP + 1e-12
    assert width / case.RESONANCE_STEP >= 5.0


def test_closure_grid_rejects_a_missing_resonance():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            case.closure_grid(bad)


def test_kk_directions_agree_on_an_exact_causal_pair():
    """Both Hilbert directions must be small on the exact pair (midpoint error only).

    The residual is evaluated at grid midpoints against the mean of the bracketing
    nodes, so it carries an O(h^2) interpolation term of its own, and the damping
    direction additionally inherits ``kkr_damping``'s tail fit.  On a 400-node grid
    over five decades the measured floor is ~1e-3 (A from B) and ~6e-2 (B from A);
    both are asserted below the 10 % gate so the estimator is demonstrably good
    enough to judge the case, and the floor is documented rather than hidden.
    """
    omegas = np.geomspace(0.01, 12.5, 400)
    _, added, damping = exact_pair(omegas=omegas)
    directions = case.kk_directions(sweep_of(omegas, added, damping), 2)
    assert directions["added_from_damping"]["max"] < 0.02
    assert directions["damping_from_added"]["max"] < 0.10
    exact_floor = directions["damping_from_added"]["max"]


def test_kk_directions_have_power_against_a_ten_percent_inconsistency():
    """A pair built from two 10 %-different kernels must read ~10 % (not vacuous).

    ``A`` comes from ``c = 2.0`` and ``B`` from ``c = 2.2``, i.e. the two halves of
    the Hilbert pair disagree by 10 % in amplitude by construction.
    """
    omegas = np.geomspace(0.01, 12.5, 400)
    _, added, _ = exact_pair(c=2.0, omegas=omegas)
    _, _, damping = exact_pair(c=2.2, omegas=omegas)
    directions = case.kk_directions(sweep_of(omegas, added, damping), 2)
    assert 0.08 <= directions["added_from_damping"]["max"] <= 0.12
    # the damping direction carries a ~6 % discretisation/tail-fit floor of its own
    # (see the exact-pair test), so the injected 10 % must still show up on top of
    # it: the reading has to grow by the injected amount, not by a rounding.
    assert 0.12 <= directions["damping_from_added"]["max"] <= 0.18


def _lorentzian_sweep(omega, *, zeta=0.005, resonance=RESONANCE_MEDIUM):
    """Analytic heave-like sweep: constant added mass, Lorentzian damping peak.

    The peak width is ``2 zeta omega_n``; the added mass is chosen so the
    undamped resonance sits at ``resonance``.  The model is analytic, so the
    *same* estimator applied to a very fine sampling is the reference.
    """
    ref = case.__dict__["wigley3_hydrostatic_reference"]()
    mass = float(ref["volume"]) * case.DENSITY
    restoring = float(ref["C33"])
    added = restoring / resonance**2 - mass
    gamma = zeta * resonance
    peak = 2.0 * zeta * (mass + added) * resonance
    omega = np.asarray(omega, dtype=float)
    damping = peak * gamma**2 / ((omega - resonance) ** 2 + gamma**2) + 1e-3
    return mass, restoring, np.full_like(omega, added), damping


def test_half_power_zeta_is_decidable_on_the_closure_grid_and_not_on_a_coarse_one():
    """Round-3's G4 defect, made into a controlled experiment.

    The same estimator is handed the same analytic resonance sampled two ways:
    on the closure grid (0.0125 rad/s across the peak) and on a 0.5 rad/s global
    grid.  The closure grid reproduces the fine-sampling reference to under 3 %;
    the coarse grid is off by tens of percent *silently* -- which is exactly why
    G4 was undecidable on the round-2 band and why the cluster is part of the
    band design rather than a convenience.
    """
    fine = np.linspace(RESONANCE_MEDIUM - 0.5, RESONANCE_MEDIUM + 0.5, 20001)
    mass, restoring, added_f, damping_f = _lorentzian_sweep(fine)
    reference = td.half_power_damping(fine, added_f, damping_f,
                                      mass=mass, restoring=restoring)
    zeta_ref = float(reference["zeta"])
    assert zeta_ref > 0.0

    closure = np.asarray(case.closure_grid(RESONANCE_MEDIUM), dtype=float)
    coarse = np.array([0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5, 7.0,
                       7.4, 7.5, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0])

    def zeta_on(grid):
        _, _, added, damping = _lorentzian_sweep(grid)
        dense = case.decay_response_grid(grid)
        return float(td.half_power_damping(
            dense, np.interp(dense, grid, added), np.interp(dense, grid, damping),
            mass=mass, restoring=restoring)["zeta"])

    assert abs(zeta_on(closure) - zeta_ref) / zeta_ref < 0.03
    assert abs(zeta_on(coarse) - zeta_ref) / zeta_ref > 0.20


def test_decay_response_grid_is_fine_enough_for_the_peak():
    grid = np.asarray(case.closure_grid(RESONANCE_MEDIUM), dtype=float)
    dense = case.decay_response_grid(grid)
    assert np.max(np.diff(dense)) <= case.DENSE_STEP + 1e-12
    assert dense[0] == grid[0] and dense[-1] == grid[-1]
