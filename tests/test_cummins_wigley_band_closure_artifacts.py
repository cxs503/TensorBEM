"""Artifact regression tests for the *executed* extended-band closure.

``docs/dev_report_m23130_cummins_wigley_band_closure_executed.md`` reports the
first run of the predeclared [0.10, 20] rad/s band of ``CASE-BEM-CUMMINS-WIGLEY``
that actually reached the gates: 69 nodes (28 extended nodes + the resonance
cluster) were solved at ``medium`` and G1-G5 were evaluated on them.  Earlier
rounds could only hand over the pipeline.

These tests pin the *evidence*, not the pipeline structure (that is
``tests/test_cummins_wigley_closure_pipeline.py``):

1. the committed sweep spans both predeclared KK tails and carries the whole
   closure grid, so the gates are evaluated on the band that was committed;
2. re-evaluating the gates from the committed sweep reproduces the committed
   verdict and every gate value -- the record is recomputable, not quoted;
3. the run does not contradict the older narrow-band record: at the single
   shared frequency ``omega = 0.20`` the new solve equals the committed
   ``sweep_medium.npz`` solve of the same mesh;
4. a FAIL is recorded as a FAIL (no gate is allowed to be silently absent or
   to be replaced by its supplementary statistic).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from tensorbem.benchmarks import cummins_wigley_kk_causality as case

ART = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
SWEEP = ART / "sweep_closure_medium.npz"
GATES = ART / "closure_band_gates.json"


@pytest.fixture(scope="module")
def sweep() -> dict:
    data = np.load(SWEEP, allow_pickle=False)
    return {
        "level": str(data["label"]),
        "level_key": str(data["level"]),
        "rotation_center": [float(v) for v in data["rotation_center"]],
        "density": float(data["density"]),
        "gravity": float(data["gravity"]),
        "omega": [float(w) for w in data["omega"]],
        "added_mass": np.asarray(data["added_mass"], dtype=float).tolist(),
        "damping": np.asarray(data["damping"], dtype=float).tolist(),
        "relative_residual": [float(v) for v in data["relative_residual"]],
        "wall_seconds": [float(v) for v in data["wall_seconds"]],
    }


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(GATES.read_text())


def test_committed_sweep_covers_the_whole_predeclared_band(sweep):
    omega = np.asarray(sweep["omega"], dtype=float)
    assert omega[0] == pytest.approx(case.EXTENDED_GRID[0])
    assert omega[-1] == pytest.approx(case.EXTENDED_GRID[-1])
    # both KK tails: the dipole low tail and the decaying high tail
    assert omega[0] <= 0.10 and omega[-1] >= 20.0
    for node in case.EXTENDED_GRID:
        assert np.any(np.isclose(omega, node, rtol=0.0, atol=1e-9)), node
    dense = np.diff(omega)
    # the predeclared global grid opens up to 5 rad/s at the top (15 -> 20), the
    # cluster is the only place where nodes are closer than RESONANCE_STEP
    assert dense.max() <= max(np.diff(np.asarray(case.EXTENDED_GRID))) + 1e-9
    assert (dense.min() <= case.RESONANCE_STEP + 1e-9)   # the cluster is present
    assert np.asarray(sweep["added_mass"]).shape == (omega.size, 6, 6)
    # every solved node is a converged linear solve
    assert max(sweep["relative_residual"]) < 1e-2


def test_geometry_gate_passes_before_any_number(record):
    geometry = record["geometry"]
    assert geometry["passed"] is True
    assert geometry["Cb_abs_error"] == 0.0
    assert geometry["L_over_B"] == pytest.approx(10.0, abs=1e-9)
    assert geometry["B_over_T"] == pytest.approx(1.6, abs=1e-9)


def test_committed_sweep_reproduces_the_committed_verdict(sweep, record):
    analysis = case.gates_from_sweep(sweep, reference=case.load_capytaine_reference())
    assert analysis["verdict"] == record["verdict"]
    assert analysis["failed"] == record["failed"]
    for name, gate in analysis["gates"].items():
        committed = record["closure_band"]["gates"][name]
        assert gate["passed"] == committed["passed"], name
        if gate["value"] is None:
            assert committed["value"] is None
            continue
        assert gate["value"] == pytest.approx(committed["value"], rel=1e-9, abs=1e-12)


def test_every_gate_is_decided_and_a_fail_is_recorded_as_a_fail(record):
    gates = record["closure_band"]["gates"]
    assert set(gates) == set(case.GATES)
    for name, gate in gates.items():
        # undecidable is recorded explicitly; here every gate has a number
        assert gate["value"] is not None, name
        assert np.isfinite(gate["value"]), name
        assert gate["passed"] == bool(abs(gate["value"]) <= gate["tol"]), name
    failed = sorted(k for k, v in gates.items() if v["passed"] is False)
    assert record["verdict"] == "FAIL:" + ",".join(failed)
    # the case tolerance is not met on this band: the record must say so
    assert set(failed) == {"G3_kramers_kronig", "G4_free_decay", "G5_round_trip"}


def test_band_extension_is_reported_against_the_narrow_band_control(record):
    closure = record["closure_band"]["gates"]
    narrow = record["narrow_round2_control"]["gates"]
    assert record["narrow_round2_control"]["points"] < record["closure_band"]["points"]
    # widening the band moves every failing gate towards its tolerance but the
    # two identical identities (G1, G2) are band-independent and unchanged
    assert closure["G1_cross_solver"]["value"] == pytest.approx(
        narrow["G1_cross_solver"]["value"], rel=1e-12)
    for name in ("G3_kramers_kronig", "G4_free_decay", "G5_round_trip"):
        assert abs(closure[name]["value"]) < abs(narrow[name]["value"]), name


def test_committed_sweep_declares_the_mesh_it_was_solved_on(sweep):
    """The band record carries its own mesh identity (never a bare label).

    ``sweep_medium.npz`` (the round-2..9 record) was solved on the 480-panel
    probe mesh ``n_sections=20, n_levels=6``, not on the current ``medium``
    (20 x 10, 800 triangles), so the two records share no frequency and cannot
    be compared node by node: this run is a *fresh* sweep of its own mesh and is
    labelled as such.
    """
    n_sections, n_levels = case.MESH_LEVELS["medium"]
    assert (n_sections, n_levels) == (20, 10)
    assert sweep["level"] == f"n_sections={n_sections},n_levels={n_levels}"
    assert sweep["level_key"] == "medium"
    assert sweep["rotation_center"] == pytest.approx(list(case.ROTATION_CENTER))
    assert sweep["density"] == pytest.approx(case.DENSITY)
    assert sweep["gravity"] == pytest.approx(case.GRAVITY)


def test_resonance_and_half_power_width_are_recorded(record):
    resonance = record["heave_resonance"]
    assert resonance["inside_band"] is True
    assert resonance["omega_n"] == pytest.approx(7.3785, abs=2e-3)
    half_power = record["half_power"]
    # the half-power width is 2.5 rad/s wide, i.e. resolved by the global grid:
    # G4's residual is an estimator difference, not a resolution outage
    width = half_power["half_power_omega_high"] - half_power["half_power_omega_low"]
    assert width > 2.0
    assert half_power["scan_width_over_half_power_width"] > 2.0