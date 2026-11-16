"""CI guard for the lid-referee adjudication of ``BEM-HULME-HEMISPHERE``.

The adjudication answers one question: the xval arm recorded
``ka = 2.58 / 3.0 / 4.0`` as ``referee_unresolved`` because the *referee*
(capytaine without a lid) had an interior-Dirichlet irregular frequency there.
With the waterplane closed by a lid, is the referee usable, and do the three
cells get a defined attribution?

These tests keep that record from rotting.  They are deliberately cheap: the
radiation half is re-derived from frozen artifacts (no solver), and the
excitation half is a committed machine record that is only *read* (recomputing
it needs ~1 min of capytaine solves, which belongs on the cluster, not in CI).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere"


def _load(name: str, filename: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def adjudication():
    return _load("_hulme_lid_adjudication", "referee_lid_adjudication.py")


@pytest.fixture(scope="module")
def committed() -> dict:
    return json.loads((HERE / "referee_lid_adjudication.json").read_text())


@pytest.fixture(scope="module")
def excitation() -> dict:
    return json.loads((HERE / "referee_lid_excitation.json").read_text())


# --------------------------------------------------------------------------- #
# the record is reproducible from the artifacts it cites
# --------------------------------------------------------------------------- #
def test_record_reproduces_from_frozen_artifacts(adjudication, committed):
    fresh = adjudication.adjudicate()
    assert fresh["overall"] == committed["overall"]
    assert list(fresh["gates"]) == list(committed["gates"])
    for name, gate in fresh["gates"].items():
        assert gate["verdict"] == committed["gates"][name]["verdict"], name
    assert len(fresh["rows"]) == len(committed["rows"])
    for new, old in zip(fresh["rows"], committed["rows"]):
        assert new["ka"] == old["ka"]
        assert new["verdict"] == old["verdict"]
        assert new["finest_tensorbem_rung"]["panels"] == old["finest_tensorbem_rung"]["panels"]
        assert new["lid_referee_4096"]["worst_relerr"] == pytest.approx(
            old["lid_referee_4096"]["worst_relerr"], rel=1e-12
        )
        assert new["cross_code_tensorbem_vs_lid"]["worst"] == pytest.approx(
            old["cross_code_tensorbem_vs_lid"]["worst"], rel=1e-12
        )


def test_inputs_are_hashed(committed):
    inputs = committed["provenance"]["inputs"]
    assert len(inputs) >= 5
    for name, digest in inputs.items():
        assert (HERE / name).exists(), name
        assert len(digest) == 64


# --------------------------------------------------------------------------- #
# the comparison is scored against one and the same reference series
# --------------------------------------------------------------------------- #
def test_both_arms_score_the_same_reference(committed):
    gap = committed["provenance"]["reference_worst_gap"]
    assert gap is not None and gap < 1e-9, gap


# --------------------------------------------------------------------------- #
# the thing the adjudication exists for
# --------------------------------------------------------------------------- #
def test_no_cell_is_left_unresolved(committed):
    gate = committed["gates"]["G3_no_referee_unresolved"]
    assert gate["verdict"] == "PASS"
    assert set(gate["measured"]) == {"2.58", "3.00", "4.00"}
    for verdict in gate["measured"].values():
        assert verdict not in ("referee_unresolved", "reference_disputed", "no_data")


def test_unresolved_cells_agree_across_codes(committed):
    tol = committed["constants"]["tolerance"]
    cells = [r for r in committed["rows"] if r["role"] == "unresolved_in_xval"]
    assert len(cells) == 3
    for row in cells:
        assert row["cross_code_tensorbem_vs_lid"]["worst"] <= tol, row["ka"]
        assert row["lid_referee_4096"]["worst_relerr"] <= tol, row["ka"]
        # the referee the xval arm could not use: at least one cell must show
        # that the lid actually did something, else this whole page is moot
    open_worst = max(r["open_referee_4096"]["worst_relerr"] for r in cells)
    assert open_worst > 0.5, open_worst


def test_lid_referee_accuracy_holds_but_is_never_sold_as_better(committed):
    gate = committed["gates"]["G1_lid_referee_accuracy"]
    assert gate["verdict"] == "PASS"
    assert gate["measured"] <= 0.03
    # G2 is the honest FAIL: only the 4096 rung of the lid referee is admissible
    assert committed["gates"]["G2_lid_self_converged"]["verdict"] == "FAIL"
    assert committed["gates"]["G2_lid_self_converged"]["measured"] > 0.05
    # a coarse rung that breaks the 5 % gate must be repaired by refinement
    g4 = committed["gates"]["G4_solver_gate_at_best_rung"]["measured"]
    repaired = [k for k, v in g4.items() if v["coarse_rungs_over_gate"]]
    assert repaired, "expected at least one mesh-limited cell to be recorded"
    for row in g4.values():
        assert row["best_rung"]["worst"] <= 0.05


def test_lid_does_not_disturb_in_band_physics(committed):
    gate = committed["gates"]["G5_lid_leaves_in_band_physics_alone"]
    assert gate["verdict"] == "PASS"
    assert set(gate["measured"]["per_ka"]) == {"1.00", "2.00"}


# --------------------------------------------------------------------------- #
# excitation channel: the second half of the same referee
# --------------------------------------------------------------------------- #
def test_excitation_record_is_complete(excitation):
    assert excitation["overall"] == "PASS"
    kas = {round(r["ka"], 2) for r in excitation["rows"]}
    assert {2.0, 2.58, 3.0, 3.5, 4.0} <= kas
    for row in excitation["rows"]:
        for key in ("open_referee", "lid_referee", "open_error", "lid_error"):
            assert key in row, (row["ka"], key)
        assert row["lid_referee"]["lid_panels"] > 0
        assert row["open_referee"]["lid_panels"] == 0


def test_excitation_pole_and_cure(excitation):
    tol = 0.05
    first = next(r for r in excitation["rows"] if abs(r["ka"] - 2.58) < 1e-9)
    # without the lid the exciting force is off by tens of percent at the pole
    assert first["open_error"]["Fz"] > 0.2, first["open_error"]
    # with the lid it is inside the case tolerance
    assert first["lid_error"]["Fz"] <= tol, first["lid_error"]
    # and it agrees with TensorBEM, which never saw either referee
    assert max(first["cross_code_vs_lid"].values()) <= tol


def test_excitation_gate_verdicts_are_the_recorded_ones(excitation):
    for name, gate in excitation["gates"].items():
        assert gate["verdict"] == "PASS", name


# --------------------------------------------------------------------------- #
# the prose does not get to disagree with the machine record
# --------------------------------------------------------------------------- #
def test_markdown_repeats_the_recorded_verdicts(committed):
    text = (HERE / "referee_lid_adjudication.md").read_text()
    assert "referee_unresolved" in text
    assert "honest FAIL" in text or "诚实" in text
    drift = committed["gates"]["G2_lid_self_converged"]["measured"]
    assert f"{100 * drift:.2f} %" in text
    for row in committed["rows"]:
        if row["role"] != "unresolved_in_xval":
            continue
        cross = 100 * row["cross_code_tensorbem_vs_lid"]["worst"]
        assert f"{cross:.2f} %" in text, (row["ka"], cross)


def test_figures_exist():
    for name in (
        "fig_referee_lid_error_vs_ka.png",
        "fig_referee_lid_b33_pole.png",
        "fig_referee_lid_crosscode.png",
    ):
        path = HERE / name
        assert path.exists() and path.stat().st_size > 10_000, name