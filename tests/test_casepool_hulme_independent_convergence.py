"""Integrity tests for the anchor BEM-HULME-HEMISPHERE's independent-axis audit.

The audit turns the anchor's "the third code agrees to 3.36 %" into "how far is
the third code from its own continuum limit".  These tests hold it to its own
evidence: stored gates must be reproducible from the raw artifacts, no gate may
pass without a numeric measurement, the verdict must be the and-reduction of the
gates, the extended ladder must still carry the published rungs unchanged, and
the estimator must first recover a known order from synthetic data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorbem.benchmarks.hemisphere_independent_convergence import (
    CASE_ARTIFACT,
    CASE_KEY,
    EXTENDED_ARTIFACT,
    FROZEN_ARTIFACT,
    INHOUSE_ARTIFACT,
    QUANTITIES,
    analyse,
    fit_convergence,
)

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere"
EVIDENCE = CASE_DIR / "evidence" / "independent_convergence.json"
DOCUMENT = ROOT / "docs" / "case_bem_hulme_independent_convergence.md"


def test_artifacts_present() -> None:
    for path in (CASE_ARTIFACT, FROZEN_ARTIFACT, EXTENDED_ARTIFACT, EVIDENCE, DOCUMENT):
        assert path.exists(), f"missing artifact {path}"


def test_extended_ladder_carries_the_published_rungs_unchanged() -> None:
    frozen = json.loads(FROZEN_ARTIFACT.read_text())
    extended = json.loads(EXTENDED_ARTIFACT.read_text())
    by_panels = {lvl["n_faces"]: lvl for lvl in extended["levels"]}
    for level in frozen["levels"]:
        assert level["n_faces"] in by_panels, level["n_faces"]
        for ka_tag, row in level["values"].items():
            for name, key in QUANTITIES:
                assert float(row[key]) == pytest.approx(
                    float(by_panels[level["n_faces"]]["values"][ka_tag][key]), rel=1e-12
                ), (level["n_faces"], ka_tag, name)
    assert max(by_panels) > max(lvl["n_faces"] for lvl in frozen["levels"]), "not an extension"
    assert len(extended["levels"]) >= 5


def test_convergence_estimator_recovers_a_known_order() -> None:
    panels = [256, 576, 1024, 1600, 2304, 3136]
    for p_true in (0.8, 1.5, 2.0):
        values = [1.0 + 0.5 * n ** (-p_true) for n in panels]
        fit = fit_convergence(panels, values)
        assert fit["p"] == pytest.approx(p_true, rel=1e-3)
        assert fit["v_inf"] == pytest.approx(1.0, abs=1e-6)
        assert fit["max_rel_residual"] < 1e-6
    with pytest.raises(ValueError):
        fit_convergence([256, 1024], [1.0, 0.9])


def test_stored_gates_are_reproducible() -> None:
    stored = json.loads(EVIDENCE.read_text())
    fresh = analyse()
    assert fresh["verdict"] == stored["verdict"]
    assert set(fresh["gates"]) == set(stored["gates"])
    for key, gate in stored["gates"].items():
        assert fresh["gates"][key]["pass"] == gate["pass"], key
        if gate.get("measured") is not None:
            assert float(fresh["gates"][key]["measured"]) == pytest.approx(
                float(gate["measured"]), rel=1e-9
            ), key


def test_no_gate_passes_without_a_measurement() -> None:
    stored = json.loads(EVIDENCE.read_text())
    for key, gate in stored["gates"].items():
        if gate["pass"]:
            assert gate.get("measured") is not None, f"{key} passed with no measurement"
            assert math.isfinite(float(gate["measured"])), key


def test_verdict_is_the_and_of_the_gates() -> None:
    stored = json.loads(EVIDENCE.read_text())
    assert stored["verdict"] == (
        "PASS" if all(g["pass"] for g in stored["gates"].values()) else "FAIL"
    )


def test_document_quotes_every_gate_and_the_verdict() -> None:
    stored = json.loads(EVIDENCE.read_text())
    text = DOCUMENT.read_text()
    for key in stored["gates"]:
        assert key in text, key
    assert stored["verdict"] in text


@pytest.mark.skipif(not INHOUSE_ARTIFACT.exists(), reason="in-house ladder not produced")
def test_inhouse_ladder_reproduces_the_graded_row() -> None:
    case = json.loads(CASE_ARTIFACT.read_text())
    ladder = json.loads(INHOUSE_ARTIFACT.read_text())
    ka = float(ladder["ka"])
    graded = [r for r in case["rows"]
              if r["kind"] == "band" and r["subdivisions"] == 4 and float(r["ka"]) == ka][0]
    rung = [r for r in ladder["rows"] if r["panels"] == 1024][0]
    for key in ("A11_dimless", "B33_dimless", "Fz_dimless"):
        assert float(rung[key]) == pytest.approx(float(graded[key]), rel=1e-9), key


# --- E7/E9: the in-house ladder must be *identified*, not merely reproduced ---
def _synthetic_ladder(path: Path, panels: list[int], ka: float = 2.0) -> Path:
    """In-house ladder artifact obeying ``v(N) = v_inf + C N^-p`` exactly."""
    rows = []
    for n in panels:
        row = {"subdivisions": int(round((n / 4) ** 0.25)), "panels": n,
               "ka": ka}
        for i, (name, _) in enumerate(QUANTITIES):
            v_inf = 0.2 + 0.1 * i
            row[CASE_KEY[name]] = v_inf * (1.0 + 0.5 * (n / 64.0) ** -0.9)
        rows.append(row)
    payload = {"case_code": "BEM-HULME-HEMISPHERE", "ka": ka,
               "quantities": [CASE_KEY[name] for name, _ in QUANTITIES],
               "rows": rows, "failures": [],
               "environment": {"host": "synthetic"}}
    path.write_text(json.dumps(payload))
    return path


def test_three_rung_inhouse_ladder_is_not_evidence(tmp_path: Path) -> None:
    """A three-parameter model on three rungs has zero residual by construction."""
    ladder = _synthetic_ladder(tmp_path / "l3.json", [64, 256, 1024])
    payload = analyse(inhouse_path=ladder)
    e7 = payload["gates"]["E7_inhouse_model_credible"]
    assert e7["n_rungs"] == 3
    assert e7["fit_identified"] is False
    assert e7["pass"] is False, "an exactly-determined fit must not pass E7"
    assert payload["gates"]["E9_inhouse_order_and_limit_stable"]["pass"] is False
    assert payload["gates"]["E9_inhouse_order_and_limit_stable"]["measured"] is None


def test_four_rung_inhouse_ladder_passes_only_on_identified_fits(tmp_path: Path) -> None:
    ladder = _synthetic_ladder(tmp_path / "l4.json", [64, 256, 1024, 4096])
    payload = analyse(inhouse_path=ladder)
    e7 = payload["gates"]["E7_inhouse_model_credible"]
    assert e7["n_rungs"] == 4 and e7["fit_identified"] is True
    assert e7["measured"] < 1e-6, "exact power law must reproduce every rung"
    assert e7["pass"] is True
    # E9 measures a genuine window change: with a clean power law the finest
    # three rungs and all four rungs must agree on order and limit
    e9 = payload["gates"]["E9_inhouse_order_and_limit_stable"]
    assert e9["pass"] is True and e9["worst_order_shift"] < 1e-6
    assert e9["worst_limit_shift"] < 1e-6


def test_committed_inhouse_ladder_has_enough_rungs_to_carry_e7() -> None:
    """The committed artifact either carries >= 4 rungs or records the failure."""
    if not INHOUSE_ARTIFACT.exists():
        pytest.skip("in-house ladder not produced")
    ladder = json.loads(INHOUSE_ARTIFACT.read_text())
    n = len(ladder["rows"])
    assert n >= 1
    if n < 4:
        assert ladder.get("failures") or ladder.get(
            "absence_reason"
        ), "fewer than four rungs must say why"
    stored = json.loads(EVIDENCE.read_text())
    e7 = stored["gates"]["E7_inhouse_model_credible"]
    assert e7["n_rungs"] == n


# --- E0: a rung that cannot prove its solve-stage health is not physics -----
def test_contaminated_rung_is_dropped_and_the_gate_fails(tmp_path: Path) -> None:
    """A 4096-panel rung produced by a broken driver must not enter the fit.

    Reproduction of the first ladder's top rung (dev report m20910): the value
    is 46 % off, the recorded residual is 4.9e-1 -- the driver did not solve the
    system it was handed.  E0 must drop it and must fail, while the four healthy
    rungs still carry E7.
    """
    ladder = json.loads(_synthetic_ladder(tmp_path / "l4.json", [64, 256, 1024, 2048]).read_text())
    for row in ladder["rows"]:
        row["radiation_relative_residual"] = 1e-3
    bad = dict(ladder["rows"][-1])
    bad.update({"subdivisions": 5, "panels": 4096, "radiation_relative_residual": 0.49})
    for name, _ in QUANTITIES:
        bad[CASE_KEY[name]] *= 0.54
    ladder["rows"].append(bad)
    path = tmp_path / "l5.json"
    path.write_text(json.dumps(ladder))

    payload = analyse(inhouse_path=path)
    e0 = payload["gates"]["E0_rungs_solver_healthy"]
    assert e0["pass"] is False
    assert [d["panels"] for d in e0["dropped"]] == [4096.0]
    assert e0["dropped"][0]["residual"] == pytest.approx(0.49)
    assert e0["n_rungs_used"] == 4, "the dropped rung must not be fitted"
    assert 4096.0 not in payload["inhouse"]["panels"]
    assert payload["verdict"] == "FAIL"


def test_unrecorded_residual_is_unproven_not_healthy(tmp_path: Path) -> None:
    """Artifacts older than the guard can be used, but must not testify."""
    ladder = _synthetic_ladder(tmp_path / "l4.json", [64, 256, 1024, 4096])
    payload = analyse(inhouse_path=ladder)
    e0 = payload["gates"]["E0_rungs_solver_healthy"]
    assert e0["measured"] is None
    assert e0["n_rungs_recorded"] == 0
    assert e0["unproven_panels"] == [64.0, 256.0, 1024.0, 4096.0]
    assert e0["pass"] is True, "no rung was *shown* unhealthy"
    assert e0["n_rungs_used"] == 4


# --- E0 provenance: a mixed ladder must be disclosed, not smoothed over -----
def test_rung_ledger_keeps_the_rungs_separable(tmp_path: Path) -> None:
    """One ledger line per admitted rung, carrying its residual and its host.

    The gate's scalar summary (worst residual, how many rungs used) is enough to
    pass it but not to audit it: a ladder whose cheap rungs ran on the control
    plane and whose expensive rung came back from a cluster node -- where the
    linear driver has already been caught returning a wrong vector (m20910) --
    looks identical to a single-host ladder once reduced to a maximum.
    """
    ladder = json.loads(_synthetic_ladder(tmp_path / "l4.json", [64, 256, 1024, 4096]).read_text())
    hosts = ["control-plane", "control-plane", "control-plane", "node315"]
    for row, host, residual in zip(ladder["rows"], hosts, [6e-3, 1e-3, 2e-4, 3e-5]):
        row["host"] = host
        row["numpy"] = "2.4.6"
        row["radiation_relative_residual"] = residual
        row["solve_warnings"] = []
    path = tmp_path / "l4_hosts.json"
    path.write_text(json.dumps(ladder))

    payload = analyse(inhouse_path=path)
    e0 = payload["gates"]["E0_rungs_solver_healthy"]
    ledger = e0["rung_ledger"]
    assert [entry["panels"] for entry in ledger] == [64.0, 256.0, 1024.0, 4096.0]
    assert [entry["residual"] for entry in ledger] == pytest.approx([6e-3, 1e-3, 2e-4, 3e-5])
    assert [entry["host"] for entry in ledger] == hosts
    assert e0["mixed_provenance"]["is_mixed"] is True
    assert e0["mixed_provenance"]["hosts"] == ["control-plane", "node315"]
    assert e0["mixed_provenance"]["n_rungs_unknown_host"] == 0
    # disclosing a mixture is not a defect: the gate still passes on residuals
    assert e0["pass"] is True


def test_unrecorded_provenance_is_counted_as_unknown_not_assumed(tmp_path: Path) -> None:
    """A rung older than the ledger must not be credited with a host."""
    path = _synthetic_ladder(tmp_path / "l4.json", [64, 256, 1024, 4096])
    ladder = json.loads(path.read_text())
    for row in ladder["rows"]:
        row.pop("host", None)
    path.write_text(json.dumps(ladder))
    payload = analyse(inhouse_path=path)
    e0 = payload["gates"]["E0_rungs_solver_healthy"]
    assert all(entry["host"] is None for entry in e0["rung_ledger"])
    assert e0["mixed_provenance"]["is_mixed"] is False
    assert e0["mixed_provenance"]["n_rungs_unknown_host"] == 4


def test_committed_healthy_ladder_proves_its_own_health() -> None:
    if not INHOUSE_ARTIFACT.exists():
        pytest.skip("healthy in-house ladder not produced")
    ladder = json.loads(INHOUSE_ARTIFACT.read_text())
    assert ladder["rows"], "ladder must carry rungs"
    for row in ladder["rows"]:
        residual = max(
            float(row[k]) for k in ("radiation_relative_residual", "diffraction_relative_residual")
            if row.get(k) is not None
        )
        assert residual < 1e-2, (row["panels"], residual)
    stored = json.loads(EVIDENCE.read_text())
    assert stored["gates"]["E0_rungs_solver_healthy"]["pass"] is True
    assert stored["gates"]["E0_rungs_solver_healthy"]["n_rungs_recorded"] == len(ladder["rows"])


def test_case_cell_records_the_residual_it_was_produced_with() -> None:
    """The graded cell must carry its own solve-stage health, not just a value."""
    from tensorbem.benchmarks.hulme_hemisphere_case import run_cell

    cell = run_cell(2.0, 2)
    assert cell["radiation_relative_residual"] < 1e-2
    assert cell["diffraction_relative_residual"] < 1e-2
    assert cell["solve_warnings"] == []
