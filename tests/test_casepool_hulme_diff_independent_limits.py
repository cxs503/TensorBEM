"""Guards for the third-code (capytaine) Richardson limit of the Hulme case.

The committed artefact
``benchmarks/verified/casepool_bem_hulme_diff_verif/independent_limits.json`` is the
independent adjudicator of the anchor's reference solution: the analytic multipole
series was compared cell-by-cell against the zero-panel limit of a 4-level capytaine
ladder (scattered force extrapolated, exact Froude-Krylov added back).

These tests keep the artefact honest:

* it must be re-derivable from the committed ladder (no hand-copied digits);
* the ladder it consumed must be the committed one (sha256);
* every cell must be covered (6 ka x 6 quantities) and every gate must be consistent
  with the recorded numbers;
* the measured reference <-> limit distance must stay inside a regression band that is
  much tighter than the case tolerance (measured worst 0.37 %, band 1 %), so a future
  change to ``analytic.py`` or to the ladder cannot silently degrade the agreement.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "benchmarks/verified/casepool_bem_hulme_diff_verif"
ARTIFACT = CASE_DIR / "independent_limits.json"
PRODUCER = CASE_DIR / "independent_limits.py"
LADDER = CASE_DIR.parent / "casepool_bem_hulme_hemisphere/capytaine_excitation.json"

#: regression band on the measured agreement (not the case tolerance): the producer
#: measured 0.37 % worst over all 36 cells, so 1 % leaves room for library drift while
#: still catching a real regression.
AGREEMENT_BAND = 0.01


def _producer():
    spec = importlib.util.spec_from_file_location("hulme_independent_limits", PRODUCER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(ARTIFACT.read_text())


def test_committed_artifact_is_currently_rederivable() -> None:
    """Recompute from the ladder and compare: the artefact carries no hand digits."""
    committed = json.loads(ARTIFACT.read_text())
    fresh = _producer().build()
    assert fresh["verdict"] == committed["verdict"]
    assert fresh["gates"].keys() == committed["gates"].keys()
    for ka, cell in fresh["cells"].items():
        assert ka in committed["cells"]
        for name, q in cell["quantities"].items():
            ref = committed["cells"][ka]["quantities"][name]
            assert q["independent_limit"] == pytest.approx(ref["independent_limit"], rel=1e-9)
            assert q["reference"] == pytest.approx(ref["reference"], rel=1e-9)
        for dof, f in cell["forces"].items():
            ref = committed["cells"][ka]["forces"][dof]
            for key in ("independent_limit", "reference", "exact_froude_krylov"):
                assert f[key] == pytest.approx(ref[key], rel=1e-9)


def test_artifact_consumed_the_committed_ladder(payload: dict) -> None:
    digest = hashlib.sha256(LADDER.read_bytes()).hexdigest()
    assert payload["independent_code"]["ladder_sha256"] == digest
    assert payload["independent_code"]["ladder_faces"] == [256, 576, 1024, 1600]


def test_every_band_cell_and_quantity_is_covered(payload: dict) -> None:
    ka = sorted(payload["cells"], key=float)
    assert ka == ["0.2", "0.5", "0.8", "1", "1.5", "2"], ka
    for cell in payload["cells"].values():
        assert set(cell["quantities"]) == {"A11", "B11", "A33", "B33"}
        assert set(cell["forces"]) == {"Surge", "Heave"}
    assert payload["counts"]["cells"] == 6 * 8  # 6 ka x (4 coefficients + 2 complex forces)


def test_gates_agree_with_the_recorded_numbers(payload: dict) -> None:
    gates = payload["gates"]
    worst_rel = max(
        [abs(q["relative"]) for c in payload["cells"].values()
         for q in c["quantities"].values()]
        + [c["forces"][d]["relative"] for c in payload["cells"].values()
           for d in ("Surge", "Heave")]
    )
    worst_abs = max(
        c["forces"][d]["absolute_normalised"]
        for c in payload["cells"].values() for d in ("Surge", "Heave")
    )
    assert gates["G1_relative"]["worst"] == pytest.approx(worst_rel, rel=1e-9)
    assert gates["G2_absolute_normalised_forces"]["worst"] == pytest.approx(worst_abs, rel=1e-9)
    for name, gate in gates.items():
        if name == "G4_extrapolation_quality":
            continue  # gated on the flagged-cell list, not on a scalar worst
        expected = "PASS" if gate["worst"] <= gate.get(
            "tolerance", gate.get("tolerance_deg", float("inf"))) else "FAIL"
        assert gate["verdict"] == expected, name
    assert payload["verdict"] == ("PASS" if all(
        g["verdict"] == "PASS" for g in gates.values()) else "FAIL")


def test_reference_agrees_with_the_independent_limit(payload: dict) -> None:
    """The adjudication this artefact exists for, held to a tighter regression band."""
    worst = payload["worst"]["relative"]
    assert worst < AGREEMENT_BAND, (
        f"reference vs independent limit {100 * worst:.2f}% exceeds the "
        f"{100 * AGREEMENT_BAND:.0f}% regression band ({payload['worst']['relative_cell']})"
    )
    assert payload["worst"]["absolute_normalised"] < AGREEMENT_BAND
    assert payload["worst"]["phase_deg"] < 3.0
    # every extrapolation must be usable, not silently averaged away
    for cell in payload["cells"].values():
        for q in cell["quantities"].values():
            assert q["extrapolation"]["limit_usable"], q
        for f in cell["forces"].values():
            assert f["extrapolation"]["limit_usable"], f
            assert f["extrapolation"]["spread_relative"] < 0.02


def test_extrapolation_order_was_estimated_not_assumed(payload: dict) -> None:
    """Orders must come out of the ladder (0.4 <= p <= 8) rather than being fixed."""
    orders = [q["extrapolation"]["order_p"] for c in payload["cells"].values()
              for q in c["quantities"].values()]
    assert min(orders) >= 0.4 and max(orders) <= 8.0
    assert len(set(round(p, 2) for p in orders)) > 1  # not a single hard-wired constant