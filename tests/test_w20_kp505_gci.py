"""Contract tests for the W20 KP505 BEM GCI artefact.

These tests do not re-run the lattice (a single fine-grid evaluation costs about
a minute on the loongarch node); they re-derive every reported statistic from the
level values stored in the artefact, which is enough to catch a hand-typed
number, a silently relaxed threshold, or a decomposition that does not close.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorbem.uq import DEFAULT_SAFETY_FACTOR, grid_convergence_index

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTEFACT = (
    REPO_ROOT / "benchmarks" / "verified" / "bem_kp505_openwater_w20" / "gci.json"
)
MISSION_J = (0.5, 0.7, 0.9)
DESIGN_J = 0.7


@pytest.fixture(scope="module")
def artefact() -> dict:
    assert ARTEFACT.is_file(), f"missing artefact {ARTEFACT}"
    return json.loads(ARTEFACT.read_text(encoding="utf-8"))


def _primary(artefact: dict) -> dict:
    return {
        cell["J"]: cell for cell in artefact["cells"] if cell["core_policy"] == "fixed"
    }


def test_artefact_shape(artefact: dict) -> None:
    assert artefact["schema"] == "tensorbem.w20_kp505_bem_gci/1"
    assert artefact["design"]["mission_J"] == list(MISSION_J)
    assert artefact["design"]["design_J"] == DESIGN_J
    assert len(artefact["design"]["panel_levels"]) == 3
    assert artefact["design"]["refinement_ratio"] == 2.0
    assert artefact["design"]["design_tolerance_pct"] == 3.0
    assert artefact["design"]["order_tolerance"] == 0.30


def test_three_densities_per_advance_ratio(artefact: dict) -> None:
    primary = _primary(artefact)
    for advance in MISSION_J:
        levels = primary[advance]["levels"]
        assert [level["level"] for level in levels] == [1, 2, 3]
        assert [level["n_radial"] for level in levels] == [8, 16, 32]
        assert [level["n_chord"] for level in levels] == [2, 4, 8]
        assert [level["wake_steps"] for level in levels] == [24, 48, 96]
        for level in levels:
            assert math.isfinite(level["KT"])
            assert math.isfinite(level["10KQ"])
            assert level["KT"] > 0.0
            assert level["10KQ"] > 0.0


def test_gci_and_observed_order_are_recomputed_from_the_levels(artefact: dict) -> None:
    primary = _primary(artefact)
    for advance in MISSION_J:
        cell = primary[advance]
        for quantity in ("KT", "10KQ"):
            block = cell["quantities"][quantity]
            fine, medium, coarse = (
                block["levels"][2],
                block["levels"][1],
                block["levels"][0],
            )
            expected = grid_convergence_index(
                fine=fine,
                medium=medium,
                coarse=coarse,
                refinement_ratio=2.0,
                safety_factor=DEFAULT_SAFETY_FACTOR,
                quantity=quantity,
            ).as_dict()
            assert block["observed_order"] == pytest.approx(
                expected["observed_order"], rel=1e-12, abs=1e-12
            )
            assert block["gci_fine_pct"] == pytest.approx(
                100.0 * expected["gci_fine"], rel=1e-12
            )
            assert block["gci"]["extrapolated"] == pytest.approx(
                expected["extrapolated"], rel=1e-12
            )


def test_order_deviation_is_relative_to_the_predeclared_theory_order(
    artefact: dict,
) -> None:
    primary = _primary(artefact)
    p_th = artefact["design"]["theory_order"]
    assert p_th == 2.0
    for advance in MISSION_J:
        for quantity in ("KT", "10KQ"):
            block = primary[advance]["quantities"][quantity]
            p = block["observed_order"]
            if not math.isfinite(p):
                continue
            assert block["order_deviation_from_theory_pct"] == pytest.approx(
                abs(p - p_th) / p_th * 100.0, rel=1e-12
            )
            assert block["order_within_tolerance"] == (
                block["order_deviation_from_theory_pct"] < 30.0
            )


def test_decomposition_closes(artefact: dict) -> None:
    for cell in artefact["cells"]:
        for quantity in ("KT", "10KQ"):
            block = cell["quantities"][quantity]
            decomposition = block["decomposition"]
            residual = abs(
                decomposition["d_discretisation_pct"]
                + decomposition["d_model_pct"]
                - decomposition["d_total_pct"]
            )
            assert residual <= 1e-9
            anchor = decomposition["f_anchor"]
            assert decomposition["d_total_pct"] == pytest.approx(
                100.0 * (decomposition["f_fine"] - anchor) / anchor, rel=1e-12
            )


def test_anchor_values_match_the_transcription(artefact: dict) -> None:
    transcription = json.loads(
        (
            REPO_ROOT / "papers" / "data" / "w20-anchors" / "kp505_transcription.json"
        ).read_text(encoding="utf-8")
    )
    anchors = {point["J"]: point for point in transcription["anchor_points"]}
    for key, values in artefact["anchor"]["values"].items():
        advance = float(key)
        assert values["KT"] == pytest.approx(anchors[advance]["KT"], abs=1e-12)
        assert values["10KQ"] == pytest.approx(anchors[advance]["10KQ"], abs=1e-12)


def test_design_point_answers_the_three_percent_question(artefact: dict) -> None:
    verdict = artefact["verdict"]["design_point"]
    primary = _primary(artefact)[DESIGN_J]["quantities"]
    assert verdict["dKT_pct"] == pytest.approx(
        primary["KT"]["delta_vs_anchor_pct"], rel=1e-12
    )
    assert verdict["dKQ_pct"] == pytest.approx(
        primary["10KQ"]["delta_vs_anchor_pct"], rel=1e-12
    )
    expected = abs(verdict["dKT_pct"]) < 3.0 and abs(verdict["dKQ_pct"]) < 3.0
    assert verdict["answer_under_3pct"] is expected
    # the threshold is not allowed to drift with the result
    assert artefact["design"]["design_tolerance_pct"] == 3.0


def test_verdict_matches_criteria(artefact: dict) -> None:
    passes = {c["id"]: c["pass"] for c in artefact["criteria"]}
    assert artefact["verdict"]["pass_by_id"] == passes
    assert artefact["verdict"]["all_criteria_pass"] == all(passes.values())
    assert len(passes) == 8


def test_inertness_probe_documents_the_dead_core_policy(artefact: dict) -> None:
    probe = artefact["core_inertness_probe"]
    assert probe["policy_range_is_inert"] is True
    for label, sweep in probe["sweeps"].items():
        first_three = [f"{v:.15g}" for v in sweep["KT_values"][:3]]
        assert len(set(first_three)) == 1, (
            f"{label}: policy ratios must be bit-identical"
        )
    # a wider ratio must still bite, i.e. the parameter is clamped, not dead
    for label in ("L1-coarse", "L3-fine"):
        sweep = probe["sweeps"][label]
        assert sweep["max_rel_change_pct"] > 0.0
        assert sweep["first_ratio_that_moves_KT"] == pytest.approx(0.4)
