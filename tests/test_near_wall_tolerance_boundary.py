"""m21776: the case tolerance must come with a measured inner bound."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorbem.benchmarks import case_cylinder_stream as case


@pytest.fixture(scope="module")
def ladder_study() -> dict:
    """One two-level near-wall study, shared: it is the expensive artefact."""
    return case.near_wall_study(ladder=(32, 64), factors=(2.0, 0.1))


def test_near_wall_tolerance_boundary_is_measured_not_assumed(
    ladder_study: dict,
) -> None:
    """The ``< 3 %`` acceptance needs a measured clearance, not an assumed one.

    The record gated two clearance *levels* (``delta = 2 h_max`` inside 1 %,
    ``delta = 0.1 h_max`` outside 3 %) and left the boundary between them
    unmeasured, so a caller could evaluate at ``0.25 h_max`` -- about 9 % error,
    outside the case tolerance -- while every criterion passed.  This checks that
    the boundary is located on every level, that it is bracketed by probes that
    actually straddle the tolerance, and that refining the mesh moves it *into*
    the body rather than shrinking the usable envelope.
    """
    # the contract levels are unioned in, so a short tuple is a sampling choice
    # and not the ``KeyError: 0.5`` crash it used to be
    assert ladder_study["factors"] == [2.0, 0.5, 0.1]
    floor = ladder_study["floor"]

    assert floor["all_levels_located"], floor["levels"]
    assert len(floor["levels"]) == 2

    for level, detail in zip(floor["levels"], ladder_study["levels"]):
        measured = detail["three_percent_floor"]
        boundary = measured["clearance_over_h_max"]
        coarse, fine = measured["bracket"]
        error_coarse, error_fine = measured["errors_at_bracket"]
        # the bracket must straddle the tolerance, or the "boundary" would be an
        # extrapolation dressed up as a measurement
        assert error_coarse >= case.TOLERANCE > error_fine
        assert coarse <= boundary <= fine
        assert boundary == pytest.approx(level["clearance_over_h_max"])
        # the crossing must be a genuine power law, not a flat or noisy window
        assert measured["observed_order_in_clearance"] > 0.0
        # and the grid itself must be monotone in the error, otherwise the single
        # bracketing pair would be an artefact of a non-monotone sweep
        errors = [row["max_relative_velocity_error"] for row in measured["grid"]]
        assert all(later < earlier for earlier, later in zip(errors, errors[1:])), errors

    shifts = floor["refinement_shifts"]
    assert len(shifts) == 1
    assert floor["boundary_moves_inward_with_refinement"]
    assert shifts[0] >= 1.0 / (1.0 + floor["monotone_tolerance"]), shifts


def test_a_single_level_cannot_claim_boundary_invariance() -> None:
    """One level cannot show the boundary is mesh-independent; do not claim it."""
    single = case.near_wall_study(ladder=(32,), factors=(2.0, 0.1))
    floor = single["floor"]
    assert floor["all_levels_located"]
    assert floor["refinement_shifts"] == []
    assert not floor["boundary_moves_inward_with_refinement"]


def test_near_wall_window_sits_clear_of_the_measured_boundary(
    ladder_study: dict,
) -> None:
    """``2 h_max`` must be far outside the boundary it is compared against."""
    floor = ladder_study["floor"]
    margin = floor["window_factor_over_finest_floor"]
    assert margin is not None
    assert margin == pytest.approx(
        case.NEAR_WALL_WINDOW_FACTOR / floor["finest_level_clearance_over_h_max"]
    )
    assert margin >= floor["window_margin_required"]
    # the boundary must lie strictly between the exclusion level and the window,
    # i.e. it is a real interior feature of the measured sweep
    assert case.NEAR_WALL_EXCLUSION_FACTOR < floor["finest_level_clearance_over_h_max"]
    assert floor["finest_level_clearance_over_h_max"] < case.NEAR_WALL_WINDOW_FACTOR


RECORD = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "verified"
    / "case_bem_laplace_cylinder"
    / "results.json"
)


def test_committed_record_gates_the_tolerance_boundary() -> None:
    """A stale record cannot carry the new gates; skip until it is regenerated."""
    if not RECORD.is_file():
        pytest.skip("committed record missing")
    record = json.loads(RECORD.read_text())
    floor = record.get("near_wall", {}).get("floor")
    if floor is None:
        pytest.skip("committed record predates the tolerance-boundary study")
    assert floor["all_levels_located"]
    by_name = {row["name"]: row for row in record["criteria"]}
    for name in (
        "near_wall_tolerance_boundary_is_located",
        "near_wall_tolerance_boundary_moves_inward_with_refinement",
        "near_wall_window_clears_measured_tolerance_boundary",
    ):
        assert name in by_name, name
        assert by_name[name]["ok"], by_name[name]
    assert record["provenance"]["combined_sha256"]
    figures = RECORD.parent / "figures"
    assert figures.is_dir(), "the rendered document references figures/"
    assert sorted(path.name for path in figures.glob("*.png")) == [
        "circulation.png",
        "convergence.png",
        "field.png",
        "multipole.png",
        "near_wall.png",
    ]