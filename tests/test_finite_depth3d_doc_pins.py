"""Gates for :mod:`tensorbem.benchmarks.finite_depth3d_pins`.

Every number published in ``docs/finite_depth3d.md`` must still be the number
the kernel produces, and the four reference checks must hold against values
that share no code with the quantity under test.
"""

from __future__ import annotations

import json
import math

import pytest

from tensorbem.benchmarks import finite_depth3d_pins as pins

LEDGER = pins.evaluate()


def _claims(prefix: str = "") -> dict[str, dict]:
    return {
        claim["id"]: claim
        for claim in LEDGER["claims"]
        if claim["id"].startswith(prefix)
    }


# ---------------------------------------------------------------------------
# structure of the document
# ---------------------------------------------------------------------------
def test_document_tables_are_parsed() -> None:
    tables = pins.parse_tables(pins.DOC.read_text())
    assert set(tables) == {
        "dispersion",
        "boundary_conditions",
        "deep_water_limit",
        "singularity",
        "truncation",
    }
    assert len(tables["dispersion"][1]) == len(pins.DEPTHS)
    assert len(tables["boundary_conditions"][1]) == len(pins.BC_DEPTHS)
    assert len(tables["deep_water_limit"][1]) == len(pins.DEEP_DEPTHS)
    assert len(tables["singularity"][1]) == len(pins.SING_RADII)
    assert len(tables["truncation"][1]) == len(pins.TRUNC_MODES)


def test_registry_covers_every_table_cell() -> None:
    """No published cell may be left outside the registry."""
    claims = LEDGER["claims"]
    protected = {claim["id"] for claim in claims}
    expected = (
        {f"dispersion/h={depth:g}/{column}"
         for depth in pins.DEPTHS
         for column in ("k0", "k0_h", "c", "cg", "residual")}
        | {f"boundary_conditions/h={depth:g}/{column}"
           for depth in pins.BC_DEPTHS
           for column in ("free_surface_residual", "bottom_residual")}
        | {f"deep_water_limit/h={depth:g}/deviation" for depth in pins.DEEP_DEPTHS}
        | {f"singularity/r={radius:g}" for radius in pins.SING_RADII}
        | {f"truncation/modes={modes}" for modes in pins.TRUNC_MODES}
    )
    assert expected <= protected
    assert LEDGER["coverage"]["protected_fraction"] > 0.8


# ---------------------------------------------------------------------------
# declared numbers reproduce
# ---------------------------------------------------------------------------
def test_every_declared_number_reproduces() -> None:
    drift = LEDGER["drift"]
    assert not drift, "documented numbers drifted:\n" + "\n".join(
        f"{row['id']}: declared {row['declared']!r} measured {row['measured']!r} "
        f"(rel {row['relative']:.2e}, tol {row['tolerance']:.2e})"
        for row in drift
    )
    assert LEDGER["verdict"] == "PASS"


def test_verdict_section_quotes_the_tables() -> None:
    """§5 must keep quoting the extremes of §3 (self-consistency of the text)."""
    for claim in _claims("verdict/").values():
        assert claim["verdict"] == "PASS", claim


# ---------------------------------------------------------------------------
# independent references
# ---------------------------------------------------------------------------
def test_wavenumber_matches_independent_bisection() -> None:
    for depth in pins.DEPTHS:
        indep = LEDGER["claims"]
        gap = next(
            row["relative"]
            for row in indep
            if row["id"] == f"independent/dispersion/h={depth:g}/k0_vs_bisection"
        )
        assert gap < 1e-6, (depth, gap)


def test_group_velocity_matches_numerical_derivative() -> None:
    measured = pins.measure()
    for depth in pins.DEPTHS:
        indep = measured["independent"]["dispersion"][depth]
        assert indep["cg_rel_gap"] < 1e-3, (depth, indep["cg_rel_gap"])
        assert indep["c_rel_gap"] < 1e-9, (depth, indep["c_rel_gap"])


def test_deep_water_limit_is_monotone_and_bounded() -> None:
    measured = pins.measure()
    assert measured["independent"]["deep_water_monotone"]
    last = measured["deep_water_limit"][pins.DEEP_DEPTHS[-1]]["max_rel_error"]
    # the document states 2.908e-07 at k0 h = 76.8
    assert last < 3.0e-7, last
    # and the limit must be reached from above: the deviation decreases
    first = measured["deep_water_limit"][pins.DEEP_DEPTHS[0]]["max_rel_error"]
    assert first > last


def test_far_field_and_singularity_against_analytic_values() -> None:
    measured = pins.measure()
    slope = measured["far_field"]["phase_slope"]
    assert abs(slope - 1.0) < 3e-3, slope
    strength = measured["singularity"][pins.SING_RADII[-1]]["four_pi_r_G"]
    assert abs(strength - 1.0) < 3e-3, strength
    # 4 pi r G must approach 1 monotonically as r -> 0
    sequence = [measured["singularity"][r]["four_pi_r_G"] for r in pins.SING_RADII]
    assert all(abs(hi - 1.0) < abs(lo - 1.0) for lo, hi in zip(sequence, sequence[1:]))


def test_free_surface_condition_holds_on_the_surface() -> None:
    measured = pins.measure()
    for depth in pins.BC_DEPTHS:
        row = measured["boundary_conditions"][depth]
        assert row["free_surface_residual"] < 1e-10, (depth, row)
        assert row["bottom_residual"] == 0.0 or row["bottom_residual"] < 1e-25


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------
def test_pin_ledger_is_current() -> None:
    """The committed ledger must be what the kernel produces today."""
    stored = json.loads(pins.PINS_FILE.read_text())
    fresh = pins.evaluate()
    assert stored["verdict"] == fresh["verdict"] == "PASS"
    stored_claims = {row["id"]: row for row in stored["claims"]}
    fresh_claims = {row["id"]: row for row in fresh["claims"]}
    assert set(stored_claims) == set(fresh_claims)
    for claim_id, row in fresh_claims.items():
        old = stored_claims[claim_id]
        assert old["declared"] == pytest.approx(row["declared"], rel=1e-12, abs=1e-300)
        assert old["measured"] == pytest.approx(row["measured"], rel=1e-9, abs=1e-30), claim_id


def test_document_matches_renderer_artifact() -> None:
    """``docs/finite_depth3d.md`` is rendered from ``study.json``; keep them coupled."""
    study = json.loads(pins.STUDY_FILE.read_text())
    tables = pins.parse_tables(pins.DOC.read_text())
    _, rows = tables["dispersion"]
    by_depth = {row["depth"]: row for row in study["dispersion"]}
    for row in rows:
        depth = pins._float(row[0])  # noqa: SLF001 - same package
        artifact = by_depth[depth]
        assert pins._float(row[1]) == pytest.approx(artifact["k0"], rel=1e-6, abs=1e-12)
        assert pins._float(row[4]) == pytest.approx(artifact["cg"], rel=1e-4, abs=1e-12)
    _, deep_rows = tables["deep_water_limit"]
    for row, artifact in zip(deep_rows, study["deep_water_limit"]):
        assert pins._float(row[2]) == pytest.approx(
            artifact["max_rel_error"], rel=1e-3, abs=1e-30
        )
    _, sing_rows = tables["singularity"]
    for row, artifact in zip(sing_rows, study["singularity"]):
        assert pins._float(row[1]) == pytest.approx(
            artifact["four_pi_r_G"], rel=1e-6, abs=1e-12
        )


def test_printed_tolerance_tracks_the_printed_digits() -> None:
    """Half-ulp of the last written digit, times the safety factor: 0.6967 -> 2.2e-4."""
    assert pins.printed_tolerance("0.6967") == pytest.approx(2.153e-4, rel=1e-3)
    assert pins.printed_tolerance("3.481518") == pytest.approx(4.31e-7, rel=1e-2)
    assert pins.printed_tolerance("1.110e-15") == pytest.approx(1.351e-3, rel=1e-2)
    assert pins.printed_tolerance("0.000e+00") == 0.0
    # more printed digits can only tighten the tolerance
    assert pins.printed_tolerance("0.1") > pins.printed_tolerance("0.100000")