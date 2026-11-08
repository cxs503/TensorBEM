"""Gate the two published deep-water deviation ladders against the current kernel.

``docs/finite_depth3d_regime.md`` §3 (m3409) and
``docs/finite_depth3d_deepwater_arbitration.md`` §3.3 (m4776) quote different
numbers for the same ``h`` ladder of ``|G/G_deep - 1|``. The m6384 study
``tensorbem.benchmarks.deepwater_ladder_geometry`` shows why: m3409 holds the
elevation at the absolute value ``z = -0.5 m`` (its driver's ``Z_MID``), while
m4776 read the prose "mid-depth" as the scaled value ``z = -h/2``. The two
conventions coincide only at ``h = 1 m``, which is precisely the one rung m4776
found to reproduce.

These tests pin that explanation as data, so that neither number can be
"corrected" again without the geometry being stated, and so that the published
JSON bundle cannot be hand-edited away from what the module computes.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from tensorbem.benchmarks.deepwater_ladder_geometry import (
    M3409_DOCUMENTED,
    M4776_RERUN,
    audit,
    ladder,
)


@pytest.fixture(scope="module")
def report() -> dict:
    return audit()


def test_m3409_ledger_reproduces_at_fixed_absolute_elevation(report):
    """``h``-ladder of docs/finite_depth3d_regime.md §3, z = -0.5 m fixed."""
    verdict = report["verdict"]["m3409_ladder_reproduces_at_fixed_elevation"]
    assert verdict["reproduces"], verdict
    assert verdict["worst_relative_vs_quoted"] < 5.0e-3
    # the ledger is not stale: every rung is within half a percent
    for row in verdict["rows"]:
        assert row["rel_to_quoted"] < 5.0e-3, row


def test_m4776_rerun_reproduces_at_scaled_mid_depth(report):
    """Same ladder re-read at z = -h/2 (the m4776 arbitration geometry)."""
    verdict = report["verdict"]["m4776_rerun_reproduces_at_scaled_mid_depth"]
    assert verdict["reproduces"], verdict
    assert verdict["worst_relative_vs_quoted"] < 5.0e-3


def test_the_two_ladders_meet_only_where_the_elevations_coincide(report):
    """Root cause: z = -0.5 m equals -h/2 only at h = 1 m."""
    assert report["verdict"]["coincident_depths"] == [1.0]
    assert report["verdict"]["ladders_coincide_only_where_z_equals_minus_h_over_2"]
    by_depth = {row["depth"]: row for row in report["fixed_elevation_ladder"]}
    assert by_depth[1.0]["z"] == pytest.approx(-0.5)
    assert by_depth[20.0]["z"] == pytest.approx(-0.5)


def test_ledger_numbers_are_not_a_commit_level_drift(report):
    """The m3409 ledger moves monotonically with k0*h, as documented, in both conventions."""
    for key in ("fixed_elevation_ladder", "scaled_mid_depth_ladder"):
        deviations = [row["relative_deviation"] for row in report[key]]
        assert deviations == sorted(deviations, reverse=True), key
    # The scaled convention is the slower-closing one; quoting one as the other
    # is a 2.5x-17x misquote at h >= 2, which is what m4776 interpreted as staleness.
    fixed = {row["depth"]: row["relative_deviation"] for row in report["fixed_elevation_ladder"]}
    scaled = {row["depth"]: row["relative_deviation"] for row in report["scaled_mid_depth_ladder"]}
    assert scaled[20.0] / fixed[20.0] > 100.0
    assert scaled[1.0] == pytest.approx(fixed[1.0], rel=1e-12)


def test_published_bundle_matches_a_fresh_build(tmp_path):
    """docs/m6384/ is build output: a fresh run must reproduce its numbers."""
    from tensorbem.benchmarks.deepwater_ladder_geometry import main

    root = pathlib.Path(__file__).resolve().parents[1]
    recorded = json.loads((root / "docs/m6384/deepwater_ladder_geometry.json").read_text())
    fresh = main(tmp_path / "fresh.json", None)
    for key in ("fixed_elevation_ladder", "scaled_mid_depth_ladder", "coincidence", "verdict"):
        assert fresh[key] == recorded[key], key


def test_documented_constants_are_the_published_ones():
    """The two quoted tables must stay verbatim copies of the two documents."""
    assert [depth for depth, _ in M3409_DOCUMENTED] == [1.0, 2.0, 5.0, 20.0, 100.0, 800.0]
    assert [depth for depth, _ in M4776_RERUN] == [1.0, 2.0, 5.0, 20.0]
    assert ladder("fixed")[1]["relative_deviation"] == pytest.approx(8.225e-02, rel=1e-3)
    assert ladder("mid-depth")[1]["relative_deviation"] == pytest.approx(2.049e-01, rel=1e-3)

def test_erratum_table_in_the_arbitration_doc_matches_the_ledger():
    """§8 of the arbitration document quotes this ledger; no number is hand-typed."""
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    text = (root / "docs/finite_depth3d_deepwater_arbitration.md").read_text()
    section = text.split("## 8. Erratum", 1)[1]
    rows = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 7 and re.fullmatch(r"\d+", cells[0]) and "e-0" in cells[2]:
            rows[float(cells[0])] = (
                float(cells[2].replace("e", "e")),
                float(cells[5]),
            )
    payload = json.loads((root / "docs/m6384/deepwater_ladder_geometry.json").read_text())
    fixed = {row["depth"]: row["relative_deviation"] for row in payload["fixed_elevation_ladder"]}
    scaled = {row["depth"]: row["relative_deviation"] for row in payload["scaled_mid_depth_ladder"]}
    assert set(rows) == set(fixed), rows
    for depth, (doc_fixed, doc_scaled) in rows.items():
        assert doc_fixed == pytest.approx(fixed[depth], rel=5e-4), depth
        assert doc_scaled == pytest.approx(scaled[depth], rel=5e-4), depth
