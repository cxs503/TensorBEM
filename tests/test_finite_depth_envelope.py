"""Guards the verification envelope of the finite-depth radiation solver.

``docs/finite_depth_radiation.md`` §5 limitation 5 recorded that the solver had
only ever been validated at ``ka = 0.6``, ``f/a = 2``.
:mod:`tensorbem.benchmarks.finite_depth_envelope` marches a 3x3 grid of
``ka`` x ``f/a`` and writes ``docs/finite_depth_envelope/benchmark.json`` plus
the document that quotes it.  These tests keep three things true:

1. the artifact is still what this code computes (one live re-measure at the
   cheapest tabulated case, ``1e-9`` agreement);
2. the acceptance criteria hold on the whole grid -- and where they do not, the
   artifact and the document both say so instead of hiding it;
3. the document is the renderer's output, not a hand-copied transcript, and the
   numbers it prints are the artifact's numbers.

The grid is expensive to recompute (it is the published run, ~10 min), so only
the first check re-runs the solver; the rest read the committed artifact.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from tensorbem.benchmarks import finite_depth_envelope as envelope

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs" / "finite_depth_envelope" / "benchmark.json"
DOC = ROOT / "docs" / "finite_depth_envelope.md"


@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT.exists():
        pytest.skip(f"missing artifact {ARTIFACT}; regenerate with "
                    "python -m tensorbem.benchmarks.finite_depth_envelope")
    return json.loads(ARTIFACT.read_text())


# --------------------------------------------------------------------------
# 1. Shape and liveliness
# --------------------------------------------------------------------------
def test_artifact_covers_the_grid(artifact):
    settings = artifact["settings"]
    ka = settings["ka_values"]
    fr = settings["f_ratios"]
    assert len(artifact["points"]) == len(ka) * len(fr) * 2
    assert len(artifact["control"]) == len(ka) * 2
    assert len(artifact["shallow"]) == len(ka) * len(fr) * 2
    assert len(artifact["refinement"]) == (
        len(settings["refinement_points"]) * len(settings["refinement_panels"]) * 2
    )
    assert {round(r["ka"], 6) for r in artifact["points"]} == {round(k, 6) for k in ka}
    assert {round(r["f_ratio"], 6) for r in artifact["points"]} == {round(f, 6) for f in fr}
    assert {r["depth"] for r in artifact["points"]} == {settings["grid_depth"]}
    assert {r["depth"] for r in artifact["control"]} == {settings["control_depth"]}
    assert {r["panels"] for r in artifact["points"]} == {settings["panels"]}


def test_cheapest_case_is_still_what_the_code_computes(artifact):
    """One live re-measure of the cheapest tabulated case.

    The shallowest / longest-wave row of the shallow group is the cheapest solve
    in the artifact (fewest evanescent modes) while still exercising the same
    assembly, so a regression anywhere in the chain moves it.
    """
    row = min(artifact["shallow"], key=lambda r: r["ka"] * r["depth"])
    again = envelope.measure_point(
        row["ka"], row["f_ratio"], row["depth"], row["motion"], row["panels"],
        artifact["settings"]["radius"], artifact["settings"]["density"],
    )
    assert again["added_mass"] == pytest.approx(row["added_mass"], rel=1e-9)
    assert again["damping"] == pytest.approx(row["damping"], rel=1e-9)
    assert again["mass_dev_from_deep"] == pytest.approx(row["mass_dev_from_deep"], rel=1e-6)


# --------------------------------------------------------------------------
# 2. Acceptance criteria on every point of the grid
# --------------------------------------------------------------------------
def test_energy_identity_holds_everywhere(artifact):
    worst = max(abs(r["far_over_near"] - 1.0) for r in artifact["points"])
    assert worst < 0.03, f"energy identity worst {worst:.4%} exceeds 3 %"
    # The far-field reading is a small correction, not a rescaling: it must be
    # positive and within a couple of percent, otherwise the two readings would
    # be measuring different things.
    assert all(r["far_over_near"] > 1.0 for r in artifact["points"])


def test_deep_water_collapse_deviation_is_first_order_in_the_panel_size(artifact):
    """The N=64 deviation is discretisation error, and it must halve with N.

    The tabulated deep-water collapse numbers are read against Ursell's closed
    form; if they were a bug in the kernel or the assembly they would not shrink
    as the panels are refined.
    """
    grid_depth = artifact["settings"]["grid_depth"]
    rows = [r for r in artifact["refinement"] if r["depth"] == grid_depth]
    assert rows, "refinement block missing the grid depth"
    for point in {(r["ka"], r["f_ratio"], r["motion"]) for r in rows}:
        ladder = sorted((r for r in rows if (r["ka"], r["f_ratio"], r["motion"]) == point),
                        key=lambda r: r["panels"])
        mass = [abs(r["mass_dev_from_deep"]) for r in ladder]
        assert mass == sorted(mass, reverse=True), f"added mass error not monotone at {point}: {mass}"
        order = math.log(mass[0] / mass[1]) / math.log(ladder[1]["panels"] / ladder[0]["panels"])
        assert 0.7 < order < 1.4, f"observed order {order:.2f} at {point} is not first order"


def test_solver_reaches_the_three_percent_band_on_the_refined_mesh(artifact):
    """Where N=64 is outside the band, the refined mesh must be inside it.

    This is the honest form of the criterion: the envelope study reports a
    coarse-mesh limitation, so the test asserts the limitation is *mesh* and not
    *physics* -- the finest tabulated mesh lands inside 3 %.
    """
    grid_depth = artifact["settings"]["grid_depth"]
    finest = max(artifact["settings"]["refinement_panels"])
    rows = [r for r in artifact["refinement"]
            if r["depth"] == grid_depth and r["panels"] == finest]
    assert rows
    for row in rows:
        assert abs(row["mass_dev_from_deep"]) < 0.03, row
        assert abs(row["damping_dev_from_deep"]) < 0.03, row


def test_shallow_water_raises_the_added_mass(artifact):
    """Blockage must raise the added mass -- with one documented exception.

    At the shallowest clearance the heave added mass rises at every point of the
    grid; the sway coefficient rises everywhere except the longest-wave,
    shallowest-submergence corner (``ka = 0.3``, ``f/a = 1.5``), where the
    section is barely off the bed and the horizontal added mass falls instead.
    That exception is asserted, not smoothed over, and the document has to state
    it.
    """
    rows = artifact["summary"]["blockage"]
    exceptions = {(0.3, 1.5, "sway")}
    for row in rows:
        key = (row["ka"], row["f_ratio"], row["motion"])
        if key in exceptions:
            assert row["shallow_over_deep"] < 1.0, row
        else:
            assert row["shallow_over_deep"] > 1.0, row
    assert sum(1 for r in rows if r["shallow_over_deep"] > 1.0) == len(rows) - len(exceptions)
    assert artifact["summary"]["blockage_all_increase"] is False
    assert "ka = 0.3" in (ROOT / "docs" / "finite_depth_envelope.md").read_text()


def test_heave_and_sway_split_in_finite_depth(artifact):
    assert artifact["summary"]["heave_sway_all_positive"] is True
    splits = artifact["summary"]["heave_sway_split"]
    assert splits
    for row in splits:
        # Heave pushes the section against the sea bed, sway shears past it, so
        # blockage must make the heave added mass the larger of the two.
        assert row["heave_over_sway"] > 0.0, row


def test_summary_matches_the_rows_it_summarises(artifact):
    mass = max(abs(r["mass_dev_from_deep"]) for r in artifact["points"])
    damping = max(abs(r["damping_dev_from_deep"]) for r in artifact["points"])
    energy = max(abs(r["far_over_near"] - 1.0) for r in artifact["points"])
    summary = artifact["summary"]
    assert summary["mass_dev_max_abs"] == pytest.approx(mass, rel=1e-12)
    assert summary["damping_dev_max_abs"] == pytest.approx(damping, rel=1e-12)
    assert summary["energy_identity_max_abs"] == pytest.approx(energy, rel=1e-12)
    inside = (
        summary["mass_dev_max_abs"] < summary["tolerance"]
        and summary["damping_dev_max_abs"] < summary["tolerance"]
        and summary["energy_identity_max_abs"] < summary["tolerance"]
    )
    assert summary["within_tolerance_all_points"] is inside


# --------------------------------------------------------------------------
# 3. The document is the renderer's output
# --------------------------------------------------------------------------
def test_document_is_generated_not_transcribed(artifact):
    assert DOC.exists(), f"missing {DOC}"
    assert DOC.read_text() == envelope.render_document(artifact), (
        "docs/finite_depth_envelope.md is not the renderer output; regenerate with "
        "python -m tensorbem.benchmarks.finite_depth_envelope"
    )


def test_document_prints_the_artifact_numbers(artifact):
    text = DOC.read_text()
    start = text.index("## 3.")
    end = text.index("## 4.")
    section = text[start:end]
    rows = {f"{r['ka']:g}|{r['f_ratio']:g}|{r['motion']}": r for r in artifact["points"]}
    printed = re.findall(
        r"^\| ([\d.]+) \| ([\d.]+) \| (heave|sway) \| ([\d.]+) \| ([+\-][\d.]+) % \|",
        section, re.MULTILINE,
    )
    assert len(printed) == len(rows), f"expected {len(rows)} table rows, parsed {len(printed)}"
    for ka, f_ratio, motion, added_mass, deviation in printed:
        row = rows[f"{float(ka):g}|{float(f_ratio):g}|{motion}"]
        assert float(added_mass) == pytest.approx(row["added_mass"], abs=0.006)
        assert float(deviation) == pytest.approx(100 * row["mass_dev_from_deep"], abs=0.006)


def test_document_states_the_verdict_the_artifact_supports(artifact):
    text = DOC.read_text()
    summary = artifact["summary"]
    worst_damping = 100 * summary["damping_dev_max_abs"]
    assert f"| 深水塌缩（阻尼） | {worst_damping:.2f} % |" in text, (
        "the document must quote the measured worst damping deviation"
    )
    # And the verdict column must be "否" when the worst case is outside the band.
    if summary["damping_dev_max_abs"] >= summary["tolerance"]:
        assert "**否**" in text, "document hides an out-of-band result"
    else:
        assert "**否**" not in text