"""Pins for the corrected zero-frequency heave anchor of the floating hemisphere.

The audited artifact is ``docs/hemisphere_zero_frequency_envelope/`` (built by
``python -m tensorbem.benchmarks.hemisphere_zero_frequency_envelope``).  The
tests are organised so that a wrong *physics* claim fails here and not in a
later audit:

* the spectral route is re-run from scratch and its self test (classical
  translating full sphere) is asserted to machine precision, so the route is
  known to integrate the right boundary data before it is used to judge the
  rigid-lid value;
* the corrected rigid-lid value is asserted against the published
  ``0.8309570 rho*grad`` and against the *exact* Hulme series at ``ka = 2e-3``
  (two independent routes, 0.85 % apart);
* the two premises of the superseded m16802 anchor are asserted to be
  **falsified** (the anchor is 16.9 % high; the "0.5 floor" is below the exact
  interior minimum 0.386), so a future edit cannot quietly restore them;
* the pointwise accuracy envelope of the committed 3-D sweep is asserted from
  the rows (max |error| < 3 % on 256 panels), and the narrative document is
  checked to quote only numbers that exist in the generated summary.
"""

from __future__ import annotations

import json
import math
import pathlib

import pytest

from tensorbem.benchmarks import hemisphere_zero_frequency_envelope as envelope

DOC_DIR = pathlib.Path(__file__).resolve().parents[1] / "docs" / "hemisphere_zero_frequency_envelope"
NARRATIVE = pathlib.Path(__file__).resolve().parents[1] / "docs" / "hemisphere_zero_frequency_envelope.md"


@pytest.fixture(scope="module")
def payload():
    path = DOC_DIR / "results.json"
    if not path.exists():
        pytest.skip("envelope artifact not built")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Route 1: the spectral solve, re-run from scratch
# ---------------------------------------------------------------------------
def test_translation_self_test_recovers_the_classical_full_sphere():
    """The projector is validated on the case whose answer is classical."""
    a33, ratio = envelope.spectral_added_mass("translation", 60)
    classical = (2.0 / 3.0) * envelope.DENSITY * math.pi * envelope.RADIUS**3
    assert a33 == pytest.approx(0.5 * classical, rel=1e-10)
    assert ratio == pytest.approx(0.5, rel=1e-12)


def test_spectral_rigid_lid_value_is_converged_and_correct():
    """Sliding-halves route: A33(ka -> 0) = 0.8309570 rho*grad."""
    values = {n: envelope.spectral_added_mass("sliding", n)[1] for n in (40, 60, 100, 200)}
    spread = max(values.values()) - min(values.values())
    assert spread < 1e-5, values
    assert values[200] == pytest.approx(0.8309570, abs=2e-7)
    # The published artifact must carry the same converged value.
    a33 = envelope.rigid_lid_heave_added_mass()
    assert a33 == pytest.approx(values[200] * 2146.754979953025, rel=1e-9)


def test_rigid_lid_limit_is_not_the_imaged_sphere():
    """The m16802 premise: the even-symmetry image is a *sliding* pair."""
    limit = envelope.spectral_added_mass("sliding", 200)[1]
    imaged = envelope.imaged_sphere_translation_added_mass() / (
        envelope.DENSITY * (2.0 / 3.0) * math.pi
    )
    assert abs(imaged - 1.0) < 1e-12
    # 16.9 % apart: the legacy anchor cannot be the rigid-lid value.
    assert (imaged - limit) / imaged > 0.15
    assert limit < 1.0


# ---------------------------------------------------------------------------
# Route 2: the exact reference, re-evaluated
# ---------------------------------------------------------------------------
def test_exact_reference_approaches_the_spectral_limit_from_above():
    rows = envelope.exact_reference_table((2e-3, 5e-3, 1e-2))
    limit = envelope.spectral_added_mass("sliding", 200)[1]
    ratios = [row["a33_over_rho_displaced"] for row in rows]
    assert all(r > limit for r in ratios), ratios
    assert ratios[0] < ratios[1] < ratios[2], "excess must grow with ka"
    assert (ratios[0] - limit) / limit < 0.01


def test_exact_table_in_the_artifact_regenerates():
    """No hand transcription: the committed table is this code's output."""
    committed = json.loads((DOC_DIR / "results.json").read_text(encoding="utf-8"))
    rows = {round(row["ka"], 6): row for row in committed["exact_table"]}
    for ka in (0.1, 2.0):
        live = envelope.exact_reference_table((ka,))[0]
        assert live["a33_over_rho_displaced"] == pytest.approx(
            rows[round(ka, 6)]["a33_over_rho_displaced"], rel=1e-9
        )


def test_exact_added_mass_dips_below_the_pressure_release_value():
    dip = envelope._minimum_exact_dip()
    release = envelope.pressure_release_heave_added_mass() / (
        envelope.DENSITY * (2.0 / 3.0) * math.pi
    )
    assert dip["a33_over_rho_displaced"] < release * 0.8
    assert dip["a33_over_rho_displaced"] == pytest.approx(0.3863, abs=5e-3)


# ---------------------------------------------------------------------------
# The 3-D solver envelope
# ---------------------------------------------------------------------------
def test_committed_bem_sweep_is_within_three_percent_of_exact(payload):
    finest = max(payload["bem_vs_exact"], key=lambda item: item["panels"])
    worst = max(finest["rows"], key=lambda row: abs(row["relative_error"]))
    assert abs(worst["relative_error"]) < 0.03, worst
    for row in finest["rows"]:
        assert row["a33_exact"] > 0.0
        assert row["relative_error"] == pytest.approx(
            (row["a33_bem"] - row["a33_exact"]) / row["a33_exact"], rel=1e-12
        )


def test_live_rigid_lid_probe_matches_the_spectral_limit(payload):
    """The solver at ka = 1e-3 *is* the rigid-lid problem; it must land on the limit."""
    probe = payload["rigid_lid_probes"]
    finest = max(probe, key=lambda row: row["subdivisions"])
    assert finest["relative_error_vs_spectral"] == pytest.approx(
        (finest["a33"] - envelope.rigid_lid_heave_added_mass())
        / envelope.rigid_lid_heave_added_mass(),
        rel=1e-9,
    )
    assert abs(finest["relative_error_vs_spectral"]) < 0.01


# ---------------------------------------------------------------------------
# Artifact self-consistency
# ---------------------------------------------------------------------------
def test_criteria_recompute_from_the_payload(payload):
    recomputed = envelope.evaluate_criteria(payload)
    stored = {item["id"]: item for item in payload["criteria"]}
    assert [item["id"] for item in recomputed] == [item["id"] for item in payload["criteria"]]
    for item in recomputed:
        assert stored[item["id"]]["pass"] == item["pass"]
        assert stored[item["id"]]["relative_error"] == pytest.approx(
            item["relative_error"], rel=1e-9, abs=1e-12
        )
    assert payload["all_pass"] == all(item["pass"] for item in payload["criteria"])
    assert payload["all_pass"] is True, "the corrected gate must close"


def test_summary_and_figures_are_generated(payload):
    summary = (DOC_DIR / "summary.md").read_text(encoding="utf-8")
    needles = [
        "| ka -> 0 (rigid lid, sliding halves) | %.4f | %.6f |"
        % (payload["limits"]["rigid_lid_a33"], payload["limits"]["rigid_lid_over_rho_displaced"]),
        "| m16802 anchor (imaged sphere translation) | %.4f | %.6f |"
        % (payload["limits"]["imaged_sphere_legacy_a33"],
           payload["limits"]["imaged_sphere_legacy_over_rho_displaced"]),
        "**all_pass = True**",
    ]
    for needle in needles:
        assert needle in summary, needle
    for name in payload["figures"]:
        path = DOC_DIR / name
        assert path.stat().st_size > 5000, name


@pytest.mark.parametrize(
    "needle",
    ["0.830957", "1783.8612", "0.38629", "0.828152", "-0.74%", "0.86275", "0.85636",
     "1777.8388", "1733.2639", "-2.84%", "-0.34%"],
)
def test_narrative_document_only_quotes_generated_numbers(needle):
    """Anti-hand-copy: every number the prose cites exists in the generated summary."""
    if not NARRATIVE.exists():
        pytest.skip("narrative document absent")
    summary = (DOC_DIR / "summary.md").read_text(encoding="utf-8")
    text = NARRATIVE.read_text(encoding="utf-8")
    assert needle in text, f"narrative no longer cites {needle!r}"
    assert needle in summary, f"cited number {needle!r} is not in the generated summary"
