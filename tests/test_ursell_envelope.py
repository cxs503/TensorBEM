"""Guards for the Ursell frequency/clearance envelope (``ursell_envelope``).

The envelope study closes the open item of ``docs/ursell_error_budget.md`` §5.2
("only one ``(ka, a/f)`` point was audited; ``ka >= 1`` unverified").  These
tests keep it honest in three ways:

* the *reference* side is asserted to have settled (40 vs 120 harmonics),
* two grid corners are recomputed live, so the envelope cannot drift away from
  the code it describes,
* the committed artifact, its summary and the published document are required
  to agree with each other, so a stale table fails the suite instead of living
  on in ``docs/`` (the Ursell 7.7e-3 defect of 2026-10-24 was exactly that).

The live solves are 128-panel 2-D solves (a few seconds each); everything else
is closed-form or file parsing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorbem.benchmarks.ursell_envelope import (
    DEPTH_RATIOS,
    FAIL_BAND,
    KA_VALUES,
    PASS_BAND,
    _observed_order,
    classify,
    reference_truncation,
    summarise,
    _panel,
)

ARTIFACT = Path("artifacts/ursell_envelope.json")
DOCUMENT = Path("docs/ursell_frequency_envelope.md")


def test_reference_plateau_holds_over_the_whole_grid() -> None:
    """The truncated multipole reference must be settled at every grid point.

    ``ursell_cylinder_coefficients`` truncates an infinite system.  If 40 and
    120 harmonics differ, the panel "error" would be measured against a
    partially converged reference and the envelope would be meaningless.
    """
    for ka in KA_VALUES:
        for ratio in DEPTH_RATIOS:
            block = reference_truncation(ka, ratio)
            shift = max(block["added_mass_shift"], block["damping_shift"])
            assert shift < 1.0e-10, f"reference not settled at ka={ka}, a/f={ratio}: {shift}"


def test_audited_point_is_reproduced() -> None:
    """``(ka, a/f) = (0.6, 0.5)`` at 128 panels: the historical numbers.

    m2173 measured 8.513e-3 / 1.336e-2 after the kernel refactor.  Pinning them
    here means a regression at the audited point cannot hide behind the new
    grid.
    """
    mu, nu = _panel(0.6, 0.5, 128)
    reference = reference_truncation(0.6, 0.5)
    error_mu = abs(mu - reference["mu_converged"]) / reference["mu_converged"]
    error_nu = abs(nu - reference["nu_converged"]) / reference["nu_converged"]
    assert error_mu == pytest.approx(8.513e-3, rel=5.0e-2)
    assert error_nu == pytest.approx(1.336e-2, rel=5.0e-2)


def test_high_frequency_corner_of_the_grid() -> None:
    """``ka = 2`` at ``a/f = 0.5``: the ``ka >= 1`` regime the audit left open."""
    mu, nu = _panel(2.0, 0.5, 128)
    reference = reference_truncation(2.0, 0.5)
    error_mu = abs(mu - reference["mu_converged"]) / reference["mu_converged"]
    error_nu = abs(nu - reference["nu_converged"]) / reference["nu_converged"]
    assert max(error_mu, error_nu) < PASS_BAND, (error_mu, error_nu)
    # The added mass is O(0.8) at both ends of the range, the damping drops by
    # 1.5 orders -- the normalisation must not be silently reused.
    assert reference["nu_converged"] < 0.05


def test_classification_bands() -> None:
    assert classify(1.0e-3, 2.0e-2) == "pass"
    assert classify(1.0e-2, 5.0e-2) == "marginal"
    assert classify(2.0e-2, 3.0e-1) == "fail"
    assert PASS_BAND < FAIL_BAND


def test_observed_order_from_an_exact_refinement() -> None:
    assert _observed_order(1.0, 0.25) == pytest.approx(2.0)
    assert _observed_order(0.0, 0.25) is None


def test_artifact_structure_and_summary_consistency() -> None:
    payload = json.loads(ARTIFACT.read_text())
    grid = payload["grid"]
    assert len(grid) == len(KA_VALUES) * len(DEPTH_RATIOS)
    keys = {(row["ka"], row["a_over_f"]) for row in grid}
    for ka in KA_VALUES:
        for ratio in DEPTH_RATIOS:
            assert (ka, ratio) in keys
    # The stored summary must be the summary of the stored grid, not a
    # hand-edited claim.
    assert payload["summary"] == summarise(grid)
    assert payload["summary"]["max_reference_shift_40_to_120"] < 1.0e-10
    # The independent (MFS) route must not share the panel error, otherwise the
    # grid would be comparing the panel code with itself.
    for row in grid:
        assert max(
            row["mfs_added_mass_relative_error"], row["mfs_damping_relative_error"]
        ) < 2.0e-2, row
    assert payload["summary"]["grid_points"] == len(grid)


def test_document_quotes_the_artifact() -> None:
    """The published document must carry the measured numbers, not older ones."""
    payload = json.loads(ARTIFACT.read_text())
    worst = payload["summary"]["worst_point"]
    text = DOCUMENT.read_text(encoding="utf-8")
    for value in (
        worst["added_mass_relative_error"],
        worst["damping_relative_error"],
    ):
        assert f"{value:.3e}" in text, f"{value:.3e} missing from {DOCUMENT}"
    counts = payload["summary"]["verdict_counts"]
    assert f"pass {counts['pass']}" in text
    # And the number of grid points must be quoted, so a shrunken grid shows up.
    assert f"{payload['summary']['grid_points']} " in text


def test_refinement_order_is_first_order_in_the_tail() -> None:
    """Constant panels on a curved body: the documented tail order is ~1."""
    payload = json.loads(ARTIFACT.read_text())
    for row in payload.get("refinement") or []:
        if row["panels"] != 256:
            continue
        for key in ("added_mass_observed_order", "damping_observed_order"):
            order = row.get(key)
            if order is None:
                continue
            assert math.isfinite(order)
            assert 0.4 < order < 1.6, (row["ka"], row["a_over_f"], key, order)