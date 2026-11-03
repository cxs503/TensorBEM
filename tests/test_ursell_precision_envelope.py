"""Regression guards for the Ursell precision envelope (docs/ursell_precision_envelope.md).

Three things are protected:

1. the *documented envelope* — every quoted cell must still be produced by the
   shipped solver (drift in either direction is a red test, not a footnote);
2. the *shape* of the envelope — first-order convergence and the fact that
   shallow submergence is the accuracy-limiting regime;
3. the bookkeeping code (orders, ``N*eps``, recommendations) — tested on
   synthetic rows so it cannot be "verified" by the same numbers it produces.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorbem.benchmarks.ursell_cylinder import run
from tensorbem.benchmarks.ursell_precision_envelope import summarise

ENVELOPE = Path(__file__).resolve().parents[1] / "docs" / "ursell_precision_envelope" / "envelope.json"


@pytest.fixture(scope="module")
def envelope() -> dict:
    if not ENVELOPE.exists():
        pytest.skip("envelope.json not generated (run the benchmark module first)")
    return json.loads(ENVELOPE.read_text())


def _row(envelope: dict, ka: float, f_over_a: float, panels: int) -> dict:
    for row in envelope["rows"]:
        if (
            math.isclose(row["ka"], ka, rel_tol=1e-12)
            and math.isclose(row["f_over_a"], f_over_a, rel_tol=1e-12)
            and row["panels"] == panels
        ):
            return row
    raise KeyError((ka, f_over_a, panels))


def _cell(envelope: dict, ka: float, f_over_a: float) -> dict:
    for cell in envelope["cells"]:
        if math.isclose(cell["ka"], ka, rel_tol=1e-12) and math.isclose(
            cell["f_over_a"], f_over_a, rel_tol=1e-12
        ):
            return cell
    raise KeyError((ka, f_over_a))


class TestBookkeeping:
    """The order/recommendation code, exercised on synthetic rows."""

    @staticmethod
    def _rows(errors: list[float]) -> list[dict]:
        panels = [32, 64, 128, 256][: len(errors)]
        rows = []
        for n, err in zip(panels, errors):
            rows.append(
                {
                    "ka": 0.6,
                    "f_over_a": 2.0,
                    "panels": n,
                    "added_mass_relative_error": err,
                    "damping_relative_error": err,
                    "added_mass_abs_error_nondim": err,
                    "damping_abs_error_nondim": err,
                    "added_mass_nondim": 1.0,
                    "damping_nondim": 1.0,
                    "reference_added_mass_nondim": 1.0,
                    "reference_damping_nondim": 1.0,
                    "damping_near_over_far": 1.0,
                    "mfs_added_mass_relative_error": 0.0,
                    "mfs_damping_relative_error": 0.0,
                    "panel_condition_number": 1.0,
                    "panel_relative_residual": 0.0,
                }
            )
        return rows

    def test_first_order_sequence_gives_order_one(self) -> None:
        rows = self._rows([1.0e-1, 5.0e-2, 2.5e-2, 1.25e-2])
        cell = summarise(rows)["cells"][0]
        assert cell["added_mass_order"]["orders"] == pytest.approx([1.0, 1.0, 1.0])
        # N * eps constant to <1% for a clean first-order sequence
        drift = cell["added_mass_order"]["n_times_error_drift"]
        assert drift == pytest.approx(1.0, rel=1e-6)

    def test_recommendation_picks_smallest_passing_mesh(self) -> None:
        rows = self._rows([1.0e-1, 5.0e-2, 2.5e-2, 1.25e-2])
        cell = summarise(rows)["cells"][0]
        assert cell["recommended_panels"]["added_mass_0.03"] == 128
        assert cell["recommended_panels"]["added_mass_0.01"] is None  # 1.25e-2 > 1e-2

    def test_scale_free_error_is_used_at_the_zero_crossing(self) -> None:
        rows = self._rows([1.0e-1, 5.0e-2, 2.5e-2])
        for row in rows:
            row["added_mass_abs_error_nondim"] = 3.0e-2
        cell = summarise(rows)["cells"][0]
        assert cell["added_mass_abs_error_nondim_at_128"] == pytest.approx(3.0e-2)


class TestMeasuredEnvelope:
    """The shipped solver must still reproduce the committed envelope."""

    def test_pinned_coarse_cells_are_reproduced(self, envelope: dict) -> None:
        for ka, f_over_a, panels in ((0.6, 2.0, 32), (0.6, 1.2, 32), (1.2, 2.0, 64)):
            expected = _row(envelope, ka, f_over_a, panels)
            measured = run(panel_count=panels, submergence=f_over_a, ka=ka)
            assert measured["panel_added_mass"] == pytest.approx(
                expected["panel_added_mass"], rel=1e-9
            )
            assert measured["panel_damping"] == pytest.approx(
                expected["panel_damping"], rel=1e-9
            )
            assert measured["panel_added_mass_relative_error"] == pytest.approx(
                expected["added_mass_relative_error"], rel=1e-9
            )

    def test_reference_is_independently_confirmed(self, envelope: dict) -> None:
        """MFS (independent discretisation) confirms the multipole solution off the corner.

        At ``f/a <= 1.2`` the 24-source MFS cannot resolve the close free surface
        (documented in the envelope, section 5 item 5), so the guard is restricted
        to the settled regime instead of being silently loosened everywhere.
        """
        for cell in envelope["cells"]:
            if cell["f_over_a"] < 1.5:
                continue
            for value in cell["mfs_added_mass_relative_error"]:
                assert value < 1.0e-2, (cell["ka"], cell["f_over_a"], value)

    def test_first_order_holds_off_the_near_surface_corner(self, envelope: dict) -> None:
        for cell in envelope["cells"]:
            if cell["f_over_a"] < 1.5:
                continue
            for order in cell["added_mass_order"]["orders"]:
                assert 0.85 < order < 1.3, (cell["ka"], cell["f_over_a"], order)
            drift = cell["added_mass_order"]["n_times_error_drift"]
            assert drift < 2.5, (cell["ka"], cell["f_over_a"], drift)

    def test_near_surface_corner_is_the_accuracy_limiting_case(self, envelope: dict) -> None:
        """The (low ka, f/a -> 1) corner is 2.8x worse than the deep case at N = 128."""
        deep = _cell(envelope, 0.4, 3.0)["added_mass_abs_error_nondim_at_128"]
        corner = _cell(envelope, 0.4, 1.1)["added_mass_abs_error_nondim_at_128"]
        assert corner > 2.0 * deep

    def test_corner_is_non_monotone(self, envelope: dict) -> None:
        """Documented negative result: refinement is not monotone at (ka=0.9, f/a=1.1)."""
        cell = _cell(envelope, 0.9, 1.1)
        assert min(cell["added_mass_order"]["orders"]) < 0.0
        assert cell["added_mass_abs_error_nondim_at_128"] < 1.0e-2

    def test_documented_bars_are_met_or_reported(self, envelope: dict) -> None:
        """Every cell is either inside 3 % at N = 256, or explicitly flagged."""
        bars = envelope["bars"]
        for cell in envelope["cells"]:
            lateral = cell["added_mass_abs_error_nondim"][-1]
            assert lateral < bars[1] or cell["recommended_panels"]["added_mass_0.03"] is None

    def test_radiation_condition_is_mesh_limited(self, envelope: dict) -> None:
        """Near/far damping offset must shrink with N and stay ~1 %."""
        for cell in envelope["cells"]:
            ratios = cell["damping_near_over_far"]
            assert all(0.99 < r < 1.03 for r in ratios), (cell["ka"], cell["f_over_a"], ratios)
            assert abs(ratios[-1] - 1.0) < abs(ratios[0] - 1.0)