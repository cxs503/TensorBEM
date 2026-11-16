"""The dense tail probe must report the same exponent convention as gate C9.

``tensorbem.causality.local_tail_exponent`` reports ``p`` in ``B ~ omega^-p``
(positive for a decaying tail).  If the probe reported the raw log-log slope
instead, every DOF would sit on the wrong side of the ``2.05`` bar and the probe
would "prove" the opposite of what the data say.  These tests pin the sign, the
span bookkeeping and the incompleteness refusal.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

WORKTREE = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location(
        "cummins_tail_dense_probe", WORKTREE / "scripts" / "cummins_tail_dense_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records(probe, exponent: float) -> dict[int, dict]:
    records = {}
    for index, omega in enumerate(probe.grid()):
        diagonal = [
            float((dof + 1) * 0.5 * omega ** (-exponent)) for dof in range(6)
        ]
        records[index] = {
            "index": index,
            "omega": float(omega),
            "damping_diagonal": diagonal,
            "added_mass_diagonal": [1.0] * 6,
            "solver_relative_residual": 1e-10,
            "panel_count": 3840,
            "max_panel_scale": 0.023,
            "n_sections": 60,
            "n_levels": 16,
        }
    return records


def test_exponent_convention_matches_causality(probe):
    payload = probe.analyze(_records(probe, 3.0))
    for row in payload["per_dof"]:
        for span, fit in row["slope_by_span"].items():
            assert fit["slope"] == pytest.approx(3.0, abs=1e-8), (row["dof_name"], span)
            assert fit["log_log_slope"] == pytest.approx(-3.0, abs=1e-8)
        assert row["decidable_long_span"] is True
        assert row["decidable_short_span"] is True


def test_shallow_tail_is_reported_undecidable(probe):
    payload = probe.analyze(_records(probe, 1.2))
    row = payload["per_dof"][0]
    assert row["decidable_long_span"] is False
    assert row["slope_spread_over_spans"] == pytest.approx(0.0, abs=1e-6)
    assert payload["verdict"]["span_sensitive_dof"] == []


def test_span_decades_grow_with_the_span(probe):
    payload = probe.analyze(_records(probe, 2.5))
    spans = payload["per_dof"][0]["slope_by_span"]
    order = [str(span) for span in probe.SPANS]
    decades = [spans[key]["decades"] for key in order]
    assert decades == sorted(decades)
    assert decades[0] < 0.05 and decades[-1] == pytest.approx(
        np.log10(probe.DENSE_BAND[1] / probe.DENSE_BAND[0]), abs=1e-6
    )


def test_analyze_refuses_incomplete_probe(probe):
    records = _records(probe, 3.0)
    records.pop(11)
    with pytest.raises(SystemExit):
        probe.analyze(records)


def test_probe_band_sits_inside_the_declared_band(probe):
    nodes = probe.grid()
    assert nodes[0] == pytest.approx(probe.DENSE_BAND[0])
    assert nodes[-1] == pytest.approx(probe.DENSE_BAND[1])
    assert probe.MESH == (60, 16)
    assert nodes.size == probe.DENSE_N_OMEGA