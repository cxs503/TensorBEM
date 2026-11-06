"""Gates for the finite-depth / deep-water kernel arbitration (m4776).

The open item in ``docs/finite_depth3d_regime.md`` §5 asked *which* of the two
3-D free-surface kernels loses accuracy at ``k0*h = 10``, ``k0*R = 50``, where
they differ by 85 %.  These gates pin the arbitration answer:

* the deep-water kernel reproduces an independent principal-value evaluation of
  its own defining Havelock integral (no region machinery, no asymptotics);
* the finite-depth eigenseries is mode-converged at the same geometry;
* the 85 % is a cancellation artefact: it collapses below 1 % once the
  difference is normalised by the local Rankine scale ``1/(4 pi r)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.benchmarks.finite_depth3d_deepwater_arbitration import (
    arbitrate,
    referee_check,
    referee_wave_function,
    verdict,
)
from tensorbem.free_surface3d import wave_function_3d

NU = 50.0


@pytest.fixture(scope="module")
def referee() -> dict:
    return referee_check(NU, portrait=50.0)


@pytest.fixture(scope="module")
def ladder() -> dict:
    return arbitrate(NU, k0h_ladder=(10.0, 50.0, 250.0), portrait=50.0)


def test_referee_reproduces_the_kernel_at_the_contested_point(referee: dict) -> None:
    row = next(item for item in referee["sweep"] if item["X"] == 50.0)
    assert row["Y"] == pytest.approx(10.0)
    kernel, _, _ = wave_function_3d(np.array([50.0]), np.array([10.0]))
    assert float(kernel[0]) == pytest.approx(row["referee_F"], rel=1.0e-5)
    # the referee is a genuinely different algorithm; it must not be exact by
    # construction, but it must agree far below the 85 % being contested
    assert 1.0e-12 < row["relative_error"] < 1.0e-5


def test_deep_water_kernel_is_certified_over_the_sweep(referee: dict) -> None:
    assert referee["max_relative_error_over_sweep"] < 1.0e-5
    # monotone decay of |F| in X must survive in both evaluations
    kernel = [abs(item["kernel_F"]) for item in referee["sweep"]]
    reference = [abs(item["referee_F"]) for item in referee["sweep"]]
    assert all(later < earlier for earlier, later in zip(kernel, kernel[1:]))
    assert all(later < earlier for earlier, later in zip(reference, reference[1:]))


def test_finite_depth_series_is_mode_converged(ladder: dict) -> None:
    for row in ladder["ladder"]:
        assert row["mode_spread"] < 1.0e-14
        auto = row["G_finite_depth_modes"]["None"]
        many = row["G_finite_depth_modes"]["160"]
        assert abs(complex(*auto) - complex(*many)) == pytest.approx(0.0, abs=1.0e-14)


def test_the_85_percent_is_a_cancellation_artefact(ladder: dict) -> None:
    row = next(item for item in ladder["ladder"] if item["k0h"] == 10.0)
    assert row["relative_deviation"] > 0.8
    assert row["cancellation_number"] > 100.0
    # the same difference is well under 1 % of the local Rankine scale
    assert row["deviation_over_local_scale"] < 1.0e-2
    # and the amplification is exactly the cancellation number
    assert row["relative_deviation"] / row["deviation_over_local_scale"] == pytest.approx(
        row["cancellation_number"], rel=1.0e-2
    )


def test_verdict_reports_neither_kernel_as_defective(referee: dict, ladder: dict) -> None:
    judgement = verdict(ladder, referee)
    assert judgement["deep_kernel_ok"] is True
    assert judgement["worst_relative_deviation_point"]["k0h"] == 10.0
    assert judgement["worst_relative_deviation"] == pytest.approx(0.8534, abs=2.0e-3)
    assert "metric artefact" in judgement["resolution"]


def test_referee_rejects_a_wrong_sign() -> None:
    """Negative control: the referee is not blind to an algebraic mistake."""
    value = referee_wave_function(50.0, 10.0)
    assert value < 0.0  # F(X, Y) is positive-definite only in magnitude, not sign
    assert abs(abs(value) - 1.9666157510e-2) < 1.0e-4