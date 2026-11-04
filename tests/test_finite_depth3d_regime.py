"""Regime gates for the 3-D finite-depth free-surface kernel.

Three properties that the shipped suite did not cover (see
:mod:`tensorbem.benchmarks.finite_depth3d_regime`):

* the long-wave (shallow-water) collapse ``G -> (i/(4h))*H_0^{(1)}(k0*R)`` and
  its closed-form leading error ``(k0*h)**2/12`` for mid-depth points;
* the deep-water collapse tested at *fixed nu* up to ``k0*h = 4e4``, i.e. beyond
  the ``cosh``/``N_0`` overflow range where the kernel used to return ``nan``;
* finiteness of the analytic ``dG/dz`` in that same regime.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.benchmarks.finite_depth3d_regime import (
    deep_water_regime,
    run,
    shallow_water_limit,
)
from tensorbem.finite_depth3d import (
    _propagating_factors_3d,
    finite_depth_wavenumber,
    green_function_3d,
    green_gradient_z_3d,
    wave_amplitude_coefficient,
)
from tensorbem.finite_depth import propagating_mode_norm

Z_MID = -0.5


# ---------------------------------------------------------------------------
# 1. scaled propagating factors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k0h", [0.05, 0.5, 5.0, 40.0])
def test_scaled_factors_reproduce_cosh_ratio(k0h):
    """``u(z)*u(z0)/w`` must equal ``cosh*cosh/N_0`` where nothing overflows."""
    depth = 1.0
    k0 = k0h / depth
    nu = k0 * math.tanh(k0h)
    u, _, w = _propagating_factors_3d(k0, depth)
    z = np.array([Z_MID, -0.25, -0.9])
    scaled = (u(z) / math.sqrt(w)) ** 2
    direct = np.cosh(k0 * (z + depth)) ** 2 / propagating_mode_norm(k0, depth)
    assert np.allclose(scaled, direct, rtol=1e-11)


def test_wave_amplitude_coefficient_stays_finite_in_deep_water():
    """``phi_0/N_0`` underflows to zero instead of hitting ``inf/inf = nan``."""
    nu, depth = 50.0, 400.0
    z = np.array([Z_MID])
    value = wave_amplitude_coefficient(z, nu, depth)
    assert np.all(np.isfinite(value))
    # phi_0/N_0 ~ 4*k0*exp(k0*(z - h)) is genuinely below the double range here,
    # so a hard zero is correct; the defect was inf/inf = nan for the same input.
    assert value[0] <= 1e-100
    k0 = finite_depth_wavenumber(nu, depth)
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.cosh(k0 * (z + depth)) / propagating_mode_norm(k0, depth)
    assert not np.all(np.isfinite(naive))


# ---------------------------------------------------------------------------
# 2. long-wave (shallow-water) collapse
# ---------------------------------------------------------------------------
def test_shallow_water_limit_collapses_to_helmholtz_kernel():
    rows = shallow_water_limit(depth=1.0, k0h_ladder=(0.02, 0.05, 0.1, 0.2))
    for row in rows:
        assert row["relative_deviation"] < 4e-3, row
        # closed-form leading correction for mid-depth elevations
        ratio = row["relative_deviation"] / row["analytic_deviation"]
        assert abs(ratio - 1.0) < 0.02, row


def test_shallow_water_limit_is_second_order_in_k0h():
    payload = run()
    for order in payload["shallow_water_limit"]["orders"][:-1]:
        assert 1.9 <= order["order"] <= 2.1, order


def test_shallow_water_limit_is_independent_of_r_over_h():
    """Evanescent modes die like exp(-n*pi*R/h): the collapse is geometry free."""
    rows = shallow_water_limit(depth=2.0, k0h_ladder=(0.1,), r_over_h=(5.0, 10.0, 20.0))
    deviations = [row["relative_deviation"] for row in rows]
    assert max(deviations) / min(deviations) < 1.001


# ---------------------------------------------------------------------------
# 3. deep-water collapse, including the previously NaN regime
# ---------------------------------------------------------------------------
def test_deep_water_collapse_survives_cosh_overflow():
    """Regression: ``k0*h > 700`` used to return ``nan`` for kernel and gradient."""
    payload = deep_water_regime(nu=50.0, depths=(20.0, 100.0, 800.0))
    for row in payload["rows"]:
        assert row["k0h"] > 700.0
        assert row["finite"], row
        assert row["relative_deviation"] < 1e-3, row
    deviations = [row["relative_deviation"] for row in payload["rows"]]
    assert deviations[-1] < deviations[1] < deviations[0]


def test_kernel_and_gradient_are_consistent_in_deep_water():
    """Analytic dG/dz must still match a finite difference at huge k0*h."""
    nu, depth = 50.0, 200.0
    target = np.array([1.0, 0.0, Z_MID])
    source = np.array([0.0, 0.0, Z_MID])
    analytic = complex(green_gradient_z_3d(target, source, nu, depth))
    step = 1e-6
    plus = complex(green_function_3d(target + [0.0, 0.0, step], source, nu, depth))
    minus = complex(green_function_3d(target - [0.0, 0.0, step], source, nu, depth))
    finite_difference = (plus - minus) / (2.0 * step)
    assert abs(analytic - finite_difference) <= 1e-6 * max(abs(analytic), 1.0)


def test_regime_verdict_is_all_pass():
    payload = run()
    assert all(payload["verdict"].values()), payload["verdict"]
    assert finite_depth_wavenumber(50.0, 800.0) * 800.0 > 7.0e3