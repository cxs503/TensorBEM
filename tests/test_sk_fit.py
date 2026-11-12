"""Gates for :mod:`tensorbem.sk_fit` -- the SK iteration must iterate, must say so,
and must beat the pre-fix fixed-pole solve on a real damping curve.

Numbers pinned here come from
``python -m tensorbem.benchmarks.sk_fit_verification`` (see
``docs/sk_verification/results.json``); the gates are set with ~2-3x slack over
the measured values so they catch regressions without being brittle.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorbem.sk_fit import fixed_pole_fit_damping, sk_fit_damping
from tensorbem.benchmarks.sk_fit_verification import (
    closed_form_kernel,
    relaxation_model,
    tau_grid,
)
from tensorbem.time_domain import radiation_memory_kernel

DATA = Path(__file__).resolve().parent / "data" / "time_domain_hemisphere_sweep.json"


@pytest.fixture(scope="module")
def hemisphere():
    record = json.loads(DATA.read_text())
    return np.asarray(record["omega"], float), np.asarray(record["damping"], float)


@pytest.fixture(scope="module")
def synthetic():
    alpha = np.array([0.4, 1.1, 3.0])
    residues = np.array([2.0, 1.2, 0.6])
    omega = np.geomspace(0.02, 60.0, 600)
    return omega, relaxation_model(omega, alpha, residues, 0.05), alpha, residues


def _rel_error(fit, damping):
    return float(np.max(np.abs(fit.B_reconstructed - damping)) / np.max(np.abs(damping)))


# ---------------------------------------------------------------------------
# the defect this module was fixed for
# ---------------------------------------------------------------------------
def test_sk_iterates_and_reports_it(hemisphere):
    omega, damping = hemisphere
    fit = sk_fit_damping(omega, damping, n_poles=12)
    # The pre-fix module hard-coded n_iter=1 / converged=True while claiming
    # "SK iteration".  The iteration must actually run.
    assert fit.n_iter >= 2
    assert isinstance(fit.converged, bool)
    assert fit.poles.size == 12
    assert np.all(np.diff(fit.poles) > 0.0)  # sorted, positive decay rates
    assert fit.constant >= 0.0               # passivity: B(inf) >= 0


def test_fixed_pole_baseline_is_the_old_single_solve(hemisphere):
    omega, damping = hemisphere
    fit = fixed_pole_fit_damping(omega, damping, n_poles=12)
    assert fit.n_iter == 1
    assert fit.converged is True
    grid = np.logspace(np.log10(0.3 * omega[0]), np.log10(3.0 * omega[-1]), 12)
    assert np.allclose(fit.poles, grid)  # poles frozen on the log grid


@pytest.mark.parametrize("kwargs,expected_iter", [({"max_iter": 1}, 1), ({"max_iter": 3}, None)])
def test_max_iter_is_honoured(hemisphere, kwargs, expected_iter):
    omega, damping = hemisphere
    fit = sk_fit_damping(omega, damping, n_poles=12, **kwargs)
    if expected_iter is not None:
        assert fit.n_iter == expected_iter
    else:
        assert 1 <= fit.n_iter <= 3


def test_tol_is_honoured(hemisphere):
    omega, damping = hemisphere
    loose = sk_fit_damping(omega, damping, n_poles=12, tol=1e9)
    assert loose.converged is True
    assert loose.n_iter == 1
    tight = sk_fit_damping(omega, damping, n_poles=12, tol=0.0, max_iter=8)
    assert tight.n_iter >= loose.n_iter


def test_deterministic(hemisphere):
    omega, damping = hemisphere
    a = sk_fit_damping(omega, damping, n_poles=8)
    b = sk_fit_damping(omega, damping, n_poles=8)
    assert np.array_equal(a.poles, b.poles)
    assert np.array_equal(a.residues, b.residues)


# ---------------------------------------------------------------------------
# accuracy: SK must beat the fixed-pole solve that it replaces
# ---------------------------------------------------------------------------
def test_sk_beats_fixed_pole_on_hemisphere(hemisphere):
    omega, damping = hemisphere
    fixed = fixed_pole_fit_damping(omega, damping, n_poles=12)
    sk = sk_fit_damping(omega, damping, n_poles=12)
    # measured: 1.6836e-01 -> 1.5471e-02 (about 11x)
    assert _rel_error(sk, damping) < 1.0 / 3.0 * _rel_error(fixed, damping)
    assert _rel_error(sk, damping) < 3.0e-2


def test_sk_fits_the_exact_model_class(synthetic):
    omega, damping, alpha, residues = synthetic
    fit = sk_fit_damping(omega, damping, n_poles=3)
    # measured: 3.86e-02 (fixed-pole baseline: 3.87e-01)
    assert _rel_error(fit, damping) < 6.0e-2
    assert _rel_error(fit, damping) < 1.0 / 3.0 * _rel_error(
        fixed_pole_fit_damping(omega, damping, n_poles=3), damping
    )


def test_reconstruction_consistent_with_coefficients(synthetic):
    omega, damping, _, _ = synthetic
    fit = sk_fit_damping(omega, damping, n_poles=4)
    direct = relaxation_model(omega, fit.poles, fit.residues, fit.constant)
    assert np.allclose(direct, fit.B_reconstructed, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# time-domain consequence: the Cummins memory kernel of the fitted model
# ---------------------------------------------------------------------------
def _kernel_error(omega, damping, fit, tau):
    table = radiation_memory_kernel(omega, damping, tau, low_frequency_model="cubic").values
    model = closed_form_kernel(tau, fit.poles, fit.residues)
    err = np.abs(model - table)[1:]  # tau = 0 carries the 2*d*delta(t) impulse
    return float(np.max(err) / np.max(np.abs(table[1:])))


def test_kernel_error_drops_with_sk(hemisphere):
    omega, damping = hemisphere
    tau = tau_grid(omega[-1], span=20.0)
    fixed = fixed_pole_fit_damping(omega, damping, n_poles=12)
    sk = sk_fit_damping(omega, damping, n_poles=12)
    err_fixed = _kernel_error(omega, damping, fixed, tau)
    err_sk = _kernel_error(omega, damping, sk, tau)
    # measured: 1.19e+02 (fixed poles) -> 5.44e-03 (SK); the fixed-pole spectrum
    # puts poles down at 0.3*omega_min and leaves an exponential tail in K(t).
    assert err_sk < 1.0e-2
    assert err_sk < err_fixed


# ---------------------------------------------------------------------------
# input contract (unchanged from the pre-fix module)
# ---------------------------------------------------------------------------
def test_input_validation():
    omega = np.linspace(0.1, 10.0, 40)
    damping = np.exp(-omega)
    with pytest.raises(ValueError):
        sk_fit_damping(omega[:5], damping[:5], n_poles=4)
    with pytest.raises(ValueError):
        sk_fit_damping(np.concatenate([[-1.0], omega]), np.concatenate([[0.0], damping]))
    with pytest.raises(ValueError):
        sk_fit_damping(omega, -damping, n_poles=4)
    with pytest.raises(ValueError):
        sk_fit_damping(omega, damping, n_poles=0)
    with pytest.raises(ValueError):
        sk_fit_damping(omega, damping, n_poles=4, max_iter=0)