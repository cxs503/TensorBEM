"""Regression pins for the m14767 state-space fit / realization fixes.

Each test here fails on the pre-m14767 code with the number quoted in the
docstring, so the fix is pinned rather than assumed:

* the unweighted nonlinear fit put five poles on the ``alpha = 1e-6`` bound with
  residues ~7.5e3 and returned 61.5 % peak error on the Hulme hemisphere curve
  (``B(omega_min)`` off by a factor 763);
* ``state_space_from_fit`` dropped the fitted constant ``d``, so the realized
  model radiated less than the fit it claimed to represent (2.8-4.6 % of the
  harmonic response on a curve with ``d = 50``);
* ``cummins_state_space`` used a bare explicit RK4 step and overflowed to NaN
  when the pole set was faster than ``2.78 / dt``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.state_space import (
    _evaluate_rational,
    _harmonic_memory_response,
    cummins_state_space,
    radiation_response,
    state_space_from_fit,
    validate_state_space,
    vector_fit_damping,
)

_DATA = Path(__file__).parent / "data" / "time_domain_hemisphere_sweep.json"


@pytest.fixture(scope="module")
def hemisphere():
    record = json.loads(_DATA.read_text())
    return (
        np.asarray(record["omega"], dtype=float),
        np.asarray(record["damping"], dtype=float),
        np.asarray(record["added_mass"], dtype=float),
        record["body"],
    )


def _synthetic(omega, constant, coefficients):
    B = np.full_like(omega, float(constant))
    for c, a in coefficients:
        B = B + c * a / (a * a + omega**2)
    return B


def test_exact_poles_recovered():
    """Three known poles on a four-decade band must come back to 1e-6."""
    coefficients = ((1000.0, 0.5), (500.0, 1.5), (200.0, 3.0))
    omega = np.logspace(-1.0, 1.0, 160)
    B = _synthetic(omega, 0.0, coefficients)
    fit = vector_fit_damping(omega, B, n_poles=3)
    assert fit.relative_error_max < 1e-6
    assert fit.relative_error_max_pointwise < 1e-6
    assert np.allclose(np.sort(fit.poles), [0.5, 1.5, 3.0], rtol=1e-3)
    assert np.allclose(np.sort(fit.residues), [200.0, 500.0, 1000.0], rtol=1e-3)


def test_wide_dynamic_range_ends_are_fitted(hemisphere):
    """The 4-decade Hulme curve: both ends, not just the peak.

    Pre-m14767: peak error 0.615 (gate 0.15) and ``B(omega_min)`` = 1075 against
    a measured 1.41 (factor 763) because unweighted absolute residuals ignore
    everything below the peak.
    """
    omega, damping, _, _ = hemisphere
    fit = vector_fit_damping(omega, damping, n_poles=10)
    assert fit.relative_error_max < 0.01
    assert fit.relative_error_max_pointwise < 0.05
    # low-frequency end (the failure that mattered physically)
    assert fit.B_reconstructed[0] == pytest.approx(damping[0], rel=0.02)
    # high-frequency tail: |B - d| stays a small fraction of the peak
    assert abs(fit.B_reconstructed[-1] - damping[-1]) < 0.05 * damping.max()
    # and no pole is parked on the search bound
    assert np.min(fit.poles) > 10.0 * 1e-3 * omega[0]


def test_positive_residues_cannot_represent_the_curve(hemisphere):
    """Negative residues are required -- documented, not assumed.

    With ``c_k >= 0`` enforced the best fit of the Hulme curve stalls near 100 %
    peak error at every order tried, so the sign constraint in the old
    ``bounds`` list (``c_k >= 0``) was itself part of the defect.
    """
    omega, damping, _, _ = hemisphere
    fit = vector_fit_damping(omega, damping, n_poles=10)
    assert np.any(fit.residues < 0.0)
    # the realised response nevertheless stays a non-negative damping in band
    band = np.linspace(omega[0], omega[-1], 400)
    B_band = _evaluate_rational(band, fit.poles, fit.residues, fit.constant)
    assert np.all(B_band >= 0.0)


def test_feedthrough_preserved():
    """``d`` must survive into the realization: K(tau) = sum c_k e^-a_k tau + d delta(tau)."""
    omega = np.linspace(0.1, 6.0, 160)
    B = _synthetic(omega, 50.0, ((900.0, 0.6), (450.0, 1.8), (220.0, 3.4)))
    fit = vector_fit_damping(omega, B, n_poles=3)
    assert fit.constant == pytest.approx(50.0, rel=1e-3)
    ss = state_space_from_fit(fit)
    assert ss.D_ss == pytest.approx(fit.constant)

    om = 2.0
    analytic = radiation_response(ss, om)
    assert analytic.real == pytest.approx(
        float(_evaluate_rational(np.array([om]), fit.poles, fit.residues, fit.constant)[0]),
        rel=1e-12,
    )
    # time-domain integration of the auxiliary ODEs reproduces the analytic response
    assert abs(_harmonic_memory_response(ss, om)) == pytest.approx(abs(analytic), rel=5e-3)

    # negative control: drop the feedthrough and the same check must fail
    ss.D_ss = 0.0
    without = abs(_harmonic_memory_response(ss, om))
    assert abs(without - abs(analytic)) / abs(analytic) > 0.02


def test_explicit_integration_guard():
    """A fast pole with a coarse ``dt`` must substep instead of overflowing."""
    omega = np.linspace(0.05, 300.0, 400)
    B = _synthetic(omega, 0.0, ((4.0e4, 250.0), (1.0e4, 40.0)))
    fit = vector_fit_damping(omega, B, n_poles=2)
    ss = state_space_from_fit(fit)
    assert np.max(ss.poles) > 100.0
    result = cummins_state_space(
        ss, mass=1000.0, added_mass_inf=500.0, restoring=1.0e4,
        initial_displacement=0.05, dt=0.05, duration=20.0,
    )
    assert result.substeps > 1
    assert np.all(np.isfinite(result.displacement))
    assert np.max(np.abs(result.displacement)) <= 0.05 + 1e-9


def test_validation_c4_c5_are_computed_not_asserted(hemisphere):
    """C4/C5 must come out of a computation.

    Before m14767 ``validate_state_space`` set ``c4_*_ss = <same value as fd>``,
    ``c4_rao_peak_*_rel_error = 0.0``, ``c5_K_negative_time_max = 0.0`` and
    ``c4_pass = c5_pass = True`` unconditionally; nothing exercised them.
    """
    omega, damping, added, body = hemisphere
    fit = vector_fit_damping(omega, damping, n_poles=10)
    ss = state_space_from_fit(fit)
    report = validate_state_space(
        omega, added, damping, body["added_mass_inf"], body["mass"], body["restoring"], fit, ss
    )
    assert math.isfinite(report.c4_memory_amp_rel_error_max)
    assert report.c4_memory_amp_rel_error_max < 0.01
    assert math.isfinite(report.c4_memory_phase_error_max)
    assert math.isfinite(report.c5_kk_residual)
    assert report.c5_min_pole > 0.0
    assert report.c2_all_stable
    # the computed criteria, and the aggregate flag, agree with each other
    assert report.all_pass == (
        report.c1_pass and report.c2_pass and report.c3_pass
        and report.c4_pass and report.c5_pass and report.c6_pass
    )