"""Verification of the Kramers-Kronig (causality) transform.

The point of these tests is that the tool must be shown to be *correct before
it is used to judge a solver*: the transform is applied to coefficient pairs
whose causality is exact, and only then (a small case) to a tabulated
reference. A transform that silently returns a plausible number for a wrong
kernel would make every downstream residual meaningless, which is exactly the
failure mode this file locks down.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.analytic import ursell_cylinder_added_mass
from tensorbem.causality import (
    fit_added_mass_asymptote,
    kkr_added_mass,
    kkr_check,
    kkr_damping,
    principal_value_integral,
)


def _exponential_pair(
    omega: np.ndarray, coefficients: tuple[tuple[float, float], ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Exact ``(A - A_inf, B)`` of the memory kernel ``sum c_j exp(-a_j s)``.

    ``Khat = sum_j c_j/(a_j + i*omega)``, so ``B = Re Khat`` and
    ``A - A_inf = Im Khat / omega`` with ``A_inf = 0``. Both relations (1) and
    (2) of :mod:`tensorbem.causality` hold identically for this family, which
    is what makes it a valid calibration of the transform.
    """
    transform = np.zeros_like(omega, dtype=np.complex128)
    for amplitude, rate in coefficients:
        transform += amplitude / (rate + 1j * omega)
    return np.imag(transform) / omega, np.real(transform)


_EXACT_PAIRS = (
    ((1.0, 0.7),),
    ((1.0, 0.7), (0.4, 2.5)),
    ((2.0, 0.05), (0.3, 1.2), (-0.5, 4.0)),
)


@pytest.mark.parametrize("coefficients", _EXACT_PAIRS)
def test_exact_causal_pair_has_zero_residual(coefficients) -> None:
    """A pair that is causal by construction must come back as causal."""
    omega = np.geomspace(1.0e-4, 1.0e4, 4001)
    added_mass, damping = _exponential_pair(omega, coefficients)
    report = kkr_check(omega, added_mass, damping)
    assert report.asymptote_relative_residual < 1.0e-3
    assert abs(report.added_mass_inf) < 1.0e-5 * np.max(np.abs(damping))
    # Limited by the piecewise-linear sweep and by the two truncated bands, not
    # by the transform: 1e-3 is a factor 20 above the measured residual.
    assert report.max_damping_relative_error < 1.0e-3
    assert report.max_added_mass_relative_error < 5.0e-3


@pytest.mark.parametrize("rate", (0.25, 1.0, 4.0))
@pytest.mark.parametrize("omega", (0.3, 1.0, 3.0))
def test_principal_value_matches_closed_form(rate: float, omega: float) -> None:
    """``PV int_0^inf h_a(s)/(s^2-w^2) ds = -(pi/(2a)) h_a(w)``, ``h_a=1/(a^2+s^2)``.

    This closed form is independent of the interval algebra in the module, so
    it pins the pole handling down on its own.
    """
    grid = np.geomspace(1.0e-5, 1.0e4, 4001)
    values = 1.0 / (rate**2 + grid**2)
    reference = -(math.pi / (2.0 * rate)) / (rate**2 + omega**2)
    computed = principal_value_integral(grid, values, omega)
    assert abs(computed - reference) / abs(reference) < 2.0e-3


def test_gap_coefficient_recovers_plateau() -> None:
    """``A(s) = A_inf + c/s^2``: the fit must return the plateau and the slope."""
    grid = np.geomspace(1.0e-2, 1.0e3, 2001)
    plateau, coefficient = 12.5, -3.0
    added_mass = plateau + coefficient / grid**2
    fit = fit_added_mass_asymptote(grid, added_mass)
    assert fit.added_mass_inf == pytest.approx(plateau, rel=1.0e-9)
    assert fit.tail_coefficient == pytest.approx(coefficient, rel=1.0e-6)
    assert fit.relative_residual < 1.0e-9


def test_inconsistent_pair_is_detected() -> None:
    """Negative control: a 5 % corruption of the damping must show up."""
    omega = np.geomspace(1.0e-2, 1.0e2, 2001)
    added_mass, damping = _exponential_pair(omega, ((1.0, 0.7), (0.4, 2.5)))
    clean = kkr_check(omega, added_mass, damping)
    corrupted = kkr_check(omega, added_mass, damping * 1.05)
    assert clean.max_damping_relative_error < 1.0e-2
    assert corrupted.max_damping_relative_error > 5.0 * clean.max_damping_relative_error
    assert corrupted.max_damping_relative_error > 5.0e-2


def test_relations_are_individually_reproducible() -> None:
    """Both eq. (1) and eq. (2) reproduce a single sample frequency."""
    omega = np.geomspace(1.0e-3, 1.0e3, 3001)
    added_mass, damping = _exponential_pair(omega, ((1.0, 0.7), (0.4, 2.5)))
    sample = float(np.sqrt(omega[1500] * omega[1501]))
    predicted_damping = kkr_damping(omega, added_mass, sample, added_mass_inf=0.0)
    predicted_mass = kkr_added_mass(omega, damping, sample, added_mass_inf=0.0)
    assert predicted_damping == pytest.approx(
        float(np.interp(sample, omega, damping)), rel=2.0e-3
    )
    assert predicted_mass == pytest.approx(
        float(np.interp(sample, omega, added_mass)), rel=2.0e-3
    )


def test_sample_frequency_on_a_node_is_rejected() -> None:
    """A kinked interpolant cannot be integrated through the pole; say so."""
    omega = np.geomspace(1.0, 10.0, 21)
    added_mass, damping = _exponential_pair(omega, ((1.0, 0.7),))
    with pytest.raises(ValueError, match="coincides with a tabulated node"):
        kkr_check(omega, added_mass, damping, sample_omegas=[omega[5]])


def test_sweep_validation() -> None:
    """Malformed sweeps are rejected rather than silently extrapolated."""
    good = np.geomspace(1.0, 10.0, 21)
    with pytest.raises(ValueError, match="strictly increasing"):
        kkr_damping(good[::-1], np.ones(21), 3.0, added_mass_inf=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        kkr_damping(good - 2.0, np.ones(21), 3.0, added_mass_inf=0.0)
    with pytest.raises(ValueError, match="at least"):
        kkr_damping(good[:4], np.ones(4), 3.0, added_mass_inf=0.0)
    with pytest.raises(ValueError, match="same length"):
        kkr_damping(good, np.ones(20), 3.0, added_mass_inf=0.0)
    with pytest.raises(ValueError, match="outside the tabulated sweep"):
        kkr_damping(good, np.ones(21), 0.5, added_mass_inf=0.0)


def test_robust_tail_exponent_clean_decay():
    """A clean power-law tail should give consensus across all windows."""
    from tensorbem.causality import robust_tail_exponent
    import numpy as np

    omega = np.linspace(1.0, 20.0, 55)
    damping = 100.0 * omega ** (-3.5)  # exact p=3.5 decay
    r = robust_tail_exponent(omega, damping)
    assert r.consensus is True
    assert abs(r.exponent - 3.5) < 0.01
    assert r.spread < 0.01
    assert len(r.per_window) == 3  # windows 3, 4, 5 all succeed


def test_robust_tail_exponent_noisy_no_consensus():
    """Noisy tail data should fail consensus."""
    from tensorbem.causality import robust_tail_exponent
    import numpy as np

    omega = np.linspace(10.0, 20.0, 10)
    # Oscillating damping: not a clean power law
    damping = np.array([10, 8, 12, 5, 15, 3, 20, 2, 25, 1], dtype=float)
    r = robust_tail_exponent(omega, damping)
    assert r.consensus is False
    assert r.spread > 1.0  # large spread


def test_robust_tail_exponent_below_bar_consensus():
    """All windows below the bar should still give consensus (all agree = below)."""
    from tensorbem.causality import robust_tail_exponent
    import numpy as np

    omega = np.linspace(0.2, 12.0, 41)
    # Shallow decay: p ~ 1.0, well below 2.05 bar
    damping = 50.0 * omega ** (-1.0)
    r = robust_tail_exponent(omega, damping)
    assert r.consensus is True  # all windows agree: below bar
    assert r.exponent < 2.05


def test_robust_tail_exponent_custom_windows():
    """Custom window set should be respected."""
    from tensorbem.causality import robust_tail_exponent
    import numpy as np

    omega = np.linspace(1.0, 20.0, 55)
    damping = 100.0 * omega ** (-3.0)
    r = robust_tail_exponent(omega, damping, windows=(3, 5))
    assert len(r.per_window) == 2
    assert 3 in r.per_window
    assert 5 in r.per_window
    assert 4 not in r.per_window


def test_robust_tail_exponent_non_positive_damping():
    """Windows with non-positive damping should be excluded."""
    from tensorbem.causality import robust_tail_exponent
    import numpy as np

    omega = np.linspace(10.0, 20.0, 10)
    # Last 3 samples all non-positive -> window=3 fails
    # Last 4 samples include a negative -> window=4 fails
    # Last 5 samples include a negative -> window=5 fails
    damping = np.array([10, 8, 6, 4, 2, 0.5, -0.1, -0.3, -0.2, -0.1], dtype=float)
    r = robust_tail_exponent(omega, damping)
    assert len(r.per_window) == 0
    assert r.consensus is False
    assert r.exponent == -float('inf')
