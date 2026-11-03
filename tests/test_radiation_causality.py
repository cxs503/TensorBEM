"""Tests for the Ogilvie-Tuck (Kramers-Kronig) causality transform.

The transform is a *verification tool*, so it is tested the way the package
tests other reference generators: first against an exact causal pair where the
answer is known, then against a deliberately broken input that must be
rejected.  The Ursell data set is scored but not asserted to close -- the study
records an honest FAIL and the test only pins that verdict, so a future fix
cannot silently change the conclusion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.benchmarks.radiation_causality import (
    SYNTHETIC_A_INF,
    added_mass_from_damping,
    damping_from_added_mass,
    fit_added_mass_infinite,
    synthetic_causal_pair,
    synthetic_validation,
    ursell_spectrum,
    ursell_study,
)


@pytest.fixture(scope="module")
def synthetic() -> dict:
    return synthetic_validation()


def test_synthetic_pair_satisfies_its_own_definitions():
    """``B`` and ``A`` are the cosine/sine transforms of ``exp(-alpha t)``."""
    alpha = 2.0
    w = 0.7
    added_mass, damping = synthetic_causal_pair(np.array([w]))
    assert damping[0] == pytest.approx(alpha / (alpha ** 2 + w ** 2))
    assert added_mass[0] - SYNTHETIC_A_INF == pytest.approx(-1.0 / (alpha ** 2 + w ** 2))


def test_added_mass_recovered_from_damping(synthetic):
    """The ``A <- B`` relation reproduces an exact pair to the quadrature floor."""
    assert synthetic["added_mass_max_relative_error"] < 5.0e-3
    assert synthetic["added_mass_median_relative_error"] < 2.0e-3


def test_added_mass_infinite_fitted_to_exact_value(synthetic):
    """The only free parameter of the relation is recovered from the data."""
    assert synthetic["added_mass_infinite_relative_error"] < 5.0e-3


def test_damping_recovered_from_added_mass(synthetic):
    """The inverse relation is looser (it differentiates the data) but closes."""
    assert synthetic["damping_max_relative_error"] < 0.06


def test_negative_control_is_rejected(synthetic):
    """Rolling the damping table by one index must destroy the relation."""
    assert synthetic["negative_control_passes"] is True
    assert synthetic["negative_control_max_relative_error"] > 1.0


def test_target_must_lie_inside_the_window():
    nodes = np.geomspace(0.1, 10.0, 21)
    damping = np.linspace(1.0, 0.1, 21)
    with pytest.raises(ValueError):
        added_mass_from_damping(nodes, damping, 10.0, 2.0)
    with pytest.raises(ValueError):
        damping_from_added_mass(nodes, damping, 0.05, 2.0)


def test_mismatched_shapes_are_rejected():
    nodes = np.geomspace(0.1, 10.0, 21)
    with pytest.raises(ValueError):
        added_mass_from_damping(nodes, np.ones(5), 1.0, 2.0)


def test_ursell_spectrum_matches_the_documented_benchmark_value():
    """Sanity anchor: ``ka = 0.6`` reproduces the README pair."""
    added_mass, damping = ursell_spectrum(np.array([0.6]))
    assert added_mass[0] == pytest.approx(2545.1672, rel=1e-6)
    assert damping[0] == pytest.approx(2718.1540, rel=1e-6)


def test_ursell_high_frequency_added_mass_creeps_towards_its_limit():
    """The documented cause of the honest FAIL: ``A(ka)`` has not converged.

    If this ever fails because ``A`` really does flatten out, the causality
    study should be re-run and the verdict revised -- not the tolerance.
    """
    added_mass, _ = ursell_spectrum(np.array([1.0, 10.0, 40.0]))
    assert added_mass[2] > added_mass[1] > 0.0
    # still more than 5 % away from the unbounded-fluid value rho*pi*a^2
    assert abs(added_mass[2] - 3220.13) / 3220.13 > 0.05


def test_ursell_study_records_the_honest_fail():
    """The study runs, produces finite numbers, and states its verdict."""
    result = ursell_study(lo=0.05, hi=10.0, intervals=60)
    assert math.isfinite(result["fitted_added_mass_infinite"])
    assert result["added_mass_median_relative_error"] > 0.01
    assert result["verdict"].startswith("TRANSFORM-DOES-NOT-CLOSE")