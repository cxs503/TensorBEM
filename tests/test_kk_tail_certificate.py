"""The ``C3`` truncation budget is a conditional bound -- audit it, do not trust it.

Two claims are pinned here.

1. The closed-form tail integral that the budget is built from is *correct*:
   :func:`tensorbem.kk_tail_certificate.tail_principal_value_quadrature` is an
   independent composite Gauss-Legendre evaluation, and it agrees with the
   production series to round-off.  The existing KK-budget tests compare the
   production series with itself, so this is the first independent check of it.

2. The budget is an **upper** bound on the omitted integral only if the damping
   above ``s_max`` does not exceed its own power-law continuation.  A resonance
   sitting just above the band top makes the continuation a *lower* bound, and
   the modulus reports that as a measured ratio instead of hiding it: the
   allowance ``tolerance / budget_relative`` says how much tail error the
   certification can absorb, and the probe-measured omission says how much is
   actually there.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from tensorbem.causality import high_band_truncation_budget, power_tail_principal_value
from tensorbem.kk_tail_certificate import (
    certify_tail_budget,
    tail_omission,
    tail_principal_value_quadrature,
)

CASE_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "verified"
    / "casepool_bem_cummins_wigley"
)
DOF_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def _band(start=0.2, top=12.0, n=41):
    grid = np.logspace(math.log10(start), math.log10(top), n)
    grid[0], grid[-1] = start, top
    return grid


# --------------------------------------------------------------------------- #
# 1. independent oracle for the closed-form tail integral
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("power", [2.2, 2.5, 3.0, 4.0, 6.0, 9.0])
@pytest.mark.parametrize("omega", [0.3, 1.0, 4.0, 8.0, 11.5])
def test_quadrature_oracle_agrees_with_the_production_series(power: float, omega: float) -> None:
    series = power_tail_principal_value(omega, 3.0, power, 12.0)
    oracle = tail_principal_value_quadrature(omega, 3.0, power, 12.0)
    assert oracle == pytest.approx(series, rel=1e-12)


def test_oracle_refuses_arguments_the_series_also_refuses() -> None:
    with pytest.raises(ValueError):
        tail_principal_value_quadrature(12.0, 3.0, 4.0, 12.0)
    with pytest.raises(ValueError):
        tail_principal_value_quadrature(1.0, 3.0, 2.0, 12.0)


# --------------------------------------------------------------------------- #
# 2. the budget as a bound: exact on a power law, defeated by an unsolved peak
# --------------------------------------------------------------------------- #
def _power_law(start=12.0, top=20.0, n=11, power=4.0, coefficient=3.0):
    grid = np.logspace(math.log10(start), math.log10(top), n)
    grid[0], grid[-1] = start, top
    return grid, coefficient * grid ** (-power)


def test_budget_is_tight_on_a_power_law_that_the_probe_confirms() -> None:
    grid = _band()
    damping = 3.0 * grid**-4.0
    probe_omega, probe_damping = _power_law()
    certificate = certify_tail_budget(
        grid, damping, scale=3.0 * grid[0] ** -4.0,
        probe_omega=probe_omega, probe_damping=probe_damping,
    )
    assert certificate.exponent == pytest.approx(4.0, rel=1e-6)
    assert certificate.verdict == "certified"
    # The probe continues the *same* law, so the full omission (measured part
    # plus the probe's own continuation) reproduces the budget.  ``worst_ratio``
    # is deliberately the *lower* bound only (measured part), so it sits just
    # below one here.
    total = certificate.omission_total_relative / certificate.budget_relative
    finite = np.isfinite(total)
    assert np.allclose(total[finite], 1.0, rtol=1e-6)
    assert certificate.worst_ratio < 1.0


def _decaying_background(omega):
    """Local slope near -3.3 at the band top: admissible power-law continuation."""
    return 5.0 / (1.0 + omega**2) ** 1.65


def _resonance(omega, *, centre=17.0, width=2.5, height=8.0):
    """A second, narrower resonance the committed band never solved."""
    return height * (width * width) / ((omega * omega - centre * centre) ** 2 + width**4)


def test_budget_underestimates_when_a_resonance_sits_above_the_band_top() -> None:
    grid = _band()
    background = _decaying_background(grid)
    budget = high_band_truncation_budget(grid, background)
    assert budget.admissible and budget.exponent > 2.05

    # The probe sees what the committed band could not: a peak at 17 rad/s.
    probe_omega, _ = _power_law(start=12.0, top=200.0, n=120)
    probe_damping = _decaying_background(probe_omega) + _resonance(probe_omega)
    certificate = certify_tail_budget(
        grid, background, scale=float(background.max()),
        probe_omega=probe_omega, probe_damping=probe_damping,
    )
    assert certificate.verdict in {"certified_within_allowance", "falsified"}
    # Power-law continuation is a lower bound here, so the "conservative
    # estimate" wording would be wrong: the measured ratio is far above one.
    assert certificate.worst_ratio > 2.0


def test_the_allowance_says_how_much_tail_error_a_certification_absorbs() -> None:
    grid = _band()
    background = _decaying_background(grid)
    budget = high_band_truncation_budget(grid, background)
    certificate = certify_tail_budget(grid, background, scale=float(background.max()))
    # allowance * budget == tolerance, sample by sample
    finite = np.isfinite(certificate.budget_relative)
    assert np.allclose(
        certificate.allowance[finite] * certificate.budget_relative[finite],
        certificate.tolerance,
        rtol=1e-12,
    )


def test_no_certification_when_the_damping_is_not_in_algebraic_decay() -> None:
    grid = _band()
    rising = 0.5 + grid**2
    probe_omega, _ = _power_law()
    probe_damping = 0.5 + probe_omega**2
    certificate = certify_tail_budget(
        grid, rising, scale=float(rising.max()),
        probe_omega=probe_omega, probe_damping=probe_damping,
    )
    assert certificate.verdict == "uncertified"
    assert certificate.n_declared_decidable == 0
    assert certificate.min_allowance == 0.0
    assert "not entered algebraic decay" in certificate.reason


def test_the_measured_omission_is_a_lower_bound_and_grows_with_the_probe() -> None:
    grid = _band()
    damping = _decaying_background(grid)
    short_w, short_b = _power_law(start=12.0, top=20.0)
    long_w, long_b = _power_law(start=12.0, top=200.0, n=60)
    short_b, long_b = _decaying_background(short_w), _decaying_background(long_w)
    short = tail_omission(grid, damping, short_w, short_b)
    long = tail_omission(grid, damping, long_w, long_b)
    assert np.all(long.measured > short.measured)
    assert np.all(np.isfinite(long.total))


def test_probe_must_start_at_or_below_the_band_top() -> None:
    grid = _band()
    damping = _decaying_background(grid)
    probe_w, probe_b = _power_law(start=14.0, top=20.0)
    with pytest.raises(ValueError):
        tail_omission(grid, damping, probe_w, probe_b)


# --------------------------------------------------------------------------- #
# 3. the committed Cummins/Wigley evidence, pinned
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def wigley_certificates():
    sweep = CASE_DIR / "sweep_medium.npz"
    probe = CASE_DIR / "sweep_extended_probe_medium.npz"
    if not (sweep.is_file() and probe.is_file()):
        pytest.skip("committed Cummins/Wigley artifacts are not in this checkout")
    band = np.load(sweep, allow_pickle=True)
    wide = np.load(probe, allow_pickle=True)
    out = {}
    for index, name in enumerate(DOF_NAMES):
        out[name] = certify_tail_budget(
            band["omega"],
            band["damping"][:, index, index],
            float(np.abs(band["added_mass"][:, index, index]).max()),
            probe_omega=wide["omega"],
            probe_damping=wide["damping"][:, index, index],
            panel_scale=float(wide["max_panel_scale"]),
        )
    return out, json.loads((CASE_DIR / "case_result.json").read_text())


def test_the_certified_dof_survive_the_measured_tail_model_error(wigley_certificates) -> None:
    certificates, _ = wigley_certificates
    for name in ("heave", "roll", "pitch"):
        certificate = certificates[name]
        assert certificate.n_declared_decidable == 12
        assert certificate.verdict == "certified"
        assert certificate.worst_ratio < 1.0
        # the certification can absorb at least a factor 2 of tail-model error
        assert certificate.min_allowance > 2.0


def test_the_uncertified_dof_have_a_tail_as_large_as_their_residual(wigley_certificates) -> None:
    certificates, result = wigley_certificates
    residual = {
        row["dof_name"]: row["max_relative_residual"]
        for row in result["gates"]["C3a"]["values"]["per_dof"]
    }
    for name in ("surge", "sway", "yaw"):
        certificate = certificates[name]
        assert certificate.verdict == "uncertified"
        assert certificate.n_declared_decidable == 0
        # the resolved tail alone carries a comparable fraction of the residual,
        # so that residual is not evidence of causality
        omission = float(np.nanmax(certificate.omission_lower_relative))
        assert 0.4 < omission / residual[name] < 1.5