"""Machine gate for the Michell unit-invariance / dispersion arbitration.

``docs/m3125/wigley_normalization_audit.md`` found that
``tensorbem.wigley.michell_wave_resistance`` does not return a force, and
concluded the repair would be a missing ``g**2 / C``. Two claims are tested
here so that the finding cannot rot back into prose:

* the *unit-exponent* gate, applied to all three shipped Michell paths (only
  the Wigley closed form was audited before; the general-hullform panel path
  in :mod:`tensorbem.michell` had never been measured);
* the *dispersion* gate: the steady free-surface condition forces the
  depth-decay wavenumber and the streamwise phase wavenumber apart
  (``k = alpha**2 / k0``), so the legacy single-wavenumber sampling carries a
  ``lambda - 1`` relative error that **no prefactor can remove** -- which is why
  ``g**2 / C`` is an incomplete diagnosis.

The two ``test_legacy_*`` cases assert the defect, not the absence of one: they
are ratchets. When a legacy path is repaired these tests fail on purpose, and
the repair must then delete them (they are the record that the defect existed).
The same convention is used by ``tests/test_wigley_normalization_audit.py``,
which pins the ``WITHHELD`` verdict rather than a value.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks import michell_unit_arbitration as arb

CASE = arb.BASE_CASE


def test_standard_form_path_is_dimensionally_a_force() -> None:
    """``thin_ship_michell`` must scale as a force under a change of units."""
    exps = arb.measure_force_exponents(arb.standard_form_resistance)
    assert np.allclose(exps, arb.FORCE_EXPONENTS, atol=1e-6, rtol=0.0), exps


@pytest.mark.parametrize(
    "name,path",
    [
        ("wigley_closed_form", arb.legacy_wigley_resistance),
        ("michell_panels", arb.legacy_panel_resistance),
    ],
)
def test_legacy_paths_are_not_forces(name: str, path) -> None:
    """RATCHET: both legacy paths return ``kg s^2 m^-1``, not a force.

    Measured exponents are ``(-1, +1, -2)`` against the force target
    ``(-1, -1, +2)``: the length and time exponents are wrong by ``+2`` and
    ``-4``.  The offset is a property of the implementation (prefactor
    ``4 rho/U^2`` times ``|H|^2 ~ m^4``), independent of grid, quadrature order
    and hull, so it is safe to pin exactly.  If this test ever passes, a legacy
    path was repaired -- delete it and re-pin the numbers it guarded.
    """
    exps = np.asarray(arb.measure_force_exponents(path), dtype=float)
    offset = exps - np.asarray(arb.FORCE_EXPONENTS)
    assert np.allclose(offset, (0.0, 2.0, -4.0), atol=1e-6, rtol=0.0), (name, exps)


def test_canonical_sampling_satisfies_the_free_surface_condition() -> None:
    """``(k, alpha) = (k0 lam^2, k0 lam)`` is an exact solution, to round-off."""
    k0 = CASE["g"] / (math.sqrt(CASE["g"] * CASE["L"]) * CASE["Fr"]) ** 2
    lam = np.array([1.0, 1.25, 2.0, 5.0, 40.0])
    residual = arb.dispersion_residual(k0 * lam**2, k0 * lam, k0)
    assert np.all(residual < 1e-14), residual


def test_legacy_sampling_violates_it_by_lambda_minus_one() -> None:
    """RATCHET: one wavenumber for both roles gives ``|dphi/dz - nu phi| ~ lam-1``."""
    k0 = CASE["g"] / (math.sqrt(CASE["g"] * CASE["L"]) * CASE["Fr"]) ** 2
    lam = np.array([1.0, 1.5, 2.0, 4.0, 12.0])
    residual = arb.dispersion_residual(k0 * lam, k0 * lam, k0)
    assert np.allclose(residual, lam - 1.0, atol=1e-12, rtol=0.0), residual


def test_prefactor_alone_cannot_repair_the_legacy_path() -> None:
    """After the only prefactor dimensional analysis admits, the residue drifts.

    ``g**2/pi`` is the unique factor that makes the legacy output a force, yet
    the residual against the standard form ranges from 0.34 to 0.80 over
    ``Fr = 0.20 ... 0.35``.  A constant ``C`` therefore cannot absorb it: the
    spectrum (wavenumber pairing) is the primary defect.
    """
    residues = [arb.residual_spectrum_ratio(fr)["after_prefactor"] for fr in (0.20, 0.25, 0.30, 0.35)]
    assert min(residues) > 0.0
    assert max(residues) / min(residues) > 1.5, residues


def test_repair_recipe_reproduces_the_independent_theta_form() -> None:
    """The Wigley closed form, corrected in place, matches the theta path.

    Different quadrature (``lam = 1 + s^2`` vs a graded theta rule) and
    different amplitude evaluations; agreement to ``< 1e-5`` means the closed
    form needs no change of theory, only the canonical pair and weight.
    """
    cross = arb.build_record()["theta_form_cross_check"]
    assert cross["relative_difference"] < 1e-5, cross


def test_committed_report_numbers_are_reproducible() -> None:
    """``docs/michell_unit_arbitration/numbers.json`` must match a fresh run."""
    path = Path("docs/michell_unit_arbitration/numbers.json")
    if not path.exists():
        pytest.skip("report artefact not present in this checkout")
    committed = json.loads(path.read_text())
    fresh = arb.build_record()
    assert committed["dimensionally_sound"] == fresh["dimensionally_sound"]
    assert np.allclose(committed["force_exponents"], fresh["force_exponents"])
    for axis in ("exponents",):
        for name in fresh[axis]:
            assert np.allclose(committed[axis][name], fresh[axis][name], atol=1e-9)
    assert np.allclose(
        committed["dispersion"]["legacy_residual"], fresh["dispersion"]["legacy_residual"], atol=1e-12
    )
    for old, new in zip(committed["spectrum"], fresh["spectrum"]):
        assert old["Fr"] == new["Fr"]
        assert math.isclose(old["standard"], new["standard"], rel_tol=1e-9)
        assert math.isclose(old["after_prefactor"], new["after_prefactor"], rel_tol=1e-9)
    assert math.isclose(
        committed["theta_form_cross_check"]["relative_difference"],
        fresh["theta_form_cross_check"]["relative_difference"],
        rel_tol=1e-6,
    )