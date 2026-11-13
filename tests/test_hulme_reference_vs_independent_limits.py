"""Independent-software bound on the Hulme (1982) hemisphere reference.

Context (audit m17235 of case-pool mission m17181): the case
``CASE-BEM-HULME-HEMISPHERE`` grades TensorBEM's Kelvin BEM against
``tensorbem.analytic.hulme_hemisphere_radiation``.  That reference is a
*re-derived* multipole series -- the published table digits were not available
-- so the reference itself has to be held to an external standard.  These tests
freeze that standard: the third-party Rankine BEM code capytaine (optional
dependency, never imported by ``src/``) was mesh-laddered from 400 to 1936
faces, Richardson-extrapolated, and stored in
``tests/data/hulme_independent_limits.json`` with its provenance and caveat.

What is asserted here is a *band*, not a fitted value: the semi-analytical
reference must stay within 3 % of the independent limits, and it must be
truncation-converged.  If a future change to ``analytic.py`` breaks either
property, these tests fail -- which is exactly the failure mode the audit
found in the case's own 5 % gate (its worst cell passes by 4.76 % against the
self-derived series but is 6.56 % off the independent limit).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorbem.analytic import hulme_hemisphere_radiation

DATA = Path(__file__).with_name("data") / "hulme_independent_limits.json"
QUAD_ORDER = 512  # >= 2 * n_terms for every n_terms used below
BAND = 0.03


def _limits() -> dict:
    return json.loads(DATA.read_text())


@pytest.mark.parametrize("ka", [1.0, 2.0])
@pytest.mark.parametrize(
    "field",
    [
        "surge_added_mass_dimless",
        "surge_damping_dimless",
        "heave_added_mass_dimless",
        "heave_damping_dimless",
    ],
)
def test_reference_within_band_of_independent_limits(ka: float, field: str) -> None:
    """Analytic reference vs Richardson-extrapolated capytaine, band 3 %."""
    payload = _limits()
    key = f"ka_{ka:.1f}"
    independent = payload[key][field]
    assert independent > 0.0
    ref = hulme_hemisphere_radiation(ka=ka, n_terms=144, quad_order=QUAD_ORDER)
    got = getattr(ref, field)
    rel = (got - independent) / independent
    assert abs(rel) < BAND, (
        f"{field} @ ka={ka}: reference {got:.6f} vs independent limit "
        f"{independent:.6f} -> {100 * rel:+.2f}% exceeds the {100 * BAND:.0f}% band"
    )


@pytest.mark.parametrize("ka", [1.0, 2.0])
def test_reference_truncation_is_negligible(ka: float) -> None:
    """96 -> 144 term drift must sit far below the band it is used to judge."""
    coarse = hulme_hemisphere_radiation(ka=ka, n_terms=96, quad_order=QUAD_ORDER)
    fine = hulme_hemisphere_radiation(ka=ka, n_terms=144, quad_order=QUAD_ORDER)
    for field in (
        "surge_added_mass_dimless",
        "surge_damping_dimless",
        "heave_added_mass_dimless",
        "heave_damping_dimless",
    ):
        a, b = getattr(coarse, field), getattr(fine, field)
        assert abs(b - a) / abs(b) < 1.0e-3, f"{field} @ ka={ka}: 96->144 drift too large"


def test_independent_limits_payload_is_documented() -> None:
    """The frozen band is only meaningful with its provenance and caveat."""
    payload = _limits()
    provenance = payload["provenance"]
    for key in ("why", "independent_code", "limit_method", "caveat", "normalisation"):
        assert provenance.get(key), f"missing provenance field {key}"
    assert "capytaine" in provenance["independent_code"]
    assert set(payload) >= {"ka_1.0", "ka_2.0"}
    worst = payload["measured_reference_deviation"]["worst_relative"]
    assert 0.0 < worst < BAND, (
        "the recorded worst deviation must itself respect the band asserted by "
        "test_reference_within_band_of_independent_limits"
    )