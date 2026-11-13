"""m17547 — reference pinning + independent route for the rigid-lid hemisphere.

``CASE-BEM-RIGID-LID-HEMISPHERE`` was delivered by m17243.  These tests guard the
two verification items this session added:

* the ``ka -> 0`` added mass is pinned by an **exact-rational** Legendre series
  (closed-form coefficients, no Gauss quadrature of the kinked ``|mu|``) with a
  measured ``O(n**-4)`` tail removed by Richardson extrapolation;
* an **independent** axisymmetric MFS ring-source solver reproduces it, and the
  delivered case's verdict survives the (ppm-level) reference correction.

The invariants asserted here are the physics/arithmetic ones, not "the code ran":
the sphere check has a closed-form answer (0.5), the coefficient values are
exact rationals, the pin is compared against two arithmetic routes (rational and
float64), and the m17243 artifact is re-read rather than re-typed.
"""

from __future__ import annotations

import json
import math
import pathlib

import pytest

from tensorbem.benchmarks.rigid_lid_reference_pin import (
    REPO_VALUES,
    abs_mu_legendre_coefficient,
    legendre_at_zero,
    mfs_added_mass_dimless,
    pin_rigid_lid_reference,
    render_summary,
    rigid_lid_series,
    run_study,
    sphere_mfs_validation,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "benchmarks" / "verified" / "bem_rigid_lid_hemisphere_axisym"
ANCHOR = ROOT / "benchmarks" / "verified" / "bem_rigid_lid_hemisphere_ka0"


@pytest.fixture(scope="module")
def payload() -> dict:
    assert (ARTIFACTS / "results.json").exists(), "missing committed payload"
    return json.loads((ARTIFACTS / "results.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# exact algebra
# ---------------------------------------------------------------------------
def test_legendre_at_zero_matches_known_values() -> None:
    assert legendre_at_zero(0) == 1
    assert legendre_at_zero(1) == 0
    assert legendre_at_zero(2) == -0.5
    assert legendre_at_zero(4) == pytest.approx(0.375)
    assert legendre_at_zero(3) == 0


def test_abs_mu_coefficients_are_the_known_ones() -> None:
    """``|mu| = 1/2 + 5/8 P_2 - 3/16 P_4 + 13/128 P_6 - ...`` (exact rationals)."""
    a0 = abs_mu_legendre_coefficient(0)
    a2 = abs_mu_legendre_coefficient(2)
    a4 = abs_mu_legendre_coefficient(4)
    assert float(a0) == 0.5
    assert float(a2) == pytest.approx(0.625, rel=0, abs=1e-15)
    assert float(a4) == pytest.approx(-0.1875, rel=0, abs=1e-15)
    # odd degrees vanish identically (|mu| is even)
    assert abs_mu_legendre_coefficient(1) == 0
    assert abs_mu_legendre_coefficient(7) == 0


def test_series_matches_closed_form_leading_terms() -> None:
    """``A33/(rho*nabla) = 3 sum a_n**2/((n+1)(2n+1))`` reproduces its first terms."""
    series = rigid_lid_series(degree_max=4)
    # n = 0: 3*(1/2)**2/(1*1) = 3/4 ; n = 2: 3*(5/8)**2/(3*5) = 5/64
    assert series["terms"][0] == pytest.approx(0.75, abs=1e-15)
    assert series["terms"][1] == pytest.approx(5.0 / 64.0, abs=1e-15)
    # 3/4 + 5/64 + 3*(3/16)**2/(5*9) = 0.83046875 exactly
    assert series["truncated_value_exact"] == pytest.approx(0.83046875, abs=1e-12)


def test_float_and_rational_arithmetic_agree() -> None:
    series = rigid_lid_series(degree_max=120)
    assert series["float_vs_exact_roundoff"] < 1e-13


# ---------------------------------------------------------------------------
# pin
# ---------------------------------------------------------------------------
def test_pin_is_truncation_independent() -> None:
    pin = pin_rigid_lid_reference()
    assert pin["measured_tail_exponent"] == pytest.approx(4.0, abs=0.5)
    assert pin["truncation_tail_estimate"] < 1e-9
    # the pinned value is below the refuted 1.0 anchor and near the delivered 0.831
    assert 0.82 < pin["pinned_value"] < 0.84
    assert pin["pinned_value"] == pytest.approx(0.830951320, abs=1e-8)


def test_pin_is_not_the_gauss_quadrature_value() -> None:
    """The 3.4 ppm gap is quadrature bias, not round-off: it must be reproduced."""
    pin = pin_rigid_lid_reference()["pinned_value"]
    gauss = REPO_VALUES["gauss_quadrature_series_m16802_m17243"]["value"]
    rel = gauss / pin - 1.0
    assert 1e-7 < rel < 1e-5, rel
    assert rel > 0.0  # the kinked Gauss rule over-estimates


# ---------------------------------------------------------------------------
# independent route
# ---------------------------------------------------------------------------
def test_route_validates_against_the_exact_sphere_answer() -> None:
    """Translating sphere in infinite fluid: A33/(rho*4 pi a**3/3) = 0.5 exactly."""
    sphere = sphere_mfs_validation(24, 48, 0.5)
    assert sphere["added_mass_dimless"] == pytest.approx(0.5, abs=1e-5)
    assert sphere["neumann_residual"] < 1e-6


def test_independent_route_reproduces_the_pin() -> None:
    pin = pin_rigid_lid_reference()["pinned_value"]
    best = min(
        (mfs_added_mass_dimless(n, 2 * n, c) for n, c in ((32, 0.6), (64, 0.8), (128, 0.8))),
        key=lambda row: abs(row["added_mass_dimless"] / pin - 1.0),
    )
    rel = best["added_mass_dimless"] / pin - 1.0
    assert abs(rel) < 1e-3, rel
    # and it is nowhere near the refuted 1.0 anchor
    assert abs(best["added_mass_dimless"] - 1.0) > 0.15


# ---------------------------------------------------------------------------
# payload / delivered-case consequences
# ---------------------------------------------------------------------------
def test_payload_is_self_consistent(payload: dict) -> None:
    pin = payload["pin"]["value"]
    for row in payload["mfs_ladder"]:
        assert row["rel_error_vs_pin"] == pytest.approx(
            row["added_mass_dimless"] / pin - 1.0, rel=1e-12
        )
        assert row["condition_number"] > 0.0
    audit = {row["key"]: row for row in payload["reference_audit"]["rows"]}
    gauss = audit["gauss_quadrature_series_m16802_m17243"]
    assert gauss["deviation"] == pytest.approx(gauss["value"] - pin, abs=1e-15)


def test_all_criteria_pass_and_summary_renders(payload: dict) -> None:
    assert payload["verdict"] == "ALL PASS"
    for key, crit in payload["criteria"].items():
        assert crit["pass"], f"{key} failed: {crit}"
    text = render_summary(payload)
    assert f"{payload['pin']['value']:.12f}" in text
    assert (ARTIFACTS / "figure.png").stat().st_size > 20_000


def test_delivered_case_verdict_survives_the_pin(payload: dict) -> None:
    """m17243's 5 % gate must still pass against the corrected reference."""
    assert ANCHOR.exists()
    committed = json.loads((ANCHOR / "results.json").read_text(encoding="utf-8"))
    pin = payload["pin"]["value"]
    headline = next(row for row in committed["mesh_ladder"] if row["panels"] == 1024)
    rel = headline["added_mass_dimless"] / pin - 1.0
    assert abs(rel) < 0.05
    assert abs(rel - headline["rel_error_vs_ref"]) < 1e-4  # ppm-level shift only
    # every row of the recomputed table tracks the committed artifact
    rows = {row["panels"]: row for row in payload["anchor_rows"]}
    for row in committed["mesh_ladder"]:
        assert rows[row["panels"]]["added_mass_dimless"] == pytest.approx(
            row["added_mass_dimless"], rel=0, abs=1e-15
        )


def test_study_is_reproducible() -> None:
    fresh = run_study()
    assert fresh["verdict"] == "ALL PASS"
    committed = json.loads((ARTIFACTS / "results.json").read_text(encoding="utf-8"))
    assert fresh["pin"]["value"] == pytest.approx(committed["pin"]["value"], rel=0, abs=1e-12)
    for new, old in zip(fresh["mfs_ladder"], committed["mfs_ladder"]):
        assert new["added_mass_dimless"] == pytest.approx(old["added_mass_dimless"], rel=1e-12)
    assert math.isfinite(fresh["sphere_validation"]["added_mass_dimless"])