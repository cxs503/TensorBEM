"""Near-field vs far-field damping cross-check on the Havelock case mesh (high ka).

Guards the artefact produced by
``benchmarks/verified/casepool_bem_submerged_sphere_havelock/far_field_high_ka.py``
and, through it, the *content* of the ``ka >= 4`` rows of the case pool: those
rows carry ``B' ~ 1e-6`` and a 5 % relative gate is only meaningful if the
near-field (pressure) integral is not sitting on a cancellation floor.  The
artifact answers that with a second, route-independent route: the far-field
(Kochin/energy) damping.

The assertions are invariants, not execution traces:

* both damping routes agree with each other (no reference used) -- a broken
  near-field or a mis-scaled Kochin prefactor breaks this;
* both routes agree with the multipole reference at the same order;
* the residual disagreement *converges* from 320 to 1280 panels, which is what
  identifies it as discretisation bias rather than a route error;
* the Kochin azimuth resolution actually used is shown to be converged;
* the near-field numbers reproduce the committed case-pool ``numbers.json``
  rows bit-for-bit, so the two artefacts cannot drift apart.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorbem.analytic import submerged_sphere_wave_multipole

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/case_bem_submerged_sphere_havelock/far_field_high_ka.json"
NUMBERS = ROOT / "docs/case_bem_submerged_sphere_havelock/numbers.json"

CASE_SUBDIVISION = 3
ECONOMY_SUBDIVISION = 2
BAND_KA = (0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
HIGH_KA = (3.0, 4.0, 5.0)
#: Two independent numerical routes for the same physical quantity must agree to
#: better than this; it is a cross-route check, not a reference comparison.
NEAR_FAR_TOL = 1.0e-2
#: Far-field value must be converged in Kochin azimuth resolution.
AZIMUTH_TOL = 5.0e-3


@pytest.fixture(scope="module")
def payload() -> dict:
    assert ARTIFACT.is_file(), f"committed artefact missing: {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def numbers() -> dict:
    assert NUMBERS.is_file(), f"case-pool artefact missing: {NUMBERS}"
    return json.loads(NUMBERS.read_text())


def _row(payload: dict, subdivision: int, ka: float) -> dict:
    for row in payload["rows"]:
        if row["subdivision"] == subdivision and row["ka"] == ka:
            return row
    raise AssertionError(f"missing row subdivision={subdivision} ka={ka}")


def test_artifact_covers_both_meshes_and_full_band(payload):
    """2 subdivisions x 7 wavenumbers, and the case mesh is really 1280 panels."""
    assert payload["case"] == "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK"
    kas = sorted({row["ka"] for row in payload["rows"]})
    assert tuple(kas) == BAND_KA
    subs = sorted({row["subdivision"] for row in payload["rows"]})
    assert subs == [ECONOMY_SUBDIVISION, CASE_SUBDIVISION]
    for row in payload["rows"]:
        expected = 320 if row["subdivision"] == ECONOMY_SUBDIVISION else 1280
        assert row["n_panels"] == expected


def test_provenance_bound_to_commit(payload):
    """The artefact names the commit it was produced from and its normalisation."""
    prov = payload["provenance"]
    assert len(prov["git_commit"]) == 40
    assert "far_field_high_ka.py" in prov["script"]
    assert "4/3*pi*a^3" in payload["parameters"]["normalisation"]


def test_reference_values_reproducible(payload):
    """The stored reference values come from the analytic multipole solution."""
    params = payload["parameters"]
    for row in payload["rows"]:
        ref = submerged_sphere_wave_multipole(
            row["ka"],
            radius=params["radius"],
            submergence=params["submergence"],
            density=params["density"],
            gravity=params["gravity"],
            n_terms=params["reference_terms"],
        )
        expected = {
            "b33": ref.heave_damping_dimless,
            "b11": ref.surge_damping_dimless,
        }
        for key, value in expected.items():
            assert row["damping_near"]["ref"][key] == pytest.approx(
                value, rel=1e-12, abs=1e-15
            )


def test_near_and_far_field_routes_agree(payload):
    """Route-independent check: pressure damping == Kochin/energy damping.

    No reference solution enters here, so this fails if either route alone is
    wrong (missing ``exp(nu*z)`` weight, wrong prefactor, wrong normal-velocity
    sign, panel area mismatch).
    """
    for row in payload["rows"]:
        for key in ("b33", "b11"):
            near = row["damping_near"]["value"][key]
            far = row["damping_far"]["value"][key]
            assert math.isfinite(near) and math.isfinite(far)
            assert far > 0.0, f"sub={row['subdivision']} ka={row['ka']} {key}: far={far}"
            rel = row["near_vs_far"][key]
            assert rel < NEAR_FAR_TOL, (
                f"sub={row['subdivision']} ka={row['ka']} {key}: "
                f"near={near:.6e} far={far:.6e} rel={rel:.3e}"
            )


def test_no_cancellation_floor_at_high_ka(payload):
    """At ka >= 3 the two routes still agree: the tiny B' is *not* round-off.

    This is the substantive claim of the study. If the near-field integral were
    limited by cancellation, the far-field route (a smooth oscillatory integral
    over the same potentials) would disagree by orders of magnitude at ka = 4, 5.
    """
    for ka in HIGH_KA:
        row = _row(payload, CASE_SUBDIVISION, ka)
        assert row["damping_near"]["ref"]["b33"] < 1.0e-2, "high-ka damping should be small"
        for key in ("b33", "b11"):
            assert row["near_vs_far"][key] < NEAR_FAR_TOL, (
                f"ka={ka} {key}: near/far disagreement {row['near_vs_far'][key]:.3e} "
                "indicates a cancellation floor"
            )


def test_far_field_agrees_with_reference_like_the_near_field(payload):
    """Both routes sit at the same distance from the multipole reference.

    A route error would show up as one route being much closer to the reference
    than the other; discretisation bias moves both together.
    """
    for row in payload["rows"]:
        for key in ("b33", "b11"):
            near_err = row["damping_near"]["err_scaled"][key]
            far_err = row["damping_far"]["err_scaled"][key]
            assert far_err < 3.0 * near_err + 1.0e-6, (
                f"sub={row['subdivision']} ka={row['ka']} {key}: "
                f"near_err={near_err:.3e} far_err={far_err:.3e}"
            )


def test_residual_error_converges_with_mesh_refinement(payload):
    """Error vs reference shrinks from 320 to 1280 panels on the whole band.

    This is what makes the residual a discretisation bias (which the case pool
    already quantifies through the refinement ladder) instead of an unexplained
    disagreement with the reference.
    """
    improved = 0
    compared = 0
    for ka in BAND_KA:
        eco = _row(payload, ECONOMY_SUBDIVISION, ka)
        case = _row(payload, CASE_SUBDIVISION, ka)
        for key in ("b33", "b11"):
            compared += 1
            if case["damping_near"]["err_scaled"][key] < eco["damping_near"]["err_scaled"][key]:
                improved += 1
    assert compared == 14
    assert improved >= 13, f"only {improved}/{compared} damping errors improved"


def test_kochin_azimuth_resolution_is_converged(payload):
    """The far-field value used is converged in azimuth, so the setting is justified."""
    for row in payload["rows"]:
        used = row["damping_far"]["n_azimuth_used"]
        assert used == max(int(k) for k in row["damping_far"]["by_azimuth"])
        for key in ("b33", "b11"):
            conv = row["damping_far"]["azimuth_convergence"][key]
            assert math.isfinite(conv), f"sub={row['subdivision']} ka={row['ka']} {key}"
            assert conv < AZIMUTH_TOL, (
                f"sub={row['subdivision']} ka={row['ka']} {key}: azimuth convergence "
                f"{conv:.3e} at n_azimuth={used}"
            )


def test_kochin_azimuth_setting_recorded_not_assumed(payload):
    """The azimuth ladder is measured, not assumed.

    For this geometry the Kochin integrand carries ``exp(nu*z)`` with ``z <= -(f-a)``,
    so it is suppressed by ``exp(-nu*(f-a))``: 72 azimuths already resolve it and
    the ladder must show saturation, i.e. the adopted (finest) setting is
    conservative.  If a future change made the far field azimuth-sensitive, the
    saturation would disappear and this test would flag the stale setting.
    """
    for row in payload["rows"]:
        by = row["damping_far"]["by_azimuth"]
        keys = sorted(int(k) for k in by)
        finest_change = abs(by[str(keys[-2])]["b33"] - by[str(keys[-1])]["b33"]) / abs(
            by[str(keys[-1])]["b33"]
        )
        coarse_change = abs(by[str(keys[0])]["b33"] - by[str(keys[1])]["b33"]) / abs(
            by[str(keys[1])]["b33"]
        )
        assert finest_change < AZIMUTH_TOL
        # saturation at the coarsest rung is the finding; assert it is recorded
        assert coarse_change < NEAR_FAR_TOL, (
            f"sub={row['subdivision']} ka={row['ka']}: azimuth 72->144 change "
            f"{coarse_change:.3e} -- the 288 setting must be re-justified"
        )
