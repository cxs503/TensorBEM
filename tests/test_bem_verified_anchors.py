"""Acceptance tests for campaign node P19-N02 (BEM verification anchors).

What is tested, in the order of the campaign table
--------------------------------------------------
1. **Hydrostatics of Wigley III** at ``200 / 800 / 3200`` wetted triangles:
   displaced volume within 1 %, wetted area within 2 %, block coefficient within
   0.5 % of the closed-form parabola integrals, plus a three-level Roache GCI
   below 3 %.  The *coarse* level is expected to sit outside the 1 % band (a
   convergence study needs a coarse member that is measurably off); that miss is
   asserted to be recorded, not swallowed.
2. **Wave resistance of the half-submerged ellipsoid** at ``Fr = 0.2/0.3/0.4``:
   panel route within 5 % of the Havelock closed form, monotone mesh convergence
   and ``U_G < 3 %`` at every Froude number.
3. **Tandem interference factor** of two identical Wigley hulls: ``sigma`` from
   the panel-discretised Kochin function reproduces the Michell-superposition
   theory (``|dsigma| < 0.02`` on the mission spacings and on a spacing sweep),
   the sign of ``sigma - 1`` agrees everywhere, ``sigma(0) = 2`` exactly, the
   un-factorised two-hull panel sum reproduces ``H(1 + e^{ikd})`` to round-off,
   and the pair resistance is within 5 % of the analytic value.

Two anti-drift gates close the loop between code, artifacts and prose: the
committed ``benchmarks/verified/*/results.json`` must reproduce a fresh run, and
``docs/bem_verified_anchors.md`` must equal :func:`render_document` of that run.
A negative control perturbs one number and proves the document gate can fail.

Modules under test: ``tensorbem.multihull`` and
``tensorbem.benchmarks.bem_verified_anchors`` (both new in this node).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks.bem_verified_anchors import (
    ELLIPSOID_LEVELS,
    FROUDES,
    HYDROSTATIC_LEVELS,
    MULTIHULL_LEVELS,
    MULTIHULL_SPACINGS_OVER_L,
    TOLERANCES,
    ellipsoid_wave_study,
    multihull_interference_study,
    render_document,
    run_all,
    wigley_hydrostatics_study,
)
from tensorbem.multihull import (
    CentreplanePanels,
    explicit_pair_kochin,
    interference_curve,
    interference_factor,
    tandem_kochin,
    wave_number_k0,
    wigley_centreplane_panels,
    wigley_panel_kochin,
)
from tensorbem.wigley import michell_kochin_function

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "benchmarks" / "verified"
DOCUMENT = REPO_ROOT / "docs" / "bem_verified_anchors.md"

WIGLEY = {"L": 3.8, "B": 0.5, "T": 0.225}


# ---------------------------------------------------------------------------
# Fixtures: one run of the three studies serves the whole module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = tmp_path_factory.mktemp("bem_verified")
    return run_all(out)


@pytest.fixture(scope="module")
def hydro(results: dict) -> dict:
    return results["bem_wigley_hydrostatics"]


@pytest.fixture(scope="module")
def wave(results: dict) -> dict:
    return results["bem_ellipsoid_wave"]


@pytest.fixture(scope="module")
def multi(results: dict) -> dict:
    return results["bem_multihull_interference"]


# ---------------------------------------------------------------------------
# 1. Wigley III hydrostatics
# ---------------------------------------------------------------------------


def test_hydrostatic_mesh_densities_are_the_mission_levels(hydro: dict) -> None:
    assert [(row["n_sections"], row["n_levels"]) for row in hydro["levels"]] == list(
        HYDROSTATIC_LEVELS
    )
    assert [row["triangles"] for row in hydro["levels"]] == [200, 800, 3200]


def test_geometry_gate_precedes_the_errors(hydro: dict) -> None:
    """The meshed hull must be the specified hull before any error is quoted."""
    finest = hydro["levels"][-1]
    assert finest["length"] == pytest.approx(WIGLEY["L"], abs=1e-12)
    assert finest["beam"] == pytest.approx(WIGLEY["B"], abs=1e-12)
    assert finest["draft"] == pytest.approx(WIGLEY["T"], abs=1e-12)
    assert finest["block_coefficient"] == pytest.approx(4.0 / 9.0, rel=1e-3)
    assert hydro["closed_form"]["volume"] == pytest.approx(4.0 * 3.8 * 0.5 * 0.225 / 9.0)


def test_wetted_area_reference_is_converged(hydro: dict) -> None:
    assert hydro["closed_form"]["wetted_area_reference_converged"] is True
    assert hydro["closed_form"]["wetted_area_quadrature"] > 0.0


def test_finest_mesh_meets_the_campaign_bands(hydro: dict) -> None:
    finest = hydro["levels"][-1]
    assert abs(finest["volume_error"]) < TOLERANCES["volume_error"]
    assert abs(finest["wetted_area_error"]) < TOLERANCES["wetted_area_error"]
    assert abs(finest["block_coefficient_error"]) < TOLERANCES["block_coefficient_error"]
    assert hydro["verdict"] == "PASS"


def test_volume_converges_second_order_and_gci_is_below_three_percent(hydro: dict) -> None:
    gci = hydro["gci"]["volume"]
    errors = [abs(row["volume_error"]) for row in hydro["levels"]]
    assert errors[0] > errors[1] > errors[2], errors  # monotone under refinement
    assert gci["converged"] is True
    assert 1.5 < gci["observed_order"] < 2.5  # quadratic faceting error
    assert gci["gci_fine"] < TOLERANCES["gci_fine"]
    assert hydro["gci"]["wetted_area"]["gci_fine"] < TOLERANCES["gci_fine"]


def test_coarse_level_miss_is_recorded_not_hidden(hydro: dict) -> None:
    """The 200-triangle mesh cannot hold the 1 % volume band; say so."""
    assert hydro["strict_all_levels_verdict"] == "FAIL"
    misses = hydro["coarse_level_misses"]
    assert [row["triangles"] for row in misses] == [200]
    error = abs(misses[0]["volume_error"])
    assert 0.01 < error < 0.02, error
    assert hydro["per_level_acceptance"][0]["volume_within_1pct"] is False


def test_brief_block_coefficient_is_audited(hydro: dict) -> None:
    """The brief's Cb = 0.667 is Cp of this hullform; the code says so."""
    audit = hydro["brief_audit"]
    assert audit["brief_block_coefficient"] == pytest.approx(0.667, abs=1e-12)
    assert audit["closed_form_block_coefficient"] == pytest.approx(4.0 / 9.0)
    assert audit["closed_form_prismatic_coefficient"] == pytest.approx(2.0 / 3.0)
    assert "prismatic" in audit["note"]


# ---------------------------------------------------------------------------
# 2. Ellipsoid wave resistance
# ---------------------------------------------------------------------------


def test_ellipsoid_geometry_gate(wave: dict) -> None:
    geometry = wave["geometry"]
    assert geometry["volume_relative_error"] == pytest.approx(0.0, abs=1e-8)
    assert geometry["block_coefficient"] == pytest.approx(math.pi / 6.0, rel=1e-12)


def test_ellipsoid_three_froude_numbers_are_covered(wave: dict) -> None:
    assert [row["froude"] for row in wave["froude_cases"]] == list(FROUDES)
    for row in wave["froude_cases"]:
        assert [level["panels"] for level in row["levels"]] == [
            n_r * n_phi for n_r, n_phi in ELLIPSOID_LEVELS
        ]
        assert row["wave_resistance_analytic"] > 0.0


def test_ellipsoid_panel_resistance_within_five_percent(wave: dict) -> None:
    for row in wave["froude_cases"]:
        assert abs(row["finest_relative_error"]) < TOLERANCES["wave_resistance_error"], (
            row["froude"],
            row["finest_relative_error"],
        )


def test_ellipsoid_convergence_is_monotone_with_gci_below_three_percent(wave: dict) -> None:
    for row in wave["froude_cases"]:
        errors = [abs(level["relative_error"]) for level in row["levels"]]
        assert errors[0] > errors[1] > errors[2], (row["froude"], errors)
        gci = row["gci"]
        assert gci["converged"] is True, row["froude"]
        assert gci["gci_fine"] < TOLERANCES["gci_fine"], row["froude"]
        assert 1.5 < gci["observed_order"] < 2.5


def test_ellipsoid_kochin_converges_to_the_bessel_form(wave: dict) -> None:
    rows = wave["kochin_convergence"]
    key_fine = f"rel_error_{ELLIPSOID_LEVELS[-1][0]}x{ELLIPSOID_LEVELS[-1][1]}"
    key_coarse = f"rel_error_{ELLIPSOID_LEVELS[0][0]}x{ELLIPSOID_LEVELS[0][1]}"
    for row in rows:
        assert row[key_fine] < row[key_coarse]
        # the elliptical rim is a sqrt-singularity of y_x, so the panel sum is
        # first-order there: sub-1 % only for k <= 2, a few 0.1 % at k = 4-8.
        assert row[key_fine] < 1.5e-2


# ---------------------------------------------------------------------------
# 3. Tandem multihull interference
# ---------------------------------------------------------------------------


def test_multihull_mission_spacings_have_the_theory_interference_factor(multi: dict) -> None:
    assert [row["spacing_over_L"] for row in multi["spacings"]] == list(
        MULTIHULL_SPACINGS_OVER_L
    )
    for row in multi["spacings"]:
        deviation = abs(row["sigma_panel_finest"] - row["sigma_theory"])
        assert deviation < TOLERANCES["sigma_absolute"], (row["spacing_over_L"], deviation)
        assert abs(row["pair_resistance_relative_error"]) < TOLERANCES[
            "sigma_pair_resistance"
        ]


def test_multihull_sigma_trend_follows_michell(multi: dict) -> None:
    sweep = multi["sweep"]
    assert all(sweep["sign_matches"])
    assert sweep["max_absolute_difference"] < TOLERANCES["sigma_absolute"]
    # sigma oscillates about 1: some spacings are beneficial, some detrimental.
    values = np.asarray(sweep["sigma_theory"])
    assert values.min() < 0.9 < 1.1 < values.max()


def test_multihull_limiting_and_asymptotic_behaviour(multi: dict) -> None:
    assert multi["sigma_at_zero_spacing"] == pytest.approx(2.0, abs=1e-9)
    assert multi["criteria"]["asymptotic_sigma_approaches_one"] is True
    for value in multi["asymptotic"]["sigma"].values():
        assert abs(value - 1.0) < TOLERANCES["asymptotic_sigma_band"]


def test_multihull_superposition_identity_is_exact(multi: dict) -> None:
    for value in multi["superposition_identity"]["relative"].values():
        assert value < 1e-12


def test_multihull_panel_kochin_converges_to_michell(multi: dict) -> None:
    rows = multi["panel_kochin_convergence"]
    for row in rows[1:]:
        assert row["rel_error_k2"] < 1e-3
    assert rows[-1]["rel_error_k2"] < rows[0]["rel_error_k2"]


# ---------------------------------------------------------------------------
# 4. Multihull module contracts (independent of the study driver)
# ---------------------------------------------------------------------------


def test_panel_grid_shapes_and_validation() -> None:
    panels = wigley_centreplane_panels(3.8, 0.5, 0.225, n_x=10, n_z=5)
    assert panels.panel_count == 50
    assert panels.y_x.shape == (10, 5)
    assert np.all(np.isfinite(panels.area))
    with pytest.raises(ValueError):
        wigley_centreplane_panels(3.8, 0.5, 0.225, n_x=0, n_z=5)
    with pytest.raises(ValueError):
        wigley_centreplane_panels(3.8, 0.5, 0.225, n_x=4, n_z=4, scheme="nope")
    with pytest.raises(ValueError):
        # x_edges must carry n_x + 1 entries
        CentreplanePanels(
            x_edges=np.zeros(2), z_edges=np.zeros(3), y_x=np.zeros((2, 2)), length=1.0,
            n_x=2, n_z=2,
        )


def test_panel_kochin_matches_the_closed_form_michell_function() -> None:
    L, B, T = WIGLEY["L"], WIGLEY["B"], WIGLEY["T"]
    for k in (1.0, 2.0, 5.0):
        analytic = michell_kochin_function(L, B, T, k)
        panel = wigley_panel_kochin(k, L, B, T, n_x=80, n_z=40)
        assert abs(panel / analytic - 1.0) < 1e-3


def test_centroid_scheme_is_worse_than_the_integrated_scheme() -> None:
    """The cruder panel rule must show a larger error -- otherwise it is untested."""
    L, B, T = WIGLEY["L"], WIGLEY["B"], WIGLEY["T"]
    k = 5.0
    analytic = michell_kochin_function(L, B, T, k)
    integrated = wigley_panel_kochin(k, L, B, T, n_x=20, n_z=10)
    centroid = wigley_panel_kochin(k, L, B, T, n_x=20, n_z=10, scheme="centroid")
    assert abs(centroid / analytic - 1.0) > abs(integrated / analytic - 1.0)


def test_tandem_kochin_is_the_superposition_phase_factor() -> None:
    k = 1.7
    spacing = 1.25
    base = 0.5 - 0.25j
    pair = tandem_kochin(lambda _k: base, spacing)(k)
    assert pair == pytest.approx(base * (1.0 + np.exp(1j * k * spacing)))
    assert tandem_kochin(lambda _k: base, 0.0)(k) == pytest.approx(2.0 * base)


def test_wave_number_convention() -> None:
    k0 = wave_number_k0(0.3, 3.8)
    assert k0 == pytest.approx(1.0 / (0.3**2 * 3.8))
    with pytest.raises(ValueError):
        wave_number_k0(0.0, 3.8)


def test_explicit_pair_sum_equals_the_factorised_form() -> None:
    L, B, T = WIGLEY["L"], WIGLEY["B"], WIGLEY["T"]
    k, spacing = 2.3, 0.8 * 3.8
    explicit = explicit_pair_kochin(k, L, B, T, spacing=spacing, n_x=40, n_z=20)
    factorised = wigley_panel_kochin(k, L, B, T, n_x=40, n_z=20) * (
        1.0 + np.exp(1j * k * spacing)
    )
    assert abs(explicit - factorised) <= 1e-12 * abs(explicit)


def test_interference_curve_matches_pointwise_calls() -> None:
    L, B, T = WIGLEY["L"], WIGLEY["B"], WIGLEY["T"]
    k0 = wave_number_k0(0.3, L)

    def kochin(k: float) -> complex:
        return michell_kochin_function(L, B, T, k)

    spacings = [0.5 * L, 1.0 * L]
    curve = interference_curve(kochin, k0=k0, spacings=spacings)
    pointwise = [interference_factor(kochin, k0=k0, spacing=d) for d in spacings]
    assert np.allclose(curve, pointwise, rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. Artifact and document drift gates
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_close(recorded, fresh, path: str = "") -> None:
    if isinstance(recorded, dict):
        assert isinstance(fresh, dict), path
        assert set(recorded) == set(fresh), (path, set(recorded) ^ set(fresh))
        for key in recorded:
            _assert_close(recorded[key], fresh[key], f"{path}/{key}")
    elif isinstance(recorded, list):
        assert isinstance(fresh, list) and len(recorded) == len(fresh), path
        for i, (a, b) in enumerate(zip(recorded, fresh)):
            _assert_close(a, b, f"{path}[{i}]")
    elif isinstance(recorded, bool) or isinstance(fresh, bool):
        assert recorded == fresh, path
    elif isinstance(recorded, (int, float)) and isinstance(fresh, (int, float)):
        assert fresh == pytest.approx(recorded, rel=1e-9, abs=1e-12), (path, recorded, fresh)
    else:
        assert recorded == fresh, path


def test_committed_artifacts_reproduce_from_the_current_code(results: dict) -> None:
    for case, result in results.items():
        path = ARTIFACTS / case / "results.json"
        assert path.is_file(), path
        recorded = _read_json(path)
        fresh = json.loads(json.dumps(result, default=str))
        _assert_close(recorded, fresh, case)


def test_case_directories_contain_report_and_figure(results: dict) -> None:
    for case, result in results.items():
        case_dir = ARTIFACTS / case
        report = case_dir / "report.md"
        assert report.is_file()
        assert case in report.read_text(encoding="utf-8")
        figure = result["figure"]
        assert figure is not None, case
        assert Path(figure).parent.name == "figures"
        image = case_dir / "figures" / Path(figure).name
        assert image.is_file(), (case, figure)
        assert image.stat().st_size < 60_000, (case, image.stat().st_size)


def test_document_is_the_rendered_result(results: dict) -> None:
    assert DOCUMENT.is_file(), DOCUMENT
    assert DOCUMENT.read_text(encoding="utf-8") == render_document(results)


def test_document_gate_detects_a_perturbed_number(results: dict) -> None:
    """Negative control: the drift gate must be able to fail."""
    tampered = json.loads(json.dumps(results, default=str))
    row = tampered["bem_ellipsoid_wave"]["froude_cases"][0]
    row["levels"][-1]["wave_resistance"] *= 1.05
    assert render_document(tampered) != DOCUMENT.read_text(encoding="utf-8")


def test_summary_lists_a_verdict_per_case(results: dict) -> None:
    summary = _read_json(ARTIFACTS / "summary.json")
    assert summary["mission"] == "P19-N02"
    assert set(summary["cases"]) == set(results)
    for case in summary["cases"]:
        assert summary["cases"][case] in {"PASS", "FAIL"}


# ---------------------------------------------------------------------------
# 6. Individual study entry points stay callable and self-consistent
# ---------------------------------------------------------------------------


def test_studies_are_reproducible_standalone() -> None:
    hydro = wigley_hydrostatics_study()
    wave = ellipsoid_wave_study()
    multi = multihull_interference_study()
    assert hydro["criteria"]["finest_volume_within_1pct"] is True
    assert wave["criteria"]["all_froudes_error_below_5pct"] is True
    assert multi["criteria"]["sigma_curve_matches_theory"] is True