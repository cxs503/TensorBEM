"""Acceptance tests for round 2 of ``CASE-BEM-ANNULUS-LAPLACE`` (eccentric annulus).

The case is a *record*, so these tests check the record: the eccentric geometry is
nested and its hole normals point into the hole, the closed forms are harmonic and
(for ``bipolar_ratio``) constant on each loop, the advertised budget meets the 3 %
threshold, convergence is second order, and both negative controls fire.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import tensorbem.benchmarks.eccentric_annulus as case_mod
from tensorbem.benchmarks.eccentric_annulus import (
    BUDGET_PANELS,
    CASE_ID,
    CONSTANCY_MAX,
    MIN_PROBE_BAND_FRACTION,
    ORDER_MIN,
    QUICK_LADDER,
    TOLERANCE,
    constancy_check,
    expected_fluxes,
    fitted_order,
    harmonicity_check,
    make_cases,
    measure_level,
    orientation_control,
    probe_band,
    probe_points,
    probe_rays,
    run_case,
)
from tensorbem.multiloop import (
    boundary_flux,
    domain_membership,
    eccentric_annulus_geometry,
    eccentric_annulus_mesh,
)

GEOMETRY = eccentric_annulus_geometry()
CASES = make_cases()
BY_NAME = {case.name: case for case in CASES}


@pytest.fixture(scope="module")
def quick_result() -> dict:
    """One ladder up to the advertised budget, sweeps off (test cost)."""
    return run_case(ladder=QUICK_LADDER, with_sweeps=False)


def test_case_id_and_thresholds_are_declared() -> None:
    assert CASE_ID == "CASE-BEM-ANNULUS-LAPLACE"
    assert BUDGET_PANELS == (64, 32)
    assert TOLERANCE == 0.03
    assert ORDER_MIN == 1.5
    assert all(pair[0] == 2 * pair[1] for pair in QUICK_LADDER)


def test_geometry_is_nested_and_foci_are_outside_the_fluid() -> None:
    """The reference is only exact if both foci stay clear of the fluid."""
    assert GEOMETRY["min_clearance"] > 0.0
    assert GEOMETRY["min_clearance"] < GEOMETRY["max_clearance"]
    assert GEOMETRY["relative_eccentricity"] > 0.0
    assert GEOMETRY["focus_plus_inside_inner"] is True
    assert GEOMETRY["focus_minus_outside_outer"] is True
    # consistency of the closed-form geometry: radius/centre of the tau=const circle
    for key in ("outer", "inner"):
        tau = GEOMETRY[f"{key}_parameter"]
        radius = GEOMETRY["half_distance"] / math.sinh(tau)
        centre = GEOMETRY["half_distance"] / math.tanh(tau)
        assert radius == pytest.approx(GEOMETRY[f"{key}_radius"], rel=1e-14)
        assert centre == pytest.approx(GEOMETRY[f"{key}_centre_x"], rel=1e-14)
    almost = GEOMETRY["outer_radius"] - (
        GEOMETRY["centre_offset"] + GEOMETRY["inner_radius"]
    )
    assert GEOMETRY["min_clearance"] == pytest.approx(almost, rel=1e-13)


def test_mesh_hole_normals_point_into_the_hole() -> None:
    """A hole normal must oppose the radial direction of the inner circle."""
    mesh = eccentric_annulus_mesh(*BUDGET_PANELS)
    assert mesh.hole_loops == (1,)
    inner = mesh.loops[1]
    radial = inner.centers - np.array([GEOMETRY["inner_centre_x"], 0.0])
    radial /= np.linalg.norm(radial, axis=1)[:, None]
    cosine = np.einsum("ij,ij->i", mesh.normals[mesh.loop_slices[1]], radial)
    assert np.all(cosine < 0.0)
    # outer loop normals point out of the fluid, i.e. along the radial direction
    outer_radial = mesh.loops[0].centers - np.array([GEOMETRY["outer_centre_x"], 0.0])
    outer_radial /= np.linalg.norm(outer_radial, axis=1)[:, None]
    outer_cosine = np.einsum("ij,ij->i", mesh.normals[mesh.loop_slices[0]], outer_radial)
    assert np.all(outer_cosine > 0.0)


def test_mesh_refuses_inscribed_polygons_that_overlap() -> None:
    """A positive circle clearance can still collapse on an inscribed mesh."""
    # (4, 3) panels on a sliver: the inscribed polygons overlap even though the
    # exact circles are still separated by a positive clearance
    with pytest.raises(ValueError, match="overlap"):
        eccentric_annulus_mesh(4, 3, inner_parameter=0.805)


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_reference_is_harmonic(case) -> None:
    """The closed form must be harmonic, not merely cited."""
    report = harmonicity_check(case, GEOMETRY)
    assert report["max_abs_laplacian_relative"] < 1.0e-9


def test_ratio_reference_is_constant_on_each_loop() -> None:
    """``bipolar_ratio`` equals ``-tau`` on both circles (bipolar coordinates)."""
    case = BY_NAME["bipolar_ratio"]
    assert case.constant_on_loops is True
    report = constancy_check(case, GEOMETRY)
    assert report["max_spread_relative"] < CONSTANCY_MAX * 1.0e-4
    assert report["constants_differ"] is True
    for loop in report["loops"]:
        assert loop["value"] == pytest.approx(loop["analytic_constant_minus_tau"], rel=1e-12)


def test_only_constant_cases_carry_the_flag() -> None:
    """The constancy gate must not be applied to a varying trace."""
    assert BY_NAME["point_source"].constant_on_loops is False
    assert BY_NAME["hole_dipole"].constant_on_loops is False


def test_exact_closed_form_flux_matches_the_source_strength() -> None:
    """Analytic gradients integrate to +-2 pi (source) and 0 (doublet)."""
    mesh = eccentric_annulus_mesh(*BUDGET_PANELS)
    for name, wanted in (
        ("point_source", (2.0 * math.pi, -2.0 * math.pi)),
        ("hole_dipole", (0.0, 0.0)),
    ):
        case = BY_NAME[name]
        assert expected_fluxes(case) == wanted
        exact_q = case.normal_derivative(mesh.centers, mesh.normals)
        flux = boundary_flux(mesh, exact_q)
        scale = float(np.sum(np.abs(exact_q) * mesh.lengths))
        # the meshes are inscribed polygons, so the *exact* data integrates to the
        # analytic flux only up to the geometric error of the polygon (~5e-3 here)
        for got, want in zip(flux, wanted):
            assert abs(got - want) / scale < 5.0e-3


def test_budget_meets_tolerance(quick_result: dict) -> None:
    """The advertised 64/32 mesh must pass on its own, for every case."""
    for entry in quick_result["cases"]:
        budget = entry["budget_level"]
        assert budget["q_relative_l2_error"] < TOLERANCE
        assert budget["q_relative_linf_error"] < TOLERANCE
        assert budget["field_relative_max_error"] < TOLERANCE
        assert budget["flux_relative_error"] < TOLERANCE
        assert budget["flux_identity_relative"] < TOLERANCE


def test_second_order_convergence(quick_result: dict) -> None:
    for entry in quick_result["cases"]:
        assert entry["fitted_order_q_relative_l2"] > ORDER_MIN
        assert entry["fitted_order_field_relative_max"] > ORDER_MIN


def test_probe_points_lie_in_the_fluid() -> None:
    mesh = eccentric_annulus_mesh(*BUDGET_PANELS)
    points = probe_points(GEOMETRY)
    in_hole, outside = domain_membership(mesh, points)
    assert not bool(np.any(in_hole))
    assert not bool(np.any(outside))
    usable, skipped = probe_rays(GEOMETRY)
    assert len(usable) + skipped == case_mod.PROBE_ANGLES
    assert points.shape[0] == len(usable) * len(case_mod.PROBE_FRACTIONS)


def test_thin_annulus_skips_probe_rays_and_eventually_refuses() -> None:
    """A sliver is recorded as a skipped ray, and a collapsed one as a refusal."""
    thin = eccentric_annulus_geometry(0.8, 0.82)
    usable, skipped = probe_rays(thin)
    assert skipped > 0
    assert len(usable) >= 4
    assert probe_points(thin).shape[0] == len(usable) * 3
    collapsed = eccentric_annulus_geometry(0.8, 0.805)
    with pytest.raises(ValueError, match="too thin"):
        probe_points(collapsed)


def test_probe_band_is_a_subset_of_the_ray() -> None:
    for phi in np.linspace(0.0, 2.0 * np.pi, 7):
        start, end = probe_band(GEOMETRY, float(phi))
        assert 0.0 <= start < end <= GEOMETRY["outer_radius"]
    assert MIN_PROBE_BAND_FRACTION > 0.0


def test_probe_guard_refuses_hole_and_exterior() -> None:
    report = case_mod.probe_guard_control()
    assert report["passed"] is True
    assert [row["probe"] for row in report["probes"]] == ["inside_hole", "outside_outer"]


def test_orientation_control_fires() -> None:
    """Unflipped hole normals: residual unchanged, flux wrong by orders."""
    control = orientation_control(BY_NAME["bipolar_ratio"])
    assert control["caught"] is True
    assert control["error_amplification"] > 10.0
    assert control["unflipped_relative_residual"] < 1.0e-10


def test_verdict_passes_and_reports_failures(quick_result: dict) -> None:
    verdict = quick_result["verdict"]
    assert verdict["passed"] is True
    assert verdict["failed_checks"] == []
    tampered = {key: value for key, value in quick_result.items()}
    tampered["cases"] = [
        dict(entry, budget_level=dict(entry["budget_level"], q_relative_l2_error=0.5))
        if entry["case"] == "bipolar_ratio"
        else entry
        for entry in quick_result["cases"]
    ]
    broken = case_mod.verdict(tampered)
    assert broken["passed"] is False
    assert any(check["check"] == "q_relative_l2_error" for check in broken["failed_checks"])


def test_markdown_has_the_eight_sections(quick_result: dict) -> None:
    text = case_mod.render_markdown(quick_result)
    for heading in (
        "## 1. Case and geometry",
        "## 2. Method",
        "## 3. Closed-form self-check",
        "## 4. Results (convergence ladder)",
        "## 5. Sensitivity studies",
        "## 6. Failure modes (measured)",
        "## 7. Error verdict",
        "## 8. Notes and limits",
    ):
        assert heading in text
    # the tolerance statement must be explicit about the 3 % answer
    assert "is it below 3%?" in text


def test_fitted_order_measures_a_known_slope() -> None:
    levels = [{"panel_count": n, "err": n ** -2.0} for n in (16, 32, 64, 128)]
    assert fitted_order(levels, "err") == pytest.approx(2.0, abs=1e-10)


def test_eccentricity_sweep_monotone_clearance() -> None:
    """Tightening the family parameter must shrink the clearance monotonically."""
    clearances = [
        eccentric_annulus_geometry(0.8, tau)["min_clearance"]
        for tau in (2.2, 1.8, 1.4, 1.2, 1.05, 1.0, 0.9)
    ]
    assert all(later < earlier for earlier, later in zip(clearances, clearances[1:]))


def test_measure_level_reports_the_gap_in_panel_lengths() -> None:
    level = measure_level(BY_NAME["point_source"], 32, 16)
    assert level["min_clearance_over_max_panel"] == pytest.approx(
        level["min_clearance"] / max(level["mean_panel_length_outer"],
                                     level["mean_panel_length_inner"]),
        rel=1e-14,
    )
    assert level["relative_residual"] < 1.0e-10