"""Machine gate for interior (lid) panel irregular-frequency suppression.

Every claim of :mod:`tensorbem.benchmarks.interior_panel_suppression` is
re-derived from the solver here, including the **negative control**: with
``interior_panels=None`` the spurious band must trigger the failure criteria the
study relies on.  Panel counts are kept at 64/320 and the frequency lists short
so the module stays well under a minute per test process.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.analytic import (
    hulme_hemisphere_radiation,
    submerged_sphere_wave_multipole,
    ursell_cylinder_added_mass,
)
from tensorbem.benchmarks import interior_panel_suppression as study
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import (
    box_hull_mesh,
    hemisphere_hull_mesh,
    icosphere,
    TriangleMesh,
)
from tensorbem.interior_panels import (
    InteriorPanelSet,
    _point_inside_hull,
    interior_panel_points,
)

RADIUS = 1.0


@pytest.fixture(scope="module")
def hemisphere():
    return hemisphere_hull_mesh(2, radius=RADIUS)  # 64 panels


@pytest.fixture(scope="module")
def lid(hemisphere):
    return interior_panel_points(hemisphere)


# ---------------------------------------------------------------------------
# Interior panel set (geometry & validation)
# ---------------------------------------------------------------------------
def test_lid_points_lie_strictly_inside_the_hull(hemisphere, lid):
    """Every lid point is below z = 0 and inside the closed hull surface."""
    assert lid.count >= 8
    assert np.all(lid.points[:, 2] < 0.0)
    inside = _point_inside_hull(lid.points, hemisphere)
    assert bool(np.all(inside))
    # The interior waterplane of a unit hemisphere has radius 1.
    assert lid.waterline_radius == pytest.approx(RADIUS, rel=1e-9)
    assert np.all(np.hypot(lid.points[:, 0], lid.points[:, 1]) < lid.waterline_radius)


def test_lid_ring_structure_follows_the_requested_resolution(hemisphere):
    coarse = interior_panel_points(hemisphere, rings=2, azimuth=6)
    fine = interior_panel_points(hemisphere, rings=3, azimuth=12)
    assert fine.count > coarse.count
    # Uniform panel areas are the point of the concentric-ring layout.
    for panel_set in (coarse, fine):
        spread = float(np.max(panel_set.areas) / np.min(panel_set.areas))
        assert spread < 25.0


def test_lid_rejects_a_closed_submerged_mesh():
    """A body without a waterplane has no interior lid to discretise."""
    base = icosphere(2, radius=RADIUS)
    submerged = TriangleMesh(
        base.vertices + np.array([0.0, 0.0, -2.0]), base.faces
    )
    with pytest.raises(ValueError):
        interior_panel_points(submerged)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rings": 0},
        {"rings": 2, "azimuth": 3},
        {"shrink": 0.0},
        {"shrink": 1.0},
        {"depth_fraction": 0.0},
    ],
)
def test_lid_rejects_invalid_resolution(hemisphere, kwargs):
    with pytest.raises(ValueError):
        interior_panel_points(hemisphere, **kwargs)


def test_lid_validation_catches_points_that_escape_the_hull(hemisphere):
    """A lid hugging the waterline at great depth must be rejected, not hidden."""
    with pytest.raises(ValueError):
        interior_panel_points(hemisphere, shrink=0.99, depth_fraction=0.95)


def test_ray_cast_classifies_obvious_points(hemisphere):
    probes = np.array(
        [[5.0, 0.0, 0.0], [0.0, 0.0, -5.0], [0.0, 0.0, 5.0], [0.5, 0.0, -0.8]]
    )
    assert list(_point_inside_hull(probes, hemisphere)) == [False, False, False, True]


# ---------------------------------------------------------------------------
# Solver plumbing
# ---------------------------------------------------------------------------
def test_solver_stacks_lid_rows_and_records_them(hemisphere, lid):
    ops = assemble_free_surface_operators_3d(hemisphere, 1.0, interior_panels="auto")
    assert ops.chief_points.shape[0] == lid.count
    assert ops.interior_lid_points is not None
    assert ops.interior_lid_points.shape[0] == lid.count
    assert ops.system_matrix.shape == (hemisphere.panel_count + lid.count, hemisphere.panel_count)

    bare = assemble_free_surface_operators_3d(hemisphere, 1.0)
    assert bare.chief_points.shape[0] == 0
    assert bare.interior_lid_points is None


def test_solver_accepts_an_interior_panel_set_object(hemisphere, lid):
    ops = assemble_free_surface_operators_3d(hemisphere, 1.0, interior_panels=lid)
    assert isinstance(lid, InteriorPanelSet)
    assert ops.chief_points.shape[0] == lid.count


def test_solver_rejects_points_on_or_above_the_waterplane(hemisphere):
    bad = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.1]])
    with pytest.raises(ValueError):
        assemble_free_surface_operators_3d(hemisphere, 1.0, interior_panels=bad)
    with pytest.raises(ValueError):
        assemble_free_surface_operators_3d(hemisphere, 1.0, interior_panels="bogus")


def test_lid_rows_are_inert_away_from_the_spurious_band(hemisphere):
    """Off-resonance the lid must not move the answer beyond discretisation noise."""
    ka = 1.0
    bare = solve_radiation_3d(hemisphere, ka, modes=(2,))
    lid = solve_radiation_3d(hemisphere, ka, modes=(2,), interior_panels="auto")
    assert abs(lid.added_mass[0, 0] / bare.added_mass[0, 0] - 1.0) < 5.0e-3
    assert abs(lid.damping[0, 0] / bare.damping[0, 0] - 1.0) < 5.0e-3


# ---------------------------------------------------------------------------
# Negative control: the unsuppressed band must fail the criteria
# ---------------------------------------------------------------------------
def _band_cases(level: int, config: str, kas):
    return [study.hemisphere_point(level, ka, config) for ka in kas]


def test_negative_control_unregularised_band_breaks_conditioning(hemisphere, lid):
    """Without interior rows the near-singular band inflates cond(A) sharply."""
    kas = (2.40, 2.50, 2.60, 2.70)
    bare = _band_cases(64, "none", kas)
    suppressed = _band_cases(64, "lid", kas)
    assert max(c["condition_number"] for c in bare) > 2.0 * min(
        c["condition_number"] for c in bare
    )
    # The lid keeps the matrix well conditioned across the same band.
    assert max(c["condition_number"] for c in suppressed) < max(
        c["condition_number"] for c in bare
    )


def test_negative_control_lid_changes_the_band_but_not_the_reference_region(hemisphere):
    """The suppression acts where it must: inside the band it moves the answer."""
    band_ka = 2.60
    bare = study.hemisphere_point(64, band_ka, "none")
    lid = study.hemisphere_point(64, band_ka, "lid")
    off_ka = 1.0
    bare_off = study.hemisphere_point(64, off_ka, "none")
    lid_off = study.hemisphere_point(64, off_ka, "lid")
    band_shift = abs(lid["added_mass_33"] / bare["added_mass_33"] - 1.0)
    off_shift = abs(lid_off["added_mass_33"] / bare_off["added_mass_33"] - 1.0)
    assert band_shift > 10.0 * off_shift


# ---------------------------------------------------------------------------
# Analytical anchors
# ---------------------------------------------------------------------------
def test_heave_rao_reduces_to_the_expected_limits():
    """At ka -> 0 with no radiation the RAO must tend to 1 (body follows the surface)."""
    from types import SimpleNamespace

    fake = SimpleNamespace(volume=1.0, waterplane_area=1.0)
    # C33 - w^2 M + i w B with tiny w -> C33; |F| = C33 gives RAO -> 1.
    ka = 1.0e-8
    c33 = study.DENSITY * study.GRAVITY * 1.0
    rao = study.heave_rao(fake, ka, 0.0, 0.0, complex(c33, 0.0))
    assert rao == pytest.approx(1.0, rel=1e-6)


def test_anchor_box_restoring_is_rho_g_awp():
    row = study.anchor_box()
    assert row["waterplane_area_mesh"] == pytest.approx(row["waterplane_area_exact"], rel=1e-12)
    assert row["c33_rel_error"] < 1.0e-9


def test_anchor_box_low_frequency_heave_rao_tends_to_one():
    row = study.anchor_box_low_frequency_rao(ka=0.05)
    assert row["rao_heave"] == pytest.approx(1.0, abs=0.05)


def test_anchor_ursell_added_mass_and_damping():
    row = study.anchor_ursell(ka=0.6, panel_count=96)
    # Reference values are recomputed from Ursell's multipole series, not quoted.
    exact_a, exact_b = ursell_cylinder_added_mass(
        RADIUS, 2.0 * RADIUS, math.sqrt(0.6 * study.GRAVITY / RADIUS)
    )
    assert row["reference_added_mass"] == pytest.approx(exact_a, rel=1e-12)
    assert row["reference_damping"] == pytest.approx(exact_b, rel=1e-12)
    assert row["added_mass_rel_error"] < 2.0e-2
    assert row["damping_rel_error"] < 2.0e-2


def test_anchor_sphere_matches_wang_linton_at_320_panels():
    row = study.anchor_sphere(panel_levels=(320,), ka=0.6)
    ref = submerged_sphere_wave_multipole(0.6, radius=RADIUS, submergence=2.0)
    assert row["reference_added_mass_33"] == pytest.approx(float(ref.heave_added_mass), rel=1e-12)
    # 320 panels is a deliberately coarse anchor point: the study pins the same
    # errors (A33 ~2.1 %, B33 ~5.5 %) and shows the h-based order ~2 toward 1280.
    assert row["max_added_mass_rel_error"] < 3.0e-2
    assert row["max_damping_rel_error"] < 6.0e-2


def test_sphere_interior_rows_do_not_degrade_a_converged_case():
    """Suppression rows are inert for a submerged body (|Delta| < 0.5 %)."""
    bare = study.anchor_sphere(panel_levels=(320,), ka=0.6)
    with_rows = study.anchor_sphere(panel_levels=(320,), ka=0.6, interior_rows=True)
    assert with_rows["rows"][0]["interior_rows"] == 5
    for key in ("added_mass_33", "damping_33"):
        delta = abs(with_rows["rows"][0][key] / bare["rows"][0][key] - 1.0)
        assert delta < 5.0e-3


def test_hemisphere_anchor_error_shrinks_with_panels():
    coarse = study.anchor_hemisphere_accuracy(level=64, ka_list=(1.0,), config="lid")
    fine = study.anchor_hemisphere_accuracy(level=256, ka_list=(1.0,), config="lid")
    assert fine["rows"][0]["added_mass_rel_error"] < coarse["rows"][0]["added_mass_rel_error"]
    ref = hulme_hemisphere_radiation(1.0, radius=RADIUS)
    assert fine["rows"][0]["reference_added_mass_33_dimless"] == pytest.approx(
        float(ref.heave_added_mass_dimless), rel=1e-12
    )


# ---------------------------------------------------------------------------
# Study bookkeeping
# ---------------------------------------------------------------------------
def test_predeclared_frequency_grids():
    assert len(study.KA_MAIN) == 30
    assert study.KA_MAIN[0] == pytest.approx(0.1)
    assert study.KA_MAIN[-1] == pytest.approx(3.0)
    assert np.allclose(np.diff(study.KA_MAIN), 0.1, atol=1e-9)
    assert np.allclose(np.diff(study.KA_FINE), 0.01, atol=1e-12)
    assert study.SPURIOUS_BAND[0] >= study.KA_FINE[0]
    assert study.SPURIOUS_BAND[1] <= study.KA_FINE[-1]
    assert study.ka_grid(64) == study.KA_MAIN + study.KA_FINE
    assert study.ka_grid(1024) == study.KA_MAIN


def test_second_difference_metric_matches_a_known_curvature():
    kas = [2.40 + 0.01 * i for i in range(11)]
    points = [{"ka": ka, "rao_heave": 3.0 * ka * ka} for ka in kas]
    metrics = study.spurious_peak_metrics(points, "rao_heave")
    assert metrics["max_curvature"] == pytest.approx(6.0, rel=1e-6)
    non_uniform = [{"ka": 2.40, "rao_heave": 1.0}, {"ka": 2.405, "rao_heave": 1.0},
                   {"ka": 2.42, "rao_heave": 1.0}]
    with pytest.raises(ValueError):
        study.spurious_peak_metrics(non_uniform, "rao_heave")


def test_summary_and_reduction_are_consistent():
    cases = []
    for config, scale in (("none", 3.0), ("lid", 0.4), ("chief", 0.6)):
        for ka in (2.42, 2.43, 2.44, 2.45, 2.46):
            cases.append(
                {
                    "level": 64,
                    "ka": ka,
                    "config": config,
                    "panels": 64,
                    "interior_rows": 0 if config == "none" else 18,
                    "condition_number": 1.0,
                    "relative_residual": 0.0,
                    "added_mass_33_dimless": scale * (ka - 2.44) ** 2,
                    "damping_33_dimless": 0.1,
                    "reference_added_mass_33_dimless": 0.0,
                    "reference_damping_33_dimless": 0.1,
                }
            )
    summary = study.summarise_sweep(cases)
    reduction = study.suppression_reduction(summary, 64)
    assert reduction["curvature_reduction_lid"] == pytest.approx(1.0 - 0.4 / 3.0, rel=1e-9)
    assert reduction["curvature_reduction_chief"] == pytest.approx(1.0 - 0.6 / 3.0, rel=1e-9)