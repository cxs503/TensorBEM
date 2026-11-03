"""Closed-form verification of the hydrostatic module (see the paired benchmark).

These tests replace the loose 5 %/10 % hemisphere acceptance checks with:

* exactness checks on wall-sided hulls, where divergence-theorem integration is
  mathematically exact, so any deviation is an implementation bug;
* a measured order of convergence on curved hulls, which separates
  discretization error from implementation error;
* a Richardson-extrapolated value that must reproduce the closed form;
* closed-form checks of the restoring matrix diagonal (``rho g A_wp`` and
  ``rho g volume GM``), which exercise the waterplane/volume/KB/BM chain end to
  end.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.benchmarks.hydrostatic_convergence import (
    RHO,
    barge_exactness,
    elliptic_cylinder_reference,
    elliptic_study,
    hemisphere_reference,
    hemisphere_restoring_matrix,
    hemisphere_study,
    rectangular_barge_reference,
)
from tensorbem.geometry3d import (
    elliptic_cylinder_hull_mesh,
    hemisphere_hull_mesh,
    rectangular_barge_hull_mesh,
)
from tensorbem.hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_hydrostatic_restoring_matrix,
    compute_waterplane_properties,
)


@pytest.fixture(scope="module")
def barge_report():
    return barge_exactness()


@pytest.fixture(scope="module")
def hemisphere_report():
    return hemisphere_study()


@pytest.fixture(scope="module")
def elliptic_report():
    return elliptic_study()


class TestWallSidedExactness:
    """A wall-sided barge is represented exactly, so results are exact."""

    def test_hull_mesh_is_accepted_with_expected_panel_count(self):
        hull = rectangular_barge_hull_mesh(10.0, 4.0, 1.0, nx=3, ny=2)
        # 2 bottom triangles per cell plus 2 wall triangles per boundary edge.
        boundary_edges = 2 * (3 + 2)
        assert len(hull.faces) == 2 * 3 * 2 + 2 * boundary_edges
        assert hull.waterline_edges.shape[0] == boundary_edges
        assert np.all(hull.vertices[:, 2] <= 0.0)

    @pytest.mark.parametrize("nx,ny", [(1, 1), (3, 2), (5, 3)])
    def test_volume_area_and_moments_are_exact(self, nx, ny):
        hull = rectangular_barge_hull_mesh(10.0, 4.0, 1.0, nx=nx, ny=ny)
        ref = rectangular_barge_reference(10.0, 4.0, 1.0)
        volume, cb = compute_displaced_volume_and_buoyancy(hull)
        area, center, moments = compute_waterplane_properties(hull)
        assert abs(volume - ref.displaced_volume) / ref.displaced_volume < 1e-13
        assert abs(area - ref.waterplane_area) / ref.waterplane_area < 1e-13
        assert abs(moments[0] - ref.i_transverse) / ref.i_transverse < 1e-13
        assert abs(moments[1] - ref.i_longitudinal) / ref.i_longitudinal < 1e-13
        assert abs(moments[2]) < 1e-12
        assert cb[2] == pytest.approx(-0.5, abs=1e-14)
        assert center[0] == pytest.approx(0.0, abs=1e-14)
        assert center[1] == pytest.approx(0.0, abs=1e-14)

    def test_restoring_matrix_matches_closed_form(self, barge_report):
        assert barge_report["max_error"] < 1e-12
        row = barge_report["rows"][-1]
        assert row["c_heave"] > 0.0
        assert row["c_roll"] > 0.0
        assert row["c_pitch"] > 0.0

    def test_restoring_matrix_is_diagonal_in_heave_roll_pitch(self):
        hull = rectangular_barge_hull_mesh(10.0, 4.0, 1.0)
        ref = rectangular_barge_reference(10.0, 4.0, 1.0)
        props = compute_hydrostatic_restoring_matrix(
            hull,
            mass=RHO * ref.displaced_volume,
            center_of_gravity=np.array([0.0, 0.0, -0.5]),
        )
        mask = np.ones((6, 6), dtype=bool)
        for index in (2, 3, 4):
            mask[index, index] = False
        assert np.max(np.abs(props.restoring_matrix[mask])) == 0.0
        assert props.restoring_matrix[2, 2] == pytest.approx(
            RHO * 9.80665 * ref.waterplane_area, rel=1e-12
        )


class TestHemisphereConvergence:
    """Curved hull: second-order convergence to the closed form."""

    def test_observed_order_is_second_order(self, hemisphere_report):
        for row in hemisphere_report["levels"][2:]:
            for name in ("displaced_volume_order", "waterplane_area_order"):
                order = row.get(name)
                assert order is not None
                assert 1.85 < order < 2.10, (name, order)

    def test_error_at_moderate_mesh_beats_platform_bar(self, hemisphere_report):
        levels = {row["subdivisions"]: row for row in hemisphere_report["levels"]}
        row = levels[4]
        assert row["displaced_volume_error"] < 0.01
        assert row["waterplane_area_error"] < 0.005
        assert row["i_transverse_error"] < 0.01
        # Platform-wide acceptance bar: 3 % relative error.
        assert max(
            row["displaced_volume_error"],
            row["waterplane_area_error"],
            row["i_transverse_error"],
            row["kb_error"],
        ) < 0.03

    def test_richardson_extrapolation_recovers_closed_form(self, hemisphere_report):
        richardson = hemisphere_report["richardson"]
        assert richardson["displaced_volume_error"] < 1e-9
        assert richardson["waterplane_area_error"] < 1e-9
        assert richardson["i_transverse_error"] < 1e-9

    def test_monotone_convergence(self, hemisphere_report):
        errors = [row["displaced_volume_error"] for row in hemisphere_report["levels"]]
        assert all(later < earlier for earlier, later in zip(errors, errors[1:]))


class TestRestoringMatrixClosedForm:
    """The full chain waterplane -> volume -> KB/BM -> C is checked end to end."""

    def test_hemisphere_gm_and_stiffness(self):
        report = hemisphere_restoring_matrix(subdivisions=4, radius=1.0, kg=0.6)
        ref = hemisphere_reference(1.0)
        assert report["computed"]["gm"] == pytest.approx(ref.kb + ref.bm_transverse - 0.6, rel=5e-3)
        for key in ("volume", "kb", "bm", "gm", "c_heave", "c_roll"):
            assert report["error"][key] < 0.01, (key, report["error"][key])
        assert report["computed"]["c_roll"] == pytest.approx(report["computed"]["c_pitch"], rel=1e-12)

    def test_roll_restoring_scales_with_gm(self):
        """Doubling KG above the keel must reduce ``C[3,3]`` by rho g volume* dKG."""
        radius = 1.0
        ref = hemisphere_reference(radius)
        hull = hemisphere_hull_mesh(subdivisions=3, radius=radius)
        z_min = float(np.min(hull.vertices[:, 2]))
        base = compute_hydrostatic_restoring_matrix(
            hull, RHO * ref.displaced_volume, np.array([0.0, 0.0, 0.4 + z_min])
        )
        raised = compute_hydrostatic_restoring_matrix(
            hull, RHO * ref.displaced_volume, np.array([0.0, 0.0, 0.6 + z_min])
        )
        delta = base.restoring_matrix[3, 3] - raised.restoring_matrix[3, 3]
        expected = RHO * 9.80665 * base.displaced_volume * 0.2
        assert delta == pytest.approx(expected, rel=1e-12)

    def test_unstable_body_yields_negative_roll_restoring(self):
        """KG above the metacentre flips the sign of ``C[3,3]`` (same convention)."""
        radius = 1.0
        ref = hemisphere_reference(radius)
        hull = hemisphere_hull_mesh(subdivisions=2, radius=radius)
        z_min = float(np.min(hull.vertices[:, 2]))
        props = compute_hydrostatic_restoring_matrix(
            hull, RHO * ref.displaced_volume, np.array([0.0, 0.0, 1.05 + z_min])
        )
        assert props.restoring_matrix[3, 3] < 0.0


class TestEllipticWaterplane:
    """Polygonal waterplane: exact for the meshed polygon, O(n^-2) to the ellipse."""

    def test_polygon_values_are_exact(self, elliptic_report):
        for row in elliptic_report["levels"]:
            assert row["polygon_area_exact_error"] < 1e-13
            assert row["i_transverse_error_vs_polygon"] < 1e-12

    def test_order_against_ellipse_is_two(self, elliptic_report):
        for row in elliptic_report["levels"][1:]:
            order = row["waterplane_area_order_vs_ellipse"]
            assert 1.9 < order < 2.1, order

    def test_reference_matches_direct_polygon_formula(self):
        panels, a, b, draft = 32, 5.0, 2.0, 1.0
        ref = elliptic_cylinder_reference(a, b, draft, panels)
        hull = elliptic_cylinder_hull_mesh(a, b, draft, panels=panels)
        volume, _ = compute_displaced_volume_and_buoyancy(hull)
        area, _, _ = compute_waterplane_properties(hull)
        assert area == pytest.approx(ref.waterplane_area, rel=1e-13)
        assert volume == pytest.approx(ref.displaced_volume, rel=1e-13)
        # The ellipse bounds the inscribed polygon from above.
        assert ref.waterplane_area < math.pi * a * b


class TestMeshGeneratorValidation:
    """Guard rails on the new hull generators."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"length": 0.0, "beam": 4.0, "draft": 1.0},
            {"length": 10.0, "beam": -1.0, "draft": 1.0},
            {"length": 10.0, "beam": 4.0, "draft": math.inf},
            {"length": 10.0, "beam": 4.0, "draft": 1.0, "nx": 0},
            {"length": 10.0, "beam": 4.0, "draft": 1.0, "nx": 1.5},
        ],
    )
    def test_barge_rejects_invalid_input(self, kwargs):
        with pytest.raises(ValueError):
            rectangular_barge_hull_mesh(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"semi_axis_x": 0.0, "semi_axis_y": 2.0, "draft": 1.0},
            {"semi_axis_x": 5.0, "semi_axis_y": 2.0, "draft": 0.0},
            {"semi_axis_x": 5.0, "semi_axis_y": 2.0, "draft": 1.0, "panels": 6},
        ],
    )
    def test_elliptic_cylinder_rejects_invalid_input(self, kwargs):
        with pytest.raises(ValueError):
            elliptic_cylinder_hull_mesh(**kwargs)