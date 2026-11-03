"""Verification tests for the hydrostatic module against closed-form solutions.

Two verdicts are asserted here:

* **barge box** - the discrete hydrostatics are exact to round-off, so any
  deviation is an implementation error (guards against future regressions in
  the volume / waterplane integration paths);
* **half-ellipsoid** - the faceting error converges at the observed second
  order, and at the finest tested mesh every key quantity is inside 1 %.

The closed-form values used as references are derived in
:mod:`tensorbem.benchmarks.hydrostatic_benchmarks`.
"""

import math

import numpy as np
import pytest

from tensorbem.benchmarks.hydrostatic_benchmarks import (
    box_reference,
    convergence_study,
    measure_hull,
    spheroid_reference,
)
from tensorbem.geometry3d import box_hull_mesh, hemisphere_hull_mesh, spheroid_hull_mesh
from tensorbem.hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_hydrostatic_restoring_matrix,
    compute_waterplane_properties,
)

DENSITY = 1025.0
GRAVITY = 9.80665


class TestBargeExactness:
    """Flat-faceted barge: hydrostatics must be exact to round-off."""

    @pytest.mark.parametrize(
        "n",
        [1, 2, 4, 8],
    )
    def test_barge_hydrostatics_are_exact(self, n):
        length, beam, draft = 10.0, 4.0, 2.0
        ref = box_reference(length, beam, draft)
        hull = box_hull_mesh(
            length, beam, draft,
            n_length=n, n_beam=max(1, n // 2), n_draft=max(1, n // 2),
        )
        measured = measure_hull(hull, -draft, ref)
        assert measured["displaced_volume"] == pytest.approx(
            ref.displaced_volume, rel=1e-12
        )
        assert measured["waterplane_area"] == pytest.approx(
            ref.waterplane_area, rel=1e-12
        )
        assert measured["i_xx"] == pytest.approx(ref.i_xx, rel=1e-12)
        assert measured["i_yy"] == pytest.approx(ref.i_yy, rel=1e-12)
        assert measured["kb"] == pytest.approx(ref.kb, rel=1e-12)
        assert measured["bm_t"] == pytest.approx(ref.bm_t, rel=1e-12)
        assert measured["bm_l"] == pytest.approx(ref.bm_l, rel=1e-12)
        assert measured["block_coefficient"] == pytest.approx(1.0, rel=1e-12)

    def test_barge_heave_restoring_is_rho_g_awp(self):
        length, beam, draft = 10.0, 4.0, 2.0
        hull = box_hull_mesh(length, beam, draft)
        area, _c, _m = compute_waterplane_properties(hull)
        props = compute_hydrostatic_restoring_matrix(
            hull,
            mass=DENSITY * length * beam * draft,
            center_of_gravity=np.array([0.0, 0.0, -1.0]),
            density=DENSITY,
            gravity=GRAVITY,
        )
        assert props.restoring_matrix[2, 2] == pytest.approx(
            DENSITY * GRAVITY * area, rel=1e-12
        )

    def test_barge_restoring_matrix_structure(self):
        """Only heave/roll/pitch restore; the matrix is symmetric and uncoupled."""
        length, beam, draft = 10.0, 4.0, 2.0
        hull = box_hull_mesh(length, beam, draft)
        volume, _cb = compute_displaced_volume_and_buoyancy(hull)
        kg = 1.0  # above keel
        cg = np.array([0.0, 0.0, -draft + kg])
        props = compute_hydrostatic_restoring_matrix(
            hull, mass=DENSITY * volume, center_of_gravity=cg,
            density=DENSITY, gravity=GRAVITY,
        )
        c = props.restoring_matrix
        assert np.allclose(c, c.T, atol=1e-12)
        # surge / sway / yaw have no hydrostatic restoring
        for i in (0, 1, 5):
            assert np.allclose(c[i, :], 0.0, atol=1e-12)
            assert np.allclose(c[:, i], 0.0, atol=1e-12)

        kb = 0.5 * draft
        bm_t = beam**2 / (12.0 * draft)
        bm_l = length**2 / (12.0 * draft)
        assert c[3, 3] == pytest.approx(
            DENSITY * GRAVITY * volume * (kb + bm_t - kg), rel=1e-12
        )
        assert c[4, 4] == pytest.approx(
            DENSITY * GRAVITY * volume * (kb + bm_l - kg), rel=1e-12
        )

    def test_barge_block_coefficient_is_one(self):
        ref = box_reference(10.0, 4.0, 2.0)
        assert ref.block_coefficient == 1.0
        assert ref.displaced_volume == pytest.approx(80.0)


class TestSpheroidConvergence:
    """Curved waterplane: verify second-order convergence and the 3 % gate."""

    @pytest.mark.parametrize(
        "quantity",
        ["displaced_volume", "waterplane_area", "i_xx", "kb", "bm_t"],
    )
    def test_second_order_convergence(self, quantity):
        study = convergence_study(
            "spheroid", (2, 3, 4), length=10.0, beam=4.0, draft=2.0
        )
        order = study["observed_order"][quantity]
        assert 1.7 < order < 2.3, f"{quantity} observed order {order:.3f}"

    def test_fine_mesh_errors_below_one_percent(self):
        study = convergence_study(
            "spheroid", (4,), length=10.0, beam=4.0, draft=2.0
        )
        errors = study["rows"][0]["relative_error"]
        for quantity in (
            "displaced_volume",
            "waterplane_area",
            "i_xx",
            "i_yy",
            "kb",
            "bm_t",
            "bm_l",
        ):
            assert errors[quantity] < 1.0e-2, f"{quantity}: {errors[quantity]:.3e}"

    def test_spheroid_block_coefficient_recovers_pi_over_six(self):
        """Cb = V / ((2a)(2b)c) = pi/6 for a half-ellipsoid."""
        ref = spheroid_reference(5.0, 2.0, 2.0)
        assert ref.block_coefficient == pytest.approx(math.pi / 6.0)
        measured = measure_hull(
            spheroid_hull_mesh((5.0, 2.0, 2.0), subdivisions=3), -2.0, ref
        )
        assert measured["block_coefficient"] == pytest.approx(math.pi / 6.0, rel=3e-2)

    def test_error_decreases_monotonically_with_refinement(self):
        study = convergence_study(
            "spheroid", (1, 2, 3, 4), length=10.0, beam=4.0, draft=2.0
        )
        for quantity in ("displaced_volume", "waterplane_area", "i_xx"):
            seq = [row["relative_error"][quantity] for row in study["rows"]]
            assert all(a > b for a, b in zip(seq, seq[1:])), (quantity, seq)


class TestClassicalMetacentreResults:
    """Classical hemisphere results: KB = 5R/8, BM = 3R/8, KM = R."""

    def test_unit_hemisphere_closed_forms(self):
        ref = spheroid_reference(1.0, 1.0, 1.0)
        assert ref.displaced_volume == pytest.approx(2.0 * math.pi / 3.0)
        assert ref.waterplane_area == pytest.approx(math.pi)
        assert ref.kb == pytest.approx(0.625)
        assert ref.bm_t == pytest.approx(0.375)
        assert ref.kb + ref.bm_t == pytest.approx(1.0)  # KM = R

    def test_hemisphere_metacentre_converges_to_radius(self):
        ref = spheroid_reference(1.0, 1.0, 1.0)
        for subdivisions, tolerance in ((2, 3.0e-2), (4, 5.0e-3)):
            measured = measure_hull(
                hemisphere_hull_mesh(subdivisions), -1.0, ref
            )
            km = measured["kb"] + measured["bm_t"]
            assert km == pytest.approx(1.0, abs=tolerance)
            assert measured["kb"] == pytest.approx(0.625, abs=tolerance)

    def test_hemisphere_coarse_mesh_facet_error_is_known(self):
        """The 64-panel geodesic hemisphere carries a ~8.9 % volume deficit.

        This pins the number quoted in the m273 development report
        ("displaced volume 8.8 %") as a discretisation artefact rather than an
        implementation error.
        """
        ref = spheroid_reference(1.0, 1.0, 1.0)
        measured = measure_hull(hemisphere_hull_mesh(2), -1.0, ref)
        assert measured["displaced_volume"] == pytest.approx(
            ref.displaced_volume * 0.91142, rel=2.0e-3
        )


class TestGeometryGeneratorGuards:
    """Invalid hull parameters must be rejected, not silently accepted."""

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_box_rejects_non_positive_dimensions(self, bad):
        with pytest.raises(ValueError):
            box_hull_mesh(10.0, bad, 2.0)
        with pytest.raises(ValueError):
            box_reference(10.0, 4.0, bad)

    @pytest.mark.parametrize("bad", [(1.0, 0.0, 1.0), (1.0, 1.0, -2.0)])
    def test_spheroid_rejects_bad_semi_axes(self, bad):
        with pytest.raises(ValueError):
            spheroid_hull_mesh(bad)
        with pytest.raises(ValueError):
            spheroid_reference(*bad)

    def test_box_rejects_bad_panel_counts(self):
        with pytest.raises(ValueError):
            box_hull_mesh(10.0, 4.0, 2.0, n_length=0)

    def test_box_mesh_is_a_valid_wetted_hull(self):
        hull = box_hull_mesh(10.0, 4.0, 2.0, n_length=3, n_beam=2, n_draft=2)
        assert hull.panel_count > 0
        assert np.all(hull.vertices[:, 2] <= 0.0)
        # the open boundary is the waterline only
        assert len(hull.waterline_edges) > 0