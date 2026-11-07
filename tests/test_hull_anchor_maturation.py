"""Acceptance tests for the potential-flow verification anchors (mission m5240).

The anchors are: (i) the closed-form hydrostatics / wetted area of the parabolic
Wigley III hull against its *meshed* panels, and (ii) the Michell/Havelock
thin-ship Kochin function of a half-submerged ellipsoid in three independent
evaluations (Bessel closed form, definition quadrature, centreplane panels).

Nothing in this file compares the solver against numbers produced by the same
routine it audits: every reference is either a closed form, an independent
quadrature, or a different formula for the same physical quantity.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.benchmarks.hull_anchor_maturation import (
    BRIEF_HULL,
    ELLIPSOID,
    PANEL_LEVELS,
    ellipsoid_wave_study,
    wigley_geometry_audit,
    wigley_hydrostatic_panel_study,
    wigley_kochin_panels,
    wigley_panel_wave_resistance,
)
from tensorbem.ellipsoid_kochin import (
    EllipsoidHull,
    kochin_analytic,
    kochin_quadrature,
    kelvin_wave_resistance,
    kelvin_wave_resistance_theta,
    wave_resistance_analytic,
    wave_resistance_panels,
    wetted_surface_area,
    wetted_volume_quadrature,
)
from tensorbem.wigley import michell_kochin_function, michell_wave_resistance

G = 9.80665


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def hull() -> EllipsoidHull:
    return EllipsoidHull(ELLIPSOID["a"], ELLIPSOID["b"], ELLIPSOID["c"])


@pytest.fixture(scope="module")
def geometry_audit() -> dict:
    return wigley_geometry_audit()


@pytest.fixture(scope="module")
def hydro_panel_study() -> dict:
    # coarse-but-representative ladder (the 3200-panel level lives in the
    # benchmark artifact; it is too slow for the unit-test budget)
    return wigley_hydrostatic_panel_study(
        levels=PANEL_LEVELS[:2],
        L=BRIEF_HULL["L"],
        B=BRIEF_HULL["B"],
        T=BRIEF_HULL["T"],
        area_quads=(120, 240),
    )


# ============================================================================
# Ellipsoid geometry gate
# ============================================================================


class TestEllipsoidGeometry:
    """Closed-form geometry must be verified before any resistance is quoted."""

    def test_principal_dimensions(self, hull: EllipsoidHull) -> None:
        assert hull.L == pytest.approx(2.0 * hull.a)
        assert hull.B == pytest.approx(2.0 * hull.b)
        assert hull.T == pytest.approx(hull.c)

    def test_coefficients_match_closed_forms(self, hull: EllipsoidHull) -> None:
        assert hull.volume == pytest.approx(2.0 * math.pi * hull.a * hull.b * hull.c / 3.0)
        assert hull.Cb == pytest.approx(math.pi / 6.0, rel=1e-14)
        assert hull.Cm == pytest.approx(math.pi / 4.0, rel=1e-14)
        assert hull.Cp == pytest.approx(2.0 / 3.0, rel=1e-14)
        assert abs(hull.Cb - 0.5236) < 1e-4  # the mesh-audit value quoted in docs
        assert hull.Cb != pytest.approx(0.5236 * 0.0)

    def test_volume_quadrature_matches_closed_form(self, hull: EllipsoidHull) -> None:
        assert wetted_volume_quadrature(hull) == pytest.approx(hull.volume, rel=1e-12)

    def test_wetted_area_is_stable_under_refinement(self, hull: EllipsoidHull) -> None:
        coarse = wetted_surface_area(hull, n_psi=48, n_phi=96)
        fine = wetted_surface_area(hull, n_psi=96, n_phi=192)
        assert coarse == pytest.approx(fine, rel=1e-6)
        # a wetted half-ellipsoid must be smaller than the full-ellipsoid bound
        assert fine < 4.0 * math.pi * hull.a * hull.b

    def test_half_breadth_and_gradient(self, hull: EllipsoidHull) -> None:
        assert hull.half_breadth(0.0, 0.0) == pytest.approx(hull.b)
        assert hull.half_breadth(hull.a, 0.0) == pytest.approx(0.0, abs=1e-15)
        assert hull.half_breadth(2.0 * hull.a, 0.0) == pytest.approx(0.0)
        # dy/dx = -b x / (a^2 sqrt(...)); check against a central difference
        x, z = 0.3 * hull.a, -0.4 * hull.c
        h = 1e-6
        fd = (hull.half_breadth(x + h, z) - hull.half_breadth(x - h, z)) / (2.0 * h)
        assert hull.half_breadth_x_gradient(x, z) == pytest.approx(fd, rel=1e-6)

    def test_rejects_non_thin_body(self) -> None:
        with pytest.raises(ValueError):
            EllipsoidHull(1.0, 1.5, 0.15)


# ============================================================================
# Ellipsoid Kochin function: three independent evaluations
# ============================================================================


class TestEllipsoidKochin:
    @pytest.mark.parametrize("k", [0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    def test_closed_form_matches_definition(self, hull: EllipsoidHull, k: float) -> None:
        closed = kochin_analytic(k, hull)
        definition = kochin_quadrature(k, hull)
        assert abs(closed - definition) / abs(closed) < 1e-10

    def test_low_wavenumber_limit(self, hull: EllipsoidHull) -> None:
        """``H(k) -> -2 i k`` times the one-side centreplane integral of ``y``.

        ``int int y dx dz = pi a b c / 3`` for the wetted half-ellipsoid, so the
        ``k -> 0`` slope is a closed-form check of the integrand normalisation
        (not just of the quadrature).
        """
        k = 1.0e-6
        ratio = kochin_analytic(k, hull) / (-2.0j * k * math.pi * hull.a * hull.b * hull.c / 3.0)
        # the leading correction is O(k), measured 5.6e-6 relative at k = 1e-4
        assert ratio.real == pytest.approx(1.0, abs=1e-6)
        assert ratio.imag == pytest.approx(0.0, abs=1e-6)

    def test_fore_aft_symmetry_is_pure_imaginary(self, hull: EllipsoidHull) -> None:
        """A symmetric body has an odd ``dy/dx``: the real part must vanish."""
        for k in (0.7, 3.0, 11.0):
            value = kochin_analytic(k, hull)
            assert abs(value.real) <= 1e-12 * abs(value)

    def test_rejects_negative_wavenumber(self, hull: EllipsoidHull) -> None:
        with pytest.raises(ValueError):
            kochin_analytic(-1.0, hull)


# ============================================================================
# Kelvin-wake integral and ellipsoid wave resistance
# ============================================================================


class TestEllipsoidWaveResistance:
    def test_kelvin_integral_reproduces_wigley_reference(self) -> None:
        """Convention check: same integrator + Wigley Kochin = Wigley ``R_W``."""
        L, B, T = 3.8, BRIEF_HULL["B"], BRIEF_HULL["T"]
        u = 1.0
        k0 = G / (u * u)
        mine = kelvin_wave_resistance(
            lambda k: michell_kochin_function(L, B, T, k),
            k0=k0,
            lambda_max=max(5.0, 10.0 / (k0 * T)),
        )
        reference = michell_wave_resistance(L, B, T, u)
        assert mine == pytest.approx(reference, rel=1e-12)

    def test_lambda_and_theta_forms_agree(self, hull: EllipsoidHull) -> None:
        froude = 0.30
        u = froude * math.sqrt(G * hull.L)
        k0 = G / (u * u)
        lam = wave_resistance_analytic(hull, froude)
        theta = kelvin_wave_resistance_theta(lambda k: kochin_analytic(k, hull), k0=k0)
        assert theta == pytest.approx(lam, rel=1e-4)

    def test_panel_route_converges_to_closed_form(self, hull: EllipsoidHull) -> None:
        froude = ELLIPSOID["Fr"]
        reference = wave_resistance_analytic(hull, froude)
        errors = {}
        for n_r, n_phi in ((10, 20), (20, 40), (40, 80)):
            value = wave_resistance_panels(hull, froude, n_r=n_r, n_phi=n_phi)
            errors[(n_r, n_phi)] = abs(value / reference - 1.0)
        assert errors[(10, 20)] < 0.02  # 200 panels, < 2 %
        assert errors[(20, 40)] < 0.01  # 800 panels, < 1 %
        assert errors[(40, 80)] < 0.005  # 3200 panels, < 0.5 %
        assert errors[(40, 80)] < errors[(20, 40)] < errors[(10, 20)]

    def test_naive_source_sampling_is_worse_than_integrated(self, hull: EllipsoidHull) -> None:
        """Rim-singularity finding: sampling ``dy/dx`` at panel centroids stalls."""
        froude = ELLIPSOID["Fr"]
        reference = wave_resistance_analytic(hull, froude)
        integrated = wave_resistance_panels(
            hull, froude, n_r=20, n_phi=40, scheme="source_integrated"
        )
        sampled = wave_resistance_panels(hull, froude, n_r=20, n_phi=40, scheme="source")
        assert abs(integrated / reference - 1.0) < abs(sampled / reference - 1.0)

    def test_wave_resistance_is_positive_and_thin_body_scales_with_b_squared(
        self,
    ) -> None:
        """``R_W`` of a thin body scales like ``b^2`` to leading order."""
        froude = 0.30
        rw_small = wave_resistance_analytic(EllipsoidHull(1.0, 0.05, 0.15), froude)
        rw_base = wave_resistance_analytic(EllipsoidHull(1.0, 0.10, 0.15), froude)
        assert rw_small > 0.0 and rw_base > 0.0
        assert rw_base / rw_small == pytest.approx(4.0, rel=0.05)

    def test_study_result_shape(self) -> None:
        study = ellipsoid_wave_study()
        assert study["geometry"]["Cb"] == pytest.approx(math.pi / 6.0, rel=1e-14)
        assert study["kochin"]["max_rel_difference"] < 1e-10
        assert study["wave_resistance"]["analytic_closed_form"] > 0.0
        schemes = {row["scheme"] for row in study["panels"]}
        assert schemes == {"source_integrated", "ordinate", "source"}


# ============================================================================
# Wigley III geometry gate and hydrostatics vs panel count
# ============================================================================


class TestWigleyGeometryAudit:
    """The brief's ``Cb = 0.667`` is audited, not inherited."""

    def test_mesh_reproduces_closed_form(self, geometry_audit: dict) -> None:
        assert geometry_audit["closed_form"]["Cb"] == pytest.approx(4.0 / 9.0, rel=1e-14)
        assert geometry_audit["mesh"]["volume"] == pytest.approx(
            geometry_audit["closed_form"]["volume"], rel=1e-3
        )
        assert geometry_audit["mesh"]["Cb"] == pytest.approx(4.0 / 9.0, rel=1e-3)

    def test_brief_claim_is_the_prismatic_coefficient(self, geometry_audit: dict) -> None:
        assert geometry_audit["verdict"]["claim_consistent_with_parabolic_hull"] is False
        assert geometry_audit["verdict"]["claimed_value_matches"].startswith("Cp")
        # the claimed volume is exactly 1.5x the true one: Cb_claim / Cb = 2/3 / (4/9)
        assert geometry_audit["brief_claim"]["volume_ratio_to_closed_form"] == pytest.approx(
            1.5, rel=2e-3
        )


class TestWigleyHydrostaticsPanels:

    def test_panel_counts_are_the_mission_ladder(self) -> None:
        assert [int(n_s * n_l * 4) for n_s, n_l in PANEL_LEVELS] == [200, 800, 3200]

    def test_volume_and_area_converge(self, hydro_panel_study: dict) -> None:
        rows = hydro_panel_study["rows"]
        assert rows[0]["volume_rel_error"] < 0.0  # inscribed mesh: under-estimates
        assert abs(rows[-1]["volume_rel_error"]) < 1.0e-2
        assert abs(rows[-1]["volume_rel_error"]) < abs(rows[0]["volume_rel_error"])
        assert max(abs(r["wetted_area_rel_error"]) for r in rows) < 2.0e-2

    def test_wetted_area_reference_is_converged(self, hydro_panel_study: dict) -> None:
        assert hydro_panel_study["reference"]["area_self_error"] < 1e-5

    def test_Cb_from_mesh_approaches_four_ninths(self, hydro_panel_study: dict) -> None:
        assert hydro_panel_study["rows"][-1]["Cb"] == pytest.approx(4.0 / 9.0, rel=1e-2)


# ============================================================================
# Wigley wave resistance: panel decomposition identity + convergence
# ============================================================================


class TestWigleyPanelKochin:
    L, B, T = BRIEF_HULL["L"], BRIEF_HULL["B"], BRIEF_HULL["T"]

    @pytest.mark.parametrize("k", [1.0, 5.0, 20.0])
    def test_moment_scheme_is_the_closed_form(self, k: float) -> None:
        """Exact per-panel moments must reproduce the analytical Kochin function."""
        reference = michell_kochin_function(self.L, self.B, self.T, k)
        for n_x, n_z in ((10, 5), (40, 20)):
            value = wigley_kochin_panels(
                k, self.L, self.B, self.T, n_x, n_z, scheme="moment"
            )
            assert abs(value - reference) / abs(reference) < 1e-12

    def test_constant_strength_panels_converge(self) -> None:
        froude = 0.30
        u = froude * math.sqrt(G * self.L)
        reference = michell_wave_resistance(self.L, self.B, self.T, u)
        coarse = wigley_panel_wave_resistance(
            self.L, self.B, self.T, froude, n_x=10, n_z=5, scheme="mean_source"
        )
        fine = wigley_panel_wave_resistance(
            self.L, self.B, self.T, froude, n_x=40, n_z=20, scheme="mean_source"
        )
        assert abs(fine / reference - 1.0) < 0.01
        assert abs(fine / reference - 1.0) < abs(coarse / reference - 1.0)

    def test_rejects_non_positive_wavenumber(self) -> None:
        with pytest.raises(ValueError):
            wigley_kochin_panels(0.0, self.L, self.B, self.T, 10, 5)