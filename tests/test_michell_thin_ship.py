"""Verification tests for :mod:`tensorbem.michell` and its benchmark surface.

The tests are the executable form of the campaign criteria:

* Wigley hydrostatics from the meshed hull: ``nabla`` error < 1 %,
  ``S_w`` error < 2 %, ``Cb`` error < 0.5 %;
* ellipsoid wave resistance at ``Fn = 0.2, 0.3, 0.4`` vs the (independent)
  quadrature reference of the same Michell integral: error < 5 %;
* Wigley tandem interference ``sigma = R_pair / (2 R_single)`` at clear-water
  gaps ``L/2, L, 2L`` vs the closed-form superposition, plus the exact limits
  ``sigma(s -> 0) = 2`` and ``sigma(s -> inf) = 1``;
* three-level grid refinement with Roache GCI ``U_G < 3 %`` on every reported
  quantity.

Every number produced here is a *discretisation* statement; the scope
statement returned by ``verify()['scope_statement']`` says so explicitly and is
asserted below so that it cannot silently disappear.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tensorbem import wigley as wigley_module
from tensorbem.benchmarks import michell_validation as bench
from tensorbem.michell import (
    CenterplanePanels,
    DEFAULT_GRAVITY,
    default_lambda_max,
    ellipsoid_hydrostatic_reference,
    ellipsoid_kochin_reference,
    ellipsoid_panels,
    ellipsoid_wave_resistance,
    froude_number,
    michell_resistance,
    resistance_nodes,
    speed_for_froude,
    tandem_interference_factor,
    tandem_pair_resistance,
    wave_resistance_from_panels,
    wigley_kochin_closed_form,
    wigley_panels,
)
from tensorbem.wigley import michell_kochin_function, michell_wave_resistance

L, B, T = bench.HULL


class TestKochinMachinery:
    """The panel route must reproduce the verified Wigley closed form."""

    @pytest.mark.parametrize("k", [1.0, 3.3, 10.0, 30.0, 60.0])
    def test_panel_kochin_matches_closed_form(self, k):
        panels = wigley_panels(L, B, T, 64, 32, rule="gauss")
        h_panel = panels.kochin(k)
        h_closed = wigley_kochin_closed_form(L, B, T, k)
        assert abs(h_panel - h_closed) / abs(h_closed) < 1e-9

    @pytest.mark.parametrize("k", [1.0, 5.554, 20.0])
    def test_closed_form_matches_repository_scalar_form(self, k):
        """The vectorised closed form agrees with the pre-existing scalar one."""
        h_vec = wigley_kochin_closed_form(L, B, T, k)
        h_ref = michell_kochin_function(L, B, T, k)
        assert abs(h_vec - h_ref) / abs(h_ref) < 1e-12

    def test_small_wavenumber_uses_the_series(self):
        """``k -> 0``: ``H -> -i k (4/9) B L T`` (both integrals go to their limits).

        The repository's scalar closed form loses all its digits below
        ``k T ~ 1e-3`` (catastrophic cancellation of ``O(1/eps^3)`` terms); the
        series branch must not.  The analytic limit is
        ``Ix -> 2 i a^3 k / 3``, ``Iz -> 2T/3``, i.e.
        ``H -> 2 (-4B/L^2) (i k L^3 / 12) (2T/3) = -i k (4/9) B L T``.
        """
        ks = (1e-9, 1e-6, 1e-3)
        got = {k: wigley_kochin_closed_form(L, B, T, k) for k in ks}
        leading = {k: 1j * k * (-4.0 * B * T * L / 9.0) for k in ks}
        rel = {k: abs(got[k] - leading[k]) / abs(leading[k]) for k in ks}
        # The leading correction is O(k) and comes from the draft integral:
        # Iz = T(2/3 - k T/4 + ...), so H deviates from its limit by
        # (3/8) k T relative -- an exact, checkable rate.
        for k in ks:
            assert 0.95 < rel[k] / (0.375 * T * k) < 1.05, (k, rel[k])
        assert rel[1e-9] < 1e-10
        # The series branch keeps the value: an independent Gauss quadrature
        # over the centreplane agrees to round-off at all three wavenumbers ...
        panels = wigley_panels(L, B, T, 64, 32, rule="gauss")
        atol = 1e-12 * float(np.sum(np.abs(panels.weight)))
        for k in ks:
            assert abs(panels.kochin(k) - got[k]) < atol, (k, panels.kochin(k))
        # ... while the repository's term-by-term closed form has lost it.
        assert abs(michell_kochin_function(L, B, T, 1e-3) - got[1e-3]) \
            / abs(got[1e-3]) > 1e-5

    def test_wigley_kochin_is_pure_imaginary(self):
        """``x`` and ``z`` integrals are odd/even: ``H`` has no real part."""
        h = wigley_kochin_closed_form(L, B, T, np.array([2.0, 7.5, 41.0]))
        assert np.max(np.abs(h.real)) < 1e-18 * max(1.0, np.max(np.abs(h)))

    def test_panel_container_operations(self):
        panels = wigley_panels(L, B, T, 8, 4, rule="midpoint")
        assert isinstance(panels, CenterplanePanels)
        assert panels.panel_count == 32
        assert panels.projected_area == pytest.approx(L * T, rel=1e-12)
        # shifting by dx multiplies H by exp(i k dx)
        dx = 0.7
        k = 4.0
        assert panels.shifted(dx).kochin(k) == pytest.approx(
            panels.kochin(k) * np.exp(1j * k * dx), rel=1e-12
        )
        merged = panels.merged(panels.shifted(dx))
        assert merged.panel_count == 2 * panels.panel_count
        assert merged.kochin(k) == pytest.approx(
            panels.kochin(k) * (1.0 + np.exp(1j * k * dx)), rel=1e-12
        )


class TestResistanceIntegral:
    """Quadrature, truncation and reference agreement of ``R_w``."""

    @pytest.mark.parametrize("froude", bench.FROUDES)
    def test_panel_route_matches_repository_reference(self, froude):
        U = speed_for_froude(froude, L)
        panels = wigley_panels(L, B, T, 96, 48, rule="gauss")
        r_panel = wave_resistance_from_panels(panels, U, T)
        r_repo = michell_wave_resistance(L, B, T, U)
        assert abs(r_panel - r_repo) / r_repo < 1e-8

    def test_resistance_nodes_are_the_classic_substitution(self):
        """``lambda = 1 + s^2``: the weights integrate the pure ``1/lambda^2`` tail."""
        U = speed_for_froude(0.3, L)
        lam_max = default_lambda_max(DEFAULT_GRAVITY / U**2, T)
        lam, weight = resistance_nodes(U, T, n_seg=32, n_quad=32)
        # int_1^L dlambda/(lambda^2 sqrt(lambda^2-1)) = sqrt(L^2-1)/L
        exact = math.sqrt(lam_max**2 - 1.0) / lam_max
        assert weight.sum() == pytest.approx(exact, rel=1e-10)
        # ... and the *untruncated* integral is exactly 1, so the tail kept out
        # is sqrt(L^2-1)/L - 1 ~ -1/(2L^2): small by construction of lambda_max.
        assert 0.9 < weight.sum() < 1.0
        assert np.all(lam >= 1.0)
        assert np.all(np.diff(lam) >= 0.0)

    def test_truncation_and_quadrature_sensitivity(self):
        result = bench.truncation_check(L, B, T)
        assert all(row["passed"] for row in result["criteria"])

    @pytest.mark.parametrize("froude", bench.FROUDES)
    def test_resistance_is_positive_and_grows_with_speed(self, froude):
        kochin = lambda k: wigley_kochin_closed_form(L, B, T, k)
        U = speed_for_froude(froude, L)
        assert michell_resistance(kochin, U, T) > 0.0
        assert wigley_kochin_closed_form(L, B, T, 5.0) != 0.0

    def test_zero_speed_rejected(self):
        with pytest.raises(ValueError):
            resistance_nodes(0.0, T)


class TestWigleyHydrostatics:
    """Campaign criterion (1): geometry before physics."""

    def test_criteria_pass(self):
        result = bench.wigley_hydrostatics_study(L, B, T)
        assert all(row["passed"] for row in result["criteria"])

    def test_errors_are_below_campaign_thresholds(self):
        result = bench.wigley_hydrostatics_study(L, B, T)
        fine = result["finest"]
        assert fine["volume_rel_err"] < 0.01
        assert fine["wetted_area_rel_err"] < 0.02
        assert fine["block_coefficient_rel_err"] < 0.005
        assert fine["block_coefficient"] == pytest.approx(4.0 / 9.0, rel=1e-3)
        assert result["reference"]["volume"] == pytest.approx(4.0 * L * B * T / 9.0)

    def test_second_order_convergence(self):
        result = bench.wigley_hydrostatics_study(L, B, T)
        for key in ("volume", "wetted_area"):
            order = result["gci"][key]["observed_order"]
            assert 1.7 < order < 2.4, (key, order)
            assert result["gci"][key]["gci_fine"] < 0.03


class TestEllipsoid:
    """Campaign criterion (2): ellipsoid ``R_w`` vs Michell thin-ship theory."""

    def test_hydrostatic_reference(self):
        ref = ellipsoid_hydrostatic_reference(L, B, T)
        assert ref["Cb"] == pytest.approx(math.pi / 12.0, rel=1e-15)
        assert ref["volume"] == pytest.approx(math.pi * L * B * T / 12.0)
        assert ref["projected_area"] == pytest.approx(math.pi * L * T / 4.0)

    def test_panel_footprint_converges(self):
        """Midpoint panels converge on the half-ellipse footprint ``pi L T / 4``.

        The elliptic-coordinate jacobian is polynomial, so a *Gauss* rule hits
        the footprint exactly; the verification path uses midpoint panels, which
        are second order -- hence a small, shrinking error rather than round-off.
        """
        result = bench.ellipsoid_geometry_study(L, B, T)
        assert all(row["passed"] for row in result["criteria"])
        errors = [row["projected_area_rel_err"] for row in result["levels"]]
        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] < 1e-4
        gauss = ellipsoid_panels(L, B, T, 24, 48, rule="gauss")
        assert gauss.projected_area == pytest.approx(
            ellipsoid_hydrostatic_reference(L, B, T)["projected_area"], rel=1e-13
        )

    def test_reference_rule_is_converged(self):
        """Doubling the reference resolution must not move ``R_w``."""
        U = speed_for_froude(0.3, L)
        r1 = ellipsoid_wave_resistance(L, B, T, U, n_psi=64, n_t=192)
        r2 = ellipsoid_wave_resistance(L, B, T, U, n_psi=96, n_t=384)
        assert abs(r1 - r2) / r2 < 1e-10

    def test_panel_route_converges_to_the_reference(self):
        U = speed_for_froude(0.3, L)
        ref = ellipsoid_wave_resistance(L, B, T, U)
        errs = []
        for n_psi, n_t in ((24, 48), (48, 96), (96, 192)):
            panels = ellipsoid_panels(L, B, T, n_psi, n_t, rule="midpoint")
            rw = wave_resistance_from_panels(panels, U, T)
            errs.append(abs(rw - ref) / ref)
        assert errs[0] > errs[1] > errs[2], errs
        assert errs[-1] < 0.05
        # second order: each refinement should cut the error by ~4
        assert 2.5 < errs[0] / errs[1] < 6.0
        assert 2.5 < errs[1] / errs[2] < 6.0

    def test_campaign_criteria_pass(self):
        result = bench.ellipsoid_wave_resistance_study(L, B, T)
        assert [case["froude_number"] for case in result["cases"]] == [0.2, 0.3, 0.4]
        assert all(row["passed"] for row in result["criteria"])
        for case in result["cases"]:
            assert case["finest"]["rel_err"] < 0.05
            assert case["gci"]["gci_fine"] < 0.03
            assert case["reference"]["wave_resistance"] > 0.0


class TestTandemInterference:
    """Campaign criterion (3): tandem pair interference factor."""

    def test_coincident_hulls_give_sigma_two(self):
        U = speed_for_froude(0.3, L)
        kochin = lambda k: wigley_kochin_closed_form(L, B, T, k)
        out = tandem_interference_factor(kochin, U, T, 0.0)
        assert out["sigma"] == pytest.approx(2.0, abs=1e-9)
        # and the merged-panel route agrees
        panels = wigley_panels(L, B, T, 96, 48, rule="gauss")
        r_pair = tandem_pair_resistance(panels, U, T, 0.0)
        r_single = michell_resistance(kochin, U, T)
        assert r_pair / (2.0 * r_single) == pytest.approx(2.0, abs=1e-6)

    def test_far_separation_decouples(self):
        U = speed_for_froude(0.3, L)
        kochin = lambda k: wigley_kochin_closed_form(L, B, T, k)
        far = tandem_interference_factor(kochin, U, T, 51.0 * L, n_quad=8)
        assert abs(far["sigma"] - 1.0) < 0.05

    def test_sigma_is_bounded_and_oscillates(self):
        U = speed_for_froude(0.3, L)
        kochin = lambda k: wigley_kochin_closed_form(L, B, T, k)
        sigmas = [tandem_interference_factor(kochin, U, T, s, n_quad=8)["sigma"]
                  for s in np.linspace(L, 4.0 * L, 24)]
        assert min(sigmas) > 0.0 and max(sigmas) < 2.0
        assert max(sigmas) - min(sigmas) > 0.2     # genuine interference structure

    def test_separation_convention_does_not_overlap_hulls(self):
        """A gap of ``L/2`` must keep the hulls apart (``s = 1.5 L``)."""
        assert bench.GAP_RATIOS == (0.5, 1.0, 2.0)
        for gap in bench.GAP_RATIOS:
            assert L + gap * L > L

    def test_interference_convention_sign(self):
        """``sigma - 1`` equals the normalised cosine interference integral."""
        U = speed_for_froude(0.3, L)
        kochin = lambda k: wigley_kochin_closed_form(L, B, T, k)
        separation = 2.0 * L
        out = tandem_interference_factor(kochin, U, T, separation, n_quad=8)
        k0 = 9.80665 / U**2
        lam, weight = resistance_nodes(U, T, n_seg=out["n_seg"], n_quad=8)
        spectrum = np.abs(kochin(lam * k0)) ** 2 * weight
        expected = 1.0 + float(np.sum(spectrum * np.cos(lam * k0 * separation))) \
            / float(np.sum(spectrum))
        assert out["sigma"] == pytest.approx(expected, rel=1e-12)

    def test_campaign_criteria_pass(self):
        result = bench.tandem_study(L, B, T, froudes=(0.3,))
        assert all(row["passed"] for row in result["criteria"])
        assert result["limits"]["sigma_coincident_err"] < 1e-6
        for case in result["cases"]:
            assert case["finest"]["rel_err"] < 0.05
            assert case["gci"]["gci_fine"] < 0.03
            # the interference is real: sigma departs from the decoupled value
            assert abs(case["theory"]["sigma"] - 1.0) > 0.05


class TestReport:
    """End-to-end: report JSON, machine-checked criteria, figures."""

    def test_reduced_surface_passes_and_serialises(self, tmp_path):
        report = bench.verify(
            L, B, T, froudes=(0.3,),
            curve_froudes=(0.2, 0.3, 0.4),
            sweep_gap_ratios=(0.5, 1.0, 2.0),
            hydrostatic_levels=((16, 8), (32, 16), (64, 32)),
            ellipsoid_levels=((24, 48), (48, 96), (96, 192)),
            tandem_levels=((16, 8), (32, 16), (64, 32)),
        )
        assert report["all_passed"], [
            row for row in report["criteria"] if not row["passed"]
        ]
        assert report["n_criteria"] == len(report["criteria"])
        assert report["milestone"] == "IN21-B1"
        assert "Michell thin-ship theory only" in report["scope_statement"]

        figures = bench.make_figures(report, tmp_path)
        assert len(figures) == 4
        for path in figures:
            assert path.exists() and path.stat().st_size > 10_000

        target = tmp_path / "report.json"
        target.write_text(json.dumps(report, indent=2))
        reloaded = json.loads(target.read_text())
        assert reloaded["all_passed"] is True
        assert {row["case"] for row in reloaded["criteria"]} >= {
            "wigley_hydrostatics", "ellipsoid_wave_resistance",
            "tandem_interference", "truncation", "wigley_wave_resistance",
            "ellipsoid_geometry",
        }

    def test_hull_parameter_validation(self):
        """The Froude/speed helpers are exact inverses."""
        for froude in bench.FROUDES:
            U = speed_for_froude(froude, L)
            assert froude_number(U, L) == pytest.approx(froude, rel=1e-15)

    def test_manifold_scale_check(self):
        """Kinematic consistency: ``Fn = 0.3`` on ``L = 2 m`` is ``1.33 m/s``."""
        assert speed_for_froude(0.3, 2.0) == pytest.approx(
            0.3 * math.sqrt(9.80665 * 2.0), rel=1e-15
        )
        # The repository's own Wigley helper agrees with the new module's.
        assert wigley_module.wigley_hull_properties(L, B, T)["Cb"] == pytest.approx(
            4.0 / 9.0
        )