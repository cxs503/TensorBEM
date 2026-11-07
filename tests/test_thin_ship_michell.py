"""Tests for the standard-form Michell thin-ship module.

The checks are layered the way the module is: the *depth kernel* first (a
one-dimensional analytic reduction against quadrature), then the *amplitude* of
the spheroid against a parameter-space quadrature that never touches that
reduction, then the two *wave-resistance forms* against each other, then the
*panel discretisation* against the reference, and finally the tandem
superposition and its analytic limits.

Every reference here is independent of the code path it checks; nothing is
compared against a stored number that this file cannot regenerate.
"""

import math

import numpy as np
import pytest

from tensorbem.thin_ship_michell import (
    CenterplanePanels,
    EllipsoidHull,
    WigleyHull,
    ellipsoid_amplitude_function,
    ellipsoid_amplitude_quadrature_2d,
    ellipsoid_amplitude_reference,
    ellipsoid_panels,
    grid_convergence_index,
    michell_wave_resistance,
    tandem_amplitude,
    wave_resistance_from_amplitude,
    wave_resistance_theta_form,
    wigley_amplitude_function,
    wigley_amplitude_reference,
    wigley_panels,
)
from tensorbem.thin_ship_michell import _depth_integral, _legendre_gauss
from tensorbem.wigley import michell_kochin_function

G = 9.80665
RHO = 1025.0
ELLIPSOID = EllipsoidHull(L=10.0, B=1.0, T=0.6)
WIGLEY = WigleyHull(L=10.0, B=1.0, T=0.625)


def _froude_speed(froude: float, length: float) -> float:
    return froude * math.sqrt(G * length)


class TestDepthKernel:
    """``G(v) = int_0^{pi/2} exp(-v sin t) dt``: series, interpolant, quadrature."""

    @staticmethod
    def _quadrature(v: float, n: int = 200) -> float:
        x, w = _legendre_gauss(n)
        t = 0.25 * math.pi * (x + 1.0)
        return float(np.sum(0.25 * math.pi * w * np.exp(-v * np.sin(t))))

    @pytest.mark.parametrize("v", [0.0, 0.5, 2.0, 5.0, 10.0, 14.0, 16.0, 20.0, 40.0, 200.0])
    def test_matches_quadrature(self, v: float) -> None:
        assert float(_depth_integral(np.array([v]))[0]) == pytest.approx(
            self._quadrature(v), rel=5.0e-6
        )

    def test_small_v_limit_is_pi_over_two(self) -> None:
        assert float(_depth_integral(np.array([0.0]))[0]) == pytest.approx(
            0.5 * math.pi, rel=1.0e-9
        )

    def test_large_v_decays_like_one_over_v(self) -> None:
        for v in (50.0, 500.0, 5000.0):
            value = float(_depth_integral(np.array([v]))[0])
            # G ~ (1/v)(1 + 1/v^2 + ...): the scaled value approaches one from above
            assert value * v == pytest.approx(1.0 + 1.0 / v**2, rel=2.0e-3)

    def test_monotone_decreasing(self) -> None:
        values = _depth_integral(np.linspace(0.0, 60.0, 200))
        assert np.all(np.diff(values) < 0.0)


class TestEllipsoidAmplitude:
    """The analytic depth reduction against an independent parameter quadrature."""

    @pytest.mark.parametrize("k, alpha", [(0.5, 0.4), (2.0, 1.0), (5.0, 2.0)])
    def test_against_parameter_space_quadrature(self, k: float, alpha: float) -> None:
        reduced = ellipsoid_amplitude_reference(ELLIPSOID, k, alpha)
        quadrature = ellipsoid_amplitude_quadrature_2d(ELLIPSOID, k, alpha)
        assert abs(reduced - quadrature) <= 1.0e-9 * abs(reduced)

    def test_small_wavenumber_limit(self) -> None:
        """``A ~ -i alpha pi a b c / 3`` as ``k, alpha -> 0``."""
        alpha = 1.0e-4
        expected = -1j * alpha * math.pi * ELLIPSOID.a * ELLIPSOID.b * ELLIPSOID.c / 3.0
        value = ellipsoid_amplitude_reference(ELLIPSOID, 0.0, alpha)
        assert abs(value - expected) <= 1.0e-6 * abs(expected)

    def test_amplitude_is_stationary_under_refinement(self) -> None:
        coarse = ellipsoid_amplitude_reference(ELLIPSOID, 40.0, 30.0, n_cell=200)
        fine = ellipsoid_amplitude_reference(ELLIPSOID, 40.0, 30.0, n_cell=800)
        assert abs(coarse - fine) <= 1.0e-6 * abs(fine)

    def test_wigley_amplitude_matches_kochin_function(self) -> None:
        """``2 A(k, alpha=k)`` is the Kochin function of :mod:`tensorbem.wigley`."""
        for k in (0.3, 1.0, 2.5, 7.0):
            assert 2.0 * wigley_amplitude_reference(WIGLEY, k, k) == pytest.approx(
                complex(michell_kochin_function(WIGLEY.L, WIGLEY.B, WIGLEY.T, k)), rel=1.0e-12
            )

    def test_wigley_vectorised_amplitude_matches_scalar(self) -> None:
        vector = wigley_amplitude_function(WIGLEY)
        k = np.array([0.5, 2.0, 9.0])
        alpha = np.array([0.25, 1.5, 4.0])
        for index in range(k.size):
            assert complex(vector(k, alpha)[index]) == pytest.approx(
                wigley_amplitude_reference(WIGLEY, float(k[index]), float(alpha[index])),
                rel=1.0e-12,
            )


class TestWaveResistanceForms:
    """The ``lambda`` and ``theta`` forms of Michell's integral must agree."""

    def test_wigley_forms_agree(self) -> None:
        amplitude = wigley_amplitude_function(WIGLEY)
        speed = _froude_speed(0.3, WIGLEY.L)
        k0 = G / speed**2
        lam_form = wave_resistance_from_amplitude(amplitude, speed, lam_max=40.0, n_lam=180)
        theta_form = wave_resistance_theta_form(
            lambda theta: amplitude(k0 / np.cos(theta) ** 2, k0 / np.cos(theta)),
            speed,
            n_theta=80,
            theta_max=math.acos(1.0 / 40.0),
        )
        assert theta_form == pytest.approx(lam_form, rel=2.0e-3)

    def test_ellipsoid_forms_agree(self) -> None:
        amplitude = ellipsoid_amplitude_function(ELLIPSOID)
        speed = _froude_speed(0.3, ELLIPSOID.L)
        k0 = G / speed**2
        lam_form = wave_resistance_from_amplitude(amplitude, speed, lam_max=40.0, n_lam=200)
        theta_form = wave_resistance_theta_form(
            lambda theta: amplitude(k0 / np.cos(theta) ** 2, k0 / np.cos(theta)),
            speed,
            n_theta=60,
            theta_max=math.acos(1.0 / 40.0),
        )
        assert theta_form == pytest.approx(lam_form, rel=5.0e-3)

    def test_wigley_resistance_grows_with_speed(self) -> None:
        amplitude = wigley_amplitude_function(WIGLEY)
        values = [
            wave_resistance_from_amplitude(amplitude, _froude_speed(f, WIGLEY.L), lam_max=40.0, n_lam=120)
            for f in (0.2, 0.3, 0.4)
        ]
        assert values[0] < values[1] < values[2]

    def test_rejects_bad_input(self) -> None:
        amplitude = wigley_amplitude_function(WIGLEY)
        with pytest.raises(ValueError):
            wave_resistance_from_amplitude(amplitude, 0.0)
        with pytest.raises(ValueError):
            wave_resistance_from_amplitude(amplitude, 1.0, lam_max=1.0)


class TestPanelDiscretisation:
    """The panel path must converge onto the reference from below."""

    def test_panel_container_rejects_bad_shapes(self) -> None:
        with pytest.raises(ValueError):
            CenterplanePanels(nodes=np.zeros((2, 3, 2)), weights=np.zeros((2, 3)), density=np.zeros((2, 3)))
        with pytest.raises(ValueError):
            CenterplanePanels(nodes=np.zeros((2, 4, 2)), weights=np.zeros((3, 4)), density=np.zeros((2, 4)))

    def test_wigley_panels_converge(self) -> None:
        speed = _froude_speed(0.3, WIGLEY.L)
        reference = wave_resistance_from_amplitude(
            wigley_amplitude_function(WIGLEY), speed, lam_max=40.0, n_lam=120
        )
        errors = []
        for n_x in (10, 20):
            panels = wigley_panels(WIGLEY, n_x, n_x)
            value = wave_resistance_from_amplitude(panels.amplitude, speed, lam_max=40.0, n_lam=120)
            errors.append(abs(value - reference) / reference)
        assert errors[0] > errors[1]
        assert errors[1] < 1.0e-2

    def test_ellipsoid_panels_converge_and_are_accurate(self) -> None:
        speed = _froude_speed(0.3, ELLIPSOID.L)
        reference = wave_resistance_from_amplitude(
            ellipsoid_amplitude_function(ELLIPSOID), speed, lam_max=40.0, n_lam=240
        )
        coarse = wave_resistance_from_amplitude(
            ellipsoid_panels(ELLIPSOID, 16, 128).amplitude, speed, lam_max=40.0, n_lam=240
        )
        fine = wave_resistance_from_amplitude(
            ellipsoid_panels(ELLIPSOID, 32, 256).amplitude, speed, lam_max=40.0, n_lam=240
        )
        assert abs(fine - reference) / reference < 0.03
        assert abs(fine - reference) < abs(coarse - reference)

    def test_panel_amplitude_of_flat_waterline_is_finite(self) -> None:
        """Panels whose Gauss points sit near the singular keel line stay finite."""
        panels = ellipsoid_panels(ELLIPSOID, 8, 32)
        assert np.all(np.isfinite(panels.density))
        assert np.all(np.isfinite(panels.amplitude(40.0, 25.0)))


class TestTandemInterference:
    """Superposition of two Kochin amplitudes and its two exact limits."""

    def test_coincident_hulls_double_the_amplitude(self) -> None:
        single = wigley_amplitude_function(WIGLEY)
        pair = tandem_amplitude(single, 0.0)
        for k, alpha in ((1.0, 0.5), (5.0, 3.0)):
            assert abs(pair(k, alpha)) == pytest.approx(2.0 * abs(single(k, alpha)), rel=1.0e-12)

    def test_sigma_limits(self) -> None:
        single = wigley_amplitude_function(WIGLEY)
        speed = _froude_speed(0.3, WIGLEY.L)
        reference = michell_wave_resistance(
            single, speed, lam_max=40.0, n_lam=180, extrapolate_tail=False
        )["Rw"]
        coincident = michell_wave_resistance(
            tandem_amplitude(single, 1.0e-9), speed, lam_max=40.0, n_lam=180, extrapolate_tail=False
        )["Rw"]
        far = michell_wave_resistance(
            tandem_amplitude(single, 60.0 * WIGLEY.L), speed, lam_max=40.0, n_lam=180,
            extrapolate_tail=False,
        )["Rw"]
        assert coincident / (2.0 * reference) == pytest.approx(2.0, rel=1.0e-6)
        assert far / (2.0 * reference) == pytest.approx(1.0, abs=0.05)

    def test_interference_is_a_ratio_of_positive_numbers(self) -> None:
        single = wigley_amplitude_function(WIGLEY)
        panels = wigley_panels(WIGLEY, 20, 20)
        speed = _froude_speed(0.3, WIGLEY.L)

        def pair(k, alpha):
            return panels.amplitude(k, alpha) + panels.amplitude(k, alpha, x_offset=15.0)

        single_value = wave_resistance_from_amplitude(panels.amplitude, speed, lam_max=40.0, n_lam=180)
        pair_value = wave_resistance_from_amplitude(pair, speed, lam_max=40.0, n_lam=180)
        sigma = pair_value / (2.0 * single_value)
        assert 0.5 < sigma < 1.5


class TestGridConvergenceIndex:
    def test_recovers_a_known_order(self) -> None:
        exact = 2.0
        # coarse, medium, fine with the step halving: second-order convergence
        values = [exact - 0.5 * h**2 for h in (1.0, 0.5, 0.25)]
        result = grid_convergence_index(values[0], values[1], values[2], refinement=2.0)
        assert result["order"] == pytest.approx(2.0, abs=0.05)
        assert 0.0 < result["gci_fine"] < 0.1

    def test_identical_values_give_zero(self) -> None:
        result = grid_convergence_index(1.0, 1.0, 1.0)
        assert result["gci_fine"] == 0.0
        assert result["order"] == 0.0

    def test_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError):
            grid_convergence_index(1.0, 1.0, 1.0, refinement=1.0)
        with pytest.raises(ValueError):
            grid_convergence_index(0.0, 1.0, 2.0)


class TestEllipsoidHydrostatics:
    def test_block_coefficient_is_pi_over_six(self) -> None:
        reference = ELLIPSOID.hydrostatic_reference()
        assert reference["Cb"] == pytest.approx(math.pi / 6.0, rel=1.0e-14)
        assert reference["waterplane_area"] == pytest.approx(
            math.pi * ELLIPSOID.a * ELLIPSOID.b, rel=1.0e-14
        )
        assert reference["KB"] == pytest.approx(5.0 * ELLIPSOID.T / 8.0, rel=1.0e-14)

    def test_wetted_area_is_stable_under_refinement(self) -> None:
        coarse = ELLIPSOID.wetted_area_reference(60, 120)
        fine = ELLIPSOID.wetted_area_reference(120, 240)
        assert coarse == pytest.approx(fine, rel=1.0e-6)
        # a slender spheroid of this fineness must have more wetted area than the
        # centreplane projection and less than the circumscribing box
        assert math.pi * ELLIPSOID.a * ELLIPSOID.b < fine < 4.0 * ELLIPSOID.a * ELLIPSOID.b * 2.0