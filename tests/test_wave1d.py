"""Tests for the 1-D wave equation solver (wave1d module)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem import wave1d as w1d


class TestDalembert:
    """d'Alembert formula tests."""

    def test_stationary_pulse(self):
        """A stationary Gaussian pulse (v0=0) should split into two half-amplitude pulses."""
        c = 1.0
        centre, sigma = 5.0, 0.5

        def u0(x):
            return w1d._gaussian(x, centre, sigma)

        def v0(x):
            return np.zeros_like(x)

        x = np.linspace(0, 10, 100)
        u = w1d.dalembert(x, 0.0, u0, v0, c)
        assert np.allclose(u, u0(x), atol=1e-10)

        u_t1 = w1d.dalembert(x, 1.0, u0, v0, c)
        expected = 0.5 * (w1d._gaussian(x, centre - c, sigma) + w1d._gaussian(x, centre + c, sigma))
        assert np.allclose(u_t1, expected, atol=1e-8)

    def test_right_moving_pulse(self):
        """A right-moving pulse should translate without changing shape."""
        c = 2.0
        centre, sigma = 3.0, 0.3

        def u0(x):
            return w1d._gaussian(x, centre, sigma)

        def v0(x):
            return -c * w1d._gaussian_prime(x, centre, sigma)

        x = np.linspace(0, 10, 100)
        t = 1.5
        u = w1d.dalembert(x, t, u0, v0, c)
        expected = w1d._gaussian(x, centre + c * t, sigma)
        assert np.allclose(u, expected, atol=1e-8)

    def test_left_moving_pulse(self):
        """A left-moving pulse should translate without changing shape."""
        c = 1.5
        centre, sigma = 7.0, 0.4

        def u0(x):
            return w1d._gaussian(x, centre, sigma)

        def v0(x):
            return c * w1d._gaussian_prime(x, centre, sigma)

        x = np.linspace(0, 10, 100)
        t = 2.0
        u = w1d.dalembert(x, t, u0, v0, c)
        expected = w1d._gaussian(x, centre - c * t, sigma)
        assert np.allclose(u, expected, atol=1e-8)


class TestSolveWave1D:
    """solve_wave1d tests."""

    def test_transparent_bc_exact(self):
        """Transparent BCs should give exact d'Alembert solution."""
        L, c = 10.0, 1.0
        centre, sigma = 5.0, 0.5

        def u0(x):
            return w1d._gaussian(x, centre, sigma)

        def v0(x):
            return np.zeros_like(x)

        result = w1d.solve_wave1d(L, c, u0, v0, t_max=3.0, dt=0.01, bc_type="transparent")
        err = np.max(np.abs(result.u_field - result.u_exact))
        assert err < 1e-12, f"Transparent BC error {err} exceeds machine precision"

    def test_wave_interaction(self):
        """Two counter-propagating pulses should pass through each other."""
        case = w1d.wave_interaction_case(L=10.0, c=1.0, t_max=8.0, dt=0.01)
        assert case["passes_5pct"], f"L2 error {case['l2_relative_error']} exceeds 5%"
        assert case["l2_relative_error"] < 1e-10

    def test_wave_interaction_dirichlet(self):
        """Wave interaction with Dirichlet BCs: before reflections, error is small."""
        case = w1d.wave_interaction_case(L=10.0, c=1.0, t_max=4.0, dt=0.005, bc_type="dirichlet")
        # Before reflections reach the interaction region (t < 5), error should be small
        assert case["l2_relative_error"] < 0.01, f"Dirichlet BC error {case['l2_relative_error']} too large"

    def test_dirichlet_bc_at_boundary(self):
        """Dirichlet BCs should enforce u=0 at boundaries."""
        L, c = 10.0, 1.0
        centre, sigma = 5.0, 0.5

        def u0(x):
            return w1d._gaussian(x, centre, sigma)

        def v0(x):
            return np.zeros_like(x)

        # Evaluate at points including the boundaries
        x_eval = np.linspace(0, L, 200)
        result = w1d.solve_wave1d(L, c, u0, v0, t_max=6.0, dt=0.01, x_eval=x_eval, bc_type="dirichlet")

        # Check u(0, t) and u(L, t) are close to 0
        u_left = result.u_field[:, 0]
        u_right = result.u_field[:, -1]
        assert np.allclose(u_left, 0, atol=0.05), f"Dirichlet BC at x=0 violated, max={np.max(np.abs(u_left))}"
        assert np.allclose(u_right, 0, atol=0.05), f"Dirichlet BC at x=L violated, max={np.max(np.abs(u_right))}"


class TestGaussianHelpers:
    """Gaussian helper function tests."""

    def test_gaussian_peak(self):
        x = np.array([3.0])
        assert np.isclose(w1d._gaussian(x, 3.0, 0.5), 1.0)

    def test_gaussian_derivative_zero_at_centre(self):
        x = np.array([3.0])
        assert np.isclose(w1d._gaussian_prime(x, 3.0, 0.5), 0.0)

    def test_gaussian_derivative_sign(self):
        x_left = np.array([2.0])
        x_right = np.array([4.0])
        assert w1d._gaussian_prime(x_left, 3.0, 0.5) > 0
        assert w1d._gaussian_prime(x_right, 3.0, 0.5) < 0
