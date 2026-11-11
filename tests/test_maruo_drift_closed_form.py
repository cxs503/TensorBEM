"""Closed-form verification of the Maruo far-field mean drift force.

This test verifies the Maruo (1957) far-field drift force formula against
analytic azimuthal integrals. The Maruo formula is:

    F_x = (rho * omega^2 / 4k) * int_0^{2pi} |K(theta)|^2 * (1 - cos(theta)) dtheta

where K(theta) is the Kochin function of the scattered wave.

For a Kochin function with |K|^2 = C^2 * (1 + a * cos(theta - gamma)), the
integrals can be evaluated in closed form:

    S = int |K|^2 dtheta = 2 * pi * C^2
    Ci = int |K|^2 * cos(theta) dtheta = a * pi * C^2 * cos(gamma)

The head-seas Maruo formula gives:
    F_x = (rho * g / 4) * [S - Ci] = (rho * g / 4) * 2*pi*C^2 * (1 - a*cos(gamma)/2)

References
----------
Maruo, H. (1957). "On the theory of ship drift in waves." J. Soc. Naval Arch. Japan, 102.
Newman, J.N. (1977). Marine Hydrodynamics, MIT Press, section 7.3.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from tensorbem.maruo_drift import maruo_mean_drift_force

RHO = 1025.0
G = 9.81
K_WAVE = 0.5
COEFF = RHO * G / 4.0


def _closed_form_fx(gamma=0.0, c=1.0, a=0.5):
    """Analytic F_x from the Maruo integral in head seas."""
    S = 2.0 * math.pi * c * c
    Ci = a * math.pi * c * c * math.cos(gamma)
    return COEFF * (S - Ci)


def _kochin_table(n=3600, gamma=0.0, c=1.0, a=0.5):
    """Sampled Kochin table with |K|^2 = c^2 * (1 + a * cos(theta - gamma))."""
    theta = np.linspace(0.0, 2.0 * math.pi, n + 1)[:-1]
    pattern = np.maximum(1.0 + a * np.cos(theta - gamma), 0.0)
    return c * np.sqrt(pattern), theta


class TestMaruoClosedForm:
    """Verify Maruo drift force against closed-form azimuthal integrals."""

    def test_isotropic_kochin_head_seas(self):
        """|K|^2 = const: F_x = (rho*g/4) * 2*pi*c^2."""
        c = 0.7
        kochin, theta = _kochin_table(c=c, a=0.0)
        result = maruo_mean_drift_force(kochin, theta, K_WAVE)
        
        expected_fx = COEFF * 2.0 * math.pi * c * c
        
        assert result["mean_drift_force_x"] == pytest.approx(expected_fx, rel=1e-4)

    def test_anisotropic_kochin_head_seas(self):
        """|K|^2 = c^2*(1 + a*cos(theta)): F_x reduced by moment term."""
        c = 1.0
        a = 0.5
        gamma = 0.0
        kochin, theta = _kochin_table(c=c, a=a, gamma=gamma)
        result = maruo_mean_drift_force(kochin, theta, K_WAVE)
        
        expected_fx = _closed_form_fx(gamma=gamma, c=c, a=a)
        
        assert result["mean_drift_force_x"] == pytest.approx(expected_fx, rel=1e-4)

    def test_energy_conservation_positive_drift(self):
        """Drift force must be positive in wave direction (energy flux)."""
        c = 1.0
        a = 0.3
        kochin, theta = _kochin_table(c=c, a=a)
        result = maruo_mean_drift_force(kochin, theta, K_WAVE)
        
        assert result["mean_drift_force_x"] > 0.0

    def test_quadrature_convergence(self):
        """Increasing azimuthal samples improves accuracy."""
        c = 1.0
        a = 0.5
        expected_fx = _closed_form_fx(c=c, a=a)
        
        errors = []
        for n in [100, 400, 1600, 3600]:
            kochin, theta = _kochin_table(n=n, c=c, a=a)
            result = maruo_mean_drift_force(kochin, theta, K_WAVE)
            error = abs(result["mean_drift_force_x"] - expected_fx) / abs(expected_fx)
            errors.append(error)
        
        # Errors should decrease monotonically
        for i in range(len(errors) - 1):
            assert errors[i] > errors[i+1], f"n={100*4**i}: error did not decrease"

    def test_wavenumber_independence(self):
        """Prefactor rho*g/4 is independent of wavenumber (omega^2 = g*k)."""
        c = 1.0
        a = 0.0
        kochin, theta = _kochin_table(c=c, a=a)
        
        expected_fx = COEFF * 2.0 * math.pi * c * c
        
        for k in [0.1, 0.5, 1.0, 2.0]:
            result = maruo_mean_drift_force(kochin, theta, k)
            assert result["mean_drift_force_x"] == pytest.approx(expected_fx, rel=1e-4)

    def test_gamma_dependence(self):
        """Varying gamma (phase shift) changes the moment term Ci."""
        c = 1.0
        a = 0.5
        n = 3600
        
        # gamma=0: Ci = a*pi*c^2 (maximum)
        # gamma=pi/2: Ci = 0
        # gamma=pi: Ci = -a*pi*c^2 (minimum, so F_x is maximum)
        
        kochin0, theta0 = _kochin_table(n=n, c=c, a=a, gamma=0.0)
        result0 = maruo_mean_drift_force(kochin0, theta0, K_WAVE)
        
        kochin_pi2, theta_pi2 = _kochin_table(n=n, c=c, a=a, gamma=math.pi/2)
        result_pi2 = maruo_mean_drift_force(kochin_pi2, theta_pi2, K_WAVE)
        
        kochin_pi, theta_pi = _kochin_table(n=n, c=c, a=a, gamma=math.pi)
        result_pi = maruo_mean_drift_force(kochin_pi, theta_pi, K_WAVE)
        
        # F_x(gamma=0) < F_x(gamma=pi/2) < F_x(gamma=pi)
        # because Ci decreases from + to 0 to -
        assert result0["mean_drift_force_x"] < result_pi2["mean_drift_force_x"]
        assert result_pi2["mean_drift_force_x"] < result_pi["mean_drift_force_x"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
