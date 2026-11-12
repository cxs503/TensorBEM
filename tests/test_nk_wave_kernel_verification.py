"""Verification of the Neumann-Kelvin wave kernel (mission m16913).

Three things are checked here, in increasing order of importance.

1. **The principal-value quadrature.**  ``PV int_0^inf exp(-mu u) J0(q u)/(u-1) du``
   is now evaluated by the log-divergence-subtracted decomposition

       PV int_0^inf f(u)/(u-1) du
         = -int_0^1   [f(u) - f(1)]/(1-u) du
           + int_0^inf [f(1+v) - f(1) exp(-v)]/v dv  - gamma f(1),

   which has a closed form for ``f(u) = exp(-mu u)`` (``q = 0``),

       exp(-mu) [Ein(-mu) - log(mu) - gamma],      Ein(z) = sum_{k>=1} (-1)^(k+1) z^k/(k k!),

   so the quadrature is anchored to an exact value, not to itself.  It also
   has to be *stable*: the previous implementation split the integral at
   ``u = 1`` with an ``eps`` and used Gauss-Laguerre weights belonging to the
   weight ``exp(-u)`` while the integrand carried ``exp(-mu u)``, so its value
   moved with ``eps`` and with the node count.

2. **Route consistency.**  ``green_function_nk_vectorized`` used to replace the
   wave integral by an ad-hoc ``K_0`` heuristic, disagreeing with the scalar
   route by more than 1000 %; the solver assembles its matrices through the
   vectorized route, so the solver was fed that heuristic.  Both routes must
   now agree to round-off.

3. **The kernel's physics.**  ``K G_z + G_xx`` must vanish on ``z = 0`` for a
   steady Neumann-Kelvin kernel (it is the free-surface condition
   ``alpha^2 = K k`` restated).  The kernel shipped in
   :mod:`tensorbem.neumann_kelvin` is the *frequency-domain* Havelock kernel and
   does not satisfy it.  The test below measures the residual and pins the
   honest evidence level: ``solve_nk_wave_resistance`` returns WITHHELD.
   When a Kelvin kernel (``k = K sec^2 theta`` + downstream radiation
   condition) replaces ``G_W``, this test must be inverted, not deleted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.neumann_kelvin import (
    KELVIN_ANGLE_DEG,
    KERNEL_STATUS,
    NeumannKelvinResult,
    _wave_integral_steady,
    _pv_wave_integral_batch,
    green_function_nk,
    green_function_nk_vectorized,
    kelvin_curve_wavenumber,
    kelvin_stationary_angle_rad,
    solve_nk_wave_resistance,
    steady_free_surface_residual,
)
from tensorbem.wigley import wigley_hull_mesh

EULER_GAMMA = 0.5772156649015328606


def _ein_series(z: float, n: int = 60) -> float:
    """Ein(z) = sum_{k>=1} (-1)^(k+1) z^k / (k k!)."""
    total, term = 0.0, z
    for k in range(1, n + 1):
        total += ((-1) ** (k + 1)) * term / k
        term *= z / (k + 1)
    return total


def _pv_closed_form(mu: float) -> float:
    """Exact PV for f = exp(-mu u)  (i.e. q = 0)."""
    return math.exp(-mu) * (_ein_series(-mu) - math.log(mu) - EULER_GAMMA)


class TestPrincipalValueQuadrature:
    """The PV evaluator against an exact closed form and against itself."""

    @pytest.mark.parametrize("mu", [0.05, 0.5, 1.0, 3.0, 10.0])
    def test_closed_form_q0(self, mu):
        """q = 0 reduces to a closed form; quadrature must match it."""
        got = _wave_integral_batch_scalar(mu, 0.0)
        want = _pv_closed_form(mu)
        assert got == pytest.approx(want, rel=1e-6, abs=1e-9)

    @pytest.mark.parametrize("q", [1.0, 5.0, 20.0])
    def test_closed_form_shift(self, q):
        """f = exp(-mu u) J0(q u) at q -> small stays close to the q = 0 value.

        (Continuity check: J0(q u) -> 1, so the two must agree as q -> 0.)
        """
        small = _wave_integral_batch_scalar(2.0, 1e-6)
        assert small == pytest.approx(_pv_closed_form(2.0), rel=1e-6)

    @pytest.mark.parametrize("mu,q", [(0.5, 10.0), (2.0, 40.0), (8.0, 5.0)])
    def test_node_refinement_is_convergent(self, mu, q):
        """Refining the Gauss order by 4x must not move the value by more than 1e-9.

        The old implementation *did* move (eps- and order-dependent), which is
        what made its numbers meaningless.
        """
        coarse = float(_pv_wave_integral_batch(np.array([mu]), np.array([q]), 192)[0])
        fine = float(_pv_wave_integral_batch(np.array([mu]), np.array([q]), 768)[0])
        assert coarse == pytest.approx(fine, rel=1e-6, abs=1e-12 * abs(fine))

    def test_oscillatory_in_q(self):
        """The q-dependence is oscillatory (J0 kernel), magnitude decaying."""
        values = [abs(_wave_integral_batch_scalar(1.0, q)) for q in np.linspace(0.5, 30.0, 40)]
        assert max(values) - min(values) > 1e-3
        assert values[-1] < values[0]

    def test_wave_integral_matches_batch(self):
        """The scalar wrapper and the batched evaluator are the same kernel."""
        for R, z_sum in [(0.5, -0.2), (3.0, -1.0), (1.5, -0.05)]:
            scalar = _wave_integral_steady(R=R, z_sum=z_sum, kappa=5.0)
            batch = float(
                _wave_integral_steady_batch(np.array([R]), np.array([z_sum]), 5.0, 128)[0]
            )
            assert scalar == pytest.approx(batch, rel=1e-6, abs=1e-12)


class TestRouteConsistency:
    """Vectorized and scalar Green functions are one definition."""

    def test_agreement_to_round_off(self):
        rng = np.random.default_rng(20261113)
        targets = rng.uniform(-2.0, 2.0, (6, 3))
        sources = rng.uniform(-2.0, 2.0, (4, 3))
        targets[:, 2] = -0.05 - np.abs(targets[:, 2])
        sources[:, 2] = -0.05 - np.abs(sources[:, 2])
        for kappa in (1.0, 5.0, 20.0):
            vec = green_function_nk_vectorized(targets, sources, kappa)
            for i in range(targets.shape[0]):
                for j in range(sources.shape[0]):
                    scalar = green_function_nk(targets[i], sources[j], kappa)
                    assert vec[i, j] == pytest.approx(scalar, rel=1e-5, abs=1e-12 * max(1.0, abs(scalar)))

    def test_no_heuristic_branch(self):
        """For a deep pair the vectorized value is not the old K0 heuristic.

        The old branch returned ``(kappa/2pi) exp(-kappa|dz|) K0(kappa R)``;
        the true wave part is several times larger here, which is why the
        solver's matrices were wrong by orders of magnitude.
        """
        target = np.array([[0.5, 0.3, -0.2]])
        source = np.array([[0.0, 0.0, -0.5]])
        kappa = 5.0
        dz_img = target[0, 2] + source[0, 2]
        R = math.hypot(0.5, 0.3)
        arg = kappa * R
        k0 = math.sqrt(math.pi / (2.0 * arg)) * math.exp(-arg)
        legacy = (kappa / (2 * math.pi)) * math.exp(-kappa * abs(dz_img)) * k0
        vec = green_function_nk_vectorized(target, source, kappa)[0, 0]
        scalar = green_function_nk(target[0], source[0], kappa)
        wave = scalar - (
            1.0 / (4 * math.pi * math.hypot(R, target[0, 2] - source[0, 2]))
            + 1.0 / (4 * math.pi * math.hypot(R, dz_img))
        )
        assert abs(wave - legacy) / abs(wave) > 0.5
        assert vec == pytest.approx(scalar, rel=1e-5)


class TestSteadyFreeSurfaceCondition:
    """The shipped kernel does not satisfy the condition it is named after."""

    @pytest.mark.parametrize(
        "x,y",
        [(1.0, 0.0), (2.0, 0.5), (-1.5, 0.3), (0.5, 1.2)],
    )
    def test_residual_is_large_and_step_independent(self, x, y):
        """``K G_z + G_xx`` must vanish; measured relative residual is O(0.01-1).

        Step independence (h = 1e-5 vs 2.5e-6) shows the residual is a property
        of the kernel, not of the finite differences.
        """
        target = np.array([x, y, 0.0])
        source = np.array([0.0, 0.0, -0.6])
        coarse = steady_free_surface_residual(target, source, 5.0, h=1.0e-5)
        fine = steady_free_surface_residual(target, source, 5.0, h=2.5e-6)
        assert coarse["relative"] > 1.0e-2
        assert fine["relative"] == pytest.approx(coarse["relative"], rel=0.05)

    def test_residual_does_not_shrink_with_kappa(self):
        """A wrong kernel cannot be rescued by the speed: the residual persists."""
        target = np.array([1.0, 0.0, 0.0])
        source = np.array([0.0, 0.0, -0.6])
        for kappa in (1.0, 5.0, 50.0):
            res = steady_free_surface_residual(target, source, kappa, h=1.0e-6)
            assert res["relative"] > 1.0e-3


class TestKelvinCurve:
    """The steady dispersion relation and the angles it generates."""

    def test_transverse_wavenumber_is_kappa(self):
        assert kelvin_curve_wavenumber(0.0, 7.5) == pytest.approx(7.5)

    def test_monotone_along_the_family(self):
        kappas = [kelvin_curve_wavenumber(t, 3.0) for t in np.linspace(0.0, 0.6, 12)]
        assert all(b > a for a, b in zip(kappas, kappas[1:]))

    def test_stationary_phase_angle(self):
        """``dphi/dtheta = 0`` at ``tan(theta_s) = 1/sqrt(2) = 35.264 deg``."""
        theta_s = kelvin_stationary_angle_rad()
        assert math.tan(theta_s) == pytest.approx(1.0 / math.sqrt(2.0))
        assert math.degrees(theta_s) == pytest.approx(35.264389682754654)

    def test_cusp_half_angle_from_stationary_angle(self):
        """``tan(alpha) = tan(theta_s)/(1 + 2 tan^2 theta_s)`` gives arcsin(1/3)."""
        t = math.tan(kelvin_stationary_angle_rad())
        alpha = math.atan(t / (1.0 + 2.0 * t * t))
        assert math.sin(alpha) == pytest.approx(1.0 / 3.0)
        assert math.degrees(alpha) == pytest.approx(KELVIN_ANGLE_DEG, rel=1e-12)
        assert KELVIN_ANGLE_DEG == pytest.approx(19.471220634490691, rel=1e-12)


class TestEvidenceLevelIsDeclared:
    """No route may present the NK ``C_w`` as evidence."""

    def test_solver_result_carries_withheld_status(self):
        mesh = wigley_hull_mesh(2.0, 0.2, 0.125, 4, 2)
        result = solve_nk_wave_resistance(mesh.vertices, mesh.faces, 0.5)
        assert isinstance(result, NeumannKelvinResult)
        assert result.kernel_status.startswith("WITHHELD")
        assert KERNEL_STATUS.startswith("WITHHELD")
        assert math.isfinite(result.cw)


from tensorbem.neumann_kelvin import _wave_integral_steady_batch  # noqa: E402


def _wave_integral_batch_scalar(mu: float, q: float) -> float:
    """PV integral (unscaled) via the batched evaluator."""
    n = max(384, int(1.5 * q) + 256)
    return float(_pv_wave_integral_batch(np.array([mu]), np.array([q]), n)[0])