"""Pins the steady (Kelvin) kernel formulation and its exact k-integral.

The shipped ``neumann_kelvin`` kernel is the frequency-domain Havelock kernel and
carries WITHHELD (it cannot satisfy ``K G_z + G_xx = 0``).  ``tensorbem.steady_kelvin``
supplies the correct Kelvin-curve formulation together with a closed form for the
singular ``k``-integral.  These tests pin what is *proved* (the identity, the
formulation's structure) and pin what is *still open* (the radiation-condition
sign), so that neither can be silently re-labelled.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import steady_kelvin as sk
from tensorbem.neumann_kelvin import steady_free_surface_residual


CASES = [(0, 1.0, 2.0), (1, 1.0, 2.0), (2, 1.0, 2.0), (1, 0.5, 0.3), (2, 0.05, 1.0)]


class TestClosedFormIdentity:
    @pytest.mark.parametrize("m, mu, pole", CASES)
    def test_matches_singularity_subtracted_quadrature(self, m, mu, pole):
        s = mu - 1j * 2.0
        closed = sk.pv_moment_int(m, pole, s)
        ref = sk.pv_moment_int_quadrature(m, pole, s)
        assert abs(closed - ref) <= 1e-6 * max(abs(ref), 1e-30)

    def test_exp_ei_small_and_large_argument_agree(self):
        # the asymptotic branch must join the direct branch at the switch-over
        z_lo = np.array([24.0 + 1.0j, 24.0 - 3.0j])
        z_hi = np.array([40.0 + 1.0j])
        direct = np.exp(-z_lo) * pytest.importorskip("scipy.special").expi(z_lo)
        assert np.allclose(sk.exp_ei(z_lo), direct, rtol=1e-13)
        # asymptotic at |z| = 40 vs. a 60-digit mpmath-free cross-check: the
        # product must be O(1/|z|) rather than exponential
        assert abs(sk.exp_ei(z_hi)[0]) < 0.05

    def test_rankine_xx_matches_finite_difference(self):
        zs = -1.0
        h = 1e-5
        fd = (sk.rankine_part(h, 0.0, 0.0, 0.0, 0.0, zs)
              - 2.0 * sk.rankine_part(0.0, 0.0, 0.0, 0.0, 0.0, zs)
              + sk.rankine_part(-h, 0.0, 0.0, 0.0, 0.0, zs)) / h ** 2
        assert abs(sk.rankine_xx(0.0, 0.0, 0.0, 0.0, 0.0, zs) - fd) <= 1e-6 * abs(fd)


class TestFormulationStructure:
    def test_kelvin_curve_matches_sec_squared(self):
        phi = np.array([0.0, math.pi / 6.0, math.pi / 4.0])
        kp = sk.kelvin_wavenumber(phi, 2.5)
        assert np.allclose(kp, 2.5 / np.cos(phi) ** 2)

    def test_kelvin_angle_is_arcsin_one_third(self):
        assert abs(sk.KELVIN_ANGLE_DEG - 19.471220634490691) < 1e-12

    def test_wave_part_decays_with_depth(self):
        """Every Kelvin mode carries exp(k_p z): the wave part must decay downward."""
        near = abs(sk.wave_part_complex(2.0, 0.0, -0.1, kappa=1.0, zs=-1.0))
        far = abs(sk.wave_part_complex(2.0, 0.0, -3.0, kappa=1.0, zs=-1.0))
        assert far < near


class TestOpenDefectIsPinned:
    """The candidate does NOT yet pass the referee: pin that, do not paper over it."""

    def test_shipped_kernel_fails_the_steady_referee(self):
        out = steady_free_surface_residual([4.0, 0.0, 0.0], [0.0, 0.0, -1.0], 1.0)
        assert out["relative"] > 1e-2, out

    @pytest.mark.parametrize("x", [0.5, 2.0])
    def test_candidate_residual_is_still_not_quadrature_limited(self, x):
        out = sk.candidate_free_surface_residual(x, -1.0, 1.0)
        assert out["relative"] > 1e-3, out

    def test_residual_is_quadrature_converged_but_nonzero(self):
        """The residual is *not* a discretization artifact: it is flat in nodes."""
        coarse = sk.candidate_free_surface_residual(2.0, -1.0, 1.0, nodes=256)["relative"]
        fine = sk.candidate_free_surface_residual(2.0, -1.0, 1.0, nodes=1024)["relative"]
        assert abs(fine - coarse) < 1e-4, (coarse, fine)   # flat vs. the 14 % level
        assert fine > 1e-3, fine

    def test_candidate_is_symmetric_upstream_downstream(self):
        up = abs(sk.wave_part_complex(-2.0, 0.0, 0.0, kappa=1.0, zs=-1.0))
        down = abs(sk.wave_part_complex(2.0, 0.0, 0.0, kappa=1.0, zs=-1.0))
        assert abs(up - down) <= 1e-9 * max(down, 1e-30)

    def test_status_keeps_the_withheld_verdict(self):
        assert sk.STATUS.startswith("CANDIDATE")
        assert "WITHHELD" in sk.STATUS

    def test_shipped_module_status_unchanged(self):
        from tensorbem import neumann_kelvin as nk
        assert nk.KERNEL_STATUS.startswith("WITHHELD")
