"""Helmholtz single-layer kernel: analytic pins and the m16451 mode contract.

The pre-existing tests only asserted the *mode string* and the output shape of
:func:`tensorbem.fmm.fmm_helmholtz_matvec`, so three silent truncations passed
CI (see ``docs/fmm_helmholtz_kernel/README.md``):

1. the returned potential was ``Re(sum q exp(ikr)/(4 pi r))`` only -- the
   imaginary quadrature was discarded, i.e. ``cos(kr)/r`` (a standing-wave
   mixture of outgoing and incoming waves) was summed in place of the outgoing
   solution: 28 %--99 % relative distance to the true complex field;
2. for ``ka < 0.1`` the *Laplace* kernel was summed (``k`` ignored, imaginary
   part dropped) and tagged ``"fmm"``: 36 % relative error at ``ka = 0.097``;
3. complex ``charges`` were cast to ``float64`` with a ``ComplexWarning``.

Every test below binds the operator to the analytic Helmholtz kernel
``q_j exp(i k r_ij) / (4 pi r_ij)`` instead of to a mode string.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tensorbem.fmm import (
    HELMHOLTZ_MODES,
    fmm_helmholtz_matvec,
    helmholtz_supported,
    source_extent,
)


def _analytic(targets: np.ndarray, sources: np.ndarray, charges: np.ndarray,
              k: float) -> np.ndarray:
    """Exact complex single-layer Helmholtz potential (independent of the operator)."""
    diff = targets[:, None, :] - sources[None, :, :]
    r = np.maximum(np.linalg.norm(diff, axis=2), 1e-15)
    return np.sum(charges[None, :] * np.exp(1j * k * r) / (4.0 * np.pi * r), axis=1)


def _cloud(n: int = 40, extent: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    sources = rng.normal(size=(n, 3)) * extent
    charges = rng.normal(size=n)
    targets = rng.normal(size=(12, 3)) * 1.5
    return targets, sources, charges


def _rel_err(num: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(num - ref)) / np.max(np.abs(ref)))


# ---------------------------------------------------------------------------
# 1. Analytic pin: single source
# ---------------------------------------------------------------------------

class TestAnalyticPin:
    def test_single_source_matches_closed_form(self):
        """One unit source: Phi(x) = exp(ik|x-y|) / (4 pi |x-y|), exactly."""
        k = 4.0
        source = np.array([[0.3, -0.2, 0.7]])
        targets = np.array([[2.0, 1.0, -1.0], [0.0, 0.0, 0.0], [-3.0, 2.5, 0.5]])
        pot, mode = fmm_helmholtz_matvec(targets, source, np.array([1.0]), k=k)
        r = np.linalg.norm(targets - source, axis=1)
        closed = np.exp(1j * k * r) / (4.0 * np.pi * r)
        assert mode == "complex_direct"
        assert np.max(np.abs(pot - closed)) <= 1e-14 * np.max(np.abs(closed))

    def test_quadrature_is_complete(self):
        """Both quadratures are needed: the imag part is a large fraction of Phi.

        A time-harmonic answer is ``Re(Phi exp(-i w t))``; with ``t`` such that
        ``w t = pi/2`` the answer is ``Im(Phi)``.  Returning ``Re(Phi)`` alone
        (pre-m16451) answers only the cosine quadrature.
        """
        targets, sources, charges = _cloud()
        k = 5.0
        pot, _ = fmm_helmholtz_matvec(targets, sources, charges, k=k)
        ref = _analytic(targets, sources, charges, k)
        imag_fraction = float(np.max(np.abs(ref.imag)) / np.max(np.abs(ref)))
        assert pot.dtype == np.complex128
        assert imag_fraction > 0.1, "test geometry must excite the sine quadrature"
        # the truncated (real-only) answer misses this much of the field:
        loss = float(np.max(np.abs(ref - ref.real)) / np.max(np.abs(ref)))
        assert loss > 0.15, "real-only truncation is above the 15 % tolerance band"


# ---------------------------------------------------------------------------
# 2. Mode contract: every tag states what was summed
# ---------------------------------------------------------------------------

class TestModeContract:
    def test_no_mode_claims_unavailable_fmm(self):
        """``"fmm"`` must not be a mode: no Helmholtz FMM exists in this module."""
        assert "fmm" not in HELMHOLTZ_MODES
        assert set(HELMHOLTZ_MODES) == {"complex_direct", "laplace_approx", "not_supported"}

    @pytest.mark.parametrize("ka", [0.02, 0.099, 0.5, 1.0, 5.0, 10.0, 12.0, 40.0])
    def test_band_scan_is_exact_and_truthfully_tagged(self, ka: float):
        """Every band returns the exact kernel sum; tags only describe cost support."""
        targets, sources, charges = _cloud()
        extent = source_extent(sources)
        assert extent > 0.0
        k = ka / extent
        pot, mode = fmm_helmholtz_matvec(targets, sources, charges, k=k)
        ref = _analytic(targets, sources, charges, k)
        assert mode in HELMHOLTZ_MODES
        assert mode == ("complex_direct" if ka <= 10.0 else "not_supported")
        assert mode != "laplace_approx", "asymptotic route must be opt-in"
        assert _rel_err(pot, ref) <= 1e-12
        assert helmholtz_supported(ka) is (ka <= 10.0)

    def test_opt_in_asymptotic_is_flagged_and_inaccurate(self):
        """The Laplace asymptotic is opt-in, tagged, and has a documented error floor.

        ``low_ka_asymptotic=True`` at ``ka ~ 0.1`` replaced the Helmholtz kernel
        by the Laplace kernel; the measured distance to the exact complex sum is
        tens of percent, so the tag must say ``laplace_approx``.
        """
        targets, sources, charges = _cloud()
        k = 0.099 / source_extent(sources)
        pot, mode = fmm_helmholtz_matvec(targets, sources, charges, k=k,
                                         low_ka_asymptotic=True)
        assert mode == "laplace_approx"
        assert np.all(pot.imag == 0.0)
        err = _rel_err(pot, _analytic(targets, sources, charges, k))
        assert err > 0.05, "documented error floor of the asymptotic route"

    def test_opt_in_asymptotic_rejects_true_complex_charges(self):
        targets, sources, _ = _cloud()
        charges = np.ones(len(sources)) * (1.0 + 1.0j)
        k = 0.01 / source_extent(sources)
        with pytest.raises(ValueError, match="imaginary quadrature"):
            fmm_helmholtz_matvec(targets, sources, charges, k=k,
                                 low_ka_asymptotic=True)


# ---------------------------------------------------------------------------
# 3. Complex charges are preserved (no silent cast)
# ---------------------------------------------------------------------------

class TestComplexCharges:
    def test_complex_charges_are_not_truncated(self):
        targets, sources, _ = _cloud(seed=3)
        rng = np.random.default_rng(11)
        charges = rng.normal(size=len(sources)) + 1j * rng.normal(size=len(sources))
        k = 6.0
        with warnings.catch_warnings():
            warnings.simplefilter("error", np.exceptions.ComplexWarning)
            pot, mode = fmm_helmholtz_matvec(targets, sources, charges, k=k)
        assert mode == "complex_direct"
        assert _rel_err(pot, _analytic(targets, sources, charges, k)) <= 1e-12

    def test_real_charges_give_real_quadrature_of_the_same_kernel(self):
        targets, sources, charges = _cloud(seed=5)
        k = 2.5
        pot, _ = fmm_helmholtz_matvec(targets, sources, charges, k=k)
        assert _rel_err(pot, _analytic(targets, sources, charges, k)) <= 1e-12


# ---------------------------------------------------------------------------
# 4. Physics: the returned field satisfies the Helmholtz equation and radiates
# ---------------------------------------------------------------------------

class TestHelmholtzPhysics:
    def test_pde_residual_is_second_order(self):
        """(nabla^2 + k^2) Phi = 0 at a field point away from the sources."""
        rng = np.random.default_rng(7)
        sources = rng.normal(size=(60, 3))
        sources /= np.linalg.norm(sources, axis=1)[:, None]
        sources *= (0.5 * rng.random(60) ** (1.0 / 3.0))[:, None]
        charges = rng.normal(size=60)
        k = 3.0
        x0 = np.array([2.0, 1.5, -1.7])

        def residual(h: float) -> float:
            pts = [x0]
            for axis in range(3):
                for sign in (1.0, -1.0):
                    e = np.zeros(3)
                    e[axis] = sign * h
                    pts.append(x0 + e)
            phi, mode = fmm_helmholtz_matvec(np.array(pts), sources, charges, k=k)
            assert mode == "complex_direct"
            res = k * k * phi[0]
            for axis in range(3):
                res += (phi[1 + 2 * axis] - 2.0 * phi[0] + phi[2 + 2 * axis]) / h ** 2
            return float(abs(res) / (k * k * abs(phi[0])))

        coarse = residual(0.02)
        fine = residual(0.005)
        assert fine < 1e-3
        assert coarse / fine == pytest.approx(16.0, rel=0.25)

    def test_outgoing_phase_and_1_over_r_decay(self):
        """Below ka_max the field must be the *outgoing* wave exp(+i k r)/r."""
        k = 2.0
        radii = np.array([1.0, 2.0, 4.0, 8.0])
        targets = np.stack([radii, np.zeros_like(radii), np.zeros_like(radii)], axis=1)
        pot, mode = fmm_helmholtz_matvec(targets, np.zeros((1, 3)), np.array([1.0]), k=k)
        assert mode == "complex_direct"
        amplitude = np.abs(pot) * 4.0 * np.pi * radii
        np.testing.assert_allclose(amplitude, 1.0, rtol=1e-12)
        phase = np.angle(pot) - k * radii
        np.testing.assert_allclose(np.cos(phase), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. Guards and blocking invariance
# ---------------------------------------------------------------------------

class TestGuards:
    def test_single_source_has_zero_extent_and_is_exact(self):
        targets = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        assert source_extent(np.array([[0.1, 0.2, 0.3]])) == 0.0
        pot, mode = fmm_helmholtz_matvec(targets, np.array([[0.1, 0.2, 0.3]]),
                                         np.array([2.0]), k=9.0)
        assert mode == "complex_direct"
        r = np.linalg.norm(targets - np.array([[0.1, 0.2, 0.3]]), axis=1)
        ref = 2.0 * np.exp(1j * 9.0 * r) / (4.0 * np.pi * r)
        assert _rel_err(pot, ref) <= 1e-12

    def test_empty_inputs_return_zeros(self):
        targets = np.zeros((0, 3))
        pot, mode = fmm_helmholtz_matvec(targets, np.array([[0.0, 0.0, 0.0]]),
                                         np.array([1.0]), k=1.0)
        assert pot.shape == (0,) and pot.dtype == np.complex128
        targets = np.array([[1.0, 1.0, 1.0]])
        pot, mode = fmm_helmholtz_matvec(targets, np.zeros((0, 3)), np.zeros(0), k=1.0)
        assert pot.shape == (1,) and np.all(pot == 0.0)
        assert mode == "complex_direct"

    @pytest.mark.parametrize(
        "targets_shape,sources_shape,charges_len",
        [((3, 2), (4, 3), 4), ((3, 3), (4, 2), 4), ((3, 3), (4, 3), 5)],
    )
    def test_shape_validation(self, targets_shape, sources_shape, charges_len):
        with pytest.raises(ValueError):
            fmm_helmholtz_matvec(np.zeros(targets_shape), np.zeros(sources_shape),
                                 np.zeros(charges_len), k=1.0)

    def test_source_extent_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            source_extent(np.zeros((4, 2)))

    @pytest.mark.parametrize("max_pairs", [7, 10_000])
    def test_blocking_is_bitwise_invariant(self, max_pairs: int):
        targets, sources, charges = _cloud(n=25, seed=9)
        k = 3.0
        ref, _ = fmm_helmholtz_matvec(targets, sources, charges, k=k, max_pairs=10_000_000)
        blocked, _ = fmm_helmholtz_matvec(targets, sources, charges, k=k, max_pairs=max_pairs)
        assert np.array_equal(ref, blocked)