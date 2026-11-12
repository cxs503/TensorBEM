"""Cross-validation of TensorBEM propeller vortex-lattice against Wageningen B-series.

This test bridges two independent propeller models in TensorBEM:

1. **Vortex-lattice BEM** (`tensorbem.propeller`) — first-principles lifting-surface
   method with horseshoe vortices, helical wake, and ITTC-1978 viscous correction.
2. **Wageningen B-series regression** (`tensorbem.wageningen`) — empirical 39/47-term
   polynomial fit to NSMB open-water test data (Oosterveld & van Oossanen, 1975).

Sign convention: thrust > 0 is positive thrust, torque < 0 is resisting.
K_T = T / (rho * n^2 * D^4), K_Q = |Q| / (rho * n^2 * D^5).
"""

from __future__ import annotations

import pytest

from tensorbem.propeller import BladeGeometry, open_water_point
from tensorbem.wageningen import B4Series

RHO = 1025.0


def _bem_to_kt_kq(result, diameter=1.0, n=10.0):
    kt = result.thrust / (RHO * n**2 * diameter**4)
    kq = abs(result.torque) / (RHO * n**2 * diameter**5)
    return kt, kq


# Module-scoped fixtures
@pytest.fixture(scope="module")
def default_geo():
    return BladeGeometry(n_radial=12, n_chord=1)


@pytest.fixture(scope="module")
def wag_default():
    return B4Series(pitch_ratio=0.95, expanded_area_ratio=0.80, blades=5)


@pytest.fixture(scope="module")
def wag_b470():
    return B4Series(pitch_ratio=0.70, expanded_area_ratio=0.70, blades=4)


class TestBEMPhysicalSanity:
    @pytest.mark.parametrize("J", [0.0, 0.3, 0.6])
    def test_kt_positive(self, J, default_geo):
        result = open_water_point(default_geo, J, wake_model="geometric",
                                  wake_length=1.0, wake_steps=24)
        kt, _ = _bem_to_kt_kq(result)
        assert kt > 0, f"K_T must be positive at J={J}, got {kt}"

    @pytest.mark.parametrize("J", [0.0, 0.3, 0.6])
    def test_kq_positive(self, J, default_geo):
        result = open_water_point(default_geo, J, wake_model="geometric",
                                  wake_length=1.0, wake_steps=24)
        _, kq = _bem_to_kt_kq(result)
        assert kq > 0, f"K_Q must be positive at J={J}, got {kq}"


class TestBEMvsWageningenOrderOfMagnitude:
    @pytest.mark.parametrize("J", [0.3, 0.5, 0.7])
    def test_kt_same_order(self, J, default_geo, wag_default):
        result = open_water_point(default_geo, J, wake_model="geometric",
                                  wake_length=1.0, wake_steps=24)
        kt_bem, _ = _bem_to_kt_kq(result)
        kt_wag = wag_default.kt(J)
        ratio = kt_bem / kt_wag if kt_wag > 0 else float('inf')
        assert 0.1 < ratio < 10.0, (
            f"K_T ratio BEM/Wag at J={J} = {ratio:.3f} "
            f"(BEM={kt_bem:.4f}, Wag={kt_wag:.4f})"
        )

    @pytest.mark.parametrize("J", [0.3, 0.5, 0.7])
    def test_kq_same_order(self, J, default_geo, wag_default):
        result = open_water_point(default_geo, J, wake_model="geometric",
                                  wake_length=1.0, wake_steps=24)
        _, kq_bem = _bem_to_kt_kq(result)
        kq_wag = wag_default.kq(J)
        ratio = kq_bem / kq_wag if kq_wag > 0 else float('inf')
        assert 0.1 < ratio < 10.0, (
            f"K_Q ratio BEM/Wag at J={J} = {ratio:.3f} "
            f"(BEM={kq_bem:.4f}, Wag={kq_wag:.4f})"
        )


class TestBEMPhysicalTrends:
    def test_kt_decreases_with_J(self, default_geo):
        kts = []
        for J in [0.0, 0.3, 0.6, 0.9]:
            result = open_water_point(default_geo, J, wake_model="geometric",
                                      wake_length=1.0, wake_steps=24)
            kt, _ = _bem_to_kt_kq(result)
            kts.append(kt)
        for i in range(len(kts) - 1):
            assert kts[i] > kts[i + 1], (
                f"K_T should decrease with J: {kts[i]:.4f} → {kts[i+1]:.4f}"
            )

    def test_kq_decreases_with_J(self, default_geo):
        kqs = []
        for J in [0.0, 0.3, 0.6, 0.9]:
            result = open_water_point(default_geo, J, wake_model="geometric",
                                      wake_length=1.0, wake_steps=24)
            _, kq = _bem_to_kt_kq(result)
            kqs.append(kq)
        for i in range(len(kqs) - 1):
            assert kqs[i] > kqs[i + 1], (
                f"K_Q should decrease with J: {kqs[i]:.4f} → {kqs[i+1]:.4f}"
            )


class TestBEMMeshConvergence:
    def test_kt_converges_with_radial_resolution(self):
        J = 0.7
        kts = []
        for n_radial in [12, 16, 24]:
            geo = BladeGeometry(n_radial=n_radial, n_chord=1)
            result = open_water_point(geo, J, wake_model="geometric",
                                      wake_length=1.0, wake_steps=24)
            kt, _ = _bem_to_kt_kq(result)
            kts.append(kt)
        rel_change = abs(kts[-1] - kts[0]) / kts[0] if kts[0] != 0 else 0
        assert rel_change < 0.10, (
            f"K_T relative change 12→24 panels = {rel_change*100:.1f}% "
            f"(12:{kts[0]:.4f}, 16:{kts[1]:.4f}, 24:{kts[2]:.4f})"
        )


class TestWageningenB470Anchors:
    def test_b470_zero_thrust_near_pitch(self, wag_b470):
        j0 = wag_b470.zero_thrust_advance_ratio()
        pd = 0.70
        ratio = j0 / pd
        assert 1.04 < ratio < 1.10, (
            f"B4-70 J0/(P/D) = {ratio:.3f}, expected [1.04, 1.10]"
        )

    def test_b470_efficiency_peak(self, wag_b470):
        eta_max = max(wag_b470.eta0(j) for j in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        assert 0.5 < eta_max < 0.8, f"B4-70 peak efficiency {eta_max:.3f} outside [0.5, 0.8]"
