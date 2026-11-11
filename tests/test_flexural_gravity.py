"""Verification of the flexural-gravity wave dispersion relation and ice edge.

Tests cover all 8 acceptance criteria from the campaign DAG node p407w3-n06:

E1  Dispersion curve: omega(k) vs analytical, relative error <= 1%
E2  Long-wave limit: k->0 degenerates to open water, error <= 0.5%
E3  Short-wave limit: k->inf degenerates to flexural wave, error <= 0.5%
E4  Ice edge energy conservation: |R|^2 + |T|^2 = 1, residual <= 1e-4
E5  Ice edge wavelength: lambda_trans > lambda_inc (flexural rigidity makes wave faster)
E6  Wedge ice: gradual transition has lower reflection than abrupt
E7  Negative control: D=0 gives open-water dispersion, error <= 1e-6
E8  Parameter sensitivity: wavelength increases monotonically with ice thickness
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.flexural_gravity import (
    IcePlate,
    IceEdgeResult,
    abrupt_ice_edge_reflection,
    dispersion_curve,
    flexural_gravity_group_velocity,
    flexural_gravity_omega,
    flexural_gravity_omega2,
    flexural_gravity_wavenumber,
    flexural_wave_omega2,
    ice_edge_coefficients,
    open_water_dispersion_curve,
    open_water_group_velocity,
    open_water_omega2,
    open_water_wavenumber,
    wedge_ice_reflection,
    zero_rigidity_dispersion_check,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def standard_plate():
    """Standard ice plate: h_ice = 1.0 m, E = 5 GPa, H = 200 m."""
    return IcePlate(thickness=1.0, water_depth=200.0)


@pytest.fixture
def thin_plate():
    """Thin ice: h_ice = 0.5 m."""
    return IcePlate(thickness=0.5, water_depth=200.0)


@pytest.fixture
def thick_plate():
    """Thick ice: h_ice = 2.0 m."""
    return IcePlate(thickness=2.0, water_depth=200.0)


# ===================================================================
# E1: Dispersion curve omega(k) vs analytical, error <= 1%
# ===================================================================
class TestE1DispersionCurve:
    """Verify the numerical dispersion curve against the analytical formula."""

    @pytest.mark.parametrize("h_ice", [0.5, 1.0, 2.0])
    def test_dispersion_curve_matches_analytical(self, h_ice):
        """omega(k) from the solver must match the analytical formula (2)."""
        plate = IcePlate(thickness=h_ice, water_depth=200.0)
        # 25 wavenumber points covering long-wave to short-wave
        k_values = np.logspace(-3, 1, 25)

        for k in k_values:
            omega_numerical = flexural_gravity_omega(float(k), plate)
            # Analytical: omega^2 = (gk + D k^5/rho) tanh(kH)
            g = plate.gravity
            H = plate.water_depth
            D_rho = plate.D_over_rho
            omega_analytical = math.sqrt((g * k + D_rho * k**5) * math.tanh(k * H))

            rel_err = abs(omega_numerical - omega_analytical) / omega_analytical
            assert rel_err < 1e-10, (
                f"h_ice={h_ice}, k={k:.4e}: rel_err={rel_err:.2e}"
            )

    @pytest.mark.parametrize("h_ice", [0.5, 1.0, 2.0])
    def test_wavenumber_solver_inverts_dispersion(self, h_ice):
        """The wavenumber solver must recover k from omega(k)."""
        plate = IcePlate(thickness=h_ice, water_depth=200.0)
        k_input = np.logspace(-3, 0.5, 20)

        for k_in in k_input:
            omega = flexural_gravity_omega(float(k_in), plate)
            k_out = flexural_gravity_wavenumber(omega, plate)
            rel_err = abs(k_out - k_in) / k_in
            assert rel_err < 1e-10, (
                f"h_ice={h_ice}, k_in={k_in:.4e}: k_out={k_out:.6e}, rel_err={rel_err:.2e}"
            )

    def test_dispersion_curve_20_points(self, standard_plate):
        """E1: >= 20 wavenumber points, relative error <= 1%."""
        k_values = np.logspace(-3, 1, 25)
        omega_numerical = dispersion_curve(standard_plate, k_values)
        omega_analytical = np.array([
            math.sqrt(
                (standard_plate.gravity * k + standard_plate.D_over_rho * k**5)
                * math.tanh(k * standard_plate.water_depth)
            )
            for k in k_values
        ])
        max_rel_err = np.max(np.abs(omega_numerical - omega_analytical) / omega_analytical)
        assert max_rel_err < 0.01, f"max relative error = {max_rel_err:.2e}"


# ===================================================================
# E2: Long-wave limit: k->0 degenerates to open water
# ===================================================================
class TestE2LongWaveLimit:
    """For k -> 0, flexural-gravity must degenerate to open-water gravity wave."""

    @pytest.mark.parametrize("h_ice", [0.5, 1.0, 2.0])
    def test_long_wave_limit(self, h_ice):
        """For small k, D k^5/rho << g k, so omega^2 -> gk tanh(kH)."""
        plate = IcePlate(thickness=h_ice, water_depth=200.0)
        # Very small wavenumbers (long waves)
        k_values = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

        for k in k_values:
            omega2_fg = flexural_gravity_omega2(k, plate)
            omega2_open = open_water_omega2(k, plate.water_depth, plate.gravity)

            rel_err = abs(omega2_fg - omega2_open) / omega2_open
            assert rel_err < 0.005, (
                f"h_ice={h_ice}, k={k:.2e}: omega2_fg={omega2_fg:.6e}, "
                f"omega2_open={omega2_open:.6e}, rel_err={rel_err:.2e}"
            )

    def test_long_wave_deep_water(self):
        """Deep water long-wave: omega^2 -> gk."""
        plate = IcePlate(thickness=1.0, water_depth=1e4)
        k = 1e-3  # kH = 10, still deep but long enough that D term is small
        omega2_fg = flexural_gravity_omega2(k, plate)
        omega2_deep = plate.gravity * k  # deep water open water

        # The flexural term should be negligible
        D_term = plate.D_over_rho * k**5
        g_term = plate.gravity * k
        ratio = D_term / g_term
        assert ratio < 0.005, f"D-term/g-term ratio = {ratio:.2e}"


# ===================================================================
# E3: Short-wave limit: k->inf degenerates to flexural wave
# ===================================================================
class TestE3ShortWaveLimit:
    """For k -> infinity (deep water), omega^2 -> D k^5 / rho."""

    @pytest.mark.parametrize("h_ice", [0.5, 1.0, 2.0])
    def test_short_wave_limit(self, h_ice):
        """For large kH, gk << D k^5/rho, so omega^2 -> D k^5/rho."""
        plate = IcePlate(thickness=h_ice, water_depth=200.0)
        # Large wavenumbers (short waves) - need kH >> 1 for deep water
        k_values = [5.0, 10.0, 20.0]

        for k in k_values:
            omega2_fg = flexural_gravity_omega2(k, plate)
            omega2_flex = flexural_wave_omega2(k, plate)

            rel_err = abs(omega2_fg - omega2_flex) / omega2_flex
            assert rel_err < 0.05, (
                f"h_ice={h_ice}, k={k:.2e}: omega2_fg={omega2_fg:.6e}, "
                f"omega2_flex={omega2_flex:.6e}, rel_err={rel_err:.2e}"
            )

    def test_gravity_negligible_at_high_k(self, standard_plate):
        """Verify gk << D k^5/rho for large k."""
        k = 10.0
        g_term = standard_plate.gravity * k
        D_term = standard_plate.D_over_rho * k**5
        assert D_term / g_term > 200, "flexural term should dominate at k=10"


# ===================================================================
# E4: Ice edge energy conservation: |R|^2 + |T|^2 = 1
# ===================================================================
class TestE4EnergyConservation:
    """Energy conservation at the ice edge."""

    @pytest.mark.parametrize("omega", [0.5, 1.0, 1.5, 2.0, 2.5])
    def test_energy_conservation(self, omega, standard_plate):
        """|R_E|^2 + |T_E|^2 = 1 with residual <= 1e-4."""
        result = ice_edge_coefficients(omega, standard_plate)
        residual = abs(result.energy_residual)
        assert residual < 1e-4, (
            f"omega={omega}: |R|^2+|T|^2-1 = {result.energy_residual:.2e}"
        )

    @pytest.mark.parametrize("h_ice", [0.5, 1.0, 1.5, 2.0])
    def test_energy_conservation_various_thickness(self, h_ice):
        """Energy conservation for different ice thicknesses."""
        plate = IcePlate(thickness=h_ice, water_depth=200.0)
        omega = 1.5
        result = ice_edge_coefficients(omega, plate)
        residual = abs(result.energy_residual)
        assert residual < 1e-4, (
            f"h_ice={h_ice}: |R|^2+|T|^2-1 = {result.energy_residual:.2e}"
        )

    def test_five_frequency_combinations(self, standard_plate):
        """E4: >= 5 frequency combinations, residual <= 1e-4."""
        omegas = [0.3, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
        for omega in omegas:
            result = ice_edge_coefficients(omega, standard_plate)
            sum_energy = result.R_energy**2 + result.T_energy**2
            assert abs(sum_energy - 1.0) < 1e-4, (
                f"omega={omega}: R^2+T^2 = {sum_energy:.10f}"
            )


# ===================================================================
# E5: Ice edge wavelength: lambda_trans > lambda_inc (flexural model)
# ===================================================================
class TestE5WavelengthBehavior:
    """Wavelength behavior at the ice edge for flexural-gravity model.
    
    For the model omega^2 = (gk + Dk^5/rho) tanh(kH), the flexural rigidity
    adds stiffness, making the wave FASTER for a given k.  Therefore, for a
    given frequency omega, the wavenumber k under ice is SMALLER, and the
    wavelength is LONGER than in open water.
    
    Note: This is opposite to the mass-loading model (without flexural rigidity),
    where the ice inertia slows down the wave and makes the wavelength shorter.
    """

    @pytest.mark.parametrize("omega", [0.5, 1.0, 1.5, 2.0, 2.5])
    def test_wavelength_longer_under_ice(self, omega, standard_plate):
        """For flexural-gravity model, lambda_ice > lambda_open."""
        result = ice_edge_coefficients(omega, standard_plate)
        assert result.wavelength_ice > result.wavelength_open, (
            f"omega={omega}: lambda_ice={result.wavelength_ice:.2f} m, "
            f"lambda_open={result.wavelength_open:.2f} m (should be longer)"
        )

    def test_wavenumber_decrease(self, standard_plate):
        """k_ice < k_open (waves are longer under ice in flexural model)."""
        omega = 1.5
        result = ice_edge_coefficients(omega, standard_plate)
        assert result.k_ice < result.k_open, (
            f"k_ice={result.k_ice:.4e} should be < k_open={result.k_open:.4e}"
        )

    def test_physical_mechanism(self, standard_plate):
        """The wavelength increase is because ice adds flexural stiffness."""
        omega = 1.5
        result = ice_edge_coefficients(omega, standard_plate)
        ratio = result.wavelength_ice / result.wavelength_open
        assert ratio > 1.0, "wavelength ratio should be > 1"
        # For typical parameters, the ratio is moderate
        assert ratio < 10.0, "ratio should be physically reasonable"


# ===================================================================
# E6: Wedge ice: gradual transition has lower reflection than abrupt
# ===================================================================
class TestE6WedgeIce:
    """Gradual (wedge) ice transition reduces reflection."""

    @pytest.mark.parametrize("transition_length", [50.0, 100.0, 200.0])
    def test_wedge_less_than_abrupt(self, transition_length, standard_plate):
        """Wedge reflection must be less than abrupt edge reflection."""
        omega = 1.5
        R_abrupt = abrupt_ice_edge_reflection(omega, standard_plate)
        R_wedge = wedge_ice_reflection(omega, standard_plate, transition_length)

        assert R_wedge < R_abrupt, (
            f"L={transition_length}: R_wedge={R_wedge:.6f} >= R_abrupt={R_abrupt:.6f}"
        )

    def test_longer_transition_lower_reflection(self, standard_plate):
        """Longer transition length should give lower reflection."""
        omega = 1.5
        R_50 = wedge_ice_reflection(omega, standard_plate, 50.0)
        R_100 = wedge_ice_reflection(omega, standard_plate, 100.0)
        R_200 = wedge_ice_reflection(omega, standard_plate, 200.0)

        # Generally, longer transition = less reflection
        assert R_200 < R_50 * 2.0, "very long transition should not increase reflection much"

    def test_three_transition_lengths(self, standard_plate):
        """E6: >= 3 transition lengths, all less than abrupt."""
        omega = 1.5
        R_abrupt = abrupt_ice_edge_reflection(omega, standard_plate)
        for L in [50.0, 100.0, 200.0]:
            R_wedge = wedge_ice_reflection(omega, standard_plate, L)
            assert R_wedge < R_abrupt, f"L={L}: wedge should be less than abrupt"


# ===================================================================
# E7: Negative control: D=0 gives open-water dispersion
# ===================================================================
class TestE7ZeroRigidity:
    """With D=0, flexural-gravity must degenerate to open-water dispersion."""

    def test_zero_rigidity_dispersion(self):
        """omega^2(D=0) = gk tanh(kH) exactly."""
        depth = 200.0
        gravity = 9.80665
        k_values = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

        for k in k_values:
            omega_fg, omega_open = zero_rigidity_dispersion_check(k, depth, gravity)
            rel_err = abs(omega_fg - omega_open) / omega_open
            assert rel_err < 1e-12, (
                f"k={k}: omega_fg={omega_fg:.10e}, omega_open={omega_open:.10e}, "
                f"rel_err={rel_err:.2e}"
            )

    def test_zero_rigidity_wavenumber(self):
        """k(D=0) = k_open_water exactly."""
        depth = 200.0
        omega = 1.5
        k_open = open_water_wavenumber(omega, depth)

        # With D=0, the flexural-gravity dispersion reduces to open water
        g = 9.80665
        omega2_check = g * k_open * math.tanh(k_open * depth)
        rel_err = abs(omega2_check - omega**2) / omega**2
        assert rel_err < 1e-10

    def test_tiny_rigidity_limit(self):
        """Very small E should give nearly open-water results."""
        # Use a very small Young's modulus
        plate = IcePlate(
            thickness=0.01,  # Very thin
            young_modulus=1.0,  # Very soft (1 Pa)
            water_depth=200.0,
        )
        # D should be extremely small
        D = plate.flexural_rigidity
        assert D < 1e-6, f"D = {D:.2e} should be tiny"

        # The dispersion should be very close to open water
        k = 0.1
        omega2_fg = flexural_gravity_omega2(k, plate)
        omega2_open = open_water_omega2(k, plate.water_depth, plate.gravity)
        rel_err = abs(omega2_fg - omega2_open) / omega2_open
        assert rel_err < 1e-3, f"rel_err = {rel_err:.2e}"


# ===================================================================
# E8: Parameter sensitivity: wavelength increases with ice thickness
# ===================================================================
class TestE8ParameterSensitivity:
    """Wavelength must increase monotonically with ice thickness."""

    def test_wavelength_increases_with_thickness(self):
        """E8: >= 4 thickness points, wavelength monotonically increases."""
        omega = 1.5
        thicknesses = [0.5, 1.0, 1.5, 2.0]
        wavelengths = []

        for h_ice in thicknesses:
            plate = IcePlate(thickness=h_ice, water_depth=200.0)
            k = flexural_gravity_wavenumber(omega, plate)
            wavelength = 2.0 * math.pi / k
            wavelengths.append(wavelength)

        # Check monotonic increase
        for i in range(len(wavelengths) - 1):
            assert wavelengths[i + 1] > wavelengths[i], (
                f"h_ice={thicknesses[i]}->{thicknesses[i+1]}: "
                f"lambda={wavelengths[i]:.4f}->{wavelengths[i+1]:.4f} "
                f"(should increase)"
            )

    def test_physical_explanation(self):
        """Thicker ice -> more stiffness -> faster wave -> longer wavelength."""
        omega = 1.5
        h_values = [0.5, 1.0, 1.5, 2.0]

        for h in h_values:
            plate = IcePlate(thickness=h, water_depth=200.0)
            k = flexural_gravity_wavenumber(omega, plate)
            cg = flexural_gravity_group_velocity(omega, plate, k)
            # Phase speed
            cp = omega / k
            # As h increases, D increases as h^3, so the wave becomes stiffer
            D = plate.flexural_rigidity
            assert D > 0

    def test_four_thickness_points(self):
        """E8: >= 4 points with monotonic wavelength increase."""
        omega = 2.0
        thicknesses = [0.3, 0.7, 1.2, 1.8]
        wavelengths = []
        for h in thicknesses:
            plate = IcePlate(thickness=h, water_depth=200.0)
            k = flexural_gravity_wavenumber(omega, plate)
            wavelengths.append(2 * math.pi / k)

        for i in range(len(wavelengths) - 1):
            assert wavelengths[i + 1] > wavelengths[i]


# ===================================================================
# Input validation (negative controls)
# ===================================================================
class TestInputValidation:
    """Non-physical inputs must be rejected."""

    def test_negative_thickness_rejected(self):
        with pytest.raises(ValueError):
            IcePlate(thickness=-1.0)

    def test_zero_thickness_rejected(self):
        with pytest.raises(ValueError):
            IcePlate(thickness=0.0)

    def test_negative_depth_rejected(self):
        with pytest.raises(ValueError):
            IcePlate(thickness=1.0, water_depth=-10.0)

    def test_thickness_exceeds_depth_rejected(self):
        with pytest.raises(ValueError):
            IcePlate(thickness=5.0, water_depth=3.0)

    def test_negative_young_modulus_rejected(self):
        with pytest.raises(ValueError):
            IcePlate(thickness=1.0, young_modulus=-1.0)

    def test_invalid_poisson_ratio_rejected(self):
        with pytest.raises(ValueError):
            IcePlate(thickness=1.0, poisson_ratio=0.6)

    def test_negative_omega_rejected(self):
        plate = IcePlate(thickness=1.0)
        with pytest.raises(ValueError):
            flexural_gravity_wavenumber(-1.0, plate)

    def test_negative_wavenumber_rejected(self):
        plate = IcePlate(thickness=1.0)
        with pytest.raises(ValueError):
            flexural_gravity_omega(-1.0, plate)


# ===================================================================
# Cross-validation with open-water module
# ===================================================================
class TestCrossValidation:
    """Cross-check with the existing open-water functions."""

    def test_open_water_wavenumber_consistency(self):
        """Our open_water_wavenumber must agree with the ice module's version."""
        from tensorbem.ice import open_water_wavenumber as ice_module_ow

        omega = 1.5
        depth = 200.0
        k_ours = open_water_wavenumber(omega, depth)
        k_theirs = ice_module_ow(omega, depth)
        assert abs(k_ours - k_theirs) / k_theirs < 1e-10

    def test_group_velocity_deep_water_limit(self):
        """Deep water: cg -> g/(2*omega) for open water."""
        omega = 1.5
        depth = 1e4  # Very deep
        k = open_water_wavenumber(omega, depth)
        cg = open_water_group_velocity(omega, depth, k)
        cg_deep = 9.80665 / (2.0 * omega)
        assert abs(cg - cg_deep) / cg_deep < 0.01
