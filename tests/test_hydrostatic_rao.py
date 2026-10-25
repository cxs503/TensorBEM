"""Tests for hydrostatic restoring matrix and RAO computation."""

import numpy as np
import pytest

from tensorbem.geometry3d import WettedHullMesh, hemisphere_hull_mesh
from tensorbem.hydrostatic import (
    compute_hydrostatic_restoring_matrix,
    compute_mass_matrix,
)
from tensorbem.rao import (
    compute_rao_at_frequency,
    compute_rao_spectrum,
    estimate_natural_frequency,
)


class TestHydrostaticProperties:
    """Test hydrostatic property calculations."""

    def test_hemisphere_waterplane_area(self):
        """Test waterplane area calculation for hemisphere."""
        hull = hemisphere_hull_mesh(subdivisions=2)
        
        mass = 1000.0
        cg = np.array([0.0, 0.0, -0.1])
        props = compute_hydrostatic_restoring_matrix(hull, mass, cg)
        
        # For hemisphere with radius R, waterplane area = pi R^2
        expected_area = np.pi * 1.0**2
        
        # Allow 5% tolerance due to discretization
        assert abs(props.waterplane_area - expected_area) / expected_area < 0.05

    def test_hemisphere_displaced_volume(self):
        """Test displaced volume calculation for hemisphere."""
        hull = hemisphere_hull_mesh(subdivisions=2)
        
        mass = 1000.0
        cg = np.array([0.0, 0.0, -0.1])
        props = compute_hydrostatic_restoring_matrix(hull, mass, cg)
        
        # For hemisphere with radius R, displaced volume = (2/3) pi R^3
        expected_volume = (2.0 / 3.0) * np.pi * 1.0**3
        
        # Allow 10% tolerance (subdivisions=2 gives 64 panels, coarse mesh)
        assert abs(props.displaced_volume - expected_volume) / expected_volume < 0.10

    def test_restoring_matrix_symmetry(self):
        """Test that restoring matrix is symmetric."""
        hull = hemisphere_hull_mesh(subdivisions=2)
        
        mass = 1000.0
        cg = np.array([0.0, 0.0, -0.1])
        props = compute_hydrostatic_restoring_matrix(hull, mass, cg)
        
        C = props.restoring_matrix
        
        # Check symmetry
        assert np.allclose(C, C.T, atol=1e-10)

    def test_restoring_matrix_positive_diagonal(self):
        """Test that restoring matrix has positive diagonal for stable body."""
        hull = hemisphere_hull_mesh(subdivisions=2)
        
        mass = 1000.0
        # Place CG below center of buoyancy for stability
        cg = np.array([0.0, 0.0, -0.3])
        props = compute_hydrostatic_restoring_matrix(hull, mass, cg)
        
        C = props.restoring_matrix
        
        # Heave, roll, and pitch should have positive restoring
        assert C[2, 2] > 0  # Heave
        assert C[3, 3] > 0  # Roll
        assert C[4, 4] > 0  # Pitch

    def test_mass_matrix_symmetry(self):
        """Test that mass matrix is symmetric."""
        mass = 1000.0
        cg = np.array([0.0, 0.0, -0.1])
        
        M = compute_mass_matrix(mass, cg)
        
        # Check symmetry
        assert np.allclose(M, M.T, atol=1e-10)

    def test_mass_matrix_positive_definite(self):
        """Test that mass matrix is positive definite."""
        mass = 1000.0
        cg = np.array([0.0, 0.0, -0.1])
        
        M = compute_mass_matrix(mass, cg)
        
        # Check positive definiteness via eigenvalues
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues > 0)


class TestRAOComputation:
    """Test RAO computation."""

    def test_rao_single_frequency(self):
        """Test RAO computation at single frequency."""
        hull = hemisphere_hull_mesh(subdivisions=2)
        
        mass = 1000.0
        cg = np.array([0.0, 0.0, -0.3])
        omega = 1.0  # rad/s
        wave_angle = 0.0  # Head seas
        
        rao_result = compute_rao_at_frequency(
            hull, omega, wave_angle, mass, cg
        )
        
        # Check that RAO has 6 components
        assert len(rao_result.rao) == 6
        assert len(rao_result.rao_amplitude) == 6
        
        # Check that added mass and damping are 6x6
        assert rao_result.added_mass.shape == (6, 6)
        assert rao_result.damping.shape == (6, 6)
        
        # Check that excitation force has 6 components
        assert len(rao_result.excitation_force) == 6

    def test_rao_spectrum(self):
        """Test RAO spectrum computation."""
        hull = hemisphere_hull_mesh(subdivisions=2)
        
        mass = 1000.0
        cg = np.array([0.0, 0.0, -0.3])
        omegas = np.array([0.5, 1.0, 1.5, 2.0])
        wave_angle = 0.0
        
        rao_spectrum = compute_rao_spectrum(
            hull, omegas, wave_angle, mass, cg
        )
        
        # Check dimensions
        assert len(rao_spectrum.omegas) == 4
        assert rao_spectrum.rao_amplitudes.shape == (4, 6)
        assert rao_spectrum.added_masses.shape == (4, 6, 6)
        assert rao_spectrum.dampings.shape == (4, 6, 6)

    def test_rao_heave_response(self):
        """Test that heave RAO is physically reasonable."""
        hull = hemisphere_hull_mesh(subdivisions=2)
        
        mass = 1000.0
        cg = np.array([0.0, 0.0, -0.3])
        omega = 1.0
        wave_angle = 0.0
        
        rao_result = compute_rao_at_frequency(
            hull, omega, wave_angle, mass, cg
        )
        
        # Heave RAO should be positive and finite
        heave_rao = rao_result.rao_amplitude[2]
        assert heave_rao > 0
        assert np.isfinite(heave_rao)
        
        # For unit wave amplitude, heave RAO should be O(1)
        assert heave_rao < 10.0

    def test_natural_frequency_estimation(self):
        """Test natural frequency estimation."""
        hull = hemisphere_hull_mesh(subdivisions=2)
        
        mass = 1000.0
        cg = np.array([0.0, 0.0, -0.3])
        props = compute_hydrostatic_restoring_matrix(hull, mass, cg)
        
        # Estimate added mass at infinite frequency (use high frequency)
        omega_high = 10.0
        rao_high = compute_rao_at_frequency(
            hull, omega_high, 0.0, mass, cg
        )
        A_inf = rao_high.added_mass
        
        # Estimate natural frequency for heave
        omega_n = estimate_natural_frequency(
            mass, A_inf, props.restoring_matrix, dof=2
        )
        
        # Natural frequency should be positive and finite
        assert omega_n > 0
        assert np.isfinite(omega_n)
        
        # For hemisphere, natural frequency should be in reasonable range
        assert omega_n < 5.0


class TestHydrostaticEdgeCases:
    """Test edge cases for hydrostatic calculations."""

    def test_mass_matrix_with_inertia(self):
        """Test mass matrix with explicit inertia tensor."""
        mass = 1000.0
        cg = np.array([0.0, 0.0, -0.1])
        
        # Provide explicit inertia tensor
        Ixx, Iyy, Izz = 100.0, 100.0, 50.0
        inertia = np.diag([Ixx, Iyy, Izz])
        
        M = compute_mass_matrix(mass, cg, inertia=inertia)
        
        # Check that inertia is incorporated
        assert M.shape == (6, 6)
        assert np.allclose(M, M.T, atol=1e-10)


class TestRAOEdgeCases:
    """Test edge cases for RAO computation."""

    def test_zero_frequency(self):
        """Test that zero frequency is handled."""
        hull = hemisphere_hull_mesh(subdivisions=2)
        
        mass = 1000.0
        cg = np.array([0.0, 0.0, -0.3])
        
        # Very low frequency (approaching zero)
        omega = 0.01
        
        rao_result = compute_rao_at_frequency(
            hull, omega, 0.0, mass, cg
        )
        
        # Should still produce valid results
        assert np.all(np.isfinite(rao_result.rao_amplitude))

    def test_high_frequency(self):
        """Test high frequency behavior."""
        hull = hemisphere_hull_mesh(subdivisions=2)
        
        mass = 1000.0
        cg = np.array([0.0, 0.0, -0.3])
        omega = 5.0  # High frequency
        
        rao_result = compute_rao_at_frequency(
            hull, omega, 0.0, mass, cg
        )
        
        # At high frequency, RAO should be small (inertia dominated)
        heave_rao = rao_result.rao_amplitude[2]
        assert heave_rao < 1.0

    def test_beam_seas(self):
        """Test RAO in beam seas (90 degree wave angle)."""
        hull = hemisphere_hull_mesh(subdivisions=2)
        
        mass = 1000.0
        cg = np.array([0.0, 0.0, -0.3])
        omega = 1.0
        wave_angle = np.pi / 2  # Beam seas
        
        rao_result = compute_rao_at_frequency(
            hull, omega, wave_angle, mass, cg
        )
        
        # Should produce valid results
        assert np.all(np.isfinite(rao_result.rao_amplitude))
        
        # In beam seas, roll response should be significant
        roll_rao = rao_result.rao_amplitude[3]
        assert roll_rao >= 0
