"""
Pytest tests for Wigley III / spheroid frequency-domain verification.

Tests verify:
1. Radiation forces (added mass and damping) are physically reasonable
2. Diffraction forces satisfy symmetry properties
3. RAO approaches correct limits
4. Energy conservation (damping matrix symmetry and positivity)
5. Mesh convergence (basic check)
"""
import sys
from pathlib import Path

# repository-relative import root (was an absolute path to another agent worktree)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from src.tensorbem.geometry3d import spheroid_hull_mesh
from src.tensorbem.free_surface_solver3d import (
    solve_diffraction_3d,
    solve_radiation_3d,
)
from src.tensorbem.rao import compute_rao_at_frequency
from src.tensorbem.hydrostatic import compute_hydrostatic_restoring_matrix

# Physical parameters
L, B, T = 3.0, 0.3, 0.1875
rho = 1025.0
g = 9.80665
target_volume = 0.091631
a, b = L / 2, B / 2
c = target_volume / ((2.0/3.0) * np.pi * a * b)
mass = rho * target_volume


@pytest.fixture(scope='module')
def mesh():
    """Generate spheroid mesh."""
    return spheroid_hull_mesh(semi_axes=(a, b, c), subdivisions=2)


@pytest.fixture(scope='module')
def hydro_props(mesh):
    """Compute hydrostatic properties."""
    cg = np.array([0.0, 0.0, -0.5*T])
    return compute_hydrostatic_restoring_matrix(
        mesh, mass=mass, center_of_gravity=cg,
        density=rho, gravity=g
    )


class TestRadiationForces:
    """Test radiation force computations."""
    
    def test_added_mass_positive(self, mesh):
        """Added mass should be positive for all frequencies."""
        omega = 1.5
        nu = omega**2 / g
        rad = solve_radiation_3d(mesh, nu, modes=[2], density=rho, gravity=g)
        assert rad.added_mass[0, 0] > 0, "Heave added mass must be positive"
    
    def test_damping_positive(self, mesh):
        """Radiation damping should be non-negative."""
        omega = 1.5
        nu = omega**2 / g
        rad = solve_radiation_3d(mesh, nu, modes=[2], density=rho, gravity=g)
        assert rad.damping[0, 0] >= 0, "Heave damping must be non-negative"
    
    def test_damping_matrix_symmetric(self, mesh):
        """Damping matrix should be symmetric (energy conservation)."""
        omega = 1.5
        nu = omega**2 / g
        rad = solve_radiation_3d(mesh, nu, modes=[0, 1, 2, 3, 4, 5], density=rho, gravity=g)
        B = rad.damping
        sym_error = np.max(np.abs(B - B.T))
        assert sym_error < 1e-2, f"Damping matrix symmetry error {sym_error} too large"
    
    def test_damping_matrix_positive_diagonal(self, mesh):
        """Damping matrix diagonal should be non-negative."""
        omega = 1.5
        nu = omega**2 / g
        rad = solve_radiation_3d(mesh, nu, modes=[0, 1, 2, 3, 4, 5], density=rho, gravity=g)
        B = rad.damping
        assert np.all(np.diag(B) >= -1e-10), "Damping matrix must have non-negative diagonal"
    
    def test_added_mass_frequency_dependence(self, mesh):
        """Added mass should vary with frequency."""
        omegas = [0.5, 1.5, 2.5]
        A33_values = []
        for omega in omegas:
            nu = omega**2 / g
            rad = solve_radiation_3d(mesh, nu, modes=[2], density=rho, gravity=g)
            A33_values.append(rad.added_mass[0, 0])
        
        # Check that values are not all the same
        assert np.std(A33_values) > 1.0, "Added mass should vary with frequency"


class TestDiffractionForces:
    """Test diffraction force computations."""
    
    def test_excitation_force_finite(self, mesh):
        """Excitation forces should be finite."""
        omega = 1.5
        nu = omega**2 / g
        diff = solve_diffraction_3d(mesh, nu, wave_angle=0.0, density=rho, gravity=g)
        assert np.all(np.isfinite(diff.excitation_force)), "Excitation forces must be finite"
    
    def test_beam_seas_symmetry(self, mesh):
        """In beam seas (β=90°), surge and pitch excitation should be small."""
        omega = 1.5
        nu = omega**2 / g
        diff = solve_diffraction_3d(mesh, nu, wave_angle=np.pi/2, density=rho, gravity=g)
        F_exc = diff.excitation_force
        
        # In beam seas, surge (F1) and pitch (F5) should be small due to symmetry
        F1_norm = np.abs(F_exc[0]) / np.abs(F_exc[2])
        F5_norm = np.abs(F_exc[4]) / np.abs(F_exc[2])
        
        assert F1_norm < 0.1, f"Surge excitation in beam seas too large: {F1_norm}"
        assert F5_norm < 0.1, f"Pitch excitation in beam seas too large: {F5_norm}"
    
    def test_head_seas_symmetry(self, mesh):
        """In head seas (β=0°), sway and yaw excitation should be small."""
        omega = 1.5
        nu = omega**2 / g
        diff = solve_diffraction_3d(mesh, nu, wave_angle=0.0, density=rho, gravity=g)
        F_exc = diff.excitation_force
        
        # In head seas, sway (F2) and yaw (F6) should be small due to symmetry
        F2_norm = np.abs(F_exc[1]) / np.abs(F_exc[2])
        F6_norm = np.abs(F_exc[5]) / np.abs(F_exc[2])
        
        assert F2_norm < 0.1, f"Sway excitation in head seas too large: {F2_norm}"
        assert F6_norm < 0.1, f"Yaw excitation in head seas too large: {F6_norm}"


class TestRAO:
    """Test RAO computations."""
    
    def test_heave_rao_low_frequency_limit(self, mesh, hydro_props):
        """Heave RAO should approach 1.0 at low frequency (perfect tracking)."""
        omega = 0.5
        cg = np.array([0.0, 0.0, -0.5*T])
        rao = compute_rao_at_frequency(
            mesh, omega, wave_angle=0.0, mass=mass,
            center_of_gravity=cg, hydrostatic_props=hydro_props,
            density=rho, gravity=g
        )
        
        rao3 = rao.rao_amplitude[2]
        assert 0.9 < rao3 < 1.1, f"Heave RAO at low freq should be ~1.0, got {rao3}"
    
    def test_rao_finite(self, mesh, hydro_props):
        """RAO should be finite for all frequencies."""
        omega = 1.5
        cg = np.array([0.0, 0.0, -0.5*T])
        rao = compute_rao_at_frequency(
            mesh, omega, wave_angle=0.0, mass=mass,
            center_of_gravity=cg, hydrostatic_props=hydro_props,
            density=rho, gravity=g
        )
        
        assert np.all(np.isfinite(rao.rao_amplitude)), "RAO must be finite"
        assert np.all(rao.rao_amplitude >= 0), "RAO amplitude must be non-negative"
    
    def test_rao_impedance_residual(self, mesh, hydro_props):
        """RAO impedance residual should be small."""
        omega = 1.5
        cg = np.array([0.0, 0.0, -0.5*T])
        rao = compute_rao_at_frequency(
            mesh, omega, wave_angle=0.0, mass=mass,
            center_of_gravity=cg, hydrostatic_props=hydro_props,
            density=rho, gravity=g
        )
        
        assert rao.impedance_residual < 1e-6, f"Impedance residual {rao.impedance_residual} too large"


class TestMeshConvergence:
    """Basic mesh convergence check."""
    
    def test_volume_convergence(self):
        """Mesh volume should converge with refinement."""
        volumes = []
        for sub in [1, 2, 3]:
            mesh = spheroid_hull_mesh(semi_axes=(a, b, c), subdivisions=sub)
            volumes.append(mesh.volume)
        
        # Check that volume converges (difference decreases with refinement)
        diff_12 = abs(volumes[1] - volumes[0])
        diff_23 = abs(volumes[2] - volumes[1])
        assert diff_23 < diff_12, "Volume should converge with mesh refinement"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
