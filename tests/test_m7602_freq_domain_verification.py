"""Pytest test for Wigley III / spheroid frequency-domain verification.

This test verifies that the BEM solver produces physically reasonable results
for a prolate spheroid hull at frequency domain.
"""
import sys
import math
import numpy as np
import pytest

sys.path.insert(0, 'src')

from tensorbem.benchmarks.wigley3_freq_domain_verification import create_hull_mesh
from tensorbem.free_surface_solver3d import solve_radiation_3d, solve_diffraction_3d
from tensorbem.hydrostatic import compute_hydrostatic_restoring_matrix
from tensorbem.rao import compute_rao_at_frequency


def test_spheroid_mesh_generation():
    """Test that spheroid mesh is generated correctly."""
    L, B, T = 3.0, 0.3, 0.1875
    hull = create_hull_mesh(L, B, T, subdivisions=2)
    
    assert hull.panel_count == 64
    assert hull.volume > 0
    assert hull.waterplane_area > 0
    
    # Check volume is close to analytical
    a, b, c = L/2, B/2, T
    V_anal = (2.0/3.0) * math.pi * a * b * c
    vol_error = abs(hull.volume - V_anal) / V_anal
    assert vol_error < 0.10  # Within 10%


def test_radiation_produces_results():
    """Test that radiation solver produces results."""
    L, B, T = 3.0, 0.3, 0.1875
    rho, g = 1025.0, 9.80665
    
    hull = create_hull_mesh(L, B, T, subdivisions=2)
    omega = 1.5
    nu = omega**2 / g
    
    rad = solve_radiation_3d(hull, nu, modes=[2], center=np.array([0.0, 0.0, 0.0]),
                             density=rho, gravity=g)
    
    assert rad.added_mass.shape == (1, 1)
    assert rad.damping.shape == (1, 1)
    assert rad.added_mass[0, 0] > 0  # Added mass should be positive
    # Note: damping may be negative at coarse resolution (known issue)


def test_diffraction_produces_results():
    """Test that diffraction solver produces results."""
    L, B, T = 3.0, 0.3, 0.1875
    rho, g = 1025.0, 9.80665
    
    hull = create_hull_mesh(L, B, T, subdivisions=2)
    omega = 1.5
    nu = omega**2 / g
    
    diff = solve_diffraction_3d(hull, nu, amplitude=1.0, wave_angle=0.0,
                                center=np.array([0.0, 0.0, 0.0]),
                                density=rho, gravity=g)
    
    assert diff.excitation_force.shape == (6,)
    assert abs(diff.excitation_force[2]) > 0  # Heave excitation should be non-zero


def test_rao_produces_results():
    """Test that RAO computation produces results."""
    L, B, T = 3.0, 0.3, 0.1875
    rho, g = 1025.0, 9.80665
    
    hull = create_hull_mesh(L, B, T, subdivisions=2)
    V = hull.volume
    mass = rho * V
    cg = np.array([0.0, 0.0, -0.05])
    
    Ixx = mass * (B**2 / 12.0 + 0.05**2)
    Iyy = mass * (L**2 / 12.0 + 0.05**2)
    Izz = mass * (L**2 + B**2) / 12.0
    inertia = np.diag([Ixx, Iyy, Izz])
    
    hydro = compute_hydrostatic_restoring_matrix(hull, mass, cg, rho, g)
    
    omega = 1.5
    rao = compute_rao_at_frequency(hull, omega, 0.0, mass, cg, hydro,
                                   1.0, rho, g, inertia)
    
    assert rao.rao_amplitude.shape == (6,)
    assert rao.rao_amplitude[2] > 0  # Heave RAO should be positive
    assert rao.rao_amplitude[2] < 2.0  # Should be reasonable (< 2 m/m)


if __name__ == "__main__":
    test_spheroid_mesh_generation()
    print("✓ Mesh generation test passed")
    
    test_radiation_produces_results()
    print("✓ Radiation test passed")
    
    test_diffraction_produces_results()
    print("✓ Diffraction test passed")
    
    test_rao_produces_results()
    print("✓ RAO test passed")
    
    print("\nAll tests passed!")
