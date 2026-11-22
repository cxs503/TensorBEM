"""Tests for Wigley III Cummins memory kernel (6DOF).

Verifies:
1. Radiation sweep produces valid added mass and damping matrices.
2. Memory kernel matrix is computed correctly.
3. Kernel matrix has correct shape and decays.
4. Sweep validation passes.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensorbem.wigley3_freq import (
    WIGLEY3_B,
    WIGLEY3_CB,
    WIGLEY3_C,
    WIGLEY3_L,
    WIGLEY3_T,
    wigley3_hydrostatic_reference,
    wigley3_wetted_hull_mesh,
)
from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem.memory_kernel_matrix import (
    coupled_memory_kernels,
    validate_coupled_sweep,
)


@pytest.fixture
def mesh():
    """Wigley III hull mesh (coarse for speed)."""
    return wigley3_wetted_hull_mesh(
        L=WIGLEY3_L,
        B=WIGLEY3_B,
        T=WIGLEY3_T,
        c=WIGLEY3_C,
        n_sections=5,  # Very coarse mesh for speed
        n_levels=3,
    )


@pytest.fixture
def omega():
    """Angular frequencies for radiation sweep."""
    return np.linspace(0.5, 2.0, 8)


@pytest.fixture
def radiation_sweep(mesh, omega):
    """Compute radiation sweep."""
    n_freq = len(omega)
    added_mass_all = np.zeros((n_freq, 6, 6))
    damping_all = np.zeros((n_freq, 6, 6))

    for i, w in enumerate(omega):
        nu = w**2 / 9.80665
        result = solve_radiation_3d(
            mesh,
            nu,
            density=1025.0,
            gravity=9.80665,
            chief_points="auto",
        )
        added_mass_all[i] = result.added_mass
        damping_all[i] = result.damping

    return omega, added_mass_all, damping_all


def test_radiation_sweep_symmetry(radiation_sweep):
    """Test that added mass and damping matrices are symmetric."""
    omega, added_mass, damping = radiation_sweep
    for i in range(len(omega)):
        # Use relative tolerance based on max entry, as in validate_coupled_sweep
        scale = max(float(np.max(np.abs(added_mass[i]))), 1e-300)
        asymmetry = float(np.max(np.abs(added_mass[i] - added_mass[i].T))) / scale
        assert asymmetry < 5e-3, (
            f"Added mass not symmetric at omega={omega[i]}: "
            f"max |A_ij - A_ji| / max|A| = {asymmetry}"
        )
        scale = max(float(np.max(np.abs(damping[i]))), 1e-300)
        asymmetry = float(np.max(np.abs(damping[i] - damping[i].T))) / scale
        assert asymmetry < 5e-3, (
            f"Damping not symmetric at omega={omega[i]}: "
            f"max |B_ij - B_ji| / max|B| = {asymmetry}"
        )


def test_radiation_sweep_passivity(radiation_sweep):
    """Test that damping matrices are positive semidefinite."""
    omega, added_mass, damping = radiation_sweep
    for i in range(len(omega)):
        eigvals = np.linalg.eigvalsh(damping[i])
        assert np.min(eigvals) >= -1e-4, (
            f"Damping not positive semidefinite at omega={omega[i]}: "
            f"min eigenvalue = {np.min(eigvals)}"
        )


def test_validate_coupled_sweep(radiation_sweep):
    """Test that the sweep passes validation."""
    omega, added_mass, damping = radiation_sweep
    # Should not raise
    omega_v, a_v, b_v = validate_coupled_sweep(omega, added_mass, damping)
    assert len(omega_v) == len(omega)
    assert a_v.shape == (len(omega), 6, 6)
    assert b_v.shape == (len(omega), 6, 6)


def test_cummins_kernel_shape(radiation_sweep):
    """Test that the memory kernel has the correct shape."""
    omega, added_mass, damping = radiation_sweep
    tau = np.linspace(0, 50.0, 1000)
    kernel = coupled_memory_kernels(omega, damping, tau)
    assert kernel.values.shape == (1000, 6, 6)


def test_cummins_kernel_decay(radiation_sweep):
    """Test that the memory kernel decays to near zero."""
    omega, added_mass, damping = radiation_sweep
    tau = np.linspace(0, 50.0, 1000)
    kernel = coupled_memory_kernels(omega, damping, tau)
    # Check that the kernel decays at the end
    # Diagonal entries should decay
    for i in range(6):
        assert abs(kernel.values[-1, i, i]) < 1.0, (
            f"Kernel K_{i+1}{i+1} did not decay: {kernel.values[-1, i, i]}"
        )


def test_heave_added_mass_positive(radiation_sweep):
    """Test that heave added mass is positive."""
    omega, added_mass, damping = radiation_sweep
    # Heave is mode 2
    for i in range(len(omega)):
        assert added_mass[i, 2, 2] > 0, (
            f"Heave added mass not positive at omega={omega[i]}: "
            f"{added_mass[i, 2, 2]}"
        )


def test_heave_damping_positive(radiation_sweep):
    """Test that heave damping is positive."""
    omega, added_mass, damping = radiation_sweep
    # Heave is mode 2
    for i in range(len(omega)):
        assert damping[i, 2, 2] > 0, (
            f"Heave damping not positive at omega={omega[i]}: "
            f"{damping[i, 2, 2]}"
        )
