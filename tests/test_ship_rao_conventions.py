"""Complex heave reference from the independent Capytaine 3.0.0 RAO solve."""

import numpy as np
import pytest

from tensorbem.geometry3d import hemisphere_hull_mesh
from tensorbem.hydrostatic import compute_hydrostatic_restoring_matrix
from tensorbem.rao import compute_rao_at_frequency, compute_rao_spectrum
from tensorbem.wigley3_hull import (
    CLASSIC_WIGLEY_ALPHA,
    LEGACY_WIGLEY_ALPHA,
    WIGLEY3_ALPHA,
    wigley3_displaced_volume,
    wigley3_half_breadth,
)


@pytest.mark.parametrize(
    "omega,modulus,phase",
    [
        (2.0, 1.059929837, 0.3140344),
        (2.5, 1.217622882, 3.0164093),
        (3.0, 1.721059037, 23.6965485),
        (3.5, 1.090033961, 87.1399271),
    ],
)
def test_public_heave_rao_and_physical_motion(omega, modulus, phase):
    mesh = hemisphere_hull_mesh(3)
    rho, g = 1025.0, 9.80665
    cg = np.array([0.0, 0.0, -0.375])
    props = compute_hydrostatic_restoring_matrix(mesh, rho * mesh.volume, cg, rho, g)
    mass = rho * props.displaced_volume
    common = dict(
        hull=mesh,
        wave_angle=0.0,
        mass=mass,
        center_of_gravity=cg,
        hydrostatic_props=props,
        inertia=0.3 * mass * np.eye(3),
        chief_points=None,
        density=rho,
        gravity=g,
    )
    unit = compute_rao_at_frequency(omega=omega, amplitude=1.0, **common)
    half = compute_rao_at_frequency(omega=omega, amplitude=0.5, **common)
    np.testing.assert_allclose(half.rao, unit.rao, atol=1e-10)
    np.testing.assert_allclose(half.motion, 0.5 * unit.motion, atol=1e-10)
    np.testing.assert_allclose(half.excitation_force, 0.5 * unit.excitation_force)
    assert half.impedance_residual < 1e-8
    assert abs(unit.rao[2]) == pytest.approx(modulus, rel=0.07)
    assert np.angle(unit.rao[2], deg=True) == pytest.approx(phase, abs=3)
    spectrum = compute_rao_spectrum(omegas=[omega], amplitude=0.5, **common)
    np.testing.assert_allclose(spectrum.complex_raos[0], unit.rao, atol=1e-10)
    np.testing.assert_allclose(spectrum.rao_amplitudes[0], np.abs(unit.rao), atol=1e-10)


def test_named_hull_dimensions_and_offsets():
    assert (CLASSIC_WIGLEY_ALPHA, WIGLEY3_ALPHA) == (0.0, 0.2)
    # Table 1-III, station at x/L=1/4, still-water line.
    assert wigley3_half_breadth(0.75, 0.0) == pytest.approx(0.118125)
    assert wigley3_displaced_volume() == pytest.approx(0.078)
    assert wigley3_displaced_volume(alpha=CLASSIC_WIGLEY_ALPHA) == pytest.approx(0.075)
    assert wigley3_displaced_volume(alpha=LEGACY_WIGLEY_ALPHA) == pytest.approx(
        0.09163125
    )
