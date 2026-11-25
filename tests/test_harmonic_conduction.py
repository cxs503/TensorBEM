import numpy as np
import pytest
from scipy.special import iv

from tensorbem.geometry import regular_polygon
from tensorbem.harmonic_conduction import solve_harmonic_conduction


def test_circular_uniform_wall_converges_to_radial_diffusion_mode():
    k, capacity, omega = 1.3, 2., 3.
    wall = .7-.4j
    wave = np.sqrt(1j*omega*capacity/k)
    radii = np.array([0., .3, .65])
    expected = wall*iv(0, wave*radii)/iv(0, wave)
    exact_flux = k*wave*wall*iv(1, wave)/iv(0, wave)
    errors = []
    for n in (24, 48):
        mesh = regular_polygon(n)
        field = solve_harmonic_conduction(mesh, wall, omega, conductivity=k, volumetric_heat_capacity=capacity)
        temp = field.temperature(np.column_stack((radii, np.zeros_like(radii))))
        errors.append(np.array([abs(temp-expected).max(), abs(field.inward_flux-exact_flux).max()]))
    assert np.all(errors[1] < errors[0]/3)
    assert np.all(errors[1] < .01)
    with pytest.raises(ValueError, match="inside"):
        field.temperature([[1.5, 0.]])
    with pytest.raises(ValueError, match="positive"):
        solve_harmonic_conduction(mesh, wall, 0., conductivity=k, volumetric_heat_capacity=capacity)
