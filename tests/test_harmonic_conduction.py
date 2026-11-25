import numpy as np
import pytest
from scipy.special import iv

from tensorbem.geometry import regular_polygon
from tensorbem.harmonic_conduction import solve_harmonic_conduction, solve_mixed_harmonic_conduction


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

@pytest.mark.parametrize("condition", ["flux", "film", "mixed"])
def test_boundary_exchange_recovers_dirichlet_solution(condition):
    mesh = regular_polygon(24)
    props = dict(conductivity=1.8, volumetric_heat_capacity=2.1)
    wall = (.3+.6j)*mesh.centers[:, 0] + .4j
    direct = solve_harmonic_conduction(mesh, wall, 1.7, **props)
    kinds = np.resize(["temperature", "flux", "film"], 24) if condition == "mixed" else np.full(24, condition)
    h = 2.+mesh.centers[:, 1]
    values = np.select([kinds=="flux", kinds=="film"], [direct.inward_flux, wall+direct.inward_flux/h], default=wall)
    result = solve_mixed_harmonic_conduction(mesh, kinds, values, 1.7, film_coefficients=h, **props)
    np.testing.assert_allclose(result.wall_temperature, wall, atol=3e-14)
    np.testing.assert_allclose(result.inward_flux, direct.inward_flux, atol=3e-14)
