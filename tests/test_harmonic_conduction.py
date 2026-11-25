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


def test_quadratic_mixed_panel_samples_recover_oblique_wave():
    from tensorbem.harmonic_conduction import harmonic_boundary_points

    mesh = regular_polygon(24)
    xy = harmonic_boundary_points(mesh, quadratic=True)
    direction = np.array([.6, .8])
    wave = np.sqrt(2j)
    exact = np.exp(wave*(xy @ direction-1))
    flux = wave*exact*np.repeat(mesh.normals @ direction, 3)
    kinds = np.repeat(["temperature", "flux", "film"], 8)
    h = np.linspace(1., 2., 24)
    sampled = np.repeat(kinds, 3)
    data = np.select([sampled=="flux", sampled=="film"], [flux, exact+flux/np.repeat(h, 3)], default=exact)
    result = solve_mixed_harmonic_conduction(mesh, kinds, data, 2., conductivity=1.,
                                             volumetric_heat_capacity=1., film_coefficients=h, quadratic=True)
    assert np.max(abs(result.inward_flux-flux)) < 4e-4
    assert np.max(abs(result.wall_temperature-exact)) < 1e-5
    probes = np.vstack(([0., 0.], mesh.centers[0]-.025*mesh.normals[0]))
    np.testing.assert_allclose(result.temperature(probes), np.exp(wave*(probes @ direction-1)), atol=2e-5)
    # Include a probe within a tenth of a panel length. The quadrature for
    # the gradient subdivides here, unlike at the central probe.
    step = 1e-5
    numerical = np.column_stack([(result.temperature(probes+step*axis)-result.temperature(probes-step*axis))/(2*step)
                                 for axis in np.eye(2)])
    np.testing.assert_allclose(result.heat_flux(probes), -numerical, atol=2e-8, rtol=0)


@pytest.mark.parametrize("quadratic", [False, True])
def test_uniform_heat_generation_insulated_and_isothermal_disk(quadratic):
    mesh = regular_polygon(64)
    k, capacity, omega = 1.3, 2., 3.
    source = 1.2-.7j
    particular = source/(1j*omega*capacity)
    opts = dict(conductivity=k, volumetric_heat_capacity=capacity, volumetric_source=source, quadratic=quadratic)
    insulated = solve_mixed_harmonic_conduction(mesh, "flux", 0., omega, **opts)
    np.testing.assert_allclose(insulated.temperature([[0., 0.]]), particular, atol=1e-13)
    np.testing.assert_allclose(insulated.wall_temperature, particular, atol=1e-13)
    cold = solve_harmonic_conduction(mesh, 0., omega, **opts)
    wave = np.sqrt(1j*omega*capacity/k)
    probes = np.array([[.3, .4], [-.4, .3], [0., 0.]])
    radii = np.linalg.norm(probes, axis=1)
    unit = np.divide(probes, radii[:, None], out=np.zeros_like(probes), where=radii[:, None]>0)
    exact_flux = k*particular*wave*iv(1, wave*radii)/iv(0, wave)
    np.testing.assert_allclose(cold.heat_flux(probes), exact_flux[:, None]*unit, atol=.001)
    expected = particular*(1-1/iv(0, wave))
    np.testing.assert_allclose(cold.temperature([[0., 0.]]), expected, atol=.001)
    weights = np.array([5., 8., 5.])/18 if quadratic else np.array([1.])
    mean_flux = (cold.inward_flux.reshape(mesh.panel_count, -1) @ weights).mean()
    np.testing.assert_allclose(mean_flux, -k*particular*wave*iv(1, wave)/iv(0, wave), atol=.002)


@pytest.mark.parametrize("quadratic", [False, True])
def test_affine_generation_has_exact_temperature_and_nonzero_normal_flux(quadratic):
    from tensorbem.harmonic_conduction import harmonic_boundary_points

    mesh = regular_polygon(12)
    points = harmonic_boundary_points(mesh, quadratic=quadratic)
    grad = np.array([.2+.4j, -.6+.3j])
    base = .7-.3j
    k, capacity, omega = 1.8, 2.1, 4.
    temperature = base+points @ grad
    q = k*np.repeat(mesh.normals @ grad, len(points)//12)
    kinds = np.resize(["temperature", "flux", "film"], 12)
    sampled = np.repeat(kinds, len(points)//12)
    data = np.select([sampled=="flux", sampled=="film"], [q, temperature+q/2], default=temperature)
    result = solve_mixed_harmonic_conduction(mesh, kinds, data, omega, conductivity=k,
        volumetric_heat_capacity=capacity, film_coefficients=2., volumetric_source=1j*omega*capacity*base,
        source_gradient=1j*omega*capacity*grad, quadratic=quadratic)
    np.testing.assert_allclose(result.wall_temperature, temperature, atol=2e-13, rtol=0)
    np.testing.assert_allclose(result.inward_flux, q, atol=2e-13, rtol=0)
    probes = np.array([[.1, -.4], [.2, .5]])
    np.testing.assert_allclose(result.temperature(probes), base+probes @ grad, atol=2e-13, rtol=0)


@pytest.mark.parametrize("quadratic", [False, True])
def test_quadratic_temperature_with_nonzero_laplacian_and_cross_term(quadratic):
    from tensorbem.harmonic_conduction import harmonic_boundary_points

    mesh = regular_polygon(16)
    points = harmonic_boundary_points(mesh, quadratic=quadratic)
    k, capacity, omega = 1.5, 2., 3.
    xx, xy, yy = .3+.2j, -.4+.1j, .2-.3j

    def field(points):
        x, y = points.T
        return .7j+xx*x**2+xy*x*y+yy*y**2

    solution = solve_harmonic_conduction(mesh, field(points), omega, conductivity=k,
        volumetric_heat_capacity=capacity, volumetric_source=1j*omega*capacity*.7j-2*k*(xx+yy),
        source_quadratic=1j*omega*capacity*np.array([xx, xy, yy]), quadratic=quadratic)
    x, y = points.T
    normals = np.repeat(mesh.normals, len(points)//mesh.panel_count, axis=0)
    expected = k*(normals[:, 0]*(2*xx*x+xy*y)+normals[:, 1]*(xy*x+2*yy*y))
    np.testing.assert_allclose(solution.inward_flux, expected, rtol=0, atol=2e-13)
    probes = np.array([[0., 0.], [.23, -.46], [.6, .1]])
    np.testing.assert_allclose(solution.temperature(probes), field(probes), rtol=0, atol=2e-13)
    x, y = probes.T
    expected_flux = -k*np.column_stack((2*xx*x+xy*y, xy*x+2*yy*y))
    np.testing.assert_allclose(solution.heat_flux(probes), expected_flux, rtol=0, atol=2e-13)
