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
@pytest.mark.parametrize("wall_capacity", [0., .8])
def test_affine_generation_has_exact_temperature_and_nonzero_normal_flux(quadratic, wall_capacity):
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
    surface_capacity = wall_capacity*(1.2+points[:, 0])
    external = q+1j*omega*surface_capacity*temperature
    data = np.select([sampled=="flux", sampled=="film"], [external, temperature+external/2], default=temperature)
    result = solve_mixed_harmonic_conduction(mesh, kinds, data, omega, conductivity=k,
        volumetric_heat_capacity=capacity, film_coefficients=2., volumetric_source=1j*omega*capacity*base,
        source_gradient=1j*omega*capacity*grad, quadratic=quadratic, wall_areal_heat_capacity=surface_capacity)
    np.testing.assert_allclose(result.wall_temperature, temperature, atol=2e-13, rtol=0)
    np.testing.assert_allclose(result.external_inward_flux, external, atol=2e-13, rtol=0)
    np.testing.assert_allclose(result.inward_flux, q, atol=2e-13, rtol=0)
    probes = np.array([[.1, -.4], [.2, .5]])
    np.testing.assert_allclose(result.temperature(probes), base+probes @ grad, atol=2e-13, rtol=0)


@pytest.mark.parametrize("quadratic", [False, True])
@pytest.mark.parametrize("loss", [0., 2.5])
def test_quadratic_temperature_with_nonzero_laplacian_and_cross_term(quadratic, loss):
    from tensorbem.harmonic_conduction import harmonic_boundary_points

    mesh = regular_polygon(16)
    points = harmonic_boundary_points(mesh, quadratic=quadratic)
    k, capacity, omega = 1.5, 2., 3.
    xx, xy, yy = .3+.2j, -.4+.1j, .2-.3j

    def field(points):
        x, y = points.T
        return .7j+xx*x**2+xy*x*y+yy*y**2

    reservoir = np.array([.25j, .4-.3j, -.3+.1j])
    admittance = loss+1j*omega*capacity
    solution = solve_harmonic_conduction(mesh, field(points), omega, conductivity=k,
        volumetric_heat_capacity=capacity, volumetric_source=admittance*.7j-2*k*(xx+yy),
        source_quadratic=admittance*np.array([xx, xy, yy])-loss*reservoir, quadratic=quadratic,
        heat_loss_coefficient=loss, bulk_ambient_quadratic=reservoir)
    x, y = points.T
    normals = np.repeat(mesh.normals, len(points)//mesh.panel_count, axis=0)
    expected = k*(normals[:, 0]*(2*xx*x+xy*y)+normals[:, 1]*(xy*x+2*yy*y))
    np.testing.assert_allclose(solution.inward_flux, expected, rtol=0, atol=2e-13)
    probes = np.array([[0., 0.], [.23, -.46], [.6, .1]])
    np.testing.assert_allclose(solution.temperature(probes), field(probes), rtol=0, atol=2e-13)
    x, y = probes.T
    expected_flux = -k*np.column_stack((2*xx*x+xy*y, xy*x+2*yy*y))
    np.testing.assert_allclose(solution.heat_flux(probes), expected_flux, rtol=0, atol=2e-13)


def test_quadratic_midpoint_single_layer_self_integral():
    from scipy.integrate import quad_vec
    from scipy.special import kv
    from tensorbem.harmonic_conduction import _quadratic_operators

    mesh = regular_polygon(8)
    wave = np.sqrt(1.3j)
    length = mesh.lengths[0]
    single, _ = _quadratic_operators(mesh, mesh.centers[:1], wave)
    # Even kernel and middle Gauss-node basis allow integration of one half.
    def integrand(s):
        return length/(2*np.pi)*kv(0, wave*s*length/2)*(1-5*s**2/3)

    exact, _ = quad_vec(integrand, 0., 1., epsabs=1e-14, epsrel=1e-13)
    np.testing.assert_allclose(single[0, 1], exact, atol=3e-11, rtol=0)


@pytest.mark.parametrize("capacity", [-1., np.nan, np.inf, 1j, [1., 2.]])
def test_invalid_wall_areal_heat_capacity(capacity):
    with pytest.raises(ValueError, match="wall areal heat capacities"):
        solve_mixed_harmonic_conduction(regular_polygon(8), "flux", 0., 1.,
            conductivity=1., volumetric_heat_capacity=1., wall_areal_heat_capacity=capacity)


@pytest.mark.parametrize("quadratic", [False, True])
def test_distributed_exchange_disk_against_radial_solution(quadratic):
    mesh = regular_polygon(64)
    omega, k, c, exchange = 1.7, 1.2, 2., 3.5
    ambient, source, wall = .3-.4j, .7j, .6+.1j
    admittance = exchange+1j*omega*c
    particular = (source+exchange*ambient)/admittance
    wave = np.sqrt(admittance/k)
    field = solve_harmonic_conduction(mesh, wall, omega, conductivity=k,
        volumetric_heat_capacity=c, heat_loss_coefficient=exchange, bulk_ambient_temperature=ambient,
        volumetric_source=source, quadratic=quadratic)
    r = np.array([0., .3, .6])
    target = particular+(wall-particular)*iv(0, wave*r)/iv(0, wave)
    np.testing.assert_allclose(field.temperature(np.column_stack([r, 0*r])), target, atol=.001)
    weights = np.array([5., 8., 5.])/18 if quadratic else np.ones(1)
    mean_flux = (field.inward_flux.reshape(mesh.panel_count, -1) @ weights).mean()
    np.testing.assert_allclose(mean_flux, k*(wall-particular)*wave*iv(1, wave)/iv(0, wave), atol=.002)


@pytest.mark.parametrize("options", [
    {"heat_loss_coefficient": -1.}, {"heat_loss_coefficient": 1j},
    {"heat_loss_coefficient": np.inf}, {"bulk_ambient_temperature": np.nan},
    {"bulk_ambient_temperature": [.2, .3]}, {"bulk_ambient_gradient": [1.]},
    {"bulk_ambient_gradient": [np.inf, 0.]}, {"bulk_ambient_quadratic": [1., 0.]},
    {"bulk_ambient_quadratic": [0., np.nan, 0.]}])
def test_invalid_distributed_exchange_parameters(options):
    with pytest.raises(ValueError, match="heat_loss_coefficient|bulk_ambient"):
        solve_harmonic_conduction(regular_polygon(8), 0., 1., conductivity=1.,
                                  volumetric_heat_capacity=1., **options)

@pytest.mark.parametrize("quadratic", [False, True])
def test_affine_reservoir_and_source_match_oblique_temperature(quadratic):
    from tensorbem.harmonic_conduction import harmonic_boundary_points

    mesh = regular_polygon(16)
    xy = harmonic_boundary_points(mesh, quadratic=quadratic)
    grad = np.array([.2+.3j, -.4+.25j])
    reservoir_grad = np.array([-.3j, .5+.1j])
    k, c, omega, loss = 1.4, 2.1, 3., 2.5
    alpha = loss+1j*omega*c
    base, ambient = .3-.2j, .45j
    field = solve_harmonic_conduction(mesh, base+xy @ grad, omega, conductivity=k,
        volumetric_heat_capacity=c, volumetric_source=alpha*base-loss*ambient,
        source_gradient=alpha*grad-loss*reservoir_grad, heat_loss_coefficient=loss,
        bulk_ambient_temperature=ambient, bulk_ambient_gradient=reservoir_grad, quadratic=quadratic)
    np.testing.assert_allclose(field.inward_flux, k*np.repeat(mesh.normals @ grad, len(xy)//16), atol=2e-13)
    np.testing.assert_allclose(field.heat_flux([[.1, .2]])[0], -k*grad, atol=2e-13)


@pytest.mark.parametrize("quadratic", [False, True])
def test_partial_panel_heat_integrates_prescribed_flux(quadratic):
    mesh = regular_polygon(8)
    nodes = np.polynomial.legendre.leggauss(3 if quadratic else 1)[0]
    fractions = (1+nodes)/2
    coeff = 2-3j if quadratic else 0.
    offset = np.arange(8)+.25j
    data = offset[:, None]+coeff*fractions**2
    field = solve_mixed_harmonic_conduction(mesh, "flux", data.ravel(), 2.,
        conductivity=1., volumetric_heat_capacity=1., quadratic=quadratic)
    panels = np.array([[1], [5]])
    start, end = np.array([0., .23, .81]), np.array([1., .74, .81])
    expected = mesh.lengths[panels]*(offset[panels]*(end-start)+coeff*(end**3-start**3)/3)
    np.testing.assert_allclose(field.panel_inward_power(panels, start, end), expected, atol=1e-13)
    assert field.panel_inward_power(1, .5, .5) == 0
    for args in ((8,), (-1,), (1.5,), (1, .8, .3), (1, 0., np.nan)):
        with pytest.raises(ValueError):
            field.panel_inward_power(*args)


@pytest.mark.parametrize("quadratic", [False, True])
def test_partial_wall_storage_resolves_sampled_capacity(quadratic):
    mesh = regular_polygon(8)
    nodes = np.polynomial.legendre.leggauss(3 if quadratic else 1)[0]
    t, omega = .7-.3j, 1.3
    fractions = (1+nodes)/2
    panel = np.arange(8)
    base = .2+.1*panel
    slope = .8 if quadratic else 0.
    capacity = (base[:, None]+slope*fractions**2).ravel()
    field = solve_mixed_harmonic_conduction(mesh, "temperature", t, omega, conductivity=1.,
        volumetric_heat_capacity=1., volumetric_source=1j*omega*t,
        wall_areal_heat_capacity=capacity, quadratic=quadratic)
    start, end = .17, .82
    exact = 1j*omega*t*mesh.lengths*(base*(end-start)+slope*(end**3-start**3)/3)
    for component in ("storage", "external"):
        np.testing.assert_allclose(field.panel_inward_power(panel, start, end, component=component),
                                   exact, atol=1e-13, rtol=0)
    np.testing.assert_allclose(field.panel_inward_power(panel), 0., atol=1e-13)
    with pytest.raises(ValueError, match="component"):
        field.panel_inward_power(panel, component="unknown")


@pytest.mark.parametrize("condition", ["flux", "film"])
@pytest.mark.parametrize("quadratic", [False, True])
def test_steady_screened_disk_radial_refinement(condition, quadratic, record_property):
    # On the unit disk T=base+A*I0(lambda*r), lambda²=H/k. Distributed
    # exchange fixes the constant even when every boundary flux is prescribed.
    k, loss, ambient, source = 1.2, 3.5, 2., .7
    wave = np.sqrt(loss/k)
    base, amplitude = ambient+source/loss, -.8
    wall = base+amplitude*iv(0, wave)
    inward = k*amplitude*wave*iv(1, wave)
    film = 2.5
    data = inward if condition == "flux" else wall+inward/film
    r = np.array([0., .2, .55, .75])
    points = np.column_stack((.6*r, .8*r))
    exact = base+amplitude*iv(0, wave*r)
    flux = (-k*amplitude*wave*iv(1, wave*r))[:, None]*[.6, .8]
    previous = None
    for panels in (16, 32, 64, 128):
        field = solve_mixed_harmonic_conduction(regular_polygon(panels), condition, data, 0.,
            conductivity=k, volumetric_heat_capacity=2., heat_loss_coefficient=loss,
            bulk_ambient_temperature=ambient, volumetric_source=source, film_coefficients=film,
            wall_areal_heat_capacity=3., quadratic=quadratic)
        errors = np.array([max(abs(field.temperature(points)-exact)), np.max(abs(field.heat_flux(points)-flux))])
        record_property(f"temperature_error_{panels}_K", errors[0])
        record_property(f"flux_error_{panels}_W_m2", errors[1])
        np.testing.assert_array_equal(field.external_inward_flux, field.inward_flux)
        if previous is not None:
            assert np.all(previous > 3*errors)
        previous = errors
    assert errors[0] < .002 and errors[1] < .003


@pytest.mark.parametrize("quadratic", [False, True])
def test_area_mean_polynomial_on_concave_polygon(quadratic, record_property):
    from tensorbem.geometry import PanelMesh
    from tensorbem.harmonic_conduction import harmonic_boundary_points

    mesh = PanelMesh(np.array([[2., 1.], [4., 1.], [4., 2.], [3., 2.], [3., 4.], [2., 4.]]))
    def exact(x, y):
        return 1.+.4*x-.2j*y+.1j*x**2-.3*x*y+.12*y**2
    k, storage = 1.3, 2.4j
    data = exact(*harmonic_boundary_points(mesh, quadratic=quadratic).T)
    field = solve_harmonic_conduction(mesh, data, 2., conductivity=k, volumetric_heat_capacity=1.2,
        quadratic=quadratic, volumetric_source=storage-2*k*(.1j+.12),
        source_gradient=storage*np.array([.4, -.2j]), source_quadratic=storage*np.array([.1j, -.3, .12]))
    # Partition the L into two rectangles, with independent tensor Gauss integration.
    nodes, weights = np.polynomial.legendre.leggauss(2)
    integral = 0j
    for x, y, width, height in [(2., 1., 2., 1.), (2., 2., 1., 2.)]:
        values = exact(x+width*(1+nodes[:, None])/2, y+height*(1+nodes)/2)
        integral += width*height*(weights @ values @ weights)/4
    # The two equal-area rectangles have centroids (3,1.5) and (2.5,3).
    x, y = 2.75, 2.25
    expected_flux = -k*np.array([.4+.2j*x-.3*y, -.2j-.3*x+.24*y])
    flux_error = np.max(abs(field.area_mean_heat_flux()-expected_flux))
    record_property("mean_flux_error_W_m2", flux_error)
    assert flux_error < 1e-13
    error = abs(field.area_mean_temperature()-integral/4)
    record_property("mean_temperature_error_K", error)
    assert error < 1e-13


@pytest.mark.parametrize("omega", [0., 2.])
def test_area_mean_screened_disk_converges(omega, record_property):
    k, loss, capacity = 1.3, 2., 1.5
    ambient, source, wall = .7, .15, 1.
    admittance = loss+1j*omega*capacity
    base = (loss*ambient+source)/admittance
    wave = np.sqrt(admittance/k)
    expected = base+(wall-base)*2*iv(1, wave)/(wave*iv(0, wave))
    previous = None
    for n in (16, 32, 64):
        field = solve_mixed_harmonic_conduction(regular_polygon(n), "temperature", wall, omega, conductivity=k,
            volumetric_heat_capacity=capacity, heat_loss_coefficient=loss,
            bulk_ambient_temperature=ambient, volumetric_source=source, quadratic=True,
            wall_areal_heat_capacity=2.)
        error = abs(field.area_mean_temperature()-expected)
        record_property(f"mean_temperature_error_{n}_K", error)
        if previous is not None:
            assert previous > 3*error
        previous = error
    assert error < .001


@pytest.mark.parametrize("quadratic", [False, True])
def test_area_mean_flux_from_oblique_wave_on_rectangle(quadratic, record_property):
    from tensorbem.geometry import PanelMesh
    from tensorbem.harmonic_conduction import harmonic_boundary_points

    k, capacity, omega = 1.3, 1.8, 2.1
    size = np.array([1.4, .9])
    direction = np.array([.6, -.8])
    rate = np.sqrt(1j*omega*capacity/k)*direction
    # Separable integral of exp(rate dot x) on the rectangle.
    expected = -k*rate*np.prod(np.expm1(rate*size)/(rate*size))
    corners = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])*size
    previous = None
    for n in ((4, 8, 16) if quadratic else (4, 8, 16, 32, 64)):
        edges = np.roll(corners, -1, axis=0)-corners
        mesh = PanelMesh((corners[:, None]+edges[:, None]*np.arange(n)[None, :, None]/n).reshape(-1, 2))
        points = harmonic_boundary_points(mesh, quadratic=quadratic)
        inward = k*np.exp(points @ rate)*np.repeat(mesh.normals @ rate, 3 if quadratic else 1)
        field = solve_mixed_harmonic_conduction(mesh, "flux", inward, omega, conductivity=k,
                                                volumetric_heat_capacity=capacity, quadratic=quadratic)
        error = np.max(abs(field.area_mean_heat_flux()-expected))
        record_property(f"max_component_error_{4*n}_W_m2", error)
        if previous is not None:
            assert previous > 3*error
        previous = error
    assert error < .002
