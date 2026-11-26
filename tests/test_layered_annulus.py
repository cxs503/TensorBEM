import numpy as np
import pytest

from tensorbem.benchmarks.layered_annulus import angular_mode, measure
from tensorbem.layered_annulus import solve_annulus_contact


@pytest.mark.parametrize("r", [0., .2])
@pytest.mark.parametrize("reverse", [False, True])
def test_radial_heat_rate_and_both_traces_converge(r, reverse):
    coarse, fine = (measure(n, r, reverse) for n in (16, 32))
    for key in ("max_rate_relative_error", "max_trace_error_K"):
        assert fine[key] < coarse[key] / 3.5
    assert fine["max_rate_relative_error"] < .003


def test_sign_changing_interface_flux_converges_locally():
    coarse, fine = angular_mode(16), angular_mode(32)
    assert fine["flux_max_error_over_amplitude"] < coarse["flux_max_error_over_amplitude"] / 3.8
    assert fine["max_abs_integrated_rate_W_m"] < 1e-12


def test_variable_panel_resistance_and_derivative_of_wall_offset():
    n = 24
    r = np.linspace(.1, .3, n)
    args = ((.25, .6, 1.), (4., 1.))
    first = solve_annulus_contact(*args, (1., 0.), r, panels=n)
    shifted = solve_annulus_contact(*args, (11., 10.), r, panels=n)
    np.testing.assert_allclose(shifted.inner_trace, first.inner_trace + 10, atol=1e-12)
    np.testing.assert_allclose(shifted.interface_flux, first.interface_flux, atol=1e-12)
    np.testing.assert_allclose(first.inner_trace-first.outer_trace, r*first.interface_flux)


def test_varying_resistance_matches_independent_harmonic_trace_and_flux():
    from tensorbem.benchmarks.layered_annulus import varying_contact

    coarse, fine = varying_contact(16), varying_contact(32)
    for key in ("max_flux_error_W_m2", "max_inner_trace_error_K", "max_outer_trace_error_K"):
        assert fine[key] < coarse[key] / 4
    assert fine["max_flux_error_W_m2"] < .001


def test_both_material_fields_refine_against_cartesian_harmonic_solution():
    coarse, fine = (angular_mode(n)["interior_fields"] for n in (16, 32))
    for layer in ("inner", "outer"):
        for key in ("temperature_max_error_K", "flux_vector_max_error_W_m2"):
            assert fine[layer][key] < coarse[layer][key] / 3.5


def test_layer_fields_require_interior_points_of_the_selected_material():
    result = solve_annulus_contact((.25, .6, 1.), (4., 1.), (310., 290.), .2, panels=24)
    for field, invalid in ((result.inner_field, [.8, 0.]), (result.outer_field, [.4, 0.])):
        for point in (invalid, field.mesh.centers[0], field.mesh.centers[-1]):
            with pytest.raises(ValueError):
                field.heat_flux(np.asarray([point]))
    row = measure(32)
    for rate in row["interior_ring_rates_W_m"]:
        assert abs(rate/row["exact_rate_W_m"] - 1) < .002


@pytest.mark.parametrize("panels", [15, 16])
def test_circular_extension_of_log_and_sine_modes(panels):
    from dataclasses import replace

    field = solve_annulus_contact((.25, .6, 1.), (4., 1.), (1., 0.), panels=panels).outer_field
    # On each nominal circle prescribe log(r) + Im(z³). Sample at panel angles.
    traces = []
    for loop in field.mesh.loops:
        radii = np.linalg.norm(loop.vertices[0])
        phase = np.arctan2(loop.centers[:, 1], loop.centers[:, 0])
        traces.append(np.log(radii) + radii**3 * np.sin(3*phase))
    field = replace(field, boundary_values=np.concatenate(traces))
    xy = np.array([[.62, .04], [-.31, .70], [.12, -.92]])
    z = xy[:, 0] + 1j*xy[:, 1]
    temperature, flux = field.circular_field(xy)
    exact_grad = xy / abs(z[:, None])**2 + 3*np.column_stack(((z*z).imag, (z*z).real))
    np.testing.assert_allclose(temperature, np.log(abs(z)) + (z**3).imag, atol=1e-13)
    np.testing.assert_allclose(flux, -exact_grad, atol=1e-12)


def test_near_contact_circular_flux_refines_on_both_sides():
    from tensorbem.benchmarks.layered_annulus import near_contact

    coarse, fine = (near_contact(n)["rings"] for n in (16, 32))
    for a, b in zip(coarse, fine):
        if a["clearance_over_panel_length"] == .125:
            assert b["circular_relative_linf"] < a["circular_relative_linf"]/3.8
            assert b["circular_relative_linf"] < .005


@pytest.mark.parametrize("n", [15, 16])
def test_circular_contact_map_handles_mean_flux_and_variable_resistance(n):
    from tensorbem.benchmarks.layered_annulus import varying_contact

    radial = measure(n, reverse=True, formulation="circular")
    assert radial["max_rate_relative_error"] < 2e-12
    assert radial["max_trace_error_K"] < 2e-11
    for rate in radial["interior_ring_rates_W_m"]:
        assert abs(rate/radial["exact_rate_W_m"] - 1) < 2e-12
    varying = varying_contact(n, formulation="circular")
    assert varying["max_flux_error_W_m2"] < 1e-12
    assert varying["max_outer_trace_error_K"] < 1e-12


def test_angular_resolution_separates_sampling_and_solved_trace_error():
    from tensorbem.benchmarks.annulus_angular_resolution import measure as angular_resolution

    aliased = angular_resolution(16, 8)["flux_relative_linf"]
    resolved = angular_resolution(32, 8)["flux_relative_linf"]
    for layer in ("inner", "outer"):
        assert aliased[layer]["exact_trace_control"] > .8
        assert aliased[layer]["circular_solve"] > .8
        assert resolved[layer]["exact_trace_control"] < 1e-12
        assert resolved[layer]["circular_solve"] < 1e-12


@pytest.mark.parametrize("formulation", ["polygonal", "circular"])
def test_convection_radial_rate_and_variable_angular_films(formulation):
    from tensorbem.benchmarks.annulus_convection import angular, radial

    for hot, cold in ((310., 290.), (290., 310.)):
        run = radial(32, formulation, hot, cold)
        assert run["max_rate_relative_error"] < .002
        assert run["max_wall_temperature_error_K"] < .012
    assert angular(32, formulation)["interface_flux_max_error_W_m2"] < .009


@pytest.mark.parametrize("films", [(0., None), (None, 0.), (0., 2.), (2., 0.)])
def test_insulated_wall_has_constant_steady_temperature(films):
    result = solve_annulus_contact((.25,.6,1.), (4.,1.), (5.,12.), .2, 16, film_coefficients=films)
    expected = 12. if films[0] == 0. else 5.
    np.testing.assert_allclose(result.inner_trace, expected, atol=1e-11)
    np.testing.assert_allclose(result.outer_trace, expected, atol=1e-11)
    np.testing.assert_allclose(result.heat_rates, 0., atol=1e-11)


@pytest.mark.parametrize("films", [(0., 0.), (np.inf, None), (-1., 3.)])
def test_unanchored_temperature_or_invalid_film_rejected(films):
    with pytest.raises(ValueError):
        solve_annulus_contact((.25,.6,1.), (4.,1.), (5.,12.), panels=16, film_coefficients=films)


@pytest.mark.parametrize("driven", [0, 1])
def test_radial_flux_drives_film_anchored_wall(driven):
    radii = (.25, .6, 1.)
    k = (4., 1.5)
    resistance = .2
    rate = 3.5
    sign = 1 if driven == 0 else -1
    fluxes = [None,None]
    fluxes[driven] = sign*rate/(2*np.pi*radii[2*driven])
    ambient, films = [None,None], [None,None]
    ambient[1-driven], films[1-driven] = 10., 3.
    result = solve_annulus_contact(radii, k, ambient, resistance, panels=24,
                                   formulation="circular", film_coefficients=films, wall_fluxes=fluxes)
    np.testing.assert_allclose(result.heat_rates, rate, atol=1e-11, rtol=0.)
    np.testing.assert_allclose(result.inner_trace-result.outer_trace,
                               resistance*rate/(2*np.pi*radii[1]), atol=1e-12, rtol=0.)
    wall = (result.outer_field.boundary_values[:24] if driven == 0
            else result.inner_field.boundary_values[24:])
    expected = 10.+sign*rate/(2*np.pi*radii[2*(1-driven)]*3.)
    np.testing.assert_allclose(wall, expected, atol=1e-12, rtol=0.)


def test_prescribed_flux_cannot_share_a_dirichlet_wall_or_leave_network_unanchored():
    with pytest.raises(ValueError, match="prescribed flux"):
        solve_annulus_contact((.25,.6,1.), (2.,1.), (1.,0.), wall_fluxes=(1.,None))
    with pytest.raises(ValueError, match="anchor"):
        solve_annulus_contact((.25,.6,1.), (2.,1.), (None,None), wall_fluxes=(1.,-1.))


def test_circular_pure_flux_selects_only_temperature_offset():
    n = 32
    a, b, c = .25, .6, 1.
    theta = 2*np.pi*(np.arange(n)+.5)/n
    flux = (3.+np.cos(2*theta), -3.*a/c+.5*np.sin(theta))
    def solve(mean, boundary_flux=flux):
        return solve_annulus_contact((a,b,c), (2.,1.), (None,None), .2, n,
                                     formulation="circular", wall_fluxes=boundary_flux,
                                     mean_bore_temperature=mean)
    base, shifted = solve(2.), solve(19.)
    np.testing.assert_allclose(base.inner_field.boundary_values[n:].mean(), 2., atol=1e-12)
    np.testing.assert_allclose(shifted.outer_trace-base.outer_trace, 17., atol=1e-11, rtol=0.)
    np.testing.assert_allclose(shifted.interface_flux, base.interface_flux, atol=1e-11, rtol=0.)
    np.testing.assert_allclose(base.heat_rates, 2*np.pi*a*3., atol=1e-11, rtol=0.)
    with pytest.raises(ValueError, match="incompatible"):
        solve(2., (flux[0], flux[1]+.001))
    zero = solve(12., (0., 0.))
    np.testing.assert_allclose(zero.outer_field.boundary_values, 12., atol=1e-11, rtol=0.)


@pytest.mark.parametrize("fraction",[0.,.35,1.])
def test_powered_contact_with_two_flux_walls_preserves_bore_gauge(fraction):
    # For uniform radial rates the temperatures are affine in log(r).
    a,b,c = 1.,1.4,2.
    source,r = 3.,.2
    qi = .8
    qo = qi+source
    result = solve_annulus_contact((a,b,c),(2.,4.),(None,None),r,16,formulation="circular",
                wall_fluxes=(b*qi/a,-b*qo/c),mean_bore_temperature=5.,
                interface_source=source,inner_resistance_fraction=fraction)
    inner = 5.-b*qi*np.log(b/a)/2
    sheet = inner-fraction*r*qi
    np.testing.assert_allclose(result.sheet_temperature,sheet,atol=1e-12)
    np.testing.assert_allclose(result.outer_trace,sheet-(1-fraction)*r*qo,atol=1e-12)
    np.testing.assert_allclose(result.heat_rates,2*np.pi*b*np.array([qi,qi,qo]),atol=1e-12)
    with pytest.raises(ValueError,match="incompatible"):
        solve_annulus_contact((a,b,c),(2.,4.),(None,None),r,16,formulation="circular",
                    wall_fluxes=(0.,0.),mean_bore_temperature=5.,interface_source=source)


def test_bulk_heating_pure_flux_and_sheet_source():
    a,b,c = .25,.55,1.
    si,so,ki,ko,sheet = 3.,-.4,1.7,2.2,.6
    bore_q = .9
    qi = (a*bore_q+.5*si*(b*b-a*a))/b
    qo = qi+sheet
    ext_q = (b*qo+.5*so*(c*c-b*b))/c
    options = dict(formulation="circular",volumetric_source=(si,so),interface_source=sheet,
                   mean_bore_temperature=4.)
    result = solve_annulus_contact((a,b,c),(ki,ko),(None,None),.12,24,wall_fluxes=(bore_q,-ext_q),**options)
    np.testing.assert_allclose(result.heat_rates,2*np.pi*np.array([a*bore_q,b*qi,c*ext_q]),atol=1e-11)
    ti = 4.-(si*(b*b-a*a)/4+(a*bore_q-si*a*a/2)*np.log(b/a))/ki
    np.testing.assert_allclose(result.inner_trace,ti,atol=1e-12)
    np.testing.assert_allclose(result.outer_trace,ti-.12*(qi+qo)/2,atol=1e-12)
    with pytest.raises(ValueError,match="incompatible"):
        solve_annulus_contact((a,b,c),(ki,ko),(None,None),.12,24,wall_fluxes=(bore_q,0.),**options)


def test_uniform_generation_matches_robin_walls_and_cartesian_gradient():
    # T=C-S*r²/(4k) creates exactly uniform heating and no log component.
    a,b,c = .3,.6,1.
    k,s = 2.,4.
    t = lambda r:5-s*r*r/(4*k)
    films = (2.,3.)
    air = (t(a)+s*a/(2*films[0]),t(c)-s*c/(2*films[1]))
    result = solve_annulus_contact((a,b,c),(k,k),air,panels=16,formulation="circular",
                                   volumetric_source=s,film_coefficients=films)
    for radius,layer in ((.45,result.inner_field),(.8,result.outer_field)):
        xy = radius*np.array([[.6,.8],[-.8,.6]])
        np.testing.assert_allclose(layer.temperature(xy),t(radius),atol=1e-12)
        np.testing.assert_allclose(layer.heat_flux(xy),s*xy/2,atol=1e-12)


def test_affine_source_cartesian_cubic_reference():
    # An independent particular solution contains no mixed cubic monomials.
    a, b, c = .3, .6, 1.
    k = 1.8
    g = np.array([3., -2.])
    n = 24
    angle = 2*np.pi*(np.arange(n)+.5)/n
    direction = np.column_stack((np.cos(angle), np.sin(angle)))

    def temperature(xy):
        return 1.2 + xy[:, 0]*xy[:, 1] - (xy**3 @ g)/(6*k)

    result = solve_annulus_contact((a,b,c), (k,k),
                                   (temperature(a*direction), temperature(c*direction)),
                                   panels=n, formulation="circular", source_gradient=g)
    for r, field in ((.44, result.inner_field), (.83, result.outer_field)):
        xy = r*np.array([[.8,.6], [-.6,.8], [-1.,0.]])
        np.testing.assert_allclose(field.temperature(xy), temperature(xy), atol=2e-13, rtol=0.)
        flux = -k*xy[:, ::-1] + .5*g*xy**2
        np.testing.assert_allclose(field.heat_flux(xy), flux, atol=2e-13, rtol=0.)
    np.testing.assert_allclose(result.heat_rates, 0., atol=2e-13)


@pytest.mark.parametrize("gradient", [1., (1.,), (1., np.nan), ((0.,0.,0.),(0.,0.,0.))])
def test_affine_source_requires_cartesian_pairs(gradient):
    with pytest.raises(ValueError, match="source_gradient"):
        solve_annulus_contact((.3,.6,1.), (2.,1.), (0.,0.), panels=8, source_gradient=gradient)


def test_radial_quadratic_heating_with_flux_walls_and_contact_jump():
    radii = (.3, .6, 1.)
    a, b, c = radii
    k = (1.4, 2.2)
    beta = (4., -1.5)
    resistance = .2
    # Integrate (r*q)' = beta*r³ independently from the specified bore flux.
    def radial_flux(radius, start, incoming, coefficient):
        return (start*incoming + coefficient*(radius**4-start**4)/4)/radius

    def temperature(radius, start, initial, incoming, coefficient, conductivity):
        return initial - ((start*incoming-coefficient*start**4/4)*np.log(radius/start)
                          + coefficient*(radius**4-start**4)/16)/conductivity

    bore_flux = .8
    contact_flux = radial_flux(b, a, bore_flux, beta[0])
    wall_flux = radial_flux(c, b, contact_flux, beta[1])
    contact_temp = temperature(b, a, 5., bore_flux, beta[0], k[0])
    result = solve_annulus_contact(radii, k, (None, None), resistance, 16, formulation="circular",
                                   radial_source_coefficient=beta, wall_fluxes=(bore_flux, -wall_flux),
                                   mean_bore_temperature=5.)
    np.testing.assert_allclose(result.heat_rates, 2*np.pi*np.array([a*bore_flux,b*contact_flux,c*wall_flux]),
                               atol=1e-12, rtol=0.)
    for field, radius, start, initial, flux, conductivity, coefficient in (
            (result.inner_field, .44, a, 5., bore_flux, k[0], beta[0]),
            (result.outer_field, .83, b, contact_temp-resistance*contact_flux, contact_flux, k[1], beta[1])):
        direction = np.array([[.6,.8], [-.8,.6]])
        np.testing.assert_allclose(field.temperature(radius*direction),
                                   temperature(radius,start,initial,flux,coefficient,conductivity), atol=1e-12, rtol=0.)
        np.testing.assert_allclose(field.heat_flux(radius*direction),
                                   direction*radial_flux(radius,start,flux,coefficient), atol=1e-12, rtol=0.)


@pytest.mark.parametrize("formulation", ["polygonal", "circular"])
def test_quadratic_source_resolves_independent_cartesian_quartic(formulation):
    from tensorbem.multiloop import annulus_mesh

    k = 1.7

    def temperature(xy):
        return 1 - (xy**4).sum(axis=1)/(12*k)

    previous = None
    for n in (16, 32, 64):
        mesh = annulus_mesh(n, n, 1., .3)
        walls = [loop.centers for loop in reversed(mesh.loops)]
        if formulation == "circular":
            walls = [xy/np.cos(np.pi/n) for xy in walls]
        result = solve_annulus_contact((.3,.6,1.), (k,k), [temperature(xy) for xy in walls],
                                       panels=n, formulation=formulation, radial_source_coefficient=1.)
        errors = np.zeros(2)
        for radius, field in ((.45, result.inner_field), (.8, result.outer_field)):
            xy = radius*np.array([[.6,.8], [-.8,.6], [1.,0.]])
            errors = np.maximum(errors, [np.max(abs(field.temperature(xy)-temperature(xy))),
                                         np.max(abs(field.heat_flux(xy)-xy**3/3))])
        if formulation == "circular":
            assert np.max(errors) < 1e-12
        elif previous is not None:
            assert np.all(errors < previous/3.5)
        previous = errors


@pytest.mark.parametrize("formulation", ["circular", "polygonal"])
def test_rotated_quartic_cartesian_source(formulation, record_property):
    from tensorbem.multiloop import annulus_mesh

    k = 1.7
    axis = np.array([1., 2.])
    # Δ[(x+2y)^4] = 60*(x+2y)^2. This quartic differs from the solver's particular field.
    previous = None
    for n in (16, 32, 64):
        mesh = annulus_mesh(n, n, 1., .3)
        walls = [loop.centers for loop in reversed(mesh.loops)]
        if formulation == "circular":
            walls = [xy/np.cos(np.pi/n) for xy in walls]
        wall_t = [1-(xy@axis)**4/(60*k) for xy in walls]
        result = solve_annulus_contact((.3,.6,1.), (k,k), wall_t, panels=n,
                                       formulation=formulation, source_quadratic=(1.,4.,4.))
        errors = np.zeros(2)
        for r, field in ((.45, result.inner_field), (.8, result.outer_field)):
            xy = r*np.array([[.6,.8], [-.8,.6], [1.,0.]])
            projection = xy@axis
            flux = projection[:, None]**3*axis/15
            errors = np.maximum(errors, [np.max(abs(field.temperature(xy)-(1-projection**4/(60*k)))),
                                         np.max(abs(field.heat_flux(xy)-flux))])
        record_property(f"temperature_error_K_{n}", errors[0])
        record_property(f"flux_error_W_m2_{n}", errors[1])
        if formulation == "circular":
            assert max(errors) < 1e-12
        elif previous is not None:
            assert np.all(errors < previous/3.)
        previous = errors


def test_quadratic_source_neumann_compatibility_per_layer():
    radii = np.array([.3, .6, 1.])
    # The x*y part integrates to zero while x² and y² have equal integrals.
    coefficients = ((1., 2., 3.), (-3., 1., 2.))
    generated = np.pi*np.diff(radii**4)*np.array([4., -1.])/4
    outer_flux = -sum(generated)/(2*np.pi*radii[-1])
    def solve(flux):
        return solve_annulus_contact(radii, (1.,2.), (None,None), panels=16, formulation="circular",
                                     wall_fluxes=(0.,flux), mean_bore_temperature=3.,
                                     source_quadratic=coefficients)
    np.testing.assert_allclose(np.diff(solve(outer_flux).heat_rates), generated, rtol=0., atol=1e-12)
    with pytest.raises(ValueError, match="incompatible"):
        solve(outer_flux+.01)


@pytest.mark.parametrize("prescribe_flux",[False,True])
@pytest.mark.parametrize("panels",[17,32])
def test_circumferential_sheet_balance(panels,prescribe_flux,record_property):
    from tensorbem.benchmarks.conducting_sheet import measure
    row = measure(panels,prescribe_flux=prescribe_flux)
    for name,error in row["errors"].items():
        record_property(name,float(error))
        assert error<2e-10
    record_property("condition_number",row["condition_number"])


def test_polygonal_conducting_sheet_refinement():
    from tensorbem.benchmarks.conducting_sheet import measure
    coarse,fine = (measure(n,"polygonal") for n in (32,64))
    for key,error in fine["errors"].items():
        assert error<coarse["errors"][key]/3.2


def test_polygonal_sheet_currents_balance_each_panel_with_variable_contact():
    n = 29
    theta = 2*np.pi*(np.arange(n)+.5)/n
    source = .25+.3*np.cos(4*theta)
    result = solve_annulus_contact((.38,.72,1.04),(1.3,3.2),
              (2+.4*np.sin(theta),.35*np.cos(2*theta)),.4+.1*np.sin(3*theta),n,
              inner_resistance_fraction=.4+.2*np.cos(theta),interface_source=source,sheet_conductivity=.36)
    width = 2*.72*np.sin(np.pi/n)
    rate = result.sheet_tangential_heat_rate
    balance = width*(result.outer_interface_flux-result.interface_flux-source)+rate-np.roll(rate,1)
    np.testing.assert_allclose(balance,0.,atol=1e-12,rtol=0.)


def test_variable_circumferential_conductivity_product_rule():
    from tensorbem.benchmarks.conducting_sheet import measure
    for n in (17,32):
        result = measure(n,"circular",prescribe_flux=True,variable=True)
        assert max(result["errors"].values())<2e-10


def test_insulating_cut_blocks_tangential_heat():
    n = 24
    ks = np.full(n,.3)
    ks[::3] = 0.
    angle = 2*np.pi*(np.arange(n)+.5)/n
    result = solve_annulus_contact((.4,.7,1.),(2.,1.),(np.sin(angle),0.),.2,n,sheet_conductivity=ks)
    np.testing.assert_array_equal(result.sheet_tangential_heat_rate[::3],0.)
    assert np.max(abs(result.sheet_tangential_heat_rate))>.01


@pytest.mark.parametrize("formulation",["polygonal","circular"])
@pytest.mark.parametrize("conductance",[0.,.4])
def test_sheet_reservoir_anchors_insulated_bulk(formulation,conductance):
    n = 31
    angle = 2*np.pi*(np.arange(n)+.5)/n
    film = np.where(np.cos(angle)>0,1.5,0.)
    ambient = 2.+np.sin(angle)
    source = film*(3.-ambient)
    result = solve_annulus_contact((.4,.7,1.1),(1.2,2.3),(None,None),.3,n,formulation=formulation,
             wall_fluxes=(0.,0.),interface_source=source,sheet_conductivity=conductance,sheet_film=(film,ambient))
    np.testing.assert_allclose(result.sheet_temperature,3.,atol=2e-12,rtol=0.)
    np.testing.assert_allclose(result.sheet_reservoir_flux,source,atol=2e-12,rtol=0.)
    np.testing.assert_allclose(result.heat_rates,0.,atol=2e-12)


def test_sheet_reservoir_gauge_and_film_validation():
    kw = dict(wall_fluxes=(0.,0.),formulation="circular")
    with pytest.raises(ValueError,match="omit mean"):
        solve_annulus_contact((.4,.7,1.),(1.,1.),(None,None),sheet_film=(1.,2.),mean_bore_temperature=2.,**kw)
    with pytest.raises(ValueError,match="sheet film"):
        solve_annulus_contact((.4,.7,1.),(1.,1.),(None,None),sheet_film=(-1.,2.),**kw)


def test_equal_area_rings_share_contact_and_integrate_on_actual_panels():
    radii = (.4,.7,1.1)
    n = 24
    result = solve_annulus_contact(radii,(1.5,2.3),(2.,0.),.3,n,
                                   formulation="polygonal_equal_area")
    for field,bounds in zip((result.inner_field,result.outer_field),
                            ((radii[1],radii[0]),(radii[2],radii[1]))):
        for loop,r in zip(field.mesh.loops,bounds):
            x,y = loop.vertices.T
            area = .5*np.sum(x*np.roll(y,-1)-y*np.roll(x,-1))
            assert area == pytest.approx(np.pi*r*r,rel=1e-14)
    inside = result.inner_field.mesh.loops[0]
    outside = result.outer_field.mesh.loops[1]
    np.testing.assert_array_equal(inside.vertices,outside.vertices)
    assert result.heat_rates[1] == pytest.approx(inside.lengths@result.interface_flux)
    np.testing.assert_allclose(result.inner_trace-result.outer_trace,.3*result.interface_flux,atol=1e-13)


@pytest.mark.parametrize("n",[15,32])
def test_regular_ring_blocks_match_direct_straight_panel_integrals(n):
    from tensorbem.layered_annulus import _polygonal_ring_matrices
    from tensorbem.multiloop import annulus_mesh
    from tensorbem.solver import assemble_matrices

    mesh = annulus_mesh(n,n,1.17,.63)
    for result,expected in zip(_polygonal_ring_matrices(mesh,2.7),assemble_matrices(mesh,2.7)):
        np.testing.assert_allclose(result,expected,rtol=1e-12,atol=2e-15)


@pytest.mark.parametrize("wall", [0, 1])
def test_equal_area_mixed_wall_flux_convergence(wall):
    from tensorbem.benchmarks.annulus_prescribed_flux import measure

    coarse, fine = (measure(n, "polygonal_equal_area", wall) for n in (64, 128))
    for key in ("temperature_error_K", "interior_flux_error_W_m2", "wall_flux_error_W_m2",
                "contact_flux_error_W_m2", "rate_error_W_m"):
        assert fine[key] < coarse[key]/3.5


def test_equal_area_sheet_generation_with_bulk_sources():
    from tensorbem.benchmarks.annulus_sheet_source import angular_bulk, radial

    for reference in (radial, angular_bulk):
        coarse, fine = (reference(n, "polygonal_equal_area") for n in (64, 128))
        for key in fine:
            if "_error_" in key:
                assert fine[key] < coarse[key]/3.5


def test_circle_mean_and_heat_rate_against_rotated_quartic(record_property):
    # T = 3 + log(r) - r²/(4k) - (x+2y)^4/(60k).
    # Its angular fourth moment uses <cos^4> = 3/8.
    k = 1.7
    for n in (16, 32, 64):
        theta = (np.arange(n) + .5) * (2*np.pi/n)
        projection = np.cos(theta) + 2*np.sin(theta)
        walls = [3 + np.log(r) - r**2/(4*k) - (r*projection)**4/(60*k) for r in (.3, 1.)]
        result = solve_annulus_contact((.3, .6, 1.), (k, k), walls, panels=n,
                                       formulation="circular", volumetric_source=1.,
                                       source_quadratic=(1., 4., 4.))
        for layer, field, r in (("inner", result.inner_field, np.array([.32, .45, .57])),
                                ("outer", result.outer_field, np.array([.63, .77, .94]))):
            mean, rate = field.circular_mean_and_heat_rate(r)
            exact_mean = 3 + np.log(r) - r**2/(4*k) - 5*r**4/(32*k)
            exact_rate = 2*np.pi*(-k + r**2/2 + 5*r**4/8)
            for label, actual, exact in (("mean_K", mean, exact_mean), ("rate_W_m", rate, exact_rate)):
                error = np.max(abs(actual-exact))
                record_property(f"{layer}_{label}_{n}", error)
                assert error < 1e-11
        with pytest.raises(ValueError, match="inside"):
            result.inner_field.circular_mean_and_heat_rate([.45, .7])
    polygon = solve_annulus_contact((.3, .6, 1.), (k, k), (1., 2.), panels=8)
    with pytest.raises(ValueError, match="circular"):
        polygon.inner_field.circular_mean_and_heat_rate(.45)


@pytest.mark.parametrize("panels", [15, 32, 64])
def test_one_sided_circle_means_at_generating_contact(panels, record_property):
    bore, contact, exterior = .3, .6, 1.
    conductivity = (1.4, 2.2)
    generation = (.8, -.3)
    resistance, inner_fraction = .25, .3
    # At the contact, sheet heating drives heat away into both materials.
    q_inner, q_outer = -.4, .8
    jump = resistance * (inner_fraction*q_inner + (1-inner_fraction)*q_outer)
    interface_temperatures = (2. + jump, 2.)

    # Integrate d(r*q)/dr = r*generation, then dT/dr = -q/k from contact.
    def wall(radius, flux, source, k, initial):
        log_term = contact*flux - source*contact**2/2
        return initial - (log_term*np.log(radius/contact) + source*(radius**2-contact**2)/4)/k

    theta = (np.arange(panels)+.5)*(2*np.pi/panels)
    walls = (wall(bore, q_inner, generation[0], conductivity[0], interface_temperatures[0]) + .2*np.cos(2*theta),
             wall(exterior, q_outer, generation[1], conductivity[1], interface_temperatures[1]) + .3*np.sin(3*theta))
    result = solve_annulus_contact(
        (bore, contact, exterior), conductivity, walls, resistance, panels, formulation="circular",
        volumetric_source=generation, interface_source=q_outer-q_inner,
        inner_resistance_fraction=inner_fraction)
    ti, qi = result.inner_field.circular_mean_and_heat_rate(contact)
    to, qo = result.outer_field.circular_mean_and_heat_rate(contact)
    circumference = 2*np.pi*contact
    for label, measured, reference in (
            ("temperature_K", [ti, to], interface_temperatures),
            ("rate_W_m", [qi, qo], circumference*np.array([q_inner, q_outer])),
            ("jump_K", ti-to, jump),
            ("generation_W_m", qo-qi, circumference*(q_outer-q_inner))):
        error = np.max(np.abs(np.subtract(measured, reference)))
        record_property(label, error)
        assert error < 1e-11


@pytest.mark.parametrize("panels", [15, 32, 64])
def test_circular_boundary_field_at_off_grid_contact_angles(panels, record_property):
    a, b, c = .45, .65, .9
    ki, ko, resistance, fraction = 1.4, 2.1, .12, .35
    mode = 3
    # Exact inner radial factor (r/b)^m. Prescribe the sheet generation
    # amplitude, then determine the two outer harmonic coefficients at b.
    qi = -ki*mode/b
    qo = qi + 2.
    outer_value = 1 - resistance*(fraction*qi + (1-fraction)*qo)
    difference = -qo*b/(ko*mode)
    growing, decaying = (outer_value+difference)/2, (outer_value-difference)/2
    nodes = (np.arange(panels)+.5)*2*np.pi/panels
    wall_mode = np.sin(mode*nodes+.23)
    walls = ((a/b)**mode*wall_mode, (growing*(c/b)**mode+decaying*(b/c)**mode)*wall_mode)
    result = solve_annulus_contact((a,b,c), (ki,ko), walls, resistance, panels,
                                   formulation="circular", interface_source=2*wall_mode,
                                   inner_resistance_fraction=fraction)
    angles = 2*np.pi*(np.arange(73)+.37)/73
    radial = np.column_stack((np.cos(angles), np.sin(angles)))
    tangent = np.column_stack((-radial[:, 1], radial[:, 0]))
    for side, field, boundary, value, flux, k in (
            ("inner", result.inner_field, "outer", 1., qi, ki),
            ("outer", result.outer_field, "inner", outer_value, qo, ko)):
        temperature, vector = field.circular_boundary_field(boundary, angles)
        harmonic = np.sin(mode*angles+.23)
        exact_flux = (flux*harmonic)[:, None]*radial - (
            k*value*mode/b*np.cos(mode*angles+.23))[:, None]*tangent
        errors = (np.max(abs(temperature-value*harmonic)), np.linalg.norm(vector-exact_flux, axis=1).max())
        for label, error in zip(("temperature_K", "vector_flux_W_m2"), errors):
            record_property(f"{side}_{label}", error)
            assert error < 1e-11
        starts = np.array([.11, 5.9, 1.7, .2, .9])
        ends = np.array([1.3, 6.8, -.1, .2+2*np.pi, .9])
        # Integrate the manufactured radial flux directly at arbitrary cuts.
        primitive_difference = (np.cos(mode*starts+.23)-np.cos(mode*ends+.23))/mode
        arc_rate = field.circular_arc_heat_rate(boundary, starts, ends)
        arc_error = np.max(abs(arc_rate - b*flux*primitive_difference))
        record_property(f"{side}_arc_W_m", arc_error)
        assert arc_error < 1e-11
        if side == "inner":
            incoming = arc_rate
        else:
            sheet_error = np.max(abs(arc_rate - incoming - 2*b*primitive_difference))
            record_property("arc_generation_balance_W_m", sheet_error)
            assert sheet_error < 1e-11


def test_arc_integral_retains_mean_and_even_grid_nyquist_mode(record_property):
    n, radius = 16, .4
    # At the bore the sampled cosine alternates sign about the nonzero mean.
    bore_flux = 2. + .7*(-1.)**np.arange(n)
    result = solve_annulus_contact((radius, .7, 1.), (1.2, 2.), (None, 0.),
                                   panels=n, formulation="circular", wall_fluxes=(bore_flux, None))
    start = .17
    ends = np.array([.171, 1.3, start+2*np.pi, start-2*np.pi])
    m, offset = n//2, np.pi/n
    reference = radius*(2*(ends-start) + .7/m*(np.sin(m*(ends-offset))-np.sin(m*(start-offset))))
    measured = result.inner_field.circular_arc_heat_rate("inner", start, ends)
    error = np.max(abs(measured-reference))
    record_property("nyquist_arc_W_m", error)
    assert error < 1e-11
