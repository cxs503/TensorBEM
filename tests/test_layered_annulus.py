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
