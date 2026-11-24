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
