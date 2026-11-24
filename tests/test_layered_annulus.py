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
