import numpy as np
import pytest

from tensorbem.harmonic_rings import assemble_ring_response


def test_massless_contacts_recover_exact_dc_with_complex_walls():
    radii = np.array([0.4, 0.5, 0.8, 1.2])
    k = [1., 3., 2.]
    contacts = [0.2, 0.5]
    walls = [1+2j, -0.3j]
    response = assemble_ring_response(radii, k, contacts, 16)
    result = response.solve(13., walls, [0., 0.])
    resistance = sum(np.log(radii[1:] / radii[:-1]) / k) + sum(contacts / radii[1:3])
    expected = (walls[0] - walls[1]) / (resistance * radii[1:3])
    np.testing.assert_allclose(result.incoming_flux, expected, rtol=1e-11)
    np.testing.assert_allclose(result.outgoing_flux, expected, rtol=1e-11)
    np.testing.assert_allclose(result.wall_heat_rates, 2*np.pi*(walls[0]-walls[1])/resistance, rtol=1e-11)
    dc = response.solve(0., walls, [1., 2.])
    np.testing.assert_allclose(dc.sheet_temperature, result.sheet_temperature)
    assert dc.join_power_residual < 1e-11


def test_mode_above_resolution_rejected():
    with pytest.raises(ValueError, match="Nyquist"):
        assemble_ring_response([1, 2, 3, 4], [1]*3, [0]*2, 16, mode=8)


@pytest.mark.parametrize("generation", [(1+2j, 0), (0, -0.7j)])
@pytest.mark.parametrize("omega", [0., 4.])
def test_localized_heater_balances_each_sheet(generation, omega):
    response = assemble_ring_response([0.4, 0.5, 0.8, 1.2], [1., 3., 2.], [0.2, 0.5], 16)
    capacities = np.array([0.2, 0.7])
    fields = response.solve(omega, [0, 0], capacities, sheet_generation=generation)
    np.testing.assert_allclose(
        fields.outgoing_flux - fields.incoming_flux + 1j * omega * capacities * fields.sheet_temperature,
        generation, atol=1e-12)
    sheet_net_input = np.asarray(generation) - 1j*omega*capacities*fields.sheet_temperature
    expected_gain = (2*np.pi*np.array([.5, .8])) @ sheet_net_input
    np.testing.assert_allclose(np.diff(fields.wall_heat_rates), expected_gain, atol=1e-12)


@pytest.mark.parametrize("fractions", [(0., 1.), (1., 0.), (.2, .7)])
def test_asymmetric_dc_sheet_temperature(fractions):
    radii = np.array([.4, .5, .8, 1.2])
    k = [1., 3., 2.]
    contact = np.array([.2, .5])
    response = assemble_ring_response(radii, k, contact, 16, inner_resistance_fractions=fractions)
    result = response.solve(0., [1., 0.], [.4, .8])
    bulk = np.log(radii[1:]/radii[:-1])/k
    jumps = contact/radii[1:-1]
    rate = 1/(sum(bulk) + sum(jumps))
    expected = 1-rate*(np.cumsum(bulk[:2]) + [0., jumps[0]] + fractions*jumps)
    np.testing.assert_allclose(result.sheet_temperature, expected, atol=1e-12, rtol=0)
    left = 1-rate*(np.cumsum(bulk[:2]) + [0., jumps[0]])
    np.testing.assert_allclose(result.contact_temperatures, np.column_stack((left, left-rate*jumps)), atol=1e-12)


@pytest.mark.parametrize("films", [(7., None), (None, 3.), (7., 3.)])
def test_axisymmetric_ring_films_against_dc_resistance(films):
    radii = np.array([.4, .55, .9, 1.1])
    contacts = np.array([.5, .9])
    k = (1., 2.5, .8)
    ambient = np.array([1.+.2j, -.3j])
    response = assemble_ring_response(radii, k, contacts, 16, film_coefficients=films)
    result = response.solve(0., ambient, (.25, .8))
    film_resistance = sum(1/(h*r) for h, r in zip(films, radii[[0, -1]]) if h is not None)
    resistance = sum(np.diff(np.log(radii))/k) + sum(contacts/radii[1:-1]) + film_resistance
    np.testing.assert_allclose(result.wall_heat_rates, 2*np.pi*(ambient[0]-ambient[1])/resistance, rtol=1e-11)
    film_drops = np.array([0. if h is None else 1/(r*h) for h, r in zip(films, radii[[0, -1]])])
    expected_surface = ambient + np.array([-1., 1.])*film_drops*(ambient[0]-ambient[1])/resistance
    np.testing.assert_allclose(result.wall_surface_temperatures, expected_surface, rtol=1e-11)


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_localized_axisymmetric_volume_source_heats_correct_annulus(layer):
    edges = np.array([.4, .55, .9, 1.1])
    response = assemble_ring_response(edges, (1., 2.5, .8), (.5, .9), 16, include_volume_sources=True)
    density = np.zeros(3, dtype=complex)
    density[layer] = 1.+.3j
    result = response.solve(0., [0., 0.], [.25, .8], volume_generation=density)
    interface_rates = 2*np.pi*edges[1:-1]*result.incoming_flux
    rates = np.r_[result.wall_heat_rates[0], interface_rates, result.wall_heat_rates[1]]
    np.testing.assert_allclose(np.diff(rates), np.pi*np.diff(edges**2)*density, rtol=1e-11, atol=1e-12)


def test_uniform_sheet_temperature_under_balanced_reservoir_loading():
    rings = assemble_ring_response([.4, .55, .9, 1.1], [1., 2.5, .8], [.5, .9], 16)
    t = .7-.2j
    h = np.array([1.3, 4.2])
    ambient = np.array([-.2j, .5+.1j])
    capacity = np.array([.25, .8])
    generation = 8j*capacity*t + h*(t-ambient)
    result = rings.solve(8., [t, t], capacity, sheet_generation=generation,
                         sheet_film_coefficients=h, sheet_ambient_temperatures=ambient)
    np.testing.assert_allclose(result.sheet_temperature, t, atol=1e-12)
    np.testing.assert_allclose(result.wall_heat_rates, 0., atol=1e-12)


@pytest.mark.parametrize("side", ["bore", "exterior"])
def test_mixed_flux_reproduces_dc_wall_temperature(side):
    radii = np.array([.4, .55, .9, 1.1])
    k = (1., 2.5, .8)
    contact = np.array([.5, .9])
    i = ("bore", "exterior").index(side)
    imposed = .4+.7j
    rate = (1 if i == 0 else -1)*2*np.pi*radii[[0, -1]][i]*imposed
    resistance = (sum(np.diff(np.log(radii))/k) + sum(contact/radii[1:-1]))/(2*np.pi)
    wall = [None, 0.] if i == 0 else [0., None]
    solution = assemble_ring_response(radii, k, contact, 16, flux_wall=side).solve(
        0., wall, (.25, .8), wall_inward_flux=imposed)
    np.testing.assert_allclose(solution.wall_heat_rates, rate, rtol=1e-11)
    np.testing.assert_allclose(-np.diff(solution.wall_surface_temperatures), rate*resistance, rtol=1e-11)


def test_double_flux_frequency_balance_and_missing_temperature_anchor():
    radii = np.array([.4, .55, .9, 1.1])
    response = assemble_ring_response(radii, [1., 2.5, .8], [.5, .9], 16, flux_wall="both")
    flux = np.array([.45-.25j, -.17+.31j])
    capacity = np.array([.25, .8])
    with pytest.raises(ValueError, match="active sheet storage"):
        response.solve(0., (None, None), capacity, wall_inward_flux=flux)
    result = response.solve(8., (None, None), capacity, wall_inward_flux=flux)
    power = 2*np.pi*radii[[0,-1]]*flux
    np.testing.assert_allclose(result.wall_heat_rates, power*[1,-1], atol=1e-12)
    np.testing.assert_allclose(8j*(2*np.pi*radii[1:-1]*capacity) @ result.sheet_temperature,
                               sum(power), atol=1e-12)


def test_double_flux_steady_cosine_mode_without_storage_or_reservoir():
    radii = np.array([.4, .55, .9, 1.1])
    # T=r*cos(theta) has constant radial flux amplitude -k.
    response = assemble_ring_response(radii, [2.]*3, [0., 0.], 16, mode=1, flux_wall="both")
    result = response.solve(0., (None, None), [0., 0.], wall_inward_flux=(-2., 2.))
    np.testing.assert_allclose(result.wall_surface_temperatures, radii[[0,-1]], atol=1e-12)
    np.testing.assert_allclose(result.sheet_temperature, radii[1:-1], atol=1e-12)


def test_floating_flux_compatibility_and_temperature_shift():
    edges = np.array([.4, .55, .9, 1.1])
    response = assemble_ring_response(edges, [1., 2.5, .8], [.5, .9], 16, flux_wall="both",
                                      include_volume_sources=True)
    flux = (1.+.2j, -edges[0]/edges[-1]*(1.+.2j))
    options = dict(omega=0., wall_temperatures=(None, None), sheet_capacities=(.25, .8), wall_inward_flux=flux)
    with pytest.raises(ValueError, match="incompatible"):
        response.solve(**options, bore_temperature_reference=0., volume_generation=(0., 1., 0.))
    base = response.solve(**options, bore_temperature_reference=0.)
    shifted = response.solve(**options, bore_temperature_reference=1e9)
    np.testing.assert_array_equal(shifted.wall_heat_rates, base.wall_heat_rates)
    np.testing.assert_array_equal(shifted.incoming_flux, base.incoming_flux)
    np.testing.assert_array_equal(shifted.contact_temperatures, base.contact_temperatures+1e9)
