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
