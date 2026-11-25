"""Harmonic conduction through three concentric materials with two storing contacts."""
from dataclasses import dataclass

import numpy as np

from .geometry import _validate_panel_count
from .layered_annulus import solve_annulus_contact


@dataclass(frozen=True)
class RingFields:
    """Complex cosine amplitudes in order of increasing radius. Flux is positive outward.

    wall_heat_rates contains the bore and exterior radial rates (W/m axial length),
    both positive toward increasing radius. Mode zero uses the integrated BEM wall
    rates. wall_surface_temperatures contains the two physical Dirichlet or Robin
    temperature traces (K), ordered bore then exterior. Higher modes report 2πr times the cosine amplitude of radial flux.
    contact_temperatures has one row per ring, with the smaller- and larger-radius
    material temperature traces (K) in its two columns.
    """

    sheet_temperature: np.ndarray
    incoming_flux: np.ndarray
    outgoing_flux: np.ndarray
    join_power_residual: float
    wall_heat_rates: np.ndarray | None = None
    wall_surface_temperatures: np.ndarray | None = None
    contact_temperatures: np.ndarray | None = None


@dataclass(frozen=True)
class RingResponse:
    """Preassembled real response for repeated frequencies or wall excitations."""

    _maps: np.ndarray
    mode_leakage: float
    _volume_maps: np.ndarray | None = None
    _flux_index: int | None = None

    def solve(self, omega, wall_temperatures, sheet_capacities, *, sheet_generation=(0., 0.), volume_generation=None,
              sheet_film_coefficients=(0., 0.), sheet_ambient_temperatures=(0., 0.), wall_inward_flux=None):
        """Apply two wall or ambient temperature phasors under exp(i omega t), with capacities in J/(m² K).

        For a prescribed flux wall, set its temperature entry to None and supply
        wall_inward_flux in W/m², positive into the annulus at either wall.
        sheet_generation gives the two imposed heat generation phasors in W/m²,
        positive for heat added to a sheet in the assembled cosine mode.
        Each sheet also receives h*(ambient - sheet temperature) in W/m² from
        its own reservoir; zero sheet film coefficient disables that exchange.
        volume_generation supplies three complex coefficients (W/m³) multiplying
        (r/exterior_radius)^m cos(mθ), when assembled with include_volume_sources.
        """
        if self._flux_index is None:
            if wall_inward_flux is not None:
                raise ValueError("select flux_wall when assembling a prescribed wall flux")
            walls = np.asarray(wall_temperatures, dtype=complex)
        else:
            if (np.shape(wall_temperatures) != (2,) or wall_temperatures[self._flux_index] is not None
                    or wall_inward_flux is None or np.ndim(wall_inward_flux) != 0):
                raise ValueError("use None at the flux wall and supply a scalar wall_inward_flux")
            data = list(wall_temperatures)
            data[self._flux_index] = wall_inward_flux
            walls = np.asarray(data, dtype=complex)
        capacities = np.asarray(sheet_capacities, dtype=float)
        generation = np.asarray(sheet_generation, dtype=complex)
        sheet_h = np.asarray(sheet_film_coefficients)
        if (sheet_h.shape != (2,) or not np.isrealobj(sheet_h) or
                not np.isfinite(sheet_h).all() or np.any(sheet_h < 0)):
            raise ValueError("two finite nonnegative sheet film coefficients required")
        ambient = np.asarray(sheet_ambient_temperatures, dtype=complex)
        if ambient.shape != (2,) or not np.isfinite(ambient).all():
            raise ValueError("two finite sheet ambient phasors required")
        if generation.shape != (2,) or not np.isfinite(generation).all():
            raise ValueError("two finite sheet generation phasors required")
        if walls.shape != (2,) or not np.isfinite(walls).all():
            raise ValueError("two finite wall temperatures required")
        if capacities.shape != (2,) or not np.isfinite(capacities).all() or np.any(capacities < 0):
            raise ValueError("two finite nonnegative areal capacities required")
        if not np.isscalar(omega) or not np.isrealobj(omega) or not np.isfinite(omega) or omega < 0:
            raise ValueError("omega must be a finite nonnegative angular frequency")

        left, right = self._maps
        admittance = sheet_h + 1j * omega * capacities
        applied = generation + sheet_h*ambient
        # Unknowns are the common join temperature and the two sheet sources.
        # Current matches at the join. The net sheet source includes applied heat, reservoir exchange and storage.
        matrix = [
            [left[1, 1] - right[0, 0], left[1, 2], -right[0, 2]],
            [admittance[0] * left[2, 1], 1 + admittance[0] * left[2, 2], 0],
            [admittance[1] * right[2, 0], 0, 1 + admittance[1] * right[2, 2]],
        ]
        base_left = left[:, 0] * walls[0]
        base_right = right[:, 1] * walls[1]
        if volume_generation is not None:
            density = np.asarray(volume_generation, dtype=complex)
            if self._volume_maps is None or density.shape != (3,) or not np.isfinite(density).all():
                raise ValueError("volume_generation requires volume source maps and three finite phasors")
            base_left += self._volume_maps[0] @ density[:2]
            base_right += self._volume_maps[1] @ density[1:]
        forcing = [base_right[0] - base_left[1], applied[0] - admittance[0]*base_left[2],
                   applied[1] - admittance[1]*base_right[2]]
        join, source1, source2 = np.linalg.solve(matrix, forcing)
        a = base_left + left[:, 1]*join + left[:, 2]*source1
        b = base_right + right[:, 0]*join + right[:, 2]*source2
        temperatures, incoming, outgoing = np.stack((a[2:5], b[2:5])).T
        return RingFields(temperatures, incoming, outgoing, float(abs(a[1] - b[0])), np.array([a[0], b[1]]),
                          np.array([a[5], b[6]]), np.stack((a[7:9], b[7:9])))


def assemble_ring_response(radii, conductivities, contact_resistances, panels=32, *,
                           mode=0, formulation="circular", join_radius=None,
                           inner_resistance_fractions=(.5, .5), sheet_conductivities=(0., 0.),
                           film_coefficients=(None, None), include_volume_sources=False, flux_wall=None):
    """Assemble one cosine mode for a quasi-steady bulk and two resistive contacts.

    Supply four increasing radii (bore, two sheets, exterior), one conductivity per
    material and one area-specific resistance per sheet. inner_resistance_fractions
    sets the share of each resistance on the smaller-radius side. The optional join radius
    splits the middle material for subdomain matching. Each circle has 'panels'
    midpoint collocation points. sheet_conductivities gives the two uniform
    circumferential conductivities times thickness (W/K). At each physical wall,
    a positive film coefficient (W/m²/K) uses its supplied temperature as the ambient
    phasor, while None prescribes the surface temperature.
    flux_wall may be "bore" or "exterior" to prescribe an inward flux there.
    include_volume_sources assembles polynomial volume loads for modes 0, 1 and 2. Both circular and polygonal formulations are supported.
    """
    n = _validate_panel_count(panels)
    if isinstance(mode, (bool, np.bool_)) or not isinstance(mode, (int, np.integer)) or not 0 <= mode < n/2:
        raise ValueError("mode must be a nonnegative integer below the Nyquist limit")
    radii = np.asarray(radii, dtype=float)
    if radii.shape != (4,) or not np.isfinite(radii).all() or radii[0] <= 0 or np.any(np.diff(radii) <= 0):
        raise ValueError("four strictly increasing positive radii required")
    if join_radius is None:
        join_radius = radii[1] + (radii[2] - radii[1]) / 2
    if not radii[1] < join_radius < radii[2]:
        raise ValueError("join radius must lie between the sheets")
    if np.shape(conductivities) != (3,) or np.shape(contact_resistances) != (2,):
        raise ValueError("three conductivities and two contact resistances required")
    fractions = np.asarray(inner_resistance_fractions, dtype=float)
    if fractions.shape != (2,) or not np.all((fractions >= 0) & (fractions <= 1)):
        raise ValueError("two inner resistance fractions in [0, 1] required")
    sheet_k = np.asarray(sheet_conductivities, dtype=float)
    if sheet_k.shape != (2,) or not np.isfinite(sheet_k).all() or np.any(sheet_k < 0):
        raise ValueError("two finite nonnegative sheet conductivities required")
    if np.shape(film_coefficients) != (2,) or any(
            h is not None and (not np.isrealobj(h) or not np.isfinite(h) or h <= 0)
            for h in film_coefficients):
        raise ValueError("supply a positive finite film coefficient or None for each physical wall")
    if flux_wall not in (None, "bore", "exterior"):
        raise ValueError("flux_wall must be None, bore or exterior")
    flux_index = None if flux_wall is None else ("bore", "exterior").index(flux_wall)
    if flux_index is not None and film_coefficients[flux_index] is not None:
        raise ValueError("a prescribed flux wall cannot also have a film")
    domain_films = ((film_coefficients[0], None), (None, film_coefficients[1]))
    domains = ((radii[0], radii[1], join_radius), (join_radius, radii[2], radii[3]))
    pattern = np.cos(mode * 2*np.pi * (np.arange(n) + .5) / n)
    projection = pattern / (pattern @ pattern)
    if include_volume_sources and mode > 2:
        raise ValueError("polynomial volume sources support modes 0, 1 and 2")
    maps = []
    volume_maps = []
    leakage = 0.
    for i, domain in enumerate(domains):
        columns = []
        for unit in np.eye(5 if include_volume_sources else 3):
            left, right, source = unit[:3]
            volume = {}
            if include_volume_sources:
                amplitudes = unit[3:] / radii[-1]**mode
                if mode == 0:
                    volume["volumetric_source"] = amplitudes
                elif mode == 1:
                    volume["source_gradient"] = np.outer(amplitudes, (1., 0.))
                else:
                    volume["source_quadratic"] = np.outer(amplitudes, (1., 0., -1.))
            temperatures = [left*pattern, right*pattern]
            inward_fluxes = [None, None]
            if i == flux_index:
                inward_fluxes[i], temperatures[i] = temperatures[i], None
            result = solve_annulus_contact(
                domain, conductivities[i:i+2], temperatures,
                contact_resistances[i], n, formulation=formulation, interface_source=source*pattern,
                inner_resistance_fraction=fractions[i], sheet_conductivity=sheet_k[i],
                film_coefficients=domain_films[i], wall_fluxes=inward_fluxes, **volume)
            bore = conductivities[i] * result.inner_field.normal_derivative[n:]
            exterior = -conductivities[i+1] * result.outer_field.normal_derivative[:n]
            traces = np.stack((2*np.pi*domain[0]*bore, 2*np.pi*domain[2]*exterior,
                               result.sheet_temperature, result.interface_flux, result.outer_interface_flux,
                               result.inner_field.boundary_values[n:], result.outer_field.boundary_values[:n],
                               result.inner_trace, result.outer_trace))
            amplitudes = traces @ projection
            leakage = max(leakage, float(abs(traces - amplitudes[:, None]*pattern).max()))
            if mode == 0:
                amplitudes[:2] = result.heat_rates[0], result.heat_rates[2]
            columns.append(amplitudes)
        maps.append(np.column_stack(columns[:3]))
        if include_volume_sources:
            volume_maps.append(np.column_stack(columns[3:]))
    return RingResponse(np.stack(maps), leakage, np.stack(volume_maps) if volume_maps else None, flux_index)
