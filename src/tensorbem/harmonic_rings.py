"""Harmonic conduction through three concentric materials with two storing contacts."""
from dataclasses import dataclass

import numpy as np

from .geometry import _validate_panel_count
from .layered_annulus import solve_annulus_contact


@dataclass(frozen=True)
class RingFields:
    """Complex cosine amplitudes in order of increasing radius. Flux is positive outward."""

    sheet_temperature: np.ndarray
    incoming_flux: np.ndarray
    outgoing_flux: np.ndarray
    join_power_residual: float


@dataclass(frozen=True)
class RingResponse:
    """Preassembled real response for repeated frequencies or wall excitations."""

    _maps: np.ndarray
    mode_leakage: float

    def solve(self, omega, wall_temperatures, sheet_capacities):
        """Apply two Dirichlet phasors under exp(i omega t), with capacities in J/(m² K)."""
        walls = np.asarray(wall_temperatures, dtype=complex)
        capacities = np.asarray(sheet_capacities, dtype=float)
        if walls.shape != (2,) or not np.isfinite(walls).all():
            raise ValueError("two finite wall temperatures required")
        if capacities.shape != (2,) or not np.isfinite(capacities).all() or np.any(capacities < 0):
            raise ValueError("two finite nonnegative areal capacities required")
        if not np.isscalar(omega) or not np.isrealobj(omega) or not np.isfinite(omega) or omega < 0:
            raise ValueError("omega must be a finite nonnegative angular frequency")

        left, right = self._maps
        storage = 1j * omega * capacities
        # Unknowns are the common join temperature and the two sheet sources.
        # Current matches at the join. Each sheet source equals -i omega C T.
        matrix = [
            [left[1, 1] - right[0, 0], left[1, 2], -right[0, 2]],
            [storage[0] * left[2, 1], 1 + storage[0] * left[2, 2], 0],
            [storage[1] * right[2, 0], 0, 1 + storage[1] * right[2, 2]],
        ]
        forcing = [right[0, 1] * walls[1] - left[1, 0] * walls[0],
                   -storage[0] * left[2, 0] * walls[0], -storage[1] * right[2, 1] * walls[1]]
        join, source1, source2 = np.linalg.solve(matrix, forcing)
        a = left @ [walls[0], join, source1]
        b = right @ [join, walls[1], source2]
        temperatures, incoming, outgoing = np.stack((a[2:], b[2:])).T
        return RingFields(temperatures, incoming, outgoing, float(abs(a[1] - b[0])))


def assemble_ring_response(radii, conductivities, contact_resistances, panels=32, *,
                           mode=0, formulation="circular", join_radius=None):
    """Assemble one cosine mode for a quasi-steady bulk and two symmetric resistive contacts.

    Supply four increasing radii (bore, two sheets, exterior), one conductivity per
    material and one area-specific resistance per sheet. The optional join radius
    splits the middle material for subdomain matching. Each circle has 'panels'
    midpoint collocation points. Both circular and polygonal formulations are supported.
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
    domains = ((radii[0], radii[1], join_radius), (join_radius, radii[2], radii[3]))
    pattern = np.cos(mode * 2*np.pi * (np.arange(n) + .5) / n)
    projection = pattern / (pattern @ pattern)
    maps = []
    leakage = 0.
    for i, domain in enumerate(domains):
        columns = []
        for left, right, source in np.eye(3):
            result = solve_annulus_contact(
                domain, conductivities[i:i+2], (left*pattern, right*pattern),
                contact_resistances[i], n, formulation=formulation, interface_source=source*pattern)
            bore = conductivities[i] * result.inner_field.normal_derivative[n:]
            exterior = -conductivities[i+1] * result.outer_field.normal_derivative[:n]
            traces = np.stack((2*np.pi*domain[0]*bore, 2*np.pi*domain[2]*exterior,
                               result.sheet_temperature, result.interface_flux, result.outer_interface_flux))
            amplitudes = traces @ projection
            leakage = max(leakage, float(abs(traces - amplitudes[:, None]*pattern).max()))
            if mode == 0:
                amplitudes[:2] = result.heat_rates[0], result.heat_rates[2]
            columns.append(amplitudes)
        maps.append(np.column_stack(columns))
    return RingResponse(np.stack(maps), leakage)
