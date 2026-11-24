"""Steady conduction in two concentric annuli with a thermal contact."""
from dataclasses import dataclass

import numpy as np

from .multiloop import MultiLoopMesh, _evaluate_traces, annulus_mesh
from .solver import assemble_matrices, _MAX_CONDITION


@dataclass(frozen=True)
class AnnulusLayerField:
    """Solved traces for one polygonal material region; targets must be interior."""
    mesh: MultiLoopMesh
    boundary_values: np.ndarray
    normal_derivative: np.ndarray
    conductivity: float
    reference_length: float
    circular_geometry: bool = False

    def temperature(self, points):
        if self.circular_geometry:
            return self.circular_field(points)[0]
        return _evaluate_traces(self.mesh, self.boundary_values, self.normal_derivative,
                                self.reference_length, points, gradient=False)

    def heat_flux(self, points):
        """Cartesian Fourier heat flux in W/m², shape (number of targets, 2)."""
        if self.circular_geometry:
            return self.circular_field(points)[1]
        return -self.conductivity * _evaluate_traces(
            self.mesh, self.boundary_values, self.normal_derivative,
            self.reference_length, points, gradient=True)


    def circular_field(self, points):
        """Temperature and Cartesian flux from harmonic extension of both traces.

        For the regular concentric rings produced by solve_annulus_contact,
        interpret each midpoint temperature as a sample on the nominal circle
        at the same angle. Targets are strictly between those two circles.
        """
        xy = np.asarray(points, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
            raise ValueError("points must be a finite array of shape (M, 2)")
        radius = np.hypot(xy[:, 0], xy[:, 1])
        outer, inner = (np.linalg.norm(loop.vertices[0]) for loop in self.mesh.loops)
        if np.any((radius <= inner) | (radius >= outer)):
            raise ValueError("targets must be inside the selected circular material")
        span = np.log(outer/inner)
        log_r = np.log(radius/inner)
        angle = np.arctan2(xy[:, 1], xy[:, 0])
        temperature = np.zeros_like(radius)
        radial = np.zeros_like(radius)  # derivative with respect to log(r)
        angular = np.zeros_like(radius)
        for loop, sl, distance, sign in zip(self.mesh.loops, self.mesh.loop_slices,
                                            (log_r, span-log_r), (1, -1)):
            n = loop.panel_count
            coeff = np.fft.rfft(self.boundary_values[sl]) / n
            temperature += coeff[0].real * distance/span
            radial += sign * coeff[0].real/span
            phase = angle - np.arctan2(loop.centers[0, 1], loop.centers[0, 0])
            for m in range(1, len(coeff)):
                # sinh(m*distance)/sinh(m*span) in bounded exponentials.
                decay = np.exp(-m*(span-distance)) / -np.expm1(-2*m*span)
                weight = decay * -np.expm1(-2*m*distance)
                slope = m * decay * (1 + np.exp(-2*m*distance))
                multiplicity = 1 if 2*m == n else 2
                wave = multiplicity * coeff[m] * np.exp(1j*m*phase)
                temperature += weight * wave.real
                radial += sign * slope * wave.real
                angular -= m * weight * wave.imag
        direction = xy / radius[:, None]
        tangent = np.column_stack((-direction[:, 1], direction[:, 0]))
        flux = -(self.conductivity/radius[:, None]) * (radial[:, None]*direction + angular[:, None]*tangent)
        return temperature, flux


@dataclass(frozen=True)
class AnnulusContactResult:
    inner_trace: np.ndarray
    outer_trace: np.ndarray
    interface_flux: np.ndarray  # W/m², positive toward increasing radius
    heat_rates: tuple[float, float, float]  # bore, contact, exterior, W/m of axis
    condition_number: float  # coupled interface system
    inner_field: AnnulusLayerField | None = None
    outer_field: AnnulusLayerField | None = None


def _circular_normal_map(outer, inner, n):
    """Map temperature samples to outward normal derivatives on exact circles."""
    span = np.log(outer/inner)
    modes = np.arange(1, n//2 + 1)
    z = modes*span
    # Harmonic radial basis in log(r). Zero mode is linear in log(r).
    diagonal = np.r_[1/span, modes / np.tanh(z)]
    coupling = np.r_[1/span, 2*modes*np.exp(-z) / -np.expm1(-2*z)]
    shifts = (np.arange(n)[:, None] - np.arange(n)) % n
    own = np.fft.irfft(diagonal, n=n)[shifts]
    other = np.fft.irfft(coupling, n=n)[shifts]
    return np.block([[own/outer, -other/outer], [-other/inner, own/inner]])


def _wall_response(normal_map, interface, wall, ambient, film, inward_flux):
    """Eliminate the wall temperature; return its constant and linear parts."""
    n = normal_map.shape[0]//2
    if inward_flux is not None:
        if ambient is not None or film is not None:
            raise ValueError("a prescribed flux requires None for wall temperature and film")
        h = np.zeros(n)
        load = np.broadcast_to(np.asarray(inward_flux, dtype=float), (n,))
        if not np.isfinite(load).all():
            raise ValueError("prescribed wall flux must be finite")
    else:
        ambient = np.broadcast_to(np.asarray(ambient, dtype=float), (n,))
        if not np.isfinite(ambient).all():
            raise ValueError("wall temperatures must be finite")
        if film is None:
            return ambient, None
        h = np.broadcast_to(np.asarray(film, dtype=float), (n,))
        if not np.isfinite(h).all() or np.any(h < 0):
            raise ValueError("film coefficients must be finite and nonnegative")
        load = h * ambient
    # k*dT/dn + h*Twall = h*Tambient, with the material-outward normal.
    loads = np.column_stack((load, -normal_map[wall, interface]))
    response = np.linalg.solve(normal_map[wall, wall] + np.diag(h), loads)
    return response[:, 0], response[:, 1:]


def solve_annulus_contact(radii, conductivities, wall_temperatures, resistance=0., panels=32, *,
                          formulation="polygonal", film_coefficients=(None, None), wall_fluxes=(None, None)):
    """Coupled two-material conduction on concentric annuli.

    Supply radii (bore, contact, exterior), two positive conductivities, and
    inner/outer Dirichlet data (scalars or panel vectors in azimuthal order).
    film_coefficients is a (bore, exterior) pair. None prescribes the wall
    temperature. A scalar or panel vector h in W/(m² K) instead uses that
    wall's temperature data as the ambient fluid temperature. Zero h insulates.
    A wall_fluxes entry prescribes inward W/m² (scalar or azimuthal samples);
    set the corresponding wall temperature and film to None. At least one
    wall must have a prescribed temperature or a nonzero film anchor.
    The areal resistance is in m² K/W and may vary by interface panel.
    The default polygonal formulation uses straight constant panels. The
    circular formulation collocates the contact law at the same angles and
    uses the Fourier Dirichlet-to-Neumann map of each nominal annulus.
    """
    if formulation not in ("polygonal", "circular"):
        raise ValueError("formulation must be polygonal or circular")
    circular = formulation == "circular"
    bore, contact, exterior = radii
    ki, ko = conductivities
    if not np.isfinite([ki, ko]).all() or min(ki, ko) <= 0:
        raise ValueError("conductivities must be finite and positive")
    inner = annulus_mesh(panels, panels, contact, bore)
    outer = annulus_mesh(panels, panels, exterior, contact)
    n = inner.loops[0].panel_count
    hot, cold = wall_temperatures
    bore_flux, exterior_flux = wall_fluxes
    r = np.broadcast_to(np.asarray(resistance, dtype=float), (n,))
    if not np.isfinite(r).all() or np.any(r < 0):
        raise ValueError("resistance must be finite and nonnegative")

    def normal_map(mesh):
        if circular:
            return _circular_normal_map(*(np.linalg.norm(loop.vertices[0]) for loop in mesh.loops), n)
        single, double = assemble_matrices(mesh, reference_length=2 * exterior)
        return np.linalg.solve(single, double + .5 * np.eye(mesh.panel_count))

    # Each map takes [outer trace, inner trace] to domain-outward derivatives.
    di = ki * normal_map(inner)
    do = ko * normal_map(outer)
    # Unknowns: outer material's contact trace and outward radial heat flux.
    # Inner trace = outer trace + R*q; Fourier flux reverses sign on the hole.
    first, second = slice(0, n), slice(n, 2*n)
    bore_film, exterior_film = film_coefficients
    if all(q is not None or (h is not None and np.all(np.asarray(h) == 0))
           for q, h in zip(wall_fluxes, film_coefficients)):
        raise ValueError("at least one wall must anchor the temperature")
    bore_base, bore_response = _wall_response(di, first, second, hot, bore_film, bore_flux)
    ext_base, ext_response = _wall_response(do, second, first, cold, exterior_film, exterior_flux)
    inner_map = di[first, first]
    outer_map = do[second, second]
    if bore_response is not None:
        inner_map = inner_map + di[first, second] @ bore_response
    if ext_response is not None:
        outer_map = outer_map + do[second, first] @ ext_response
    eye = np.eye(n)
    system = np.block([[inner_map, inner_map * r + eye],
                       [outer_map, -eye]])
    rhs = -np.concatenate((di[first, second] @ bore_base, do[second, first] @ ext_base))
    condition = float(np.linalg.cond(system))
    if not np.isfinite(condition) or condition > _MAX_CONDITION:
        raise ValueError(f"ill-conditioned interface system ({condition:.3e})")
    outer_trace, flux = np.split(np.linalg.solve(system, rhs), 2)
    inner_trace = outer_trace + r * flux
    bore_temp = bore_base if bore_response is None else bore_base + bore_response @ inner_trace
    ext_temp = ext_base if ext_response is None else ext_base + ext_response @ outer_trace
    ti = np.concatenate((inner_trace, bore_temp))
    to = np.concatenate((ext_temp, outer_trace))
    dni = (di @ ti) / ki
    dno = (do @ to) / ko
    rates = (float(ki * dni[n:] @ inner.loops[1].lengths), float(flux @ inner.loops[0].lengths),
             float(-ko * dno[:n] @ outer.loops[0].lengths))
    if circular:
        # Equal angular panels on the nominal circles use arc length.
        arc_to_chord = (np.pi/n) / np.sin(np.pi/n)
        rates = tuple(float(q * arc_to_chord) for q in rates)
    return AnnulusContactResult(inner_trace, outer_trace, flux, rates, condition,
                                AnnulusLayerField(inner, ti, dni, ki, 2 * exterior, circular),
                                AnnulusLayerField(outer, to, dno, ko, 2 * exterior, circular))
