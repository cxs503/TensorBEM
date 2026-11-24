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

    def temperature(self, points):
        return _evaluate_traces(self.mesh, self.boundary_values, self.normal_derivative,
                                self.reference_length, points, gradient=False)

    def heat_flux(self, points):
        """Cartesian Fourier heat flux in W/m², shape (number of targets, 2)."""
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


def solve_annulus_contact(radii, conductivities, wall_temperatures, resistance=0., panels=32):
    """Constant straight-panel collocation with coincident interface panels.

    Supply radii (bore, contact, exterior), two positive conductivities, and
    inner/outer Dirichlet data (scalars or panel vectors in azimuthal order).
    The areal resistance is in m² K/W and may vary by interface panel.
    """
    bore, contact, exterior = radii
    ki, ko = conductivities
    if not np.isfinite([ki, ko]).all() or min(ki, ko) <= 0:
        raise ValueError("conductivities must be finite and positive")
    inner = annulus_mesh(panels, panels, contact, bore)
    outer = annulus_mesh(panels, panels, exterior, contact)
    n = inner.loops[0].panel_count
    hot, cold, r = (np.broadcast_to(np.asarray(v, dtype=float), (n,))
                    for v in (*wall_temperatures, resistance))
    if not all(np.isfinite(v).all() for v in (hot, cold, r)) or np.any(r < 0):
        raise ValueError("wall temperatures must be finite and resistance finite and nonnegative")

    def normal_map(mesh):
        single, double = assemble_matrices(mesh, reference_length=2 * exterior)
        return np.linalg.solve(single, double + .5 * np.eye(mesh.panel_count))

    # Each map takes [outer trace, inner trace] to domain-outward derivatives.
    di = ki * normal_map(inner)
    do = ko * normal_map(outer)
    # Unknowns: outer material's contact trace and outward radial heat flux.
    # Inner trace = outer trace + R*q; Fourier flux reverses sign on the hole.
    eye = np.eye(n)
    system = np.block([[di[:n, :n], di[:n, :n] * r + eye],
                       [do[n:, n:], -eye]])
    rhs = -np.concatenate((di[:n, n:] @ hot, do[n:, :n] @ cold))
    condition = float(np.linalg.cond(system))
    if not np.isfinite(condition) or condition > _MAX_CONDITION:
        raise ValueError(f"ill-conditioned interface system ({condition:.3e})")
    outer_trace, flux = np.split(np.linalg.solve(system, rhs), 2)
    inner_trace = outer_trace + r * flux
    ti = np.concatenate((inner_trace, hot))
    to = np.concatenate((cold, outer_trace))
    dni = (di @ ti) / ki
    dno = (do @ to) / ko
    rates = (float(ki * dni[n:] @ inner.loops[1].lengths), float(flux @ inner.loops[0].lengths),
             float(-ko * dno[:n] @ outer.loops[0].lengths))
    return AnnulusContactResult(inner_trace, outer_trace, flux, rates, condition,
                                AnnulusLayerField(inner, ti, dni, ki, 2 * exterior),
                                AnnulusLayerField(outer, to, dno, ko, 2 * exterior))
