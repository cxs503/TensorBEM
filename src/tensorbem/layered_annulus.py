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
