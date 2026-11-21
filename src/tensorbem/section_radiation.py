"""Deep-water sectional heave radiation and double-ended heave/pitch strip theory."""

from dataclasses import dataclass

import numpy as np

from .geometry import PanelMesh
from .free_surface_solver import (
    _rankine_normal_derivative_matrix,
    _free_surface_only_normal_derivative,
    _evaluate_source_potential,
    _validate_panel_nodes,
)
from .stf_fullspeed import _double_ended_speed_terms


@dataclass(frozen=True)
class SectionHeaveRadiation:
    """Per unit ship length, for unit heave velocity in exp(-i omega t)."""

    added_mass: float
    damping: float
    energy_damping: float
    relative_residual: float
    condition_number: float


@dataclass(frozen=True)
class _WettedPanels:
    """Geometry of the open wetted polygon, excluding its waterline chord."""

    vertices: np.ndarray
    vectors: np.ndarray
    normals: np.ndarray
    centers: np.ndarray
    lengths: np.ndarray

    @property
    def panel_count(self):
        return len(self.lengths)


def solve_section_heave(
    vertices, omega, *, density=1025.0, gravity=9.80665, panel_nodes=8
):
    """Solve for the radiation of a 2-D section crossing the calm free surface.

    Supply the wetted contour in (y,z), starting at the left waterline, tracing
    below the body, and ending at the right waterline. The source density is
    constant on each physical segment. Both the collocation points and all
    integration nodes lie below z=0. No source is placed on the waterplane.
    Returned mass is kg/m and damping is kg/(m s). The independent wave-energy
    damping estimate helps assess resolution when the pressure integral is small.
    """
    if not all(np.isfinite(x) and x > 0 for x in (omega, density, gravity)):
        raise ValueError("omega, density and gravity must be finite and positive")
    nodes = _validate_panel_nodes(panel_nodes)
    mesh = PanelMesh(vertices)
    v = mesh.vertices
    if np.any(v[[0, -1], 1] != 0) or v[0, 0] >= v[-1, 0] or np.any(v[1:-1, 1] >= 0):
        raise ValueError(
            "the wetted contour must connect its left and right waterline endpoints below z=0"
        )
    wet = _WettedPanels(
        v[:-1],
        mesh.vectors[:-1],
        mesh.normals[:-1],
        mesh.centers[:-1],
        mesh.lengths[:-1],
    )
    k = omega**2 / gravity
    system = _rankine_normal_derivative_matrix(
        wet
    ) + _free_surface_only_normal_derivative(wet, k, nodes)
    system.flat[:: wet.panel_count + 1] += 0.5
    q = wet.normals[:, 1]
    sources = np.linalg.solve(system, q)
    phi = _evaluate_source_potential(wet, sources, k, nodes)
    force = -density * np.dot(phi, q * wet.lengths)
    # The free-surface Green function and its outward-wave residue give the
    # two outgoing waves directly from the source distribution (unit velocity).
    # Integrate exp(k z +/- i k y) exactly along every straight segment.
    radiated = []
    for direction in (-1, 1):
        z_plus_iy = wet.vertices[:, 1] + 1j * direction * wet.vertices[:, 0]
        step = k * (wet.vectors[:, 1] + 1j * direction * wet.vectors[:, 0])
        # Integrate towards deeper water to keep both exponentials bounded.
        reverse = step.real > 0
        start = np.where(reverse, k * z_plus_iy + step, k * z_plus_iy)
        step = np.where(reverse, -step, step)
        radiated.append(
            np.dot(sources, wet.lengths * np.exp(start) * np.expm1(step) / step)
        )
    b_energy = density * omega * np.mean(np.abs(radiated) ** 2)
    residual = np.linalg.norm(system @ sources - q) / np.linalg.norm(q)
    return SectionHeaveRadiation(
        float(force.real),
        float(omega * force.imag),
        float(b_energy),
        float(residual),
        float(np.linalg.cond(system)),
    )


def double_ended_strip_radiation(
    stations, weights, added_mass, damping, encounter_omega, forward_speed
):
    """Integrate 2-D heave coefficients on a hull with vanishing end sections.

    Stations are x forward from the pitch origin, weights are longitudinal
    quadrature weights in m. Inputs are per unit length at encounter_omega.
    Return dimensional A,B with force rows and heave/pitch displacement columns.
    Fully wetted transoms require additional end terms.
    """
    x, w, a, b = (
        np.asarray(v, dtype=float) for v in (stations, weights, added_mass, damping)
    )
    if (
        x.ndim != 1
        or x.size < 2
        or any(v.shape != x.shape or not np.all(np.isfinite(v)) for v in (x, w, a, b))
        or np.any(w <= 0)
        or np.any(np.diff(x) <= 0)
    ):
        raise ValueError(
            "supply increasing stations with matching finite coefficients and positive weights"
        )
    if not (
        np.isfinite(encounter_omega)
        and encounter_omega > 0
        and np.isfinite(forward_speed)
        and forward_speed >= 0
    ):
        raise ValueError(
            "encounter_omega must be positive and forward_speed non-negative, both finite"
        )
    modes = np.column_stack((np.ones_like(x), -x))
    A = modes.T @ ((w * a)[:, None] * modes)
    B = modes.T @ ((w * b)[:, None] * modes)
    da, db = _double_ended_speed_terms(A[0, 0], B[0, 0], encounter_omega, forward_speed)
    return A + da, B + db
