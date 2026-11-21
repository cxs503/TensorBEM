"""Linear deep-water heave/pitch solver with uniform or double-body hull flow."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .double_body_flow import DoubleBodyFlow
from .drift import panel_surface_gradient
from .free_surface_solver3d import (
    _solve_system,
    assemble_free_surface_operators_3d,
    incident_airy_wave_3d,
    rigid_body_normals_3d,
)
from .geometry3d import WettedHullMesh
from .stf_fullspeed import encounter_frequency


@dataclass(frozen=True)
class ForwardSpeedHeavePitch:
    """Rows are generalized forces and columns are radiating motions (heave, pitch).

    Excitation is per metre of incident amplitude. Frequencies are in rad/s.
    The impedance convention is C - omega_e² (M + A) - i omega_e B.
    """

    wave_omega: float
    encounter_omega: float
    added_mass: NDArray[np.float64]
    damping: NDArray[np.float64]
    excitation: NDArray[np.complex128]
    relative_residual: float
    condition_number: float


def solve_forward_speed_heave_pitch(
    mesh: WettedHullMesh,
    omega: float,
    forward_speed: float,
    *,
    center=(0.0, 0.0, 0.0),
    density: float = 1025.0,
    gravity: float = 9.80665,
    base_flow: DoubleBodyFlow | None = None,
) -> ForwardSpeedHeavePitch:
    """Solve the radiation and scattering BVPs for a body advancing along +x.

    Incident waves propagate along -x at the intrinsic omega. The outgoing
    Green function uses the encounter frequency (small-speed approximation).
    The base flow in body axes is uniform at -U. With exp(-i omega_e t),
    pressure is rho (i omega_e phi + U dphi/dx). The rigid pitch velocity
    includes the gradient of its displacement field. Integration is about
    the supplied rotation center for both radiation and excitation.

    Pass a prepared DoubleBodyFlow for hull-tangent convection and double-body
    m-terms. This option retains the encounter-frequency Green function, the
    mean wetted surface and the original hydrostatics. It computes Eulerian
    unsteady pressure. Steady-pressure shape forces and base-flow distortion
    of the free-surface condition are outside this approximation.
    """
    if not all(np.isfinite(x) and x > 0 for x in (omega, density, gravity)):
        raise ValueError("omega, density and gravity must be positive and finite")
    if not np.isfinite(forward_speed) or forward_speed < 0:
        raise ValueError("forward_speed must be non-negative and finite")
    if base_flow is not None and base_flow.mesh is not mesh:
        raise ValueError("base_flow must be prepared on the supplied mesh")
    we = encounter_frequency(omega, forward_speed, np.pi, gravity)
    modes = rigid_body_normals_3d(mesh, center=center)[:, [2, 4]]
    incident, grad_incident = incident_airy_wave_3d(
        mesh.centroids, omega**2 / gravity, wave_angle=np.pi, gravity=gravity
    )
    # Body-outward normal velocity for unit displacement. For pitch about y,
    # d(e_y cross r)/dx = -e_z. Scattering cancels the incident normal flow.
    normal_rad = -1j * we * modes
    if base_flow is None:
        normal_rad[:, 1] += forward_speed * mesh.normals[:, 2]
    else:
        offset = np.zeros(3) if center is None else np.asarray(center)
        pitch_m = (base_flow.rotation_m - np.cross(offset, base_flow.translation_m))[
            :, 1
        ]
        normal_rad += forward_speed * np.column_stack(
            (base_flow.translation_m[:, 2], pitch_m)
        )
    scattered_normal = -np.einsum("ij,ij->i", grad_incident, mesh.normals)
    normal = np.column_stack([normal_rad, scattered_normal])

    ops = assemble_free_surface_operators_3d(mesh, we**2 / gravity)
    # The direct BIE uses the fluid-domain exterior normal, opposite to n_body.
    potential, residual, _, _ = _solve_system(ops, -normal)
    pressure = 1j * we * potential
    if base_flow is None:
        # The incident Airy pressure in uniform flow uses the intrinsic frequency.
        if forward_speed:
            for j in range(3):
                tangent = panel_surface_gradient(mesh, potential[:, j], method="nodal")
                dx = tangent[:, 0] + mesh.normals[:, 0] * normal[:, j]
                pressure[:, j] += forward_speed * dx
        pressure[:, 2] += 1j * omega * incident
    else:
        pressure[:, 2] += 1j * we * incident
        if forward_speed:
            velocity = forward_speed * base_flow.velocity
            for j in range(3):
                tangent = panel_surface_gradient(mesh, potential[:, j], method="nodal")
                pressure[:, j] -= np.einsum("ij,ij->i", velocity, tangent)
            pressure[:, 2] -= np.einsum("ij,ij->i", velocity, grad_incident)
    forces = -density * (mesh.areas[:, None] * modes).T @ pressure
    return ForwardSpeedHeavePitch(
        omega,
        we,
        forces[:, :2].real / we**2,
        forces[:, :2].imag / we,
        forces[:, 2],
        residual,
        ops.condition_number,
    )
