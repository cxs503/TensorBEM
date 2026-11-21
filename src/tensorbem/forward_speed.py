"""Linear deep-water rigid-body hydrodynamics at forward speed."""

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
    """Rows are generalized forces and columns are radiating motions in requested order.

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


# Generic code can refer to the result independently of the requested mode set.
ForwardSpeedModes = ForwardSpeedHeavePitch


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
    return solve_forward_speed_modes(
        mesh,
        omega,
        forward_speed,
        wave_angle=np.pi,
        dofs=(2, 4),
        center=center,
        density=density,
        gravity=gravity,
        base_flow=base_flow,
    )


def solve_forward_speed_modes(
    mesh: WettedHullMesh,
    omega: float,
    forward_speed: float,
    *,
    wave_angle: float,
    dofs=(0, 1, 2, 3, 4, 5),
    center=(0.0, 0.0, 0.0),
    density: float = 1025.0,
    gravity: float = 9.80665,
    base_flow: DoubleBodyFlow | None = None,
) -> ForwardSpeedModes:
    """Radiation matrices and complex excitation in the caller's rigid-mode order.

    The propagation heading is measured from positive x in radians, with pi
    denoting head seas. The approximation uses the encounter-frequency deep-water
    Green function. A negative encounter frequency takes the conjugate outgoing
    radiation branch. At exact wave synchronism the harmonic solver is singular.
    Mode numbers 0..5 correspond to surge, sway, heave, roll, pitch and yaw.
    """
    selected = np.asarray(dofs)
    if (
        selected.ndim != 1
        or not len(selected)
        or selected.dtype.kind not in "iu"
        or np.any((selected < 0) | (selected > 5))
        or len(np.unique(selected)) != len(selected)
    ):
        raise ValueError("dofs must contain distinct rigid-mode indices from 0 to 5")
    if not np.isfinite(wave_angle):
        raise ValueError("wave_angle must be finite")
    if not all(np.isfinite(x) and x > 0 for x in (omega, density, gravity)):
        raise ValueError("omega, density and gravity must be positive and finite")
    if not np.isfinite(forward_speed) or forward_speed < 0:
        raise ValueError("forward_speed must be non-negative and finite")
    if base_flow is not None and base_flow.mesh is not mesh:
        raise ValueError("base_flow must be prepared on the supplied mesh")
    we = encounter_frequency(omega, forward_speed, wave_angle, gravity)
    if abs(we) <= 32 * np.finfo(float).eps * omega:
        raise ValueError("zero encounter frequency is outside the harmonic solver")
    modes = rigid_body_normals_3d(mesh, center=center)[:, selected]
    incident, grad_incident = incident_airy_wave_3d(
        mesh.centroids, omega**2 / gravity, wave_angle=wave_angle, gravity=gravity
    )
    # Body-outward normal velocity for unit displacement. For pitch about y,
    # d(e_y cross r)/dx = -e_z. Scattering cancels the incident normal flow.
    normal_rad = -1j * we * modes
    if base_flow is None:
        # -U times the x derivative of e_j cross r for each rotational mode.
        correction = np.zeros((mesh.panel_count, 6))
        correction[:, 4] = mesh.normals[:, 2]
        correction[:, 5] = -mesh.normals[:, 1]
    else:
        offset = np.zeros(3) if center is None else np.asarray(center)
        correction = np.column_stack(
            (
                base_flow.translation_m,
                base_flow.rotation_m - np.cross(offset, base_flow.translation_m),
            )
        )
    normal_rad += forward_speed * correction[:, selected]
    scattered_normal = -np.einsum("ij,ij->i", grad_incident, mesh.normals)
    normal = np.column_stack([normal_rad, scattered_normal])

    ops = assemble_free_surface_operators_3d(mesh, we**2 / gravity)
    # The direct BIE uses the fluid-domain exterior normal, opposite to n_body.
    potential, residual, _, _ = _solve_system(
        ops, -normal if we > 0 else -normal.conj()
    )
    if we < 0:
        potential = potential.conj()
    pressure = 1j * we * potential
    if base_flow is None:
        # The incident Airy pressure in uniform flow uses the intrinsic frequency.
        if forward_speed:
            for j in range(normal.shape[1]):
                tangent = panel_surface_gradient(mesh, potential[:, j], method="nodal")
                dx = tangent[:, 0] + mesh.normals[:, 0] * normal[:, j]
                pressure[:, j] += forward_speed * dx
        pressure[:, -1] += 1j * omega * incident
    else:
        pressure[:, -1] += 1j * we * incident
        if forward_speed:
            velocity = forward_speed * base_flow.velocity
            for j in range(normal.shape[1]):
                tangent = panel_surface_gradient(mesh, potential[:, j], method="nodal")
                pressure[:, j] -= np.einsum("ij,ij->i", velocity, tangent)
            pressure[:, -1] -= np.einsum("ij,ij->i", velocity, grad_incident)
    forces = -density * (mesh.areas[:, None] * modes).T @ pressure
    return ForwardSpeedModes(
        omega,
        we,
        forces[:, :-1].real / we**2,
        forces[:, :-1].imag / we,
        forces[:, -1],
        residual,
        ops.condition_number,
    )
