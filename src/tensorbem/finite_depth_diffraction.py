"""Wave diffraction and Haskind excitation for a 2-D body in water of finite depth.

Why this module exists
----------------------
:mod:`tensorbem.finite_depth_solver` delivered the *radiation* problem in a
strip of constant depth ``h`` (added mass, damping, far-field amplitude).  The
deep-water solver :mod:`tensorbem.free_surface_solver` also carries the
diffraction problem, the incident Airy wave and the Haskind-Newman reciprocity
identity.  Without this module the finite-depth capability stopped at
radiation: a ship or section in a channel could not be loaded by an incident
wave, which is the quantity a seakeeping calculation actually needs.

What is added
-------------
* :func:`incident_wave` -- the finite-depth incident Airy potential and its
  outward-normal derivative.  With ``z`` up, the mean free surface at ``z = 0``,
  the bottom at ``z = -h`` and the factor ``exp(-i*omega*t)``,

      phi_I = -(i g A / omega) * cosh(k0 (z + h)) / cosh(k0 h)
                                  * exp(i * direction * k0 * x),

  where ``k0`` solves ``k0 tanh(k0 h) = nu``, ``nu = omega**2 / g``.  It
  satisfies ``d(phi_I)/dz = nu phi_I`` on ``z = 0`` and ``d(phi_I)/dz = 0`` on
  ``z = -h`` identically, and reduces to the deep-water
  ``-(i g A / omega) exp(nu z + i nu x)`` as ``k0 h -> infinity``.
* :func:`solve_diffraction_indirect` -- the diffraction problem with the same
  indirect (single-layer) finite-depth operator as the radiation solve,
  ``(K' + I/2) sigma_S = -d(phi_I)/dn``, the total potential
  ``phi = phi_I + phi_S``, the exciting force, and the reflection/transmission
  coefficients read off the propagating mode of the kernel.
* :func:`haskind_excitation_force` -- the exciting force from *radiation data
  alone*,

      X_j = -i omega rho * integral_Gamma (phi_I n_j - phi_j d(phi_I)/dn) ds,

  which uses no diffraction solve.  Its agreement with the direct pressure
  integral of :func:`solve_diffraction_indirect` is an exact reciprocity check
  on the discretisation, and it is the finite-depth counterpart of the
  deep-water identity already verified for the Ursell cylinder.

Conventions are copied verbatim from the deep-water code (``-i*omega*rho`` for
the force, ``exp(-i*omega*t)`` time factor, outward normal ``n`` pointing into
the fluid, ``direction = +1`` for a wave travelling towards ``+x``) so that the
two solvers can be compared element by element in the deep-water limit.
"""

from __future__ import annotations

import math

import numpy as np

from .finite_depth import (
    finite_depth_wavenumber,
    green_function,
    green_gradient,
    propagating_mode_norm,
)
from .finite_depth_solver import (
    _evaluate_source_potential,
    _validate_depth_inputs,
    assemble_indirect_matrix,
)
from .free_surface_solver import (
    RadiationResult,
    _GRAVITY,
    _validate_positive_scalar,
    _validate_wave_direction,
)
from .geometry import PanelMesh

__all__ = [
    "FiniteDepthDiffractionResult",
    "incident_wave",
    "solve_diffraction_indirect",
    "haskind_excitation_force",
    "diffraction_reference",
    "excitation_reference_limit",
]


class FiniteDepthDiffractionResult:
    """Solution of one finite-depth diffraction problem.

    Attributes mirror :class:`tensorbem.free_surface_solver.DiffractionResult`
    so that the two solvers can be diffed field by field.
    """

    __slots__ = (
        "total_potential",
        "scattered_potential",
        "incident_potential",
        "incident_normal_velocity",
        "source_density",
        "excitation_force",
        "reflection_coefficient",
        "transmission_coefficient",
        "energy_balance_error",
        "condition_number",
        "relative_residual",
    )

    def __init__(
        self,
        *,
        total_potential: np.ndarray,
        scattered_potential: np.ndarray,
        incident_potential: np.ndarray,
        incident_normal_velocity: np.ndarray,
        source_density: np.ndarray,
        excitation_force: np.ndarray,
        reflection_coefficient: complex,
        transmission_coefficient: complex,
        energy_balance_error: float,
        condition_number: float,
        relative_residual: float,
    ) -> None:
        self.total_potential = total_potential
        self.scattered_potential = scattered_potential
        self.incident_potential = incident_potential
        self.incident_normal_velocity = incident_normal_velocity
        self.source_density = source_density
        self.excitation_force = excitation_force
        self.reflection_coefficient = reflection_coefficient
        self.transmission_coefficient = transmission_coefficient
        self.energy_balance_error = energy_balance_error
        self.condition_number = condition_number
        self.relative_residual = relative_residual

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            "FiniteDepthDiffractionResult("
            f"excitation_force={self.excitation_force!r}, "
            f"reflection={self.reflection_coefficient!r}, "
            f"transmission={self.transmission_coefficient!r}, "
            f"energy_balance_error={self.energy_balance_error!r})"
        )


def incident_wave(
    points: np.ndarray,
    normals: np.ndarray,
    nu: float,
    depth: float,
    amplitude: float = 1.0,
    *,
    gravity: float = _GRAVITY,
    direction: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-depth incident Airy potential and outward-normal velocity.

    ``phi_I = -(i g A / omega) * cosh(k0 (z + h)) / cosh(k0 h) * exp(i s k0 x)``
    with ``s = direction``.  Returns ``(phi_I, d(phi_I)/dn)`` evaluated at the
    supplied points.  Points must lie strictly inside the strip ``-h < z < 0``.
    """
    k = _validate_positive_scalar("nu", nu)
    h = _validate_positive_scalar("depth", depth)
    amp = _validate_positive_scalar("amplitude", amplitude)
    g = _validate_positive_scalar("gravity", gravity)
    sign = _validate_wave_direction(direction)
    coords = np.asarray(points, dtype=float)
    nrm = np.asarray(normals, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or nrm.shape != coords.shape:
        raise ValueError("points and normals must both have shape (N, 2)")
    if np.any(coords[:, 1] >= 0.0):
        raise ValueError("evaluation points must lie strictly below z = 0")
    if np.any(coords[:, 1] <= -h):
        raise ValueError("evaluation points must lie strictly above the bottom")

    k0 = finite_depth_wavenumber(k, h)
    omega = math.sqrt(k * g)
    vertical = np.cosh(k0 * (coords[:, 1] + h)) / math.cosh(k0 * h)
    phase = vertical * np.exp(1.0j * sign * k0 * coords[:, 0])
    phi_inc = -(1.0j * g * amp / omega) * phase
    # d/dx = i*s*k0, d/dz = k0 tanh(k0 (z+h)); the bottom derivative vanishes.
    dphi_dn = phi_inc * (
        1.0j * sign * k0 * nrm[:, 0] + k0 * np.tanh(k0 * (coords[:, 1] + h)) * nrm[:, 1]
    )
    return phi_inc, dphi_dn


def _mode_amplitudes(
    x: np.ndarray,
    z: np.ndarray,
    weights: np.ndarray,
    nu: float,
    depth: float,
    *,
    gravity: float,
) -> tuple[complex, complex]:
    """Elevation amplitudes of the ``+x`` and ``-x`` travelling waves.

    The propagating mode of the finite-depth kernel has far field
    ``-i/(2 k0 N0) phi0(z) phi0(z0) exp(i k0 |x - x0|)``, so a distribution of
    strengths ``weights`` (area element included) radiates, towards ``+x``,

        eta_a = (i omega / g) cosh(k0 h) * [-i/(2 k0 N0)]
                * sum_j exp(-i k0 x_j) phi0(z_j) w_j,

    and the ``-x`` amplitude is the same sum with ``exp(+i k0 x_j)``.
    ``wave_amplitude`` in :mod:`tensorbem.finite_depth_solver` is exactly the
    first of these for a panel density; the second is built here so that the
    reflection and transmission coefficients can be normalised consistently.
    """
    k0 = finite_depth_wavenumber(nu, depth)
    norm = propagating_mode_norm(k0, depth)
    omega = math.sqrt(nu * gravity)
    phi0 = np.cosh(k0 * (np.asarray(z, dtype=float) + depth))
    scale = 1.0j * omega / gravity * math.cosh(k0 * depth) * (-1.0j / (2.0 * k0 * norm))
    density = np.asarray(weights, dtype=complex) * phi0
    plus = scale * complex(np.sum(density * np.exp(-1.0j * k0 * np.asarray(x, dtype=float))))
    minus = scale * complex(np.sum(density * np.exp(1.0j * k0 * np.asarray(x, dtype=float))))
    return plus, minus


def _propagating_amplitudes(
    mesh: PanelMesh,
    sources: np.ndarray,
    nu: float,
    depth: float,
    *,
    gravity: float,
) -> tuple[complex, complex]:
    """Panel-density wrapper of :func:`_mode_amplitudes`."""
    return _mode_amplitudes(
        mesh.centers[:, 0],
        mesh.centers[:, 1],
        sources * mesh.lengths,
        nu,
        depth,
        gravity=gravity,
    )


def solve_diffraction_indirect(
    mesh: PanelMesh,
    nu: float,
    depth: float,
    amplitude: float = 1.0,
    *,
    density: float = 1025.0,
    gravity: float = _GRAVITY,
    direction: int = 1,
    panel_nodes: int = 8,
    modes: int | None = None,
) -> FiniteDepthDiffractionResult:
    """Solve the finite-depth diffraction problem for a submerged closed body.

    The scattered potential is a single-layer finite-depth source distribution
    with ``(K' + I/2) sigma_S = -d(phi_I)/dn``; ``K'`` is the same matrix used
    by the radiation solve, so the only new ingredient is the incident wave.
    Reflection and transmission are normalised by the incident elevation
    amplitude, and ``|R|^2 + |T|^2 = 1`` for a body that neither absorbs nor
    transmits energy is the corresponding energy check (the group velocity is
    the same in both directions, so the flux ratio is the elevation ratio).
    """
    nu, depth = _validate_depth_inputs(mesh, nu, depth)
    amp = _validate_positive_scalar("amplitude", amplitude)
    rho = _validate_positive_scalar("density", density)
    g = _validate_positive_scalar("gravity", gravity)
    sign = _validate_wave_direction(direction)
    if panel_nodes < 2:
        raise ValueError("panel_nodes must be at least 2")

    matrix = assemble_indirect_matrix(mesh, nu, depth, panel_nodes=panel_nodes, modes=modes)
    system = matrix + 0.5 * np.eye(mesh.panel_count)
    phi_inc, dphi_inc_dn = incident_wave(
        mesh.centers, mesh.normals, nu, depth, amp, gravity=g, direction=sign
    )
    rhs = -dphi_inc_dn
    sources = np.linalg.solve(system, rhs)
    phi_scat = _evaluate_source_potential(mesh, sources, nu, depth, panel_nodes, modes)
    phi_total = phi_inc + phi_scat

    omega = math.sqrt(nu * g)
    excitation_force = -1.0j * omega * rho * ((phi_total * mesh.lengths) @ mesh.normals)

    plus, minus = _propagating_amplitudes(mesh, sources, nu, depth, gravity=g)
    scattered_plus = plus / amp
    scattered_minus = minus / amp
    if sign == 1:
        reflection = scattered_minus
        transmission = 1.0 + scattered_plus
    else:
        reflection = scattered_plus
        transmission = 1.0 + scattered_minus
    energy_error = float(abs(reflection) ** 2 + abs(transmission) ** 2 - 1.0)

    residual = system @ sources - rhs
    return FiniteDepthDiffractionResult(
        total_potential=phi_total,
        scattered_potential=phi_scat,
        incident_potential=phi_inc,
        incident_normal_velocity=dphi_inc_dn,
        source_density=sources,
        excitation_force=np.asarray(excitation_force, dtype=complex),
        reflection_coefficient=complex(reflection),
        transmission_coefficient=complex(transmission),
        energy_balance_error=energy_error,
        condition_number=float(np.linalg.cond(system)),
        relative_residual=float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-300)),
    )


def haskind_excitation_force(
    mesh: PanelMesh,
    radiation: RadiationResult,
    nu: float,
    depth: float,
    amplitude: float = 1.0,
    *,
    density: float = 1025.0,
    gravity: float = _GRAVITY,
    direction: int = 1,
) -> complex:
    """Exciting force from radiation data alone (finite-depth Haskind-Newman).

    Green's second identity applied to the unit-velocity radiation potential
    ``phi_j`` (``d(phi_j)/dn = n_j``) and to ``phi = phi_I + phi_S`` cancels
    both the body integral of ``phi_S`` and the contribution at infinity
    (both potentials radiate outward), leaving

        X_j = -i omega rho * integral_Gamma (phi_I n_j - phi_j d(phi_I)/dn) ds.

    Nothing about the diffraction solve enters, so comparing this with the
    pressure integral of :func:`solve_diffraction_indirect` is a reciprocity
    check that the two independent discretisations agree.
    """
    if not isinstance(radiation, RadiationResult):
        raise TypeError("radiation must be a RadiationResult")
    nu, depth = _validate_depth_inputs(mesh, nu, depth)
    amp = _validate_positive_scalar("amplitude", amplitude)
    rho = _validate_positive_scalar("density", density)
    g = _validate_positive_scalar("gravity", gravity)
    sign = _validate_wave_direction(direction)
    if radiation.potential.shape != (mesh.panel_count,) or radiation.normal_velocity.shape != (
        mesh.panel_count,
    ):
        raise ValueError("radiation traces must match mesh.panel_count")

    phi_inc, dphi_inc_dn = incident_wave(
        mesh.centers, mesh.normals, nu, depth, amp, gravity=g, direction=sign
    )
    omega = math.sqrt(nu * g)
    integrand = phi_inc * radiation.normal_velocity - radiation.potential * dphi_inc_dn
    return complex(-1.0j * omega * rho * np.sum(integrand * mesh.lengths))


def diffraction_reference(
    mesh: PanelMesh,
    nu: float,
    depth: float,
    amplitude: float = 1.0,
    *,
    source_count: int = 32,
    source_radius_ratio: float = 0.55,
    density: float = 1025.0,
    gravity: float = _GRAVITY,
    direction: int = 1,
    modes: int | None = None,
) -> FiniteDepthDiffractionResult:
    """Independent diffraction reference by the method of fundamental solutions.

    Point sources are placed on a circle inside the body and the body Neumann
    condition is collocated at the panel centres, exactly as in
    :func:`tensorbem.finite_depth_solver.radiation_reference` for the radiation
    problem.  The kernel is the same finite-depth one, but the discretisation
    shares nothing with the panel method: no piecewise-constant density, no
    singular quadrature, no matrix entries in common.  It therefore tests the
    diffraction solve itself rather than only the reciprocity identity, which a
    consistently wrong incident-wave derivative would satisfy as well.
    """
    nu, depth = _validate_depth_inputs(mesh, nu, depth)
    amp = _validate_positive_scalar("amplitude", amplitude)
    rho = _validate_positive_scalar("density", density)
    g = _validate_positive_scalar("gravity", gravity)
    sign = _validate_wave_direction(direction)
    if isinstance(source_count, (bool, np.bool_)) or int(source_count) < 4:
        raise ValueError("source_count must be an integer >= 4")
    if not 0.0 < float(source_radius_ratio) < 1.0:
        raise ValueError("source_radius_ratio must lie strictly between 0 and 1")

    centre = np.mean(mesh.centers, axis=0)
    radius = float(np.mean(np.linalg.norm(mesh.centers - centre, axis=1)))
    angle = 2.0 * np.pi * np.arange(int(source_count)) / int(source_count)
    sources_xy = centre + float(source_radius_ratio) * radius * np.column_stack(
        (np.cos(angle), np.sin(angle))
    )
    if np.any(sources_xy[:, 1] >= 0.0):
        raise ValueError("fundamental solutions must lie below the free surface")

    targets = mesh.centers
    normals = mesh.normals
    phi_inc, dphi_inc_dn = incident_wave(
        targets, normals, nu, depth, amp, gravity=g, direction=sign
    )
    gx, gz = green_gradient(
        targets[:, 0][:, None],
        targets[:, 1][:, None],
        sources_xy[None, :, 0],
        sources_xy[None, :, 1],
        nu,
        depth,
        modes=modes,
        include_rankine=True,
    )
    kernel = gx * normals[:, 0][:, None] + gz * normals[:, 1][:, None]
    rhs = -dphi_inc_dn
    strengths, *_ = np.linalg.lstsq(kernel, rhs, rcond=None)
    phi_scat = (
        green_function(
            targets[:, 0][:, None],
            targets[:, 1][:, None],
            sources_xy[None, :, 0],
            sources_xy[None, :, 1],
            nu,
            depth,
            modes=modes,
        )
        @ strengths
    )
    phi_total = phi_inc + phi_scat

    omega = math.sqrt(nu * g)
    excitation_force = -1.0j * omega * rho * ((phi_total * mesh.lengths) @ normals)
    plus, minus = _mode_amplitudes(
        sources_xy[:, 0], sources_xy[:, 1], strengths, nu, depth, gravity=g
    )
    scattered_plus = plus / amp
    scattered_minus = minus / amp
    if sign == 1:
        reflection = scattered_minus
        transmission = 1.0 + scattered_plus
    else:
        reflection = scattered_plus
        transmission = 1.0 + scattered_minus
    energy_error = float(abs(reflection) ** 2 + abs(transmission) ** 2 - 1.0)
    return FiniteDepthDiffractionResult(
        total_potential=phi_total,
        scattered_potential=phi_scat,
        incident_potential=phi_inc,
        incident_normal_velocity=dphi_inc_dn,
        source_density=strengths,
        excitation_force=np.asarray(excitation_force, dtype=complex),
        reflection_coefficient=complex(reflection),
        transmission_coefficient=complex(transmission),
        energy_balance_error=energy_error,
        condition_number=float(np.linalg.cond(kernel)),
        relative_residual=float(np.linalg.norm(kernel @ strengths - rhs) / max(np.linalg.norm(rhs), 1e-300)),
    )


def excitation_reference_limit(
    radius: float,
    submergence: float,
    nu: float,
    amplitude: float = 1.0,
    *,
    density: float = 1025.0,
    gravity: float = _GRAVITY,
    added_mass_coefficient: float = 1.0,
) -> complex:
    """Long-wave (Froude-Krylov + added-mass) heave excitation of a deep section.

    For ``k a -> 0`` and a body far from the free surface the fluid acceleration
    is uniform over the body, so the heave exciting force per unit span tends to

        X_3 -> rho * pi * a**2 * (1 + C_a) * omega**2 * A,

    with ``C_a = 1`` the exact added-mass coefficient of a circular cylinder in
    unbounded fluid.  This is an independent analytical anchor for the
    *magnitude* (not just the reciprocity) of the finite-depth excitation: it
    requires ``k0 a -> 0`` and ``k0 f -> 0`` (free surface invisible) at the same
    time, i.e. a deep, small, long-wave configuration.  It is only used in the
    benchmark at large ``depth/a`` and small ``ka``.
    """
    a = _validate_positive_scalar("radius", radius)
    _validate_positive_scalar("submergence", submergence)
    k = _validate_positive_scalar("nu", nu)
    amp = _validate_positive_scalar("amplitude", amplitude)
    rho = _validate_positive_scalar("density", density)
    g = _validate_positive_scalar("gravity", gravity)
    coefficient = _validate_positive_scalar("added_mass_coefficient", added_mass_coefficient)
    omega = math.sqrt(k * g)
    return complex(rho * math.pi * a * a * (1.0 + coefficient) * omega * omega * amp)