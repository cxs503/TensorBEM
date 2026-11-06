"""Radiation of a submerged body in water of *finite* depth.

:mod:`tensorbem.free_surface_solver` solves the same radiation problem in
infinite depth with the deep-water Green function.  The finite-depth kernel in
:mod:`tensorbem.finite_depth` was delivered on its own (kernel-level
verification only, see ``docs/finite_depth_2d.md`` §7); this module is the
end-to-end counterpart: the body boundary condition is imposed on a panel mesh
with that kernel, so the added mass and damping of a section in a channel of
depth ``h`` become available.

The formulation is the same indirect (single-layer) one as in deep water, which
is the only option in two dimensions: the radiated wave does not decay, so a
Green's identity step would have to survive a non-decaying field.  Only the
kernel changes.  Writing the finite-depth kernel as

    G = log(r)/(2*pi) + G_smooth,

the singular part is integrated over every straight panel in closed form (the
same :func:`~tensorbem.free_surface_solver._log_panel_primitive` used by the
deep-water solver) and only ``G_smooth`` is quadratured.  The normal-derivative
matrix is assembled the same way: the exact Rankine row integrals plus
``d(G_smooth)/dn`` from :func:`tensorbem.finite_depth.green_gradient` with
``include_rankine=False``.  The single-layer normal derivative jumps by one
half across the surface, so the system is ``A + I/2``, exactly as in deep water.

Far field and the energy identity
---------------------------------
In deep water the radiated wave amplitude is obtained from a Haskind integral
over the body surface (:func:`tensorbem.free_surface_solver.wave_amplitude`).
With the source representation here it is cleaner to read the amplitude off the
*propagating mode* of the kernel, whose far field is known exactly::

    G -> -i/(2 k0 N0) * phi0(z) phi0(z0) * exp(i k0 |x - x0|)

with ``phi0(z) = cosh(k0*(z + h))`` and ``N0`` the mode norm.  A source
distribution ``sigma`` therefore radiates the elevation

    eta_a = (i*omega/g) * phi0(0) * C,
    C     = -i/(2 k0 N0) * integral exp(-i k0 x) phi0(z) sigma ds.

The factor two in :func:`far_field_damping` is the two directions of
propagation: each carries the mean flux ``rho*g*|eta_a|**2*c_g/2``, the body
radiates that in both directions, and mean power ``B*U**2/2`` equals the total
flux, so ``B = 2*rho*g*c_g*|eta_a|**2`` for unit velocity.  The constant is not
fitted: it is the standard wave-energy balance, and in the deep-water limit the
same formula reproduces
:func:`tensorbem.free_surface_solver.far_field_damping` (tested).  That identity
is the real test of the radiation condition; the added mass alone looks
plausible with the wrong kernel.

Verification summary (see ``docs/finite_depth_radiation.md``):

* ``h -> infinity``: the solve collapses onto the frozen deep-water solver to
  ~1e-6 relative in added mass and damping (both motions);
* energy identity: near-field damping ``omega*rho*Im(force)`` and far-field
  damping agree to the mesh discretisation error (<2% at 64 panels);
* a method-of-fundamental-solutions reference built from the same kernel but a
  different discretisation (point sources, no panels) agrees to ~1e-3.

Limits: constant depth, no current, linear potential flow, and a body strictly
inside ``-h < z < 0``.  Accuracy degrades as ``nu*h`` grows because the
evanescent modes crowd together (``nu*h`` around 100 should use the deep-water
solver instead).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss

from .finite_depth import (
    finite_depth_wavenumber,
    green_function,
    green_gradient,
    group_velocity,
    propagating_mode_norm,
)
from .free_surface_solver import (
    RadiationResult,
    _GRAVITY,
    _log_panel_primitive,
    _rankine_normal_derivative_matrix,
)
from .geometry import PanelMesh

__all__ = [
    "assemble_indirect_matrix",
    "solve_radiation_indirect",
    "radiation_reference",
    "wave_amplitude",
    "far_field_damping",
    "solve_radiation",
]


def _validate_depth_inputs(mesh: PanelMesh, nu: float, depth: float) -> tuple[float, float]:
    """Check that the body sits strictly inside the fluid strip ``-h < z < 0``."""
    if not isinstance(mesh, PanelMesh):
        raise TypeError("mesh must be a PanelMesh")
    if isinstance(nu, (bool, np.bool_)) or not math.isfinite(float(nu)) or float(nu) <= 0.0:
        raise ValueError("nu must be a positive finite number")
    if (
        isinstance(depth, (bool, np.bool_))
        or not math.isfinite(float(depth))
        or float(depth) <= 0.0
    ):
        raise ValueError("depth must be a positive finite number")
    if np.any(mesh.vertices[:, 1] >= 0.0) or np.any(mesh.centers[:, 1] >= 0.0):
        raise ValueError(
            "a submerged body must lie strictly below the free surface; "
            "panels touching z = 0 are outside the supported formulation"
        )
    if np.any(mesh.vertices[:, 1] <= -float(depth)):
        raise ValueError(
            "the body must lie strictly above the sea bed; panels touching "
            "z = -depth are outside the supported formulation"
        )
    return float(nu), float(depth)


def _smooth_normal_derivative_matrix(
    mesh: PanelMesh, nu: float, depth: float, panel_nodes: int, modes: int | None
) -> np.ndarray:
    """Gauss quadrature of ``d(G_smooth)/dn``, whose integrand is smooth."""
    count = mesh.panel_count
    nodes, weights = leggauss(panel_nodes)
    fraction = 0.5 * (nodes + 1.0)
    source_x = (
        mesh.vertices[:, 0][:, None] + fraction[None, :] * mesh.vectors[:, 0][:, None]
    ).ravel()
    source_z = (
        mesh.vertices[:, 1][:, None] + fraction[None, :] * mesh.vectors[:, 1][:, None]
    ).ravel()
    jacobian = np.tile(weights, count) * np.repeat(0.5 * mesh.lengths, panel_nodes)
    matrix = np.zeros((count, count), dtype=complex)
    for row in range(count):
        target = mesh.centers[row]
        normal = mesh.normals[row]
        gradient_x, gradient_z = green_gradient(
            target[0],
            target[1],
            source_x,
            source_z,
            nu,
            depth,
            modes=modes,
            include_rankine=False,
        )
        values = (gradient_x * normal[0] + gradient_z * normal[1]) * jacobian
        matrix[row] = values.reshape(count, panel_nodes).sum(axis=1)
    return matrix


def assemble_indirect_matrix(
    mesh: PanelMesh,
    nu: float,
    depth: float,
    *,
    panel_nodes: int = 8,
    modes: int | None = None,
) -> np.ndarray:
    """Normal-derivative matrix of the finite-depth single-layer potential.

    The singular part is ``d(log r / 2pi)/dn`` on straight panels, integrated in
    closed form by :func:`_rankine_normal_derivative_matrix`; the smooth
    remaining part comes from the finite-depth kernel.  The jump relation is
    *not* included here, it is applied by :func:`solve_radiation_indirect`.
    """
    _validate_depth_inputs(mesh, nu, depth)
    if panel_nodes < 2:
        raise ValueError("panel_nodes must be at least 2")
    matrix = _rankine_normal_derivative_matrix(mesh)
    matrix += _smooth_normal_derivative_matrix(
        mesh, float(nu), float(depth), panel_nodes, modes
    )
    return matrix


def _evaluate_source_potential(
    mesh: PanelMesh,
    sources: np.ndarray,
    nu: float,
    depth: float,
    panel_nodes: int,
    modes: int | None,
) -> np.ndarray:
    """``phi(P_i) = integral G(P_i, Q) sigma(Q) ds(Q)`` at the panel centres."""
    count = mesh.panel_count
    starts = mesh.vertices
    lengths = mesh.lengths
    tangents = mesh.vectors / lengths[:, None]
    normals = mesh.normals

    delta_a = mesh.centers[:, None, :] - starts[None, :, :]
    xi = np.einsum("mpk,pk->mp", delta_a, tangents)
    h = np.abs(np.einsum("mpk,pk->mp", delta_a, normals))

    rankine_potential = (
        _log_panel_primitive(lengths[None, :] - xi, h) - _log_panel_primitive(-xi, h)
    ) / (2.0 * math.pi)

    smooth_nodes = min(panel_nodes, 6)
    nodes, weights = leggauss(smooth_nodes)
    fraction = 0.5 * (nodes + 1.0)
    source_x = (
        starts[:, 0][:, None] + fraction[None, :] * mesh.vectors[:, 0][:, None]
    ).ravel()
    source_z = (
        starts[:, 1][:, None] + fraction[None, :] * mesh.vectors[:, 1][:, None]
    ).ravel()
    jacobian = np.tile(weights, count) * np.repeat(0.5 * lengths, smooth_nodes)
    kernel = green_function(
        mesh.centers[:, 0][:, None],
        mesh.centers[:, 1][:, None],
        source_x[None, :],
        source_z[None, :],
        nu,
        depth,
        modes=modes,
    )
    radius2 = (
        (mesh.centers[:, 0][:, None] - source_x[None, :]) ** 2
        + (mesh.centers[:, 1][:, None] - source_z[None, :]) ** 2
    )
    # The kernel carries the Rankine part; remove it before quadrature so that
    # the log singularity is only ever integrated in closed form.
    smooth = kernel - np.log(np.sqrt(radius2)) / (2.0 * math.pi)
    smooth_potential = (smooth * jacobian[None, :]).reshape(
        count, count, smooth_nodes
    ).sum(axis=2)
    return (rankine_potential + smooth_potential) @ sources


def _solve_sources(
    mesh: PanelMesh,
    nu: float,
    depth: float,
    direction: str,
    panel_nodes: int,
    modes: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(sources, system, data, potential)`` for one motion.

    Shared by :func:`solve_radiation_indirect` and :func:`solve_radiation` so
    that the density is computed once even when the far-field damping is also
    wanted.
    """
    if direction == "sway":
        direction_vector = np.array([1.0, 0.0])
    elif direction == "heave":
        direction_vector = np.array([0.0, 1.0])
    else:
        raise ValueError("direction must be 'sway' or 'heave'")
    nu, depth = _validate_depth_inputs(mesh, nu, depth)
    matrix = assemble_indirect_matrix(
        mesh, nu, depth, panel_nodes=panel_nodes, modes=modes
    )
    data = mesh.normals @ direction_vector
    system = matrix + 0.5 * np.eye(mesh.panel_count)
    sources = np.linalg.solve(system, data)
    potential = _evaluate_source_potential(
        mesh, sources, nu, depth, panel_nodes, modes
    )
    return sources, system, data, potential


def solve_radiation_indirect(
    mesh: PanelMesh,
    nu: float,
    depth: float,
    direction: str,
    *,
    density: float = 1025.0,
    panel_nodes: int = 8,
    modes: int | None = None,
    gravity: float = 9.80665,
) -> RadiationResult:
    """Radiation solve in water of constant depth ``h``.

    ``d(phi)/dn_body = U . n_body`` with unit velocity, so the added mass keeps
    the classical minus sign, ``A = -rho * Re integral(phi n)``, and the damping
    is ``B = omega * rho * Im integral(phi n)``.
    """
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("density must be a positive finite number")
    sources, system, data, potential = _solve_sources(
        mesh, nu, depth, direction, panel_nodes, modes
    )
    force = -(potential * mesh.lengths) @ mesh.normals
    omega = math.sqrt(_validate_depth_inputs(mesh, nu, depth)[0] * gravity)
    residual = system @ sources - data
    return RadiationResult(
        potential=potential,
        normal_velocity=data,
        added_mass=density * np.real(force),
        damping=omega * density * np.imag(force),
        condition_number=float(np.linalg.cond(system)),
        relative_residual=float(
            np.linalg.norm(residual) / max(np.linalg.norm(data), 1e-300)
        ),
    )


def radiation_reference(
    mesh: PanelMesh,
    nu: float,
    depth: float,
    direction: str,
    *,
    source_count: int = 24,
    source_radius_ratio: float = 0.55,
    density: float = 1025.0,
    modes: int | None = None,
    gravity: float = 9.80665,
) -> RadiationResult:
    """Independent reference solution by the method of fundamental solutions.

    Point sources are placed on a circle inside the body and the Neumann
    condition is collocated at the panel centres.  The kernel is the same
    finite-depth one, but the discretisation shares nothing with the panel
    method: no piecewise-constant density, no singular quadrature, no matrix
    entries in common.  It therefore catches assembly mistakes that the
    energy identity alone cannot see.
    """
    if direction == "sway":
        direction_vector = np.array([1.0, 0.0])
    elif direction == "heave":
        direction_vector = np.array([0.0, 1.0])
    else:
        raise ValueError("direction must be 'sway' or 'heave'")
    if source_count < 4:
        raise ValueError("source_count must be at least 4")
    if not 0.0 < source_radius_ratio < 1.0:
        raise ValueError("source_radius_ratio must lie strictly between 0 and 1")
    nu, depth = _validate_depth_inputs(mesh, nu, depth)

    centre = np.mean(mesh.centers, axis=0)
    radius = float(np.mean(np.linalg.norm(mesh.centers - centre, axis=1)))
    angle = 2.0 * np.pi * np.arange(source_count) / source_count
    sources = centre + source_radius_ratio * radius * np.column_stack(
        (np.cos(angle), np.sin(angle))
    )
    if np.any(sources[:, 1] >= 0.0):
        raise ValueError("fundamental solutions must lie below the free surface")

    targets = mesh.centers
    gradient_x, gradient_z = green_gradient(
        targets[:, 0][:, None],
        targets[:, 1][:, None],
        sources[None, :, 0],
        sources[None, :, 1],
        nu,
        depth,
        modes=modes,
        include_rankine=True,
    )
    normals = mesh.normals
    kernel = gradient_x * normals[:, 0][:, None] + gradient_z * normals[:, 1][:, None]
    data = normals @ direction_vector
    strengths, *_ = np.linalg.lstsq(kernel, data, rcond=None)

    potential = green_function(
        targets[:, 0][:, None],
        targets[:, 1][:, None],
        sources[None, :, 0],
        sources[None, :, 1],
        nu,
        depth,
        modes=modes,
    ) @ strengths
    force = -(potential * mesh.lengths) @ normals
    omega = math.sqrt(nu * gravity)
    return RadiationResult(
        potential=potential,
        normal_velocity=data,
        added_mass=density * np.real(force),
        damping=omega * density * np.imag(force),
        condition_number=float(np.linalg.cond(kernel)),
        relative_residual=float(
            np.linalg.norm(kernel @ strengths - data) / max(np.linalg.norm(data), 1e-300)
        ),
    )


def wave_amplitude(
    mesh: PanelMesh,
    sources: np.ndarray,
    nu: float,
    depth: float,
    *,
    gravity: float = 9.80665,
    modes: int | None = None,
) -> complex:
    """Complex elevation amplitude radiated by a source distribution.

    Returns ``eta_a`` for unit body velocity: the outgoing wave travelling
    towards ``+x`` has ``eta(x) -> eta_a * exp(i k0 x)``, and the wave travelling
    towards ``-x`` is its conjugate (same modulus) in this symmetric
    formulation.  ``sources`` is the density returned by the solve; the caller
    that owns the energy identity turns this into a damping coefficient, so the
    same amplitude is not normalised twice.
    """
    nu, depth = _validate_depth_inputs(mesh, nu, depth)
    k0 = finite_depth_wavenumber(nu, depth)
    norm = propagating_mode_norm(k0, depth)
    phi0 = np.cosh(k0 * (mesh.centers[:, 1] + depth))
    integral = np.sum(
        phi0 * np.exp(-1j * k0 * mesh.centers[:, 0]) * sources * mesh.lengths
    )
    coefficient = -1j / (2.0 * k0 * norm) * integral
    omega = math.sqrt(nu * gravity)
    return 1j * omega / gravity * math.cosh(k0 * depth) * coefficient


def far_field_damping(
    mesh: PanelMesh,
    sources: np.ndarray,
    nu: float,
    depth: float,
    density: float = 1025.0,
    *,
    gravity: float = 9.80665,
    modes: int | None = None,
) -> float:
    """Damping from the radiated wave amplitude, ``B = 2*rho*g*c_g*|eta_a|^2``."""
    amplitude = wave_amplitude(
        mesh, sources, nu, depth, gravity=gravity, modes=modes
    )
    c_g = group_velocity(nu, depth, gravity=gravity)
    return 2.0 * density * gravity * float(c_g) * float(abs(amplitude) ** 2)


def solve_radiation(
    mesh: PanelMesh,
    nu: float,
    depth: float,
    direction: str,
    *,
    density: float = 1025.0,
    panel_nodes: int = 8,
    modes: int | None = None,
    gravity: float = 9.80665,
) -> tuple[RadiationResult, float]:
    """Solve once and return both the near-field result and the far-field damping.

    The source density is not exposed by :class:`RadiationResult`, so this
    convenience wrapper returns ``(result, far_field_damping)`` from the same
    densities instead of solving twice.
    """
    sources, system, data, potential = _solve_sources(
        mesh, nu, depth, direction, panel_nodes, modes
    )
    nu_checked, depth_checked = _validate_depth_inputs(mesh, nu, depth)
    force = -(potential * mesh.lengths) @ mesh.normals
    omega = math.sqrt(nu_checked * gravity)
    residual = system @ sources - data
    result = RadiationResult(
        potential=potential,
        normal_velocity=data,
        added_mass=density * np.real(force),
        damping=omega * density * np.imag(force),
        condition_number=float(np.linalg.cond(system)),
        relative_residual=float(
            np.linalg.norm(residual) / max(np.linalg.norm(data), 1e-300)
        ),
    )
    far = far_field_damping(
        mesh,
        sources,
        nu_checked,
        depth_checked,
        density,
        gravity=gravity,
        modes=modes,
    )
    return result, far
