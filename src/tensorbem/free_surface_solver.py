"""Frequency-domain radiation of a two-dimensional submerged body.

Formulation
-----------
The body condition is imposed directly on the potential of a source
distribution spread over the body surface,

    phi(P) = integral_body G(P, Q) sigma(Q) ds(Q),

so that ``d(phi)/dn = U . n`` on the body. Differentiating the representation
gives the boundary integral equation

    (1/2 I + K') sigma = U . n,

where ``K'`` is the normal derivative of the free-surface Green function taken
at the field point. The half comes from the jump of the normal derivative of
the single-layer potential; its sign is fixed by the convention of
:mod:`tensorbem.free_surface` (where the logarithm enters with a plus sign) and
is asserted by the tests against the closed-form free-space circle, where the
exact source density is ``sigma = 2 U cos(theta)``.

This is the *indirect* formulation, and it is the one Ursell used. It matters:
Green's identity applied to the same kernel (the direct formulation, which
solves for the potential itself) does **not** reproduce the damping for this
problem, because the two-dimensional wave field does not decay and the identity
does not close on a body of finite extent. The indirect route never takes that
step - the body condition is imposed on the representation itself - and it is
the form the classical references are written in. Do not reintroduce the direct
version: it produced plausible-looking added mass with damping wrong by three
to four orders of magnitude.

Only the body is discretised. The free surface never appears as panels, because
the kernel already satisfies the free-surface condition and radiates outgoing
waves. That is why a two-dimensional radiation problem needs only a few dozen
panels.

The singular (logarithmic) part of the kernel is integrated in closed form over
straight panels, and only the smooth remainder is quadratured; see
:func:`_rankine_normal_derivative_matrix`.

Forces
------
With the time factor ``exp(-i*omega*t)`` and unit velocity amplitude the
hydrodynamic force per unit length is ``F_i = -rho * integral(phi n_i) ds``,
which gives

    A_ij = rho * Re integral(phi_j n_i) ds,   B_ij = omega * rho * Im integral(phi_j n_i) ds,

the classical added-mass and damping coefficients (note the minus sign: the
potential here is the unit-velocity potential of the *fluid*, whose pressure
force opposes the motion).

Verification
------------
:func:`tensorbem.benchmarks.ursell_cylinder` compares against Ursell's
multipole solution for a deeply submerged circular cylinder and enforces the
energy identity between the near-field damping and the far-field wave
amplitude. Both are tests, not assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from .free_surface import green_function, green_gradient_field
from .geometry import PanelMesh

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_GRAVITY = 9.80665


@dataclass(frozen=True)
class RadiationResult:
    """Solution of one radiation problem."""

    potential: ComplexArray
    normal_velocity: ComplexArray
    added_mass: FloatArray
    damping: FloatArray
    condition_number: float
    relative_residual: float


@dataclass(frozen=True)
class DiffractionResult:
    """Solution of one two-dimensional deep-water wave-diffraction problem."""

    total_potential: ComplexArray
    scattered_potential: ComplexArray
    incident_potential: ComplexArray
    incident_normal_velocity: ComplexArray
    source_density: ComplexArray
    excitation_force: ComplexArray
    reflection_coefficient: complex
    transmission_coefficient: complex
    energy_balance_error: float
    condition_number: float
    relative_residual: float


def _validate_panel_nodes(panel_nodes: int) -> int:
    if isinstance(panel_nodes, (bool, np.bool_)) or not isinstance(
        panel_nodes, (int, np.integer)
    ):
        raise ValueError("panel_nodes must be an integer >= 4")
    nodes = int(panel_nodes)
    if nodes < 4:
        raise ValueError("panel_nodes must be at least 4")
    return nodes


def _validate_mesh_depth(mesh: PanelMesh, nu: float) -> float:
    if not isinstance(mesh, PanelMesh):
        raise TypeError("mesh must be a PanelMesh")
    if np.any(mesh.vertices[:, 1] >= 0.0) or np.any(mesh.centers[:, 1] >= 0.0):
        raise ValueError(
            "a submerged body must lie strictly below the free surface; "
            "panels touching z = 0 are outside the supported formulation"
        )
    if isinstance(nu, (bool, np.bool_)) or not math.isfinite(float(nu)) or float(nu) <= 0.0:
        raise ValueError("nu must be a positive finite number")
    return float(nu)


def radiation_reference(
    mesh: PanelMesh, nu: float, direction: str, *, source_count: int = 48,
    source_radius_ratio: float = 0.55, density: float = 1025.0,
) -> RadiationResult:
    """Independent reference solution by the method of fundamental solutions.

    The free-surface Green function already satisfies every condition except
    the body boundary condition, so a distribution of point sources placed
    *inside* the body is a complete representation. Collocating the Neumann
    condition on the body gives a small dense system; for a circular body the
    error falls exponentially in ``source_count`` because the kernel is
    analytic, which is what makes this a usable reference for the panel method.

    The sources are placed on a circle of radius ``source_radius_ratio`` times
    the mean panel-centre radius, concentric with the body's mean panel centre.
    The representation is independent of the panel discretisation: no panels,
    no piecewise-constant basis, and no matrix entries shared with
    :func:`assemble_matrices`.

    Note the sign: this routine imposes the physical condition
    ``d(phi)/dn_body = U . n_body``, so its potential is minus the bound-domain
    potential :func:`solve_radiation` computes from the fluid-side normal
    derivative. The added mass therefore carries an explicit minus sign here;
    ``A = -rho * integral(phi n)`` is the classical form. Getting this wrong
    flips the sign of the added mass while leaving every residual small, which
    is why the two formulations are compared in the tests.
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

    centre = np.mean(mesh.centers, axis=0)
    radius = float(np.mean(np.linalg.norm(mesh.centers - centre, axis=1)))
    angle = 2.0 * np.pi * np.arange(source_count) / source_count
    sources = centre + source_radius_ratio * radius * np.column_stack(
        (np.cos(angle), np.sin(angle))
    )
    if np.any(sources[:, 1] >= 0.0):
        raise ValueError("fundamental solutions must lie below the free surface")

    targets = mesh.centers
    gradient_x, gradient_z = green_gradient_field(
        targets[:, 0][:, None],
        targets[:, 1][:, None],
        sources[None, :, 0],
        sources[None, :, 1],
        nu,
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
    ) @ strengths
    force = -(potential * mesh.lengths) @ normals
    omega = math.sqrt(nu * _GRAVITY)
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


def assemble_indirect_matrix(mesh: PanelMesh, nu: float, *,
                             panel_nodes: int = 24) -> ComplexArray:
    """Normal-derivative matrix ``K'`` of the indirect (source) formulation.

    ``K'[i, j] = integral_{panel j} dG(P_i, Q)/dn_i ds``, the derivative taken
    at the field point along its normal. Unlike the direct formulation this
    uses field derivatives, and it is the formulation Ursell used: the body
    condition is imposed directly on the potential of a source distribution,
    with no Green's-identity step that would have to survive the non-decaying
    two-dimensional wave field. The MFS in :func:`radiation_reference` is the
    same formulation with point sources, so the two are genuinely comparable.

    The kernel's logarithmic part contributes nothing to the normal derivative
    on a flat panel: a collocation point on the panel's own line sees
    ``(Q - P) . n = 0``. Away from the self panel the kernel is bounded, so a
    plain Gauss rule converges; no exact primitive is needed here.
    """
    wavenumber = _validate_mesh_depth(mesh, nu)
    nodes_count = _validate_panel_nodes(panel_nodes)
    matrix = _rankine_normal_derivative_matrix(mesh)
    matrix += _free_surface_only_normal_derivative(mesh, wavenumber, nodes_count)
    return matrix


def _evaluate_source_potential(
    mesh: PanelMesh, sources: ComplexArray, nu: float, panel_nodes: int
) -> ComplexArray:
    """Evaluate ``phi(P_i) = integral G(P_i, Q) sigma(Q) ds(Q)`` at panel centres.

    The singular ``log(r)/(2*pi)`` part is integrated analytically over every
    straight panel (including the self panel), while the smooth free-surface
    remainder (whose image sources lie at depth ``-(z + z_0) > 0``) is
    integrated with Gauss-Legendre quadrature without ever evaluating ``log(0)``.
    """
    count = mesh.panel_count
    starts = mesh.vertices
    lengths = mesh.lengths
    tangents = mesh.vectors / lengths[:, None]
    normals = mesh.normals

    delta_a = mesh.centers[:, None, :] - starts[None, :, :]
    xi = np.einsum("mpk,pk->mp", delta_a, tangents)
    h = np.abs(np.einsum("mpk,pk->mp", delta_a, normals))

    def _vec_log_primitive(z: FloatArray) -> FloatArray:
        coplanar = h <= 1.0e-14
        h_safe = np.where(coplanar, 1.0, h)
        z_abs = np.where(np.abs(z) <= 1.0e-14, 1.0, np.abs(z))
        collinear = np.where(np.abs(z) <= 1.0e-14, 0.0, z * np.log(z_abs) - z)
        general = 0.5 * z * np.log(z * z + h_safe * h_safe) - z + h_safe * np.arctan(z / h_safe)
        return np.where(coplanar, collinear, general)

    rankine_potential = (
        _vec_log_primitive(lengths[None, :] - xi) - _vec_log_primitive(-xi)
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
    smooth = green_function(
        mesh.centers[:, 0][:, None],
        mesh.centers[:, 1][:, None],
        source_x[None, :],
        source_z[None, :],
        nu,
        include_rankine=False,
    )
    smooth_potential = (smooth * jacobian[None, :]).reshape(
        count, count, smooth_nodes
    ).sum(axis=2)
    return (rankine_potential + smooth_potential) @ sources


def solve_radiation_indirect(
    mesh: PanelMesh, nu: float, direction: str, *, density: float = 1025.0,
    panel_nodes: int = 24,
) -> RadiationResult:
    """Radiation solve through the indirect formulation.

    The body condition is ``d(phi)/dn_body = U . n``, so this potential is the
    classical unit-velocity potential and the added mass carries the classical
    minus sign, ``A = -rho * Re integral(phi n)``.
    """
    if direction == "sway":
        direction_vector = np.array([1.0, 0.0])
    elif direction == "heave":
        direction_vector = np.array([0.0, 1.0])
    else:
        raise ValueError("direction must be 'sway' or 'heave'")
    nodes_count = _validate_panel_nodes(panel_nodes)
    wavenumber = _validate_mesh_depth(mesh, nu)
    count = mesh.panel_count
    matrix = assemble_indirect_matrix(mesh, wavenumber, panel_nodes=nodes_count)
    data = mesh.normals @ direction_vector
    # The derivative of the single-layer potential jumps by one half across the
    # surface. The sign is fixed once, by requiring the potential to satisfy the
    # body condition, and is asserted in the tests.
    system = matrix + 0.5 * np.eye(count)
    sources = np.linalg.solve(system, data)
    potential = _evaluate_source_potential(mesh, sources, wavenumber, nodes_count)
    force = -(potential * mesh.lengths) @ mesh.normals
    omega = math.sqrt(wavenumber * _GRAVITY)
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


def _rankine_normal_derivative_matrix(mesh: PanelMesh) -> ComplexArray:
    """Exact ``integral_{panel j} d(log r / 2pi)/dn_i ds`` for ``i != j``.

    ``G_R = log(r)/(2*pi)`` is the singular part of the free-surface Green
    function, and over a straight panel its normal derivative is elementary:

        dG_R/dn = -(Q-P).n / (2*pi*|Q-P|^2),

    which integrates to a log plus an arc tangent. Doing this in closed form
    removes the only singular quadrature from the assembly; everything left in
    :func:`assemble_indirect_matrix` is smooth. The self panel contributes
    exactly zero: its own line passes through the collocation point, so
    ``(Q-P).n`` vanishes there, and the missing jump is added separately.

    The perpendicular distance is formed directly as ``|d - (d.u)u|`` rather
    than as ``|d|^2 - (d.u)^2``: for a neighbouring panel the two are ``O(h)``
    and ``O(h^2)`` and the difference of squares would lose half the digits.
    """
    count = mesh.panel_count
    matrix = np.zeros((count, count), dtype=complex)
    for row in range(count):
        point = mesh.centers[row]
        normal = mesh.normals[row]
        tangent = mesh.vectors / mesh.lengths[:, None]
        delta = mesh.vertices - point                       # (count, 2)
        projection = np.einsum("pj,pj->p", delta, tangent)
        offset = delta - projection[:, None] * tangent
        height = np.sqrt(np.einsum("pj,pj->p", offset, offset))
        # The self panel's own line passes through the collocation point: its
        # closed-form contribution is exactly zero, and the formula below is
        # singular there, so the placeholder is overwritten after the row.
        height[row] = 1.0
        constant_term = delta @ normal                       # d.n
        slope = tangent @ normal                             # u.n
        lower = projection
        upper = lower + mesh.lengths
        def primitive(u: FloatArray) -> FloatArray:
            return (
                0.5 * slope * np.log(u * u + height * height)
                + (constant_term - slope * lower)
                / height
                * np.arctan(u / height)
            )
        matrix[row] = -(primitive(upper) - primitive(lower)) / (2.0 * math.pi)
        matrix[row, row] = 0.0
    return matrix


def _free_surface_only_normal_derivative(
    mesh: PanelMesh, nu: float, panel_nodes: int
) -> ComplexArray:
    """Gauss quadrature of ``d(dG/dn - dG_R/dn)``, whose integrand is smooth."""
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
        gradient_x, gradient_z = green_gradient_field(
            target[0],
            target[1],
            source_x,
            source_z,
            nu,
            include_rankine=False,
        )
        values = (gradient_x * normal[0] + gradient_z * normal[1]) * jacobian
        matrix[row] = values.reshape(count, panel_nodes).sum(axis=1)
    return matrix


def wave_amplitude(
    mesh: PanelMesh, potential: ComplexArray, normal_velocity: ComplexArray,
    nu: float,
) -> complex:
    """Outgoing-wave coefficient of the radiated field (Haskind integral).

    In deep water the far field of the Green function is a plane wave
    ``exp(nu*z) * exp(i*nu*x)``, so the wave amplitude radiated by a body with
    surface potential ``phi`` and normal velocity ``q`` is

        a_w = i * integral [ exp(nu*z) exp(-i*nu*x)
                             (q - nu*phi*(n_z - i*n_x)) ] ds.

    Inside the modulus this gives the classical Haskind relation
    ``B = omega*rho*|a_w|**2``. The normalisation is *not* fitted: the
    principal-value part of the Green function radiates exactly as strongly as
    its residue term, so the far-field coefficient is the full residue strength
    rather than half of it. The benchmark asserts that this far-field damping
    agrees with the near-field ``omega*rho*Im(force)`` as the mesh refines,
    which is the check that pins the constant down.
    """
    phase = np.exp(nu * mesh.centers[:, 1]) * np.exp(-1j * nu * mesh.centers[:, 0])
    stream = normal_velocity - nu * potential * (
        mesh.normals[:, 1] - 1j * mesh.normals[:, 0]
    )
    return 1j * np.sum(phase * stream * mesh.lengths)


def far_field_damping(
    mesh: PanelMesh, potential: ComplexArray, normal_velocity: ComplexArray,
    nu: float, density: float = 1025.0, *, gravity: float = 9.80665,
) -> float:
    """Damping from the far-field wave amplitude, ``B = omega*rho*|a_w|**2``."""
    amplitude = wave_amplitude(mesh, potential, normal_velocity, nu)
    omega = math.sqrt(nu * gravity)
    return omega * density * float(abs(amplitude) ** 2)


# --------------------------------------------------------------------------
# Incident wave, diffraction and Haskind reciprocity
# --------------------------------------------------------------------------
def _validate_positive_scalar(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _validate_wave_direction(direction: int) -> int:
    if isinstance(direction, (bool, np.bool_)) or int(direction) not in (-1, 1):
        raise ValueError("direction must be +1 (+x wave) or -1 (-x wave)")
    return int(direction)


def incident_airy_wave(
    points: FloatArray,
    normals: FloatArray,
    nu: float,
    amplitude: float = 1.0,
    *,
    gravity: float = _GRAVITY,
    direction: int = 1,
) -> tuple[ComplexArray, ComplexArray]:
    """Deep-water incident Airy potential and outward-normal velocity.

    With time factor ``exp(-i*omega*t)`` and ``nu = omega**2 / g``, a regular
    wave of elevation amplitude ``A`` propagating in ``direction * (+x)`` has

        phi_I(x, z) = -(i * g * A / omega) * exp(nu * z + i * direction * nu * x),

    satisfying ``d(phi_I)/dz - nu * phi_I = 0`` on ``z = 0`` and surface
    elevation ``Re((i*omega/g)*phi_I(x, 0)*exp(-i*omega*t)) = A*cos(direction*nu*x - omega*t)``.
    """
    k = _validate_positive_scalar("nu", nu)
    amp = _validate_positive_scalar("amplitude", amplitude)
    g = _validate_positive_scalar("gravity", gravity)
    sign = _validate_wave_direction(direction)
    coords = np.asarray(points, dtype=float)
    nrm = np.asarray(normals, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or nrm.shape != coords.shape:
        raise ValueError("points and normals must both have shape (N, 2)")
    if np.any(coords[:, 1] >= 0.0):
        raise ValueError("evaluation points must lie strictly below z = 0")

    omega = math.sqrt(k * g)
    phase = np.exp(k * coords[:, 1] + 1.0j * sign * k * coords[:, 0])
    phi_inc = -(1.0j * g * amp / omega) * phase
    dphi_dn = omega * amp * (sign * nrm[:, 0] - 1.0j * nrm[:, 1]) * phase
    return phi_inc, dphi_dn


def solve_diffraction_indirect(
    mesh: PanelMesh,
    nu: float,
    amplitude: float = 1.0,
    *,
    density: float = 1025.0,
    gravity: float = _GRAVITY,
    direction: int = 1,
    panel_nodes: int = 24,
) -> DiffractionResult:
    """Solve 2-D deep-water wave diffraction around a submerged closed body.

    Splits the total potential into ``phi = phi_I + phi_S`` where ``phi_S`` is
    represented by a single-layer free-surface source distribution on the body
    satisfying ``(1/2 I + K') sigma_S = -d(phi_I)/dn``.

    Far upstream and downstream, ``G(x, z; Q) ~ -i*exp(nu*(z + z_Q))*exp(i*nu*|x - x_Q|)``,
    so the scattered wave-elevation ratios toward ``+x`` and ``-x`` follow directly
    from the Kochin integrals of ``sigma_S``.
    """
    wavenumber = _validate_mesh_depth(mesh, nu)
    amp = _validate_positive_scalar("amplitude", amplitude)
    rho = _validate_positive_scalar("density", density)
    g = _validate_positive_scalar("gravity", gravity)
    sign = _validate_wave_direction(direction)
    nodes_count = _validate_panel_nodes(panel_nodes)

    count = mesh.panel_count
    matrix = assemble_indirect_matrix(mesh, wavenumber, panel_nodes=nodes_count)
    phi_inc, dphi_inc_dn = incident_airy_wave(
        mesh.centers,
        mesh.normals,
        wavenumber,
        amp,
        gravity=g,
        direction=sign,
    )
    system = matrix + 0.5 * np.eye(count)
    rhs = -dphi_inc_dn
    sources = np.linalg.solve(system, rhs)
    phi_scat = _evaluate_source_potential(mesh, sources, wavenumber, nodes_count)
    phi_total = phi_inc + phi_scat

    omega = math.sqrt(wavenumber * g)
    # Dynamic pressure p = i*omega*rho*phi; force on body = int p*(-n_body) ds.
    excitation_force = -1.0j * omega * rho * ((phi_total * mesh.lengths) @ mesh.normals)

    nodes, weights = leggauss(nodes_count)
    fraction = 0.5 * (nodes + 1.0)
    qx = mesh.vertices[:, 0][:, None] + fraction[None, :] * mesh.vectors[:, 0][:, None]
    qz = mesh.vertices[:, 1][:, None] + fraction[None, :] * mesh.vectors[:, 1][:, None]
    panel_weights = 0.5 * mesh.lengths[:, None] * weights[None, :]
    c_plus = (omega / (g * amp)) * complex(
        np.sum(
            np.sum(np.exp(wavenumber * qz - 1.0j * wavenumber * qx) * panel_weights, axis=1)
            * sources
        )
    )
    c_minus = (omega / (g * amp)) * complex(
        np.sum(
            np.sum(np.exp(wavenumber * qz + 1.0j * wavenumber * qx) * panel_weights, axis=1)
            * sources
        )
    )
    if sign == 1:
        reflection = c_minus
        transmission = 1.0 + c_plus
    else:
        reflection = c_plus
        transmission = 1.0 + c_minus

    energy_error = float(abs(reflection) ** 2 + abs(transmission) ** 2 - 1.0)
    residual = system @ sources - rhs
    return DiffractionResult(
        total_potential=phi_total,
        scattered_potential=phi_scat,
        incident_potential=phi_inc,
        incident_normal_velocity=dphi_inc_dn,
        source_density=sources,
        excitation_force=excitation_force,
        reflection_coefficient=complex(reflection),
        transmission_coefficient=complex(transmission),
        energy_balance_error=energy_error,
        condition_number=float(np.linalg.cond(system)),
        relative_residual=float(
            np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-300)
        ),
    )


def diffraction_reference(
    mesh: PanelMesh,
    nu: float,
    amplitude: float = 1.0,
    *,
    source_count: int = 48,
    source_radius_ratio: float = 0.55,
    density: float = 1025.0,
    gravity: float = _GRAVITY,
    direction: int = 1,
) -> DiffractionResult:
    """Independent MFS wave-diffraction reference with interior point sources."""
    wavenumber = _validate_mesh_depth(mesh, nu)
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
    phi_inc, dphi_inc_dn = incident_airy_wave(
        targets, normals, wavenumber, amp, gravity=g, direction=sign
    )
    gx, gz = green_gradient_field(
        targets[:, 0][:, None],
        targets[:, 1][:, None],
        sources_xy[None, :, 0],
        sources_xy[None, :, 1],
        wavenumber,
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
            wavenumber,
        )
        @ strengths
    )
    phi_total = phi_inc + phi_scat
    omega = math.sqrt(wavenumber * g)
    excitation_force = -1.0j * omega * rho * ((phi_total * mesh.lengths) @ normals)

    c_plus = (omega / (g * amp)) * complex(
        np.sum(
            np.exp(wavenumber * sources_xy[:, 1] - 1.0j * wavenumber * sources_xy[:, 0])
            * strengths
        )
    )
    c_minus = (omega / (g * amp)) * complex(
        np.sum(
            np.exp(wavenumber * sources_xy[:, 1] + 1.0j * wavenumber * sources_xy[:, 0])
            * strengths
        )
    )
    reflection = c_minus if sign == 1 else c_plus
    transmission = (1.0 + c_plus) if sign == 1 else (1.0 + c_minus)
    energy_error = float(abs(reflection) ** 2 + abs(transmission) ** 2 - 1.0)
    return DiffractionResult(
        total_potential=phi_total,
        scattered_potential=phi_scat,
        incident_potential=phi_inc,
        incident_normal_velocity=dphi_inc_dn,
        source_density=strengths,
        excitation_force=excitation_force,
        reflection_coefficient=complex(reflection),
        transmission_coefficient=complex(transmission),
        energy_balance_error=energy_error,
        condition_number=float(np.linalg.cond(kernel)),
        relative_residual=float(
            np.linalg.norm(kernel @ strengths - rhs) / max(np.linalg.norm(rhs), 1e-300)
        ),
    )


def haskind_excitation_force(
    mesh: PanelMesh,
    radiation: RadiationResult,
    nu: float,
    amplitude: float = 1.0,
    *,
    density: float = 1025.0,
    gravity: float = _GRAVITY,
    direction: int = 1,
) -> complex:
    """Wave exciting force from radiation data alone (Haskind-Newman relation).

    Applying Green's second identity to the unit-velocity radiation potential
    ``phi_j`` (satisfying ``d(phi_j)/dn = n_j``) and the diffraction potential
    ``phi_D = phi_I + phi_S`` cancels the ``(phi_S, phi_j)`` boundary integral
    at infinity because both satisfy the outgoing radiation condition, yielding

        X_j = -i * omega * rho * integral_Gamma (phi_I * n_j - phi_j * d(phi_I)/dn) ds.

    This computes the diffraction exciting force without solving a diffraction
    problem; agreement between :func:`haskind_excitation_force` and
    :func:`solve_diffraction_indirect` is an exact reciprocity check.
    """
    if not isinstance(radiation, RadiationResult):
        raise TypeError("radiation must be a RadiationResult")
    wavenumber = _validate_mesh_depth(mesh, nu)
    amp = _validate_positive_scalar("amplitude", amplitude)
    rho = _validate_positive_scalar("density", density)
    g = _validate_positive_scalar("gravity", gravity)
    sign = _validate_wave_direction(direction)
    if radiation.potential.shape != (mesh.panel_count,) or radiation.normal_velocity.shape != (
        mesh.panel_count,
    ):
        raise ValueError("radiation traces must match mesh.panel_count")

    phi_inc, dphi_inc_dn = incident_airy_wave(
        mesh.centers, mesh.normals, wavenumber, amp, gravity=g, direction=sign
    )
    omega = math.sqrt(wavenumber * g)
    integrand = phi_inc * radiation.normal_velocity - radiation.potential * dphi_inc_dn
    return complex(-1.0j * omega * rho * np.sum(integrand * mesh.lengths))
