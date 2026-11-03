"""Constant-panel collocation solver for the 2-D Laplace Dirichlet problem.

Boundary equation at a smooth collocation point:
    c*u = integral(G*q ds) - integral((dG/dn_y)*u ds)
with G = -(1/(2*pi))*log(r/L), -Delta(G)=delta, and c=1/2.
Thus Gmat*q = (0.5*I + Hmat)*u for a bounded domain, and
Gmat*q_e = (0.5*I - Hmat)*u for the exterior domain, where q_e is the
derivative along the fluid-domain normal (opposite the outward normal).

All straight-panel matrix entries are integrated analytically. This removes
the off-diagonal midpoint-quadrature error and improves near-panel accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from numpy.typing import NDArray

from .geometry import PanelMesh
from .kernels import (
    REFERENCE_LENGTH,
    integrate_double_layer_panel,
    integrate_single_layer_panel,
    validate_reference_length,
)

FloatArray = NDArray[np.float64]
_DOMAINS = ("interior", "exterior")
# Beyond this condition number the dense solve is no longer meaningful in
# double precision; it is the logarithmic-capacity degeneracy in practice.
_MAX_CONDITION = 1.0e13
# Relative tolerance on the exterior net flux below which the exterior
# potential is treated as decaying (O(1/r)) instead of logarithmically growing.
_NET_FLUX_TOL = 1.0e-8


@dataclass(frozen=True)
class LaplaceDirichletResult:
    """Solution and matrices from a constant-panel collocation solve."""

    normal_derivative: FloatArray
    boundary_values: FloatArray
    single_layer_matrix: FloatArray
    double_layer_matrix: FloatArray
    condition_number: float
    residual_l2: float
    relative_residual_l2: float = 0.0
    domain: str = "interior"
    reference_length: float = REFERENCE_LENGTH


def _validate_domain(domain: str) -> str:
    if domain not in _DOMAINS:
        raise ValueError(f"domain must be one of {_DOMAINS}")
    return domain


def assemble_matrices(
    mesh: PanelMesh, reference_length: float = REFERENCE_LENGTH
) -> tuple[FloatArray, FloatArray]:
    """Assemble analytically integrated single- and double-layer matrices."""
    reference = validate_reference_length(reference_length)
    centers = mesh.centers
    starts = mesh.vertices
    tangents = mesh.vectors / mesh.lengths[:, None]
    normals = mesh.normals
    lengths = mesh.lengths
    n = mesh.panel_count
    single = np.empty((n, n), dtype=float)
    double = np.empty((n, n), dtype=float)

    for i, target in enumerate(centers):
        for j in range(n):
            single[i, j] = integrate_single_layer_panel(
                target, starts[j], tangents[j], float(lengths[j]), reference
            )
            double[i, j] = integrate_double_layer_panel(
                target, starts[j], tangents[j], normals[j], float(lengths[j])
            )
    return single, double


def _solve(
    mesh: PanelMesh,
    boundary_values: FloatArray,
    domain: str,
    reference_length: float,
) -> LaplaceDirichletResult:
    values = np.asarray(boundary_values, dtype=float)
    n = mesh.panel_count
    if values.shape != (n,):
        raise ValueError(f"boundary_values must have shape ({n},)")
    if not np.isfinite(values).all():
        raise ValueError("boundary_values must all be finite")
    reference = validate_reference_length(reference_length)

    single, double = assemble_matrices(mesh, reference_length=reference)
    sign = 1.0 if domain == "interior" else -1.0
    rhs = 0.5 * values + sign * (double @ values)
    condition = float(np.linalg.cond(single))
    if not math.isfinite(condition) or condition > _MAX_CONDITION:
        raise ValueError(
            "the single-layer matrix is numerically singular "
            f"(condition number {condition:.3e}); for a closed 2-D boundary this "
            "happens when its logarithmic capacity approaches reference_length "
            f"(currently {reference:g}). Pass a different reference_length."
        )
    q = np.linalg.solve(single, rhs)
    residual = float(np.linalg.norm(single @ q - rhs))
    rhs_norm = float(np.linalg.norm(rhs))
    return LaplaceDirichletResult(
        normal_derivative=q,
        boundary_values=values.copy(),
        single_layer_matrix=single,
        double_layer_matrix=double,
        condition_number=condition,
        residual_l2=residual,
        relative_residual_l2=residual / rhs_norm if rhs_norm > 0.0 else residual,
        domain=domain,
        reference_length=reference,
    )


def solve_laplace_dirichlet(
    mesh: PanelMesh,
    boundary_values: FloatArray,
    reference_length: float = REFERENCE_LENGTH,
) -> LaplaceDirichletResult:
    """Solve for the unknown outward normal derivative given boundary values.

    The baseline targets smooth, closed, simply connected 2-D domains.
    Boundary values are piecewise constant at panel midpoints. The returned
    ``normal_derivative`` is the derivative along the polygon's outward normal.
    """
    return _solve(mesh, boundary_values, "interior", reference_length)


def solve_laplace_exterior_dirichlet(
    mesh: PanelMesh,
    boundary_values: FloatArray,
    reference_length: float = REFERENCE_LENGTH,
) -> LaplaceDirichletResult:
    """Solve the exterior 2-D Laplace Dirichlet problem.

    The fluid occupies the unbounded complement of the closed polygon. The
    returned ``normal_derivative`` is ``q_e = grad(u) . n_e`` with ``n_e`` the
    fluid-domain normal, i.e. the negative of the polygon's outward normal,
    matching the three-dimensional convention.

    The integral equation is of the first kind,

        S q_e = (1/2 I - H) u,

    so ``S`` must be well conditioned. Integrating a constant density over a
    closed curve gives ``a*log(reference_length/a)`` for a circle of radius
    ``a``, so ``S`` approaches rank deficiency as the curve's logarithmic
    capacity approaches ``reference_length``: the unshifted kernel (capacity 1)
    is degenerate on a unit circle, and the default ``reference_length = 2`` is
    degenerate on a circle of radius 2. For a fixed inscribed polygon the
    capacity is slightly below the circumscribed radius, so the system is
    ill-conditioned rather than exactly singular, and the condition number
    grows with refinement. A numerically singular system raises ``ValueError``
    instead of being solved.

    The far field is not imposed explicitly: the single-layer representation
    with a decaying density gives a decaying potential, so the boundary
    equation plus the decay of the density is the formulation. Unlike the
    3-D solver there is no explicit far-field or irregular-frequency
    treatment, and the solve is intended for verification meshes.
    """
    return _solve(mesh, boundary_values, "exterior", reference_length)


@dataclass(frozen=True)
class RigidBodyAddedMass2DResult:
    """Full ``(3, 3)`` rigid-body added-mass tensor in 2-D unbounded fluid.

    Modes are ``(surge_x, heave_y, roll_xy)`` about ``center``.
    """

    added_mass_matrix: FloatArray
    mode_potentials: FloatArray
    mode_normals: FloatArray
    center: tuple[float, float]
    condition_number: float
    max_relative_residual_l2: float
    symmetry_error: float


def solve_laplace_exterior_neumann(
    mesh: PanelMesh,
    normal_derivative: FloatArray,
    reference_length: float = REFERENCE_LENGTH,
) -> LaplaceDirichletResult:
    """Solve the exterior 2-D Laplace Neumann boundary-value problem.

    Given the fluid-domain normal derivative ``q_e = grad(u) . n_e`` (with
    ``n_e = -n_body`` opposite the polygon's outward normal, matching
    :func:`solve_laplace_exterior_dirichlet`), solve the second-kind boundary
    integral equation

        (1/2 I - H) u = S q_e

    for the boundary potential ``u``. Unlike the first-kind Dirichlet operator
    ``S``, the second-kind operator ``1/2 I - H`` has ``O(1)`` condition number
    independent of mesh refinement and no logarithmic-capacity degeneracy.
    When ``q_e`` has zero net flux ``integral(q_e ds) = 0`` (which holds
    identically for rigid-body translations and rotations of a closed body),
    the exterior potential decays as ``O(1/r)`` and ``u`` is invariant to
    ``reference_length``.
    """
    if not isinstance(mesh, PanelMesh):
        raise TypeError("mesh must be a PanelMesh")
    flux = np.asarray(normal_derivative, dtype=float)
    n = mesh.panel_count
    if flux.shape != (n,):
        raise ValueError(f"normal_derivative must have shape ({n},)")
    if not np.isfinite(flux).all():
        raise ValueError("normal_derivative must all be finite")
    reference = validate_reference_length(reference_length)

    single, double = assemble_matrices(mesh, reference_length=reference)
    system = 0.5 * np.eye(n) - double
    rhs = single @ flux
    condition = float(np.linalg.cond(system))
    potential = np.linalg.solve(system, rhs)
    residual = float(np.linalg.norm(system @ potential - rhs))
    rhs_norm = float(np.linalg.norm(rhs))
    return LaplaceDirichletResult(
        normal_derivative=flux.copy(),
        boundary_values=potential,
        single_layer_matrix=single,
        double_layer_matrix=double,
        condition_number=condition,
        residual_l2=residual,
        relative_residual_l2=residual / rhs_norm if rhs_norm > 0.0 else residual,
        domain="exterior",
        reference_length=reference,
    )


def compute_added_mass_matrix_2d(
    mesh: PanelMesh,
    density: float = 1.0,
    center: tuple[float, float] | FloatArray = (0.0, 0.0),
    reference_length: float = REFERENCE_LENGTH,
) -> RigidBodyAddedMass2DResult:
    """Compute the ``(3, 3)`` rigid-body added-mass matrix per unit span.

    Solves the exterior Neumann boundary-value problem ``d(phi_j)/dn_body = n_j``
    simultaneously for the three planar rigid-body modes ``(surge_x, heave_y,
    roll_xy)`` about ``center`` using ``(1/2 I - H) phi_j = S (-n_j)``, then
    evaluates ``A_ij = -rho * integral(phi_j * n_i ds)``.
    """
    if not isinstance(mesh, PanelMesh):
        raise TypeError("mesh must be a PanelMesh")
    if isinstance(density, (bool, np.bool_)) or not math.isfinite(float(density)) or float(density) <= 0.0:
        raise ValueError("density must be a positive finite number")
    rho = float(density)
    origin = np.asarray(center, dtype=float)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("center must be a finite 2-vector")
    reference = validate_reference_length(reference_length)

    single, double = assemble_matrices(mesh, reference_length=reference)
    n = mesh.panel_count
    rel = mesh.centers - origin[None, :]
    normals = mesh.normals
    mode_normals = np.column_stack(
        (
            normals[:, 0],
            normals[:, 1],
            rel[:, 0] * normals[:, 1] - rel[:, 1] * normals[:, 0],
        )
    )
    system = 0.5 * np.eye(n) - double
    rhs = single @ (-mode_normals)
    condition = float(np.linalg.cond(system))
    potentials = np.linalg.solve(system, rhs)
    residuals = np.linalg.norm(system @ potentials - rhs, axis=0)
    rhs_norms = np.linalg.norm(rhs, axis=0)
    rel_residuals = np.where(rhs_norms > 0.0, residuals / rhs_norms, residuals)

    added_mass = -rho * ((mode_normals * mesh.lengths[:, None]).T @ potentials)
    symmetry_error = float(np.max(np.abs(added_mass - added_mass.T)))
    return RigidBodyAddedMass2DResult(
        added_mass_matrix=added_mass,
        mode_potentials=potentials,
        mode_normals=mode_normals,
        center=(float(origin[0]), float(origin[1])),
        condition_number=condition,
        max_relative_residual_l2=float(np.max(rel_residuals)),
        symmetry_error=symmetry_error,
    )


def _validate_field_request(
    mesh: PanelMesh, result: LaplaceDirichletResult, points: FloatArray
) -> tuple[FloatArray, str]:
    """Shared validation for the interior/exterior field evaluators."""
    if not isinstance(mesh, PanelMesh):
        raise TypeError("mesh must be a PanelMesh")
    if not isinstance(result, LaplaceDirichletResult):
        raise TypeError("result must be a LaplaceDirichletResult")
    domain = _validate_domain(result.domain)
    targets = np.asarray(points, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("points must have shape (M, 2)")
    if not np.isfinite(targets).all():
        raise ValueError("points must all be finite")
    if (
        result.normal_derivative.shape != (mesh.panel_count,)
        or result.boundary_values.shape != (mesh.panel_count,)
    ):
        raise ValueError("result boundary traces must match the mesh panel count")
    return targets, domain


def _panel_representation_field(
    mesh: PanelMesh,
    result: LaplaceDirichletResult,
    targets: FloatArray,
    *,
    double_layer_sign: float,
) -> FloatArray:
    """Single/double-layer representation of the harmonic field at off-body points.

    The constant-panel kernels are integrated analytically over each straight
    panel, so the only error left is the piecewise-constant boundary trace.
    ``double_layer_sign = -1`` reproduces the interior representation
    ``u = S[q] - D[u]``; ``+1`` gives the exterior one ``u = S[q_e] + D[u]``.
    """
    lengths = mesh.lengths
    tangents = mesh.vectors / lengths[:, None]
    normals = mesh.normals
    reference = result.reference_length
    values = np.zeros(len(targets), dtype=float)
    for i, target in enumerate(targets):
        for j in range(mesh.panel_count):
            single = integrate_single_layer_panel(
                target, mesh.vertices[j], tangents[j], float(lengths[j]), reference
            )
            double = integrate_double_layer_panel(
                target,
                mesh.vertices[j],
                tangents[j],
                normals[j],
                float(lengths[j]),
            )
            values[i] += (
                single * result.normal_derivative[j]
                + double_layer_sign * double * result.boundary_values[j]
            )
    return values


def evaluate_interior_field(
    mesh: PanelMesh, result: LaplaceDirichletResult, points: FloatArray
) -> FloatArray:
    """Evaluate the harmonic potential at interior points using exact panel integrals.

    Points must be strictly inside the domain and not on the boundary. The
    representation integrates each constant-panel kernel analytically rather
    than sampling the kernel at the panel midpoint.
    """
    targets, domain = _validate_field_request(mesh, result, points)
    if domain != "interior":
        raise ValueError(
            "evaluate_interior_field requires a result from "
            "solve_laplace_dirichlet; use evaluate_exterior_field for the "
            "exterior formulation"
        )
    return _panel_representation_field(mesh, result, targets, double_layer_sign=-1.0)


def exterior_net_flux(mesh: PanelMesh, result: LaplaceDirichletResult) -> float:
    """Return ``integral(q_e ds)`` over the boundary for an exterior result.

    A closed body in an unbounded fluid admits a *bounded* exterior potential
    only when this integral vanishes (rigid-body translations and rotations
    always satisfy it). A non-zero value means the trace pair has no decaying
    harmonic extension, so the field grows like ``-(Q/2) log r`` and no
    bounded solution exists.
    """
    if not isinstance(mesh, PanelMesh):
        raise TypeError("mesh must be a PanelMesh")
    if not isinstance(result, LaplaceDirichletResult):
        raise TypeError("result must be a LaplaceDirichletResult")
    flux = np.asarray(result.normal_derivative, dtype=float)
    if flux.shape != (mesh.panel_count,):
        raise ValueError("result boundary traces must match the mesh panel count")
    return float(np.sum(flux * mesh.lengths))


def exterior_dirichlet_decay_ratio(
    mesh: PanelMesh, result: LaplaceDirichletResult
) -> float:
    """Return ``|integral(q_e ds)| / integral(|q_e| ds)``, the decay diagnostic.

    ``0`` means the boundary data lie in the decaying class the exterior
    single-layer representation spans; ``1`` means a pure capacity/constant
    mode (necessarily spurious, since the bounded solution then has zero flux).
    """
    if not isinstance(mesh, PanelMesh):
        raise TypeError("mesh must be a PanelMesh")
    if not isinstance(result, LaplaceDirichletResult):
        raise TypeError("result must be a LaplaceDirichletResult")
    flux = np.asarray(result.normal_derivative, dtype=float)
    if flux.shape != (mesh.panel_count,):
        raise ValueError("result boundary traces must match the mesh panel count")
    scale = float(np.sum(np.abs(flux) * mesh.lengths))
    if scale == 0.0:
        return 0.0
    return abs(float(np.sum(flux * mesh.lengths))) / scale


def evaluate_exterior_field(
    mesh: PanelMesh,
    result: LaplaceDirichletResult,
    points: FloatArray,
    *,
    require_decaying: bool = True,
) -> FloatArray:
    """Evaluate the harmonic potential in the *exterior* fluid domain.

    Accepts a result from :func:`solve_laplace_exterior_dirichlet` or
    :func:`solve_laplace_exterior_neumann` and evaluates

        u(x) = integral(G q_e ds) + integral(u dG/dn ds),   x outside the body,

    where ``n`` is the body outward normal and ``q_e = grad(u) . n_e`` is the
    trace along the fluid-domain normal ``n_e = -n`` (the convention used by
    both exterior solvers). The sign of the double-layer term is the only
    difference from :func:`evaluate_interior_field`: the jump relation flips it
    as the observation point leaves the closed curve, and getting it wrong
    doubles the double-layer contribution instead of cancelling it.

    Points must lie outside the body and off its boundary. Both panel kernels
    are integrated analytically, so the remaining error is the
    piecewise-constant boundary trace.

    Scope: the representation is exact for any exterior trace pair
    ``(q_e, u)`` that satisfies the exterior boundary integral equation. When
    ``integral(q_e ds) != 0`` no bounded harmonic exterior solution exists and
    the representation reproduces the ``log r`` growth, whose additive
    constant is set by ``result.reference_length``; rigid-body translations and
    rotations of a closed body always have zero net flux, so this does not
    arise for the added-mass modes.
    """
    targets, domain = _validate_field_request(mesh, result, points)
    if domain != "exterior":
        raise ValueError(
            "evaluate_exterior_field requires a result from an exterior solve "
            "(domain='exterior'); use evaluate_interior_field for "
            "solve_laplace_dirichlet"
        )
    if require_decaying:
        ratio = exterior_dirichlet_decay_ratio(mesh, result)
        if ratio > _NET_FLUX_TOL:
            raise ValueError(
                "the exterior trace pair carries a non-zero net flux "
                f"(|Q|/sum|q|ds = {ratio:.3e} > {_NET_FLUX_TOL:.0e}), so it has "
                "no bounded harmonic extension: the reconstructed potential "
                "grows like -(Q/2) log r and is not a solution of the exterior "
                "Dirichlet problem. Non-zero net flux arises when the Dirichlet "
                "data have a non-zero boundary mean, which the decaying "
                "single-layer representation cannot represent; see "
                "docs/exterior_field_verification.md. Pass "
                "require_decaying=False to obtain the (unbounded) "
                "representation anyway."
            )
    return _panel_representation_field(mesh, result, targets, double_layer_sign=1.0)
