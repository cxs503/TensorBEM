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

from .conditioning import condition_number_with_fallback
from .geometry import PanelMesh
from .kernels import (
    REFERENCE_LENGTH,
    validate_clearance_factor,
    integrate_double_layer_panel,
    integrate_double_layer_panel_gradient,
    integrate_double_layer_panel_gradient_many,
    integrate_double_layer_panel_many,
    integrate_single_layer_panel,
    integrate_single_layer_panel_gradient_many,
    integrate_single_layer_panel_many,
    integrate_double_layer_panel_first_moment,
    integrate_double_layer_panel_first_moment_gradient,
    integrate_single_layer_panel_first_moment,
    integrate_single_layer_panel_first_moment_gradient,
    integrate_single_layer_panel_gradient,
    validate_reference_length,
)

FloatArray = NDArray[np.float64]
_DOMAINS = ("interior", "exterior")
# Beyond this condition number the dense solve is no longer meaningful in
# double precision; it is the logarithmic-capacity degeneracy in practice.
_MAX_CONDITION = 1.0e13


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
    linear_density: bool = False
    condition_number_source: str = "svd"


def _validate_domain(domain: str) -> str:
    if domain not in _DOMAINS:
        raise ValueError(f"domain must be one of {_DOMAINS}")
    return domain


def _slope_weights(mesh: PanelMesh) -> tuple[FloatArray, FloatArray]:
    """Coefficients of backward and forward increments of the local quadratic."""
    lengths = mesh.lengths
    left = 0.5 * (lengths + np.roll(lengths, 1))
    right = 0.5 * (lengths + np.roll(lengths, -1))
    span = left + right
    return (right / span) / left, (left / span) / right


def _difference_operator(mesh: PanelMesh) -> FloatArray:
    """Periodic derivative at the midpoints of possibly unequal panels."""
    n = mesh.panel_count
    backward, forward = _slope_weights(mesh)
    operator = np.zeros((n, n), dtype=float)
    rows = np.arange(n)
    operator[rows, (rows + 1) % n] = forward
    operator[rows, rows] = backward - forward
    operator[rows, (rows - 1) % n] = -backward
    return operator


def assemble_matrices(
    mesh: PanelMesh,
    reference_length: float = REFERENCE_LENGTH,
    linear_density: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Assemble analytically integrated single- and double-layer matrices.

    ``linear_density=True`` integrates the densities of both layers as
    ``sigma(s) = sigma_i + sigma'_i (s - L_i/2)``, with the tangential slopes
    ``sigma'`` supplied by the arc-length difference operator
    :func:`_difference_operator`. The extra term is added in closed form with
    the first-moment panel integrals, so no quadrature error enters and the
    boundary equation stays a linear system in the panel-midpoint values; the
    resulting solve is second-order accurate on a smooth boundary instead of
    first-order. The jump coefficient ``1/2 I`` is unchanged: the double-layer
    kernel on the collocation panel is a delta at the midpoint, where the
    linear basis equals the midpoint value.
    """
    reference = validate_reference_length(reference_length)
    centers = mesh.centers
    starts = mesh.vertices
    tangents = mesh.vectors / mesh.lengths[:, None]
    normals = mesh.normals
    lengths = mesh.lengths
    n = mesh.panel_count
    single = np.empty((n, n), dtype=float)
    double = np.empty((n, n), dtype=float)
    if linear_density:
        single_moment = np.empty((n, n), dtype=float)
        double_moment = np.empty((n, n), dtype=float)

    for i, target in enumerate(centers):
        for j in range(n):
            single[i, j] = integrate_single_layer_panel(
                target, starts[j], tangents[j], float(lengths[j]), reference
            )
            double[i, j] = integrate_double_layer_panel(
                target, starts[j], tangents[j], normals[j], float(lengths[j])
            )
            if linear_density:
                single_moment[i, j] = integrate_single_layer_panel_first_moment(
                    target, starts[j], tangents[j], float(lengths[j]), reference
                ) - 0.5 * float(lengths[j]) * single[i, j]
                double_moment[i, j] = integrate_double_layer_panel_first_moment(
                    target, starts[j], tangents[j], normals[j], float(lengths[j])
                ) - 0.5 * float(lengths[j]) * double[i, j]
    if not linear_density:
        return single, double
    difference = _difference_operator(mesh)
    return single + single_moment @ difference, double + double_moment @ difference


def _solve(
    mesh: PanelMesh,
    boundary_values: FloatArray,
    domain: str,
    reference_length: float,
    linear_density: bool = False,
) -> LaplaceDirichletResult:
    values = np.asarray(boundary_values, dtype=float)
    n = mesh.panel_count
    if values.shape != (n,):
        raise ValueError(f"boundary_values must have shape ({n},)")
    if not np.isfinite(values).all():
        raise ValueError("boundary_values must all be finite")
    reference = validate_reference_length(reference_length)

    single, double = assemble_matrices(
        mesh, reference_length=reference, linear_density=linear_density
    )
    sign = 1.0 if domain == "interior" else -1.0
    rhs = 0.5 * values + sign * (double @ values)
    condition, condition_source = condition_number_with_fallback(single)
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
        linear_density=bool(linear_density),
        condition_number_source=condition_source,
    )


def solve_laplace_dirichlet(
    mesh: PanelMesh,
    boundary_values: FloatArray,
    reference_length: float = REFERENCE_LENGTH,
    linear_density: bool = False,
) -> LaplaceDirichletResult:
    """Solve for the unknown outward normal derivative given boundary values.

    The baseline targets smooth, closed, simply connected 2-D domains.
    Boundary values are piecewise constant at panel midpoints. The returned
    ``normal_derivative`` is the derivative along the polygon's outward normal.
    ``linear_density=True`` uses the first-moment (linearly varying density)
    panel basis, which is second-order accurate on a smooth boundary.
    """
    return _solve(
        mesh,
        boundary_values,
        "interior",
        reference_length,
        linear_density=linear_density,
    )


def solve_laplace_exterior_dirichlet(
    mesh: PanelMesh,
    boundary_values: FloatArray,
    reference_length: float = REFERENCE_LENGTH,
    linear_density: bool = False,
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
    linear_density: bool = False,
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

    ``linear_density=True`` uses the first-moment (linearly varying density)
    panel basis for both layers, making the solve second-order accurate on a
    smooth boundary; this is the setting that
    :func:`evaluate_exterior_field` documents for its ``linear_density``
    near-wall mode.
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

    single, double = assemble_matrices(
        mesh, reference_length=reference, linear_density=linear_density
    )
    system = 0.5 * np.eye(n) - double
    rhs = single @ flux
    condition, condition_source = condition_number_with_fallback(system)
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
        linear_density=bool(linear_density),
        condition_number_source=condition_source,
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
    condition, _ = condition_number_with_fallback(system)
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


def _interior_targets(mesh, result, points, minimum_clearance_factor):
    if not isinstance(result, LaplaceDirichletResult):
        raise TypeError("result must be a LaplaceDirichletResult")
    if result.domain != "interior":
        raise ValueError(
            "evaluate_interior_field requires a result from "
            "solve_laplace_dirichlet; the exterior formulation is not implemented "
            "for field evaluation"
        )
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

    factor = validate_clearance_factor(minimum_clearance_factor)
    h_max = float(np.max(mesh.lengths))
    clearance = factor * h_max
    distance = mesh.boundary_clearance(targets)
    too_close = (distance <= 1.0e-14 * h_max) | (distance < clearance)
    if np.any(too_close):
        index = int(np.flatnonzero(too_close)[0])
        raise ValueError(
            "evaluation point is on or too close to the boundary; choose a "
            "point strictly inside the polygon "
            f"(first: index {index} at distance {distance[index]:.3e}, "
            f"threshold {max(clearance, 1.0e-14 * h_max):.3e} = "
            "minimum_clearance_factor * h_max)"
        )
    outside = ~mesh.contains_point(targets)
    if np.any(outside):
        index = int(np.flatnonzero(outside)[0])
        raise ValueError(
            "evaluation points must lie strictly inside the domain; "
            f"{int(np.sum(outside))} of {len(targets)} point(s) are outside the "
            f"polygon (first: index {index} at {targets[index, :].tolist()}). "
            "The interior Green representation is not valid outside the domain, "
            "and it returns a finite but meaningless value there."
        )

    return targets


def evaluate_interior_field(
    mesh: PanelMesh,
    result: LaplaceDirichletResult,
    points: FloatArray,
    minimum_clearance_factor: float = 0.0,
) -> FloatArray:
    """Evaluate the harmonic potential at interior points using exact panel integrals.

    Points must be strictly inside the domain and not on the boundary. The
    representation integrates each constant-panel kernel analytically rather
    than sampling the kernel at the panel midpoint.

    Only the interior Green representation is valid, so targets are screened
    before the representation is evaluated: a target outside the polygon, or
    closer to the boundary than ``minimum_clearance_factor * h_max`` (``h_max``
    the longest panel), is refused with ``ValueError``. Outside the domain the
    Green representation converges to a finite but meaningless number, so an
    unscreened call returns a plausible-looking wrong value -- the failure the
    guard exists to convert into an exception. The factor is validated like the
    3-D field evaluators (finite and non-negative).
    """
    targets = _interior_targets(mesh, result, points, minimum_clearance_factor)
    lengths = mesh.lengths
    tangents = mesh.vectors / lengths[:, None]
    normals = mesh.normals
    reference = result.reference_length
    # One numpy pass per panel over every target, instead of one python-level
    # kernel call per (target, panel) pair: the per-target sum still runs over
    # the panels in increasing order, and the vectorised kernels are the same
    # closed forms (tests/test_kernels_batch.py pins them against the scalar
    # ones to round-off).
    values = np.zeros(len(targets), dtype=float)
    for j in range(mesh.panel_count):
        single = integrate_single_layer_panel_many(
            targets, mesh.vertices[j], tangents[j], float(lengths[j]), reference
        )
        double = integrate_double_layer_panel_many(
            targets,
            mesh.vertices[j],
            tangents[j],
            normals[j],
            float(lengths[j]),
        )
        values += (
            single * result.normal_derivative[j]
            - double * result.boundary_values[j]
        )
    return values


@dataclass(frozen=True)
class ExteriorField:
    """Potential and its gradient at points of the unbounded exterior domain."""

    potential: FloatArray
    gradient: FloatArray


def _screen_exterior_targets(
    mesh: PanelMesh,
    points: FloatArray,
    minimum_clearance_factor: float,
) -> FloatArray:
    """Validate exterior field-evaluation targets and return them as an array."""
    if not isinstance(mesh, PanelMesh):
        raise TypeError("mesh must be a PanelMesh")
    targets = np.asarray(points, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("points must have shape (M, 2)")
    if not np.isfinite(targets).all():
        raise ValueError("points must all be finite")
    factor = validate_clearance_factor(minimum_clearance_factor)
    h_max = float(np.max(mesh.lengths))
    clearance = factor * h_max
    distance = mesh.boundary_clearance(targets)
    too_close = (distance <= 1.0e-12 * h_max) | (distance < clearance)
    if np.any(too_close):
        index = int(np.flatnonzero(too_close)[0])
        raise ValueError(
            "evaluation point is on or too close to the boundary; choose a "
            "point strictly outside the polygon "
            f"(first: index {index} at distance {distance[index]:.3e}, "
            f"threshold {max(clearance, 1.0e-12 * h_max):.3e} = "
            "minimum_clearance_factor * h_max)"
        )
    inside = mesh.contains_point(targets)
    if np.any(inside):
        index = int(np.flatnonzero(inside)[0])
        raise ValueError(
            "evaluation points must lie strictly outside the polygon; "
            f"{int(np.sum(inside))} of {len(targets)} point(s) are inside "
            f"(first: index {index} at {targets[index, :].tolist()}). The "
            "exterior Green representation is not valid inside the body, and "
            "it returns a finite but meaningless value there."
        )
    return targets


def tangential_slope(mesh: PanelMesh, values: FloatArray) -> FloatArray:
    r"""Estimate the arc-length derivative of a panel-midpoint trace.

    Interpolate the trace at each midpoint and its two periodic neighbours
    by a quadratic in arc length, then differentiate at the central point.
    This retains second-order accuracy with unequal neighbouring spacings.
    """
    if not isinstance(mesh, PanelMesh):
        raise TypeError("mesh must be a PanelMesh")
    trace = np.asarray(values, dtype=float)
    n = mesh.panel_count
    if trace.shape != (n,):
        raise ValueError(f"values must have shape ({n},)")
    if not np.isfinite(trace).all():
        raise ValueError("values must all be finite")
    backward, forward = _slope_weights(mesh)
    return backward * (trace - np.roll(trace, 1)) + forward * (np.roll(trace, -1) - trace)


def evaluate_exterior_field(
    mesh: PanelMesh,
    result: LaplaceDirichletResult,
    points: FloatArray,
    minimum_clearance_factor: float = 0.0,
    linear_density: bool | None = None,
) -> ExteriorField:
    r"""Evaluate the exterior harmonic potential and its gradient at fluid points.

    The exterior Green representation of a solution that decays at infinity is

        u(x) = integral( G q_e ds ) + integral( u dG/dn_body ds ),

    with ``q_e = grad(u) . n_e`` the fluid-domain normal derivative of the
    result and ``n_e = -n_body`` (the fluid normal is opposite the polygon's
    outward normal used by :func:`solve_laplace_exterior_dirichlet` /
    :func:`solve_laplace_exterior_neumann`). The two signs follow from the
    boundary equation ``(1/2 I - H) u = S q_e`` that those solvers invert.

    Both panel integrals are evaluated in closed form, including their target
    gradients, so no finite-difference step enters the reported velocity field.
    The gradient is insensitive to ``reference_length``; the potential itself
    is only defined up to a constant by an exterior Neumann solve, so callers
    that need absolute potentials must use the Dirichlet formulation.

    Targets strictly inside the polygon, on the boundary, or closer than
    ``minimum_clearance_factor * h_max`` are refused: outside their domain the
    panel integrals still return finite numbers, so an unscreened call would
    look plausible and be wrong.

    Accuracy window. The panel densities are piecewise constant, so the recovered
    gradient degrades within a few panel lengths of the body even though every
    panel integral is closed form. Case ``CASE-BEM-LAPLACE-CYLINDER`` measures
    this (`benchmarks/verified/case_bem_laplace_cylinder/results.json`,
    ``near_wall``): with the cylinder at ``a = 1`` the relative velocity error at
    ``delta = 2 h_max`` stays below 1% on every ladder level, while at
    ``delta = 0.1 h_max`` it reaches ~18% and does *not* shrink when the mesh is
    refined (it is a property of the constant-density representation, not of the
    linear solve). The default clearance factor stays zero for backward
    compatibility, so a caller evaluating close to the body should pass
    ``minimum_clearance_factor`` of order one to two.

    ``linear_density=True`` expands the densities as
    ``sigma(s) = sigma_i + sigma'_i (s - L_i/2)`` on each panel, where
    ``sigma'_i`` is the arc-length slope of the panel trace estimated by
    :func:`tangential_slope`, and the first-moment panel integrals of
    :mod:`tensorbem.kernels` supply the extra term in closed form (value and
    target gradient, no extra quadrature). That basis is *not* a uniform
    improvement; the extent of its gain is measured, not assumed
    (``near_wall`` in
    ``benchmarks/verified/case_bem_laplace_cylinder/results.json``):

    * above a panel *midpoint* it buys an order of magnitude at
      ``delta <= 0.5 h_max`` (cylinder: 2.8e-2 -> 3.2e-3 at ``delta = 0.5 h_max``)
      and moves the 1%-accurate window in from ``2 h_max`` to ``0.5 h_max``;
    * inside the documented window it is ~1.2x *worse* than constant panels, and
      its 1% window does not move in at all once the target is allowed to sit
      anywhere across the panel, not just above the midpoint;
    * towards a vertex it degrades sharply, because the arc-length slope
      extrapolates a density that is discontinuous across the kink: on the
      cylinder at ``delta = 0.1 h_max`` the worst-azimuth error is 1.40, 0.129
      and 1.26 on the 32/64/128-panel levels, i.e. not even monotone in ``h``.

    Use it for smooth (curved) boundaries and targets away from vertices; keep
    the constant basis plus a clearance guard as the default elsewhere.
    """
    if not isinstance(result, LaplaceDirichletResult):
        raise TypeError("result must be a LaplaceDirichletResult")
    if result.domain != "exterior":
        raise ValueError(
            "evaluate_exterior_field requires a result from an exterior solve "
            "(solve_laplace_exterior_dirichlet or solve_laplace_exterior_neumann)"
        )
    n = mesh.panel_count
    if (
        result.normal_derivative.shape != (n,)
        or result.boundary_values.shape != (n,)
    ):
        raise ValueError("result boundary traces must match the mesh panel count")
    if linear_density is None:
        linear_density = bool(getattr(result, "linear_density", False))
    elif not isinstance(linear_density, (bool, np.bool_)):
        raise ValueError("linear_density must be a boolean")
    targets = _screen_exterior_targets(mesh, points, minimum_clearance_factor)

    potential, gradient = _panel_field_and_gradient(
        mesh, targets, result.normal_derivative, result.boundary_values,
        result.reference_length, linear_density)
    return ExteriorField(potential=potential, gradient=gradient)


def evaluate_interior_field_and_gradient(
    mesh: PanelMesh, result: LaplaceDirichletResult, points: FloatArray,
    minimum_clearance_factor: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Return interior potential (M,) and Cartesian gradient (M,2).

    Differentiate the interior Green representation S*q - D*u using the
    density basis used in the solve. Targets must lie strictly inside the
    polygon and satisfy the same clearance rule as evaluate_interior_field.
    """
    targets = _interior_targets(mesh, result, points, minimum_clearance_factor)
    return _panel_field_and_gradient(
        mesh, targets, result.normal_derivative, -result.boundary_values,
        result.reference_length, result.linear_density)


def _panel_field_and_gradient(mesh, targets, flux, values, reference, linear_density):
    n = mesh.panel_count
    lengths = mesh.lengths
    tangents = mesh.vectors / lengths[:, None]
    normals = mesh.normals
    if linear_density:
        half = 0.5 * lengths
        flux_slope = tangential_slope(mesh, flux)
        value_slope = tangential_slope(mesh, values)
    potential = np.zeros(len(targets), dtype=float)
    gradient = np.zeros((len(targets), 2), dtype=float)
    if not linear_density:
        # Constant-density basis: batch the targets, keep the panel order.
        for j in range(n):
            start = mesh.vertices[j]
            tangent = tangents[j]
            length = float(lengths[j])
            panel_flux = flux[j]
            value = values[j]
            potential += (
                integrate_single_layer_panel_many(
                    targets, start, tangent, length, reference
                )
                * panel_flux
                + integrate_double_layer_panel_many(
                    targets, start, tangent, normals[j], length
                )
                * value
            )
            gradient += (
                integrate_single_layer_panel_gradient_many(
                    targets, start, tangent, length, reference
                )
                * panel_flux
                + integrate_double_layer_panel_gradient_many(
                    targets, start, tangent, normals[j], length
                )
                * value
            )
        return potential, gradient
    for i, target in enumerate(targets):
        for j in range(n):
            start = mesh.vertices[j]
            tangent = tangents[j]
            length = float(lengths[j])
            single = integrate_single_layer_panel(
                target, start, tangent, length, reference
            )
            double = integrate_double_layer_panel(
                target, start, tangent, normals[j], length
            )
            potential[i] += single * flux[j] + double * values[j]
            gradient[i] += (
                integrate_single_layer_panel_gradient(
                    target, start, tangent, length, reference
                )
                * flux[j]
                + integrate_double_layer_panel_gradient(
                    target, start, tangent, normals[j], length
                )
                * values[j]
            )
            if linear_density:
                # First moment of the density about the panel midpoint:
                #   integral (s - L/2) kernel ds = K1 - (L/2) K0.
                potential[i] += (
                    (
                        integrate_single_layer_panel_first_moment(
                            target, start, tangent, length, reference
                        )
                        - half[j] * single
                    )
                    * flux_slope[j]
                    + (
                        integrate_double_layer_panel_first_moment(
                            target, start, tangent, normals[j], length
                        )
                        - half[j] * double
                    )
                    * value_slope[j]
                )
                gradient[i] += (
                    (
                        integrate_single_layer_panel_first_moment_gradient(
                            target, start, tangent, length, reference
                        )
                        - half[j]
                        * integrate_single_layer_panel_gradient(
                            target, start, tangent, length, reference
                        )
                    )
                    * flux_slope[j]
                    + (
                        integrate_double_layer_panel_first_moment_gradient(
                            target, start, tangent, normals[j], length
                        )
                        - half[j]
                        * integrate_double_layer_panel_gradient(
                            target, start, tangent, normals[j], length
                        )
                    )
                    * value_slope[j]
                )
    return potential, gradient


def _boundary_integral_wall_slope(mesh, values, flux, tangents, linear_density):
    """Differentiate the exterior trace equation along each straight panel.

    In (I/2 - H)u = S q, represent the potential on panel j by its
    supplied midpoint value plus an unknown slope times (s-L_j/2).
    Collocating the differentiated equation at all midpoints determines
    those slopes. The on-panel H principal value is zero along the face,
    and its exterior jump is already included in the half-identity.
    """
    centers, lengths, normals = mesh.centers, mesh.lengths, mesh.normals
    system = 0.5 * np.eye(mesh.panel_count)
    rhs = np.zeros(mesh.panel_count)
    flux_slopes = tangential_slope(mesh, flux) if linear_density else None
    for j, (start, tangent, normal, length) in enumerate(zip(mesh.vertices, tangents, normals, lengths)):
        single = integrate_single_layer_panel_gradient_many(centers, start, tangent, length)
        double = integrate_double_layer_panel_gradient_many(centers, start, tangent, normal, length)
        ds = np.sum(single * tangents, axis=1)
        dh = np.sum(double * tangents, axis=1)
        ds[j] = dh[j] = 0.0
        # A constant double-layer density contributes zero tangential derivative.
        rhs += ds * flux[j] + dh * (values[j] - values)
        for i, target in enumerate(centers):
            if i != j:
                moment = integrate_double_layer_panel_first_moment_gradient(
                    target, start, tangent, normal, length
                )
                system[i, j] -= moment @ tangents[i] - 0.5 * length * dh[i]
            if flux_slopes is not None:
                if i == j:
                    weight = length / (2.0 * np.pi)
                else:
                    moment = integrate_single_layer_panel_first_moment_gradient(
                        target, start, tangent, length
                    )
                    weight = moment @ tangents[i] - 0.5 * length * ds[i]
                rhs[i] += weight * flux_slopes[j]
    return np.linalg.solve(system, rhs)


def evaluate_exterior_surface_velocity(
    mesh: PanelMesh, result: LaplaceDirichletResult, *, method: str = "trace_difference"
) -> FloatArray:
    r"""Fluid-side velocity ``grad(u)`` at panel midpoints of an exterior solve.

    The normal component is the solved ``q_e``. The tangential component is the
    arc-length derivative ``d(u)/ds`` of the panel-midpoint trace, taken with a
    periodic central difference on the closed boundary. For a smooth periodic
    trace that difference is second-order accurate, so the surface velocity is not
    limited by a first-order one-sided stencil.

    The polygon's counter-clockwise tangent is ``t`` and the panel spacing
    ``ds`` varies, so the difference is centred in arc length rather than in
    panel index.
    With ``method="boundary_integral"``, recover the tangential derivative
    from the differentiated exterior boundary integral instead. This solves
    an additional dense system for the wall slopes and accommodates jumps
    in the prescribed normal derivative at polygon corners.
    """
    if not isinstance(result, LaplaceDirichletResult):
        raise TypeError("result must be a LaplaceDirichletResult")
    if result.domain != "exterior":
        raise ValueError(
            "evaluate_exterior_surface_velocity requires a result from an "
            "exterior solve"
        )
    n = mesh.panel_count
    values = np.asarray(result.boundary_values, dtype=float)
    flux = np.asarray(result.normal_derivative, dtype=float)
    if values.shape != (n,) or flux.shape != (n,):
        raise ValueError("result boundary traces must match the mesh panel count")
    lengths = mesh.lengths
    tangents = mesh.vectors / lengths[:, None]
    # Arc-length difference of the panel midpoints (equally spaced in arc
    # length for an inscribed polygon of a circle, but not in general).
    if method == "trace_difference":
        derivative = tangential_slope(mesh, values)
    elif method == "boundary_integral":
        derivative = _boundary_integral_wall_slope(mesh, values, flux, tangents, result.linear_density)
    else:
        raise ValueError("method must be trace_difference or boundary_integral")
    return flux[:, None] * (-mesh.normals) + derivative[:, None] * tangents
