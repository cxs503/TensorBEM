"""Two-dimensional Laplace boundary-integral kernels."""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
_TWO_PI = 2.0 * np.pi
REFERENCE_LENGTH = 2.0


def validate_reference_length(reference_length: float) -> float:
    """Return a validated logarithmic reference length.

    The single-layer kernel is ``-(1/(2*pi))*log(r/reference_length)``. The
    reference length shifts the kernel by a constant. For a bounded domain that
    shift is immaterial (a harmonic Dirichlet problem has zero net flux, so the
    solution is unchanged), and for the exterior problem as well the solution
    is unchanged. What the shift does change is *conditioning*: integrating a
    constant density over a closed curve gives ``a*log(reference_length/a)``
    for a circle of radius ``a``, so the single-layer matrix approaches rank
    deficiency when the curve's logarithmic capacity equals the reference
    length. The value must therefore be positive, finite and explicit.
    """
    if isinstance(reference_length, (bool, np.bool_)):
        raise ValueError("reference_length must be a positive finite number")
    value = float(reference_length)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("reference_length must be a positive finite number")
    return value


def validate_clearance_factor(factor: float) -> float:
    """Return a validated field-evaluation clearance factor.

    ``factor`` multiplies the largest panel size to give the distance below
    which a field-evaluation target is rejected. Negative, non-finite and
    boolean inputs are refused: a factor of ``0`` or less silently disables the
    guard, and the evaluation still returns a finite (wrong) number, which the
    caller cannot detect from the value itself.
    """
    if isinstance(factor, (bool, np.bool_)):
        raise ValueError("minimum_clearance_factor must be a finite non-negative number")
    value = float(factor)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("minimum_clearance_factor must be a finite non-negative number")
    return value


def fundamental_solution(
    distance: FloatArray | float, reference_length: float = REFERENCE_LENGTH
) -> FloatArray:
    r"""Return G(r) = -(1/(2*pi))*log(r/reference_length), with -Laplacian(G)=delta.

    In two dimensions the fundamental solution is defined up to an additive
    constant. The reference length carries that constant: it avoids the
    unit-capacity degeneracy of the single-layer operator for a unit circle
    with the unshifted logarithm. A closed curve whose logarithmic capacity
    equals ``reference_length`` still makes the single-layer matrix singular;
    see :func:`validate_reference_length`.
    """
    reference = validate_reference_length(reference_length)
    r = np.asarray(distance, dtype=float)
    if np.any(r <= 0.0):
        raise ValueError("distance must be positive; handle singular panel integrals separately")
    return -np.log(r / reference) / _TWO_PI


def double_layer_kernel(
    target: FloatArray, source: FloatArray, source_normal: FloatArray
) -> float:
    r"""Evaluate dG/dn_y = (x-y).n_y/(2*pi*|x-y|^2) for x != y."""
    delta = np.asarray(target, dtype=float) - np.asarray(source, dtype=float)
    r2 = float(np.dot(delta, delta))
    if r2 <= 0.0:
        raise ValueError("double-layer point kernel is singular at coincident points")
    return float(np.dot(delta, source_normal) / (_TWO_PI * r2))


def _log_distance_primitive(z: float, normal_distance: float) -> float:
    """Primitive of log(sqrt(z*z + normal_distance**2)); stable at z=0."""
    h = abs(normal_distance)
    if h <= 1.0e-14:
        if abs(z) <= 1.0e-14:
            return 0.0
        return z * math.log(abs(z)) - z
    return 0.5 * z * math.log(z * z + h * h) - z + h * math.atan(z / h)


def integrate_single_layer_panel(
    target: FloatArray,
    start: FloatArray,
    tangent: FloatArray,
    length: float,
    reference_length: float = REFERENCE_LENGTH,
) -> float:
    r"""Integrate G(target, y) exactly over a straight constant panel.

    The source panel is y(s) = start + s*tangent, 0 <= s <= length.
    Unlike midpoint quadrature, this analytic integral accurately handles
    near-panel interactions as well as the logarithmic self-panel singularity.
    """
    x = np.asarray(target, dtype=float)
    a = np.asarray(start, dtype=float)
    t = np.asarray(tangent, dtype=float)
    if x.shape != (2,) or a.shape != (2,) or t.shape != (2,):
        raise ValueError("target, start, and tangent must each have shape (2,)")
    if not np.isfinite(x).all() or not np.isfinite(a).all() or not np.isfinite(t).all():
        raise ValueError("panel integral inputs must be finite")
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("length must be a positive finite number")
    reference = validate_reference_length(reference_length)
    xi = float(np.dot(x - a, t))
    normal_distance = float(np.dot(x - a, np.array([-t[1], t[0]])))
    integral_log_r = (
        _log_distance_primitive(length - xi, normal_distance)
        - _log_distance_primitive(-xi, normal_distance)
    )
    return -(integral_log_r - length * math.log(reference)) / _TWO_PI


def integrate_double_layer_panel(
    target: FloatArray,
    start: FloatArray,
    tangent: FloatArray,
    normal: FloatArray,
    length: float,
) -> float:
    r"""Integrate dG/dn_y over a straight panel analytically.

    The target is assumed not to lie on the panel. For a target on the same
    straight panel, the principal-value contribution is zero.
    """
    x = np.asarray(target, dtype=float)
    a = np.asarray(start, dtype=float)
    t = np.asarray(tangent, dtype=float)
    n = np.asarray(normal, dtype=float)
    if any(v.shape != (2,) for v in (x, a, t, n)):
        raise ValueError("target, start, tangent, and normal must each have shape (2,)")
    if not all(np.isfinite(v).all() for v in (x, a, t, n)):
        raise ValueError("panel integral inputs must be finite")
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("length must be a positive finite number")
    delta = x - a
    xi = float(np.dot(delta, t))
    eta = float(np.dot(delta, n))
    h = abs(eta)
    if h <= 1.0e-14:
        return 0.0
    z0, z1 = -xi, length - xi
    return math.copysign(1.0, eta) * (
        math.atan(z1 / h) - math.atan(z0 / h)
    ) / _TWO_PI


def _panel_local_coordinates(
    target: FloatArray,
    start: FloatArray,
    tangent: FloatArray,
    normal: FloatArray,
) -> tuple[float, float]:
    """Return ``(xi, eta)``: target coordinates along and across the panel."""
    x = np.asarray(target, dtype=float)
    a = np.asarray(start, dtype=float)
    t = np.asarray(tangent, dtype=float)
    n = np.asarray(normal, dtype=float)
    if any(v.shape != (2,) for v in (x, a, t, n)):
        raise ValueError("target, start, tangent, and normal must each have shape (2,)")
    if not all(np.isfinite(v).all() for v in (x, a, t, n)):
        raise ValueError("panel integral inputs must be finite")
    delta = x - a
    return float(np.dot(delta, t)), float(np.dot(delta, n))


def _validate_panel_length(length: float) -> float:
    if isinstance(length, (bool, np.bool_)) or not math.isfinite(float(length)):
        raise ValueError("length must be a positive finite number")
    value = float(length)
    if value <= 0.0:
        raise ValueError("length must be a positive finite number")
    return value


def integrate_single_layer_panel_gradient(
    target: FloatArray,
    start: FloatArray,
    tangent: FloatArray,
    length: float,
    reference_length: float = REFERENCE_LENGTH,
) -> FloatArray:
    r"""Gradient of :func:`integrate_single_layer_panel` with respect to the target.

    With ``xi``/``eta`` the target coordinates along the panel tangent and its
    left normal, the closed form is

        d/dx integral G ds = -(1/2 pi) * [
            t * (1/2) log((xi^2 + eta^2) / ((xi - L)^2 + eta^2))
            + n * (atan(z1/eta) - atan(z0/eta)) ],

    with ``z0 = -xi``, ``z1 = L - xi``. The reference-length shift of the
    potential is constant, so it drops out of the gradient.
    """
    length = _validate_panel_length(length)
    validate_reference_length(reference_length)
    t = np.asarray(tangent, dtype=float)
    n = np.array([-t[1], t[0]], dtype=float)
    xi, eta = _panel_local_coordinates(target, start, t, n)
    z0, z1 = -xi, length - xi
    along = 0.5 * math.log((xi * xi + eta * eta) / (z1 * z1 + eta * eta))
    if eta == 0.0:
        # Target on the panel's line: the integral of eta/rho^2 vanishes off
        # the panel and is undefined on it, so the caller must stay clear.
        across = 0.0
    else:
        across = math.atan(z1 / eta) - math.atan(z0 / eta)
    return -(t * along + n * across) / _TWO_PI


def integrate_double_layer_panel_gradient(
    target: FloatArray,
    start: FloatArray,
    tangent: FloatArray,
    normal: FloatArray,
    length: float,
) -> FloatArray:
    r"""Gradient of :func:`integrate_double_layer_panel` with respect to the target.

    Differentiating the constant-panel kernel ``eta/(2*pi*rho^2)`` (the panel
    normal ``n`` is fixed, the target moves) gives, with ``u = xi - s``,

        d/dx integral dG/dn ds = (1/2 pi) [ A n - 2 eta B t ],
        A = integral (u^2 - eta^2)/(u^2 + eta^2)^2 ds
          = -xi/(xi^2 + eta^2) - z1/(z1^2 + eta^2),
        B = integral u/(u^2 + eta^2)^2 ds
          = -1/(2 (xi^2 + eta^2)) + 1/(2 (z1^2 + eta^2)).

    ``B`` and ``A`` stay finite for a target on the panel's line (``eta = 0``)
    as long as it is not at an endpoint; that case is common for collinear
    panels of a polygonal boundary.
    """
    length = _validate_panel_length(length)
    n = np.asarray(normal, dtype=float)
    t = np.asarray(tangent, dtype=float)
    xi, eta = _panel_local_coordinates(target, start, t, n)
    z1 = length - xi
    r0 = xi * xi + eta * eta
    r1 = z1 * z1 + eta * eta
    if r0 <= 1.0e-28 or r1 <= 1.0e-28:
        raise ValueError(
            "double-layer gradient is singular at a panel endpoint; "
            "keep the target a finite distance from the boundary"
        )
    a_term = -xi / r0 - z1 / r1
    b_term = 0.5 * (1.0 / r1 - 1.0 / r0)
    return (a_term * n - 2.0 * eta * b_term * t) / _TWO_PI


# ---------------------------------------------------------------------------
# First-moment (linearly varying density) panel integrals.
#
# A piecewise-constant density makes the recovered field first-order accurate
# near the body: the representation error is set by the tangential variation of
# the density *within* a panel, which the basis cannot carry.  These integrals
# add the missing first moment, i.e. the contribution of a density that varies
# linearly across the panel, so a caller that can estimate the tangential slope
# recovers a second-order-accurate near-field.
#
# The abscissa ``s`` is measured from the panel start (0 <= s <= L), the same
# convention as the density traces in :mod:`tensorbem.solver`.
# ---------------------------------------------------------------------------


def _log_sqrt_first_moment_primitive(z: float, normal_distance: float) -> float:
    r"""Primitive of ``z*log(rho)`` with ``rho = sqrt(z*z + h*h)``, ``h=|eta|``.

    ``B(z) = ((z^2 + h^2) log(z^2 + h^2) - z^2) / 4``.  At ``r2 -> 0`` both the
    ``r2*log(r2)`` and ``z^2`` terms vanish, so ``B = 0`` is the continuous
    branch there and no singularity is introduced for a target at a panel
    endpoint.
    """
    h = abs(normal_distance)
    r2 = z * z + h * h
    if r2 <= 0.0:
        return 0.0
    return 0.25 * (r2 * math.log(r2) - z * z)


def integrate_single_layer_panel_first_moment(
    target: FloatArray,
    start: FloatArray,
    tangent: FloatArray,
    length: float,
    reference_length: float = REFERENCE_LENGTH,
) -> float:
    r"""Integrate ``s * G(target, y(s))`` exactly over a straight panel.

    With ``xi``/``eta`` the target coordinates along the panel tangent and its
    left normal, ``z0 = -xi``, ``z1 = L - xi`` and ``P``/``B`` the primitives of
    ``log(rho)`` and ``z*log(rho)``, the closed form is

        integral s G ds = -(1/2 pi) [ xi (P(z1)-P(z0)) + B(z1)-B(z0)
                                     - (L^2/2) log(L_ref) ].
    """
    length = _validate_panel_length(length)
    reference = validate_reference_length(reference_length)
    t = np.asarray(tangent, dtype=float)
    n = np.array([-t[1], t[0]], dtype=float)
    xi, eta = _panel_local_coordinates(target, start, t, n)
    z0, z1 = -xi, length - xi
    integral_log_r = _log_distance_primitive(z1, eta) - _log_distance_primitive(z0, eta)
    first_moment = xi * integral_log_r + (
        _log_sqrt_first_moment_primitive(z1, eta)
        - _log_sqrt_first_moment_primitive(z0, eta)
    )
    return -(first_moment - 0.5 * length * length * math.log(reference)) / _TWO_PI


def integrate_double_layer_panel_first_moment(
    target: FloatArray,
    start: FloatArray,
    tangent: FloatArray,
    normal: FloatArray,
    length: float,
) -> float:
    r"""Integrate ``s * dG/dn_y`` exactly over a straight panel.

    Same local coordinates as :func:`integrate_double_layer_panel`;

        integral s (dG/dn) ds = (1/2 pi) [ xi (atan(z1/eta) - atan(z0/eta))
                                           + eta log(rho1/rho0) ],

    which also vanishes on the panel's own line, matching the principal-value
    convention of the constant-density integral.
    """
    length = _validate_panel_length(length)
    n = np.asarray(normal, dtype=float)
    t = np.asarray(tangent, dtype=float)
    xi, eta = _panel_local_coordinates(target, start, t, n)
    h = abs(eta)
    if h <= 1.0e-14:
        return 0.0
    z0, z1 = -xi, length - xi
    angle = math.copysign(1.0, eta) * (math.atan(z1 / h) - math.atan(z0 / h))
    r2_0 = z0 * z0 + h * h
    r2_1 = z1 * z1 + h * h
    logarithm = 0.5 * math.log(r2_1 / r2_0)
    return (xi * angle + eta * logarithm) / _TWO_PI


def integrate_single_layer_panel_first_moment_gradient(
    target: FloatArray,
    start: FloatArray,
    tangent: FloatArray,
    length: float,
    reference_length: float = REFERENCE_LENGTH,
) -> FloatArray:
    r"""Gradient of :func:`integrate_single_layer_panel_first_moment`.

    Differentiating the closed form in ``(xi, eta)`` and mapping back with
    ``grad = t*d/dxi + n*d/deta`` (the panel frame is orthonormal) gives

        d/dxi  J = I + xi log(rho0/rho1) + z0 log(rho0) - z1 log(rho1),
        d/deta J = xi (atan(z1/eta) - atan(z0/eta)) + eta log(rho1/rho0),
        J = xi I + B(z1) - B(z0),  I = P(z1) - P(z0),

    and ``grad(integral s G ds) = -(t d/dxi + n d/deta) J / (2 pi)``.  The
    reference-length term is constant and drops out.
    """
    length = _validate_panel_length(length)
    validate_reference_length(reference_length)
    t = np.asarray(tangent, dtype=float)
    n = np.array([-t[1], t[0]], dtype=float)
    xi, eta = _panel_local_coordinates(target, start, t, n)
    h = abs(eta)
    z0, z1 = -xi, length - xi
    r2_0 = z0 * z0 + h * h
    r2_1 = z1 * z1 + h * h
    if r2_0 <= 1.0e-28 or r2_1 <= 1.0e-28:
        raise ValueError(
            "first-moment gradient is singular at a panel endpoint; "
            "keep the target a finite distance from the boundary"
        )
    log_rho0 = 0.5 * math.log(r2_0)
    log_rho1 = 0.5 * math.log(r2_1)
    integral_log_r = _log_distance_primitive(z1, eta) - _log_distance_primitive(z0, eta)
    along = integral_log_r + xi * (log_rho0 - log_rho1) + z0 * log_rho0 - z1 * log_rho1
    angle = math.copysign(1.0, eta) * (math.atan(z1 / h) - math.atan(z0 / h))
    across = xi * angle + eta * (log_rho1 - log_rho0)
    return -(t * along + n * across) / _TWO_PI


def integrate_double_layer_panel_first_moment_gradient(
    target: FloatArray,
    start: FloatArray,
    tangent: FloatArray,
    normal: FloatArray,
    length: float,
) -> FloatArray:
    r"""Gradient of :func:`integrate_double_layer_panel_first_moment`.

    With ``R = xi*Delta + eta*log(rho1/rho0)``, ``Delta = atan(z1/eta) -
    atan(z0/eta)`` and ``r2_i = z_i^2 + eta^2``,

        d R/d xi  = Delta + xi*eta (1/r2_0 - 1/r2_1) + eta (z0/r2_0 - z1/r2_1),
        d R/d eta = xi (z0/r2_0 - z1/r2_1) + log(rho1/rho0)
                    + eta^2 (1/r2_1 - 1/r2_0),

    and the gradient is ``(t d/dxi + n d/deta) R / (2 pi)``.
    """
    length = _validate_panel_length(length)
    n = np.asarray(normal, dtype=float)
    t = np.asarray(tangent, dtype=float)
    xi, eta = _panel_local_coordinates(target, start, t, n)
    h = abs(eta)
    z0, z1 = -xi, length - xi
    r2_0 = z0 * z0 + h * h
    r2_1 = z1 * z1 + h * h
    if r2_0 <= 1.0e-28 or r2_1 <= 1.0e-28:
        raise ValueError(
            "first-moment gradient is singular at a panel endpoint; "
            "keep the target a finite distance from the boundary"
        )
    angle = math.copysign(1.0, eta) * (math.atan(z1 / h) - math.atan(z0 / h))
    logarithm = 0.5 * math.log(r2_1 / r2_0)
    along = (
        angle
        + xi * eta * (1.0 / r2_0 - 1.0 / r2_1)
        + eta * (z0 / r2_0 - z1 / r2_1)
    )
    across = (
        xi * (z0 / r2_0 - z1 / r2_1)
        + logarithm
        + eta * eta * (1.0 / r2_1 - 1.0 / r2_0)
    )
    return (t * along + n * across) / _TWO_PI
