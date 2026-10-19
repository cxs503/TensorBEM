"""Two-dimensional Laplace boundary-integral kernels."""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
_TWO_PI = 2.0 * np.pi
REFERENCE_LENGTH = 2.0


def fundamental_solution(distance: FloatArray | float) -> FloatArray:
    r"""Return G(r) = -(1/(2*pi))*log(r/2), with -Laplacian(G)=delta.

    In two dimensions the fundamental solution is defined up to an additive
    constant. The reference length avoids the unit-capacity degeneracy of the
    single-layer operator for a unit circle with the unshifted logarithm.
    """
    r = np.asarray(distance, dtype=float)
    if np.any(r <= 0.0):
        raise ValueError("distance must be positive; handle singular panel integrals separately")
    return -np.log(r / REFERENCE_LENGTH) / _TWO_PI


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
    target: FloatArray, start: FloatArray, tangent: FloatArray, length: float
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
    xi = float(np.dot(x - a, t))
    normal_distance = float(np.dot(x - a, np.array([-t[1], t[0]])))
    integral_log_r = (
        _log_distance_primitive(length - xi, normal_distance)
        - _log_distance_primitive(-xi, normal_distance)
    )
    return -(integral_log_r - length * math.log(REFERENCE_LENGTH)) / _TWO_PI


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
