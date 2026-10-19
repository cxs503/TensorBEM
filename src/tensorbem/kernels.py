"""Two-dimensional Laplace boundary-integral kernels."""

from __future__ import annotations

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
