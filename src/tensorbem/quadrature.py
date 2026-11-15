"""Gauss-Legendre rules that survive a broken ``leggauss``.

``numpy.polynomial.legendre.leggauss`` is a Golub-Welsch implementation: it
assembles the symmetric tridiagonal Jacobi matrix and calls ``eigvalsh``.  That
call is not reliable on every LAPACK build.  On the loongarch64 nodes this
package is exercised on, ``leggauss(80)`` raises

    numpy.linalg.LinAlgError: Eigenvalues did not converge

and because ``tensorbem/free_surface.py`` built its 80-node rule at *import*
time, the failure was not a rare quadrature problem but a dead import: no
module in the package could be imported, so every cluster run of this
repository died before its first line of physics.  The same defect had already
been worked around once, locally, in
``benchmarks/interior_panel_suppression.py`` by stepping the Hulme quadrature
order down -- a mitigation that only helped that one study.

This module replaces the single point of failure.  ``gauss_legendre(n)`` keeps
using NumPy's rule wherever NumPy's rule is usable, so every number already
pinned in the verification records is reproduced bit for bit, and falls back to
a Newton iteration on the Legendre polynomial (three-term recurrence plus the
differentiated recurrence) when ``leggauss`` raises ``LinAlgError``.

The fallback is not a degraded substitute.  Measured against a 50-digit
``mpmath`` reference for the endpoint weight (the most sensitive one), the
Newton rule at ``n = 2560`` is accurate to ``2.0e-11`` relative while NumPy's
Golub-Welsch rule is accurate to ``1.4e-7`` -- the eigvalsh route loses weight
accuracy at high order, the Newton route does not.  A run that falls back
therefore still targets the accuracy the kernels claim; the choice of route is
recorded by :func:`rule_source` rather than assumed.

``free_surface`` also grows its rule along a ladder (80 ... 2560 nodes) chosen
from the oscillation phase, so the fallback is exercised at exactly the orders
where eigvalsh is least reliable -- the reason for fixing the highest order,
not just the import-time one.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: Newton is quadratically convergent from the ``cos`` initial guess (error
#: ``~1e-2`` for the smallest root, ``~n**-2`` in node units): 12 sweeps reach
#: ``4e-16`` in every order used here.  The bound exists so a pathological
#: convergence cannot spin, not because the iteration is expected to need it.
_NEWTON_MAX_SWEEPS = 24
_NEWTON_TOL = 4.0e-16
#: A node this close to zero is the odd-order centre node; it must not be
#: mirrored or the rule gains a duplicate node.
_ZERO_NODE_TOL = 1.0e-14
#: Tolerance of the cheap integrity check applied to NumPy's rule: the weights
#: of a Gauss rule on ``[-1, 1]`` sum to the length of the interval.
_WEIGHT_SUM_TOL = 1.0e-10

_RULES: dict[int, tuple[FloatArray, FloatArray]] = {}
_SOURCE: dict[int, str] = {}


def _validate_order(n: object) -> int:
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise TypeError("the quadrature order must be an integer")
    order = int(n)
    if order < 1:
        raise ValueError("the quadrature order must be at least 1")
    return order


def _legendre_and_derivative(x: FloatArray, n: int) -> tuple[FloatArray, FloatArray]:
    """``P_n(x)`` and ``P_n'(x)`` by the three-term recurrence.

    The derivative is carried along by differentiating the recurrence
    ``k P_k = (2k-1) x P_{k-1} - (k-1) P_{k-2}`` rather than recovered from
    ``n (P_{n-1} - x P_n) / (1 - x**2)``: near the endpoints the latter
    subtracts two ``O(1)`` numbers whose difference is ``O(1/n)``.
    """
    p_prev = np.ones_like(x)
    p_curr = x.copy()
    d_prev = np.zeros_like(x)
    d_curr = np.ones_like(x)
    for k in range(2, n + 1):
        p_next = ((2 * k - 1) * x * p_curr - (k - 1) * p_prev) / k
        d_next = ((2 * k - 1) * (p_curr + x * d_curr) - (k - 1) * d_prev) / k
        p_prev, p_curr = p_curr, p_next
        d_prev, d_curr = d_curr, d_next
    return p_curr, d_curr


def _newton_gauss_legendre(n: int) -> tuple[FloatArray, FloatArray]:
    """Nodes and weights by Newton iteration on ``P_n`` (no ``eigvalsh``)."""
    half = (n + 1) // 2
    index = np.arange(1, half + 1, dtype=float)
    # Chebyshev-like asymptotic guess for the positive roots, largest first.
    x = np.cos(np.pi * (index - 0.25) / (n + 0.5))
    for _ in range(_NEWTON_MAX_SWEEPS):
        p, dp = _legendre_and_derivative(x, n)
        step = p / dp
        x = x - step
        if np.all(np.abs(step) < _NEWTON_TOL):
            break
    p, dp = _legendre_and_derivative(x, n)
    # w = 2 / ((1 - x**2) P_n'(x)**2) is the Christoffel weight of the rule.
    weights_half = 2.0 / ((1.0 - x * x) * dp * dp)
    mirror = x > _ZERO_NODE_TOL
    nodes = np.concatenate((-x[mirror], x[::-1]))
    weights = np.concatenate((weights_half[mirror], weights_half[::-1]))
    order = np.argsort(nodes)
    return nodes[order], weights[order]


def _numpy_rule_is_usable(nodes: FloatArray, weights: FloatArray, n: int) -> bool:
    """Cheap integrity check on a rule NumPy returned without raising.

    ``eigvalsh`` is also capable of returning a converged but wrong
    decomposition on a broken build; a Gauss rule on ``[-1, 1]`` has weights
    that are positive, finite and sum to 2, so that is what is checked.
    """
    if nodes.shape != (n,) or weights.shape != (n,):
        return False
    if not (np.all(np.isfinite(nodes)) and np.all(np.isfinite(weights))):
        return False
    if np.any(weights <= 0.0):
        return False
    if np.any(np.abs(nodes) > 1.0):
        return False
    return abs(float(weights.sum()) - 2.0) <= _WEIGHT_SUM_TOL


def gauss_legendre(n: int) -> tuple[FloatArray, FloatArray]:
    """Nodes and weights of the ``n``-point Gauss-Legendre rule on ``[-1, 1]``.

    NumPy's ``leggauss`` is used when it works, so recorded results do not
    move.  If it raises ``LinAlgError`` (or returns a rule that fails the
    integrity check) the Newton rule is used instead, and
    :func:`rule_source` reports which one produced the numbers.
    """
    order = _validate_order(n)
    cached = _RULES.get(order)
    if cached is not None:
        return cached
    try:
        nodes, weights = leggauss(order)
    except np.linalg.LinAlgError:
        nodes, weights = _newton_gauss_legendre(order)
        source = "newton-fallback"
    else:
        if _numpy_rule_is_usable(np.asarray(nodes), np.asarray(weights), order):
            source = "numpy"
        else:
            nodes, weights = _newton_gauss_legendre(order)
            source = "newton-fallback"
    nodes = np.asarray(nodes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    _RULES[order] = (nodes, weights)
    _SOURCE[order] = source
    return nodes, weights


def rule_source(n: int) -> str:
    """``"numpy"`` or ``"newton-fallback"`` for the rule cached for ``n``.

    Raises ``KeyError`` if the rule has not been requested yet, so a report
    cannot claim a route that was never taken.
    """
    order = _validate_order(n)
    if order not in _SOURCE:
        raise KeyError(f"no rule has been built for order {order}")
    return _SOURCE[order]


def clear_cache() -> None:
    """Drop the cached rules (used by the tests, and by anything that wants a
    route re-decided after the linear algebra backend changed underneath)."""
    _RULES.clear()
    _SOURCE.clear()


def rule_sums_to_two(n: int) -> float:
    """``sum(weights)`` for order ``n``; the invariant every Gauss rule meets."""
    _, weights = gauss_legendre(n)
    return float(math.fsum(weights))


__all__ = [
    "FloatArray",
    "clear_cache",
    "gauss_legendre",
    "rule_source",
    "rule_sums_to_two",
]