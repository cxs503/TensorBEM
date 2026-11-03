"""Kramers-Kronig (causality) relations for hydrodynamic coefficients.

A linear, causal, time-invariant fluid force satisfies a memory integral

    F(t) = -a(inf) * accel(t) - int_0^inf K(tau) * vel(t - tau) dtau,   K real,

so the complex response ``Phi(omega) = B(omega) + i*omega*(a(inf) - a(omega))``
is the Fourier transform of a one-sided kernel and is therefore analytic in the
upper half of the complex ``omega`` plane.  Analyticity plus decay at infinity
is exactly the Hilbert-transform pair

    a(omega) - a(inf) =  (2/pi) * PV int_0^inf  B(s) / (s**2 - omega**2) ds      (A)
    B(omega)          =  (2/pi) * PV int_0^inf  s**2 * (a(inf) - a(s))
                                                 / (s**2 - omega**2) ds         (B)

with ``PV`` the Cauchy principal value (Wehausen 1971; Newman, *Marine
Hydrodynamics*, 1977, ch. 6).  Equations (A) and (B) are not consequences of the
boundary-integral discretisation: they follow from causality alone, so a solver
that reproduces them on an independent frequency sweep has been checked against
a law of physics rather than against another solution of the same equations.

Two practical notes:

* ``a(inf)`` is needed, but it can be avoided.  Because ``a(inf)`` is a constant,
  differences ``a(w1) - a(w2)`` are fixed by ``B`` alone; equivalently the
  quantity ``a(w) - (2/pi)*PV int ...`` must be independent of ``w``.  That
  residual-constancy is what :func:`implied_infinite_added_mass` reports.
* In the deep-water submerged-cylinder problem the damping decays exponentially
  (no wave can be radiated once the free surface has become effectively rigid),
  so the infinite integration range can be truncated at modest ``ka`` with a
  negligible tail.

The principal-value quadrature is the only numerically delicate ingredient and
lives in :func:`principal_value_integral`; the rest of this module is bookkeeping
for relations (A) and (B).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss

__all__ = [
    "principal_value_integral",
    "added_mass_gap",
    "damping_from_added_mass",
    "implied_infinite_added_mass",
    "kk_residual_metrics",
    "sum_rule_residual",
]


def _as_grid(grid, values, name_grid="grid", name_values="values"):
    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=float)
    if grid.ndim != 1 or values.ndim != 1:
        raise ValueError(f"{name_grid} and {name_values} must be one-dimensional")
    if grid.size != values.size:
        raise ValueError(f"{name_grid} and {name_values} must have the same length")
    if grid.size < 5:
        raise ValueError("at least five samples are required for the quadrature")
    if not np.all(np.isfinite(grid)) or not np.all(np.isfinite(values)):
        raise ValueError("grid and values must be finite")
    if not np.all(np.diff(grid) > 0.0):
        raise ValueError(f"{name_grid} must be strictly increasing")
    if grid[0] <= 0.0:
        raise ValueError("the frequency grid must start above zero")
    return grid, values


def _local_lagrange(
    nodes: np.ndarray, values: np.ndarray, targets: np.ndarray, stencil: int = 4
) -> np.ndarray:
    """Evaluate the local ``stencil``-point interpolant of ``values`` at ``targets``.

    A fixed-width stencil keeps the interpolation local (no Runge ringing) while
    remaining exact for polynomials of degree ``stencil-1``; with four nodes the
    interpolation error inside a grid cell is ``O(h**4)``.
    """
    nodes = np.asarray(nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    targets = np.atleast_1d(np.asarray(targets, dtype=float))
    count = nodes.size
    width = min(int(stencil), count)
    if width < 2:
        raise ValueError("stencil must cover at least two nodes")
    start = np.clip(np.searchsorted(nodes, targets) - width // 2, 0, count - width)
    pick = start[:, None] + np.arange(width)[None, :]
    xs = nodes[pick]
    ys = values[pick]

    delta = targets[:, None] - xs
    exact = delta == 0.0
    safe_delta = np.where(exact, 1.0, delta)
    numerator_all = np.prod(safe_delta, axis=1)
    numerators = numerator_all[:, None] / safe_delta

    differences = xs[:, :, None] - xs[:, None, :]
    mask = ~np.eye(width, dtype=bool)[None, :, :]
    denominators = np.prod(np.where(mask, differences, 1.0), axis=2)

    weights = numerators / denominators
    result = np.sum(weights * ys, axis=1)
    if np.any(exact):
        rows, cols = np.nonzero(exact)
        result[rows] = ys[rows, cols]
    return result


def _lagrange_derivative(nodes: np.ndarray, values: np.ndarray, x: float) -> float:
    """Derivative at ``x`` of the polynomial interpolating ``(nodes, values)``."""
    total = 0.0
    for i, xi in enumerate(nodes):
        term = 0.0
        for j, xj in enumerate(nodes):
            if j == i:
                continue
            product = 1.0 / (xi - xj)
            for k, xk in enumerate(nodes):
                if k == i or k == j:
                    continue
                product *= (x - xk) / (xi - xk)
            term += product
        total += values[i] * term
    return total


def _gauss_legendre(a: float, b: float, order: int):
    nodes, weights = leggauss(order)
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    return mid + half * nodes, half * weights


def _principal_value_segment(l: float, r: float, omega: float) -> float:
    """``PV int_l^r ds / (s**2 - omega**2)`` for a segment not straddling omega."""
    return (1.0 / (2.0 * omega)) * math.log(
        abs((r - omega) * (l + omega)) / abs((r + omega) * (l - omega))
    )


def principal_value_integral(grid, values, omega: float, *, order: int = 8) -> float:
    r"""Cauchy principal value of ``int values(s) / (s**2 - omega**2) ds``.

    ``values`` is sampled on ``grid``; ``omega`` must lie strictly inside the
    grid.  The one or two cells that touch ``omega`` are integrated with the
    second-order subtraction

        F(s)/(s**2-omega**2) = [F(s) - F(omega) - F'(omega)(s-omega)]/(s**2-omega**2)
                             + F(omega)/(s**2-omega**2)
                             + F'(omega)/(s+omega),

    whose last two terms are analytic on the merged cell (the two logarithmic
    divergences of the half-cells cancel in the principal value).  Every other
    cell uses the cheaper subtraction ``[F(s)-F(omega)]/(s**2-omega**2)`` plus
    the closed-form principal value of ``1/(s**2-omega**2)``, which keeps
    near-singular cells smooth without changing the exact value.
    """
    grid, values = _as_grid(grid, values)
    omega = float(omega)
    if not math.isfinite(omega) or omega <= grid[0] or omega >= grid[-1]:
        raise ValueError("omega must lie strictly inside the grid")
    if order < 2:
        raise ValueError("order must be at least 2")

    f_omega = float(_local_lagrange(grid, values, omega)[0])
    nearest = int(np.argmin(np.abs(grid - omega)))
    if abs(grid[nearest] - omega) <= 1e-12 * max(1.0, omega):
        # omega is (numerically) a grid node: merge its two adjacent cells.
        left_index, right_index = nearest - 1, nearest + 1
    else:
        right_index = int(np.searchsorted(grid, omega))
        left_index = right_index - 1
    if left_index < 0 or right_index > grid.size - 1:
        raise ValueError("omega is too close to the grid boundary")

    lo = float(grid[left_index])
    hi = float(grid[right_index])

    span = min(5, grid.size)
    start = min(max(nearest - span // 2, 0), grid.size - span)
    f_prime = _lagrange_derivative(
        grid[start : start + span], values[start : start + span], omega
    )

    total = 0.0
    for i in range(grid.size - 1):
        if left_index <= i < right_index:
            continue  # merged singular block, handled below
        a, b = float(grid[i]), float(grid[i + 1])
        s, w = _gauss_legendre(a, b, order)
        smooth = _local_lagrange(grid, values, s) - f_omega
        total += float(np.sum(w * smooth / (s * s - omega * omega)))
        total += f_omega * _principal_value_segment(a, b, omega)

    sub_edges = [lo] + ([omega] if lo < omega < hi else []) + [hi]
    for a, b in zip(sub_edges[:-1], sub_edges[1:]):
        s, w = _gauss_legendre(a, b, order)
        regularised = (_local_lagrange(grid, values, s) - f_omega - f_prime * (s - omega)) / (
            s * s - omega * omega
        )
        total += float(np.sum(w * regularised))
    total += f_omega * _principal_value_segment(lo, hi, omega)
    total += f_prime * math.log((hi + omega) / (lo + omega))
    return float(total)


def added_mass_gap(omega_grid, damping, omega_eval, *, order: int = 8) -> np.ndarray:
    """Relation (A): ``a(omega) - a(inf)`` from the damping sweep."""
    grid, values = _as_grid(omega_grid, damping, "omega_grid", "damping")
    targets = np.atleast_1d(np.asarray(omega_eval, dtype=float))
    out = np.empty(targets.size, dtype=float)
    for index, omega in enumerate(targets):
        out[index] = (2.0 / math.pi) * principal_value_integral(
            grid, values, float(omega), order=order
        )
    return out


def damping_from_added_mass(
    omega_grid, added_mass, infinite_added_mass: float, omega_eval, *, order: int = 8
) -> np.ndarray:
    """Relation (B): ``B(omega)`` from the added-mass sweep and ``a(inf)``."""
    grid, values = _as_grid(omega_grid, added_mass, "omega_grid", "added_mass")
    integrand = grid * grid * (float(infinite_added_mass) - values)
    targets = np.atleast_1d(np.asarray(omega_eval, dtype=float))
    out = np.empty(targets.size, dtype=float)
    for index, omega in enumerate(targets):
        out[index] = (2.0 / math.pi) * principal_value_integral(
            grid, integrand, float(omega), order=order
        )
    return out


def implied_infinite_added_mass(
    omega_grid, damping, added_mass, *, order: int = 8
) -> dict:
    """Estimate ``a(inf)`` from causality and report how constant it is.

    Relation (A) fixes ``a(omega) - a(inf)``; subtracting it from the tabulated
    ``a(omega)`` must leave a constant.  The mean is the implied ``a(inf)`` and
    the spread around that mean measures how well the data satisfy causality.
    """
    grid, damping_values = _as_grid(omega_grid, damping, "omega_grid", "damping")
    _, mass_values = _as_grid(omega_grid, added_mass, "omega_grid", "added_mass")
    # The principal value needs omega strictly inside the grid, so the two
    # outermost samples (which carry a negligible amount of damping anyway) are
    # excluded from the comparison.
    margin = 2 if grid.size > 8 else 1
    eval_grid = grid[margin:-margin]
    gap = added_mass_gap(grid, damping_values, eval_grid, order=order)
    implied = mass_values[margin:-margin] - gap
    mean = float(np.mean(implied))
    spread = float(np.max(np.abs(implied - mean)))
    return {
        "implied_infinite_added_mass": mean,
        "implied_added_mass_max_deviation": spread,
        "implied_added_mass_relative_spread": spread / abs(mean) if mean else float("inf"),
        "omega_evaluated": eval_grid,
        "added_mass_reference": mass_values[margin:-margin],
        "reconstructed_added_mass": mean + gap,
    }


def kk_residual_metrics(reference, reconstructed, *, floor: float = 0.0) -> dict:
    """Relative-error summary of a reconstruction against a reference sweep."""
    reference = np.asarray(reference, dtype=float)
    reconstructed = np.asarray(reconstructed, dtype=float)
    scale = max(float(np.max(np.abs(reference))), float(floor), 1e-300)
    error = reconstructed - reference
    return {
        "max_absolute_error": float(np.max(np.abs(error))),
        "max_relative_error": float(np.max(np.abs(error)) / scale),
        "rms_relative_error": float(math.sqrt(float(np.mean(error**2))) / scale),
    }


def sum_rule_residual(omega_grid, added_mass, infinite_added_mass: float) -> float:
    """The ``omega -> 0`` limit of relation (B): ``int_0^inf (a - a(inf)) ds = 0``."""
    grid, values = _as_grid(omega_grid, added_mass, "omega_grid", "added_mass")
    integrand = values - float(infinite_added_mass)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(integrand, grid))
    return float(np.trapz(integrand, grid))