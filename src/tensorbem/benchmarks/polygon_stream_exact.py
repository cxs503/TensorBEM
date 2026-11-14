"""Closed-form uniform flow past a regular polygon (2-D potential flow).

The anchor case ``CASE-BEM-LAPLACE-CYLINDER`` measures the BEM field of a
constant-panel polygon discretisation against the *circular* cylinder solution
``W(z) = U (z + a**2 / z)``.  On an inscribed ``n``-gon that reference is not
exact: the discrete body is a polygon, not a circle, so the measured error
mixes two different things

* the boundary-element error of the panel approximation of the polygon, and
* the *faceting* error of the polygon itself (an ``O(h**2)`` geometric defect
  that no amount of panel refinement removes, it only shrinks with ``h``).

This module supplies the exact solution for the polygon, so the two can be
separated.  Uniform flow past a regular ``n``-gon (circumradius ``a``) is the
image of the unit-circle solution under the exterior Schwarz-Christoffel map

.. math::

    z(w) = c \\int^{w} (1 - t^{-n})^{2/n}\\, dt,
    \\qquad c = \\frac{a \\sin(\\pi/n)}{\\alpha\\, 2^{\\alpha} I(\\alpha)},
    \\quad \\alpha = \\frac{2}{n},

with :math:`I(\\alpha) = \\int_0^{\\pi/2} \\sin^{\\alpha}\\psi\\, d\\psi
= \\frac{\\sqrt{\\pi}}{2}\\,\\Gamma\\!\\left(\\frac{1+\\alpha}{2}\\right)
\\Big/ \\Gamma\\!\\left(1+\\frac{\\alpha}{2}\\right)`.
The map is fixed by the requirement that the vertices sit at
:math:`a e^{2\\pi i k/n}`: it sends the arc midpoint of each unit-circle
segment to the midpoint of the corresponding edge, and

.. math:: z(e^{i\\varphi}) - z(1) = i e^{i\\pi/n} c \\int_0^{\\varphi}
          (2\\sin(n\\varphi'/2))^{\\alpha} d\\varphi',

so the image of the unit circle is exactly the polygon.  The complex potential
is :math:`W = U c\\, (w + 1/w)`; the extra factor ``c`` is what makes the
far-field speed equal to ``U`` in the physical plane.  The complex velocity is

.. math::

    \\frac{dW}{dz} = U \\frac{1 - w^{-2}}{(1 - w^{-n})^{\\alpha}},

which is finite and non-zero everywhere outside the body.  At the midpoint of
every edge :math:`w^{-n} = -1`, hence

.. math:: \\left|v\\right| = \\frac{2 U \\left|\\sin\\theta_w\\right|}{2^{\\alpha}}

with no numerical quadrature at all: the wall velocity used for the surface
comparison is closed form.

Everything here is independent of the boundary-integral machinery in
:mod:`tensorbem.solver`; the only shared geometry is the vertex list of
:func:`tensorbem.geometry.regular_polygon`.

Provenance: Lamb (Hydrodynamics, 6th ed., 1932) for the circle solution,
Milne-Thomson (Theoretical Hydrodynamics, 5th ed., 1968) for the circle theorem
and its use with conformal maps; the Schwarz-Christoffel exterior map of a
regular polygon is classical (Driscoll & Trefethen, Schwarz-Christoffel
Mapping, CUP 2002, ch. 2, exterior maps).  No page numbers are claimed: the
formulas are re-derived and gated numerically by :func:`geometry_gate`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

MAX_SERIES_TERMS = 512

__all__ = [
    "scale_factor",
    "half_edge_moment",
    "map_points",
    "map_derivative",
    "inverse_map",
    "exact_velocity",
    "exact_potential",
    "surface_midpoint_velocity",
    "surface_midpoint_angles",
    "apothem",
    "geometry_gate",
]


def _alpha(panel_count: int) -> float:
    if panel_count < 3:
        raise ValueError("a regular polygon needs at least three edges")
    return 2.0 / float(panel_count)


def half_edge_moment(alpha: float) -> float:
    """``I(alpha) = int_0^{pi/2} sin(psi)**alpha dpsi`` in closed form."""
    if alpha <= -1.0:
        raise ValueError("alpha must exceed -1 for the moment to converge")
    return (
        0.5
        * math.sqrt(math.pi)
        * math.gamma(0.5 * (1.0 + alpha))
        / math.gamma(1.0 + 0.5 * alpha)
    )


def scale_factor(panel_count: int, radius: float = 1.0) -> float:
    """Map dilation ``c`` that puts the vertices at ``radius`` from the centre."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    alpha = _alpha(panel_count)
    return radius * math.sin(math.pi / panel_count) / (
        alpha * 2.0**alpha * half_edge_moment(alpha)
    )


def apothem(panel_count: int, radius: float = 1.0) -> float:
    """Distance from the centre to an edge of the regular polygon."""
    return radius * math.cos(math.pi / panel_count)


def _binomial_row(alpha: float, count: int) -> np.ndarray:
    """``binom(alpha, k)`` for ``k = 0 .. count-1`` evaluated by recurrence."""
    row = np.empty(count, dtype=float)
    row[0] = 1.0
    for k in range(1, count):
        row[k] = row[k - 1] * (alpha - k + 1) / k
    return row


def _series_terms(w: np.ndarray, panel_count: int, extra: int = 0) -> int:
    """Number of Laurent terms that make the ``w**(-n)`` series machine exact."""
    radial = np.min(np.abs(w))
    if radial <= 1.0:
        raise ValueError(
            "the exterior series needs |w| > 1; boundary values are closed form"
        )
    # truncate when |w|**(-n*K) < 1e-18 (at least four terms for the gate)
    needed = int(math.ceil(18.0 * math.log(10.0) / (panel_count * math.log(radial))))
    if needed > MAX_SERIES_TERMS:
        raise ValueError(
            "target point is too close to the wall for the exterior Laurent "
            f"series ({needed} terms needed); the series is only used for "
            f"|w| >= 1.02, boundary data are closed form"
        )
    return max(4, needed) + extra


def map_points(
    w: np.ndarray, panel_count: int, radius: float = 1.0, extra_terms: int = 0
) -> np.ndarray:
    """Physical image ``z(w)`` of exterior points ``|w| > 1`` (Laurent series)."""
    w = np.asarray(w, dtype=complex)
    alpha = _alpha(panel_count)
    c = scale_factor(panel_count, radius)
    count = _series_terms(w, panel_count, extra_terms)
    coefficients = _binomial_row(alpha, count)
    powers = 1.0 - panel_count * np.arange(count)
    terms = (
        (coefficients * (-1.0) ** np.arange(count) / powers)[None, :]
        * w[:, None] ** powers[None, :]
    )
    return c * np.sum(terms, axis=1)


def map_derivative(
    w: np.ndarray, panel_count: int, radius: float = 1.0, extra_terms: int = 0
) -> np.ndarray:
    """``dz/dw = c (1 - w**(-n))**(2/n)`` for ``|w| > 1``."""
    w = np.asarray(w, dtype=complex)
    if np.any(np.abs(w) <= 1.0):
        raise ValueError("dz/dw is evaluated only outside the unit circle")
    alpha = _alpha(panel_count)
    c = scale_factor(panel_count, radius)
    count = _series_terms(w, panel_count, extra_terms)
    coefficients = _binomial_row(alpha, count)
    powers = -panel_count * np.arange(count)
    terms = (
        (coefficients * (-1.0) ** np.arange(count))[None, :]
        * w[:, None] ** powers[None, :]
    )
    return c * np.sum(terms, axis=1)


def inverse_map(
    z: np.ndarray,
    panel_count: int,
    radius: float = 1.0,
    iterations: int = 40,
    tolerance: float = 1e-14,
) -> np.ndarray:
    """Solve ``z(w) = z`` by Newton iteration in the exterior.

    The initial guess ``w = z/c`` is the far-field behaviour; convergence is
    quadratic and the returned residual is checked against ``tolerance``.
    """
    z = np.asarray(z, dtype=complex)
    c = scale_factor(panel_count, radius)
    alpha = _alpha(panel_count)
    w = z / c
    # the far-field guess |w| = |z|/c is poor within a panel width of the wall,
    # where the local behaviour of the map around the nearest edge midpoint is
    # z - m_k ~ (dz/dphi) (w - e^{i theta_k}) with dz/dphi = i e^{i theta_k} c 2^alpha
    near = np.abs(w) < 1.0 + 0.1
    if np.any(near):
        theta = np.angle(z[near])
        index = np.round((theta * panel_count / np.pi - 1.0) / 2.0)
        theta_k = (2.0 * index + 1.0) * np.pi / panel_count
        midpoints = apothem(panel_count, radius) * np.exp(1j * theta_k)
        w[near] = np.exp(1j * theta_k) + (z[near] - midpoints) / (c * 2.0**alpha)
    for _ in range(iterations):
        residual = map_points(w, panel_count, radius) - z
        step = residual / map_derivative(w, panel_count, radius)
        w = w - step
        if np.max(np.abs(step)) <= tolerance * np.max(np.abs(w)):
            break
    else:  # pragma: no cover - only reachable for degenerate input
        raise RuntimeError("inverse Schwarz-Christoffel map did not converge")
    if np.any(np.abs(w) <= 1.0):
        raise ValueError("target points must lie outside the polygon")
    return w


def _complex_velocity(
    w: np.ndarray, panel_count: int, radius: float = 1.0, speed: float = 1.0
) -> np.ndarray:
    """``dW/dz = U (1 - w**(-2)) / (1 - w**(-n))**(2/n)``."""
    alpha = _alpha(panel_count)
    w = np.asarray(w, dtype=complex)
    return speed * (1.0 - w**-2) / (1.0 - w**-panel_count) ** alpha


def exact_velocity(
    points: np.ndarray,
    panel_count: int,
    radius: float = 1.0,
    speed: float = 1.0,
) -> np.ndarray:
    """Exact velocity ``(u, v)`` of uniform flow past the regular polygon."""
    pts = np.asarray(points, dtype=float)
    w = inverse_map(pts[:, 0] + 1j * pts[:, 1], panel_count, radius)
    complex_velocity = _complex_velocity(w, panel_count, radius, speed)
    return np.column_stack((complex_velocity.real, -complex_velocity.imag))


def _map_on_unit_circle(phi: np.ndarray, panel_count: int, radius: float = 1.0) -> np.ndarray:
    """``z`` for boundary angles ``phi`` measured from vertex 0.

    Path: straight edge integral along the first edge, ``w = e^{i phi}``, with
    the analytic branch ``dz/dphi = i e^{i pi/n} c (2 sin(n phi/2))**alpha``.
    """
    alpha = _alpha(panel_count)
    c = scale_factor(panel_count, radius)
    nodes, weights = np.polynomial.legendre.leggauss(240)
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    out = np.empty(phi.shape, dtype=complex)
    # the integrand behaves like phi**alpha at the vertex, so grade the
    # substitution phi = upper * t**p with p = 2/(1+alpha) to make it smooth
    power = 2.0 / (1.0 + alpha)
    for index, value in enumerate(phi):
        upper = float(value)
        if upper < 0.0 or upper > 2.0 * math.pi / panel_count + 1e-15:
            raise ValueError("phi must lie on the first edge, [0, 2 pi/n]")
        t = 0.5 * (nodes + 1.0)
        sample = upper * t**power
        integrand = (2.0 * np.sin(panel_count * sample / 2.0)) ** alpha
        integral = 0.5 * upper * float(
            np.sum(weights * integrand * power * t ** (power - 1.0))
        )
        out[index] = radius + 1j * np.exp(1j * math.pi / panel_count) * c * integral
    return out


def surface_midpoint_angles(panel_count: int) -> np.ndarray:
    """Wall angles (from the +x axis) of every edge midpoint."""
    return np.pi * (2.0 * np.arange(panel_count) + 1.0) / panel_count


def surface_midpoint_positions(panel_count: int, radius: float = 1.0) -> np.ndarray:
    """Edge midpoints of the regular polygon (circumradius ``radius``)."""
    theta = surface_midpoint_angles(panel_count)
    r = apothem(panel_count, radius)
    return np.column_stack((r * np.cos(theta), r * np.sin(theta)))


def surface_midpoint_velocity(
    panel_count: int, radius: float = 1.0, speed: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form wall velocity at the edge midpoints.

    Returns ``(velocity, speed)`` with ``velocity`` of shape ``(n, 2)`` and the
    complex velocity evaluated exactly at ``w**(-n) = -1``, so no quadrature or
    inversion enters the surface comparison.
    """
    theta = surface_midpoint_angles(panel_count)
    w = np.exp(1j * theta)
    alpha = _alpha(panel_count)
    complex_velocity = speed * (1.0 - w**-2) / 2.0**alpha
    velocity = np.column_stack((complex_velocity.real, -complex_velocity.imag))
    norm = np.hypot(velocity[:, 0], velocity[:, 1])
    return velocity, norm


def exact_potential(
    points: np.ndarray,
    panel_count: int,
    radius: float = 1.0,
    speed: float = 1.0,
) -> np.ndarray:
    """Total potential ``Re[U c (w + 1/w)]`` of the polygon flow."""
    pts = np.asarray(points, dtype=float)
    c = scale_factor(panel_count, radius)
    w = inverse_map(pts[:, 0] + 1j * pts[:, 1], panel_count, radius)
    return speed * c * (w + 1.0 / w).real


def geometry_gate(panel_count: int, radius: float = 1.0) -> dict[str, Any]:
    """Self-consistency of the map: scale, edge length, placement, symmetry.

    Every quantity is produced twice, once from the closed-form Gamma function
    and once from an independent Gauss-Legendre quadrature of the map itself,
    so a mistake in either route shows up as a non-zero residual.
    """
    alpha = _alpha(panel_count)
    c = scale_factor(panel_count, radius)
    nodes, weights = np.polynomial.legendre.leggauss(400)
    phi = 0.5 * (2.0 * math.pi / panel_count) * (nodes + 1.0)
    edge_quadrature = (
        c * 0.5 * (2.0 * math.pi / panel_count) * float(
            np.sum(weights * (2.0 * np.sin(panel_count * phi / 2.0)) ** alpha)
        )
    )
    edge_closed_form = 2.0 * radius * math.sin(math.pi / panel_count)
    midpoint = _map_on_unit_circle(np.array([math.pi / panel_count]), panel_count, radius)[0]
    midpoint_closed_form = apothem(panel_count, radius) * np.exp(
        1j * np.pi / panel_count
    )
    vertex = _map_on_unit_circle(np.array([2.0 * math.pi / panel_count]), panel_count, radius)[0]
    # probe the symmetry: rotating w by 2 pi/n must rotate z by the same angle
    w_probe = 1.6 * np.exp(1j * np.array([0.3, 0.3 + 2.0 * math.pi / panel_count]))
    z_probe = map_points(w_probe, panel_count, radius)
    rotation_residual = abs(
        z_probe[1] - z_probe[0] * np.exp(1j * 2.0 * math.pi / panel_count)
    ) / abs(z_probe[0])
    # round trip: z(w) -> w must return the input
    w_back = inverse_map(z_probe, panel_count, radius)
    roundtrip = float(np.max(np.abs(w_back - w_probe)) / np.max(np.abs(w_probe)))
    return {
        "panel_count": panel_count,
        "radius": radius,
        "alpha": alpha,
        "scale_factor": c,
        "half_edge_moment_gamma": half_edge_moment(alpha),
        "edge_length_quadrature": edge_quadrature,
        "edge_length_closed_form": edge_closed_form,
        "edge_length_relative_error": abs(edge_quadrature - edge_closed_form)
        / edge_closed_form,
        "midpoint_quadrature": [midpoint.real, midpoint.imag],
        "midpoint_closed_form": [midpoint_closed_form.real, midpoint_closed_form.imag],
        "midpoint_relative_error": abs(midpoint - midpoint_closed_form)
        / abs(midpoint_closed_form),
        "vertex_radius": abs(vertex),
        "vertex_radius_relative_error": abs(abs(vertex) - radius) / radius,
        "rotation_symmetry_residual": rotation_residual,
        "inverse_map_roundtrip": roundtrip,
    }