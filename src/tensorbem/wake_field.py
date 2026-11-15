"""Nominal-wake inflow and relative rotative efficiency for a propeller behind a hull.

Why this module exists
----------------------
The ship-performance platform's propulsion block returns
``relative_rot_eff`` as the constant ``1.0``.  That is not a bug in one
function: the block evaluates the *open-water* B-series regression at the
behind-hull advance ratio ``J = V (1 - w) / (n D)`` and then feeds those same
coefficients into both sides of the definition

    eta_R = eta_0,open-water(J) / eta_0,behind-hull(J) ,               (1)

so the ratio is identically one.  A number that cannot be other than one is not
an estimate of anything.  What is missing is a *behind-hull* operating point:
the propeller working in the non-uniform afterbody wake, at the same mean
advance ratio.

This module supplies exactly that, and nothing else:

* :func:`nominal_wake_fraction` -- a radial nominal-wake profile whose
  area-weighted mean equals the hull's wake fraction ``w`` (the only wake
  quantity the platform reports);
* :func:`wake_inflow` -- the blade-fixed inflow field of that profile, ready for
  ``propeller.open_water_point(..., inflow_field=...)``: axial component
  ``J (1 - w(r)) / (1 - w_bar)`` plus the rotational ``-omega r e_theta`` term;
* :func:`relative_rotative_efficiency` -- Eq. (1) with the guards that make it
  refuse a tautological input;
* :func:`interpolate_open_water_curve` -- the open-water curve read at an
  arbitrary ``J`` (the "敞水曲线在对应 J 处插值" step).

Declared modelling assumptions (fixed before any number was produced)
--------------------------------------------------------------------
1. The wake is **axial only**: no swirl, no radial velocity.  The rotational
   inflow is the undisturbed ``-omega r e_theta``.  A wake swirl would change
   ``K_Q`` directly; it is omitted because the platform reports no swirl, and the
   omission is stated wherever the resulting ``eta_R`` is used.
2. The radial shape is the one-parameter family

       w(xi) = w_bar * phi(xi) / <phi>,   phi(xi) = (1 - xi)**q,
       xi = (r - r_hub) / (R - r_hub)                                   (2)

   i.e. the deficit is concentrated inboard (the counter/deadwater region behind
   a full single-screw afterbody) and vanishes at the tip.  ``q`` is an
   **assumed** shape exponent: the measured KP505/KCS nominal wake is not
   available offline in this worktree (same provenance limit as the anchor in
   ``benchmarks/anchors/kp505_openwater.json``), so ``q`` is never fitted -- the
   declared primary value and the sensitivity band are fixed in the caller and
   ``q = 0`` (uniform wake) is carried as the **null control**: for a uniform
   wake the field equals the open-water inflow and Eq. (1) must return exactly
   1, which is what a working implementation has to reproduce.
3. ``<phi>`` is the area-weighted mean over the blade annulus,
   ``<phi> = int phi r dr / int r dr``, so that ``w`` is the mass-flux mean.  With
   that normalisation the *mean* advance ratio of the family is ``J`` for every
   ``q``, and any departure of Eq. (1) from 1 is caused by the radial
   non-uniformity alone -- not by a change of the operating point.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Nominal wake profile
# ---------------------------------------------------------------------------


def _phi(xi: FloatArray, exponent: float) -> FloatArray:
    return np.power(np.clip(1.0 - np.asarray(xi, dtype=float), 0.0, None), exponent)


def _annulus_mean(exponent: float, hub_ratio: float) -> float:
    """Area-weighted mean of ``phi`` over ``r/R in [hub_ratio, 1]`` -- closed form.

    With ``a = r_hub/R``, ``b = 1 - a`` and ``phi = (1-xi)**q`` on
    ``xi = (r/R - a)/b``,

        int_0^1 phi (a + b xi) dxi = a/(q+1) + b/((q+1)(q+2))
        int_0^1 (a + b xi) dxi     = (1 + a)/2

    The closed form matters: ``phi`` has an infinite slope at ``xi = 1`` for
    non-integer ``q``, so a quadrature leaves the mean -- and therefore the
    mean advance ratio of the inflow -- off by O(1e-4), which is the same order
    as the effect being measured.
    """

    if exponent == 0.0:
        return 1.0
    a = hub_ratio
    b = 1.0 - hub_ratio
    numerator = a / (exponent + 1.0) + b / ((exponent + 1.0) * (exponent + 2.0))
    return float(numerator / (0.5 * (1.0 + a)))


def nominal_wake_fraction(
    radius_ratio: FloatArray | float,
    wake_fraction: float,
    *,
    exponent: float = 0.5,
    hub_ratio: float = 0.18,
) -> FloatArray:
    """Radial nominal wake fraction ``w(r/R)`` with area mean ``wake_fraction``.

    ``exponent = 0`` gives the uniform wake ``w == wake_fraction`` (null
    control); larger ``exponent`` concentrates the deficit towards the hub.
    """

    if not 0.0 <= hub_ratio < 1.0:
        raise ValueError("hub_ratio must satisfy 0 <= hub_ratio < 1")
    if exponent < 0.0:
        raise ValueError("exponent must be non-negative")
    radius_ratio = np.asarray(radius_ratio, dtype=float)
    if np.any(radius_ratio < hub_ratio - 1e-12) or np.any(radius_ratio > 1.0 + 1e-12):
        raise ValueError("radius_ratio must lie in [hub_ratio, 1]")
    xi = (radius_ratio - hub_ratio) / (1.0 - hub_ratio)
    return wake_fraction * _phi(xi, exponent) / _annulus_mean(exponent, hub_ratio)


def wake_inflow(
    advance_ratio: float,
    wake_fraction: float,
    *,
    exponent: float = 0.5,
    hub_ratio: float = 0.18,
    angular_rate: float = TWO_PI,
    diameter: float = 1.0,
) -> Callable[[FloatArray], FloatArray]:
    """Blade-fixed inflow field ``W(P)`` of the nominal wake (axial only).

    The axial component is ``J (1 - w(r)) / (1 - w_bar)``: the local advance
    rate normalised so that its area average is ``J``.  The tangential component
    is the undisturbed ``-omega r e_theta``.  The returned field has the same
    signature and units as :func:`tensorbem.propeller.inflow_velocity`, so it
    plugs into ``open_water_point(..., inflow_field=...)`` unchanged.
    """

    if not 0.0 <= wake_fraction < 1.0:
        raise ValueError("wake_fraction must lie in [0, 1)")

    def inflow(points: FloatArray) -> FloatArray:
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        r = np.hypot(points[:, 1], points[:, 2])
        radius_ratio = np.clip(r / (0.5 * diameter), hub_ratio, 1.0)
        w_local = nominal_wake_fraction(
            radius_ratio, wake_fraction, exponent=exponent, hub_ratio=hub_ratio
        )
        axial = advance_ratio * (1.0 - w_local) / (1.0 - wake_fraction)
        e_theta = np.stack(
            [
                np.zeros_like(r),
                -points[:, 2] / np.maximum(r, 1e-12),
                points[:, 1] / np.maximum(r, 1e-12),
            ],
            axis=-1,
        )
        return np.stack([axial, np.zeros_like(r), np.zeros_like(r)], axis=-1) - (
            angular_rate * r[:, None] * e_theta
        )

    return inflow


# ---------------------------------------------------------------------------
# eta_R from the open-water / behind-hull pair
# ---------------------------------------------------------------------------


def relative_rotative_efficiency(
    eta0_open_water: float,
    eta0_behind_hull: float,
    *,
    tolerance: float = 1e-9,
) -> float:
    """``eta_R = eta_0,open-water / eta_0,behind-hull`` (ITTC 7.5-02-03-01.5).

    The function refuses the tautology it replaces: if the two efficiencies are
    the *same* number the input is not a behind-hull point but the same
    open-water point counted twice, and the result carries no information.  That
    is reported, not hidden: ``nan`` with a ``RuntimeWarning`` instead of a
    silent ``1.0``.
    """

    import warnings

    eta0_ow = float(eta0_open_water)
    eta0_bh = float(eta0_behind_hull)
    if not (math.isfinite(eta0_ow) and math.isfinite(eta0_bh)):
        return float("nan")
    if eta0_bh <= 0.0:
        warnings.warn(
            f"eta_0,behind-hull = {eta0_bh:g} <= 0: eta_R undefined",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")
    if abs(eta0_ow - eta0_bh) <= tolerance * max(abs(eta0_ow), 1e-30):
        warnings.warn(
            "eta_0,open-water == eta_0,behind-hull: the behind-hull point is the "
            "open-water point, so eta_R carries no information (the platform's "
            "constant 1.0 failure mode)",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")
    return eta0_ow / eta0_bh


def open_water_efficiency(advance_ratio: float, thrust: float, torque: float) -> float:
    """``eta_0 = J K_T / (2 pi K_Q)`` (the definition, in one place)."""

    if torque <= 0.0:
        return float("nan")
    return advance_ratio * thrust / (TWO_PI * torque)


def interpolate_open_water_curve(
    advance_ratios: Sequence[float],
    values: Sequence[float],
    advance_ratio: float,
) -> float:
    """Read an open-water curve (``K_T``, ``K_Q`` or ``eta_0``) at ``J``.

    Piecewise-linear in ``J`` on the tabulated ladder, extended linearly (not
    clipped) outside it so that an out-of-ladder request is visible as such in
    the recorded interpolation residual instead of being frozen at the end
    value.
    """

    js = np.asarray(advance_ratios, dtype=float)
    vs = np.asarray(values, dtype=float)
    order = np.argsort(js)
    js, vs = js[order], vs[order]
    j = float(advance_ratio)
    if js.size < 2:
        raise ValueError("an open-water curve needs at least two points")
    return float(np.interp(j, js, vs))