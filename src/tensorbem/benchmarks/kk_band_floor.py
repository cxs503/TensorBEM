"""How much of a Kramers-Kronig residual is the *estimator's* own discretisation?

Why this module exists
----------------------
:mod:`tensorbem.causality` answers *how far* a tabulated ``(omega, A, B)`` sweep
is from being causal.  On the Wigley III radiation sweep
(``CASE-BEM-CUMMINS-WIGLEY``) that answer is 40-80 % on several DOFs, and the
round-2 write-up could only *assert* that this was a band effect: "the KK
residual is normalised by ``max|A - A_inf|`` and the principal value is truncated
at the lowest tabulated frequency, so the remaining error is a band/scale
effect."  An assertion is not evidence, and the distinction decides the response:
a band-limited estimator asks for a wider/denser sweep or a band-consistent
tolerance, a non-causal solve asks for a solver fix.

Two error sources act on the estimator and they can be separated *without a
model of the physics*:

``discretisation``
    :func:`tensorbem.causality.kkr_added_mass` treats the sweep as a
    piecewise-linear interpolant and integrates it exactly.  A sharp resonance
    (heave here: 26 kg of damping on a 0.1 kg baseline within a 0.5 rad/s
    spacing) is badly represented by straight segments, and the principal-value
    integral of a badly represented peak is wrong *at the peak*.  This source
    disappears as the sampling of the same datum is refined.
``band``
    the integral runs over ``s in (0, inf)`` but the sweep stops at ``s_min`` and
    ``s_max``; the omitted tails are closed analytically (``A -> A_inf + c/s**2``
    above the band, constant below) so the residual contribution is a property of
    the *datum's* extent, not of its sampling.  This source only moves if the
    sweep itself is extended below ``s_min`` / above ``s_max`` -- i.e. it needs
    new solves, and :func:`estimator_floor` reports it as ``band_attributable``
    only when such an extended sweep is supplied.

The measurement here is deliberately model-free: the *same* tabulated datum is
re-expressed on a denser grid with a shape-preserving interpolant (PCHIP: monotone
where the data are monotone, no new extrema), and the estimator is re-run.  What
survives is band truncation; what disappears was discretisation.  Re-expression
adds no information, so the refined residual is not a "better answer" -- it is the
*floor* that the coarse sampling was adding.

A first attempt at this attribution used a least-squares exponential-sum surrogate
(28 rates) fitted to the sweep.  It over-fits a 13-node sweep exactly, so its
"floor" collapsed onto the measured residual by construction and the dense
extrapolation diverged -- recorded here because it is the obvious trap in this
kind of analysis: **a surrogate with more free parameters than data cannot bound
anything.**  The shape-preserving re-expression below has no free parameters.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tensorbem.causality import fit_added_mass_asymptote, kkr_added_mass

FloatArray = NDArray[np.float64]

__all__ = [
    "estimator_floor",
    "kk_residual",
    "refine_sweep",
]


def _validate(omega, added_mass, damping) -> tuple[FloatArray, FloatArray, FloatArray]:
    grid = np.asarray(omega, dtype=float).reshape(-1)
    a = np.asarray(added_mass, dtype=float).reshape(-1)
    b = np.asarray(damping, dtype=float).reshape(-1)
    if not (grid.size == a.size == b.size):
        raise ValueError("omega, added_mass and damping must have the same length")
    if grid.size < 8:
        raise ValueError("at least 8 frequencies are required")
    if np.any(grid <= 0.0) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("omegas must be strictly positive and increasing")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("added_mass and damping must be finite")
    return grid, a, b


def refine_sweep(
    omega: FloatArray, added_mass: FloatArray, damping: FloatArray, *, points: int = 601
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Re-express a tabulated sweep on a denser grid with a shape-preserving map.

    PCHIP is used rather than a cubic spline on purpose: it cannot introduce an
    extremum the datum does not have, so a damping peak is reproduced as one peak
    of the same height at the same place.  The refinement *adds no information*;
    it only removes the piecewise-linear model from the estimator's input.
    """
    from scipy.interpolate import PchipInterpolator

    grid, a, b = _validate(omega, added_mass, damping)
    dense = np.linspace(float(grid[0]), float(grid[-1]), int(points))
    return (
        dense,
        PchipInterpolator(grid, a)(dense),
        PchipInterpolator(grid, b)(dense),
    )


def kk_residual(
    omega: FloatArray,
    added_mass: FloatArray,
    damping: FloatArray,
    *,
    added_mass_inf: float | None = None,
    scale: float | None = None,
) -> dict:
    """Max KK residual at the grid midpoints, normalised as the case's G3 is.

    The normalisation is ``max|A - A_inf|`` over the tabulated sweep, i.e. the
    same scale the anchor uses, so the numbers are directly comparable with the
    gate that failed.  Only interior midpoints are evaluated: the principal value
    needs the pole strictly inside an interval (see
    :mod:`tensorbem.causality`).
    """
    grid, a, b = _validate(omega, added_mass, damping)
    infinity = (float(fit_added_mass_asymptote(grid, a).added_mass_inf)
                if added_mass_inf is None else float(added_mass_inf))
    if scale is None:
        scale = max(float(np.max(np.abs(a - infinity))), 1e-30)
    mid = 0.5 * (grid[:-1] + grid[1:])
    measured = 0.5 * (a[:-1] + a[1:])
    predicted = np.array([
        kkr_added_mass(grid, b, float(w), added_mass_inf=infinity) for w in mid
    ])
    residual = np.abs(predicted - measured) / max(float(scale), 1e-30)
    return {
        "omega_mid": mid.tolist(),
        "residual": residual.tolist(),
        "max": float(np.max(residual)),
        "rms": float(np.sqrt(np.mean(residual**2))),
        "argmax_omega": float(mid[int(np.argmax(residual))]),
        "added_mass_inf": infinity,
        "scale": float(scale),
        "points": int(grid.size),
    }


def estimator_floor(
    omega,
    added_mass,
    damping,
    *,
    added_mass_inf: float | None = None,
    dense_points: int = 601,
    extended=None,
) -> dict:
    """Separate the estimator's discretisation error from band truncation.

    Parameters
    ----------
    omega, added_mass, damping
        One diagonal of one DOF, as tabulated by the solver.
    extended
        Optional ``(omega, added_mass, damping)`` of the *same* DOF on a wider
        band (more solves, same estimator).  When supplied, the difference
        between the two refined residuals is the band-attributable part -- the
        only way to *measure* band truncation rather than assume it.
    """
    grid, a, b = _validate(omega, added_mass, damping)
    infinity = (float(fit_added_mass_asymptote(grid, a).added_mass_inf)
                if added_mass_inf is None else float(added_mass_inf))
    scale = max(float(np.max(np.abs(a - infinity))), 1e-30)

    coarse = kk_residual(grid, a, b, added_mass_inf=infinity, scale=scale)
    dense_grid, dense_a, dense_b = refine_sweep(grid, a, b, points=dense_points)
    refined = kk_residual(dense_grid, dense_a, dense_b,
                          added_mass_inf=infinity, scale=scale)

    report = {
        "band": [float(grid[0]), float(grid[-1])],
        "points": int(grid.size),
        "added_mass_inf": infinity,
        "scale": scale,
        "residual_coarse": coarse["max"],
        "residual_refined": refined["max"],
        "residual_coarse_rms": coarse["rms"],
        "residual_refined_rms": refined["rms"],
        "argmax_omega_coarse": coarse["argmax_omega"],
        "argmax_omega_refined": refined["argmax_omega"],
        "dense_points": int(dense_points),
        "discretisation_share": (
            1.0 - refined["max"] / coarse["max"] if coarse["max"] > 0.0 else 0.0
        ),
        "band_attributable": None,
        "extended_band": None,
        "coarse_profile": {"omega_mid": coarse["omega_mid"],
                           "residual": coarse["residual"]},
        "refined_profile": {"omega_mid": refined["omega_mid"],
                            "residual": refined["residual"]},
    }
    if extended is not None:
        ext_grid, ext_a, ext_b = _validate(*extended)
        ext_coarse = kk_residual(ext_grid, ext_a, ext_b,
                                 added_mass_inf=infinity, scale=scale)
        ext_dense_grid, ext_dense_a, ext_dense_b = refine_sweep(
            ext_grid, ext_a, ext_b, points=dense_points)
        ext_refined = kk_residual(ext_dense_grid, ext_dense_a, ext_dense_b,
                                 added_mass_inf=infinity, scale=scale)
        report["extended_band"] = {
            "band": [float(ext_grid[0]), float(ext_grid[-1])],
            "points": int(ext_grid.size),
            "residual_coarse": ext_coarse["max"],
            "residual_refined": ext_refined["max"],
        }
        report["band_attributable"] = float(
            max(refined["max"] - ext_refined["max"], 0.0))
    return report