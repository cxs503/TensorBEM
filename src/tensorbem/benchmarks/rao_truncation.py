"""Truncation-convergence helpers for frequency-domain RAO studies.

Why this module exists
----------------------
A frequency-domain RAO study is only as good as the *band* it is evaluated on.
Two truncations are always present and are almost never quantified:

* **truncation frequency** ``omega_max`` -- the upper end of the sweep.  In the
  ITTC recommended procedure for motions in waves the response spectrum is
  integrated over a finite band, and the residual (truncated) spectrum area is
  the acceptance quantity, not the pointwise RAO.
* **truncation radius** ``R_lambda = lambda_max / L`` -- how far into the long
  wave region the sweep reaches.  ``R_lambda`` is a *length* (a radius in
  wavelength space); in a panel method with an explicitly truncated fluid domain
  its geometric counterpart is the outer radius of the discretised patch.

The functions here are deliberately free of any reference/experimental number:
they compute the truncated and the full-band statistics of the *same* discrete
RAO curve, so the truncation error is a self-referential quantity that cannot be
faked by quoting somebody else's table.

The sea state is the ITTC (1978) two-parameter spectrum

    S(omega) = (A / omega**5) * exp(-B / omega**4),
    A = 173 * Hs**2 / T1**4,   B = 691 / T1**4,

whose zeroth moment is the analytic ``m0 = A / (4B) = Hs**2 / 16`` — that
identity is asserted in the tests, so the spectrum implementation is anchored to
closed form rather than to a digitised table.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

GRAVITY = 9.80665

#: ITTC (1978) two-parameter spectrum coefficients (closed-form definitions).
ITTC_A_COEFFICIENT = 173.0
ITTC_B_COEFFICIENT = 691.0


# ---------------------------------------------------------------------------
# spectrum
# ---------------------------------------------------------------------------

def ittc_two_parameter_spectrum(
    omega: FloatArray | Sequence[float],
    hs: float,
    t1: float,
    gravity: float = GRAVITY,
) -> FloatArray:
    """ITTC (1978) two-parameter wave spectrum ``S(omega)`` in m^2 s.

    ``hs`` is the significant wave height [m], ``t1`` the mean period [s].
    ``gravity`` is accepted for API symmetry with the dispersion relation used
    elsewhere in the package; the ITTC coefficients are already dimensional.
    """
    om = np.atleast_1d(np.asarray(omega, dtype=float))
    if not np.isfinite(om).all() or np.any(om <= 0.0):
        raise ValueError("omega must be finite and strictly positive")
    hs = float(hs)
    t1 = float(t1)
    if not math.isfinite(hs) or hs <= 0.0:
        raise ValueError("hs must be positive and finite")
    if not math.isfinite(t1) or t1 <= 0.0:
        raise ValueError("t1 must be positive and finite")
    if not math.isfinite(float(gravity)) or float(gravity) <= 0.0:
        raise ValueError("gravity must be positive and finite")
    a = ITTC_A_COEFFICIENT * hs**2 / t1**4
    b = ITTC_B_COEFFICIENT / t1**4
    return (a / om**5) * np.exp(-b / om**4)


def ittc_spectral_peak_omega(t1: float) -> float:
    """Peak circular frequency of the ITTC two-parameter spectrum [rad/s].

    ``dS/domega = 0`` gives ``omega_p = (4B/5)**(1/4)``.
    """
    t1 = float(t1)
    if not math.isfinite(t1) or t1 <= 0.0:
        raise ValueError("t1 must be positive and finite")
    return (4.0 * ITTC_B_COEFFICIENT / 5.0) ** 0.25 / t1


# ---------------------------------------------------------------------------
# dispersion helpers
# ---------------------------------------------------------------------------

def omega_from_wavelength(wavelength: float, gravity: float = GRAVITY) -> float:
    """Deep-water circular frequency of a given wavelength [rad/s]."""
    lam = float(wavelength)
    if not math.isfinite(lam) or lam <= 0.0:
        raise ValueError("wavelength must be positive and finite")
    return math.sqrt(2.0 * math.pi * float(gravity) / lam)


def truncation_radius_omega_min(
    radius_over_length: float, length: float, gravity: float = GRAVITY
) -> float:
    """Lower band edge ``omega_min`` for a long-wave truncation radius.

    ``radius_over_length`` is ``R_lambda = lambda_max / L``; the returned value is
    the deep-water circular frequency of ``lambda_max = R_lambda * L``.
    """
    r = float(radius_over_length)
    length = float(length)
    if not math.isfinite(r) or r <= 0.0:
        raise ValueError("radius_over_length must be positive and finite")
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("length must be positive and finite")
    return omega_from_wavelength(r * length, gravity=gravity)


# ---------------------------------------------------------------------------
# peak of a discrete curve
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Peak:
    """Peak of a sampled curve with a parabolic (3-point) refinement."""

    omega: float
    value: float
    index: int
    refined: bool


def curve_peak(omega: Sequence[float], values: Sequence[float]) -> Peak:
    """Locate the maximum of ``values`` on the ``omega`` grid.

    A 3-point parabola through the maximum and its two neighbours refines the
    abscissa (and the ordinate) between grid nodes; the refinement is *not*
    applied at the array ends, where no parabola can be fitted.
    """
    om = np.asarray(omega, dtype=float)
    val = np.asarray(values, dtype=float)
    if om.ndim != 1 or val.ndim != 1 or om.size != val.size or om.size < 2:
        raise ValueError("omega and values must be 1-D of equal length >= 2")
    if not np.isfinite(om).all() or not np.isfinite(val).all():
        raise ValueError("omega and values must be finite")
    order = np.argsort(om)
    om = om[order]
    val = val[order]
    i = int(np.argmax(val))
    if i == 0 or i == om.size - 1:
        return Peak(omega=float(om[i]), value=float(val[i]), index=i, refined=False)
    x0, x1, x2 = om[i - 1], om[i], om[i + 1]
    y0, y1, y2 = val[i - 1], val[i], val[i + 1]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if denom == 0.0:
        return Peak(omega=float(x1), value=float(y1), index=i, refined=False)
    c2 = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    c1 = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom
    if c2 >= 0.0:  # not a concave peak between the nodes; keep the grid maximum
        return Peak(omega=float(x1), value=float(y1), index=i, refined=False)
    x_star = -c1 / (2.0 * c2)
    if not (x0 <= x_star <= x2):
        return Peak(omega=float(x1), value=float(y1), index=i, refined=False)
    c0 = y1 - c1 * x1 - c2 * x1**2
    y_star = c0 + c1 * x_star + c2 * x_star**2
    return Peak(omega=float(x_star), value=float(y_star), index=i, refined=True)


# ---------------------------------------------------------------------------
# truncated response statistics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BandStatistics:
    """Response statistics of one discrete RAO curve over one band."""

    omega_min: float
    omega_max: float
    m0_full: float
    m0_band: float
    area_fraction: float
    sigma_full: float
    sigma_band: float
    peak_omega: float
    peak_value: float
    peak_refined: bool
    peak_inside_band: bool


def band_statistics(
    omega: Sequence[float],
    rao_amplitude: Sequence[float],
    spectrum: Sequence[float],
    omega_min: float = 0.0,
    omega_max: float = math.inf,
) -> BandStatistics:
    """Truncated vs full-band RAO statistics.

    ``m0_full`` integrates ``S(omega) * |RAO(omega)|**2`` over the whole supplied
    grid (trapezoidal); ``m0_band`` restricts the same integrand to
    ``omega_min <= omega <= omega_max``.  ``area_fraction`` is the share of the
    response variance (not of the wave variance) inside the band, which is the
    quantity the truncation convergence of a RAO-derived statistic rides on.
    """
    om = np.asarray(omega, dtype=float)
    rao = np.asarray(rao_amplitude, dtype=float)
    spec = np.asarray(spectrum, dtype=float)
    if om.ndim != 1 or rao.ndim != 1 or spec.ndim != 1:
        raise ValueError("omega, rao_amplitude and spectrum must be 1-D")
    if not (om.size == rao.size == spec.size) or om.size < 3:
        raise ValueError("omega, rao_amplitude and spectrum must share one length >= 3")
    if not np.isfinite(om).all() or not np.isfinite(rao).all() or not np.isfinite(spec).all():
        raise ValueError("inputs must be finite")
    order = np.argsort(om)
    om, rao, spec = om[order], rao[order], spec[order]

    integrand = spec * rao**2
    m0_full = float(np.trapezoid(integrand, om))
    # Inclusive band edges with a rounding slack: a level that was inserted into
    # the grid as a decimal literal must not be dropped by a 1-ulp difference.
    lo = float(omega_min)
    hi = float(omega_max)
    slack = 1.0e-9 * max(1.0, abs(lo), abs(hi))
    mask = (om >= lo - slack) & (om <= hi + slack)
    if int(mask.sum()) < 2:
        raise ValueError("band contains fewer than two grid points")
    m0_band = float(np.trapezoid(integrand[mask], om[mask]))
    area_fraction = m0_band / m0_full if m0_full > 0.0 else float("nan")
    peak = curve_peak(om, rao)
    return BandStatistics(
        omega_min=float(omega_min),
        omega_max=float(omega_max),
        m0_full=m0_full,
        m0_band=m0_band,
        area_fraction=area_fraction,
        sigma_full=math.sqrt(m0_full),
        sigma_band=math.sqrt(max(m0_band, 0.0)),
        peak_omega=peak.omega,
        peak_value=peak.value,
        peak_refined=peak.refined,
        peak_inside_band=bool(float(omega_min) <= peak.omega <= float(omega_max)),
    )


def relative_change_pct(a: float, b: float) -> float:
    """`` |a - b| / max(|a|, |b|) `` in percent; ``nan`` if both vanish."""
    a = float(a)
    b = float(b)
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return float("nan")
    return 100.0 * abs(a - b) / scale


def doubling_pairs(levels: Sequence[float]) -> list[tuple[float, float]]:
    """Pairs ``(lo, 2*lo)`` present in ``levels`` (exact factor-2 tolerance 1e-9)."""
    vals = sorted(float(v) for v in levels)
    out: list[tuple[float, float]] = []
    for lo in vals:
        for hi in vals:
            if lo <= 0.0:
                continue
            if abs(hi / lo - 2.0) <= 1e-9:
                out.append((lo, hi))
    return out


# ---------------------------------------------------------------------------
# hull inertia of the Wigley III form (uniform density)
# ---------------------------------------------------------------------------

def wigley3_body_inertia(
    L: float,
    B: float,
    T: float,
    alpha: float,
    density: float,
    n_x: int = 801,
    n_z: int = 401,
) -> dict[str, float]:
    """Mass moments of inertia of the Wigley III lower hull at uniform density.

    The hull is fore-aft symmetric, so the centroid sits at ``(0, 0, z_B)`` and
    the products of inertia vanish.  With the analytic half breadth
    ``b(x, z) = (B/2)(1 - xi^2)(1 - zeta^2)(1 + alpha zeta^2)``,
    ``xi = 2x/L``, ``zeta = z/T`` the volume integrals

        V = 2 * int b dx dz,   z_B = (2/V) * int z b dx dz,
        I_xx = rho * int 2 (y^2 + (z - z_B)^2) b dx dz,
        I_yy = rho * int 2 (x^2 + (z - z_B)^2) b dx dz,

    are evaluated by trapezoidal quadrature on a uniform ``n_x x n_z`` grid; the
    volume is returned as well so the quadrature can be checked against the
    closed form ``V = L B T (4/9 + 4 alpha/45)``.
    """
    if n_x < 9 or n_z < 9:
        raise ValueError("n_x and n_z must be >= 9")
    if density <= 0.0:
        raise ValueError("density must be positive")
    x = np.linspace(-0.5 * L, 0.5 * L, int(n_x))
    z = np.linspace(-T, 0.0, int(n_z))
    X, Z = np.meshgrid(x, z, indexing="ij")
    xi = 2.0 * X / L
    zeta = Z / T
    b = 0.5 * B * (1.0 - xi**2) * (1.0 - zeta**2) * (1.0 + alpha * zeta**2)

    def integrate(field: FloatArray) -> float:
        return float(np.trapezoid(np.trapezoid(field, z, axis=1), x, axis=0))

    volume = 2.0 * integrate(b)
    z_b = (2.0 / volume) * integrate(Z * b) if volume > 0.0 else 0.0
    dz2 = (Z - z_b) ** 2
    # y^2 integrated over the width: int_{-b}^{b} y^2 dy = 2 b^3 / 3
    i_xx = density * integrate(2.0 * b**3 / 3.0 + 2.0 * dz2 * b)
    i_yy = density * integrate(2.0 * X**2 * b + 2.0 * dz2 * b)
    i_zz = density * integrate(2.0 * X**2 * b + 2.0 * b**3 / 3.0)
    return {
        "volume": volume,
        "z_centroid": z_b,
        "Ixx": i_xx,
        "Iyy": i_yy,
        "Izz": i_zz,
    }


def gt_style_mesh_waterplane(mesh: object) -> float:
    """Convenience accessor: waterplane area of a :class:`WettedHullMesh`."""
    area = float(getattr(mesh, "waterplane_area"))
    return area


def block_coefficient(volume: float, L: float, B: float, T: float) -> float:
    """``Cb = V / (L B T)``."""
    denom = float(L) * float(B) * float(T)
    if denom <= 0.0:
        raise ValueError("L, B and T must be positive")
    return float(volume) / denom


def relative_deviation_pct(value: float, reference: float) -> float:
    """``100 * |value - reference| / |reference|`` (``nan`` for a zero reference)."""
    ref = float(reference)
    if ref == 0.0:
        return float("nan")
    return 100.0 * abs(float(value) - ref) / abs(ref)


def summarise_levels(
    levels: Iterable[float], values: Iterable[float]
) -> list[dict[str, float]]:
    """Tabulate ``levels`` against ``values`` with consecutive relative changes.

    Convenience for report tables: returns one dict per level with
    ``rel_change_to_next_pct`` and ``rel_change_to_last_pct``.
    """
    lv = [float(v) for v in levels]
    va = [float(v) for v in values]
    if len(lv) != len(va):
        raise ValueError("levels and values must have equal length")
    last = va[-1] if va else float("nan")
    rows: list[dict[str, float]] = []
    for i, (l, v) in enumerate(zip(lv, va)):
        nxt = relative_change_pct(v, va[i + 1]) if i + 1 < len(va) else float("nan")
        rows.append(
            {
                "level": l,
                "value": v,
                "rel_change_to_next_pct": nxt,
                "rel_change_to_last_pct": relative_change_pct(v, last),
            }
        )
    return rows