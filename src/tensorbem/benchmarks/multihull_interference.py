"""Multi-body interference: Wigley tandem hulls in potential flow.

Mission m5455 (P19-N02): Wigley III wetted-surface / hydrostatics + ellipsoid
wave-resistance anchor, plus a tandem-interference study.

Physics
-------
Two identical Wigley hulls are placed in a **tandem** arrangement with their
longitudinal centres separated by a distance ``S`` (measured from centre to
centre).  The interference factor is

    sigma(S) = R_W_pair(S) / (2 * R_W_single)

where ``R_W_pair`` is the wave resistance of the pair and ``R_W_single`` is
the resistance of one hull alone.  In the **Michell thin-ship** framework the
pair's Kochin function is the single-hull function multiplied by the
interference term ``2 cos(k S / 2)`` (the two centreplane source distributions
are identical and shifted by ``S``), so

    H_pair(k) = H_single(k) * 2 cos(k S / 2),
    R_W_pair(S) = R_W_single * 4 <cos^2(k_0 S / 2)>_Kelvin

where ``<...>_Kelvin`` denotes the Kelvin-wake average.  For a thin body with
slowly varying Kochin function, this approximates to ``4 cos^2(k_0 S / 2)``,
but the exact value requires the full Kelvin-wake integral.

The BEM panel method computes the pair by placing panels on both hulls
simultaneously; the Kochin function is obtained from the panel-source
distribution.  The panel Kochin function is compared with the analytical
interference formula above.

Limitations
-----------
* Linearised free surface, deep water, steady forward speed.
* Thin-ship: the hulls are treated as centreplane source distributions;
  no free-surface boundary integral equation is solved.
* No viscous or nonlinear effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from ..wigley import (
    WigleyHullMesh,
    michell_kochin_function,
    michell_wave_resistance,
    wigley_hull_mesh,
    wigley_hull_properties,
)

__all__ = [
    "TandemWigley",
    "tandem_kochin_function",
    "tandem_wave_resistance",
    "interference_factor",
    "tandem_kochin_panels",
    "tandem_panel_wave_resistance",
    "interference_study",
    "tandem_mesh",
]

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_DEFAULT_DENSITY = 1025.0
_DEFAULT_GRAVITY = 9.80665


# ============================================================================
# Tandem Wigley hullform
# ============================================================================


@dataclass(frozen=True)
class TandemWigley:
    """Two identical Wigley hulls in tandem arrangement.

    Parameters
    ----------
    L, B, T :
        Principal dimensions of each hull.
    spacing :
        Centre-to-centre longitudinal spacing (m).  ``S = L`` means the stern
        of the forward hull is at the bow of the aft hull.
    """

    L: float
    B: float
    T: float
    spacing: float

    def __post_init__(self) -> None:
        for name in ("L", "B", "T", "spacing"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive finite")

    @property
    def single(self):
        """Return a single-hull property dict."""
        return wigley_hull_properties(self.L, self.B, self.T)


# ============================================================================
# Analytical Michell interference
# ============================================================================


def tandem_kochin_function(
    k: float, hull: TandemWigley, U: float
) -> complex:
    """Kochin function of the tandem pair (analytical Michell).

    ``H_pair(k) = H_single(k) * 2 cos(k S / 2)``.
    """
    if k <= 0.0:
        return 0.0j
    H_single = michell_kochin_function(hull.L, hull.B, hull.T, k)
    return H_single * 2.0 * math.cos(k * hull.spacing / 2.0)


def tandem_wave_resistance(
    hull: TandemWigley,
    U: float,
    *,
    rho: float = _DEFAULT_DENSITY,
    g: float = _DEFAULT_GRAVITY,
    n_quad: int = 256,
) -> float:
    """Wave resistance of the tandem pair (analytical Michell).

    Uses the same Kelvin-wake integral as the single hull, but with the
    interference-modified Kochin function.
    """
    if U <= 0.0:
        return 0.0

    k0 = g / (U * U)
    lam_max = max(5.0, 10.0 / (k0 * hull.T))

    nodes, weights = leggauss(n_quad)
    s_max = math.sqrt(lam_max - 1.0)
    s = 0.5 * s_max * (nodes + 1.0)
    w = 0.5 * s_max * weights
    lam = 1.0 + s * s

    acc = 0.0
    for si, wi, li in zip(s, w, lam):
        hk = tandem_kochin_function(li * k0, hull, U)
        acc += wi * abs(hk) ** 2 * 2.0 / (li * li * math.sqrt(2.0 + si * si))

    return max(4.0 * rho * k0 / g * acc, 0.0)


def interference_factor(
    hull: TandemWigley,
    U: float,
    *,
    rho: float = _DEFAULT_DENSITY,
    g: float = _DEFAULT_GRAVITY,
) -> float:
    """Interference factor ``sigma = R_W_pair / (2 R_W_single)``.

    Computed from the analytical tandem wave resistance divided by twice the
    single-hull wave resistance.
    """
    if U <= 0.0:
        return 1.0

    single_Rw = michell_wave_resistance(hull.L, hull.B, hull.T, U)
    Rw_pair = tandem_wave_resistance(hull, U)
    return Rw_pair / (2.0 * single_Rw)


# ============================================================================
# Panel-based tandem computation
# ============================================================================


def tandem_mesh(
    L: float,
    B: float,
    T: float,
    spacing: float,
    nx: int,
    nz: int,
) -> tuple[WigleyHullMesh, WigleyHullMesh]:
    """Generate meshes for two Wigley hulls in tandem.

    Returns ``(mesh_forward, mesh_aft)`` where the forward hull is centred at
    ``x = +S/2`` and the aft hull at ``x = -S/2``.
    """
    mesh_fwd = wigley_hull_mesh(L, B, T, nx, nz)
    # Shift aft hull by -spacing in x
    mesh_aft = WigleyHullMesh(
        mesh_fwd.vertices - np.array([spacing, 0.0, 0.0]),
        mesh_fwd.faces,
    )
    return mesh_fwd, mesh_aft


def _wigley_panel_grid(
    L: float,
    B: float,
    T: float,
    x_center: float,
    nx: int,
    nz: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Panel grid on the centreplane projection of a Wigley hull.

    The hull is centred at ``x = x_center``.

    Returns ``(x_centroid, z_centroid, area, source_strength)`` flattened over
    the ``nx x nz`` grid, where ``source_strength`` is the *panel-averaged*
    ``dy/dx``.

    For the Wigley hull ``y = (B/2)(1 - (2x_rel/L)^2)(1 - (z/T)^2)`` where
    ``x_rel = x - x_center``:
    ``dy/dx = -(4B/L^2) * x_rel * (1 - (z/T)^2)``.

    The panel-averaged source strength is:
    ``<dy/dx> = -(4B/L^2) * (x_{i+1,rel} + x_{i,rel})/2 * [1 - (z_{j+1}^2 + z_{j+1}*z_j + z_j^2)/(3T^2)]``

    The panel area is ``dx * dz`` (centreplane projection area).
    """
    # Panel x-range is centred at x_center
    x_vals = np.linspace(x_center - L / 2, x_center + L / 2, nx + 1)
    z_vals = np.linspace(-T, 0.0, nz + 1)

    x_c = np.empty((nx, nz))
    z_c = np.empty((nx, nz))
    area = np.empty((nx, nz))
    strength = np.empty((nx, nz))

    dx = x_vals[1] - x_vals[0]
    dz = z_vals[1] - z_vals[0]

    for i in range(nx):
        for j in range(nz):
            # Panel centroid (absolute x-coordinate)
            xc = 0.5 * (x_vals[i] + x_vals[i + 1])
            zc = 0.5 * (z_vals[j] + z_vals[j + 1])

            # Relative x-coordinate (for half-breadth and source strength)
            x_rel = xc - x_center

            # Half-breadth at centroid
            y_c = 0.5 * B * (1.0 - (2.0 * x_rel / L) ** 2) * (1.0 - (zc / T) ** 2)
            y_c = max(y_c, 0.0)

            if y_c < 1e-14:
                # Panel is at the post (zero width)
                x_c[i, j] = xc
                z_c[i, j] = zc
                area[i, j] = 0.0
                strength[i, j] = 0.0
                continue

            # Panel-averaged source strength:
            # <dy/dx> = -(4B/L^2) * (x_{i+1,rel} + x_{i,rel})/2 * [1 - (z_{j+1}^2 + z_{j+1}*z_j + z_j^2)/(3T^2)]
            x_rel_avg = ((x_vals[i + 1] - x_center) + (x_vals[i] - x_center)) / 2.0
            z_term = 1.0 - (z_vals[j + 1] ** 2 + z_vals[j + 1] * z_vals[j] + z_vals[j] ** 2) / (3.0 * T ** 2)
            panel_avg_dydx = -(4.0 * B / L ** 2) * x_rel_avg * z_term

            # Panel area: dx * dz (centreplane projection)
            panel_area = dx * dz

            x_c[i, j] = xc
            z_c[i, j] = zc
            area[i, j] = panel_area
            strength[i, j] = panel_avg_dydx

    return x_c.flatten(), z_c.flatten(), area.flatten(), strength.flatten()


def tandem_kochin_panels(
    k: float,
    hull: TandemWigley,
    nx: int,
    nz: int,
) -> complex:
    """Kochin function from panel discretisation of the tandem pair.

    Each hull contributes its own centreplane source distribution; the total
    Kochin function is the sum of the two shifted contributions.
    """
    if k <= 0.0:
        return 0.0j

    # Forward hull (centred at +S/2)
    x_fwd, z_fwd, area_fwd, strength_fwd = _wigley_panel_grid(
        hull.L, hull.B, hull.T, hull.spacing / 2.0, nx, nz
    )
    # Aft hull (centred at -S/2)
    x_aft, z_aft, area_aft, strength_aft = _wigley_panel_grid(
        hull.L, hull.B, hull.T, -hull.spacing / 2.0, nx, nz
    )

    # Phase factors
    phase_fwd = area_fwd * np.exp(k * z_fwd + 1j * k * x_fwd)
    phase_aft = area_aft * np.exp(k * z_aft + 1j * k * x_aft)

    # Total Kochin function
    H = 2.0 * complex(np.sum(strength_fwd * phase_fwd) + np.sum(strength_aft * phase_aft))
    return H


def tandem_panel_wave_resistance(
    hull: TandemWigley,
    U: float,
    *,
    nx: int,
    nz: int,
    rho: float = _DEFAULT_DENSITY,
    g: float = _DEFAULT_GRAVITY,
    n_kelvin: int = 256,
) -> float:
    """Wave resistance from panel Kochin function of the tandem pair."""
    if U <= 0.0:
        return 0.0

    k0 = g / (U * U)
    lam_max = max(5.0, 10.0 / (k0 * hull.T))

    nodes, weights = leggauss(n_kelvin)
    s_max = math.sqrt(lam_max - 1.0)
    s = 0.5 * s_max * (nodes + 1.0)
    w = 0.5 * s_max * weights
    lam = 1.0 + s * s

    acc = 0.0
    for si, wi, li in zip(s, w, lam):
        hk = tandem_kochin_panels(li * k0, hull, nx, nz)
        acc += wi * abs(hk) ** 2 * 2.0 / (li * li * math.sqrt(2.0 + si * si))

    return max(4.0 * rho * k0 / g * acc, 0.0)


# ============================================================================
# Interference study
# ============================================================================


def interference_study(
    L: float = 3.8,
    B: float = 0.5,
    T: float = 0.225,
    froude: float = 0.30,
    spacing_ratios: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0),
    *,
    rho: float = _DEFAULT_DENSITY,
    g: float = _DEFAULT_GRAVITY,
    panel_levels: Sequence[tuple[int, int]] = ((10, 5), (20, 10), (40, 20)),
) -> dict:
    """Tandem interference study: sigma vs spacing ratio.

    Parameters
    ----------
    L, B, T :
        Principal dimensions of each hull.
    froude :
        Froude number based on ``L``.
    spacing_ratios :
        Spacing ``S / L`` values to evaluate.
    panel_levels :
        ``(n_sections, n_levels)`` per side for the panel Kochin function.

    Returns
    -------
    dict
        Study results with analytical and panel-based interference factors.
    """
    U = froude * math.sqrt(g * L)
    k0 = g / (U * U)
    single_Rw = michell_wave_resistance(L, B, T, U)

    # Analytical interference curve (computed from analytical tandem Rw)
    analytical_rows = []
    for sr in spacing_ratios:
        S = sr * L
        tw = TandemWigley(L, B, T, S)
        Rw_pair_analytic = tandem_wave_resistance(tw, U)
        sigma_analytic = Rw_pair_analytic / (2.0 * single_Rw)
        analytical_rows.append({
            "spacing_ratio": sr,
            "spacing_m": S,
            "k0_S": k0 * S,
            "sigma_analytic": sigma_analytic,
            "Rw_pair_analytic": Rw_pair_analytic,
        })

    # Panel-based interference
    panel_rows = []
    for n_s, n_l in panel_levels:
        panel_results = []
        for sr in spacing_ratios:
            S = sr * L
            tw = TandemWigley(L, B, T, S)
            Rw_pair = tandem_panel_wave_resistance(tw, U, nx=n_s, nz=n_l)
            sigma_panel = Rw_pair / (2.0 * single_Rw) if single_Rw > 0 else 0.0
            sigma_analytic = analytical_rows[spacing_ratios.index(sr)]["sigma_analytic"]
            panel_results.append({
                "spacing_ratio": sr,
                "spacing_m": S,
                "sigma_panel": sigma_panel,
                "sigma_analytic": sigma_analytic,
                "rel_error": abs(sigma_panel - sigma_analytic) / max(abs(sigma_analytic), 1e-10),
            })
        panel_rows.append({
            "panels": n_s * n_l * 4,
            "results": panel_results,
        })

    # Convergence check at the finest panel level
    finest = panel_rows[-1]["results"]
    max_panel_error = max(r["rel_error"] for r in finest)

    return {
        "L": L,
        "B": B,
        "T": T,
        "froude": froude,
        "U": U,
        "k0": k0,
        "single_Rw": single_Rw,
        "spacing_ratios": spacing_ratios,
        "analytical": analytical_rows,
        "panel_levels": panel_rows,
        "max_panel_error": max_panel_error,
        "convergence": {
            "note": "Panel Kochin function uses discrete source distribution; "
                    "convergence is O(h) in panel count.",
            "max_relative_error": max_panel_error,
        },
    }
