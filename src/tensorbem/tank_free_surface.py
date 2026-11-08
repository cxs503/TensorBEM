"""Slack-tank (liquid free-surface) correction to the hydrostatic restoring matrix.

Why this module exists
----------------------
:func:`tensorbem.hydrostatic.compute_hydrostatic_restoring_matrix` builds the 6x6
restoring matrix ``C`` from the *ship's* waterplane only.  A ship carrying liquid
in partially filled tanks does not behave that way: the liquid keeps its free
surface horizontal when the ship heels or trims, so its centre of gravity slides
in the body frame and the righting moment drops.  That is the classical *free
surface effect* (``free surface correction``, FSC) of naval architecture, and
without it the roll stiffness of a ship with slack tanks is over-predicted --
the error is first order in the tank free-surface area, not a small correction
(see ``docs/tank_free_surface_correction.md``, where two wing tanks remove
7 % of ``GM_T`` and lengthen the roll period by 4 %).

Physics and convention
----------------------
Let the ship rotate by the small angles ``(phi, theta)`` about ``+x`` (roll) and
``+y`` (pitch).  A body-fixed point undergoes the vertical displacement

    w = y*phi - x*theta            (the convention used in ``hydrostatic.py``)

so the liquid free surface, which stays horizontal in the inertial frame, sits
at ``z = H - y*phi + x*theta`` in the body frame.  The liquid volume is fixed
(``H`` is chosen so), and its centre of gravity -- measured in the inertial frame
-- is

    Z_cg = z_cg + phi*y_cg - theta*x_cg .

Integrating ``z_cg`` (mean of ``H - y phi + x theta``) and ``y_cg, x_cg`` (weighted
means) over the free-surface footprint ``A`` about its own centroid gives

    Z_cg - H = (1/V_l) * [ -1/2 phi^2 I_xx + phi theta I_xy - 1/2 theta^2 I_yy ]

with ``V_l`` the liquid volume and ``I_xx = int y^2 dA``, ``I_xy = int x y dA``,
``I_yy = int x^2 dA`` taken **about the free-surface centroid** (so the mean level
drops out -- which is why the correction is independent of where the tank sits).
The ship's potential energy therefore loses

    dC[3,3] = -rho_l g I_xx,   dC[4,4] = -rho_l g I_yy,   dC[3,4] = +rho_l g I_xy

and, dividing by ``rho g nabla``, the metacentric heights drop by

    dGM_T = sum(rho_l I_xx) / (rho nabla),   dGM_L = sum(rho_l I_yy) / (rho nabla),

which is the textbook free surface correction (Comstock, *Principles of Naval
Architecture*; IMO IS Code 2008, intact-stability free-surface correction).

Two facts follow from the centroidal moments and are used as checks:

* the correction is **invariant under translation** of the tank footprint;
* a slack tank adds **no heave restoring** and no first moment (an off-centre tank
  produces a static heel moment at equilibrium, which is a trim/heel problem, not
  a restoring entry).

Verification
------------
``tensorbem.benchmarks.tank_free_surface_verification`` compares three
independent evaluations of the free-surface moments -- the closed form for a
rectangle, exact Green's-theorem integration over the polygon, and midpoint
quadrature on a refining grid -- and checks the sign and magnitude of the energy
Hessian against a direct finite-difference differentiation of the liquid
potential energy obtained by quadrature (``docs/tank_free_surface_correction.md``).
``tests/test_tank_free_surface.py`` pins those checks.

Conventions match the rest of the package: metres, kilograms, radians,
``(surge, sway, heave, roll, pitch, yaw)`` ordering, seawater
``rho = 1025 kg/m^3``, ``g = 9.80665 m/s^2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

DEFAULT_DENSITY = 1025.0
DEFAULT_GRAVITY = 9.80665

__all__ = [
    "DEFAULT_DENSITY",
    "DEFAULT_GRAVITY",
    "PolygonMoments",
    "TankFreeSurface",
    "FreeSurfaceCorrection",
    "polygon_moments",
    "rectangular_free_surface",
    "free_surface_correction",
    "apply_free_surface_correction",
]


@dataclass(frozen=True)
class PolygonMoments:
    """Area, centroid and **centroidal** second moments of a planar polygon."""

    area: float
    centroid: FloatArray
    i_xx: float
    i_yy: float
    i_xy: float


def _triangle_raw_moments(
    a: FloatArray, b: FloatArray, c: FloatArray
) -> tuple[float, float, float, float, float, float]:
    """Signed ``(A, S_x, S_y, I_xx, I_yy, I_xy)`` of the triangle ``a b c``."""
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    area = 0.5 * cross
    xa, ya = float(a[0]), float(a[1])
    xb, yb = float(b[0]), float(b[1])
    xc, yc = float(c[0]), float(c[1])
    s_x = area * (xa + xb + xc) / 3.0
    s_y = area * (ya + yb + yc) / 3.0
    i_xx = area / 6.0 * (ya * ya + yb * yb + yc * yc + ya * yb + yb * yc + yc * ya)
    i_yy = area / 6.0 * (xa * xa + xb * xb + xc * xc + xa * xb + xb * xc + xc * xa)
    i_xy = area / 12.0 * (
        2.0 * (xa * ya + xb * yb + xc * yc)
        + xa * yb + xb * ya + xa * yc + xc * ya + xb * yc + xc * yb
    )
    return area, s_x, s_y, i_xx, i_yy, i_xy


def _polygon_raw_moments(vertices: FloatArray) -> tuple[float, float, float, float, float, float]:
    """``(A, S_x, S_y, I_xx, I_yy, I_xy)`` about the origin, exact for a polygon.

    Closed-form triangle moments summed over the fan ``(v0, v_i, v_{i+1})``. The
    triangle areas are *signed*, so the fan is exact for non-convex polygons as
    well: the parts that leave the polygon cancel pairwise.
    """
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2 or v.shape[0] < 3:
        raise ValueError("polygon vertices must be an (n>=3, 2) array")
    area = s_x = s_y = i_xx = i_yy = i_xy = 0.0
    for k in range(1, v.shape[0] - 1):
        t = _triangle_raw_moments(v[0], v[k], v[k + 1])
        area += t[0]
        s_x += t[1]
        s_y += t[2]
        i_xx += t[3]
        i_yy += t[4]
        i_xy += t[5]
    return area, s_x, s_y, i_xx, i_yy, i_xy


def polygon_moments(vertices: FloatArray) -> PolygonMoments:
    """Area, centroid and centroidal second moments of a simple polygon.

    The polygon may be given clockwise or counter-clockwise (orientations are
    normalised to counter-clockwise); the reported ``i_xx, i_yy, i_xy`` are about
    the axes through the centroid, i.e. the quantities the free-surface
    correction consumes. Exact up to floating-point rounding -- no quadrature.
    """
    area, s_x, s_y, i_xx_o, i_yy_o, i_xy_o = _polygon_raw_moments(vertices)
    if area < 0.0:  # clockwise input: flip the orientation and recompute
        return polygon_moments(np.asarray(vertices, dtype=float)[::-1])
    if area == 0.0:
        raise ValueError("polygon must enclose a positive area")
    x_c = s_x / area
    y_c = s_y / area
    return PolygonMoments(
        area=area,
        centroid=np.array([x_c, y_c]),
        i_xx=i_xx_o - area * y_c * y_c,
        i_yy=i_yy_o - area * x_c * x_c,
        i_xy=i_xy_o - area * x_c * y_c,
    )


@dataclass(frozen=True)
class TankFreeSurface:
    """One free surface: area, centroid, centroidal second moments, density."""

    area: float
    centroid: FloatArray
    i_xx: float
    i_yy: float
    i_xy: float
    liquid_density: float = DEFAULT_DENSITY
    name: str = ""

    @classmethod
    def from_polygon(
        cls,
        vertices: FloatArray,
        liquid_density: float = DEFAULT_DENSITY,
        name: str = "",
    ) -> "TankFreeSurface":
        m = polygon_moments(vertices)
        return cls(
            area=m.area,
            centroid=m.centroid,
            i_xx=m.i_xx,
            i_yy=m.i_yy,
            i_xy=m.i_xy,
            liquid_density=float(liquid_density),
            name=name,
        )

    @property
    def free_surface_moment_transverse(self) -> float:
        """``rho_l * i_xx``: the moment that reduces roll restoring (kg m^2)."""
        return self.liquid_density * self.i_xx

    @property
    def free_surface_moment_longitudinal(self) -> float:
        """``rho_l * i_yy``: the moment that reduces pitch restoring (kg m^2)."""
        return self.liquid_density * self.i_yy


def rectangular_free_surface(
    length: float,
    breadth: float,
    centroid: Sequence[float] = (0.0, 0.0),
    liquid_density: float = DEFAULT_DENSITY,
    name: str = "",
) -> TankFreeSurface:
    """Closed-form free surface of a rectangular tank (length along ``x``)."""
    if length <= 0.0 or breadth <= 0.0:
        raise ValueError("length and breadth must be positive")
    x_c, y_c = float(centroid[0]), float(centroid[1])
    return TankFreeSurface(
        area=length * breadth,
        centroid=np.array([x_c, y_c]),
        i_xx=length * breadth**3 / 12.0,
        i_yy=breadth * length**3 / 12.0,
        i_xy=0.0,
        liquid_density=float(liquid_density),
        name=name,
    )


@dataclass(frozen=True)
class FreeSurfaceCorrection:
    """Aggregate slack-tank correction for a floating body."""

    tanks: tuple[TankFreeSurface, ...]
    displaced_volume: float
    water_density: float
    gravity: float
    delta_gm_transverse: float
    delta_gm_longitudinal: float
    restoring_correction: FloatArray

    @property
    def n_tanks(self) -> int:
        return len(self.tanks)

    def summary(self) -> dict[str, float]:
        return {
            "n_tanks": float(self.n_tanks),
            "delta_gm_transverse": float(self.delta_gm_transverse),
            "delta_gm_longitudinal": float(self.delta_gm_longitudinal),
            "restoring_correction_roll": float(self.restoring_correction[3, 3]),
            "restoring_correction_pitch": float(self.restoring_correction[4, 4]),
            "restoring_correction_roll_pitch": float(self.restoring_correction[3, 4]),
        }


def free_surface_correction(
    tanks: Iterable[TankFreeSurface],
    displaced_volume: float,
    water_density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> FreeSurfaceCorrection:
    """6x6 restoring correction and metacentric-height reduction from slack tanks.

    Parameters
    ----------
    tanks :
        Free surfaces of the partially filled tanks (``TankFreeSurface`` or
        polygons via :meth:`TankFreeSurface.from_polygon`).
    displaced_volume :
        Ship displaced volume ``nabla`` (m^3, at the loading condition). Must be
        positive: an unbalanced ``m != rho nabla`` mass is a different problem.
    water_density, gravity :
        Fluid density (kg/m^3) and gravity (m/s^2) of the surrounding water.

    Returns
    -------
    correction :
        Aggregated correction, including the 6x6 matrix to be **added** to
        ``compute_hydrostatic_restoring_matrix(...).restoring_matrix``.

    Notes
    -----
    The matrix is symmetric and diagonal-negative on the roll/pitch block:
    a free surface can only remove restoring, never add it. ``C[3,4] = C[4,3]``
    is non-zero only for a footprint with ``i_xy != 0`` (an L-shaped tank,
    or a tank whose free surface is not symmetric about its own axes).
    """
    if displaced_volume <= 0.0:
        raise ValueError("displaced_volume must be positive")
    if water_density <= 0.0:
        raise ValueError("water_density must be positive")

    tank_list = tuple(tanks)
    delta_c = np.zeros((6, 6), dtype=float)
    total_i_x = 0.0
    total_i_y = 0.0
    for tank in tank_list:
        if tank.area <= 0.0:
            raise ValueError("tank free-surface area must be positive")
        w = tank.liquid_density * gravity
        delta_c[3, 3] -= w * tank.i_xx
        delta_c[3, 4] += w * tank.i_xy
        delta_c[4, 3] += w * tank.i_xy
        delta_c[4, 4] -= w * tank.i_yy
        total_i_x += tank.liquid_density * tank.i_xx
        total_i_y += tank.liquid_density * tank.i_yy

    return FreeSurfaceCorrection(
        tanks=tank_list,
        displaced_volume=float(displaced_volume),
        water_density=float(water_density),
        gravity=float(gravity),
        delta_gm_transverse=total_i_x / (water_density * displaced_volume),
        delta_gm_longitudinal=total_i_y / (water_density * displaced_volume),
        restoring_correction=delta_c,
    )


def apply_free_surface_correction(
    restoring_matrix: FloatArray,
    correction: FreeSurfaceCorrection,
) -> FloatArray:
    """Return a new 6x6 restoring matrix with the slack-tank correction added."""
    c = np.asarray(restoring_matrix, dtype=float)
    if c.shape != (6, 6):
        raise ValueError("restoring_matrix must be (6, 6)")
    return c + correction.restoring_correction