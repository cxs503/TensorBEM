"""Wigley III modified hull mesh generator for 3-D frequency-domain BEM.

The Wigley III hull (Journee 1992) is a modified parabolic hull form:

    y(x, z) = (B/2) * (1 - xi^2) * (1 - zeta^2) * (1 + alpha * zeta^2)

where xi = 2x/L in [-1, 1] and zeta = z/T in [-1, 0].

Reference:
    Journee, J.M.J. (1992). "Experiments and Calculations on 4 Wigley
    Hull Forms in Head Waves." Report 09-V-09, Delft University of
    Technology, Department of Ship Hydromechanics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .geometry3d import WettedHullMesh

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# Wigley III parameters (Journee 1992)
WIGLEY3_L = 3.0       # Length (m)
WIGLEY3_B = 0.3       # Beam (m)
WIGLEY3_T = 0.1875    # Draft (m)
WIGLEY3_CB = 0.543    # Block coefficient
WIGLEY3_ALPHA = (WIGLEY3_CB - 4.0 / 9.0) * 45.0 / 4.0  # ~1.109


def wigley3_half_breadth(
    x: float | FloatArray, z: float | FloatArray,
    L: float = WIGLEY3_L,
    B: float = WIGLEY3_B,
    T: float = WIGLEY3_T,
    alpha: float = WIGLEY3_ALPHA,
) -> float | FloatArray:
    """Compute the half-breadth of the Wigley III hull at (x, z)."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    xi = 2.0 * x / L
    zeta = z / T
    return (B / 2.0) * (1.0 - xi**2) * (1.0 - zeta**2) * (1.0 + alpha * zeta**2)


def wigley3_displaced_volume(
    L: float = WIGLEY3_L,
    B: float = WIGLEY3_B,
    T: float = WIGLEY3_T,
    alpha: float = WIGLEY3_ALPHA,
) -> float:
    """Analytical displaced volume of the Wigley III hull."""
    return L * B * T * (4.0 / 9.0 + 4.0 * alpha / 45.0)


def wigley3_block_coefficient(alpha: float = WIGLEY3_ALPHA) -> float:
    """Block coefficient from the modification parameter."""
    return 4.0 / 9.0 + 4.0 * alpha / 45.0


def wigley3_wetted_hull_mesh(
    nx: int = 20,
    nz: int = 10,
    L: float = WIGLEY3_L,
    B: float = WIGLEY3_B,
    T: float = WIGLEY3_T,
    alpha: float = WIGLEY3_ALPHA,
) -> WettedHullMesh:
    """Generate a triangular panel mesh of the Wigley III wetted hull.

    The wetted surface is swept as a single closed strip: the ring index
    ``k = 0 .. 2*nz`` runs from the port waterline (``z = 0``) down the port
    side to the keel (``k = nz``, ``z = -T``) and back up the starboard side,
    with ``z_k = -T sin(pi k / (2 nz))`` so that ``z = 0`` and ``z = -T`` are
    hit exactly.  Longitudinal stations are cosine spaced,
    ``x_i = -(L/2) cos(pi i / (nx - 1))``, so the stations land exactly on the
    ends ``x = +-L/2``, where the half breadth vanishes and the port and
    starboard sheets meet.

    Vertices are welded on shared coordinates, so the collapse at the keel
    (``y = 0``) and at both ends (``y = 0`` for every ``z``) produces a closed
    lower hull; the only open edges are the two waterline strips at ``z = 0``,
    which is exactly the :class:`~tensorbem.geometry3d.WettedHullMesh`
    contract.  Panel normals point out of the body into the fluid.

    Parameters
    ----------
    nx : int
        Number of longitudinal stations (``nx - 1`` panels per ring strip).
    nz : int
        Number of panels in the vertical direction, per side.
    L, B, T : float
        Length, beam and draft (m).
    alpha : float
        Modification parameter (``1.109`` reproduces Journee's Wigley III,
        ``Cb = 0.543``; ``alpha = 0`` is the parabolic Wigley hull with
        ``Cb = 4/9``).

    Returns
    -------
    mesh : WettedHullMesh
        Triangulated wetted hull surface with outward normals and a closed
        waterline loop at ``z = 0``.
    """
    if nx < 4:
        raise ValueError("nx must be >= 4")
    if nz < 2:
        raise ValueError("nz must be >= 2")
    for name, value in (("L", L), ("B", B), ("T", T)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")

    stations = -0.5 * float(L) * np.cos(np.linspace(0.0, np.pi, int(nx)))
    rings = -float(T) * np.sin(
        np.pi * np.arange(2 * int(nz) + 1, dtype=float) / (2.0 * int(nz))
    )

    vertices: list[tuple[float, float, float]] = []
    index: dict[tuple[float, float, float], int] = {}

    def _vertex(point: tuple[float, float, float]) -> int:
        key = (round(point[0], 12) + 0.0, round(point[1], 12) + 0.0, round(point[2], 12) + 0.0)
        if key not in index:
            index[key] = len(vertices)
            vertices.append(key)
        return index[key]

    grid: list[list[int]] = []
    for x in stations:
        row = []
        for k, z in enumerate(rings):
            half = float(wigley3_half_breadth(x, z, L, B, T, alpha))
            y = half if k <= int(nz) else -half
            row.append(_vertex((float(x), float(y), float(z))))
        grid.append(row)

    faces: list[tuple[int, int, int]] = []
    for i in range(int(nx) - 1):
        for k in range(2 * int(nz)):
            a, b = grid[i][k], grid[i + 1][k]
            c, d = grid[i + 1][k + 1], grid[i][k + 1]
            for triangle in ((a, b, c), (a, c, d)):
                if len(set(triangle)) == 3:
                    faces.append(triangle)

    if len(faces) < 3:
        raise ValueError("failed to generate sufficient panels")

    return WettedHullMesh(
        np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)
    )


@dataclass
class Wigley3HullProperties:
    """Physical properties of the Wigley III hull."""
    L: float
    B: float
    T: float
    alpha: float
    Cb: float
    volume: float
    mass: float
    waterplane_area: float
    Ixx: float
    Iyy: float
    KB: float
    BM_T: float
    BM_L: float
    KG_default: float


def compute_wigley3_properties(
    L: float = WIGLEY3_L,
    B: float = WIGLEY3_B,
    T: float = WIGLEY3_T,
    alpha: float = WIGLEY3_ALPHA,
    rho: float = 1025.0,
) -> Wigley3HullProperties:
    """Compute analytical hydrostatic properties of the Wigley III hull."""
    Cb = wigley3_block_coefficient(alpha)
    volume = L * B * T * (4.0 / 9.0 + 4.0 * alpha / 45.0)
    mass = rho * volume

    A_wp = 2.0 * B * L / 3.0
    I_xx = 4.0 * B**3 * L / 105.0
    I_yy = B * L**3 / 30.0
    # Centre of buoyancy.  With the analytic half breadth
    # b(x, z) = (B/2)(1 - xi^2)(1 - zeta^2)(1 + alpha zeta^2) the vertical
    # moment integrates in closed form,
    #     int_{-1}^{0} zeta (1 - zeta^2)(1 + alpha zeta^2) d zeta
    #         = -(1/4 + alpha/12),
    #     int_{-1}^{0}      (1 - zeta^2)(1 + alpha zeta^2) d zeta
    #         =  (2/3 + 2 alpha/15),
    # hence KB = T [1 - (15 + 5 alpha)/(40 + 8 alpha)], which reduces to the
    # classical KB = 5T/8 at alpha = 0.  (The previous formula
    # T (65 + 11 alpha)/(16 (5 + alpha)) returned 0.8125 T instead of 0.625 T
    # at alpha = 0, i.e. a +30 % centre-of-buoyancy error.)
    KB = T * (1.0 - (15.0 + 5.0 * alpha) / (40.0 + 8.0 * alpha))
    BM_T = I_xx / volume
    BM_L = I_yy / volume
    KG_default = T * 0.5

    return Wigley3HullProperties(
        L=L, B=B, T=T, alpha=alpha, Cb=Cb,
        volume=volume, mass=mass,
        waterplane_area=A_wp,
        Ixx=I_xx, Iyy=I_yy,
        KB=KB, BM_T=BM_T, BM_L=BM_L,
        KG_default=KG_default,
    )
