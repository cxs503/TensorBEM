"""Wigley III modified hull mesh generator for 3-D frequency-domain BEM.

The Wigley III hull (Journee 1992) is a modified parabolic hull form:

    y(x, z) = (B/2) * (1 - xi^2) * (1 - zeta^2) * (1 + alpha * xi^2)

where xi = 2x/L in [-1, 1] and zeta = z/T in [-1, 0].  The fullness term is
longitudinal; see :func:`wigley3_half_breadth` for the adjudication.

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
    """Compute the half-breadth of the Wigley III hull at (x, z).

    ``y = (B/2) (1 - xi^2) (1 - zeta^2) (1 + alpha xi^2)`` with
    ``xi = 2x/L`` and ``zeta = z/T``: the fullness parameter acts on the
    **longitudinal** coordinate, which is what "fuller ends than the standard
    parabolic Wigley (Wigley I)" in
    :mod:`tensorbem.wigley3_freq` / ``docs/wigley3_freq_verification_report.md``
    means.  Applying it to ``zeta`` instead (the pre-m16935 code) keeps
    ``Cwp = 2/3`` while making the sections full -- a hull that carries the
    same ``Cb = 0.543`` but is a different ship, 36 % apart in heave added
    mass (mission m16892 geometry gate).  Adjudicated and fixed in m16935; the
    two builders are now guarded to agree pointwise by
    ``tests/test_wigley3_hull_alpha_axis.py``.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    xi = 2.0 * x / L
    zeta = z / T
    return (B / 2.0) * (1.0 - xi**2) * (1.0 - zeta**2) * (1.0 + alpha * xi**2)


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

    # Waterplane area.  At the waterplane zeta = 0 every vertical factor is 1,
    # hence y(x, 0) = (B/2)(1 - xi^2)(1 + alpha xi^2) and
    #     A_wp = 2 * int_{-L/2}^{L/2} y dx = B L (2/3 + 2 alpha/15).
    A_wp = B * L * (2.0 / 3.0 + 2.0 * alpha / 15.0)
    # Transverse second moment of the waterplane, I_xx = (2/3) int y(x,0)^3 dx.
    # With u = xi the polynomial integral is exact:
    #     int_{-1}^{1} (1-u^2)^3 (1+alpha u^2)^3 du
    #         = 32/35 + alpha 32/105 + alpha^2 32/385 + alpha^3 32/3003,
    # so I_xx = (B^3 L / 24) * E.
    e_moment = (
        32.0 / 35.0
        + alpha * 32.0 / 105.0
        + alpha**2 * 32.0 / 385.0
        + alpha**3 * 32.0 / 3003.0
    )
    I_xx = B**3 * L / 24.0 * e_moment
    # I_yy = int x^2 2y dx = (B L^3/8) int_{-1}^{1} u^2 (1-u^2)(1+alpha u^2) du
    #     = (B L^3/8) (4/15 + 4 alpha/35).
    I_yy = B * L**3 / 8.0 * (4.0 / 15.0 + 4.0 * alpha / 35.0)
    # Centre of buoyancy.  Every section is A(x) (1 - zeta^2) with the same
    # vertical shape, so the vertical centroid is zeta_c = -3/8 independently of
    # alpha: KB = T - 3T/8 = 5T/8.  (For the retired zeta-placed alpha the
    # section shape itself changed with alpha and KB drifted with it.)
    KB = 5.0 * T / 8.0
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
