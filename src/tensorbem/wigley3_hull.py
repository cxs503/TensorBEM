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

    Uses a cosine distribution for x-stations to cluster panels near the ends
    while avoiding the singular points at x=±L/2 where y=0.

    Parameters
    ----------
    nx : int
        Number of panels in the longitudinal direction.
    nz : int
        Number of panels in the vertical direction (each side).
    L, B, T : float
        Hull dimensions (m).
    alpha : float
        Modification parameter.

    Returns
    -------
    mesh : WettedHullMesh
        Triangulated wetted hull surface with outward normals.
    """
    if nx < 4:
        raise ValueError("nx must be >= 4")
    if nz < 2:
        raise ValueError("nz must be >= 2")

    # Use cosine distribution for x-stations to avoid singular ends
    # xi goes from -1+eps to 1-eps where eps is small
    eps = 1e-6
    xi_vals = -np.cos(np.linspace(0, np.pi, nx + 1)) * (1 - eps)
    x_vals = xi_vals * L / 2.0

    n_theta = 2 * nz
    theta_vals = np.linspace(0.0, np.pi, n_theta + 1)

    # Build vertex grid
    vertices = []
    grid = {}

    for i, x in enumerate(x_vals):
        for j, theta in enumerate(theta_vals):
            if theta <= np.pi / 2:
                # Port side
                zeta = -np.sin(theta)
                z = zeta * T
                y_abs = float(wigley3_half_breadth(x, z, L, B, T, alpha))
                y = -max(y_abs, 0.0)
            else:
                # Starboard side
                zeta = -np.cos(theta - np.pi / 2)
                z = zeta * T
                y_abs = float(wigley3_half_breadth(x, z, L, B, T, alpha))
                y = max(y_abs, 0.0)

            idx = len(vertices)
            vertices.append((x, y, z))
            grid[(i, j)] = idx

    # Build triangles
    faces = []
    for i in range(nx):
        for j in range(n_theta):
            v00 = grid[(i, j)]
            v10 = grid[(i + 1, j)]
            v11 = grid[(i + 1, j + 1)]
            v01 = grid[(i, j + 1)]

            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    verts_array = np.array(vertices, dtype=float)
    faces_array = np.array(faces, dtype=np.int64)

    if len(faces_array) < 3:
        raise ValueError("Failed to generate sufficient panels")

    return WettedHullMesh(verts_array, faces_array)


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
    KB = T * (65.0 + 11.0 * alpha) / (16.0 * (5.0 + alpha))
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
