"""Wigley III hullform for frequency-domain radiation/diffraction/RAO verification.

The Wigley III hullform is a mathematical ship hull with fuller ends than the
standard parabolic Wigley (Wigley I).  It is defined by

    y(x, z) = (B/2) * (1 - u^2) * (1 - v^2) * (1 + c * u^2)

where ``u = 2x/L``, ``v = z/T``, and ``c`` controls the fullness of the ends.
The block coefficient is

    Cb = 4/9 + 4c/45.

For the canonical Wigley III used in Journee (1992) and WAMIT verification,
``Cb = 0.543`` which corresponds to ``c = (Cb - 4/9) * 45/4 ≈ 1.1088``.

Principal dimensions (Journee 1992):
    L = 3.0 m,  B = 0.3 m,  T = 0.1875 m,  Cb = 0.543.

References
----------
- Journee, J.M.J. (1992). "Hydrodynamic coefficients of the Wigley III hull."
  Delft University of Technology, Report 922.
- WAMIT User Manual (v6.x), Wigley III verification case.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .geometry3d import WettedHullMesh

__all__ = [
    "WIGLEY3_L",
    "WIGLEY3_B",
    "WIGLEY3_T",
    "WIGLEY3_CB",
    "WIGLEY3_C",
    "wigley3_halfbreadth",
    "wigley3_wetted_hull_mesh",
    "wigley3_hydrostatic_reference",
    "wigley3_wetted_area_reference",
]

FloatArray = NDArray[np.float64]

#: Principal dimensions of the canonical Wigley III (Journee 1992).
WIGLEY3_L = 3.0
WIGLEY3_B = 0.3
WIGLEY3_T = 0.1875
#: Block coefficient Cb = 0.543.
WIGLEY3_CB = 0.543
#: Fullness coefficient: c = (Cb - 4/9) * 45/4.
WIGLEY3_C = (WIGLEY3_CB - 4.0 / 9.0) * 45.0 / 4.0


def wigley3_halfbreadth(
    x: FloatArray, z: FloatArray, L: float, B: float, T: float, c: float = WIGLEY3_C
) -> FloatArray:
    """Wigley III half-breadth ``y(x, z)``.

    Parameters
    ----------
    x, z :
        Longitudinal and vertical coordinates (m).
    L, B, T :
        Length, beam, draft (m).
    c :
        Fullness coefficient (default gives Cb = 0.543).
    """
    u = 2.0 * np.asarray(x, dtype=float) / L
    v = np.asarray(z, dtype=float) / T
    return 0.5 * B * (1.0 - u * u) * (1.0 - v * v) * (1.0 + c * u * u)


def wigley3_wetted_hull_mesh(
    L: float = WIGLEY3_L,
    B: float = WIGLEY3_B,
    T: float = WIGLEY3_T,
    c: float = WIGLEY3_C,
    n_sections: int = 20,
    n_levels: int = 10,
    *,
    waterline_tol: float = 1.0e-12,
) -> WettedHullMesh:
    """Wetted-surface triangle mesh of the Wigley III hull (``z <= 0``).

    The grid is ``(n_sections + 1) x (n_levels + 1)`` per side; the post
    meridians (``i = 0``, ``i = n_sections``) and the keel row
    (``j = n_levels``) are shared by the port and starboard sheets, so the
    surface is closed along ``y = 0`` and its only boundary is the waterline.

    Parameters
    ----------
    L, B, T :
        Length, beam and draft of the hullform (m).
    c :
        Fullness coefficient.
    n_sections, n_levels :
        Number of panel columns along the length and rows along the draft.
        Panel count ≈ ``2 * n_sections * n_levels * 2`` (both sides).
    waterline_tol :
        Passed to :class:`~tensorbem.geometry3d.WettedHullMesh`.

    Returns
    -------
    WettedHullMesh
        Mesh with outward normals (into the fluid) and one closed waterline
        loop.
    """
    if L <= 0 or B <= 0 or T <= 0:
        raise ValueError("L, B and T must be positive")
    if n_sections < 2 or n_levels < 1:
        raise ValueError("need n_sections >= 2 and n_levels >= 1")

    s_axis = np.linspace(0.0, 1.0, n_sections + 1)
    t_axis = np.linspace(0.0, 1.0, n_levels + 1)

    def _halfbreadth(s: float, t: float) -> float:
        u = 2.0 * s - 1.0  # u in [-1, 1]
        v = t  # v in [0, 1], z = -v*T
        return float(0.5 * B * (1.0 - u * u) * (1.0 - v * v) * (1.0 + c * u * u))

    # Port sheet: y >= 0 for every (i, j)
    port = np.empty(((n_sections + 1) * (n_levels + 1), 3), dtype=float)
    for i, s in enumerate(s_axis):
        for j, t in enumerate(t_axis):
            idx = i * (n_levels + 1) + j
            port[idx] = (
                -0.5 * L + s * L,
                _halfbreadth(s, t),
                -t * T,
            )

    # Starboard sheet: mirror of the interior columns only
    starboard_index: dict[tuple[int, int], int] = {}
    starboard_vertices = []
    for i in range(1, n_sections):
        for j in range(n_levels):  # j == n_levels is the shared keel
            xx = -0.5 * L + s_axis[i] * L
            yy = -_halfbreadth(s_axis[i], t_axis[j])
            starboard_index[(i, j)] = len(port) + len(starboard_vertices)
            starboard_vertices.append((xx, yy, -t_axis[j] * T))

    vertices = (
        np.vstack([port, np.asarray(starboard_vertices, dtype=float)])
        if starboard_vertices
        else port
    )

    def port_id(i: int, j: int) -> int:
        return i * (n_levels + 1) + j

    def star_id(i: int, j: int) -> int:
        if i == 0 or i == n_sections or j == n_levels:
            return port_id(i, j)
        return starboard_index[(i, j)]

    tol_y = 1.0e-14 * max(1.0, B)

    def _abs_y(vertex: int) -> float:
        return abs(float(vertices[vertex, 1]))

    faces: list[tuple[int, int, int]] = []
    for i in range(n_sections):
        for j in range(n_levels):
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            p = [port_id(*c) for c in corners]
            q = [star_id(*c) for c in corners]
            span_ac = _abs_y(p[0]) + _abs_y(p[2])
            span_bd = _abs_y(p[1]) + _abs_y(p[3])
            for ids in (p, q):
                if span_ac >= span_bd:
                    tris = [(ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])]
                else:
                    tris = [(ids[0], ids[1], ids[3]), (ids[1], ids[2], ids[3])]
                if ids is q:
                    tris = [(t[0], t[2], t[1]) for t in tris]
                faces.extend(tris)

    return WettedHullMesh(
        vertices, np.asarray(faces, dtype=np.int64), waterline_tol=waterline_tol
    )


def wigley3_hydrostatic_reference(
    L: float = WIGLEY3_L,
    B: float = WIGLEY3_B,
    T: float = WIGLEY3_T,
    c: float = WIGLEY3_C,
    *,
    density: float = 1025.0,
    gravity: float = 9.80665,
    keel_of_gravity: float | None = None,
) -> dict[str, Any]:
    """Closed-form hydrostatic properties of the Wigley III hull.

    The volume integral separates in ``u`` and ``v``:

        V = B * L/2 * integral_{-1}^{1} (1-u^2)(1+c*u^2) du * T * integral_{-1}^{0} (1-v^2) dv
          = B * L/2 * [4/3 + 4c/15] * T * 2/3
          = B * L * T * (4/9 + 4c/45)
          = L * B * T * Cb.

    The centre of buoyancy ``z_B`` is the 1-D centroid of ``(1-v^2)`` on ``[-1, 0]``:
    ``z_B = -3T/8`` (independent of ``c``).

    Waterplane area: ``A_wp = 2 * integral_{-L/2}^{L/2} y(x, 0) dx``
        = B * L/2 * integral_{-1}^{1} (1-u^2)(1+c*u^2) du
        = B * L * (2/3 + 2c/15).
    """
    if L <= 0 or B <= 0 or T <= 0:
        raise ValueError("L, B and T must be positive")

    # Volume
    int_u = 4.0 / 3.0 + 4.0 * c / 15.0
    int_v = 2.0 / 3.0
    volume = B * L / 2.0 * int_u * T * int_v
    Cb = volume / (L * B * T)

    # Centre of buoyancy (z-direction only; x_B = y_B = 0 by symmetry)
    z_b = -3.0 * T / 8.0
    kb = z_b + T

    # Waterplane area
    a_wp = B * L / 2.0 * int_u

    # Second moments of waterplane area
    # I_xx = (2/3) * integral y^3 dx  (transverse, about x-axis)
    # I_yy = integral x^2 * 2y dx     (longitudinal, about y-axis)
    # These require numerical integration for c != 0.
    from numpy.polynomial.legendre import leggauss
    nodes, weights = leggauss(200)
    u_q = nodes
    w_q = weights

    # I_yy: integral_{-L/2}^{L/2} x^2 * 2*y(x,0) dx
    # x = L*u/2, dx = L/2 du, y(x,0) = B/2 * (1-u^2)(1+c*u^2)
    # I_yy = integral_{-1}^{1} (L*u/2)^2 * B * (1-u^2)(1+c*u^2) * L/2 du
    #      = B * L^3 / 8 * integral_{-1}^{1} u^2 * (1-u^2)(1+c*u^2) du
    integrand_yy = u_q**2 * (1 - u_q**2) * (1 + c * u_q**2)
    i_yy = B * L**3 / 8.0 * float(np.sum(w_q * integrand_yy))

    # I_xx: (2/3) * integral_{-L/2}^{L/2} y(x,0)^3 dx
    # y(x,0) = B/2 * (1-u^2)(1+c*u^2)
    # I_xx = (2/3) * integral_{-1}^{1} [B/2 * (1-u^2)(1+c*u^2)]^3 * L/2 du
    y_cubed = (0.5 * B * (1 - u_q**2) * (1 + c * u_q**2)) ** 3
    i_xx = 2.0 / 3.0 * L / 2.0 * float(np.sum(w_q * y_cubed))

    bm_t = i_xx / volume
    bm_l = i_yy / volume

    kg = 0.5 * T if keel_of_gravity is None else float(keel_of_gravity)
    gm_t = kb + bm_t - kg
    gm_l = kb + bm_l - kg

    return {
        "L": float(L),
        "B": float(B),
        "T": float(T),
        "c": float(c),
        "volume": volume,
        "Cb": Cb,
        "center_of_buoyancy": np.array([0.0, 0.0, z_b]),
        "KB": kb,
        "KG": kg,
        "waterplane_area": a_wp,
        "I_xx": i_xx,
        "I_yy": i_yy,
        "BM_T": bm_t,
        "BM_L": bm_l,
        "GM_T": gm_t,
        "GM_L": gm_l,
        "C33": density * gravity * a_wp,
        "C44": density * gravity * volume * gm_l,
        "C55": density * gravity * volume * gm_t,
        "density": float(density),
        "gravity": float(gravity),
    }


def wigley3_wetted_area_reference(
    L: float = WIGLEY3_L,
    B: float = WIGLEY3_B,
    T: float = WIGLEY3_T,
    c: float = WIGLEY3_C,
    n_quad: int = 240,
) -> float:
    """Wetted surface area via 2-D Gauss-Legendre quadrature."""
    from numpy.polynomial.legendre import leggauss

    x, wx = leggauss(n_quad)
    z, wz = leggauss(n_quad)
    xs = 0.5 * L * x
    wxs = 0.5 * L * wx
    zs = 0.5 * T * (z - 1.0)
    wzs = 0.5 * T * wz
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    ww = np.outer(wxs, wzs)

    u = 2.0 * xx / L
    v = zz / T
    y = 0.5 * B * (1.0 - u * u) * (1.0 - v * v) * (1.0 + c * u * u)

    # Partial derivatives
    y_x = 0.5 * B * (-8.0 * xx / L**2) * (1.0 - v * v) * (1.0 + c * u * u) + \
          0.5 * B * (1.0 - u * u) * (1.0 - v * v) * (2.0 * c * 2.0 * xx / L**2)
    y_z = 0.5 * B * (1.0 - u * u) * (-2.0 * zz / T**2) * (1.0 + c * u * u)

    jac = np.sqrt(1.0 + y_x**2 + y_z**2)
    return float(2.0 * np.sum(ww * jac))
