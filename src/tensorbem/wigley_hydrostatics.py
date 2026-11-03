"""Hydrostatics of the mathematical Wigley hull: wetted mesh + closed-form reference.

Why this module exists
----------------------
:mod:`tensorbem.hydrostatic` computes displaced volume, centre of buoyancy,
waterplane area/second moments and the 6x6 restoring matrix from a
:class:`~tensorbem.geometry3d.WettedHullMesh`.  It was verified against a
hemisphere and a box (``docs/hydrostatic_verification.md``), but the one hull
form with *ship-like fine ends* -- the parabolic Wigley hull -- could not be
meshed there:

    "Wigley parabolic hull cannot be meshed with the quad-sheet approach ... a
    Wigley generator needs an explicit triangle-fan closure at the posts"
    (``dev_report_tensorbem-hydrostatic-verification.md`` §5)

That is the gap this module closes.  The Wigley surface

    y(x, z) = (B/2) * (1 - (2x/L)^2) * (1 - (z/T)^2),   x in [-L/2, L/2],
                                                        z in [-T, 0],

is parameterised here in ``(s, t)`` with ``x = -L/2 + s L`` and ``z = -t T``,
so

    y(s, t) = 2 B s (1 - s) (1 - t^2).

The two zero-width curves of the hull, the fore/stern **posts**
(``s = 0``, ``s = 1``) and the **keel** (``t = 1``), are *grid lines*, so no
panel is degenerate: a post panel spans two distinct points on the post
meridian (different ``z``) and two points with ``y > 0``.  The port and
starboard sheets therefore share the post and keel vertices exactly, the
wetted surface is closed there, and the only boundary is the calm-waterplane
waterline ``z = 0`` -- one closed loop, which is what
:class:`~tensorbem.geometry3d.WettedHullMesh` requires.  No triangle fan and no
degenerate sliver is involved.

Closed-form hydrostatic reference
---------------------------------
Every quantity below is an elementary integral of the parabola/paraboloid
above, so the reference used by the benchmarks is independent of any mesh:

======================  ==========================================  ====================
quantity                closed form                                 value
======================  ==========================================  ====================
displaced volume        ``V = 4 L B T / 9``                         ``Cb = 4/9``
centre of buoyancy      ``z_B = -3 T / 8`` (below ``z = 0``)        ``KB = 5T/8`` from keel
waterplane area         ``A_wp = 2 L B / 3``                        --
transverse moment       ``I_xx = 4 B^3 L / 105``  (about x)         ``BM_T = 3B^2/(35T)``
longitudinal moment     ``I_yy = B L^3 / 30``     (about y)         ``BM_L = 3L^2/(40T)``
======================  ==========================================  ====================

Derivation (one line each, all used as test references):

* ``V = 2 * (B/2) * integral_{-L/2}^{L/2}(1 - (2x/L)^2) dx *
  integral_{-T}^{0}(1 - (z/T)^2) dz = 2*(B/2)*(2L/3)*(2T/3) = 4LBT/9``;
* the factors separate in ``x`` and ``z``, so ``z_B`` is the 1-D centroid of
  ``1 - (z/T)^2`` on ``[-T, 0]``: ``(-T^2/4)/(2T/3) = -3T/8``;
* ``A_wp = integral 2 y(x,0) dx = 2LB/3``; ``I_yy = integral x^2 * 2 y dx =
  B L^3 / 30`` and ``I_xx = integral (2/3) y^3 dx = 4 B^3 L / 105``.

The wetted surface area has no elementary closed form; it is reported from a
2-D Gauss-Legendre quadrature of ``sqrt(1 + y_x^2 + y_z^2)`` over the
``(x, z)`` projection, which is a reference for the *mesh* rather than for the
solver.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .geometry3d import WettedHullMesh

__all__ = [
    "wigley_wetted_hull_mesh",
    "wigley_hydrostatic_reference",
    "wigley_wetted_area_reference",
]

FloatArray = NDArray[np.float64]


def _halfbreadth(s: FloatArray, t: FloatArray, B: float) -> FloatArray:
    """``y(s, t) = 2 B s (1 - s) (1 - t^2)`` for the Wigley surface."""
    return 2.0 * B * s * (1.0 - s) * (1.0 - t * t)


def wigley_wetted_hull_mesh(
    L: float,
    B: float,
    T: float,
    n_sections: int = 32,
    n_levels: int = 16,
    *,
    waterline_tol: float = 1.0e-12,
) -> WettedHullMesh:
    """Wetted-surface triangle mesh of the parabolic Wigley hull (``z <= 0``).

    The grid is ``(n_sections + 1) x (n_levels + 1)`` per side; the post
    meridians (``i = 0``, ``i = n_sections``) and the keel row
    (``j = n_levels``) are shared by the port and starboard sheets, so the
    surface is closed along ``y = 0`` and its only boundary is the waterline.

    Parameters
    ----------
    L, B, T :
        Length, beam and draft of the hullform (m).
    n_sections, n_levels :
        Number of panel columns along the length and rows along the draft.
    waterline_tol :
        Passed to :class:`~tensorbem.geometry3d.WettedHullMesh`.

    Returns
    -------
    WettedHullMesh
        Mesh with outward normals (into the fluid) and one closed waterline
        loop.  The displaced volume of the closed body is positive.
    """
    if L <= 0 or B <= 0 or T <= 0:
        raise ValueError("L, B and T must be positive")
    if n_sections < 2 or n_levels < 1:
        raise ValueError("need n_sections >= 2 and n_levels >= 1")

    s_axis = np.linspace(0.0, 1.0, n_sections + 1)
    t_axis = np.linspace(0.0, 1.0, n_levels + 1)

    # Port sheet: y >= 0 for every (i, j); the shared curves are y == 0.
    port = np.empty(((n_sections + 1) * (n_levels + 1), 3), dtype=float)
    for i, s in enumerate(s_axis):
        for j, t in enumerate(t_axis):
            idx = i * (n_levels + 1) + j
            port[idx] = (
                -0.5 * L + s * L,
                float(_halfbreadth(np.array(s), np.array(t), B)),
                -t * T,
            )

    # Starboard sheet: mirror of the interior columns only; the post columns and
    # the keel row are shared with the port sheet, so they are not duplicated.
    starboard_index: dict[tuple[int, int], int] = {}
    starboard_vertices = []
    for i in range(1, n_sections):
        for j in range(n_levels):  # j == n_levels is the shared keel
            xx = -0.5 * L + s_axis[i] * L
            yy = -float(_halfbreadth(np.array(s_axis[i]), np.array(t_axis[j]), B))
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
        # y == 0 curves are shared with the port sheet
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
            # Diagonal choice: never split along a chord whose two endpoints both
            # lie on the zero-width seam y = 0 (post meridian or keel line).  Such
            # a chord is shared by the port and starboard sheets of the same cell
            # and shows up as a duplicated directed edge; it is exactly the
            # defect that made this hull unmeshable in the earlier attempt.
            span_ac = _abs_y(p[0]) + _abs_y(p[2])
            span_bd = _abs_y(p[1]) + _abs_y(p[3])
            for ids in (p, q):
                if span_ac >= span_bd:
                    tris = [(ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])]
                else:
                    tris = [(ids[0], ids[1], ids[3]), (ids[1], ids[2], ids[3])]
                if ids is q:  # mirrored sheet: reversed winding keeps n outward
                    tris = [(t[0], t[2], t[1]) for t in tris]
                faces.extend(tris)

    return WettedHullMesh(
        vertices, np.asarray(faces, dtype=np.int64), waterline_tol=waterline_tol
    )


def wigley_hydrostatic_reference(
    L: float,
    B: float,
    T: float,
    *,
    density: float = 1025.0,
    gravity: float = 9.80665,
    keel_of_gravity: float | None = None,
) -> dict[str, Any]:
    """Closed-form hydrostatic properties of the parabolic Wigley hull.

    Parameters
    ----------
    L, B, T :
        Hullform dimensions (m).
    density, gravity :
        Fluid properties for the restoring matrix.
    keel_of_gravity :
        Height of the centre of gravity above the keel (m), used for the
        metacentric heights.  ``None`` puts it at mid-draft ``T/2``.

    Returns
    -------
    dict
        Closed-form volumes, centres, waterplane moments, metacentric radii,
        metacentric heights and the three non-zero restoring terms.
    """
    if L <= 0 or B <= 0 or T <= 0:
        raise ValueError("L, B and T must be positive")

    volume = 4.0 * L * B * T / 9.0
    z_b = -3.0 * T / 8.0
    kb = z_b + T  # height of the centre of buoyancy above the keel
    a_wp = 2.0 * L * B / 3.0
    i_xx = 4.0 * B**3 * L / 105.0  # transverse (about x, for roll)
    i_yy = B * L**3 / 30.0  # longitudinal (about y, for pitch)
    bm_t = i_xx / volume
    bm_l = i_yy / volume

    kg = 0.5 * T if keel_of_gravity is None else float(keel_of_gravity)
    gm_t = kb + bm_t - kg
    gm_l = kb + bm_l - kg

    return {
        "L": float(L),
        "B": float(B),
        "T": float(T),
        "volume": volume,
        "Cb": volume / (L * B * T),
        "Cm": 2.0 / 3.0,
        "Cp": 2.0 / 3.0,
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


def wigley_wetted_area_reference(
    L: float, B: float, T: float, n_quad: int = 240
) -> float:
    """Wetted surface area ``2 * int int sqrt(1 + y_x^2 + y_z^2) dx dz``.

    Evaluated with a 2-D Gauss-Legendre rule on the ``(x, z)`` projection; the
    integrand is smooth there (the knife edges at the posts are finite-slope,
    not singular), so the rule converges spectrally and serves as a reference
    for the faceted mesh area.
    """
    x, wx = np.polynomial.legendre.leggauss(n_quad)
    z, wz = np.polynomial.legendre.leggauss(n_quad)
    xs = 0.5 * L * x  # [-L/2, L/2]
    wxs = 0.5 * L * wx
    zs = 0.5 * T * (z - 1.0)  # [-T, 0]
    wzs = 0.5 * T * wz
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    ww = np.outer(wxs, wzs)

    y_x = 0.5 * B * (-8.0 * xx / L**2) * (1.0 - (zz / T) ** 2)
    y_z = 0.5 * B * (1.0 - (2.0 * xx / L) ** 2) * (-2.0 * zz / T**2)
    jac = np.sqrt(1.0 + y_x**2 + y_z**2)
    return float(2.0 * np.sum(ww * jac))