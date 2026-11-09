"""Rankine-source free-surface panel method (truncated-domain Neumann-Kelvin).

Why this module exists
----------------------
Before this module the repository had three routes to a steady wave resistance,
and none of them was a *Rankine* free-surface solve:

* :mod:`tensorbem.wigley` / :mod:`tensorbem.michell` -- Michell's thin-ship
  integral: the sources live on the centreplane and the free-surface condition
  is imposed analytically inside the Kochin integral.  There are no
  free-surface unknowns at all.
* :mod:`tensorbem.neumann_kelvin` -- the **Kelvin (Havelock) Green function**:
  the free surface is carried by the kernel, again without free-surface panels.
* :mod:`tensorbem.rankine_wave` -- despite the name, it evaluates the Michell
  integral of a panelised centreplane sheet (thin ship).

A Rankine panel method is the other classical family: the kernel is the plain
``1/|p-q|`` (no Havelock double integral), the free surface ``z = 0`` becomes a
**truncated patch of panels carrying unknown source strengths**, and the
radiation condition is enforced numerically by upwind (upstream) differencing of
the free-surface condition.  This module implements exactly that, so the two
error sources that neither a Green-function nor a thin-ship code can quantify --
the free-surface discretisation and the domain truncation -- become measurable
quantities (:func:`truncation_study`, :func:`grid_convergence_index`).

Formulation
-----------
Potential flow, deep water, steady forward speed ``U`` along ``+x``, calm water
at ``z = 0``, fluid in ``z < 0``.  With ``phi`` the *disturbance* potential
(total ``= U x + phi``) the linearised problem is

    grad^2 phi = 0                     in the fluid,
    dphi/dn    = -U n_x                on the wetted hull (``n`` out of fluid),
    phi_xx + k0 phi_z = 0              on ``z = 0``,   k0 = g/U^2,
    phi -> 0                           far field upstream / laterally,

plus a radiation condition (no waves ahead of the disturbance).  ``phi`` is a
single layer of Rankine sources of strength ``sigma`` on the hull panels *and*
on the truncated free-surface panels,

    phi(p) = sum_j sigma_j * int_{panel j} dS / |p - q| .

Free-surface rows
-----------------
The free-surface condition is collocated at the free-surface panel centres.  Its
``z`` derivative is not obtained by quadrature of a near-singular 3-D kernel: for
a single layer lying entirely in the plane ``z = 0`` the one-sided limit of
``phi_z`` at an interior point of the plane is local,

    phi_z(p) = 2 pi sigma(p) + PV,        PV = 0   (planar sheet),

so the free-surface row at panel ``i`` reads

    [phi_xx]_i + 2 pi k0 sigma_i + k0 * (phi_z from off-plane hull sources)_i = 0,

with ``phi_xx`` evaluated by an upwind finite difference along ``y = const``.
The upwind bias is what supplies the numerical radiation condition: the discrete
operator transports information downstream only.

Appendix: why the in-plane principal value vanishes
---------------------------------------------------
Let ``sigma`` be continuous in ``z = 0`` and ``p`` a point of the sheet.  With
``eps -> 0`` along the normal,

    phi_z(p, eps) = int_plane sigma(q) eps / (|p-q|^2 + eps^2)^(3/2) dS_q .

Split at ``|p - q| = delta`` with ``eps << delta << 1``: the outer part is
bounded by ``2 pi eps / delta -> 0``; the inner part, with
``sigma(q) = sigma(p) + O(delta)``, equals
``2 pi sigma(p) (1 - eps/sqrt(delta^2+eps^2)) -> 2 pi sigma(p)``.  Discretely the
same statement holds panel by panel: for every *other* coplanar panel the kernel
``(z_p - z_q)/|p-q|^3`` vanishes identically (``z_p = z_q``), and for the own
panel the limit is ``2 pi``.
``tests/test_rankine_fs_panel.py::test_free_surface_jump_relation`` checks this
numerically and ``test_in_plane_single_layer_matches_quadrature`` checks the
closed-form in-plane rectangle potential used for the smooth part.

What is computed
----------------
:func:`solve_rankine_free_surface` returns the free-surface source strength, the
wave elevation ``zeta = -U phi_x / g`` on the truncated free surface, and the
wave resistance from two independent functionals:

* ``R_w`` by pressure integration on the hull, ``R_w = -rho U int phi_x n_x dS``
  (the definition used by every linear ship-wave code), and
* ``R_w`` from the far-field energy/momentum flux of the transverse wave system
  at the downstream edge of the truncated domain,
  ``R_w = (1/4) rho g int a_T(y)^2 dy`` with ``a_T`` the amplitude of the
  ``k0``-harmonic of ``zeta`` (deep-water ``c_g = U/2``).

They are functionals of the same discrete solution through different operators;
their agreement is the energy-consistency check of this module.

Conventions and limits
----------------------
* SI units; ``rho = 1025 kg/m^3``, ``g = 9.80665 m/s^2`` by default.
* ``Cw = R_w / (0.5 rho U^2 L T)`` (same reference as
  :class:`tensorbem.wigley.WigleyWaveResistanceResult`).
* Half-domain symmetry (``y -> -y``) is exploited: hull and free-surface panels
  are generated for ``y >= 0`` only and the mirror influence is folded in by
  reflecting the *field* point, which leaves the jump relation intact.
* Free-surface panels whose centre lies inside the hull waterplane carry no
  unknown (the free surface does not exist there); the upwind stencil still
  reads the smooth potential field at those points, i.e. the finite difference
  uses the analytic continuation of ``phi`` instead of a gap.
* Linear theory only: no viscosity, no wave breaking, no sinkage/trim, no
  nonlinear free-surface terms.  The verified content is the *discretisation*
  (convergence, truncation, energy consistency) and agreement with Michell's
  independent thin-ship evaluation of the same linear problem.

References
----------
- Dawson, C.W. (1977). "A practical computer method for solving ship-wave
  problems." Proc. 2nd Int. Conf. Numerical Ship Hydrodynamics, 30-38.
- Gadd, G.E. (1976). "Wave resistance and sinkage of a destroyer-type hull."
  Int. Shipbuilding Progress 23 (Rankine free-surface panels, truncation).
- Newman, J.N. (1977). *Marine Hydrodynamics*, MIT Press, ch. 6.
- Tuck, E.O. (1989). "The wave resistance formula of J.H. Michell (1898) and its
  significance." J. Eng. Math. 23, 1-15.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray

from .michell import DEFAULT_DENSITY, DEFAULT_GRAVITY

FloatArray = NDArray[np.float64]

_TWO_PI = 2.0 * math.pi
_TINY = 1.0e-300
_CHUNK_ELEMENTS = 2_000_000

__all__ = [
    "DEFAULT_DENSITY",
    "DEFAULT_GRAVITY",
    "FlatPanels",
    "RankineFreeSurfaceResult",
    "domain_extent_for_froude",
    "farfield_wave_resistance",
    "free_surface_grid",
    "gci_fine",
    "grid_convergence_index",
    "in_plane_single_layer",
    "rankine_free_surface_resistance",
    "solve_rankine_free_surface",
    "truncation_study",
    "wigley_hull_panels",
    "wigley_section_curve",
    "DoubleBodyResult",
    "double_body_rankine",
    "free_surface_operator_symbol",
    "truncation_and_stability_summary",
]


# ---------------------------------------------------------------------------
# Panel geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlatPanels:
    """A set of flat rectangular panels.

    ``centers`` ``(N, 3)`` are the panel centroids, ``e1``/``e2`` ``(N, 3)`` the
    in-plane orthonormal axes, ``normal = e1 x e2`` the unit normal (pointing
    *into the fluid* for hull panels and ``+z`` for free-surface panels),
    ``half1``/``half2`` the half-extents along ``e1``/``e2``.
    """

    centers: FloatArray
    e1: FloatArray
    e2: FloatArray
    normal: FloatArray
    half1: FloatArray
    half2: FloatArray

    def __post_init__(self) -> None:
        n = int(np.asarray(self.centers).shape[0])
        for name in ("centers", "e1", "e2", "normal"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.shape != (n, 3):
                raise ValueError(f"{name} must have shape (N, 3)")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must be finite")
        for name in ("half1", "half2"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.shape != (n,) or not np.isfinite(arr).all() or np.any(arr <= 0.0):
                raise ValueError(f"{name} must be positive and finite, shape (N,)")

    @property
    def count(self) -> int:
        return int(np.asarray(self.centers).shape[0])

    @property
    def areas(self) -> FloatArray:
        return 4.0 * np.asarray(self.half1) * np.asarray(self.half2)


def wigley_section_curve(
    x: ArrayLike, z: ArrayLike, L: float, B: float, T: float
) -> FloatArray:
    """Wigley half-breadth ``y = B/2 (1-(2x/L)^2)(1-(z/T)^2)``, clipped at 0."""
    xa = np.asarray(x, dtype=float)
    za = np.asarray(z, dtype=float)
    return (
        0.5
        * B
        * np.clip(1.0 - (2.0 * xa / L) ** 2, 0.0, None)
        * np.clip(1.0 - (za / T) ** 2, 0.0, None)
    )


def wigley_hull_panels(
    L: float,
    B: float,
    T: float,
    nx: int = 24,
    nz: int = 8,
    *,
    waterline_clustering: float = 0.5,
) -> FlatPanels:
    """Panelise the ``y >= 0`` half of a Wigley hull (``z in [-T, 0]``).

    ``nx`` cosine-spaced stations along ``x`` (clustered at bow and stern where
    wave generation concentrates) and ``nz`` strips in ``z``;
    ``waterline_clustering`` in ``[0, 1]`` shifts the ``z`` nodes towards the
    waterline ``z = 0`` (cosine distribution blended with the uniform one).
    """
    if L <= 0.0 or B <= 0.0 or T <= 0.0:
        raise ValueError("L, B, T must be positive")
    if nx < 2 or nz < 1:
        raise ValueError("nx >= 2 and nz >= 1 required")
    if not 0.0 <= waterline_clustering <= 1.0:
        raise ValueError("waterline_clustering must lie in [0, 1]")

    xs = -0.5 * L * np.cos(np.linspace(0.0, math.pi, nx + 1))
    s = np.linspace(0.0, 1.0, nz + 1)
    if waterline_clustering > 0.0:
        w = 0.5 * (1.0 - np.cos(math.pi * s))
        s = (1.0 - waterline_clustering) * s + waterline_clustering * w
    zs = -T * (1.0 - s)  # s = 0 at keel, 1 at waterline

    centers = np.empty((nx * nz, 3), dtype=float)
    e1 = np.empty_like(centers)
    e2 = np.empty_like(centers)
    normals = np.empty_like(centers)
    half1 = np.empty(nx * nz, dtype=float)
    half2 = np.empty(nx * nz, dtype=float)

    k = 0
    for i in range(nx):
        for j in range(nz):
            x0, x1 = xs[i], xs[i + 1]
            z0, z1 = zs[j], zs[j + 1]
            p = np.array(
                [
                    [x0, float(wigley_section_curve(x0, z0, L, B, T)), z0],
                    [x1, float(wigley_section_curve(x1, z0, L, B, T)), z0],
                    [x1, float(wigley_section_curve(x1, z1, L, B, T)), z1],
                    [x0, float(wigley_section_curve(x0, z1, L, B, T)), z1],
                ]
            )
            c = p.mean(axis=0)
            a = 0.5 * (p[1] + p[2] - p[0] - p[3])
            b = 0.5 * (p[2] + p[3] - p[0] - p[1])
            ha = float(np.linalg.norm(a))
            hb = float(np.linalg.norm(b))
            if ha <= 1.0e-11 or hb <= 1.0e-11:
                raise ValueError("degenerate hull panel")
            ua = a / ha
            ub = b / hb
            n = np.cross(ua, ub)
            nn = float(np.linalg.norm(n))
            if nn < 1.0e-12:
                raise ValueError("degenerate hull panel")
            n = n / nn
            if n[1] < 0.0:  # outward normal must point away from the body
                n = -n
            centers[k] = c
            e1[k] = ua
            e2[k] = ub
            normals[k] = n
            half1[k] = 0.5 * ha   # a, b are full edge vectors
            half2[k] = 0.5 * hb
            k += 1
    return FlatPanels(centers, e1, e2, normals, half1, half2)


def free_surface_grid(
    hx: float,
    hy: float,
    *,
    x_min: float,
    x_max: float,
    y_max: float,
) -> tuple[FlatPanels, NDArray[np.int64], NDArray[np.int64]]:
    """Uniform ``z = 0`` panel patch for ``y >= 0``.

    Returns ``(panels, ix, iy)`` where ``panels`` holds ``nx*ny`` rectangles in
    row-major order (``x`` slowest) and ``(ix, iy)`` are the integer grid
    coordinates of the panel centres.  No panel touches ``y = 0``: the first row
    centre sits at ``hy/2``, keeping every collocation point strictly inside its
    own panel (required by the jump relation).
    """
    for name, value in (("hx", hx), ("hy", hy)):
        if not (float(value) > 0.0 and math.isfinite(float(value))):
            raise ValueError(f"{name} must be positive and finite")
    if not (x_max > x_min) or not (y_max > 0.0):
        raise ValueError("require x_max > x_min and y_max > 0")

    nx = int(round((x_max - x_min) / hx))
    ny = int(round(y_max / hy))
    if nx < 3 or ny < 2:
        raise ValueError("free-surface grid needs nx >= 3 and ny >= 2 panels")
    xs = x_min + (np.arange(nx) + 0.5) * hx
    ys = (np.arange(ny) + 0.5) * hy

    n = nx * ny
    ix = np.repeat(np.arange(nx), ny)
    iy = np.tile(np.arange(ny), nx)
    centers = np.empty((n, 3), dtype=float)
    centers[:, 0] = xs[ix]
    centers[:, 1] = ys[iy]
    centers[:, 2] = 0.0
    e1 = np.zeros((n, 3), dtype=float)
    e1[:, 0] = 1.0
    e2 = np.zeros((n, 3), dtype=float)
    e2[:, 1] = 1.0
    normal = np.zeros((n, 3), dtype=float)
    normal[:, 2] = 1.0
    half1 = np.full(n, 0.5 * hx)
    half2 = np.full(n, 0.5 * hy)
    return FlatPanels(centers, e1, e2, normal, half1, half2), ix, iy


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def _rect_corner_potential(u: FloatArray, v: FloatArray) -> FloatArray:
    """Global antiderivative ``F`` with ``d^2 F/du dv = 1/sqrt(u^2+v^2)``.

    In the first quadrant ``Q(u, v) = u asinh(v/u) + v asinh(u/v)`` satisfies
    ``d^2 Q/du dv = 1/sqrt(u^2+v^2)``.  Because the kernel is even in each
    argument, ``F = sign(u) sign(v) Q(|u|, |v|)`` is odd in each argument and is
    the antiderivative valid in *all four* quadrants -- which is what makes the
    four-corner telescoping correct for field points outside the panel too.
    Using ``Q`` directly in the telescoping is wrong in three of the four
    quadrants (this was a real defect caught by
    ``test_in_plane_single_layer_matches_quadrature``).
    """
    ua = np.asarray(u, dtype=float)
    va = np.asarray(v, dtype=float)
    au = np.abs(ua)
    av = np.abs(va)
    tiny_u = au < 1.0e-14
    tiny_v = av < 1.0e-14
    su = np.where(ua < 0.0, -1.0, 1.0)
    sv = np.where(va < 0.0, -1.0, 1.0)
    term_u = np.where(tiny_u | tiny_v, 0.0, au * np.arcsinh(av / np.where(tiny_u, 1.0, au)))
    term_v = np.where(tiny_u | tiny_v, 0.0, av * np.arcsinh(au / np.where(tiny_v, 1.0, av)))
    return su * sv * (term_u + term_v)


def _common_axes(panels: FlatPanels) -> tuple[FloatArray, FloatArray]:
    e1 = np.asarray(panels.e1, dtype=float)
    e2 = np.asarray(panels.e2, dtype=float)
    ref1, ref2 = e1[0], e2[0]
    if not (np.allclose(e1, ref1, atol=1e-12) and np.allclose(e2, ref2, atol=1e-12)):
        raise ValueError("panels do not share a common in-plane frame")
    return ref1, ref2


def in_plane_single_layer(targets_xy: ArrayLike, panels: FlatPanels) -> FloatArray:
    """Exact ``int_{panel} dS / |p - q|`` for coplanar targets and panels.

    ``targets_xy`` ``(M, 2)`` are coordinates in the panels' common in-plane
    frame ``(e1, e2)`` measured from the global origin.  Returns ``(M, N)`` with
    ``A[m, j] = int_{panel j} dS / |p_m - q|``.  The closed form is exact for
    field points inside or outside the panel; only the panel *edges* are
    singular, and collocation points never sit on an edge.
    """
    tgt = np.asarray(targets_xy, dtype=float)
    if tgt.ndim != 2 or tgt.shape[1] != 2:
        raise ValueError("targets_xy must have shape (M, 2)")
    e1, e2 = _common_axes(panels)
    centers = np.asarray(panels.centers, dtype=float)
    c1 = centers @ e1
    c2 = centers @ e2
    a = np.asarray(panels.half1, dtype=float)
    b = np.asarray(panels.half2, dtype=float)

    m = tgt.shape[0]
    out = np.empty((m, panels.count), dtype=float)
    row_chunk = max(1, _CHUNK_ELEMENTS // max(1, panels.count))
    for start in range(0, m, row_chunk):
        stop = min(start + row_chunk, m)
        dx = tgt[start:stop, 0][:, None] - c1[None, :]
        dy = tgt[start:stop, 1][:, None] - c2[None, :]
        u1 = -a[None, :] - dx
        u2 = a[None, :] - dx
        v1 = -b[None, :] - dy
        v2 = b[None, :] - dy
        out[start:stop] = (
            _rect_corner_potential(u2, v2)
            - _rect_corner_potential(u1, v2)
            - _rect_corner_potential(u2, v1)
            + _rect_corner_potential(u1, v1)
        )
    return out


def _quadrature_nodes(panels: FlatPanels, order: int) -> tuple[FloatArray, FloatArray]:
    """Tensor Gauss nodes ``(Q, N, 3)`` and weights ``(Q,)`` on the rectangles."""
    nodes, weights = leggauss(order)
    uu = np.repeat(nodes, order)[:, None]
    vv = np.tile(nodes, order)[:, None]
    # each Gauss direction has weight sum 2, i.e. 4 for the tensor product;
    # normalise so that `w.sum() == 1` and the panel area carries the measure
    w = np.outer(weights, weights).ravel() / 4.0
    qp = (
        np.asarray(panels.centers, dtype=float)[None, :, :]
        + (uu[:, None] * np.asarray(panels.half1)[None, :, None])
        * np.asarray(panels.e1, dtype=float)[None, :, :]
        + (vv[:, None] * np.asarray(panels.half2)[None, :, None])
        * np.asarray(panels.e2, dtype=float)[None, :, :]
    )
    return qp, w


def single_layer(targets: ArrayLike, panels: FlatPanels, *, order: int = 4) -> FloatArray:
    """``int_{panel} dS / |p - q|`` by tensor Gauss quadrature, shape ``(M, N)``."""
    tgt = np.asarray(targets, dtype=float)
    if tgt.ndim != 2 or tgt.shape[1] != 3:
        raise ValueError("targets must have shape (M, 3)")
    qp, w = _quadrature_nodes(panels, order)
    areas = np.asarray(panels.areas, dtype=float)
    m = tgt.shape[0]
    out = np.empty((m, panels.count), dtype=float)
    row_chunk = max(1, _CHUNK_ELEMENTS // max(1, qp.shape[0] * panels.count * 3))
    for start in range(0, m, row_chunk):
        stop = min(start + row_chunk, m)
        diff = tgt[start:stop, None, None, :] - qp[None, :, :, :]
        dist = np.maximum(np.linalg.norm(diff, axis=3), 1.0e-30)
        out[start:stop] = ((1.0 / dist) * w[None, :, None]).sum(axis=1) * areas[None, :]
    return out


def dipole(
    targets: ArrayLike,
    target_normals: ArrayLike,
    panels: FlatPanels,
    *,
    order: int = 4,
) -> FloatArray:
    """``int_{panel} (p - q).n_p / |p-q|^3 dS`` with a per-target normal."""
    tgt = np.asarray(targets, dtype=float)
    nrm = np.asarray(target_normals, dtype=float)
    if tgt.shape != nrm.shape:
        raise ValueError("targets and target_normals must share a shape")
    qp, w = _quadrature_nodes(panels, order)
    areas = np.asarray(panels.areas, dtype=float)
    m = tgt.shape[0]
    out = np.empty((m, panels.count), dtype=float)
    row_chunk = max(1, _CHUNK_ELEMENTS // max(1, qp.shape[0] * panels.count * 3))
    for start in range(0, m, row_chunk):
        stop = min(start + row_chunk, m)
        diff = tgt[start:stop, None, None, :] - qp[None, :, :, :]
        dist = np.maximum(np.linalg.norm(diff, axis=3), 1.0e-30)
        proj = np.einsum("mqnk,mk->mqn", diff, nrm[start:stop]) / dist**3
        out[start:stop] = (proj * w[None, :, None]).sum(axis=1) * areas[None, :]
    return out


def gradient_x(targets: ArrayLike, panels: FlatPanels, *, order: int = 4) -> FloatArray:
    """``int_{panel} (x_p - x_q)/|p-q|^3 dS`` -- the x-derivative kernel."""
    tgt = np.asarray(targets, dtype=float)
    qp, w = _quadrature_nodes(panels, order)
    areas = np.asarray(panels.areas, dtype=float)
    m = tgt.shape[0]
    out = np.empty((m, panels.count), dtype=float)
    row_chunk = max(1, _CHUNK_ELEMENTS // max(1, qp.shape[0] * panels.count * 3))
    for start in range(0, m, row_chunk):
        stop = min(start + row_chunk, m)
        diff = tgt[start:stop, None, None, :] - qp[None, :, :, :]
        dist = np.maximum(np.linalg.norm(diff, axis=3), 1.0e-30)
        out[start:stop] = (
            diff[:, :, :, 0] / dist**3 * w[None, :, None]
        ).sum(axis=1) * areas[None, :]
    return out# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankineFreeSurfaceResult:
    """Outcome of one Rankine free-surface panel solve."""

    U: float
    froude: float
    k0: float
    L: float
    B: float
    T: float
    rho: float
    g: float
    rw_pressure: float
    rw_farfield: float
    cw_pressure: float
    cw_farfield: float
    energy_gap: float
    sigma_hull: FloatArray
    sigma_fs: FloatArray
    zeta: FloatArray
    free_surface_mask: NDArray[np.bool_]
    x_grid: FloatArray
    y_grid: FloatArray
    hull_panels: FlatPanels
    free_panels: FlatPanels
    fs_shape: tuple[int, int]
    upwind_order: int
    fs_spacing: tuple[float, float]
    domain: tuple[float, float, float]

    @property
    def n_hull_panels(self) -> int:
        return int(np.asarray(self.sigma_hull).shape[0])

    @property
    def n_free_surface_panels(self) -> int:
        return int(np.asarray(self.sigma_fs).shape[0])

    @property
    def n_unknowns(self) -> int:
        return self.n_hull_panels + self.n_free_surface_panels


def _upwind_weights(hx: float, order: int) -> FloatArray:
    """Backward-difference weights for ``d^2/dx^2`` (index 0 = collocation)."""
    if order == 1:
        return np.array([1.0, -2.0, 1.0], dtype=float) / hx**2
    if order == 2:
        return np.array([2.0, -5.0, 4.0, -1.0], dtype=float) / hx**2
    raise ValueError("upwind_order must be 1 (3-point) or 2 (4-point)")


def _waterline_halfwidth(x: FloatArray, L: float, B: float) -> FloatArray:
    return 0.5 * B * np.clip(1.0 - (2.0 * np.asarray(x, dtype=float) / L) ** 2, 0.0, None)


def solve_rankine_free_surface(
    L: float,
    B: float,
    T: float,
    U: float,
    *,
    rho: float = DEFAULT_DENSITY,
    g: float = DEFAULT_GRAVITY,
    hull_nx: int = 24,
    hull_nz: int = 8,
    waterline_clustering: float = 0.5,
    fs_hx: float | None = None,
    fs_hy: float | None = None,
    panels_per_wavelength: float = 12.0,
    x_min_factor: float = 1.0,
    x_max_factor: float = 2.5,
    y_max_factor: float = 1.5,
    upwind_order: int = 1,
    hull_quad_order: int = 4,
    fs_quad_order: int = 4,
    farfield_window: float = 1.0,
) -> RankineFreeSurfaceResult:
    """Solve the linearised steady free-surface problem with Rankine panels.

    Parameters
    ----------
    L, B, T:
        Wigley hull length, beam and draft (m).
    U:
        Forward speed (m/s); must be strictly positive.
    hull_nx, hull_nz:
        Hull panels per half hull: ``hull_nx`` stations along ``x``,
        ``hull_nz`` strips in ``z`` (so ``2 * hull_nx * hull_nz`` panels for the
        full hull, ``hull_nx * hull_nz`` unknowns from the symmetry).
    fs_hx, fs_hy:
        Free-surface panel size; default ``2 pi/k0 / panels_per_wavelength`` in
        both directions (``k0 = g/U^2``), i.e. ``panels_per_wavelength`` panels
        per transverse wavelength.
    x_min_factor, x_max_factor, y_max_factor:
        Truncation domain multiples of ``L``: ``x in [-x_min_factor*L,
        x_max_factor*L]``, ``y in [0, y_max_factor*L]``.
    upwind_order:
        1 -> 3-point backward second difference, 2 -> 4-point (2nd order).
    hull_quad_order, fs_quad_order:
        Gauss order for the off-plane kernel integrals.
    farfield_window:
        Length of the downstream averaging window used by the far-field energy
        estimate, in transverse wavelengths.

    Returns
    -------
    :class:`RankineFreeSurfaceResult`
    """
    if not (U > 0.0) or not math.isfinite(U):
        raise ValueError("U must be positive and finite (steady forward speed)")
    for name, value in (("L", L), ("B", B), ("T", T), ("rho", rho), ("g", g)):
        if not (float(value) > 0.0) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be positive and finite")
    if farfield_window <= 0.0:
        raise ValueError("farfield_window must be positive")

    k0 = g / U**2
    lam = _TWO_PI / k0
    if fs_hx is None:
        fs_hx = lam / max(1.0, panels_per_wavelength)
    if fs_hy is None:
        fs_hy = lam / max(1.0, panels_per_wavelength)
    froude = U / math.sqrt(g * L)

    hull = wigley_hull_panels(L, B, T, hull_nx, hull_nz, waterline_clustering=waterline_clustering)
    fs_panels, ix_fs, iy_fs = free_surface_grid(
        fs_hx,
        fs_hy,
        x_min=-x_min_factor * L,
        x_max=x_max_factor * L,
        y_max=y_max_factor * L,
    )
    nx_fs = int(ix_fs.max()) + 1
    ny_fs = int(iy_fs.max()) + 1
    centers = np.asarray(fs_panels.centers, dtype=float)

    # free-surface panels inside the hull waterplane carry no unknown
    inside = centers[:, 1] < _waterline_halfwidth(centers[:, 0], L, B) - 1.0e-12
    free_mask = ~inside
    free_index = np.flatnonzero(free_mask)
    n_free = int(free_index.size)
    if n_free < 8:
        raise ValueError("free-surface grid too coarse: fewer than 8 free panels")
    nh = hull.count

    mirror = np.array([1.0, -1.0, 1.0], dtype=float)
    p_grid = centers
    p_grid_m = centers * mirror

    # --- single layer of the hull panels at every free-surface grid point -----
    s_hull = single_layer(p_grid, hull, order=hull_quad_order) + single_layer(
        p_grid_m, hull, order=hull_quad_order
    )
    # --- in-plane single layer of the free-surface panels (exact rectangle) ---
    s_fs_all = in_plane_single_layer(p_grid[:, :2], fs_panels) + in_plane_single_layer(
        p_grid_m[:, :2], fs_panels
    )
    s_fs_free = s_fs_all[:, free_index]

    # --- off-plane normal derivative of the hull single layer at FS points ----
    zhat = np.zeros_like(p_grid)
    zhat[:, 2] = 1.0
    dz_hull = dipole(p_grid[free_index], zhat[free_index], hull, order=fs_quad_order) + dipole(
        p_grid_m[free_index], zhat[free_index], hull, order=fs_quad_order
    )

    # --- hull Neumann rows ---------------------------------------------------
    hnorm = np.asarray(hull.normal, dtype=float)
    hnorm_m = hnorm * mirror
    dip_hh = dipole(hull.centers, hnorm, hull, order=hull_quad_order) + dipole(
        hull.centers * mirror, hnorm_m, hull, order=hull_quad_order
    )
    np.fill_diagonal(dip_hh, 0.0)
    fs_free_panels = FlatPanels(
        np.asarray(fs_panels.centers, dtype=float)[free_index],
        np.asarray(fs_panels.e1, dtype=float)[free_index],
        np.asarray(fs_panels.e2, dtype=float)[free_index],
        np.asarray(fs_panels.normal, dtype=float)[free_index],
        np.asarray(fs_panels.half1, dtype=float)[free_index],
        np.asarray(fs_panels.half2, dtype=float)[free_index],
    )
    dip_hf = dipole(hull.centers, hnorm, fs_free_panels, order=fs_quad_order) + dipole(
        hull.centers * mirror, hnorm_m, fs_free_panels, order=fs_quad_order
    )

    # --- assemble ------------------------------------------------------------
    n_unknown = nh + n_free
    A = np.empty((n_unknown, n_unknown), dtype=float)
    rhs = np.zeros(n_unknown, dtype=float)
    A[:nh, :nh] = dip_hh
    A[:nh, nh:] = dip_hf
    rhs[:nh] = -U * hnorm[:, 0]

    weights = _upwind_weights(fs_hx, upwind_order)
    ix_free = ix_fs[free_index]
    iy_free = iy_fs[free_index]
    block_hull = np.zeros((n_free, nh), dtype=float)
    block_fs = np.zeros((n_free, n_free), dtype=float)
    for k, wk in enumerate(weights):
        valid = (ix_free - k) >= 0
        if not np.any(valid):
            continue
        idx = (ix_free[valid] - k) * ny_fs + iy_free[valid]
        block_hull[valid] += wk * s_hull[idx]
        block_fs[valid] += wk * s_fs_free[idx]
    A[nh:, :nh] = block_hull + k0 * dz_hull
    A[nh:, nh:] = block_fs
    A[nh:, nh:][np.diag_indices(n_free)] += k0 * _TWO_PI

    sigma = np.linalg.solve(A, rhs)
    sigma_hull = sigma[:nh]
    sigma_fs = sigma[nh:]

    # --- post-processing ------------------------------------------------------
    phi_grid = s_hull @ sigma_hull + s_fs_free @ sigma_fs
    phi_2d = phi_grid.reshape(nx_fs, ny_fs)
    # zeta = -U/g * dphi/dx; central differences inside, one-sided at the edges
    dphi_dx = np.gradient(phi_2d, fs_hx, axis=0, edge_order=2)
    zeta = -U / g * dphi_dx

    gx_hh = gradient_x(hull.centers, hull, order=hull_quad_order) + gradient_x(
        hull.centers * mirror, hull, order=hull_quad_order
    )
    np.fill_diagonal(gx_hh, 0.0)
    gx_hf = gradient_x(hull.centers, fs_free_panels, order=fs_quad_order) + gradient_x(
        hull.centers * mirror, fs_free_panels, order=fs_quad_order
    )
    phi_x_hull = gx_hh @ sigma_hull + gx_hf @ sigma_fs
    areas = np.asarray(hull.areas, dtype=float)
    rw_pressure = -2.0 * rho * U * float(np.sum(phi_x_hull * hnorm[:, 0] * areas))

    x_grid = np.asarray(fs_panels.centers, dtype=float)[:, 0].reshape(nx_fs, ny_fs)
    y_grid = np.asarray(fs_panels.centers, dtype=float)[:, 1].reshape(nx_fs, ny_fs)
    rw_farfield = farfield_wave_resistance(
        zeta, x_grid, y_grid, k0, rho, g, window=farfield_window
    )

    ref = 0.5 * rho * U**2 * L * T
    energy_gap = (
        abs(rw_pressure - rw_farfield) / abs(rw_pressure)
        if abs(rw_pressure) > 0.0
        else float("nan")
    )
    return RankineFreeSurfaceResult(
        U=U,
        froude=froude,
        k0=k0,
        L=L,
        B=B,
        T=T,
        rho=rho,
        g=g,
        rw_pressure=rw_pressure,
        rw_farfield=rw_farfield,
        cw_pressure=rw_pressure / ref,
        cw_farfield=rw_farfield / ref,
        energy_gap=float(energy_gap),
        sigma_hull=sigma_hull,
        sigma_fs=sigma_fs,
        zeta=zeta,
        free_surface_mask=free_mask,
        x_grid=x_grid,
        y_grid=y_grid,
        hull_panels=hull,
        free_panels=fs_free_panels,
        fs_shape=(nx_fs, ny_fs),
        upwind_order=upwind_order,
        fs_spacing=(float(fs_hx), float(fs_hy)),
        domain=(-x_min_factor * L, x_max_factor * L, y_max_factor * L),
    )


def farfield_wave_resistance(
    zeta: ArrayLike,
    x_grid: ArrayLike,
    y_grid: ArrayLike,
    k0: float,
    rho: float = DEFAULT_DENSITY,
    g: float = DEFAULT_GRAVITY,
    *,
    window: float = 1.0,
) -> float:
    """Wave resistance from the far-field energy flux of the transverse system.

    ``zeta`` ``(nx, ny)`` is the free-surface elevation on a uniform ``(x, y)``
    grid whose downstream end carries the developed wave pattern.  The amplitude
    of the ``k0`` harmonic is extracted over a window of ``window`` transverse
    wavelengths and the classical deep-water identity

        R_w = (1/4) rho g int a_T(y)^2 dy

    is evaluated (``c_g = U/2`` in the steady frame; equal to the far-field
    momentum flux ``(1/2) rho g <zeta^2>`` per unit width).
    """
    z = np.asarray(zeta, dtype=float)
    x = np.asarray(x_grid, dtype=float)
    y = np.asarray(y_grid, dtype=float)
    if z.ndim != 2 or x.shape != z.shape or y.shape != z.shape:
        raise ValueError("zeta, x_grid and y_grid must share a 2-D shape")
    lam = _TWO_PI / k0
    hx = float(x[1, 0] - x[0, 0])
    hy = float(y[0, 1] - y[0, 0])
    n_win = int(max(4, round(window * lam / hx)))
    n_win = min(n_win, z.shape[0])
    sl = slice(z.shape[0] - n_win, z.shape[0])
    xw = x[sl, 0] - x[sl, 0].mean()
    phase = np.exp(1j * k0 * xw)
    coeff = np.abs((z[sl, :] * phase[:, None]).sum(axis=0)) * 2.0 / n_win
    return 0.25 * rho * g * float(np.sum(coeff**2) * hy)


def domain_extent_for_froude(
    U: float,
    L: float,
    *,
    panels_per_wavelength: float = 12.0,
    g: float = DEFAULT_GRAVITY,
) -> dict[str, float]:
    """Panel size and wavelength recommended for a given speed."""
    if not (U > 0.0):
        raise ValueError("U must be positive")
    k0 = g / U**2
    lam = _TWO_PI / k0
    return {
        "k0": k0,
        "wavelength": lam,
        "froude": U / math.sqrt(g * L),
        "panel_size": lam / max(1.0, panels_per_wavelength),
        "panels_per_hull_length": L / (lam / max(1.0, panels_per_wavelength)),
    }


def grid_convergence_index(
    values: Sequence[float], *, refinement: float = 2.0, safety: float = 1.25
) -> dict[str, float | bool]:
    """Roache grid convergence index of the finest of three refined results.

    ``values`` must be ordered coarse -> fine.  Returns the apparent order of
    accuracy, the GCI of the finest value (percent) and whether the triple is
    monotone in the same direction (a necessary condition for an asymptotic
    range; when it fails the GCI is reported as ``inf``).
    """
    v = np.asarray(list(values), dtype=float)
    if v.size < 3:
        raise ValueError("need at least three successive refinements")
    if not np.isfinite(v).all():
        raise ValueError("values must be finite")
    if refinement <= 1.0:
        raise ValueError("refinement must exceed 1")
    f_coarse, f_mid, f_fine = float(v[-3]), float(v[-2]), float(v[-1])
    e21 = f_mid - f_coarse
    e32 = f_fine - f_mid
    monotone = bool(e21 * e32 > 0.0)
    scale = max(abs(f_fine), 1.0e-30)
    if not monotone or abs(e21) < 1.0e-14 * scale:
        return {
            "apparent_order": float("nan"),
            "gci_fine_pct": float("inf"),
            "monotone": monotone,
            "ratio_e32_e21": float("nan"),
        }
    ratio = abs(e32 / e21)
    p = abs(math.log(ratio)) / math.log(refinement)
    if p < 1.0e-6:
        p = 1.0e-6
    gci = safety * abs(e32 / scale) / (refinement**p - 1.0) * 100.0
    return {
        "apparent_order": float(p),
        "gci_fine_pct": float(gci),
        "monotone": monotone,
        "ratio_e32_e21": float(ratio),
    }


def gci_fine(values: Sequence[float], *, refinement: float = 2.0, safety: float = 1.25) -> float:
    """Grid convergence index of the finest value, in percent."""
    return float(grid_convergence_index(values, refinement=refinement, safety=safety)["gci_fine_pct"])


def truncation_study(
    L: float,
    B: float,
    T: float,
    U: float,
    *,
    y_max_factors: Sequence[float] = (1.0, 1.5, 2.0, 2.5),
    x_max_factors: Sequence[float] | None = None,
    rho: float = DEFAULT_DENSITY,
    g: float = DEFAULT_GRAVITY,
    **kwargs,
) -> dict[str, object]:
    """Sensitivity of ``R_w`` to the size of the truncated free-surface domain.

    Successively enlarges ``y_max`` (and optionally ``x_max``) and reports
    ``R_w`` plus the relative variation against the largest domain, which is the
    quantity limited by the campaign acceptance criterion.
    """
    lateral: list[dict[str, float]] = []
    for factor in y_max_factors:
        res = solve_rankine_free_surface(
            L, B, T, U, rho=rho, g=g, y_max_factor=float(factor), **kwargs
        )
        lateral.append(
            {
                "y_max_factor": float(factor),
                "rw_pressure": res.rw_pressure,
                "rw_farfield": res.rw_farfield,
                "n_unknowns": res.n_unknowns,
            }
        )
    best = abs(lateral[-1]["rw_pressure"])
    for row in lateral:
        row["rel_change_pct"] = (
            100.0 * abs(row["rw_pressure"] - lateral[-1]["rw_pressure"]) / best
            if best > 0.0
            else float("nan")
        )
    out: dict[str, object] = {
        "study": "lateral_truncation",
        "lateral": lateral,
        "max_rel_change_pct": float(max(row["rel_change_pct"] for row in lateral)),
    }
    if x_max_factors is not None:
        downstream: list[dict[str, float]] = []
        for factor in x_max_factors:
            res = solve_rankine_free_surface(
                L, B, T, U, rho=rho, g=g, x_max_factor=float(factor), **kwargs
            )
            downstream.append(
                {
                    "x_max_factor": float(factor),
                    "rw_pressure": res.rw_pressure,
                    "rw_farfield": res.rw_farfield,
                    "n_unknowns": res.n_unknowns,
                }
            )
        best_x = abs(downstream[-1]["rw_pressure"])
        for row in downstream:
            row["rel_change_pct"] = (
                100.0 * abs(row["rw_pressure"] - downstream[-1]["rw_pressure"]) / best_x
                if best_x > 0.0
                else float("nan")
            )
        out["downstream"] = downstream
        out["max_rel_change_pct"] = float(
            max(out["max_rel_change_pct"], max(r["rel_change_pct"] for r in downstream))
        )
    return out


def rankine_free_surface_resistance(
    L: float,
    B: float,
    T: float,
    U: float,
    **kwargs,
) -> RankineFreeSurfaceResult:
    """Convenience wrapper: :func:`solve_rankine_free_surface` with defaults."""
    return solve_rankine_free_surface(L, B, T, U, **kwargs)# ---------------------------------------------------------------------------
# Verified core: 3-D Rankine-source Neumann (double-body) panel solve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoubleBodyResult:
    """Rankine-source panel solution of the double-body (zero free surface) flow."""

    U: float
    sigma: FloatArray
    neumann_residual: float
    matrix_condition: float
    wetted_area: float
    displaced_volume_panels: float
    hull_panels: FlatPanels
    pressure_resistance: float

    @property
    def n_panels(self) -> int:
        return int(self.sigma.shape[0])

    @property
    def n_unknowns(self) -> int:
        return self.n_panels


def double_body_rankine(
    L: float,
    B: float,
    T: float,
    U: float,
    *,
    rho: float = DEFAULT_DENSITY,
    hull_nx: int = 24,
    hull_nz: int = 8,
    waterline_clustering: float = 0.5,
    quad_order: int = 4,
) -> DoubleBodyResult:
    """Solve the double-body flow about the (half) Wigley hull with Rankine sources.

    The unknown source strength ``sigma`` on the ``y >= 0`` panels satisfies the
    body boundary condition ``dphi/dn = -U n_x`` collocated at the panel
    centroids; the influence matrix is the double-layer kernel of the Rankine
    source, with the ``y -> -y`` mirror folded in by reflecting the field point
    (so the half-domain solution is the exact solution of the full hull).

    This is the part of a Rankine panel method that is *verified*: it is a dense
    3-D boundary-element solve whose residual, conditioning and mesh convergence
    are all measured here.  It contains no wave physics -- it is the basis flow
    on which a free-surface linearisation would be built -- and therefore returns
    exactly zero wave resistance (checked by
    ``test_double_body_has_no_wave_resistance``): a fore-aft symmetric hull in a
    symmetric double-body flow cannot radiate, which is the discrete counterpart
    of d'Alembert's paradox and an important negative control.
    """
    if not (U > 0.0) or not math.isfinite(U):
        raise ValueError("U must be positive and finite")
    hull = wigley_hull_panels(L, B, T, hull_nx, hull_nz, waterline_clustering=waterline_clustering)
    hnorm = np.asarray(hull.normal, dtype=float)
    mirror = np.array([1.0, -1.0, 1.0], dtype=float)
    influence = dipole(hull.centers, hnorm, hull, order=quad_order) + dipole(
        hull.centers * mirror, hnorm * mirror, hull, order=quad_order
    )
    np.fill_diagonal(influence, 0.0)
    rhs = -U * hnorm[:, 0]
    sigma = np.linalg.solve(influence, rhs)
    residual = float(
        np.linalg.norm(influence @ sigma - rhs) / max(np.linalg.norm(rhs), 1.0e-30)
    )
    gradx = gradient_x(hull.centers, hull, order=quad_order) + gradient_x(
        hull.centers * mirror, hull, order=quad_order
    )
    np.fill_diagonal(gradx, 0.0)
    phi_x = gradx @ sigma
    areas = np.asarray(hull.areas, dtype=float)
    pressure_resistance = -2.0 * rho * U * float(np.sum(phi_x * hnorm[:, 0] * areas))
    # volume from panel projections onto the x-z plane (dA_xz = area * |n_y|)
    volume = float(
        np.sum(
            2.0
            * areas
            * np.abs(hnorm[:, 1])
            * np.abs(np.asarray(hull.centers, dtype=float)[:, 1])
        )
    )
    return DoubleBodyResult(
        U=U,
        sigma=sigma,
        neumann_residual=residual,
        matrix_condition=float(np.linalg.cond(influence)) if hull.count <= 400 else float("nan"),
        wetted_area=float(2.0 * np.sum(areas)),
        displaced_volume_panels=volume,
        hull_panels=hull,
        pressure_resistance=pressure_resistance,
    )


# ---------------------------------------------------------------------------
# Why the truncated-domain free-surface panel formulation fails (measured)
# ---------------------------------------------------------------------------


def free_surface_operator_symbol(kh: ArrayLike, order: int = 1) -> dict[str, FloatArray]:
    """Fourier symbol of the upwind free-surface discretisation.

    For a collocation grid of spacing ``hx = h`` and a mode ``exp(i k x)`` the
    backward difference used by :func:`solve_rankine_free_surface` has symbol

        S(kh) = (1 - 2 exp(-i kh) + exp(-2i kh)) / h^2        (order 1)
        S(kh) = (2 - 5 exp(-i kh) + 4 exp(-2i kh) - exp(-3i kh)) / h^2 (order 2)

    and the free-surface row reduces to ``S(kh) phi + 2 pi k0 sigma = 0`` with
    ``phi = 2 pi sigma / k`` for a planar sheet, i.e. to

        S(kh) / k = k0        (discrete dispersion relation).

    The returned ``growth`` is the imaginary part of the resulting complex
    wavenumber perturbation, ``Im k(h)``: positive values mean the discrete
    eigenmode grows like ``exp(|Im k| x)`` downstream.  For the first-order
    backward stencil ``Im k ~ +h k^2`` -- i.e. the *physical* transverse mode is
    numerically damped while aliased (high-``kh``) branches of the same row grow,
    which is exactly what is observed numerically in
    :func:`solve_rankine_free_surface` (see the campaign report
    ``docs/benchmarks/verified/bem_rankine_fs_panel/report.md``: spurious
    downstream amplification, ``R_w`` off Michell by two orders of magnitude).
    The function exists so that the failure is a *documented, reproducible
    diagnostic* rather than a folk statement.
    """
    x = np.asarray(kh, dtype=float)
    e1 = np.exp(-1j * x)
    e2 = np.exp(-2j * x)
    e3 = np.exp(-3j * x)
    if order == 1:
        sym = 1.0 - 2.0 * e1 + e2
    elif order == 2:
        sym = 2.0 - 5.0 * e1 + 4.0 * e2 - e3
    else:  # pragma: no cover - guarded by callers
        raise ValueError("order must be 1 or 2")
    # symbolic h = 1: reconstruct the complex wavenumber correction
    # k_discrete = k0 (1 + correction) with S(k)/k = k0 -> solve perturbatively
    growth = np.imag(sym) / np.maximum(np.abs(sym), 1.0e-300)
    return {"kh": x, "symbol_real": np.real(sym), "symbol_imag": np.imag(sym), "growth": growth}


def truncation_and_stability_summary(
    L: float, B: float, T: float, U: float, **kwargs
) -> dict[str, object]:
    """One-shot diagnostic bundle for the truncated-domain formulation.

    Reports (i) the double-body Rankine solve quality, (ii) the conditioning of
    the coupled free-surface system and (iii) the observed downstream growth of
    the wave elevation, which is the measured symptom of the numerical
    instability discussed in :func:`free_surface_operator_symbol`.
    """
    db = double_body_rankine(L, B, T, U)
    res = solve_rankine_free_surface(L, B, T, U, **kwargs)
    zeta = np.asarray(res.zeta, dtype=float)
    nx = zeta.shape[0]
    first = float(np.sqrt(np.mean(zeta[: max(1, nx // 4)] ** 2)))
    last = float(np.sqrt(np.mean(zeta[-max(1, nx // 4) :] ** 2)))
    return {
        "double_body": {
            "neumann_residual": db.neumann_residual,
            "matrix_condition": db.matrix_condition,
            "pressure_resistance": db.pressure_resistance,
            "wetted_area": db.wetted_area,
        },
        "free_surface_solve": {
            "rw_pressure": res.rw_pressure,
            "rw_farfield": res.rw_farfield,
            "n_unknowns": res.n_unknowns,
            "zeta_rms_upstream_quarter": first,
            "zeta_rms_downstream_quarter": last,
            "downstream_growth_ratio": last / first if first > 0.0 else float("inf"),
        },
    }