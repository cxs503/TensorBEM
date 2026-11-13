"""3D steady Neumann-Kelvin panel method (Rankine sources + Havelock free-surface kernel).

Motivation
----------
The repository already contains thin-ship (Michell/Kochin) wave-resistance
machinery and an axisymmetric ("cylindrical") steady kernel in
:mod:`tensorbem.neumann_kelvin`.  Neither is a *panel* method on the real
three-dimensional hull surface, and the vectorised kernel of the latter uses
the surrogate ``exp(-kappa|z|) K_0(kappa R)`` instead of the Havelock
integral, so it does not satisfy the steady free-surface condition.

This module implements the classical steady Neumann-Kelvin (NK) formulation
directly:

* the unknown is a source strength ``sigma`` on the wetted hull panels;
* the kernel is the steady deep-water Green function ``G = G_R + G_W`` with

  .. math::
     G_R = \\frac{1}{4\\pi r} + \\frac{1}{4\\pi r_1},\\qquad
     G_W = \\frac{1}{2\\pi^2}\\int_{-\\pi/2}^{\\pi/2}
           \\left[e^{-\\beta}\\mathrm{Ei}(\\beta) + C\\,e^{-\\beta}\\right]d\\theta,

  .. math::
     \\beta = \\frac{\\kappa\\,(a - i D)}{\\cos^2\\theta},\\quad
     a = -(z+\\zeta),\\quad D = (x-\\xi)\\cos\\theta + (y-\\eta)\\sin\\theta,

  with ``kappa = g/U^2``.  The derivation is the double Fourier transform of
  ``W_xx + kappa W_z = -R_xx`` on ``z = 0``; the polar form of the wavenumber
  integral has a simple pole at ``k = kappa sec^2 theta`` whose principal value
  is the exponential integral and whose indentation gives the radiation term
  ``C = -i pi q``.  The choice of ``q`` (``radiation="downstream"``) is
  *verified* numerically by :func:`verify_free_surface_condition`,
  :func:`verify_radiation_direction` and :func:`verify_laplace`.
* the body boundary condition ``dphi/dn = U n_x`` is imposed in the
  single-layer (indirect) form with the jump ``-sigma/2``;
* output quantities are the near-field pressure integral
  (:func:`wave_resistance_near` path, returned by :func:`solve_nk_panel`) and
  the free-surface elevation (:func:`free_surface_elevation`), a direct
  by-product via ``zeta = (U/g) phi_x``.

All kernel gradients are analytic (``d beta / d(x,y,z)`` is constant per theta
node), so no finite differencing is used inside the solver.

Sign conventions: uniform stream ``(U, 0, 0)``, total potential
``Phi = -U x + phi``, ``n`` the outward normal of the body (into the fluid).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray
from scipy.special import expi

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_TWO_PI2 = 2.0 * math.pi**2
#: overall normalisation of the wave part; determined by the free-surface
#: condition verification (see ``verify_free_surface_condition``): the literal
#: Fourier inversion of the polar form leaves an overall factor 2 that comes
#: from the source/image doubling of the Rankine part at ``z = 0``.
_WAVE_SCALE = 0.5
_FOUR_PI = 4.0 * math.pi
_G = 9.80665
_RHO = 1025.0


# ---------------------------------------------------------------------------
# theta quadrature
# ---------------------------------------------------------------------------


def theta_nodes(n_theta: int = 48) -> tuple[FloatArray, FloatArray]:
    """Gauss-Legendre nodes/weights on ``(-pi/2, pi/2)``."""
    if n_theta < 8:
        raise ValueError("n_theta must be >= 8")
    x, w = leggauss(int(n_theta))
    return 0.5 * math.pi * x, 0.5 * math.pi * w



def _exp_ei(beta: ComplexArray) -> ComplexArray:
    """``exp(-beta) * Ei(beta)`` for ``Re(beta) > 0`` (asymptotics for |beta| large).

    ``scipy.special.expi`` overflows for ``Re(beta)`` beyond ~700, but the
    product ``exp(-beta) Ei(beta)`` decays like ``1/beta``; the asymptotic
    series ``sum_k k!/beta^(k+1)`` is used instead.  Truncation error at
    ``|beta| = 60`` with 16 terms is ``~16!/60^17 ~ 1e-17``.
    """
    out = np.zeros(beta.shape, dtype=complex)
    small = np.abs(beta) < 60.0
    if np.any(small):
        b = beta[small]
        out[small] = np.exp(-b) * expi(b)
    big = ~small
    if np.any(big):
        b = beta[big]
        acc = np.zeros(b.shape, dtype=complex)
        for k in range(16, 0, -1):
            acc = 1.0 + (k / b) * acc
        out[big] = acc / b
    return out


# ---------------------------------------------------------------------------
# steady Neumann-Kelvin wave kernel
# ---------------------------------------------------------------------------


def _radiation_mask(D: FloatArray, radiation: str) -> FloatArray:
    """Residue indicator for the wavenumber pole.

    ``D = (x-xi) cos(theta) + (y-eta) sin(theta)`` is the projection of the
    source-to-field vector on the wave direction.  For a stream flowing in
    ``+x`` the wake lies at ``x < 0``, so the physical choice is
    ``radiation="downstream"`` -> residue retained where ``D < 0``.
    ``"upstream"`` is the unphysical mirror image and ``"none"`` is the pure
    principal value (both are kept as numerical controls).
    """
    if radiation == "none":
        return np.zeros_like(D)
    if radiation == "downstream":
        return (D < 0.0).astype(float)
    if radiation == "upstream":
        return (D > 0.0).astype(float)
    if radiation == "halfplane":
        # classical Havelock choice: the residue is retained for every
        # direction of the downstream half-plane (cos(theta) > 0), which is
        # how the theta integral is already restricted
        return np.ones_like(D)
    raise ValueError(f"unknown radiation mode {radiation!r}")


def nk_wave_kernel(
    targets: ArrayLike,
    sources: ArrayLike,
    kappa: float,
    n_theta: int = 48,
    radiation: str = "downstream",
    sign: float = -1.0,
    chunk: int = 96,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Wave part ``G_W`` of the steady Green function and its gradient.

    Returns ``(G, dG/dx, dG/dy, dG/dz)``, derivatives with respect to the
    *field* point, each of shape ``(M, N)``.

    The double wavenumber integral is reduced to the half range
    ``theta in (-pi/2, pi/2)`` by pairing ``theta`` with ``theta + pi``: the
    principal-value parts are complex conjugates and the two residue terms
    combine into ``-i pi [q e^{-beta} + (1-q) e^{-conj(beta)}]``, which yields
    the compact real form used below.
    """
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    tgt = np.atleast_2d(np.asarray(targets, dtype=float))
    src = np.atleast_2d(np.asarray(sources, dtype=float))
    if tgt.shape[1] != 3 or src.shape[1] != 3:
        raise ValueError("targets and sources must be (M, 3) and (N, 3)")

    th, w = theta_nodes(n_theta)
    c = np.cos(th)
    sn = np.sin(th)
    c2 = c * c

    M, N = tgt.shape[0], src.shape[0]
    G = np.zeros((M, N))
    Gx = np.zeros((M, N))
    Gy = np.zeros((M, N))
    Gz = np.zeros((M, N))

    dbx = (-1j * kappa * c / c2)[None, None, :]
    dby = (-1j * kappa * sn / c2)[None, None, :]
    dbz = (-kappa / c2)[None, None, :]
    pref = _WAVE_SCALE / _TWO_PI2
    pi = math.pi

    for i0 in range(0, M, chunk):
        sl = slice(i0, min(i0 + chunk, M))
        dx = tgt[sl, 0][:, None] - src[None, :, 0]
        dy = tgt[sl, 1][:, None] - src[None, :, 1]
        a = -(tgt[sl, 2][:, None] + src[None, :, 2])
        D = dx[:, :, None] * c + dy[:, :, None] * sn
        beta = (kappa / c2) * (a[:, :, None] - 1j * D)
        re = np.maximum(beta.real, 1e-14)
        beta = re + 1j * beta.imag
        q = _radiation_mask(D, radiation)
        rad = sign * (2.0 * q - 1.0)
        ee = _exp_ei(beta)
        e = np.exp(-beta)
        # integrand: kappa sec^2(theta) * [2 Re(e^{-beta} Ei(beta) - 1/beta)
        #            + (2q-1) pi Im(e^{-beta})]   (theta + theta+pi pairing)
        wgt = (kappa / c2)[None, None, :]
        F = 2.0 * (ee - 1.0 / beta) - 1j * rad * pi * e
        Fp = 2.0 * (-ee + 1.0 / beta + 1.0 / beta**2) + 1j * rad * pi * e
        G[sl] = pref * np.einsum("p,ijp->ij", w * wgt.reshape(-1)[0], F.real) if False else pref * np.einsum("p,ijp->ij", w, (wgt * F).real)
        Gx[sl] = pref * np.einsum("p,ijp->ij", w, (wgt * Fp * dbx).real)
        Gy[sl] = pref * np.einsum("p,ijp->ij", w, (wgt * Fp * dby).real)
        Gz[sl] = pref * np.einsum("p,ijp->ij", w, (wgt * Fp * dbz).real)
    return G, Gx, Gy, Gz


# ---------------------------------------------------------------------------
# Rankine + image part
# ---------------------------------------------------------------------------


def _rankine_pair(
    field: FloatArray, src: FloatArray, mirror: bool = True
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Element-wise rankine+image kernel for paired point lists."""
    dx = field[:, 0] - src[:, 0]
    dy = field[:, 1] - src[:, 1]
    dz = field[:, 2] - src[:, 2]
    r = np.maximum(np.sqrt(dx * dx + dy * dy + dz * dz), 1e-300)
    G = 1.0 / (_FOUR_PI * r)
    inv3 = 1.0 / r**3
    Gx = -dx / _FOUR_PI * inv3
    Gy = -dy / _FOUR_PI * inv3
    Gz = -dz / _FOUR_PI * inv3
    if mirror:
        dzi = field[:, 2] + src[:, 2]
        ri = np.maximum(np.sqrt(dx * dx + dy * dy + dzi * dzi), 1e-300)
        G = G + 1.0 / (_FOUR_PI * ri)
        inv3i = 1.0 / ri**3
        Gx = Gx - dx / _FOUR_PI * inv3i
        Gy = Gy - dy / _FOUR_PI * inv3i
        Gz = Gz - dzi / _FOUR_PI * inv3i
    return G, Gx, Gy, Gz


def rankine_kernel(
    targets: ArrayLike,
    sources: ArrayLike,
    mirror: bool = True,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """``G_R = 1/(4 pi r) + 1/(4 pi r1)`` and its gradient."""
    tgt = np.atleast_2d(np.asarray(targets, dtype=float))
    src = np.atleast_2d(np.asarray(sources, dtype=float))
    dx = tgt[:, 0][:, None] - src[None, :, 0]
    dy = tgt[:, 1][:, None] - src[None, :, 1]
    dz = tgt[:, 2][:, None] - src[None, :, 2]
    r = np.sqrt(dx * dx + dy * dy + dz * dz)
    safe = np.maximum(r, 1e-300)
    G = np.where(r > 0.0, 1.0 / (_FOUR_PI * safe), 0.0)
    inv3 = np.where(r > 0.0, 1.0 / safe**3, 0.0)
    Gx = -dx / _FOUR_PI * inv3
    Gy = -dy / _FOUR_PI * inv3
    Gz = -dz / _FOUR_PI * inv3
    if mirror:
        dzi = tgt[:, 2][:, None] + src[None, :, 2]
        ri = np.sqrt(dx * dx + dy * dy + dzi * dzi)
        safei = np.maximum(ri, 1e-300)
        G = G + np.where(ri > 0.0, 1.0 / (_FOUR_PI * safei), 0.0)
        inv3i = np.where(ri > 0.0, 1.0 / safei**3, 0.0)
        Gx = Gx - dx / _FOUR_PI * inv3i
        Gy = Gy - dy / _FOUR_PI * inv3i
        Gz = Gz - dzi / _FOUR_PI * inv3i
    return G, Gx, Gy, Gz


def green_kernel(
    targets: ArrayLike,
    sources: ArrayLike,
    kappa: float,
    n_theta: int = 48,
    radiation: str = "downstream",
    sign: float = 1.0,
    mirror: bool = True,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Full steady NK Green function ``G_R + G_W`` and its gradient."""
    G0, Gx0, Gy0, Gz0 = rankine_kernel(targets, sources, mirror=mirror)
    Gw, Gxw, Gyw, Gzw = nk_wave_kernel(
        targets, sources, kappa, n_theta=n_theta, radiation=radiation, sign=sign
    )
    return G0 + Gw, Gx0 + Gxw, Gy0 + Gyw, Gz0 + Gzw


# ---------------------------------------------------------------------------
# panel geometry
# ---------------------------------------------------------------------------


@dataclass
class PanelMesh:
    """Structured quadrilateral panel mesh on a hull half-body (``y >= 0``)."""

    centroids: FloatArray
    normals: FloatArray
    areas: FloatArray
    corners: FloatArray  # (F, 4, 3)

    @property
    def n_panels(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def mean_edge(self) -> float:
        e1 = np.linalg.norm(self.corners[:, 1] - self.corners[:, 0], axis=1)
        e2 = np.linalg.norm(self.corners[:, 2] - self.corners[:, 1], axis=1)
        return float(np.mean(np.minimum(e1, e2)))

    @property
    def total_area(self) -> float:
        return float(np.sum(self.areas))


def _quad_from_corners(c: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Centroid, area and unit normal of quads ``(F, 4, 3)`` (Newell)."""
    nx = np.zeros(c.shape[0])
    ny = np.zeros(c.shape[0])
    nz = np.zeros(c.shape[0])
    for k in range(4):
        p = c[:, k, :]
        q = c[:, (k + 1) % 4, :]
        nx += (p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])
        ny += (p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])
        nz += (p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])
    n = np.stack([nx, ny, nz], axis=1)
    nn = np.linalg.norm(n, axis=1)
    nn = np.where(nn > 0.0, nn, 1.0)
    return np.mean(c, axis=1), 0.5 * nn, n / nn[:, None]


def wigley_half_panels(L: float, B: float, T: float, nx: int, nz: int) -> PanelMesh:
    """Quadrilateral panels of the Wigley half-body.

    ``y(x, z) = (B/2)(1 - (2x/L)^2)(1 - (z/T)^2)`` for ``z in [-T, 0]``;
    degenerate panels at the extreme ends are dropped.
    """
    if not (L > 0 and B > 0 and T > 0):
        raise ValueError("L, B, T must be positive")
    if nx < 4 or nz < 2:
        raise ValueError("nx >= 4 and nz >= 2 required")

    def half_breadth(x: float, z: float) -> float:
        return 0.5 * B * (1.0 - (2.0 * x / L) ** 2) * (1.0 - (z / T) ** 2)

    def surf_normal(x: float, z: float) -> FloatArray:
        fx = -(4.0 * B / L**2) * x * (1.0 - (z / T) ** 2)
        fz = -(B / T**2) * (1.0 - (2.0 * x / L) ** 2) * z
        n = np.array([-fx, 1.0, -fz], dtype=float)
        return n / np.linalg.norm(n)

    xs = np.linspace(-0.5 * L, 0.5 * L, nx + 1)
    zs = np.linspace(-T, 0.0, nz + 1)
    quads = []
    for i in range(nx):
        for j in range(nz):
            xq = [xs[i], xs[i + 1], xs[i + 1], xs[i]]
            zq = [zs[j], zs[j], zs[j + 1], zs[j + 1]]
            ys = [half_breadth(x, z) for x, z in zip(xq, zq)]
            if max(ys) < 1e-6 * B:
                continue
            quads.append(
                np.array([[xq[k], ys[k], zq[k]] for k in range(4)], dtype=float)
            )
    corners = np.array(quads)
    centroid, area, normal = _quad_from_corners(corners)
    ref = np.array([surf_normal(c[0], c[2]) for c in centroid])
    flip = np.einsum("ij,ij->i", normal, ref) < 0.0
    normal[flip] *= -1.0
    return PanelMesh(centroid, normal, area, corners)


def centreplane_panels(L: float, B: float, T: float, nx: int, nz: int) -> PanelMesh:
    """Thin-ship (Michell) centreplane panel sheet at ``y = 0``."""
    xs = np.linspace(-0.5 * L, 0.5 * L, nx + 1)
    zs = np.linspace(-T, 0.0, nz + 1)
    quads = []
    for i in range(nx):
        for j in range(nz):
            quads.append(
                np.array(
                    [
                        [xs[i], 0.0, zs[j]],
                        [xs[i + 1], 0.0, zs[j]],
                        [xs[i + 1], 0.0, zs[j + 1]],
                        [xs[i], 0.0, zs[j + 1]],
                    ]
                )
            )
    corners = np.array(quads)
    centroid, area, _ = _quad_from_corners(corners)
    normal = np.tile(np.array([0.0, 1.0, 0.0]), (corners.shape[0], 1))
    return PanelMesh(centroid, normal, area, corners)


# ---------------------------------------------------------------------------
# influence matrices
# ---------------------------------------------------------------------------


def _subcell_rule(corners: FloatArray, nsub: int = 3) -> tuple[FloatArray, FloatArray]:
    """Centroid rule on ``nsub x nsub`` bilinear sub-cells of one quad."""
    u = (np.arange(nsub) + 0.5) / nsub
    uu, vv = np.meshgrid(u, u, indexing="ij")
    uu = uu.ravel()
    vv = vv.ravel()
    Nu = 0.25 * (1 - uu) * (1 - vv)
    Nv = 0.25 * (1 + uu) * (1 - vv)
    Nw = 0.25 * (1 + uu) * (1 + vv)
    Ns = 0.25 * (1 - uu) * (1 + vv)
    pts = (
        Nu[:, None] * corners[0]
        + Nv[:, None] * corners[1]
        + Nw[:, None] * corners[2]
        + Ns[:, None] * corners[3]
    )
    quad_area = 0.5 * np.linalg.norm(
        np.cross(corners[1] - corners[0], corners[2] - corners[0])
        + np.cross(corners[2] - corners[0], corners[3] - corners[0])
    )
    return pts, np.full(pts.shape[0], quad_area / pts.shape[0])


def _panel_potential_self(corners: FloatArray, n_gauss: int = 8) -> float:
    """``int_panel 1/(4 pi r) dS`` evaluated at the panel centroid."""
    x, w = leggauss(n_gauss)
    uu, vv = np.meshgrid(x, x, indexing="ij")
    uu = uu.ravel()
    vv = vv.ravel()
    ww = (np.tile(w, n_gauss) * np.repeat(w, n_gauss))
    Nu = 0.25 * (1 - uu) * (1 - vv)
    Nv = 0.25 * (1 + uu) * (1 - vv)
    Nw = 0.25 * (1 + uu) * (1 + vv)
    Ns = 0.25 * (1 - uu) * (1 + vv)
    pts = (
        Nu[:, None] * corners[0]
        + Nv[:, None] * corners[1]
        + Nw[:, None] * corners[2]
        + Ns[:, None] * corners[3]
    )
    jac = 0.25 * float(np.linalg.norm(np.cross(corners[0] - corners[1], corners[2] - corners[3])))
    r = np.linalg.norm(pts - np.mean(corners, axis=0), axis=1)
    return float(np.sum(ww * jac / (_FOUR_PI * np.maximum(r, 1e-300))))


def rankine_influences(
    mesh: PanelMesh,
    mirror: bool = True,
    near_factor: float = 2.5,
    nsub: int = 3,
) -> tuple[FloatArray, FloatArray]:
    """Rankine (image included) matrices ``S`` (potential) and ``K`` (normal deriv.).

    The diagonal of ``K`` is set to zero: the flat-panel finite part of the
    own-panel normal derivative vanishes by symmetry, only the jump remains and
    is added by the caller.
    """
    cen = mesh.centroids
    area = mesh.areas
    normals = mesh.normals
    F = mesh.n_panels
    G, Gx, Gy, Gz = rankine_kernel(cen, cen, mirror=mirror)
    S = G * area[None, :]
    K = (
        Gx * normals[:, 0][:, None]
        + Gy * normals[:, 1][:, None]
        + Gz * normals[:, 2][:, None]
    ) * area[None, :]

    for j in range(F):
        S[j, j] = _panel_potential_self(mesh.corners[j])
        K[j, j] = 0.0

    h = np.sqrt(area)
    dist = np.linalg.norm(cen[:, None, :] - cen[None, :, :], axis=2)
    idx = np.argwhere(dist < near_factor * h[None, :])
    idx = idx[idx[:, 0] != idx[:, 1]]
    if idx.size:
        sp, sw, sidx = [], [], []
        for (i, j) in idx:
            pts, wts = _subcell_rule(mesh.corners[j], nsub=nsub)
            sp.append(pts)
            sw.append(wts)
            sidx.append(np.full(pts.shape[0], i * F + j, dtype=np.int64))
        sp = np.vstack(sp)
        sw = np.concatenate(sw)
        sidx = np.concatenate(sidx)
        field_idx = sidx // F
        Gp, Gxp, Gyp, Gzp = _rankine_pair(cen[field_idx], sp, mirror=mirror)
        cS = Gp * sw
        cK = (
            Gxp * normals[field_idx, 0]
            + Gyp * normals[field_idx, 1]
            + Gzp * normals[field_idx, 2]
        ) * sw
        S_near = np.bincount(sidx, weights=cS, minlength=F * F).reshape(F, F)
        K_near = np.bincount(sidx, weights=cK, minlength=F * F).reshape(F, F)
        mask = np.zeros((F, F), dtype=bool)
        mask[idx[:, 0], idx[:, 1]] = True
        S[mask] = S_near[mask]
        K[mask] = K_near[mask]
    return S, K


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------


@dataclass
class NKPanelResult:
    """Result of a steady Neumann-Kelvin panel solve."""

    kappa: float
    speed: float
    froude: float
    sigma: FloatArray
    phi_hull: FloatArray
    dphidx_hull: FloatArray
    wave_resistance: float
    cw: float
    s_ref: float
    n_panels: int
    bc_residual: float
    mesh: PanelMesh | None = field(default=None, repr=False)
    extra: dict = field(default_factory=dict)


def _grad_field(
    field: FloatArray,
    mesh: PanelMesh,
    kappa: float,
    n_theta: int,
    radiation: str,
    sign: float,
    mirror: bool = True,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Gradient influence matrices (field-point derivatives, unit source)."""
    _, Gx0, Gy0, Gz0 = rankine_kernel(field, mesh.centroids, mirror=mirror)
    _, Gxw, Gyw, Gzw = nk_wave_kernel(
        field, mesh.centroids, kappa, n_theta=n_theta, radiation=radiation, sign=sign
    )
    Mx = Gx0 + Gxw
    My = Gy0 + Gyw
    Mz = Gz0 + Gzw
    if mirror:
        src_m = mesh.centroids.copy()
        src_m[:, 1] *= -1.0
        _, Gx0m, Gy0m, Gz0m = rankine_kernel(field, src_m, mirror=False)
        _, Gxwm, Gywm, Gzwm = nk_wave_kernel(
            field, src_m, kappa, n_theta=n_theta, radiation=radiation, sign=sign
        )
        Mx = Mx + Gx0m + Gxwm
        My = My + Gy0m + Gywm
        Mz = Mz + Gz0m + Gzwm
    a = mesh.areas[None, :]
    return Mx * a, My * a, Mz * a


def solve_nk_panel(
    mesh: PanelMesh,
    U: float,
    rho: float = _RHO,
    g: float = _G,
    n_theta: int = 48,
    radiation: str = "downstream",
    sign: float = 1.0,
    near_factor: float = 2.5,
    nsub: int = 3,
    s_ref: float | None = None,
    L_ref: float | None = None,
) -> NKPanelResult:
    """Solve the steady NK single-layer problem on a half-body mesh.

    The symmetry plane is handled exactly by the mirror image, so only
    ``mesh.n_panels`` unknowns appear.
    """
    if U <= 0.0:
        raise ValueError("U must be positive")
    kappa = g / U**2
    cen = mesh.centroids
    area = mesh.areas
    normals = mesh.normals
    F = mesh.n_panels

    S, K = rankine_influences(mesh, mirror=True, near_factor=near_factor, nsub=nsub)
    Gw, Gxw, Gyw, Gzw = nk_wave_kernel(
        cen, cen, kappa, n_theta=n_theta, radiation=radiation, sign=sign
    )
    src_m = cen.copy()
    src_m[:, 1] *= -1.0
    Gwm, Gxwm, Gywm, Gzwm = nk_wave_kernel(
        cen, src_m, kappa, n_theta=n_theta, radiation=radiation, sign=sign
    )
    S = S + (Gw + Gwm) * area[None, :]
    K = K + (
        Gxw * normals[:, 0][:, None]
        + Gyw * normals[:, 1][:, None]
        + Gzw * normals[:, 2][:, None]
        + Gxwm * normals[:, 0][:, None]
        + Gywm * normals[:, 1][:, None]
        + Gzwm * normals[:, 2][:, None]
    ) * area[None, :]

    A = -0.5 * np.eye(F) + K
    rhs = U * normals[:, 0]
    sigma = np.linalg.solve(A, rhs)
    bc_res = float(np.linalg.norm(A @ sigma - rhs) / max(np.linalg.norm(rhs), 1e-30))

    phi = S @ sigma
    Mx, _, _ = _grad_field(cen, mesh, kappa, n_theta, radiation, sign, mirror=True)
    dphi_dx = Mx @ sigma
    Rw = -rho * U * float(np.sum(dphi_dx * normals[:, 0] * area))
    Sref = float(s_ref) if s_ref is not None else 2.0 * float(np.sum(area))
    Cw = Rw / (0.5 * rho * U**2 * Sref)
    Lref = float(L_ref) if L_ref is not None else 1.0
    return NKPanelResult(
        kappa=kappa,
        speed=U,
        froude=U / math.sqrt(g * Lref),
        sigma=sigma,
        phi_hull=phi,
        dphidx_hull=dphi_dx,
        wave_resistance=Rw,
        cw=Cw,
        s_ref=Sref,
        n_panels=F,
        bc_residual=bc_res,
        mesh=mesh,
        extra={"condition_number": float(np.linalg.cond(A))},
    )


# ---------------------------------------------------------------------------
# free-surface elevation (wave cuts)
# ---------------------------------------------------------------------------


def free_surface_elevation(
    points_xy: ArrayLike,
    mesh: PanelMesh,
    sigma: FloatArray,
    U: float,
    g: float = _G,
    n_theta: int = 48,
    radiation: str = "downstream",
    sign: float = 1.0,
) -> FloatArray:
    """Wave elevation ``zeta = (U/g) dphi/dx`` at free-surface points ``(M, 2)``."""
    xy = np.atleast_2d(np.asarray(points_xy, dtype=float))
    if xy.shape[1] != 2:
        raise ValueError("points_xy must be (M, 2)")
    kappa = g / U**2
    pts = np.column_stack([xy, np.zeros(xy.shape[0])])
    Mx, _, _ = _grad_field(pts, mesh, kappa, n_theta, radiation, sign, mirror=True)
    return (U / g) * (Mx @ sigma)


# ---------------------------------------------------------------------------
# far-field (Kochin) wave resistance
# ---------------------------------------------------------------------------


def kochin_amplitude(
    sigma: FloatArray,
    mesh: PanelMesh,
    kappa: float,
    thetas: ArrayLike,
    mirror: bool = True,
) -> ComplexArray:
    """``H(t) = int_S sigma exp(kappa sec^2 t (z + i(x cos t + y sin t))) dS``."""
    th = np.atleast_1d(np.asarray(thetas, dtype=float))
    c = np.cos(th)
    s = np.sin(th)
    sec2 = 1.0 / np.maximum(c * c, 1e-12)
    H = np.zeros(th.shape[0], dtype=complex)
    cen = mesh.centroids
    sa = sigma * mesh.areas
    for side in ([1.0, -1.0] if mirror else [1.0]):
        phase = kappa * sec2[:, None] * (
            cen[:, 2][None, :]
            + 1j * (cen[:, 0][None, :] * c[:, None] + side * cen[:, 1][None, :] * s[:, None])
        )
        H += np.sum(sa[None, :] * np.exp(phase), axis=1)
    return H


def kochin_prefactor(kappa: float, g: float = _G) -> float:
    """Exact far-field (Havelock/Kochin) prefactor ``1/(2 pi U^2) = kappa/(2 pi g)``.

    See :func:`wave_resistance_far` for the derivation.  The quantity is *not*
    free: it is fixed by the residue of the steady kernel's wavenumber pole and
    carries the units of ``1/U^2`` that the ``kappa``-normalised Kochin
    amplitude ``H`` (units ``m^3/s``) leaves open.
    """
    if kappa <= 0.0 or g <= 0.0:
        raise ValueError("kappa and g must be positive")
    return kappa / (2.0 * math.pi * g)


def wave_resistance_far(
    sigma: FloatArray,
    mesh: PanelMesh,
    kappa: float,
    rho: float = _RHO,
    g: float = _G,
    n_theta: int = 7201,
    mirror: bool = True,
    prefactor: float | None = None,
) -> tuple[float, FloatArray, ComplexArray]:
    """Far-field (Havelock/Kochin) resistance of a steady source distribution.

    .. math::

        R_w = \frac{\rho\kappa^2}{2\pi}\int_{-\pi/2}^{\pi/2}
              |H(\theta)|^2 \sec^3\theta\,d\theta,
        \qquad
        H(\theta) = \int_S \sigma\,e^{\kappa\sec^2\theta
                     (z + i(x\cos\theta + y\sin\theta))}\,dS .

    Derivation (what fixes the constant, *not* fitted to any data)

    The steady wave field is the residue of the pole ``k = kappa sec^2 theta``
    of the double Fourier transform of ``G_W`` (see :func:`nk_wave_kernel`), so
    every wave component carries the depth decay ``exp(kappa sec^2 theta z)``
    and the streamwise phase ``kappa sec theta x = kappa sec^2 theta x cos
    theta``.  Substituting the residue amplitude into the pressure integral over
    a control surface in the radiation field gives the classical Kochin form;
    written with the *source strength* ``sigma`` (units m/s, the single-layer
    density of :func:`solve_nk_panel`) the amplitude is ``H`` above and the
    constant is ``rho kappa^2/(2 pi)``.  For the thin-ship sheet the same
    formula with ``sigma = -2 U db/dx`` is Michell's integral
    ``(4 rho g^2/(pi U^2)) int_0^{pi/2} sec^3 theta |A|^2 dtheta`` with
    ``H = -2 U A`` -- i.e. ``rho kappa^2/(2 pi) = (rho g kappa)/(2 pi U^2)``
    over the full range equals ``4 rho g^2/(pi U^2)`` over the half range.

    The historical defect (mission m10003 "implied prefactor drifting 4.8x
    between Fr = 0.2 and 0.4") is now explained: the old default was the
    dimensionless ``0.5``, but the exact value is ``kappa/(2 pi g) = 1/(2 pi
    U^2)``, which varies by ``(0.4/0.2)^2 = 4`` over that Froude range.  The
    constant was never Froude-dependent -- the missing ``1/U^2`` was.

    ``prefactor=None`` (default) uses :func:`kochin_prefactor`; passing a number
    is kept for linearity/sensitivity tests only.
    """
    if prefactor is None:
        prefactor = kochin_prefactor(kappa, g)
    th = np.linspace(-0.5 * math.pi + 1e-7, 0.5 * math.pi - 1e-7, n_theta)
    H = kochin_amplitude(sigma, mesh, kappa, th, mirror=mirror)
    sec3 = 1.0 / np.cos(th) ** 3
    integrand = np.abs(H) ** 2 * sec3
    val = prefactor * rho * g * kappa * float(np.trapezoid(integrand, th))
    return val, th, H


# ---------------------------------------------------------------------------
# verification helpers
# ---------------------------------------------------------------------------


def _total_kernel(points, sources, kappa, n_theta, radiation, sign):
    G0, Gx0, Gy0, Gz0 = rankine_kernel(points, sources, mirror=True)
    Gw, Gxw, Gyw, Gzw = nk_wave_kernel(points, sources, kappa, n_theta, radiation, sign)
    return G0 + Gw, Gx0 + Gxw, Gy0 + Gyw, Gz0 + Gzw


def verify_free_surface_condition(
    kappa: float,
    n_theta: int = 64,
    radiation: str = "downstream",
    sign: float = -1.0,
    h: float = 1e-3,
) -> dict:
    """Check ``(G_R + G_W)_xx + kappa (G_R + G_W)_z = 0`` on ``z = 0-``.

    The kernel must satisfy the steady free-surface condition; the Rankine part
    alone does not, so the *total* kernel is tested (this is the check that
    pins the principal-value structure of the wavenumber integral).
    """
    src = np.array([[0.0, 0.0, -0.3 / kappa]])
    hs = h / kappa
    eps = 1e-7 / kappa
    res = []
    for xr in (0.6, 1.2, 2.0):
        for yr in (0.0, 0.7):
            x = xr / kappa
            y = yr / kappa
            base = np.array([[x, y, 0.0]])
            G0 = _total_kernel(base, src, kappa, n_theta, radiation, sign)[0].ravel()[0].real
            Gp = _total_kernel(np.array([[x + hs, y, 0.0]]), src, kappa, n_theta, radiation, sign)[0].ravel()[0].real
            Gm = _total_kernel(np.array([[x - hs, y, 0.0]]), src, kappa, n_theta, radiation, sign)[0].ravel()[0].real
            Gd = _total_kernel(np.array([[x, y, -eps]]), src, kappa, n_theta, radiation, sign)[0].ravel()[0].real
            gxx = (Gp - 2.0 * G0 + Gm) / hs**2
            gz = (G0 - Gd) / eps
            scale = max(abs(gxx), abs(kappa * gz), 1e-30)
            res.append(abs(gxx + kappa * gz) / scale)
    return {
        "max_relative_residual": float(max(res)),
        "mean_relative_residual": float(np.mean(res)),
        "kappa": kappa,
        "radiation": radiation,
        "sign": sign,
        "n_theta": n_theta,
    }


def verify_radiation_direction(
    kappa: float,
    n_theta: int = 64,
    radiation: str = "downstream",
    sign: float = -1.0,
    distance: float = 8.0,
) -> dict:
    """Upstream/downstream wave amplitude ratio on ``z = 0``.

    The steady kernel must produce a wave train *behind* the source (``x < 0``
    for a stream flowing in ``+x``) and only the decaying local field ahead.
    """
    src = np.array([[0.0, 0.0, -0.25 / kappa]])
    xs = np.linspace(-distance / kappa, distance / kappa, 81)
    pts = np.column_stack([xs, np.zeros(xs.size), np.zeros(xs.size)])
    G = nk_wave_kernel(pts, src, kappa, n_theta, radiation, sign)[0]
    amp = np.abs(G.ravel())
    up = float(np.mean(amp[xs > 0.6 * distance / kappa]))
    down = float(np.mean(amp[xs < -0.6 * distance / kappa]))
    return {
        "upstream_mean_abs": up,
        "downstream_mean_abs": down,
        "ratio_down_over_up": down / max(up, 1e-30),
        "radiation": radiation,
        "sign": sign,
    }


def verify_laplace(
    kappa: float,
    n_theta: int = 64,
    radiation: str = "downstream",
    sign: float = -1.0,
    h: float = 1e-5,
) -> dict:
    """Check that the wave kernel is harmonic in ``z < 0``."""
    src = np.array([[0.0, 0.3 / kappa, -0.5 / kappa]])
    p = np.array([[0.7 / kappa, -0.2 / kappa, -0.8 / kappa]])
    Gw0 = nk_wave_kernel(p, src, kappa, n_theta, radiation, sign)[0].ravel()[0]
    lap = 0.0
    for k in range(3):
        ep = p.copy()
        em = p.copy()
        ep[0, k] += h
        em[0, k] -= h
        Gp = nk_wave_kernel(ep, src, kappa, n_theta, radiation, sign)[0].ravel()[0]
        Gm = nk_wave_kernel(em, src, kappa, n_theta, radiation, sign)[0].ravel()[0]
        lap += float((Gp - 2.0 * Gw0 + Gm) / h**2)
    return {"laplacian": lap, "value": float(Gw0), "h": h}


def kelvin_wake_extent(
    kappa: float,
    n_theta: int = 96,
    radiation: str = "halfplane",
    sign: float = -1.0,
    span: float = 16.0,
    n: int = 161,
    threshold: float = 0.05,
) -> dict:
    """Outermost extent of the Kelvin wake of a point source.

    For each downstream station the largest ``|y|`` at which the wave amplitude
    still exceeds ``threshold`` times the *global* maximum is located; the mean
    angle of those points is compared with the classical caustic half-angle
    ``arcsin(1/3) = 19.47 deg``.
    """
    src = np.array([[0.0, 0.0, -0.3 / kappa]])
    xs = np.linspace(-span / kappa, -0.1 / kappa, n)
    ys = np.linspace(-span / kappa, span / kappa, 2 * n - 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    amp = np.abs(
        nk_wave_kernel(pts, src, kappa, n_theta, radiation, sign)[0].reshape(X.shape)
    )
    thr = threshold * amp.max()
    ang = []
    for i in range(len(xs) // 2, len(xs)):
        row = amp[i]
        idx = np.where(row > thr)[0]
        if idx.size == 0:
            continue
        y_edge = float(np.max(np.abs(ys[idx])))
        ang.append(math.degrees(math.atan2(y_edge, abs(xs[i]))))
    return {
        "mean_wedge_extent_deg": float(np.mean(ang)) if ang else float("nan"),
        "std_wedge_extent_deg": float(np.std(ang)) if ang else float("nan"),
        "theoretical_deg": float(math.degrees(math.asin(1.0 / 3.0))),
        "threshold_fraction": threshold,
        "radiation": radiation,
        "sign": sign,
    }


def thin_ship_reference(
    L: float,
    B: float,
    T: float,
    U: float,
    nx: int = 120,
    nz: int = 60,
    g: float = _G,
    rho: float = _RHO,
    prefactor: float | None = None,
    near_field: bool = False,
) -> dict:
    """Thin-ship (Michell) source sheet on the centreplane.

    Source strength per unit centreplane area ``m = -2 U db/dx`` (the two sides
    of the thin hull add up); used to validate the far-field operator against
    Michell's resistance and to provide the thin-ship wave-cut reference.
    """
    mesh = centreplane_panels(L, B, T, nx, nz)
    x = mesh.centroids[:, 0]
    z = mesh.centroids[:, 2]
    b_x = 0.5 * B * (-8.0 * x / L**2) * (1.0 - (z / T) ** 2)
    sigma = -2.0 * U * b_x
    kappa = g / U**2
    Rw_far, th, H = wave_resistance_far(
        sigma, mesh, kappa, rho=rho, g=g, mirror=False, prefactor=prefactor
    )
    # near-field pressure integral on the centreplane (optional: O(N^2) kernel)
    Rw_near = float("nan")
    if near_field:
        _, Gxw, _, _ = nk_wave_kernel(
            mesh.centroids, mesh.centroids, kappa, radiation="halfplane", sign=-1.0
        )
        dphi_dx = (Gxw * mesh.areas[None, :]) @ sigma
        Rw_near = -rho * U * float(np.sum(dphi_dx * (-b_x) * mesh.areas)) * 2.0
    s_ref = L * T
    from .wigley import michell_wave_resistance_canonical

    rw_michell = float(michell_wave_resistance_canonical(L, B, T, U, rho, g))
    return {
        "mesh": mesh,
        "sigma": sigma,
        "b_x": b_x,
        "rw_far": Rw_far,
        "rw_near": Rw_near,
        "rw_michell": rw_michell,
        "rel_error_far_vs_michell": abs(Rw_far - rw_michell) / rw_michell,
        "cw_far": Rw_far / (0.5 * rho * U**2 * s_ref),
        "cw_michell": rw_michell / (0.5 * rho * U**2 * s_ref),
        "cw_near": Rw_near / (0.5 * rho * U**2 * s_ref),
        "s_ref": s_ref,
        "H": H,
        "theta": th,
        "kappa": kappa,
    }