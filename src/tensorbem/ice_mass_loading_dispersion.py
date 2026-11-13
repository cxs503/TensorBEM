"""BEM verification of the Weitz-Keller mass-loading dispersion relation.

Physical model
--------------
An ice sheet of thickness ``h_i`` and density ``rho_i`` floats on water of
density ``rho_w`` and depth ``H``.  The sheet is treated as a *mass load*: it
carries inertia but no bending stiffness (the classical Weitz & Keller 1950
model; the elastic extension is :mod:`tensorbem.flexural_gravity`).  With the
harmonic convention ``phi ~ exp(-i omega t)`` the linearised free-surface
condition is the Robin condition

    dphi/dz = rho_w omega^2 phi / (rho_w g - rho_i h_i omega^2) =: Lambda phi
                                                                        (1)

on ``z = 0`` (derivation in ``docs/case_bem_ice_dispersion/REPORT.md``,
appendix A).  Writing the vertical structure as
``phi = cos(k x) cosh(k (z + H))`` turns (1) into the classical relation

    k tanh(k H) = Lambda,                                            (2)
    omega^2 = g k tanh(k H) / (1 + mu k tanh(k H)),  mu = rho_i h_i / rho_w
                                                                        (3)

which is the **reference** used here.  Its limits are the anchors of the
benchmark: ``k H -> infinity`` gives the deep-water Weitz-Keller form
``omega^2 = g k / (1 + mu k)``, and ``k H -> 0`` gives the open-water long-wave
limit ``omega -> sqrt(g H) k`` (the ice inertia becomes negligible).

What this module computes (the "got" side)
------------------------------------------
Nothing above is used to obtain the numerical answer.  Instead one period cell
``0 <= x <= L``, ``-H <= z <= 0`` of the ice-covered basin (``L = 2 pi / k``,
periodic in ``x``, rigid bottom) is discretised with **constant panels** and the
Laplace boundary integral equation

    1/2 phi(x_i) - sum_j D_ij phi_j + sum_j S_ij q_j = 0                 (4)
    S_ij = int_j (1/2 pi) ln r ds,   D_ij = int_j dG/dn ds,  q = dphi/dn

is solved (:func:`assemble_strip_matrices`, :func:`strip_bem_depth_factor`).
Two independent numerical routes are provided:

* **driven mode** -- prescribe ``phi = cos(k x)`` on the free-surface panels,
  solve for the flux and form the depth factor as the ``L2`` Rayleigh quotient
  ``Lambda = sum(q_i phi_i l_i) / sum(phi_i^2 l_i)`` (a single number even
  though the numerical flux is panel-wise);
* **DtN map** -- assemble the dense Dirichlet-to-Neumann map ``T`` of the free
  surface (:func:`dirichlet_to_neumann`) and evaluate the same quotient,
  ``Lambda = <T c, c> / <c, c>``.  This is a separate code path (matrix
  assembly + matvec) over the same layer matrices, so agreement of the two
  routes to round-off checks the assembly; the accuracy against ``Lambda`` is
  separate and reported on its own.

Higher periodic modes are solved with :func:`strip_bem_mode_sweep` (the cell
length stays ``2 pi / k`` and ``phi = cos(m k x)`` is prescribed, so mode ``m``
is resolved by ``panels_per_wavelength / m`` panels per wavelength).

Either way ``omega^2`` follows algebraically from (2),
``omega^2 = g Lambda / (1 + mu Lambda)``
(:func:`mass_loading_omega2_from_depth_factor`), so the BEM discretisation error
in the depth factor is the only error in "got".

Conventions and units
---------------------
All quantities are SI.  ``q = dphi/dn`` with ``n`` the **outward** normal, so on
the free surface ``q = dphi/dz``.  Phases are absent: the problem is a real
eigenproblem.

References
----------
* Weitz, M. & Keller, J.B. (1950), *Some remarks on sea ice* -- mass-loading
  dispersion relation for an ice-covered sea surface.
* Fox, C. & Squire, V.A. (1994), *On the oblique reflexion and transmission of
  ocean waves*, Phil. Trans. R. Soc. A 352 -- restates the mass-loading form (3).
* Brebbia, C.A. & Dominguez, J. (1992), *Boundary Elements: An Introductory
  Course* -- constant-panel Laplace BIE (4) with free term ``1/2``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

DEFAULT_GRAVITY = 9.80665
DEFAULT_WATER_DENSITY = 1025.0
DEFAULT_ICE_DENSITY = 917.0
_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# reference (analytic) side
# ---------------------------------------------------------------------------
def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return value


def mass_loading_mu(
    ice_thickness: float,
    *,
    ice_density: float = DEFAULT_ICE_DENSITY,
    water_density: float = DEFAULT_WATER_DENSITY,
) -> float:
    """Mass-loading parameter ``mu = rho_i h_i / rho_w`` (m)."""
    h = _positive("ice thickness", ice_thickness)
    rho_i = _positive("ice density", ice_density)
    rho_w = _positive("water density", water_density)
    return rho_i * h / rho_w


def weitz_keller_depth_factor(k: float, depth: float) -> float:
    """``k tanh(k H)``: the depth factor that the Robin condition (2) must match."""
    k = _positive("wavenumber", k)
    H = _positive("water depth", depth)
    return k * math.tanh(k * H)


def mass_loading_omega2_from_depth_factor(
    lam: float, *, mu: float, gravity: float = DEFAULT_GRAVITY
) -> float:
    """Invert the Robin condition: ``omega^2 = g Lambda / (1 + mu Lambda)``.

    ``Lambda`` is ``(dphi/dz) / phi`` on the free surface.  This map is the only
    place where the ice enters once the BEM has produced ``Lambda``.
    """
    lam = float(lam)
    if not math.isfinite(lam) or lam <= 0.0:
        raise ValueError(f"depth factor must be finite positive, got {lam!r}")
    mu = _positive("mass-loading parameter", mu)
    g = _positive("gravity", gravity)
    return g * lam / (1.0 + mu * lam)


def weitz_keller_omega2(
    k: float,
    *,
    depth: float,
    mu: float,
    gravity: float = DEFAULT_GRAVITY,
) -> float:
    """Reference ``omega^2`` of the finite-depth mass-loading relation (3)."""
    lam = weitz_keller_depth_factor(k, depth)
    return mass_loading_omega2_from_depth_factor(lam, mu=mu, gravity=gravity)


def weitz_keller_omega(
    k: float,
    *,
    depth: float,
    mu: float,
    gravity: float = DEFAULT_GRAVITY,
) -> float:
    """Reference ``omega`` (rad/s) of the finite-depth mass-loading relation."""
    return math.sqrt(weitz_keller_omega2(k, depth=depth, mu=mu, gravity=gravity))


def weitz_keller_deep_omega2(
    k: float, *, mu: float, gravity: float = DEFAULT_GRAVITY
) -> float:
    """Deep-water Weitz-Keller limit ``omega^2 = g k / (1 + mu k)``."""
    k = _positive("wavenumber", k)
    mu = _positive("mass-loading parameter", mu)
    g = _positive("gravity", gravity)
    return g * k / (1.0 + mu * k)


def long_wave_omega(
    k: float, *, depth: float, gravity: float = DEFAULT_GRAVITY
) -> float:
    """Shallow-water (long-wave) limit ``omega = sqrt(g H) k``.

    This is the ``k H -> 0`` limit of (3): the mass-loading term enters at
    ``O((k H)^2)``, so the long wave travels at the open-water speed
    ``sqrt(g H)`` and is insensitive to the ice.
    """
    k = _positive("wavenumber", k)
    H = _positive("water depth", depth)
    g = _positive("gravity", gravity)
    return math.sqrt(g * H) * k


def mass_loading_cutoff_omega(
    mu: float, *, gravity: float = DEFAULT_GRAVITY
) -> float:
    """Short-wave cutoff ``sqrt(g / mu)``: ``omega`` can never exceed it."""
    mu = _positive("mass-loading parameter", mu)
    g = _positive("gravity", gravity)
    return math.sqrt(g / mu)


# ---------------------------------------------------------------------------
# periodic strip mesh
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StripMesh:
    """Constant-panel discretisation of one periodic cell of the basin.

    The boundary consists of ``free_surface`` panels (``z = 0``, outward normal
    ``+z``), ``bottom`` panels (``z = -H``), and ``left`` / ``right`` panels
    (``x = 0`` and ``x = L``), the two side sets sharing the same ``z``
    subdivision so that the periodic ties are one-to-one.
    """

    period: float
    depth: float
    starts: np.ndarray
    ends: np.ndarray
    normals: np.ndarray
    lengths: np.ndarray
    index: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_panels(self) -> int:
        return int(self.starts.shape[0])

    @property
    def centres(self) -> np.ndarray:
        return 0.5 * (self.starts + self.ends)

    def describe(self) -> dict[str, float]:
        return {
            "period": self.period,
            "depth": self.depth,
            "n_panels": float(self.n_panels),
            "n_free_surface": float(self.index["free_surface"].size),
            "n_bottom": float(self.index["bottom"].size),
            "n_side": float(self.index["left"].size),
        }


def _depth_nodes(
    s0: float,
    depth: float,
    *,
    growth: float = 1.18,
    min_panels: int = 8,
    max_panels: int = 800,
) -> np.ndarray:
    """``z`` breakpoints from ``0`` down to ``-depth``.

    Deep basins are graded geometrically away from the free surface (the motion
    decays like ``exp(k z)``); shallow ones are split uniformly.
    """
    if s0 >= 0.25 * depth:
        n = int(min(max(min_panels, math.ceil(depth / s0)), max_panels))
        return -depth * np.arange(n + 1) / n
    nodes = [0.0]
    z = 0.0
    step = s0
    while z > -depth and len(nodes) < max_panels:
        z = max(z - step, -depth)
        nodes.append(z)
        step *= growth
    if len(nodes) - 1 < min_panels:
        n = min_panels
        return -depth * np.arange(n + 1) / n
    return np.asarray(nodes, dtype=float)


def build_periodic_strip(
    k: float,
    depth: float,
    *,
    panels_per_wavelength: int = 32,
    min_side_panels: int = 8,
    max_side_panels: int = 800,
) -> StripMesh:
    """Mesh one period cell ``L = 2 pi / k`` of the ice-covered basin."""
    k = _positive("wavenumber", k)
    H = _positive("water depth", depth)
    n_x = int(panels_per_wavelength)
    if n_x < 4:
        raise ValueError("panels_per_wavelength must be at least 4")
    period = _TWO_PI / k
    hx = period / n_x
    x_breaks = np.linspace(0.0, period, n_x + 1)
    z_breaks = _depth_nodes(
        hx, H, min_panels=min_side_panels, max_panels=max_side_panels
    )
    n_z = z_breaks.size - 1

    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    index: dict[str, np.ndarray] = {}

    def add(a, b, unit_normal, count):
        idx = np.arange(len(starts), len(starts) + count)
        starts.extend(a)
        ends.extend(b)
        normals.extend([unit_normal] * count)
        return idx

    a = np.column_stack([x_breaks[:-1], np.zeros(n_x)])
    b = np.column_stack([x_breaks[1:], np.zeros(n_x)])
    index["free_surface"] = add(a, b, (0.0, 1.0), n_x)
    a = np.column_stack([x_breaks[:-1], np.full(n_x, -H)])
    b = np.column_stack([x_breaks[1:], np.full(n_x, -H)])
    index["bottom"] = add(a, b, (0.0, -1.0), n_x)
    a = np.column_stack([np.zeros(n_z), z_breaks[:-1]])
    b = np.column_stack([np.zeros(n_z), z_breaks[1:]])
    index["left"] = add(a, b, (-1.0, 0.0), n_z)
    a = np.column_stack([np.full(n_z, period), z_breaks[:-1]])
    b = np.column_stack([np.full(n_z, period), z_breaks[1:]])
    index["right"] = add(a, b, (1.0, 0.0), n_z)

    starts_a = np.asarray(starts, dtype=float)
    ends_a = np.asarray(ends, dtype=float)
    normals_a = np.asarray(normals, dtype=float)
    lengths = np.linalg.norm(ends_a - starts_a, axis=1)
    return StripMesh(
        period=period,
        depth=H,
        starts=starts_a,
        ends=ends_a,
        normals=normals_a,
        lengths=lengths,
        index=index,
    )


# ---------------------------------------------------------------------------
# Laplace BEM assembly (constant panels)
# ---------------------------------------------------------------------------
def _self_integrals(length: float) -> tuple[float, float]:
    """Analytic ``(S_ii, D_ii)`` for a straight constant panel.

    ``int_{-l/2}^{l/2} (1/2 pi) ln|s| ds = (l / 2 pi) (ln(l/2) - 1)`` and the
    principal-value double layer vanishes on a straight panel.
    """
    return (length / _TWO_PI) * (math.log(0.5 * length) - 1.0), 0.0


def _integrate_segment(
    colloc: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    normal: np.ndarray,
    order_x: np.ndarray,
    order_w: np.ndarray,
    *,
    max_depth: int = 8,
) -> tuple[float, float]:
    """``int_j G ds`` and ``int_j dG/dn ds`` with recursive subdivision."""
    mid = 0.5 * (a + b)
    span = b - a
    length = float(np.linalg.norm(span))
    if length <= 0.0:
        return 0.0, 0.0
    if float(np.linalg.norm(colloc - mid)) < 2.0 * length and max_depth > 0:
        s1 = _integrate_segment(
            colloc, a, mid, normal, order_x, order_w, max_depth=max_depth - 1
        )
        s2 = _integrate_segment(
            colloc, mid, b, normal, order_x, order_w, max_depth=max_depth - 1
        )
        return s1[0] + s2[0], s1[1] + s2[1]
    tangent = span / length
    pts = mid[None, :] + 0.5 * length * order_x[:, None] * tangent[None, :]
    rv = colloc[None, :] - pts
    r2 = np.einsum("ij,ij->i", rv, rv)
    jac = 0.5 * length * order_w
    single = float(np.sum((1.0 / _TWO_PI) * np.log(np.sqrt(r2)) * jac))
    double = float(
        np.sum(-(1.0 / _TWO_PI) * np.einsum("ij,j->i", rv, normal) / r2 * jac)
    )
    return single, double


def assemble_strip_matrices(
    mesh: StripMesh, *, order: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """Dense single- (``S``) and double-layer (``D``) matrices of the strip.

    ``S_ij = int_j (1/2 pi) ln r ds`` and ``D_ij = int_j dG/dn ds`` with
    ``G = (1/2 pi) ln r`` and ``n`` the outward normal, so that the interior
    limit of the double-layer potential is ``(D phi)_i + phi_i / 2`` and the
    boundary integral equation (4) uses the free term ``1/2``.
    """
    xi, w = np.polynomial.legendre.leggauss(int(order))
    n = mesh.n_panels
    starts, ends = mesh.starts, mesh.ends
    normals, lengths = mesh.normals, mesh.lengths
    centres = mesh.centres
    mid = 0.5 * (starts + ends)
    tangent = (ends - starts) / lengths[:, None]

    single = np.zeros((n, n))
    double = np.zeros((n, n))
    for i in range(n):
        colloc = centres[i]
        dist = np.linalg.norm(mid - colloc[None, :], axis=1)
        near = dist < 2.0 * lengths
        near[i] = True
        far = ~near
        if np.any(far):
            pts = (
                mid[far][None, :, :]
                + 0.5
                * lengths[far][None, :, None]
                * xi[:, None, None]
                * tangent[far][None, :, :]
            )
            rv = colloc[None, None, :] - pts
            r2 = np.einsum("ijk,ijk->ij", rv, rv)
            jac = 0.5 * lengths[far][None, :] * w[:, None]
            single[i, far] = np.sum(
                (1.0 / _TWO_PI) * np.log(np.sqrt(r2)) * jac, axis=0
            )
            double[i, far] = np.sum(
                -(1.0 / _TWO_PI)
                * np.einsum("ijk,jk->ij", rv, normals[far])
                / r2
                * jac,
                axis=0,
            )
        for j in np.nonzero(near)[0]:
            if j == i:
                s_self, d_self = _self_integrals(float(lengths[i]))
                single[i, j] = s_self
                double[i, j] = d_self
            else:
                s_near, d_near = _integrate_segment(
                    colloc, starts[j], ends[j], normals[j], xi, w
                )
                single[i, j] = s_near
                double[i, j] = d_near
    return single, double


# ---------------------------------------------------------------------------
# tied mixed problem: Dirichlet on the free surface, Neumann elsewhere,
# periodic ties between the left and right walls
# ---------------------------------------------------------------------------
def tie_operators(mesh: StripMesh) -> tuple[np.ndarray, np.ndarray]:
    """Maps from the tied unknown vector to panel values / fluxes.

    Unknowns ``u = [q_fs, phi_bottom, phi_side, q_side]``; the right wall is the
    periodic image of the left one (``phi_R = phi_L`` and ``q_R = -q_L`` because
    the outward normals are opposite).  Returns ``(P, Q)`` with
    ``phi_panel = P u + data`` and ``q_panel = Q u``.
    """
    n = mesh.n_panels
    fs = mesh.index["free_surface"]
    bot = mesh.index["bottom"]
    left = mesh.index["left"]
    right = mesh.index["right"]
    n_fs, n_bot, n_side = fs.size, bot.size, left.size
    if right.size != n_side:
        raise ValueError("left and right walls must have the same panel count")
    m = n_fs + n_bot + 2 * n_side
    if m != n:
        raise ValueError(
            f"tied unknown count {m} does not match panel count {n}; the "
            "Dirichlet/Neumann split must cover every panel exactly once"
        )
    p = np.zeros((n, m))
    q = np.zeros((n, m))
    p[bot, n_fs + np.arange(n_bot)] = 1.0
    p[left, n_fs + n_bot + np.arange(n_side)] = 1.0
    p[right, n_fs + n_bot + np.arange(n_side)] = 1.0
    q[fs, np.arange(n_fs)] = 1.0
    q[left, n_fs + n_bot + n_side + np.arange(n_side)] = 1.0
    q[right, n_fs + n_bot + n_side + np.arange(n_side)] = -1.0
    return p, q


def _tied_operator(
    single: np.ndarray,
    double: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``(A, B)`` with ``A u = B data`` for equation (4) of the module docstring."""
    a = 0.5 * p - double @ p + single @ q
    b = double - 0.5 * np.eye(double.shape[0])
    return a, b


@dataclass(frozen=True)
class StripSolve:
    """One driven-mode BEM solve on the periodic strip."""

    wavenumber: float
    depth: float
    panels_per_wavelength: int
    n_panels: int
    mode: int
    depth_factor: float
    depth_factor_rq: float
    panel_scatter: float
    exact_depth_factor: float
    residual: float
    bv_error: float
    q_free_surface: np.ndarray
    phi_free_surface: np.ndarray
    x_free_surface: np.ndarray

    @property
    def depth_factor_error(self) -> float:
        return self.depth_factor_rq / self.exact_depth_factor - 1.0

    def describe(self) -> dict[str, float]:
        return {
            "wavenumber": self.wavenumber,
            "mode": float(self.mode),
            "depth": self.depth,
            "panels_per_wavelength": float(self.panels_per_wavelength),
            "n_panels": float(self.n_panels),
            "depth_factor_bem": self.depth_factor_rq,
            "depth_factor_bem_panel_mean": self.depth_factor,
            "depth_factor_exact": self.exact_depth_factor,
            "depth_factor_rel_error": self.depth_factor_error,
            "panel_scatter": self.panel_scatter,
            "linear_system_residual": self.residual,
            "free_surface_flux_bv_error": self.bv_error,
        }


def strip_bem_depth_factor(
    k: float,
    depth: float,
    *,
    panels_per_wavelength: int = 32,
    order: int = 12,
    mode: int = 1,
) -> StripSolve:
    """BEM depth factor ``(dphi/dz)/phi`` for the periodic mode ``cos(k x)``.

    The free-surface trace ``cos(k x)`` (unit amplitude) is prescribed, the tied
    mixed BIE is solved, and ``Lambda`` is reported panel-wise (mean and
    scatter) and as the ``L2`` Rayleigh quotient, which is the value used for
    the dispersion comparison.
    """
    k = _positive("wavenumber", k)
    H = _positive("water depth", depth)
    mode = int(mode)
    if mode < 1:
        raise ValueError(f"mode must be a positive integer, got {mode!r}")
    km = mode * k
    mesh = build_periodic_strip(k, H, panels_per_wavelength=panels_per_wavelength)
    single, double = assemble_strip_matrices(mesh, order=order)
    p, q = tie_operators(mesh)
    a, b = _tied_operator(single, double, p, q)
    fs = mesh.index["free_surface"]
    x_fs = mesh.centres[fs, 0]
    phi_fs = np.cos(km * x_fs)
    data = np.zeros(mesh.n_panels)
    data[fs] = phi_fs
    u = _solve_tied_system(a, b @ data)
    q_fs = u[: fs.size]
    residual = float(
        np.linalg.norm(a @ u - b @ data) / max(1.0, np.linalg.norm(b @ data))
    )
    lengths = mesh.lengths[fs]
    lam_rq = float(np.sum(q_fs * phi_fs * lengths) / np.sum(phi_fs * phi_fs * lengths))
    ratio = q_fs / phi_fs
    lam_mean = float(np.mean(ratio))
    scatter = float(np.max(np.abs(ratio - lam_mean)) / abs(lam_mean))
    # Independent boundary-value check on the *solved* unknowns: the exact
    # field of the driven mode is phi = cos(k x) cosh(k (z + H)) / cosh(k H).
    # The rigid bottom and the periodic walls were not prescribed with that
    # data, so recovering it is a genuine check of the interior solution.
    bot = mesh.index["bottom"]
    left = mesh.index["left"]
    phi_bot = u[fs.size : fs.size + bot.size]
    phi_side = u[fs.size + bot.size : fs.size + bot.size + left.size]
    cosh_kH = math.cosh(km * H)
    exact_bot = np.cos(km * mesh.centres[bot, 0]) / cosh_kH
    exact_side = np.cosh(km * (mesh.centres[left, 1] + H)) / cosh_kH
    bv_error = float(
        max(
            np.max(np.abs(phi_bot - exact_bot)),
            np.max(np.abs(phi_side - exact_side)),
        )
    )
    return StripSolve(
        wavenumber=km,
        mode=mode,
        depth=H,
        panels_per_wavelength=int(panels_per_wavelength),
        n_panels=mesh.n_panels,
        depth_factor=lam_mean,
        depth_factor_rq=lam_rq,
        panel_scatter=scatter,
        exact_depth_factor=weitz_keller_depth_factor(km, H),
        residual=residual,
        bv_error=bv_error,
        q_free_surface=q_fs,
        phi_free_surface=phi_fs,
        x_free_surface=x_fs,
    )


def _solve_tied_system(a: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve the tied BIE system with numpy's LAPACK ``dgesv``.

    ``scipy.linalg.solve`` was observed to return a solution with a relative
    residual of ``3.6e3`` on this operator (condition number 5.4e2, residual of
    the numpy solution ``6e-15``) in the reference environment, so the numpy
    path is used here; the residual is reported by
    :class:`StripSolve` and asserted in the tests, so a silent loss of accuracy
    cannot pass unnoticed.
    """
    return np.linalg.solve(a, rhs)


def dirichlet_to_neumann(
    k: float,
    depth: float,
    *,
    panels_per_wavelength: int = 32,
    order: int = 12,
) -> tuple[np.ndarray, StripMesh]:
    """Dense free-surface Dirichlet-to-Neumann map ``T`` of the strip.

    Column ``j`` is the free-surface flux produced by a unit Dirichlet trace on
    panel ``j`` (one LU factorisation, ``n_fs`` back substitutions).  The
    continuous operator has eigenvalues ``k_n tanh(k_n H)`` on the modes
    ``cos(k_n x)``, ``k_n = 2 pi n / L``; ``T`` is the discretised version of
    that statement, and its *accuracy* is measured through
    :func:`dtn_depth_factor` rather than through an eigenvalue solve (a dense
    eigenproblem in the piecewise-constant basis is spectrally polluted by the
    panel corners, which is why it is not used here).
    """
    k = _positive("wavenumber", k)
    H = _positive("water depth", depth)
    mesh = build_periodic_strip(k, H, panels_per_wavelength=panels_per_wavelength)
    single, double = assemble_strip_matrices(mesh, order=order)
    p, q = tie_operators(mesh)
    a, b = _tied_operator(single, double, p, q)
    fs = mesh.index["free_surface"]
    n_fs = fs.size
    basis = np.zeros((mesh.n_panels, n_fs))
    basis[fs, np.arange(n_fs)] = 1.0
    solved = _solve_tied_system(a, b @ basis)
    t = solved[:n_fs, :]
    return t, mesh


@dataclass(frozen=True)
class StripSolveField:
    """Interior potential recovered from a strip solution, vs the exact mode."""

    points: np.ndarray
    potential: np.ndarray
    exact: np.ndarray

    @property
    def max_error(self) -> float:
        return float(np.max(np.abs(self.potential - self.exact)))

    @property
    def relative_error(self) -> float:
        return self.max_error / float(np.max(np.abs(self.exact)))


def evaluate_strip_interior_field(
    mesh: StripMesh,
    phi_panel: np.ndarray,
    q_panel: np.ndarray,
    points: np.ndarray,
    *,
    order: int = 12,
) -> np.ndarray:
    """Recover ``phi`` at interior points from the boundary representation.

    ``phi(x) = sum_j int_j [phi dG/dn - G q] ds`` for ``x`` inside the cell; the
    same layer kernels as the assembly are used, so this is a check of the
    boundary data (and of the interior solution), not a new discretisation.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    xi, w = np.polynomial.legendre.leggauss(int(order))
    mid = 0.5 * (mesh.starts + mesh.ends)
    tangent = (mesh.ends - mesh.starts) / mesh.lengths[:, None]
    out = np.zeros(pts.shape[0])
    for i, x in enumerate(pts):
        dist = np.linalg.norm(mid - x[None, :], axis=1)
        near = dist < 1.5 * mesh.lengths
        far = ~near
        value = 0.0
        if np.any(far):
            g_pts = (
                mid[far][None, :, :]
                + 0.5 * mesh.lengths[far][None, :, None] * xi[:, None, None]
                * tangent[far][None, :, :]
            )
            rv = x[None, None, :] - g_pts
            r2 = np.einsum("ijk,ijk->ij", rv, rv)
            jac = 0.5 * mesh.lengths[far][None, :] * w[:, None]
            single = np.sum((1.0 / _TWO_PI) * np.log(np.sqrt(r2)) * jac, axis=0)
            double = np.sum(
                -(1.0 / _TWO_PI) * np.einsum("ijk,jk->ij", rv, mesh.normals[far])
                / r2 * jac,
                axis=0,
            )
            value += float(
                np.sum(double * phi_panel[far] - single * q_panel[far])
            )
        for j in np.nonzero(near)[0]:
            single_j, double_j = _integrate_segment(
                x, mesh.starts[j], mesh.ends[j], mesh.normals[j], xi, w
            )
            value += double_j * phi_panel[j] - single_j * q_panel[j]
        out[i] = value
    return out


def strip_bem_field(
    k: float,
    depth: float,
    *,
    panels_per_wavelength: int = 64,
    nx: int = 41,
    nz: int = 21,
    mode: int = 1,
    order: int = 12,
) -> StripSolveField:
    """Recover the driven-mode potential on an interior grid of the cell.

    The exact driven mode is ``phi = cos(k_m x) cosh(k_m (z + H)) / cosh(k_m H)``;
    comparing the BEM recovery with it inside the domain (not just on the
    boundary) is the field-level evidence of the case.
    """
    k = _positive("wavenumber", k)
    H = _positive("water depth", depth)
    km = int(mode) * k
    mesh = build_periodic_strip(k, H, panels_per_wavelength=panels_per_wavelength)
    single, double = assemble_strip_matrices(mesh, order=order)
    p, q = tie_operators(mesh)
    a, b = _tied_operator(single, double, p, q)
    fs = mesh.index["free_surface"]
    phi_fs = np.cos(km * mesh.centres[fs, 0])
    data = np.zeros(mesh.n_panels)
    data[fs] = phi_fs
    u = _solve_tied_system(a, b @ data)
    phi_panel = data.copy()
    phi_panel[mesh.index["bottom"]] = u[fs.size : fs.size + mesh.index["bottom"].size]
    phi_panel[mesh.index["left"]] = u[
        fs.size + mesh.index["bottom"].size : fs.size + mesh.index["bottom"].size
        + mesh.index["left"].size
    ]
    phi_panel[mesh.index["right"]] = phi_panel[mesh.index["left"]]
    q_panel = np.zeros(mesh.n_panels)
    q_panel[fs] = u[: fs.size]
    q_panel[mesh.index["left"]] = u[fs.size + mesh.index["bottom"].size + mesh.index["left"].size :]
    q_panel[mesh.index["right"]] = -q_panel[mesh.index["left"]]

    xs = np.linspace(0.0, mesh.period, int(nx))[:-1] + mesh.period / (2.0 * int(nx))
    zs = -H + (H * (np.arange(int(nz)) + 0.5) / int(nz))
    grid = np.array([(x, z) for z in zs for x in xs])
    values = evaluate_strip_interior_field(mesh, phi_panel, q_panel, grid, order=order)
    cosh_kH = math.cosh(km * H)
    exact = np.cos(km * grid[:, 0]) * np.cosh(km * (grid[:, 1] + H)) / cosh_kH
    return StripSolveField(points=grid, potential=values, exact=exact)


def dtn_depth_factor(
    k: float,
    depth: float,
    *,
    panels_per_wavelength: int = 32,
    order: int = 12,
    mode: int = 1,
) -> float:
    """Depth factor from the *assembled* Dirichlet-to-Neumann map.

    ``Lambda = <T c, c> / <c, c>`` with ``c_i = cos(k_m x_i)`` on the free-surface
    panels.  This route never calls :func:`strip_bem_depth_factor`: it is an
    independent code path over the same layer matrices, so equality of the two
    routes to round-off is a check of the DtN assembly.
    """
    k = _positive("wavenumber", k)
    H = _positive("water depth", depth)
    km = int(mode) * k
    t, mesh = dirichlet_to_neumann(
        k, H, panels_per_wavelength=panels_per_wavelength, order=order
    )
    fs = mesh.index["free_surface"]
    x_fs = mesh.centres[fs, 0]
    lengths = mesh.lengths[fs]
    c = np.cos(km * x_fs)
    tc = t @ c
    return float(np.sum(tc * c * lengths) / np.sum(c * c * lengths))


def strip_bem_mode_sweep(
    k: float,
    depth: float,
    *,
    n_modes: int = 3,
    panels_per_wavelength: int = 32,
    order: int = 12,
) -> list[dict[str, float]]:
    """Depth factors of the first ``n_modes`` periodic modes of one cell.

    Mode ``m`` is the driven solve with ``phi = cos(m k x)`` prescribed on the
    free surface of the cell of length ``L = 2 pi / k``, so its analytic depth
    factor is ``m k tanh(m k H)``.  Higher modes are resolved by fewer panels per
    wavelength in the same mesh, which is why their error is larger -- the sweep
    measures that loss of resolution rather than the physics.
    """
    rows = []
    for m in range(1, int(n_modes) + 1):
        solve = strip_bem_depth_factor(
            k, depth, panels_per_wavelength=panels_per_wavelength, order=order,
            mode=m,
        )
        rows.append(
            {
                "mode": float(m),
                "wavenumber": solve.wavenumber,
                "panels_per_wavelength_effective": float(panels_per_wavelength)
                / float(m),
                "depth_factor_bem": solve.depth_factor_rq,
                "depth_factor_exact": solve.exact_depth_factor,
                "rel_error": solve.depth_factor_error,
                "panel_scatter": solve.panel_scatter,
                "boundary_value_error": solve.bv_error,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# got side: BEM dispersion
# ---------------------------------------------------------------------------
def _route_depth_factor(
    k: float,
    depth: float,
    *,
    route: str = "driven",
    panels_per_wavelength: int = 32,
) -> float:
    """Depth factor of mode 1 by the requested numerical route."""
    if route == "driven":
        return strip_bem_depth_factor(
            k, depth, panels_per_wavelength=panels_per_wavelength
        ).depth_factor_rq
    if route == "dtn":
        return dtn_depth_factor(
            k, depth, panels_per_wavelength=panels_per_wavelength
        )
    raise ValueError(f"unknown route {route!r}; use 'driven' or 'dtn'")


def strip_bem_mass_loading_omega2(
    k: float,
    *,
    depth: float,
    mu: float,
    gravity: float = DEFAULT_GRAVITY,
    panels_per_wavelength: int = 32,
    route: str = "driven",
) -> float:
    """``omega^2`` from the BEM depth factor (the "got" side of the benchmark).

    ``route='driven'`` uses the Rayleigh quotient of the driven mode solve,
    ``route='dtn'`` the same quotient evaluated through the assembled
    Dirichlet-to-Neumann map.
    """
    lam = _route_depth_factor(
        k, depth, route=route, panels_per_wavelength=panels_per_wavelength
    )
    return mass_loading_omega2_from_depth_factor(lam, mu=mu, gravity=gravity)


def strip_bem_mass_loading_omega(
    k: float,
    *,
    depth: float,
    mu: float,
    gravity: float = DEFAULT_GRAVITY,
    panels_per_wavelength: int = 32,
    route: str = "driven",
) -> float:
    """``omega`` (rad/s) from the BEM depth factor."""
    return math.sqrt(
        strip_bem_mass_loading_omega2(
            k,
            depth=depth,
            mu=mu,
            gravity=gravity,
            panels_per_wavelength=panels_per_wavelength,
            route=route,
        )
    )


def dispersion_row(
    k: float,
    *,
    depth: float,
    ice_thickness: float,
    ice_density: float = DEFAULT_ICE_DENSITY,
    water_density: float = DEFAULT_WATER_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    panels_per_wavelength: int = 32,
    route: str = "driven",
) -> dict[str, float]:
    """One ref-vs-got row of the mass-loading dispersion benchmark."""
    mu = mass_loading_mu(
        ice_thickness, ice_density=ice_density, water_density=water_density
    )
    solve = strip_bem_depth_factor(
        k, depth, panels_per_wavelength=panels_per_wavelength
    )
    lam = _route_depth_factor(
        k, depth, route=route, panels_per_wavelength=panels_per_wavelength
    )
    other = "dtn" if route == "driven" else "driven"
    lam_other = _route_depth_factor(
        k, depth, route=other, panels_per_wavelength=panels_per_wavelength
    )
    omega2 = mass_loading_omega2_from_depth_factor(lam, mu=mu, gravity=gravity)
    omega_ref2 = weitz_keller_omega2(k, depth=depth, mu=mu, gravity=gravity)
    omega_deep2 = weitz_keller_deep_omega2(k, mu=mu, gravity=gravity)
    cutoff = mass_loading_cutoff_omega(mu, gravity=gravity)
    return {
        "k": float(k),
        "kH": float(k * depth),
        "wavelength": _TWO_PI / float(k),
        "depth": float(depth),
        "mu": mu,
        "depth_factor_bem": lam,
        "depth_factor_bem_other_route": lam_other,
        "route": route,
        "depth_factor_routes_rel_diff": lam_other / lam - 1.0,
        "depth_factor_ref": solve.exact_depth_factor,
        "depth_factor_rel_error": lam / solve.exact_depth_factor - 1.0,
        "omega2_ref": omega_ref2,
        "omega2_got": omega2,
        "omega_ref": math.sqrt(omega_ref2),
        "omega_got": math.sqrt(omega2),
        "omega_rel_error": math.sqrt(omega2) / math.sqrt(omega_ref2) - 1.0,
        "omega2_deep_limit": omega_deep2,
        "omega_deep_rel_error": math.sqrt(omega_deep2) / math.sqrt(omega_ref2) - 1.0,
        "omega_long_wave": long_wave_omega(k, depth=depth, gravity=gravity),
        "cutoff_omega": cutoff,
        "omega_over_cutoff": math.sqrt(omega_ref2) / cutoff,
        "n_panels": float(solve.n_panels),
        "panels_per_wavelength": float(solve.panels_per_wavelength),
        "panel_scatter": solve.panel_scatter,
        "boundary_value_error": solve.bv_error,
    }


def long_wave_rel_error(
    k: float,
    *,
    depth: float,
    ice_thickness: float,
    ice_density: float = DEFAULT_ICE_DENSITY,
    water_density: float = DEFAULT_WATER_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> float:
    """Relative error of the long-wave anchor against the full relation (3)."""
    mu = mass_loading_mu(
        ice_thickness, ice_density=ice_density, water_density=water_density
    )
    ref = weitz_keller_omega(k, depth=depth, mu=mu, gravity=gravity)
    return long_wave_omega(k, depth=depth, gravity=gravity) / ref - 1.0


def convergence_study(
    k: float,
    *,
    depth: float,
    levels: tuple[int, ...] = (8, 16, 32, 64),
) -> list[dict[str, float]]:
    """Depth-factor error of the driven BEM route against panel refinement."""
    rows = []
    for level in levels:
        solve = strip_bem_depth_factor(k, depth, panels_per_wavelength=level)
        rows.append(
            {
                "panels_per_wavelength": float(level),
                "n_panels": float(solve.n_panels),
                "depth_factor_bem": solve.depth_factor_rq,
                "depth_factor_exact": solve.exact_depth_factor,
                "rel_error": solve.depth_factor_error,
                "abs_rel_error": abs(solve.depth_factor_error),
                "panel_scatter": solve.panel_scatter,
            }
        )
    for prev, cur in zip(rows[:-1], rows[1:]):
        if cur["abs_rel_error"] > 0.0 and prev["abs_rel_error"] > 0.0:
            cur["observed_order"] = float(
                math.log(prev["abs_rel_error"] / cur["abs_rel_error"]) / math.log(2.0)
            )
        else:
            cur["observed_order"] = float("nan")
    return rows


def ice_thickness_sensitivity(
    k: float,
    *,
    depth: float,
    thicknesses: tuple[float, ...] = (0.5, 1.0, 2.0),
    ice_density: float = DEFAULT_ICE_DENSITY,
    water_density: float = DEFAULT_WATER_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    panels_per_wavelength: int = 32,
) -> list[dict[str, float]]:
    """``omega`` vs ice thickness: the mass-loading sensitivity of the case."""
    rows = []
    for h in thicknesses:
        row = dispersion_row(
            k,
            depth=depth,
            ice_thickness=h,
            ice_density=ice_density,
            water_density=water_density,
            gravity=gravity,
            panels_per_wavelength=panels_per_wavelength,
        )
        row["ice_thickness"] = float(h)
        rows.append(row)
    return rows


def depth_sensitivity(
    k: float,
    *,
    depths: tuple[float, ...] = (2.0, 5.0, 20.0, 200.0),
    ice_thickness: float = 1.0,
    ice_density: float = DEFAULT_ICE_DENSITY,
    water_density: float = DEFAULT_WATER_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    panels_per_wavelength: int = 32,
) -> list[dict[str, float]]:
    """``omega`` vs water depth: shallow-to-deep transition at fixed ``k``."""
    rows = []
    for H in depths:
        row = dispersion_row(
            k,
            depth=H,
            ice_thickness=ice_thickness,
            ice_density=ice_density,
            water_density=water_density,
            gravity=gravity,
            panels_per_wavelength=panels_per_wavelength,
        )
        row["depth"] = float(H)
        rows.append(row)
    return rows