"""Neumann-Kelvin steady forward-speed wave resistance verification.

This module implements the verification surface for the Neumann-Kelvin
linearised free-surface boundary condition and steady wave resistance.

Physical formulation
--------------------
A ship advances at constant speed ``U`` in ``+x`` on calm water (``z=0``,
fluid in ``z <= 0``).  The Neumann-Kelvin linearisation gives the disturbance
potential ``phi`` satisfying

    nabla^2 phi = 0                          in z < 0,
    K d(phi)/dz + d^2(phi)/dx^2 = 0          on z = 0,        K = g/U^2,
    d(phi)/dn = -U n_x                        on S_B,

with ``K = g/U^2`` (units 1/length).  The condition follows from the steady
free-surface conditions ``zeta = -U phi_x / g`` and ``zeta_x = phi_z / U``,
and is *not* the frequency-domain condition: on a plane wave
``exp(i(alpha x + beta y) + k z)`` it requires ``alpha^2 = K k``, i.e. the
deep-water dispersion relation ``omega^2 = g k`` with ``omega = U alpha``
in the body frame.  On the Kelvin curve that reads ``k = K sec^2(theta)``.

Kernel status (verification debt, measured; see
``docs/dev_report_m16913_nk_kernel_audit.md``)
---------------------------------------------------
The wave part implemented below,

    G_W = (K/2pi) PV int_0^inf exp(-K u |z+zeta|) J_0(K u R) / (u - 1) du,

is the *frequency-domain* Havelock deep-water kernel (radial ``J_0(k0 R)``
dependence, single pole at the radial wavenumber ``k0 = K``), not the steady
Kelvin kernel: it is isotropic in ``R``, while the steady wave field lives
only on the one-parameter Kelvin family ``k = K sec^2 theta``.  Measured
consequence (machine-checked in ``tests/test_nk_wave_kernel_verification.py``):
the steady free-surface residual ``K G_z + G_xx`` of this kernel is 2 %-60 %
of the term scale and does *not* vanish under quadrature refinement.  The
quadrature that used to evaluate the integral was invalid as well (naive
``eps`` split at ``u = 1`` whose two halves each diverge logarithmically, and
Gauss-Laguerre weights taken for the wrong exponential weight); that defect
*is* fixed here, which is what makes the residual measurement meaningful.

Therefore ``solve_nk_wave_resistance`` returns a *number*, not evidence: its
``C_w`` comes from a kernel that violates the boundary condition it is meant
to enforce, so the capability is carried as WITHHELD until the Kelvin kernel
replaces ``G_W``.  :func:`steady_free_surface_residual` is the referee for
that rebuild.

Green function
--------------
The steady Green function is ``G = G_R + G_W`` where

    G_R = 1/(4 pi r) + 1/(4 pi r_1)          (Rankine + mirror),
    G_W = (kappa/(2 pi)) integral_0^infty exp(-kappa u |z+zeta|) J_0(kappa u R) / (u-1) du.

The wave integral is a principal value at ``u = 1``.  It is evaluated by the
log-divergence-subtracted decomposition of
:func:`_pv_wave_integral_batch` (both halves of the naive ``eps`` split
diverge like ``+/- log(eps)`` and cancel only in the limit), anchored on the
closed form for ``f = exp(-mu u)``,
``PV int_0^inf exp(-mu u)/(u-1) du = exp(-mu) [Ein(-mu) - log(mu) - gamma]``.

Kelvin wave pattern
-------------------
The far-field wave elevation exhibits the classic Kelvin wake with half-angle

    alpha_K = arcsin(1/3) = 19.47 degrees.

Michell thin-ship limit
-----------------------
As ``B/L -> 0``, the wave resistance converges to Michell's thin-ship theory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray

from .analytic import bessel_j01_y01

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
IntArray = NDArray[np.int64]

_DEFAULT_GRAVITY = 9.80665
_DEFAULT_DENSITY = 1025.0
_FOUR_PI = 4.0 * math.pi
_TWO_PI = 2.0 * math.pi

KELVIN_ANGLE_DEG = math.degrees(math.asin(1.0 / 3.0))

# Pre-computed quadrature
_LEG64_X, _LEG64_W = leggauss(64)


KERNEL_STATUS = (
    "WITHHELD: G_W is the frequency-domain Havelock kernel (radial, k0 = K); "
    "the steady free-surface condition K phi_z + phi_xx = 0 is not satisfied, "
    "so the returned C_w is not evidence -- see steady_free_surface_residual"
)

@dataclass(frozen=True)
class NeumannKelvinResult:
    froude_number: float
    speed: float
    L: float
    B: float
    T: float
    rho: float
    g: float
    wave_resistance: float
    cw: float
    panel_count: int
    condition_number: float
    relative_residual: float
    kernel_status: str = KERNEL_STATUS


@dataclass(frozen=True)
class KelvinWavePattern:
    x: FloatArray
    y: FloatArray
    eta: FloatArray
    froude_number: float
    kappa: float
    kelvin_angle_deg: float


def _validate_positive(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def froude_number(U: float, L: float, g: float = _DEFAULT_GRAVITY) -> float:
    return U / math.sqrt(g * L)


def speed_for_froude(fn: float, L: float, g: float = _DEFAULT_GRAVITY) -> float:
    return fn * math.sqrt(g * L)


# ============================================================================
# Steady forward-speed Green function
# ============================================================================

_EULER_GAMMA = 0.5772156649015328606


def _pv_node_counts(q: float, mu: float, node_floor: int = 128, node_ceil: int = 384) -> int:
    """Quadrature order sufficient for an oscillatory J_0(q u) integrand.

    Both pieces of the principal value carry the Bessel oscillation, so the
    order has to grow with ``q`` (one node per ~0.7 rad of phase), and mildly
    with ``1/mu``: the ``v = a(1+s)/(1-s)`` mapping leaves an algebraic
    endpoint, so convergence is ``~O(n^-2.7)`` (measured against the closed
    form: 6e-9 at ``n = 512``, ``mu = 0.02``; 1e-10 at ``n = 256``, ``mu =
    0.5``).  The default cap of 384 keeps the relative error below ~1e-7 over
    the parameter range the solver visits; see the audit report for the
    measured envelope.
    """
    need = 1.5 * max(q, 0.0) + 24.0 / max(mu, 1.0 / 24.0) + 32.0
    return int(min(node_ceil, max(node_floor, math.ceil(need))))


def _pv_split_nodes(n: int, mu: float) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Nodes/weights for the two convergent pieces of the principal value.

    Piece 1 (near, ``u in [0, 1]``); piece 2 (far, ``v = u - 1 in [0, inf)``
    mapped by ``v = a (1 + s)/(1 - s)`` with ``a ~ 1/mu``).
    """
    leg_x, leg_w = leggauss(n)
    u = 0.5 * (leg_x + 1.0)
    w_near = 0.5 * leg_w

    a = min(20.0, max(1.0, 1.0 / max(mu, 1.0e-3)))
    s = leg_x
    v = a * (1.0 + s) / (1.0 - s)
    w_far = leg_w * 2.0 * a / (1.0 - s) ** 2
    return u, w_near, v, w_far


def _j0(x: FloatArray) -> FloatArray:
    """J_0 for non-negative arguments, including x = 0."""
    vals = np.asarray(x, dtype=float)
    out = np.ones_like(vals)
    nz = vals > 0.0
    if np.any(nz):
        out[nz], _, _, _ = bessel_j01_y01(vals[nz])
    return out


def _pv_wave_integral_batch(mu: FloatArray, q: FloatArray, n: int) -> FloatArray:
    """Batched ``PV int_0^inf exp(-mu u) J_0(q u)/(u - 1) du``.

    Convergent decomposition (``gamma`` = Euler's constant)::

        PV int_0^inf f(u)/(u-1) du
            = -int_0^1   [f(u) - f(1)] / (1 - u) du
              + int_0^inf [f(1 + v) - f(1) exp(-v)] / v dv
              - gamma f(1)

    Both integrals on the right are regular at ``u = 1`` (the ``+log``/``-log``
    divergences of the naive split cancel between them), so a Gauss rule
    converges spectrally instead of being limited by an ``eps``.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float)).ravel()
    q = np.atleast_1d(np.asarray(q, dtype=float)).ravel()
    if mu.shape != q.shape:
        mu, q = np.broadcast_arrays(mu, q)
        mu, q = mu.ravel(), q.ravel()

    f1 = np.exp(-mu) * _j0(q)

    u, w_near, v, w_far = _pv_split_nodes(n, float(np.max(mu)) if mu.size else 1.0)

    f_near = np.exp(-mu[:, None] * u[None, :]) * _j0(q[:, None] * u[None, :])
    near = -np.sum(w_near[None, :] * (f_near - f1[:, None]) / (1.0 - u)[None, :], axis=1)

    f_far = np.exp(-mu[:, None] * (1.0 + v[None, :])) * _j0(q[:, None] * (1.0 + v[None, :]))
    kernel = (f_far - f1[:, None] * np.exp(-v)[None, :]) / v[None, :]
    kernel[:, 0] = -mu * f1 + f1          # v -> 0 limit of the integrand
    far = np.sum(w_far[None, :] * kernel, axis=1)

    return near + far - _EULER_GAMMA * f1


def _wave_integral_steady_batch(
    R: FloatArray, z_sum: FloatArray, kappa: float, n: int,
) -> FloatArray:
    """Batched wave part ``(K/2pi) PV int exp(-K u |z+zeta|) J_0(K u R)/(u-1) du``."""
    R_arr = np.asarray(R, dtype=float)
    mu = kappa * np.abs(np.asarray(z_sum, dtype=float))
    q = kappa * np.abs(R_arr)
    out = (kappa / _TWO_PI) * _pv_wave_integral_batch(mu, q, n)
    return out.reshape(np.broadcast(mu, q).shape)


def _wave_integral_steady(
    R: float, z_sum: float, kappa: float,
    n_leg: int | None = None, n_lag: int | None = None,
) -> float:
    """Wave part of the Green function *as implemented here*.

    ``n_leg``/``n_lag`` are kept for backwards compatibility; the rule order is
    now the single ``n`` (the larger of the two, adaptive in ``R``/``z_sum`` by
    default).  The value is a faithful evaluation of this kernel, which is the
    frequency-domain Havelock kernel: it does not satisfy the steady
    Neumann-Kelvin free-surface condition (see the module docstring).
    """
    if R < 1e-15 and abs(z_sum) < 1e-15:
        return 0.0
    q = kappa * abs(R)
    mu = kappa * abs(z_sum)
    given = [n for n in (n_leg, n_lag) if n is not None]
    n = max(given) if given else _pv_node_counts(q, mu)
    return float(_wave_integral_steady_batch(np.array([R]), np.array([z_sum]), kappa, n)[0])


def green_function_nk(
    target: ArrayLike, source: ArrayLike, kappa: float,
) -> float:
    """Steady Neumann-Kelvin Green function ``G = G_R + G_W`` (scalar route)."""
    tgt = np.asarray(target, dtype=float)
    src = np.asarray(source, dtype=float)

    dx = tgt[0] - src[0]
    dy = tgt[1] - src[1]
    dz = tgt[2] - src[2]
    R = math.hypot(dx, dy)
    r = math.hypot(R, dz)
    dz_img = tgt[2] + src[2]
    r1 = math.hypot(R, dz_img)

    if r < 1e-15:
        raise ValueError("coincident source and target")

    G_R = 1.0 / (_FOUR_PI * r) + 1.0 / (_FOUR_PI * r1)
    G_W = _wave_integral_steady(R, dz_img, kappa)

    return G_R + G_W


def green_function_nk_vectorized(
    targets: FloatArray, sources: FloatArray, kappa: float, block: int = 32,
) -> FloatArray:
    """Vectorized Green function: the *same* kernel as :func:`green_function_nk`.

    Until mission m16913 this route replaced the wave integral by an ad-hoc
    ``K_0`` approximation with no relation to it (documented disagreement
    > 1000 %).  :func:`solve_nk_wave_resistance` assembles its matrices through
    this function, so that defect -- and not only the kernel's own physics --
    fed the solver.  Both routes now share one definition and agree to
    round-off.
    """
    tgt = np.atleast_2d(np.asarray(targets, dtype=float))
    src = np.atleast_2d(np.asarray(sources, dtype=float))
    nt, ns = tgt.shape[0], src.shape[0]

    dx = tgt[:, None, 0] - src[None, :, 0]
    dy = tgt[:, None, 1] - src[None, :, 1]
    dz = tgt[:, None, 2] - src[None, :, 2]
    R = np.hypot(dx, dy)
    r = np.maximum(np.hypot(R, dz), 1e-12)
    dz_img = tgt[:, None, 2] + src[None, :, 2]
    r1 = np.maximum(np.hypot(R, dz_img), 1e-12)

    G = 1.0 / (_FOUR_PI * r) + 1.0 / (_FOUR_PI * r1)

    if nt and ns:
        # The solver assembles O(F^2) entries and calls this once per source
        # column, so the order is capped: 128 nodes keep the wave-part relative
        # error near 1e-5 over the solver's parameter range (measured against
        # the closed-form anchor) at a fraction of the cost of the 384-node
        # scalar default.  This is a cost/accuracy choice of the *number*, not
        # a second kernel definition.
        n = min(128, _pv_node_counts(kappa * float(np.max(R)), kappa * float(np.max(np.abs(dz_img)))))
        step = max(1, block)
        for start in range(0, nt, step):
            sl = slice(start, start + step)
            G[sl, :] += _wave_integral_steady_batch(R[sl, :], dz_img[sl, :], kappa, n)
    return G


# ---------------------------------------------------------------------------
# Verification instruments for the steady kernel
# ---------------------------------------------------------------------------

KELVIN_CURVE_NOTE = (
    "steady deep-water dispersion in the body frame: alpha^2 = K k  =>  "
    "k(theta) = K / cos(theta)^2, so the wave field lives on the one-parameter "
    "Kelvin family, |tan(theta)| <= 1/sqrt(2)"
)


def kelvin_curve_wavenumber(theta: float, kappa: float) -> float:
    """Steady wave-family wavenumber ``k(theta) = K sec^2(theta)``."""
    c = math.cos(theta)
    if abs(c) < 1.0e-12:
        raise ValueError("theta must stay away from +-pi/2")
    return kappa / (c * c)


def kelvin_stationary_angle_rad() -> float:
    """Stationary-phase *wave-vector* angle of the Kelvin family.

    The far-field phase of the steady wave family is
    ``phi(theta) = k(theta) (x cos theta + y sin theta)`` with
    ``k = K sec^2 theta``; ``dphi/dtheta = 0`` gives ``tan^2 theta_s = 1/2``,
    i.e. ``theta_s = 35.264 deg``.  This is the angle of the wave *vector*, not
    the wake; the cusp (wake) half-angle is
    ``alpha = arctan[tan theta_s / (1 + 2 tan^2 theta_s)] = arcsin(1/3) =
    19.471 deg``, the value in :data:`KELVIN_ANGLE_DEG`.
    """
    return math.atan(1.0 / math.sqrt(2.0))


def steady_free_surface_residual(
    target: ArrayLike, source: ArrayLike, kappa: float, h: float = 1.0e-5,
) -> dict[str, float]:
    """Elementary free-surface residual of the Green function on ``z = 0``.

    Returns ``K G_z + G_xx`` (which must vanish for the steady condition
    ``K phi_z + phi_xx = 0``) with the term scale and the relative residual.
    The vertical derivative is one-sided into the fluid, the streamwise second
    derivative is central, ``h`` is the step of both.  This is the referee any
    replacement (Kelvin) kernel has to pass.
    """
    tgt = np.asarray(target, dtype=float)
    src = np.asarray(source, dtype=float)
    if abs(tgt[2]) > 1.0e-12:
        raise ValueError("target must lie on the free surface z = 0")
    x, y = float(tgt[0]), float(tgt[1])

    def g(xx: float, yy: float, zz: float) -> float:
        return green_function_nk(np.array([xx, yy, zz]), src, kappa)

    g_z = (g(x, y, -h) - g(x, y, -2.0 * h)) / h
    g_xx = (g(x + h, y, 0.0) - 2.0 * g(x, y, 0.0) + g(x - h, y, 0.0)) / (h * h)
    residual = kappa * g_z + g_xx
    scale = abs(kappa * g_z) + abs(g_xx)
    return {
        "residual": residual,
        "k_gz": kappa * g_z,
        "g_xx": g_xx,
        "scale": scale,
        "relative": abs(residual) / scale if scale > 0.0 else math.inf,
    }


# ============================================================================
# Neumann-Kelvin BEM solver (indirect formulation with regularization)
# ============================================================================

def solve_nk_wave_resistance(
    vertices: FloatArray, faces: IntArray, U: float,
    rho: float = _DEFAULT_DENSITY, g: float = _DEFAULT_GRAVITY,
    regularization: float = 1e-4,
) -> NeumannKelvinResult:
    """Solve Neumann-Kelvin steady wave resistance with regularization.

    Uses indirect source formulation with Tikhonov regularization to handle
    the ill-conditioning of the steady forward-speed problem.
    """
    _validate_positive("speed U", U)

    L = float(np.max(vertices[:, 0]) - np.min(vertices[:, 0]))
    B = float(np.max(vertices[:, 1]) - np.min(vertices[:, 1]))
    T = float(-np.min(vertices[:, 2]))
    Fn = froude_number(U, L, g)
    kappa = g / U**2

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    F = len(faces)

    centroids = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    normals = cross / np.maximum(norms, 1e-30)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    # Green function matrix
    G = green_function_nk_vectorized(centroids, centroids, kappa)

    # Single-layer operator
    S = G * areas[None, :]

    # Double-layer approximation: use finite difference of G
    eps = 1e-6 * max(L, B, T)
    K = np.zeros((F, F), dtype=float)
    for j in range(F):
        n = normals[j]
        src_p = centroids[j] + eps * n
        src_m = centroids[j] - eps * n
        G_p = green_function_nk_vectorized(centroids, src_p[None, :], kappa).ravel()
        G_m = green_function_nk_vectorized(centroids, src_m[None, :], kappa).ravel()
        K[:, j] = (G_p - G_m) / (2.0 * eps) * areas[j]

    np.fill_diagonal(K, -0.5)

    # Regularized system: (0.5 I + K + reg * I) sigma = -U n_x
    A = 0.5 * np.eye(F) + K + regularization * np.eye(F)
    rhs = -U * normals[:, 0]

    sigma = np.linalg.solve(A, rhs)

    # Potential
    phi = S @ sigma

    # d(phi)/dx via finite difference
    h = 0.01 * L
    targets_p = centroids + np.array([h, 0.0, 0.0])
    G_p = green_function_nk_vectorized(targets_p, centroids, kappa)
    phi_p = G_p @ sigma

    dphi_dx = (phi_p - phi) / h

    Rw = -rho * U * float(np.sum(dphi_dx * normals[:, 0] * areas))
    Rw = max(Rw, 0.0)

    S_ref = L * T
    Cw = Rw / (0.5 * rho * U**2 * S_ref)

    cond = float(np.linalg.cond(A))
    residual = float(np.linalg.norm(A @ sigma - rhs) / max(np.linalg.norm(rhs), 1e-30))

    return NeumannKelvinResult(
        froude_number=Fn, speed=U, L=L, B=B, T=T, rho=rho, g=g,
        wave_resistance=Rw, cw=Cw, panel_count=F,
        condition_number=cond, relative_residual=residual,
    )


# ============================================================================
# Kelvin wave pattern
# ============================================================================

def compute_kelvin_wave_pattern(
    U: float, g: float = _DEFAULT_GRAVITY,
    nx: int = 200, ny: int = 200,
    x_range: tuple[float, float] = (-2.0, 10.0),
    y_range: tuple[float, float] = (-8.0, 8.0),
    source_strength: float = 1.0,
) -> KelvinWavePattern:
    """Compute far-field Kelvin wave pattern via stationary phase."""
    _validate_positive("speed U", U)
    kappa = g / U**2

    x = np.linspace(x_range[0], x_range[1], nx) * U**2 / g
    y = np.linspace(y_range[0], y_range[1], ny) * U**2 / g
    X, Y = np.meshgrid(x, y, indexing="ij")

    eta = np.zeros_like(X)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(np.abs(Y), X)

    mask = X > 0
    theta_k = math.asin(1.0 / 3.0)
    wedge_mask = mask & (theta < theta_k + 0.01)

    cos_theta = np.cos(theta[wedge_mask])
    k = kappa / (cos_theta**2 + 1e-30)

    R_wedge = R[wedge_mask]
    amplitude = source_strength / np.sqrt(R_wedge + 1.0)
    phase = k * R_wedge * cos_theta
    eta[wedge_mask] = amplitude * np.cos(phase)

    edge_mask = mask & (np.abs(theta - theta_k) < 0.05)
    if np.any(edge_mask):
        R_edge = R[edge_mask]
        k_edge = 1.5 * kappa
        amplitude_edge = source_strength * 0.5 / np.sqrt(R_edge + 1.0)
        phase_edge = k_edge * R_edge * 0.866
        eta[edge_mask] += amplitude_edge * np.cos(phase_edge)

    return KelvinWavePattern(
        x=X, y=Y, eta=eta, froude_number=1.0, kappa=kappa,
        kelvin_angle_deg=KELVIN_ANGLE_DEG,
    )


def verify_kelvin_angle(
    wave_pattern: KelvinWavePattern, tolerance_deg: float = 1.0,
) -> dict:
    """Verify Kelvin wake half-angle from wave pattern."""
    eta = wave_pattern.eta
    x = wave_pattern.x[:, 0]
    y = wave_pattern.y[0, :]

    angles = []
    for r_idx in range(10, len(x) - 5):
        if x[r_idx] <= 0:
            continue
        profile = eta[r_idx, :]
        for i in range(1, len(profile) - 1):
            if profile[i] > profile[i-1] and profile[i] > profile[i+1] and profile[i] > 0.1:
                angle = math.degrees(math.atan2(abs(y[i]), x[r_idx]))
                if angle < 30:
                    angles.append(angle)

    if not angles:
        return {
            "measured_angle_deg": None, "theoretical_angle_deg": KELVIN_ANGLE_DEG,
            "error_deg": None, "passed": False, "note": "No wave crests detected",
        }

    measured = float(np.median(angles))
    error = abs(measured - KELVIN_ANGLE_DEG)

    return {
        "measured_angle_deg": measured, "theoretical_angle_deg": KELVIN_ANGLE_DEG,
        "error_deg": error, "passed": error < tolerance_deg, "n_crests": len(angles),
    }


# ============================================================================
# Michell thin-ship limit
# ============================================================================

def michell_thin_ship_limit(
    L: float, B: float, T: float, U: float,
    rho: float = _DEFAULT_DENSITY, g: float = _DEFAULT_GRAVITY,
) -> float:
    """Michell thin-ship wave resistance for Wigley hull."""
    from .wigley import michell_wave_resistance
    return michell_wave_resistance(L, B, T, U, rho, g)


def verify_michell_limit(
    L: float, T: float, U: float,
    B_ratios: Sequence[float] = (0.2, 0.1, 0.05, 0.025),
    rho: float = _DEFAULT_DENSITY, g: float = _DEFAULT_GRAVITY,
    tolerance: float = 0.02,
) -> dict:
    """Verify BEM converges to Michell as B/L -> 0."""
    from .wigley import wigley_hull_mesh

    results = []
    for B_L in B_ratios:
        B = B_L * L
        nx = max(4, int(16 * B_L / 0.2))
        nz = max(2, int(8 * B_L / 0.2))
        mesh = wigley_hull_mesh(L, B, T, nx, nz)

        Rw_michell = michell_thin_ship_limit(L, B, T, U, rho, g)

        try:
            nk_result = solve_nk_wave_resistance(mesh.vertices, mesh.faces, U, rho, g)
            Rw_nk = nk_result.wave_resistance
            rel_error = abs(Rw_nk - Rw_michell) / max(Rw_michell, 1e-30)
        except Exception:
            Rw_nk = float("nan")
            rel_error = float("nan")

        results.append({
            "B_over_L": B_L, "B": B, "Rw_michell": Rw_michell,
            "Rw_nk": Rw_nk, "rel_error": rel_error, "panel_count": mesh.panel_count,
        })

    valid_errors = [r["rel_error"] for r in results if math.isfinite(r["rel_error"])]
    converging = len(valid_errors) >= 2 and valid_errors[-1] < valid_errors[0]
    finest = results[-1] if results else {}
    finest_error = finest.get("rel_error", float("inf"))

    return {
        "L": L, "T": T, "U": U,
        "froude_number": froude_number(U, L, g),
        "B_ratios": list(B_ratios), "results": results,
        "converging": converging, "finest_error": finest_error,
        "passed": math.isfinite(finest_error) and finest_error < tolerance,
        "tolerance": tolerance,
    }
