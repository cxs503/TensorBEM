"""Patch tensorbem/neumann_kelvin.py: correct PV wave integral + kernel consistency."""
from __future__ import annotations

import subprocess
from pathlib import Path

p = Path("src/tensorbem/neumann_kelvin.py")
lines = p.read_text().split("\n")


def replace_span(first_marker: str, last_marker: str, new_lines: list[str]) -> None:
    i = next(k for k, l in enumerate(lines) if first_marker in l)
    j = next(k for k in range(i, len(lines)) if last_marker in lines[k])
    lines[i:j + 1] = new_lines


# --- 1. free-surface condition + kernel status note in the module docstring ---
replace_span(
    "d(phi)/dz - kappa d^2(phi)/dx^2",
    "the frequency-domain condition.",
    [
        "    K d(phi)/dz + d^2(phi)/dx^2 = 0          on z = 0,        K = g/U^2,",
        "    d(phi)/dn = -U n_x                        on S_B,",
        "",
        "with ``K = g/U^2`` (units 1/length).  The condition follows from the steady",
        "free-surface conditions ``zeta = -U phi_x / g`` and ``zeta_x = phi_z / U``,",
        "and is *not* the frequency-domain condition: on a plane wave",
        "``exp(i(alpha x + beta y) + k z)`` it requires ``alpha^2 = K k``, i.e. the",
        "deep-water dispersion relation ``omega^2 = g k`` with ``omega = U alpha``",
        "in the body frame.  On the Kelvin curve that reads ``k = K sec^2(theta)``.",
        "",
        "Kernel status (verification debt, measured; see",
        "``docs/dev_report_m16913_nk_kernel_audit.md``)",
        "---------------------------------------------------",
        "The wave part implemented below,",
        "",
        "    G_W = (K/2pi) PV int_0^inf exp(-K u |z+zeta|) J_0(K u R) / (u - 1) du,",
        "",
        "is the *frequency-domain* Havelock deep-water kernel (radial ``J_0(k0 R)``",
        "dependence, single pole at the radial wavenumber ``k0 = K``), not the steady",
        "Kelvin kernel: it is isotropic in ``R``, while the steady wave field lives",
        "only on the one-parameter Kelvin family ``k = K sec^2 theta``.  Measured",
        "consequence (machine-checked in ``tests/test_nk_wave_kernel_verification.py``):",
        "the steady free-surface residual ``K G_z + G_xx`` of this kernel is 2 %-60 %",
        "of the term scale and does *not* vanish under quadrature refinement.  The",
        "quadrature that used to evaluate the integral was invalid as well (naive",
        "``eps`` split at ``u = 1`` whose two halves each diverge logarithmically, and",
        "Gauss-Laguerre weights taken for the wrong exponential weight); that defect",
        "*is* fixed here, which is what makes the residual measurement meaningful.",
        "",
        "Therefore ``solve_nk_wave_resistance`` returns a *number*, not evidence: its",
        "``C_w`` comes from a kernel that violates the boundary condition it is meant",
        "to enforce, so the capability is carried as WITHHELD until the Kelvin kernel",
        "replaces ``G_W``.  :func:`steady_free_surface_residual` is the referee for",
        "that rebuild.",
    ],
)

# --- 2. Green function paragraph ---
replace_span(
    "The wave integral is evaluated by splitting at",
    "on ``[0, 1-eps]`` and Gauss-Laguerre on ``[1+eps, infty)``.",
    [
        "The wave integral is a principal value at ``u = 1``.  It is evaluated by the",
        "log-divergence-subtracted decomposition of",
        ":func:`_pv_wave_integral_batch` (both halves of the naive ``eps`` split",
        "diverge like ``+/- log(eps)`` and cancel only in the limit), anchored on the",
        "closed form for ``f = exp(-mu u)``,",
        "``PV int_0^inf exp(-mu u)/(u-1) du = exp(-mu) [Ein(-mu) - log(mu) - gamma]``.",
    ],
)

src = "\n".join(lines)

# --- 3. replace the wave integral + both Green functions ---------------------
start = src.index("def _wave_integral_steady(")
end = src.index("# ============================================================================\n# Neumann-Kelvin BEM solver")
new_block = '''_EULER_GAMMA = 0.5772156649015328606


def _pv_node_counts(q: float, mu: float, node_floor: int = 48, node_ceil: int = 512) -> int:
    """Quadrature order sufficient for an oscillatory J_0(q u) integrand.

    Both pieces of the principal value carry the Bessel oscillation, so the
    order has to grow with ``q`` (one node per ~0.7 rad of phase); ``mu`` only
    shortens the range, which the mapped quadrature absorbs.
    """
    need = 1.5 * max(q, 0.0) + 4.0 * max(mu, 0.0) + 32.0
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
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    q = np.atleast_1d(np.asarray(q, dtype=float))
    if mu.shape != q.shape:
        mu, q = np.broadcast_arrays(mu, q)

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
    mu = kappa * np.abs(np.asarray(z_sum, dtype=float))
    q = kappa * np.abs(np.asarray(R, dtype=float))
    return (kappa / _TWO_PI) * _pv_wave_integral_batch(mu, q, n)


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
        n = _pv_node_counts(kappa * float(np.max(R)), kappa * float(np.max(np.abs(dz_img))))
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
    """Stationary-phase angle of the Kelvin phase: ``tan^2(theta_s) = 1/2``.

    The far-field phase of the steady wave family is
    ``phi(theta) = k(theta) (x cos theta + y sin theta)`` with
    ``k = K sec^2 theta``; ``dphi/dtheta = 0`` there gives
    ``tan^2 theta_s = 1/2``, i.e. ``theta_s = arcsin(1/3) = 19.47 deg`` -- the
    Kelvin half-angle, recovered from the dispersion relation alone.
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


'''
src = src[:start] + new_block + src[end:]

# --- 4. solver result carries the kernel status -----------------------------
old_dc = "    panel_count: int\n    condition_number: float\n    relative_residual: float"
new_dc = old_dc + "\n    kernel_status: str = KERNEL_STATUS"
assert old_dc in src
src = src.replace(old_dc, new_dc)

anchor = "@dataclass(frozen=True)\nclass NeumannKelvinResult:"
const = ('KERNEL_STATUS = (\n'
         '    "WITHHELD: G_W is the frequency-domain Havelock kernel (radial, k0 = K); "\n'
         '    "the steady free-surface condition K phi_z + phi_xx = 0 is not satisfied, "\n'
         '    "so the returned C_w is not evidence -- see steady_free_surface_residual"\n'
         ')\n\n')
assert anchor in src
src = src.replace(anchor, const + anchor)
assert src.index("KERNEL_STATUS = (") < src.index("class NeumannKelvinResult")

# --- 5. drop the now-unused Laguerre import --------------------------------
src = src.replace("from numpy.polynomial.laguerre import laggauss\n", "")
src = src.replace("_LAG64_X, _LAG64_W = laggauss(64)\n", "")

p.write_text(src)
print("patched", len(src), "bytes")
r = subprocess.run(
    ["python3", "-c",
     "import sys; sys.path.insert(0,'src'); import tensorbem.neumann_kelvin as nk;"
     "print(nk.KERNEL_STATUS[:50]);"
     "print(nk._wave_integral_steady(1.0,-0.5,5.0));"
     "print(nk.kelvin_stationary_angle_rad()*180/3.141592653589793)"],
    capture_output=True, text=True)
print(r.stdout, r.stderr[-1500:])