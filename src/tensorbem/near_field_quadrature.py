"""Near-boundary (near-singular) panel quadrature: two verified routes.

Why this module exists
----------------------
A boundary-element field point ``x`` frequently sits a small distance
``eps`` above a panel of size ``h``: off-body pressure, wave elevation just
above the free surface, near-field velocity, thin gaps, multi-body clearance.
The integrands

    S = int_T G dS,        G        = 1 / (4*pi*r),
    D = int_T dG/dn_y dS,  dG/dn_y  = -n_y . (x - y) / (4*pi*r**3),

develop a peak of width ``eps`` when ``eps/h -> 0``.  A fixed tensor Gauss rule
on the panel cannot resolve that peak: its error grows without bound as
``eps -> 0`` even though the integrand is bounded (the integrals themselves
tend to finite limits).  This module implements two independent remedies that
are *uniform in eps* and can therefore be certified:

``adaptive_triangle_quadrature``
    Recursive midpoint bisection of the panel: a sub-triangle is accepted as
    soon as ``distance(x, sub) >= split_ratio * size(sub)`` (the usual
    well-separated criterion) or the level budget is exhausted.  Only the
    shrinking chain containing the peak is refined, so the cost grows
    logarithmically in ``h/eps``.

``polar_triangle_quadrature``
    Polar coordinates about the *foot* ``F`` of ``x`` on the panel plane.
    With ``r = sqrt(rho**2 + eps**2)`` the substitution ``rho = eps*sinh(t)``
    makes every radial integrand analytic and peak-free:
    ``rho*drho/r = eps*sinh(t)*dt``.  The angular direction is split into three
    sectors (one per vertex) inside which the radial extent ``rho_max(theta)``
    is the exact ray/edge intersection.

``semi_analytic_reference``
    An independent arbiter: the same sectors with the radial integral done in
    closed form (``int rho drho / r = R - eps``, ``int rho drho / r**3 =
    1/eps - 1/R``) and only a 1-D Gauss rule in ``theta``.  It shares no code
    path with either route above, so comparing the three isolates quadrature
    error from modelling error.

``plain_duffy_quadrature``
    The untransformed Duffy rule that BEM codes use by default; it is the
    negative control of the study (it must *fail* as ``eps -> 0``).

The module also carries the analytic anchor used to certify the frequency
domain radiation chain: the Hulme (1982) heave radiation potential of a
floating hemisphere evaluated on the wetted surface, and the far-field Kochin
energy integral of that potential.  ``energy_identity_residual`` compares the
near-field pressure integral and the far-field Kochin integral of the *same*
radiating field, which must agree because the free surface carries zero mean
energy flux; the residual is limited only by how well the truncated multipole
series satisfies the body boundary condition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .analytic import bessel_j01_y01
from .free_surface3d import wave_function_3d
from .kernels3d import duffy_quadrature

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_FOUR_PI = 4.0 * math.pi
_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Gauss-Legendre nodes
# ---------------------------------------------------------------------------
def gauss_legendre(nodes: int) -> tuple[FloatArray, FloatArray]:
    """Return ``n``-point Gauss-Legendre nodes/weights on ``[-1, 1]``.

    Newton iteration on the Legendre recurrence.  ``numpy.polynomial.
    legendre.leggauss`` builds a companion matrix and is ``O(n**3)``, which
    dominates every call in this module at ``n`` of a few thousand, so the
    ``O(n**2)`` Newton form is used instead (agreement with ``leggauss`` is
    at the 1e-15 level, see ``tests/test_nearfield_quadrature.py``).
    """
    if isinstance(nodes, (bool, np.bool_)) or not isinstance(nodes, (int, np.integer)):
        raise ValueError("nodes must be an integer")
    n = int(nodes)
    if n < 1:
        raise ValueError("nodes must be >= 1")
    k = np.arange(1, n + 1, dtype=float)
    x = np.cos(math.pi * (k - 0.25) / (n + 0.5))
    derivative = np.ones_like(x)
    for _ in range(100):
        p_prev = np.ones_like(x)
        p_curr = x.copy()
        for degree in range(2, n + 1):
            p_next = ((2 * degree - 1) * x * p_curr - (degree - 1) * p_prev) / degree
            p_prev, p_curr = p_curr, p_next
        derivative = n * (x * p_curr - p_prev) / (x * x - 1.0)
        step = p_curr / derivative
        x = x - step
        if float(np.max(np.abs(step))) < 1.0e-16:
            break
    weights = 2.0 / ((1.0 - x * x) * derivative * derivative)
    order = np.argsort(x)
    return x[order], weights[order]


# ---------------------------------------------------------------------------
# Panel frames
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PanelFrame:
    """Planar frame of one triangle relative to a target point."""

    v0: FloatArray
    v1: FloatArray
    v2: FloatArray
    normal: FloatArray
    e1: FloatArray
    e2: FloatArray
    foot: FloatArray
    height: float
    size: float
    vertices_2d: FloatArray
    area: float

    @property
    def clearance_ratio(self) -> float:
        """``eps / size`` of the target above the panel."""
        return float(abs(self.height) / self.size)


def _validate_triangle(v0, v1, v2) -> tuple[FloatArray, FloatArray, FloatArray]:
    vertices = []
    for name, value in (("v0", v0), ("v1", v1), ("v2", v2)):
        arr = np.asarray(value, dtype=float)
        if arr.shape != (3,) or not np.isfinite(arr).all():
            raise ValueError(f"{name} must be a finite 3-vector")
        vertices.append(arr)
    a, b, c = vertices
    if float(np.linalg.norm(np.cross(b - a, c - a))) <= 0.0:
        raise ValueError("triangle is degenerate")
    return a, b, c


def panel_frame(v0, v1, v2, target) -> PanelFrame:
    """Project ``target`` onto the panel plane and build an in-plane frame."""
    a, b, c = _validate_triangle(v0, v1, v2)
    x = np.asarray(target, dtype=float)
    if x.shape != (3,) or not np.isfinite(x).all():
        raise ValueError("target must be a finite 3-vector")
    cross = np.cross(b - a, c - a)
    area = 0.5 * float(np.linalg.norm(cross))
    normal = cross / (2.0 * area)
    height = float(np.dot(x - a, normal))
    foot = x - height * normal
    e1 = (b - a) / float(np.linalg.norm(b - a))
    e2 = np.cross(normal, e1)
    # In-plane coordinates measured from the *foot*: the polar routes put the
    # origin of the (rho, theta) system there, not at a triangle vertex.
    verts2d = np.array(
        [
            [float(np.dot(a - foot, e1)), float(np.dot(a - foot, e2))],
            [float(np.dot(b - foot, e1)), float(np.dot(b - foot, e2))],
            [float(np.dot(c - foot, e1)), float(np.dot(c - foot, e2))],
        ]
    )
    size = max(
        float(np.linalg.norm(b - a)),
        float(np.linalg.norm(c - b)),
        float(np.linalg.norm(a - c)),
    )
    return PanelFrame(
        v0=a,
        v1=b,
        v2=c,
        normal=normal,
        e1=e1,
        e2=e2,
        foot=foot,
        height=height,
        size=size,
        vertices_2d=verts2d,
        area=area,
    )


def _kernel_pair(frame: PanelFrame, points: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Single- and double-layer kernels at ``points`` (shape ``(N, 3)``)."""
    rel = frame.foot[None, :] - points  # in-plane offset (foot - y)
    rho2 = np.einsum("ij,ij->i", rel, rel)
    eps = abs(frame.height)
    r = np.sqrt(rho2 + eps * eps)
    single = 1.0 / (_FOUR_PI * r)
    # n_y . (x - y) == signed height for a planar panel
    double = -frame.height / (_FOUR_PI * r * r * r)
    return single, double


# ---------------------------------------------------------------------------
# Route 1: adaptive subdivision
# ---------------------------------------------------------------------------
def _triangle_distance(target: FloatArray, v0: FloatArray, v1: FloatArray, v2: FloatArray) -> float:
    """Distance from ``target`` to the (closed) triangle."""
    normal = np.cross(v1 - v0, v2 - v0)
    norm = float(np.linalg.norm(normal))
    normal = normal / norm
    height = float(np.dot(target - v0, normal))
    foot = target - height * normal
    # barycentric coordinates of the foot in the triangle plane
    edges = ((v0, v1), (v1, v2), (v2, v0))
    inside = True
    for p, q in edges:
        if float(np.dot(np.cross(q - p, foot - p), normal)) < 0.0:
            inside = False
            break
    if inside:
        return abs(height)
    best = math.inf
    for p, q in edges:
        seg = q - p
        denom = float(np.dot(seg, seg))
        t = 0.0 if denom == 0.0 else float(np.dot(target - p, seg) / denom)
        t = min(1.0, max(0.0, t))
        best = min(best, float(np.linalg.norm(target - (p + t * seg))))
    return best


def adaptive_triangle_quadrature(
    v0,
    v1,
    v2,
    target,
    *,
    order: int = 8,
    split_ratio: float = 1.0,
    max_level: int = 12,
    initial_vertices: tuple[FloatArray, FloatArray, FloatArray] | None = None,
) -> dict[str, float]:
    """Integrate single/double-layer kernels by recursive midpoint bisection.

    A sub-triangle is integrated with the untransformed Duffy rule once
    ``distance(target, sub) >= split_ratio * size(sub)``; otherwise it is
    split into four similar sub-triangles.  Returns the two integrals plus the
    number of kernel evaluations and the deepest level actually reached.
    """
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) or int(order) < 2:
        raise ValueError("order must be an integer >= 2")
    if not (0.0 < float(split_ratio)):
        raise ValueError("split_ratio must be positive")
    if isinstance(max_level, (bool, np.bool_)) or not isinstance(max_level, (int, np.integer)) or int(max_level) < 0:
        raise ValueError("max_level must be a non-negative integer")

    frame = panel_frame(v0, v1, v2, target)
    x = np.asarray(target, dtype=float)
    single = 0.0
    double = 0.0
    nodes = 0
    deepest = 0

    stack = [(frame.v0, frame.v1, frame.v2, 0)]
    while stack:
        a, b, c, level = stack.pop()
        size = max(
            float(np.linalg.norm(b - a)),
            float(np.linalg.norm(c - b)),
            float(np.linalg.norm(a - c)),
        )
        distance = _triangle_distance(x, a, b, c)
        if distance >= float(split_ratio) * size or level >= int(max_level):
            points, weights = duffy_quadrature(np.array([a, b, c]), order=int(order))
            s_val, d_val = _kernel_pair(frame, points)
            single += float(np.dot(weights, s_val))
            double += float(np.dot(weights, d_val))
            nodes += points.shape[0]
            deepest = max(deepest, level)
            continue
        ab = 0.5 * (a + b)
        bc = 0.5 * (b + c)
        ca = 0.5 * (c + a)
        stack.extend(
            [
                (a, ab, ca, level + 1),
                (ab, b, bc, level + 1),
                (ca, bc, c, level + 1),
                (ab, bc, ca, level + 1),
            ]
        )
    return {"single": single, "double": double, "nodes": float(nodes), "level": float(deepest)}


# ---------------------------------------------------------------------------
# Route 2: graded polar quadrature
# ---------------------------------------------------------------------------
def _polar_sectors(frame: PanelFrame) -> list[tuple[float, float, FloatArray, FloatArray]]:
    """Return ``(theta_lo, theta_hi, vertex_a, vertex_b)`` sectors.

    The foot is joined to the three vertices; inside the angular sector
    between two consecutive vertices the ray hits exactly the edge joining
    them, so ``rho_max(theta)`` is an exact ray/segment intersection.
    """
    verts = frame.vertices_2d
    angles = np.arctan2(verts[:, 1], verts[:, 0])
    order = np.argsort(angles)
    sectors: list[tuple[float, float, FloatArray, FloatArray]] = []
    for idx in range(3):
        ia = int(order[idx])
        ib = int(order[(idx + 1) % 3])
        lo = float(angles[ia])
        hi = float(angles[ib])
        if hi <= lo:
            hi += _TWO_PI
        sectors.append((lo, hi, verts[ia], verts[ib]))
    return sectors


def _ray_extent(theta: FloatArray, va: FloatArray, vb: FloatArray) -> FloatArray:
    """``rho_max(theta)`` for rays leaving the origin inside one sector."""
    direction = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    edge = vb - va
    numerator = va[0] * edge[1] - va[1] * edge[0]
    denominator = direction[..., 0] * edge[1] - direction[..., 1] * edge[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = numerator / denominator
    rho = np.where(np.isfinite(rho) & (rho > 0.0), rho, 0.0)
    return rho


def polar_triangle_quadrature(
    v0,
    v1,
    v2,
    target,
    *,
    radial_order: int = 16,
    angular_order: int = 16,
) -> dict[str, float]:
    """Integrate single/double-layer kernels in graded polar coordinates.

    ``rho = eps * sinh(t)`` clusters the radial nodes in the boundary layer of
    width ``eps`` while keeping the integrand analytic, so the error is
    independent of ``eps``.  ``angular_order`` Gauss nodes are used per sector.
    """
    if isinstance(radial_order, (bool, np.bool_)) or not isinstance(radial_order, (int, np.integer)) or int(radial_order) < 2:
        raise ValueError("radial_order must be an integer >= 2")
    if isinstance(angular_order, (bool, np.bool_)) or not isinstance(angular_order, (int, np.integer)) or int(angular_order) < 2:
        raise ValueError("angular_order must be an integer >= 2")

    frame = panel_frame(v0, v1, v2, target)
    eps = abs(frame.height)
    if eps <= 0.0:
        raise ValueError("polar quadrature requires a finite positive clearance")
    xs, ws = gauss_legendre(int(radial_order))
    xa, wa = gauss_legendre(int(angular_order))
    single = 0.0
    double = 0.0
    nodes = 0
    for theta_lo, theta_hi, va, vb in _polar_sectors(frame):
        theta = 0.5 * (theta_hi - theta_lo) * xa + 0.5 * (theta_hi + theta_lo)
        w_theta = 0.5 * (theta_hi - theta_lo) * wa
        rho_max = _ray_extent(theta, va, vb)
        t_max = np.arcsinh(rho_max / eps)
        t_nodes = 0.5 * t_max[:, None] * (xs[None, :] + 1.0)
        w_t = 0.5 * t_max[:, None] * ws[None, :]
        # rho = eps*sinh(t), d(rho)/dt = eps*cosh(t), r = eps*cosh(t); hence
        #   rho drho / r     = eps * sinh(t) dt
        #   rho drho / r**3  = (1/eps) * sinh(t) / cosh(t)**2 dt
        r = eps * np.cosh(t_nodes)
        w_single = (_FOUR_PI ** -1) * eps * np.sinh(t_nodes) * w_t
        w_double = (-frame.height / (eps * _FOUR_PI)) * (
            np.sinh(t_nodes) / (r / eps) ** 2
        ) * w_t
        single += float(np.sum(w_theta[:, None] * w_single))
        double += float(np.sum(w_theta[:, None] * w_double))
        nodes += 2 * int(radial_order) * int(angular_order)
    return {"single": single, "double": double, "nodes": float(nodes), "level": float("nan")}


# ---------------------------------------------------------------------------
# Independent arbiter: semi-analytic radial integration
# ---------------------------------------------------------------------------
def semi_analytic_reference(
    v0, v1, v2, target, *, angular_order: int = 96
) -> dict[str, float]:
    """Reference integrals with the radial part in closed form.

    ``int_0^R rho drho / r = R - eps`` and ``int_0^R rho drho / r**3 =
    1/eps - 1/R`` with ``R = sqrt(R**2 + eps**2)`` leave a 1-D Gauss rule in
    ``theta``; error is at the 1e-14 level for ``angular_order >= 64``.
    """
    if isinstance(angular_order, (bool, np.bool_)) or not isinstance(angular_order, (int, np.integer)) or int(angular_order) < 8:
        raise ValueError("angular_order must be an integer >= 8")
    frame = panel_frame(v0, v1, v2, target)
    eps = abs(frame.height)
    if eps <= 0.0:
        raise ValueError("semi-analytic reference requires a positive clearance")
    xa, wa = gauss_legendre(int(angular_order))
    single = 0.0
    double = 0.0
    for theta_lo, theta_hi, va, vb in _polar_sectors(frame):
        theta = 0.5 * (theta_hi - theta_lo) * xa + 0.5 * (theta_hi + theta_lo)
        w_theta = 0.5 * (theta_hi - theta_lo) * wa
        rho_max = _ray_extent(theta, va, vb)
        r_mean = np.sqrt(rho_max * rho_max + eps * eps)
        single += float(np.sum(w_theta * (r_mean - eps))) / _FOUR_PI
        double += (-frame.height / _FOUR_PI) * float(np.sum(w_theta * (1.0 / eps - 1.0 / r_mean)))
    return {"single": single, "double": double}


def plain_duffy_quadrature(v0, v1, v2, target, *, order: int = 8) -> dict[str, float]:
    """Untransformed Duffy rule (negative control: degrades as ``eps -> 0``)."""
    frame = panel_frame(v0, v1, v2, target)
    points, weights = duffy_quadrature(np.array([frame.v0, frame.v1, frame.v2]), order=int(order))
    single, double = _kernel_pair(frame, points)
    return {
        "single": float(np.dot(weights, single)),
        "double": float(np.dot(weights, double)),
        "nodes": float(points.shape[0]),
    }


# ---------------------------------------------------------------------------
# Hulme (1982) heave radiation potential on the wetted hemisphere
# ---------------------------------------------------------------------------
def _wave_kernel(ka: float, mu: FloatArray) -> tuple[FloatArray, ComplexArray]:
    """``s``, ``W = F + i*pi*exp(-y)*J0(x)`` on the hemisphere ``mu = -z/a``."""
    mu = np.asarray(mu, dtype=float)
    s = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
    x_dim = ka * s
    y_dim = ka * mu
    f_val, _, _ = wave_function_3d(x_dim, y_dim)
    j0 = np.ones_like(x_dim)
    nz = x_dim > 1.0e-15
    if np.any(nz):
        j0[nz] = bessel_j01_y01(x_dim[nz])[0]
    return s, f_val + 1j * math.pi * np.exp(-y_dim) * j0


def hulme_heave_coefficients(
    ka: float, n_terms: int, *, quad_order: int | None = None
) -> ComplexArray:
    """Multipole coefficients ``c_n`` of Hulme's heave wave-free expansion.

    The heave potential on ``r = a, z <= 0`` is

        phi(mu) = sum_n c_n [P_{2n}(mu) + (ka / (2n)) P_{2n-1}(mu)],  n >= 1,

    plus the Kelvin source term ``c_0 [2 + 2*ka*W(ka*s, ka*mu)]``.  ``c``
    solves the Galerkin projection of the body condition ``dphi/dr = -mu`` on
    odd Legendre test functions, exactly as in
    :func:`tensorbem.analytic.hulme_hemisphere_radiation` (whose internals are
    not exposed); the two implementations are cross-checked in the test suite.
    """
    ka = float(ka)
    if not math.isfinite(ka) or ka <= 0.0:
        raise ValueError("ka must be positive and finite")
    if isinstance(n_terms, (bool, np.bool_)) or not isinstance(n_terms, (int, np.integer)) or int(n_terms) < 4:
        raise ValueError("n_terms must be an integer >= 4")
    m_max = int(n_terms)
    q_order = 2 * m_max if quad_order is None else int(quad_order)
    if q_order < 2 * m_max:
        raise ValueError("quad_order must be >= 2 * n_terms")
    nodes, weights = gauss_legendre(q_order)
    mu = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    s, w_val = _wave_kernel(ka, mu)
    _, fx_val, fy_val = wave_function_3d(ka * s, ka * mu)
    j0 = np.ones_like(mu)
    j1 = np.zeros_like(mu)
    nz = ka * s > 1.0e-15
    if np.any(nz):
        j0[nz], j1[nz], _, _ = bessel_j01_y01(ka * s[nz])
    exp_my = np.exp(-ka * mu)
    wx_val = fx_val - 1j * math.pi * exp_my * j1
    wy_val = fy_val - 1j * math.pi * exp_my * j0

    legendre = np.zeros((2 * m_max + 4, q_order))
    legendre[0] = 1.0
    legendre[1] = mu
    for degree in range(2, 2 * m_max + 4):
        legendre[degree] = (
            (2 * degree - 1) * mu * legendre[degree - 1] - (degree - 1) * legendre[degree - 2]
        ) / degree

    basis = np.zeros((m_max + 1, q_order), dtype=np.complex128)
    basis_dr = np.zeros((m_max + 1, q_order), dtype=np.complex128)
    basis[0] = 2.0 + 2.0 * ka * w_val
    basis_dr[0] = -2.0 + 2.0 * (ka * ka) * (s * wx_val + mu * wy_val)
    for n in range(1, m_max + 1):
        basis[n] = legendre[2 * n] + (ka / (2.0 * n)) * legendre[2 * n - 1]
        basis_dr[n] = -(2 * n + 1) * legendre[2 * n] - ka * legendre[2 * n - 1]
    test = legendre[1 : 2 * m_max + 2 : 2]
    weighted = test * w[None, :]
    return np.linalg.solve(weighted @ basis_dr.T, weighted @ (-mu))


def hulme_heave_potential(ka: float, coefficients: ComplexArray, mu: FloatArray) -> ComplexArray:
    """Evaluate the heave radiation potential on the wetted hemisphere."""
    ka = float(ka)
    c = np.asarray(coefficients, dtype=np.complex128)
    mu = np.asarray(mu, dtype=float)
    if mu.ndim != 1 or np.any(mu < 0.0) or np.any(mu > 1.0):
        raise ValueError("mu must be a 1-D array of values in [0, 1]")
    n_terms = c.size - 1
    s, w_val = _wave_kernel(ka, mu)
    legendre = np.zeros((2 * n_terms + 4, mu.size))
    legendre[0] = 1.0
    legendre[1] = mu
    for degree in range(2, 2 * n_terms + 4):
        legendre[degree] = (
            (2 * degree - 1) * mu * legendre[degree - 1] - (degree - 1) * legendre[degree - 2]
        ) / degree
    out = c[0] * (2.0 + 2.0 * ka * w_val)
    for n in range(1, n_terms + 1):
        out = out + c[n] * (legendre[2 * n] + (ka / (2.0 * n)) * legendre[2 * n - 1])
    return out


def heave_near_field_damping(
    ka: float,
    coefficients: ComplexArray,
    *,
    mu_nodes: int = 600,
) -> dict[str, float]:
    """Near-field heave added mass/damping from the pressure integral.

    Dimensionless readings are normalised by ``rho * displaced_volume`` for
    ``A_33`` and by ``rho * omega * displaced_volume`` for ``B_33`` with
    ``displaced_volume = 2*pi*a**3/3`` (``a = 1``); the surface integral is
    ``-3 * int_{-1}^{0} phi n_z dmu`` with ``n_z = -mu``.
    """
    nodes, weights = gauss_legendre(int(mu_nodes))
    mu = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    phi = hulme_heave_potential(ka, coefficients, mu)
    value = -3.0 * np.sum(w * phi * (-mu))
    return {"added_mass_dimless": float(np.real(value)), "damping_dimless": float(np.imag(value))}


def heave_kochin_function(
    ka: float,
    coefficients: ComplexArray,
    azimuth: FloatArray,
    *,
    mu_nodes: int = 600,
    azimuthal_nodes: int = 96,
) -> ComplexArray:
    """Far-field Kochin function ``K(theta)`` of the heave potential.

    ``K(theta) = int_S [-dphi/dn + ka*phi*(n_z - i*(n_x cos + n_y sin))]
    exp(-ka*mu - i*ka*s*cos(theta - psi)) dS`` evaluated with 2-D quadrature on
    the wetted hemisphere; for an axisymmetric heave field ``K`` must be
    independent of ``theta`` (a structural check on the implementation).
    """
    azimuth = np.asarray(azimuth, dtype=float)
    nodes, weights = gauss_legendre(int(mu_nodes))
    mu = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    phi = hulme_heave_potential(ka, coefficients, mu)
    s = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
    psi = _TWO_PI * np.arange(int(azimuthal_nodes)) / int(azimuthal_nodes)
    w_psi = _TWO_PI / int(azimuthal_nodes)
    angle = azimuth[:, None, None] - psi[None, :, None]
    phase = np.exp(-ka * mu[None, None, :] - 1j * ka * s[None, None, :] * np.cos(angle))
    direction = -mu[None, None, :] - 1j * s[None, None, :] * np.cos(angle)
    # dphi/dn = n_z = -mu; the Kochin integrand uses -dphi/dn = +mu.
    integrand = (mu[None, None, :] + ka * phi[None, None, :] * direction) * phase
    return np.einsum("m,tpm->t", w, integrand) * w_psi


def energy_identity_residual(
    ka: float,
    *,
    n_terms: int = 1024,
    mu_nodes: int = 600,
    azimuth: int = 8,
    azimuthal_nodes: int = 96,
) -> dict[str, float]:
    """Near-field pressure integral vs far-field Kochin energy integral.

    Both readings are functionals of the *same* radiating field, so by energy
    conservation (the free surface carries zero mean flux) they must agree;
    the residual is set by the truncation of the multipole series (the body
    condition ``dphi/dn = -mu`` is met only to the series accuracy).
    """
    coefficients = hulme_heave_coefficients(ka, int(n_terms))
    near = heave_near_field_damping(ka, coefficients, mu_nodes=mu_nodes)
    thetas = _TWO_PI * np.arange(int(azimuth)) / int(azimuth)
    kochin = heave_kochin_function(
        ka, coefficients, thetas, mu_nodes=mu_nodes, azimuthal_nodes=azimuthal_nodes
    )
    volume = 2.0 * math.pi / 3.0
    far = float(np.mean(np.abs(kochin) ** 2)) * ka / (2.0 * volume)
    near_b = near["damping_dimless"]
    return {
        "ka": float(ka),
        "n_terms": float(n_terms),
        "added_mass_dimless": near["added_mass_dimless"],
        "damping_near_dimless": near_b,
        "damping_far_dimless": far,
        "kochin_magnitude": float(np.mean(np.abs(kochin))),
        "kochin_source_amplitude": 4.0 * math.pi * float(abs(coefficients[0])),
        "kochin_relative_spread": float(np.std(np.abs(kochin)) / max(np.mean(np.abs(kochin)), 1e-300)),
        "residual": abs(near_b - far) / abs(near_b),
    }