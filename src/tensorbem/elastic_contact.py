"""Elastic normal contact by the boundary-integral (Boussinesq) route.

Why this module exists
----------------------
``TensorBEM`` had no elasticity at all: every existing kernel is either a
Laplace/Helmholtz layer potential (acoustics) or a free-surface Green function
(hydrodynamics).  The classical contact problem, however, *is* a
boundary-integral problem and is the natural independent second numerical route
for the Hertz anchor ``CASE-FEM-HERTZ-CONTACT`` -- contact radius ``a`` and
penetration ``delta`` versus the Hertz closed form (Hertz 1882; Johnson,
*Contact Mechanics* 1985, Sec. 4.2).  The finite-element route lives in
``TensorFEM`` (``tensorfem.hertz_axisymmetric``); this module shares no code,
no mesh and no discretisation with it, so agreement between the two routes is a
genuine cross-method verification rather than a restatement.

Formulation
-----------
For an isotropic elastic half-space ``z >= 0`` (Young's modulus ``E``, Poisson
ratio ``nu``) the Boussinesq fundamental solution gives the normal surface
displacement under a pressure ``p`` supported on ``A``::

    w(x) = 1/(pi E*) * integral_A p(y)/|x - y| dA(y),      E* = E/(1 - nu^2)

with ``E*`` the semi-space modulus of the indented body; for an indenter of
finite stiffness the usual series combination
``1/E* = (1-nu1^2)/E1 + (1-nu2^2)/E2`` applies.  ``1/|x - y|`` is the Green
function of the *half-space*, not an approximation of a full-space kernel: the
traction-free surface is built in, so no free-surface discretisation is needed
and the only unknowns are the pressures on the contact patch.  That reduction
is what makes the contact problem a one-dimensional (axisymmetric) integral
equation instead of a three-dimensional field problem.

The mechanical statement is the complementarity (Signorini) problem

    w(r) - delta + g(r) >= 0,   p(r) >= 0,   p(r) * [w(r) - delta + g(r)] = 0,

solved here as the convex quadratic programme
``min_{p >= 0} 1/2 p^T C p - p^T b`` with ``C`` the influence matrix of the
kernel above and ``b = delta - g`` the rigid-body approach minus the indenter
gap.  Contact radius (where ``p`` reaches zero), peak pressure and the load
corresponding to a prescribed approach are all OUTPUTS; the indenter enters
only through the purely geometric gap ``g(r)`` and no Hertz formula is used
anywhere inside the solver.

Indenter gaps
-------------
``parabolic_gap(r, radius)`` -- ``r^2/(2R)``, the rigid sphere (exact to
``O((r/R)^4)``); the Hertz anchor.
``flat_gap``                 -- rigid flat punch, used as an independent check
because its closed form is *different*: ``P = 2 E* a delta`` with a
square-root pressure singularity at the rim instead of a vanishing pressure.

Units are SI throughout (m, Pa, N).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import integrate, optimize
from scipy.integrate import IntegrationWarning
import warnings
from scipy.special import ellipk

__all__ = [
    "HertzContact",
    "ContactSolution",
    "effective_modulus",
    "hertz_sphere_halfspace",
    "hertz_sphere_sphere",
    "flat_punch_load",
    "parabolic_gap",
    "flat_gap",
    "ring_kernel",
    "displacement_influence",
    "graded_edges",
    "solve_normal_contact",
    "hertz_comparison",
]


# ---------------------------------------------------------------------------
# Closed-form references (benchmark targets; never used by the solver)
# ---------------------------------------------------------------------------
def effective_modulus(young: float, poisson: float, young2: float | None = None,
                      poisson2: float | None = None) -> float:
    """Effective modulus ``E*`` from the series combination of both bodies."""
    inv = (1.0 - poisson ** 2) / young
    if young2 is not None:
        inv += (1.0 - (0.0 if poisson2 is None else poisson2) ** 2) / young2
    return 1.0 / inv


@dataclass(frozen=True)
class HertzContact:
    """Hertz solution for a sphere of (effective) radius ``radius``.

    Hertz (1882); Johnson (1985) Sec. 4.2::

        a     = (3 P R / (4 E*))^(1/3)
        delta = a^2 / R
        p0    = 3 P / (2 pi a^2)
        p(r)  = p0 sqrt(1 - (r/a)^2)
    """

    load: float
    radius: float
    modulus: float

    @property
    def contact_radius(self) -> float:
        return (3.0 * self.load * self.radius / (4.0 * self.modulus)) ** (1.0 / 3.0)

    @property
    def penetration(self) -> float:
        return self.contact_radius ** 2 / self.radius

    @property
    def peak_pressure(self) -> float:
        return 3.0 * self.load / (2.0 * math.pi * self.contact_radius ** 2)

    @property
    def mean_pressure(self) -> float:
        return self.load / (math.pi * self.contact_radius ** 2)

    def pressure(self, r):
        """Hertz pressure profile (zero outside the contact patch)."""
        r = np.asarray(r, dtype=float)
        a = self.contact_radius
        val = self.peak_pressure * np.sqrt(np.clip(1.0 - (r / a) ** 2, 0.0, None))
        return np.where(r <= a, val, 0.0)


def hertz_sphere_halfspace(load: float, radius: float, modulus: float) -> HertzContact:
    """Hertz anchor for a rigid sphere of radius ``radius`` on the half-space."""
    return HertzContact(load=float(load), radius=float(radius), modulus=float(modulus))


def hertz_sphere_sphere(load: float, radius1: float, radius2: float,
                        young1: float, poisson1: float,
                        young2: float, poisson2: float) -> HertzContact:
    """Two elastic spheres: same closed form with ``R`` and ``E*`` combined."""
    radius = 1.0 / (1.0 / radius1 + 1.0 / radius2)
    modulus = effective_modulus(young1, poisson1, young2, poisson2)
    return HertzContact(load=float(load), radius=radius, modulus=modulus)


def flat_punch_load(radius: float, penetration: float, modulus: float) -> float:
    """Rigid flat punch of radius ``radius``: ``P = 2 E* a delta`` (Johnson 3.3)."""
    return 2.0 * modulus * radius * penetration


# ---------------------------------------------------------------------------
# Indenter gaps (pure geometry)
# ---------------------------------------------------------------------------
def parabolic_gap(r, radius: float):
    """Rigid sphere gap ``r^2/(2R)``."""
    return np.asarray(r, dtype=float) ** 2 / (2.0 * radius)


def flat_gap(r):
    """Rigid flat punch gap."""
    return np.zeros_like(np.asarray(r, dtype=float))


# ---------------------------------------------------------------------------
# Axisymmetric Boussinesq kernel and its ring average
# ---------------------------------------------------------------------------
def ring_kernel(r, rho):
    """Angular average of ``1/|x - y|`` over the ring of radius ``rho``.

    ``int_0^{2 pi} dphi / sqrt(r^2 + rho^2 - 2 r rho cos phi)
       = 4 K(k)/(r + rho)``,  ``k^2 = 4 r rho/(r + rho)^2``,

    with ``K`` the complete elliptic integral of the first kind, parameter
    ``m = k^2`` to match ``scipy.special.ellipk``.  ``k -> 1`` at ``r = rho`` is a
    logarithmic singularity: integrable, and handled by the quadrature split in
    :func:`displacement_influence`.
    """
    r = np.asarray(r, dtype=float)
    rho = np.asarray(rho, dtype=float)
    r, rho = np.broadcast_arrays(r, rho)
    denom = np.maximum(r + rho, np.finfo(float).tiny)
    k2 = np.clip(4.0 * r * rho / denom ** 2, 0.0, 1.0 - 1e-16)
    return 4.0 * ellipk(k2) / denom


def _ring_kernel_scalar(r: float, rho: float) -> float:
    if rho <= 0.0:
        return 0.0
    denom = r + rho
    k2 = min(4.0 * r * rho / denom ** 2, 1.0 - 1e-16)
    return 4.0 * float(ellipk(k2)) / denom


def displacement_influence(centers, edges, modulus: float, order: int = 16) -> np.ndarray:
    """Influence matrix: displacement at ``centers[i]`` from unit pressure on ring ``j``.

    ``C[i, j] = 1/(pi E*) * int_{edges[j]}^{edges[j+1]} rho K_ring(centers[i], rho) d rho``

    Regular elements use 16-point Gauss-Legendre.  The self element contains the
    collocation radius and hence the logarithmic singularity; it is integrated
    with adaptive quadrature split exactly at ``r`` (``O(n)`` extra evaluations).
    """
    centers = np.asarray(centers, dtype=float)
    edges = np.asarray(edges, dtype=float)
    n = centers.size
    gx, gw = np.polynomial.legendre.leggauss(order)

    C = np.empty((n, n), dtype=float)
    for j in range(n):
        lo, hi = edges[j], edges[j + 1]
        half = 0.5 * (hi - lo)
        mid = 0.5 * (hi + lo)
        rho = mid + half * gx
        weights = half * gw * rho
        C[:, j] = (ring_kernel(centers[:, None], rho[None, :]) * weights[None, :]).sum(axis=1)

    with warnings.catch_warnings():
        # the split point r sits exactly on the log singularity: QUADPACK reports
        # non-convergence there every time, although the integrand is integrable
        # and the two half-intervals are resolved.  The assembler is checked
        # instead against Love's uniform-disk displacement (test suite).
        warnings.simplefilter("ignore", IntegrationWarning)
        for i in range(n):
            lo, hi = edges[i], edges[i + 1]
            r = centers[i]
            val, _err = integrate.quad(
                lambda x: x * _ring_kernel_scalar(r, x), lo, hi,
                points=[r], limit=300, epsabs=0.0, epsrel=1e-10,
            )
            C[i, i] = val

    return C / (math.pi * modulus)


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
def graded_edges(domain_radius: float, n_elements: int, grading: float = 2.5) -> np.ndarray:
    """Node radii ``r_j = R_d (j/n)^grading``: graded towards the axis.

    ``grading > 1`` clusters elements at ``r = 0`` where the sphere gap is flat
    and the pressure peak sits.  The rim resolution that this leaves at ``r = a``
    is reported with every solution (``ContactSolution.rim_resolution``) instead
    of being trusted silently.
    """
    j = np.arange(n_elements + 1, dtype=float)
    return domain_radius * (j / n_elements) ** grading


# ---------------------------------------------------------------------------
# Contact solution object
# ---------------------------------------------------------------------------
@dataclass
class ContactSolution:
    """Result of the axisymmetric contact solve (all quantities are outputs)."""

    penetration: float
    load: float
    radii: np.ndarray
    pressures: np.ndarray
    areas: np.ndarray
    edges: np.ndarray
    domain_radius: float
    modulus: float
    contact_radius: float
    contact_radius_rim: float
    edge_exponent: float
    peak_pressure: float
    rim_resolution: float
    history: dict = field(default_factory=dict)

    @property
    def n_elements(self) -> int:
        return int(self.radii.size)

    @property
    def active_mask(self) -> np.ndarray:
        return self.pressures > 1e-9 * max(float(self.pressures.max()), 1e-300)

    def profile(self):
        """``(r, p)`` of the elements carrying pressure."""
        mask = self.active_mask
        return self.radii[mask], self.pressures[mask]


def _restricted_solve(sub: np.ndarray, rhs: np.ndarray,
                      rcond: float = 1e-12) -> np.ndarray:
    """Solve a working-set subproblem, degrading gracefully when rank-deficient.

    Truncated SVD (``rcond``) and not a Cholesky solve, deliberately.  The
    Boussinesq operator ``1/|x-y|`` is a smoothing operator of order -1, so its
    discrete inverse amplifies whatever is not identifiable: near the axis the
    surface displacement is almost constant, two neighbouring small elements
    therefore have nearly identical rows, and an *exact* solve of the restricted
    system returns a spiked pressure distribution (measured: peak pressure 25x
    the Hertz value, growing linearly with the element count) that reproduces the
    right displacements but a meaningless local pressure.  Truncating the
    spectrum below ``rcond * sigma_max`` is the standard regularisation for
    influence matrices and recovers the smooth solution; the sensitivity of the
    reported quantities to ``rcond`` is measured in
    ``tests/test_elastic_contact.py`` (it is flat over four decades).

    Note for maintainers: ``scipy.linalg.cho_factor`` segfaults for ``n >= 60``
    in this environment (reproducible with a plain random SPD matrix, outside
    this module); do not reintroduce it here.
    """
    return np.linalg.lstsq(sub, rhs, rcond=rcond)[0]


def _solve_complementarity(C: np.ndarray, b: np.ndarray, tol: float = 1e-11,
                           max_iter: int = 500, max_pass: int = 40,
                           warm: np.ndarray | None = None,
                           regularization: float = 1e-12) -> np.ndarray:
    """Primal active-set (Lawson-Hanson style) solve of the contact QP.

    ``min_{p >= 0} 1/2 p^T C p - p^T b`` with ``C`` the symmetric positive
    (semi)definite influence matrix.  This is the *primal* method with a line
    search along the segment from the current feasible iterate to the
    equality-constrained subproblem solution, so every iterate stays feasible and
    the objective decreases monotonically (a naive "solve, clip, re-solve" loop
    can cycle, which is what makes this code longer than it looks).

    The starting working set is the geometrically admissible patch ``b > 0``;
    subproblems are solved by least squares because a rank-deficient working set
    does occur when the discretisation cannot separate two neighbouring elements
    (``solve`` would then return garbage).
    """
    n = b.size
    if warm is not None and warm.size == n:
        p = np.maximum(np.asarray(warm, dtype=float), 0.0)
        working = p > 0.0
    else:
        p = np.zeros(n)
        working = np.zeros(n, dtype=bool)
    scale = max(1.0, float(np.abs(b).max()))
    order = np.argsort(-b)  # add the most-violating (largest gap shortfall) first
    pending = [int(i) for i in order if b[i] > 0.0]

    for _ in range(max_iter):
        idx = np.flatnonzero(working)
        if idx.size:
            pW = _restricted_solve(C[np.ix_(idx, idx)], b[idx], regularization)
            if np.any(pW < 0.0):
                bad = np.flatnonzero(pW < 0.0)
                direction = pW - p[idx]
                ratios = np.full(bad.size, np.inf)
                for k, loc in enumerate(bad):
                    if direction[loc] < 0.0:
                        ratios[k] = p[idx][loc] / (p[idx][loc] - pW[loc])
                step = float(ratios.min()) if ratios.size else 0.0
                step = min(max(step, 0.0), 1.0)
                p[idx] = p[idx] + step * direction
                blocking = idx[bad[int(np.argmin(ratios))]]
                p[blocking] = 0.0
                working[blocking] = False
                continue
            p[idx] = pW
        # KKT: the gradient must be non-negative on the variables held at zero
        grad = C @ p - b
        inactive = ~working
        grad_inactive = np.where(inactive & (p <= 0.0), grad, np.inf)
        worst = int(np.argmin(grad_inactive))
        if not np.isfinite(grad_inactive[worst]) or grad_inactive[worst] >= -tol * scale:
            return np.maximum(p, 0.0)
        add = 0
        for _pass in range(max_pass):
            worst = int(np.argmin(grad_inactive))
            if not np.isfinite(grad_inactive[worst]) or grad_inactive[worst] >= -tol * scale:
                break
            working[worst] = True
            grad_inactive[worst] = np.inf
            add += 1
        if add == 0:  # pragma: no cover - safety net against cycling
            return np.maximum(p, 0.0)
    return np.maximum(p, 0.0)


def _contact_edge(radii: np.ndarray, pressures: np.ndarray, edges: np.ndarray,
                  n_fit: int = 4):
    """Estimate the rim radius from the computed pressure field alone.

    Two estimators, both functions of ``(r, p)`` only:

    * ``rim`` -- outer boundary of the last pressurised element: shape-free but
      resolution-limited (uncertainty of order one element width);
    * ``fit`` -- three-parameter power-law fit ``p = K (a - r)^m`` over the
      outermost pressurised elements with the exponent ``m`` left FREE.  The
      recovered ``m`` is a reported diagnostic, not an assumption: Hertz theory
      predicts ``m = 1/2``, so obtaining it from the fit is evidence that the rim
      behaviour is resolved rather than imposed.
    """
    pmax = float(pressures.max())
    if not np.isfinite(pmax) or pmax <= 0.0:
        return 0.0, 0.0, float("nan"), 0.0
    active = np.flatnonzero(pressures > 1e-8 * pmax)
    if active.size == 0:
        return 0.0, 0.0, float("nan"), 0.0
    last = int(active[-1])
    rim = float(edges[last + 1])
    h_rim = float(edges[last + 1] - edges[last])

    outer = active[-min(n_fit, active.size):]
    if outer.size < 3:
        return rim, rim, float("nan"), h_rim
    r_o = radii[outer]
    p_o = pressures[outer]
    a_hi = float(edges[min(last + 2, edges.size - 1)])
    guess = float(r_o[-1]) + 0.75 * h_rim

    def residual(theta):
        logk, m, a = theta
        model = math.exp(logk) * np.maximum(a - r_o, 1e-300) ** m
        return np.log(model) - np.log(p_o)

    try:
        sol = optimize.least_squares(
            residual,
            x0=[math.log(p_o[-1] / max(h_rim, 1e-30) ** 0.5), 0.5, guess],
            bounds=([-np.inf, 0.05, float(r_o[-1]) * (1.0 + 1e-9)],
                    [np.inf, 3.0, max(a_hi, guess)]),
            xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000,
        )
        fit = float(sol.x[2])
        exponent = float(sol.x[1])
    except Exception:  # pragma: no cover - a fit must never abort a solve
        fit, exponent = rim, float("nan")
    return rim, fit, exponent, h_rim


def solve_normal_contact(gap: Callable, modulus: float, *, load: float | None = None,
                         penetration: float | None = None,
                         domain_radius: float | None = None,
                         n_elements: int = 400, grading: float = 2.5,
                         rim_points: int = 4,
                         regularization: float = 1e-12,
                         adequacy_tol: float = 1e-6,
                         max_rebuilds: int = 3) -> ContactSolution:
    """Solve the axisymmetric normal contact problem on an elastic half-space.

    Exactly one of ``load`` (force control, the anchor's question: "``P`` is
    given") or ``penetration`` (displacement control) must be supplied.

    ``regularization`` is the truncated-SVD threshold of the influence matrix
    (see :func:`_restricted_solve`); it is an explicit input so that a
    verification run can demonstrate the reported quantities do not depend on it.

    ``domain_radius`` defaults to ``20 sqrt(load/modulus)`` -- a purely
    dimensional length from the data, not the Hertz radius -- and is then grown
    automatically until the pressure decays at the outer boundary (a truncated
    contact patch would silently inflate the compliance).

    Returns a :class:`ContactSolution` whose ``contact_radius`` is the power-law
    rim fit and whose ``contact_radius_rim`` is the shape-free outer-edge value.
    """
    if (load is None) == (penetration is None):
        raise ValueError("give exactly one of load / penetration")
    if modulus <= 0.0:
        raise ValueError("modulus must be positive")

    if domain_radius is None:
        if load is not None:
            domain_radius = 20.0 * math.sqrt(load / modulus)
        else:
            domain_radius = 20.0 * math.sqrt(penetration * 1.0) if penetration else 1.0

    result: ContactSolution | None = None
    for rebuild in range(max_rebuilds + 1):
        edges = graded_edges(domain_radius, n_elements, grading)
        centers = 0.5 * (edges[1:] + edges[:-1])
        areas = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        gap_values = np.asarray(gap(centers), dtype=float)
        if gap_values.shape != centers.shape:
            gap_values = np.broadcast_to(gap_values, centers.shape).copy()
        C = displacement_influence(centers, edges, modulus)

        cache: dict = {}
        last: list = [None]

        def pressures_at(delta: float) -> np.ndarray:
            key = float(delta)
            if key in cache:
                return cache[key]
            warm = cache.get(last[0]) if last[0] is not None else None
            p = _solve_complementarity(C, delta - gap_values, warm=warm,
                                       regularization=regularization)
            cache[key] = p
            last[0] = key
            return p

        def total_load(delta: float) -> float:
            return float((areas * pressures_at(delta)).sum())

        history: dict = {}
        if penetration is not None:
            delta = float(penetration)
            pressures = pressures_at(delta)
            force = float((areas * pressures).sum())
        else:
            lo = 0.0
            hi = float(domain_radius) * 1e-6
            for _ in range(200):
                if total_load(hi) >= load:
                    break
                hi *= 4.0
            else:  # pragma: no cover - would need a pathological gap
                raise RuntimeError("could not bracket the target load")
            bracket = (lo, hi)
            root = optimize.brentq(lambda d: total_load(d) - load, lo, hi,
                                   xtol=1e-16 * hi, rtol=1e-13, maxiter=300)
            delta = float(root)
            pressures = pressures_at(delta)
            force = float((areas * pressures).sum())
            history["bracket"] = bracket

        rim, fit, exponent, h_rim = _contact_edge(centers, pressures, edges, rim_points)
        pmax = float(pressures.max())
        outer = centers > 0.75 * domain_radius
        outer_max = float(pressures[outer].max()) if outer.any() else 0.0
        result = ContactSolution(
            penetration=delta,
            load=force,
            radii=centers,
            pressures=pressures,
            areas=areas,
            edges=edges,
            domain_radius=float(domain_radius),
            modulus=float(modulus),
            contact_radius=fit,
            contact_radius_rim=rim,
            edge_exponent=exponent,
            peak_pressure=pmax,
            rim_resolution=h_rim,
            history=history,
        )
        if outer_max <= adequacy_tol * max(pmax, 1e-300) or penetration is not None:
            break
        # the patch reaches the truncation boundary: the half-space is being
        # cut, so the compliance would be overestimated -> enlarge and redo
        domain_radius = domain_radius * 1.8
        result.history["rebuilds"] = rebuild + 1
    assert result is not None
    return result


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------
def hertz_comparison(solution: ContactSolution, hertz: HertzContact) -> dict:
    """Relative errors of the BEM solution against the Hertz closed form.

    The pressure profile is compared on a common set of radii (the BEM element
    centroids inside the *computed* patch) so that the numbers quoted in a report
    come straight from this function.
    """
    r, p = solution.profile()
    inside = r < min(solution.contact_radius, hertz.contact_radius)
    ref = hertz.pressure(r[inside])
    err = p[inside] - ref
    rms = float(np.sqrt(np.mean(err ** 2)) / hertz.peak_pressure) if err.size else float("nan")
    peak_ref = hertz.peak_pressure
    out = {
        "load": solution.load,
        "load_ref": hertz.load,
        "penetration": solution.penetration,
        "penetration_ref": hertz.penetration,
        "contact_radius": solution.contact_radius,
        "contact_radius_rim": solution.contact_radius_rim,
        "contact_radius_ref": hertz.contact_radius,
        "peak_pressure": solution.peak_pressure,
        "peak_pressure_ref": peak_ref,
        "edge_exponent": solution.edge_exponent,
        "rim_resolution": solution.rim_resolution,
    }
    out["penetration_rel_error"] = (solution.penetration / hertz.penetration) - 1.0
    out["contact_radius_rel_error"] = (solution.contact_radius / hertz.contact_radius) - 1.0
    out["contact_radius_rim_rel_error"] = (solution.contact_radius_rim / hertz.contact_radius) - 1.0
    out["peak_pressure_rel_error"] = (solution.peak_pressure / peak_ref) - 1.0
    out["pressure_profile_rms_rel_error"] = rms
    return out