"""Steady three-dimensional lifting surfaces: vortex-lattice solver + analytic references.

Motivation
----------
:mod:`tensorbem.lifting` ships the *two-dimensional* steady lifting solve (Joukowski
aerofoil, Kutta closure at a cusp trailing edge).  README roadmap item 6 lists
"three-dimensional lifting flows" as the remaining lifting gap: every 3-D solver in
this package is non-lifting, because the exterior Dirichlet/Neumann problems it
solves have a unique circulation-free solution, and the frequency-domain free-surface
solvers are radiation/diffraction problems of the same kind.

This module closes that gap with the classical *vortex-lattice method* (VLM): the
lifting surface is represented by horseshoe vortex filaments and the circulation of
each filament is fixed by the zero-normal-flow condition at control points downstream
of the bound segments -- the discrete counterpart of the Kutta condition that fixes
the 2-D circulation.  It is an indirect boundary-element method: the unknowns are
vortex strengths, not surface potentials.

What is *verified* here (see ``docs/lifting3d.md`` and
``tests/test_lifting3d.py``) is the physics, against closed-form lifting-line
theory, not merely internal consistency:

* elliptic planform:  ``C_L = 2*pi*AR/(AR+2) * alpha`` (Prandtl's lifting line) and
  ``C_Di = C_L**2/(pi*AR)`` (Munk's minimum-induced-drag theorem, exact for
  elliptic loading);
* ``AR -> infinity``:  ``C_L/alpha -> 2*pi`` (thin-aerofoil limit, the same value the
  2-D solver of :mod:`tensorbem.lifting` returns for a thin Joukowski section);
* cambered sections:  ``C_L/(2*pi*AR/(AR+2)) - alpha -> 2*z_m/c``, the thin-aerofoil
  camber increment of a parabolic camber line.

Conventions
-----------
``x`` downstream, ``y`` to starboard, ``z`` up; freestream ``(U, 0, 0)``.  A bound
vortex segment along ``+y`` with circulation ``Gamma > 0`` produces lift along ``+z``
(Kutta-Joukowski ``F = rho*Gamma*(V x dl)``).  Each lattice panel carries one
horseshoe: bound segment on the quarter-chord line spanning the panel's own
spanwise extent, two semi-infinite trailing legs downstream, control point at the
three-quarter chord.  The wing is rotated by the angle of attack ``alpha`` about
``y`` so that the chord runs along ``(cos(alpha), 0, -sin(alpha))`` -- leading edge
up, positive lift.

Limitations
-----------
Steady, incompressible, inviscid, unbounded flow (no free surface, no forward-speed
free-surface term); zero thickness, linearised tangency on the mean camber surface;
no boundary-layer or viscosity (no profile drag); the trailing wake is fixed to the
chord plane, not force-free.  Those are the standard assumptions of lifting-line /
lifting-surface theory and match the 2-D module's scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "LiftingWing",
    "VortexLattice",
    "build_lattice",
    "solve_circulation",
    "panel_forces",
    "induced_drag",
    "wake_downwash",
    "loads",
    "spanwise_loading",
    "lifting_line_lift_slope",
    "elliptic_induced_drag",
    "thin_airfoil_lift",
    "lifting_line_solve",
    "elliptic_loading_shape",
    "segment_velocity",
    "semi_infinite_velocity",
]

_FOUR_PI = 4.0 * math.pi


# --------------------------------------------------------------------------------------
# Biot-Savart primitives (vectorised over field points)
# --------------------------------------------------------------------------------------
def segment_velocity(a, b, points, gamma: float = 1.0) -> np.ndarray:
    """Velocity induced by a straight finite vortex segment ``a -> b``.

    ``V = gamma/(4*pi) * (r1 x r2)/|r1 x r2|^2 * (r0 . (r1/|r1| - r2/|r2|))`` with
    ``r0 = b - a``, ``r1 = P - a``, ``r2 = P - b``.  For a point on the segment the
    velocity is set to zero (the self-term is excluded by construction in the lattice).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p = np.atleast_2d(np.asarray(points, dtype=float))
    r1 = p - a
    r2 = p - b
    r0 = b - a
    cross = np.cross(r1, r2)
    denom = np.einsum("ij,ij->i", cross, cross)
    n1 = np.linalg.norm(r1, axis=1)
    n2 = np.linalg.norm(r2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = (r1 / n1[:, None] - r2 / n2[:, None]) @ r0
        scale = np.where(denom > 1e-30, k / denom, 0.0)
    return (gamma / _FOUR_PI) * cross * scale[:, None]


def semi_infinite_velocity(a, direction, points, gamma: float = 1.0) -> np.ndarray:
    """Velocity induced by a semi-infinite vortex line starting at ``a`` along ``direction``.

    Closed form of ``V = gamma/(4*pi) * (d x r)/h^2 * (1 + (r.d)/|r|)`` with
    ``r = P - a`` and ``h = |r x d|`` for a unit direction ``d``.
    """
    a = np.asarray(a, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    p = np.atleast_2d(np.asarray(points, dtype=float))
    r = p - a
    cross_dr = np.cross(d, r)
    h2 = np.einsum("ij,ij->i", cross_dr, cross_dr)
    rn = np.linalg.norm(r, axis=1)
    ud = r @ d
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(h2 > 1e-30, (1.0 + ud / rn) / h2, 0.0)
    return (gamma / _FOUR_PI) * cross_dr * scale[:, None]


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LiftingWing:
    """Thin lifting surface: planform + parabolic camber + angle of attack.

    ``chord(y)`` is the local chord, ``camber_ratio = z_m/c`` the maximum camber of
    the parabolic camber line ``z = 4*z_m*(X/c)*(1 - X/c)`` (``X`` measured from the
    leading edge), ``alpha`` the geometric angle of attack in radians.
    """

    span: float = 8.0
    planform: str = "elliptic"  # "elliptic" | "rectangular" | "tapered"
    chord_root: float = 1.0
    taper_ratio: float = 1.0
    camber_ratio: float = 0.0
    alpha: float = math.radians(4.0)

    def __post_init__(self) -> None:
        if self.span <= 0.0 or self.chord_root <= 0.0:
            raise ValueError("span and chord_root must be positive")
        if self.planform not in ("elliptic", "rectangular", "tapered"):
            raise ValueError(f"unknown planform {self.planform!r}")

    # ------------------------------------------------------------------ geometry
    def chord(self, y):
        """Local chord at spanwise station(s) ``y`` (clipped at zero for elliptic tips)."""
        y = np.asarray(y, dtype=float)
        eta = np.abs(2.0 * y / self.span)
        if self.planform == "rectangular":
            c = np.full_like(eta, self.chord_root)
        elif self.planform == "tapered":
            c = self.chord_root * (1.0 - (1.0 - self.taper_ratio) * eta)
        else:
            c = self.chord_root * np.sqrt(np.clip(1.0 - eta**2, 0.0, None))
        return np.maximum(c, 1e-12)

    def camber(self, x_local, chord):
        """Camber height above the chord plane at chordwise position ``x_local``."""
        xc = np.clip(np.asarray(x_local, dtype=float) / chord, 0.0, 1.0)
        return 4.0 * self.camber_ratio * chord * xc * (1.0 - xc)

    def camber_slope(self, x_local, chord):
        """``d z_c / d X`` of the parabolic camber line (dimensionless)."""
        xc = np.asarray(x_local, dtype=float) / chord
        return 4.0 * self.camber_ratio * (1.0 - 2.0 * xc)

    def area(self) -> float:
        """Planform area (closed form per planform)."""
        if self.planform == "rectangular":
            return self.span * self.chord_root
        if self.planform == "tapered":
            return 0.5 * self.span * self.chord_root * (1.0 + self.taper_ratio)
        return math.pi / 4.0 * self.span * self.chord_root

    def aspect_ratio(self) -> float:
        return self.span**2 / self.area()

    def to_global(self, x_local, y, z_local):
        """Rotate chord-local coordinates into the global frame (leading edge up)."""
        ca, sa = math.cos(self.alpha), math.sin(self.alpha)
        x = x_local * ca + z_local * sa
        z = -x_local * sa + z_local * ca
        return np.stack([x, np.asarray(y, dtype=float), z], axis=-1)

    def normal_global(self, x_local, chord):
        """Unit normal of the camber surface (global frame) at chordwise station."""
        slope = self.camber_slope(x_local, chord)
        n_local = np.stack(
            [
                -slope,
                np.zeros_like(slope),
                np.ones_like(slope),
            ],
            axis=-1,
        )
        n_local = n_local / np.linalg.norm(n_local, axis=-1, keepdims=True)
        ca, sa = math.cos(self.alpha), math.sin(self.alpha)
        nx = n_local[..., 0] * ca + n_local[..., 2] * sa
        nz = -n_local[..., 0] * sa + n_local[..., 2] * ca
        return np.stack([nx, n_local[..., 1], nz], axis=-1)


@dataclass
class VortexLattice:
    """Horseshoe-vortex discretisation of a :class:`LiftingWing`."""

    wing: LiftingWing
    span_panels: int
    chord_panels: int
    y_edges: np.ndarray
    control_points: np.ndarray  # (n, 3) global, three-quarter chord
    normals: np.ndarray  # (n, 3) global
    bound_edges: np.ndarray  # (n, 2, 3) global, a -> b
    bound_mid: np.ndarray  # (n, 3) global, quarter-chord midpoint
    dy: np.ndarray  # (n,) spanwise width of each horseshoe
    strip_index: np.ndarray  # (n,) index of the spanwise strip
    chord: np.ndarray  # (n,) local chord of the strip
    y_mid: np.ndarray  # (n,) spanwise station of the strip
    chords: list = field(default_factory=list, repr=False)


def build_lattice(
    wing: LiftingWing, span_panels: int, chord_panels: int = 1
) -> VortexLattice:
    """Build the horseshoe lattice (cosine clustering in span and chord)."""
    if span_panels < 1 or chord_panels < 1:
        raise ValueError("panel counts must be >= 1")
    k = np.arange(span_panels + 1)
    y_edges = -0.5 * wing.span * np.cos(math.pi * k / span_panels)

    cps, nrm, ea, eb, mids, dys, strips, chords, ymids = [], [], [], [], [], [], [], [], []
    index = 0
    for s in range(span_panels):
        y0, y1 = y_edges[s], y_edges[s + 1]
        y_mid = 0.5 * (y0 + y1)
        c = float(wing.chord(y_mid))
        i = np.arange(chord_panels + 1)
        x_edges = 0.5 * c * (1.0 - np.cos(math.pi * i / chord_panels))
        for p in range(chord_panels):
            x1, x2 = float(x_edges[p]), float(x_edges[p + 1])
            x_v = x1 + 0.25 * (x2 - x1)
            x_c = x1 + 0.75 * (x2 - x1)
            # control point and its normal
            zc = float(wing.camber(x_c, c))
            cps.append(wing.to_global(x_c, y_mid, zc))
            nrm.append(wing.normal_global(np.array([x_c]), c)[0])
            # bound segment on the quarter-chord line
            z_v = float(wing.camber(x_v, c))
            ea.append(wing.to_global(x_v, y0, z_v))
            eb.append(wing.to_global(x_v, y1, z_v))
            mids.append(wing.to_global(x_v, y_mid, z_v))
            dys.append(y1 - y0)
            strips.append(s)
            chords.append(c)
            ymids.append(y_mid)
            index += 1

    return VortexLattice(
        wing=wing,
        span_panels=span_panels,
        chord_panels=chord_panels,
        y_edges=y_edges,
        control_points=np.asarray(cps),
        normals=np.asarray(nrm),
        bound_edges=np.asarray(list(zip(ea, eb))),
        bound_mid=np.asarray(mids),
        dy=np.asarray(dys),
        strip_index=np.asarray(strips, dtype=int),
        chord=np.asarray(chords),
        y_mid=np.asarray(ymids),
    )


def _horseshoe_velocity(lattice: VortexLattice, j: int, points: np.ndarray) -> np.ndarray:
    """Velocity induced by horseshoe ``j`` at ``points`` (unit circulation)."""
    a, b = lattice.bound_edges[j]
    v = segment_velocity(a, b, points, 1.0)
    downstream = np.array([1.0, 0.0, 0.0])
    # The horseshoe is a closed loop P1 -> P2 (bound, +y), P2 -> +inf, and +inf -> P1
    # along the other leg, i.e. the leg at ``a`` is directed upstream (-x) with +gamma,
    # which is the same as downstream with -gamma.  Both legs therefore go downstream
    # but carry opposite signs; adding both with +1 would cancel the downwash.
    v += semi_infinite_velocity(b, downstream, points, 1.0)
    v -= semi_infinite_velocity(a, downstream, points, 1.0)
    return v


def influence_matrix(lattice: VortexLattice) -> np.ndarray:
    """Normal-velocity influence matrix ``A[i, j] = V_j(x_i) . n_i``."""
    n = len(lattice.dy)
    A = np.empty((n, n))
    for j in range(n):
        v = _horseshoe_velocity(lattice, j, lattice.control_points)
        A[:, j] = np.einsum("ij,ij->i", v, lattice.normals)
    return A


def solve_circulation(lattice: VortexLattice, speed: float = 1.0) -> np.ndarray:
    """Solve the zero-normal-flow system for the horseshoe circulations.

    Tangency: ``(U + V_ind) . n = 0``  ->  ``A gamma = -U n_x``.
    """
    A = influence_matrix(lattice)
    rhs = -speed * lattice.normals[:, 0]
    gamma, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return gamma


def panel_forces(
    lattice: VortexLattice, gamma: np.ndarray, speed: float = 1.0, rho: float = 1.0
) -> dict:
    """Kutta-Joukowski forces per panel and the integrated lift coefficient.

    ``F_j = rho*Gamma_j*(V_j x dl_j)`` with ``V_j`` the total velocity at the
    quarter-chord midpoint of horseshoe ``j`` (freestream plus the induced velocity of
    *all* horseshoes; the bound-segment self-term is exactly zero there).

    Only the lift is taken from this near-field form.  The induced drag is **not**
    read off the quarter-chord wash: at the bound vortex the horseshoe is inside its
    own lattice, its two legs contribute a self-wash that diverges as ``1/dy`` with
    the strip width, and the finite sum is a difference of two large nearly cancelling
    numbers (measured in ``docs/lifting3d.md``: the quarter-chord drag changes sign and
    grows in magnitude with refinement).  :func:`induced_drag` uses the far-wake
    (Trefftz-plane) form instead, which is the standard and convergent one.
    """
    gamma = np.asarray(gamma, dtype=float)
    n = len(gamma)
    v_ind = np.zeros((n, 3))
    for j in range(n):
        v_ind += _horseshoe_velocity(lattice, j, lattice.bound_mid) * gamma[j]
    v_total = v_ind + np.array([speed, 0.0, 0.0])
    dl = np.stack(
        [np.zeros(n), lattice.bound_edges[:, 1, 1] - lattice.bound_edges[:, 0, 1], np.zeros(n)],
        axis=1,
    )
    forces = rho * gamma[:, None] * np.cross(v_total, dl)
    S = lattice.wing.area()
    q = 0.5 * rho * speed**2
    return {
        "forces": forces,
        "circulation": gamma,
        "lift": float(np.sum(forces[:, 2])),
        "CL": float(np.sum(forces[:, 2]) / (q * S)),
        "condition_number": float(np.linalg.cond(influence_matrix(lattice))),
    }


def wake_downwash(
    lattice: VortexLattice,
    gamma: np.ndarray,
    speed: float = 1.0,
    downstream_spans: float = 50.0,
) -> np.ndarray:
    """Positive-downward wash on the trailing vortex sheet far downstream.

    The lattice wake is the straight (unrolled) sheet of the trailing legs, so the
    wash at ``50*span`` behind the trailing edge is produced by the trailing vorticity
    alone (the bound term decays like ``1/r^2``); this is the Trefftz-plane wash.
    """
    gamma = np.asarray(gamma, dtype=float)
    n = len(gamma)
    x = float(np.max(lattice.control_points[:, 0])) + downstream_spans * lattice.wing.span
    points = np.stack([np.full(n, x), lattice.y_mid, np.zeros(n)], axis=1)
    v = np.zeros((n, 3))
    for j in range(n):
        v += _horseshoe_velocity(lattice, j, points) * gamma[j]
    return -v[:, 2]


def induced_drag(
    lattice: VortexLattice,
    gamma: np.ndarray,
    speed: float = 1.0,
    downstream_spans: float = 50.0,
) -> float:
    """Induced drag coefficient from the Trefftz-plane momentum balance.

    ``D_i = (rho/2) * sum_j Gamma_j * w_j * dy_j`` with ``w_j`` the far-wake downwash
    (:func:`wake_downwash`); the factor ``1/2`` is the standard wing-to-wake relation
    (the wash at the wing is half the fully developed wake wash, e.g. for elliptic
    loading ``Gamma_0/(2b)`` at the wing against ``Gamma_0/b`` downstream).  For exact
    elliptic loading this evaluates to Munk's ``C_L^2/(pi*AR)``, which is the gate used
    in the tests.
    """
    gamma = np.asarray(gamma, dtype=float)
    w = wake_downwash(lattice, gamma, speed=speed, downstream_spans=downstream_spans)
    S = lattice.wing.area()
    return float(0.5 * np.sum(gamma * w * lattice.dy) / (0.5 * speed**2 * S))


def loads(
    lattice: VortexLattice,
    gamma: np.ndarray | None = None,
    speed: float = 1.0,
    alpha: float | None = None,
) -> dict:
    """Lift, induced drag, span efficiency and lift slope of a converged lattice."""
    if gamma is None:
        gamma = solve_circulation(lattice, speed)
    f = panel_forces(lattice, gamma, speed)
    CDi = induced_drag(lattice, gamma, speed)
    CL = f["CL"]
    AR = lattice.wing.aspect_ratio()
    a = math.radians(0.0) if alpha is None else alpha
    a = lattice.wing.alpha if alpha is None else alpha
    return {
        "CL": CL,
        "CDi": CDi,
        "span_efficiency": (CL**2 / (math.pi * AR * CDi)) if CDi > 0 else float("nan"),
        "CL_alpha": CL / a if a else float("nan"),
        "condition_number": f["condition_number"],
        "gamma": gamma,
    }


def spanwise_loading(lattice: VortexLattice, gamma: np.ndarray, speed: float = 1.0) -> dict:
    """Strip-averaged circulation and sectional lift coefficient vs span station."""
    gamma = np.asarray(gamma, dtype=float)
    ys, gams, cls, cs = [], [], [], []
    for s in range(lattice.span_panels):
        sel = lattice.strip_index == s
        g = float(np.mean(gamma[sel]))
        c = float(lattice.chord[sel][0])
        ys.append(float(lattice.y_mid[sel][0]))
        gams.append(g)
        cs.append(c)
        cls.append(2.0 * g / (speed * c))
    return {
        "y": np.asarray(ys),
        "gamma": np.asarray(gams),
        "cl": np.asarray(cls),
        "chord": np.asarray(cs),
    }


# --------------------------------------------------------------------------------------
# Closed-form references (lifting-line / thin-aerofoil theory)
# --------------------------------------------------------------------------------------
def lifting_line_lift_slope(aspect_ratio: float) -> float:
    """Prandtl lifting-line lift slope ``2*pi*AR/(AR+2)`` per radian (elliptic planform)."""
    return 2.0 * math.pi * aspect_ratio / (aspect_ratio + 2.0)


def elliptic_induced_drag(CL: float, aspect_ratio: float) -> float:
    """Munk's minimum induced drag ``CL**2/(pi*AR)`` for elliptic spanwise loading."""
    return CL**2 / (math.pi * aspect_ratio)


def thin_airfoil_lift(alpha: float, camber_ratio: float = 0.0) -> float:
    """Thin-aerofoil lift of a parabolic camber line: ``2*pi*(alpha + 2*z_m/c)``.

    Derived from the Fourier solution ``C_L = 2*pi*A0 + pi*A1`` with
    ``A0 = alpha - (1/pi) Int_0^pi (dz/dx) dtheta`` and
    ``A1 = (2/pi) Int_0^pi (dz/dx) cos(theta) dtheta``; for
    ``z/c = 4*(z_m/c)*xc*(1-xc)`` the slope is ``dz/dxc = 4*(z_m/c)*cos(theta)``, so
    ``A0 = alpha`` and ``A1 = 4*(z_m/c)``.
    """
    return 2.0 * math.pi * (alpha + 2.0 * camber_ratio)


def lifting_line_solve(wing: LiftingWing, n_modes: int = 24) -> dict:
    """Glauert Fourier solution of the lifting-line equation for an arbitrary planform.

    This is a *second, independent discretisation of the lifting-line model*: the
    Fourier-Glauert series ``Gamma(theta) = 2*b*U*sum_n A_n sin(n*theta)`` with
    ``y = -(b/2) cos(theta)`` turns Prandtl's integro-differential equation

        ``2*pi*(alpha + 2*z_m/c - alpha_i(y))*c(y) = 2*Gamma(y)``,
        ``alpha_i = -(1/(4*pi*U)) Int (dGamma/dy')/(y-y') dy'``

    into the algebraic system ``sum_n A_n sin(n*theta) [4b/c(theta) + 2*pi*n/sin(theta)]
    = 2*pi*(alpha + 2*z_m/c)``, collocated at ``theta_i = i*pi/(N+1)``.  The section
    slope is the thin-aerofoil value ``2*pi`` and the camber term is the same
    ``2*z_m/c`` increment as in :func:`thin_airfoil_lift`.

    Returns the coefficients together with ``C_L = pi*AR*A_1`` and the induced drag
    ``C_Di = pi*AR*sum_n n*A_n**2`` (both exact consequences of the series).
    For the elliptic planform this reduces to ``2*pi*AR/(AR+2)*alpha`` and
    ``C_L**2/(pi*AR)`` (checked in ``tests/test_lifting3d.py``); for other planforms it
    supplies a reference the closed forms cannot give, so the vortex lattice can be
    checked on a rectangular or tapered wing as well.
    """
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    theta = np.arange(1, n_modes + 1) * math.pi / (n_modes + 1)
    n = np.arange(1, n_modes + 1)
    y = -0.5 * wing.span * np.cos(theta)
    chord = wing.chord(y)
    b = wing.span
    A = np.empty((n_modes, n_modes))
    for m, th in enumerate(theta):
        A[m, :] = np.sin(n * th) * (
            4.0 * b / chord[m] + 2.0 * math.pi * n / math.sin(th)
        )
    rhs = np.full(n_modes, 2.0 * math.pi * (wing.alpha + 2.0 * wing.camber_ratio))
    coef = np.linalg.solve(A, rhs)
    AR = wing.aspect_ratio()
    gamma = 2.0 * b * np.sin(np.outer(theta, n)) @ coef
    return {
        "coefficients": coef,
        "theta": theta,
        "y": y,
        "chord": chord,
        "gamma": gamma,
        "CL": math.pi * AR * float(coef[0]),
        "CDi": math.pi * AR * float(np.sum(n * coef**2)),
    }


def elliptic_loading_shape(y, span: float) -> np.ndarray:
    """Normalised elliptic spanwise circulation ``sqrt(1 - (2y/b)**2)``."""
    eta = np.abs(2.0 * np.asarray(y, dtype=float) / span)
    return np.sqrt(np.clip(1.0 - eta**2, 0.0, None))