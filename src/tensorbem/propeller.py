"""Propeller open-water performance by a vortex-lattice (lifting-surface) BEM.

Motivation
----------
The package had no first-principles propeller kernel: every open-water number in
the ecosystem came from regression tables or hand fits, so no discretisation
parameter existed and a grid-convergence study was *undefined* (see
``docs/m3538_propeller_openwater_uncertainty.md`` §5, which also records that an
earlier vortex-lattice attempt produced ``K_Q < 0`` and was deleted because its
saddle-vortex return legs closed downstream and its Kutta-Joukowski sign was
unverified).

This module is the replacement.  Every sign that used to be wrong is pinned by
an analytic case before any propeller number is produced:

* :func:`segment_induced_velocity` -- Biot-Savart on a straight vortex segment
  with the right-hand-rule convention; ``tests`` check it against the
  infinite-line value ``Gamma / (2 pi h)`` and its direction.
* :func:`solve_lifting_surface` -- horseshoe vortex lattice.  The trailing legs
  belong to one continuous vortex *path* ``far field -> A -> B -> far field``,
  so the upstream/downstream closure of the return legs is a property of the
  data structure instead of a sign the caller can get wrong.
* The Kutta-Joukowski force is ``F = rho * Gamma * (W x dl)`` with ``dl`` along
  the bound segment and ``Gamma`` the right-hand-rule circulation of the same
  convention as the induction kernel; ``tests/test_propeller_open_water.py``
  checks it on an elliptic wing against ``C_L = 2 pi AR/(AR+2) alpha`` and on a
  high-aspect-ratio wing against the 2-D limit ``2 pi alpha``.

Open-water model
----------------
Non-dimensionalised with ``D = 1`` and ``n = 1`` rev/s so that the advance ratio
``J = V_a/(n D)`` equals the axial inflow velocity and ``omega = 2 pi``.  In the
blade-fixed frame the inflow at a point ``P`` is

    W_inf(P) = J x_hat - omega r (x_hat x e_r(P))

i.e. the axial inflow plus the tangential velocity the blade sees because it
rotates.  Blade sections are flat (no camber) and sit on the helical chord line
of pitch ratio ``P/D``; the chord distribution is a smooth hump scaled so that
the expanded area ratio ``Ae/Ao`` is matched exactly (see
:class:`BladeGeometry`).

Three wake models are available through ``wake_model``:

``"geometric"``
    helix follows the blade geometric pitch ``P/D`` (fixed-pitch helical wake,
    no slipstream contraction);
``"kinematic"``
    helix follows the free-stream advance ``J`` (a free streamline of the
    undisturbed inflow: it advances ``J`` per revolution);
``"induced"``
    helix follows the *local* total velocity of a converged ``"kinematic"``
    solution (one force-free alignment step).

Viscosity
---------
An inviscid lattice cannot predict ``K_Q`` to better than ~20%: section profile
drag is the dominant torque contribution and inviscid theory has none (the
lifting-line comparison documented in ``docs/m3538...md`` §7 underpredicts
``10 K_Q`` by 13-21%).  The lattice is therefore coupled to the ITTC-1978
section drag law

    C_D = 2 (1 + 2 t/c) [0.044 / Re^(1/6) - 5 / Re^(2/3)]

added as a distributed force opposing the local relative velocity on every
panel; it raises torque and reduces thrust (through ``dD sin(beta)``).  The law
is the standard ITTC-78 performance-prediction formulation (ITTC Recommended
Procedures 7.5-02-03-02.1 "Open Water Test"; Carlton, *Marine Propellers and
Propulsion*, 4th ed.).  The standard text is not available offline, so the study
artefact cites it instead of quoting it and marks the provenance depth.

Scope
-----
Steady, incompressible, unbounded flow; no free surface, no cavitation, no hub
or blade-thickness blockage.  Segment singularities are desingularised with a
van Garrel style core radius.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

TWO_PI = 2.0 * math.pi

#: Kinematic viscosity of water at 15 C in m^2/s.  A physical constant, not a
#: benchmark value: it only sets the section Reynolds number of the model run.
WATER_NU = 1.14e-6

#: Reference density for non-dimensionalisation (cancels in K_T / K_Q).
RHO = 1.0


# ---------------------------------------------------------------------------
# Vortex-segment induction
# ---------------------------------------------------------------------------


def segment_induced_velocity(
    points: FloatArray,
    start: FloatArray,
    end: FloatArray,
    core_ratio: float | FloatArray = 0.05,
) -> FloatArray:
    """Velocity induced at ``points`` by straight vortex segments.

    ``start`` / ``end`` are ``(S, 3)`` end points: the vortex line is traversed
    from ``start`` to ``end`` and the circulation is positive by the right-hand
    rule about that direction.  ``points`` is ``(Q, 3)``.

    Returns ``(Q, S, 3)`` unit-circulation velocities.  The finite core
    regularises ``1/|r|^2`` with the van Garrel form ``K + (core_ratio
    |end - start|)^2`` so that a collocation point lying on a segment returns a
    finite value instead of a division by zero.
    """

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    start = np.asarray(start, dtype=float).reshape(-1, 3)
    end = np.asarray(end, dtype=float).reshape(-1, 3)
    if start.shape != end.shape:
        raise ValueError("start and end must have the same shape")

    r1 = points[:, None, :] - start[None, :, :]
    r2 = points[:, None, :] - end[None, :, :]
    n1 = np.linalg.norm(r1, axis=-1)
    n2 = np.linalg.norm(r2, axis=-1)

    cross = np.cross(r1, r2)
    dot = np.einsum("qsk,qsk->qs", r1, r2)
    length = np.linalg.norm(end - start, axis=-1)
    core_radius = np.asarray(core_ratio, dtype=float) * length
    if core_radius.ndim and core_radius.shape != length.shape:
        raise ValueError("core_ratio must be a scalar or match the segment count")
    core = core_radius**2

    denominator = n1 * n2 * (n1 * n2 + dot) + core[None, :]
    denominator = np.where(denominator == 0.0, np.finfo(float).tiny, denominator)

    factor = (n1 + n2) / denominator
    return cross * factor[:, :, None] / (4.0 * math.pi)


# ---------------------------------------------------------------------------
# Vortex-lattice solve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatticeSolution:
    """Circulations and panel geometry of a solved vortex lattice."""

    circulation: FloatArray  # (P,)
    bound_a: FloatArray  # (P, 3)
    bound_b: FloatArray  # (P, 3)
    midpoints: FloatArray  # (P, 3) bound-vortex midpoint
    control_points: FloatArray  # (P, 3)
    normals: FloatArray  # (P, 3)
    chords: FloatArray  # (P,)
    spans: FloatArray  # (P,)
    induced_velocity: FloatArray  # (P, 3) at the bound-vortex midpoints
    influence: FloatArray  # (P, P) normal-wash matrix

    @property
    def bound_vectors(self) -> FloatArray:
        return self.bound_b - self.bound_a


def _rotation(axis: FloatArray, angle: float):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    def apply(array: FloatArray) -> FloatArray:
        c, s = math.cos(angle), math.sin(angle)
        flat = np.asarray(array, dtype=float).reshape(-1, 3)
        out = (
            flat * c
            + np.cross(axis, flat) * s
            + np.outer(flat @ axis, axis) * (1.0 - c)
        )
        return out.reshape(np.shape(array))

    return apply


def solve_lifting_surface(
    bound_a: FloatArray,
    bound_b: FloatArray,
    control_points: FloatArray,
    normals: FloatArray,
    wake_legs: FloatArray,
    freestream,
    *,
    chords: FloatArray | None = None,
    spans: FloatArray | None = None,
    copies: int = 1,
    copy_axis: FloatArray | None = None,
    core_ratio: float = 0.05,
) -> LatticeSolution:
    """Solve the horseshoe vortex lattice for the unknown circulations.

    Parameters
    ----------
    bound_a, bound_b
        ``(P, 3)`` endpoints of each panel's bound vortex, traversed ``A -> B``
        with right-hand-rule circulation ``Gamma``.
    control_points, normals
        ``(P, 3)`` collocation points (three-quarter chord) and unit normals.
    wake_legs
        ``(P, 2, M, 3)`` trailing-leg polylines leaving ``bound_a`` (index 0)
        and ``bound_b`` (index 1), ordered downstream, ``M >= 2``.
    freestream
        Callable ``points -> (P, 3)`` inflow, or a constant ``(3,)``.
    copies, copy_axis
        ``copies`` rotationally repeated copies of the whole vortex system
        (``copies=Z`` for a ``Z``-bladed propeller evaluated in the blade-fixed
        frame, where rotational symmetry gives all blades the same circulation
        distribution).
    """

    bound_a = np.asarray(bound_a, dtype=float).reshape(-1, 3)
    bound_b = np.asarray(bound_b, dtype=float).reshape(-1, 3)
    control_points = np.asarray(control_points, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    wake_legs = np.asarray(wake_legs, dtype=float)
    n_panels = bound_a.shape[0]
    if wake_legs.shape[:2] != (n_panels, 2) or wake_legs.ndim != 4:
        raise ValueError("wake_legs must have shape (P, 2, M, 3)")
    # Panel extents.  ``spans`` sets the near-field core radius, so it must be
    # the strip width (the bound-vortex length) when the caller does not say.
    if chords is None:
        chords = np.linalg.norm(bound_b - bound_a, axis=-1)
    chords = np.asarray(chords, dtype=float).reshape(-1)
    if spans is None:
        spans = np.linalg.norm(bound_b - bound_a, axis=-1)
    spans = np.asarray(spans, dtype=float).reshape(-1)

    if callable(freestream):
        w_inf = np.asarray(freestream(control_points), dtype=float).reshape(-1, 3)
    else:
        w_inf = np.broadcast_to(
            np.asarray(freestream, dtype=float).reshape(1, 3), control_points.shape
        )

    # --- flatten the horseshoe system into one segment list ----------------
    seg_start = []
    seg_end = []
    seg_panel = []

    leg_a = wake_legs[:, 0, :, :]
    leg_b = wake_legs[:, 1, :, :]
    m_points = leg_a.shape[1]

    # leg at A traversed far field -> A (reverse of the downstream ordering)
    for k in range(m_points - 1, 0, -1):
        seg_start.append(leg_a[:, k, :])
        seg_end.append(leg_a[:, k - 1, :])
        seg_panel.append(np.arange(n_panels))
    # bound vortex A -> B
    seg_start.append(bound_a)
    seg_end.append(bound_b)
    seg_panel.append(np.arange(n_panels))
    # leg at B traversed B -> far field
    for k in range(m_points - 1):
        seg_start.append(leg_b[:, k, :])
        seg_end.append(leg_b[:, k + 1, :])
        seg_panel.append(np.arange(n_panels))

    seg_start = np.concatenate(seg_start, axis=0)
    seg_end = np.concatenate(seg_end, axis=0)
    seg_panel = np.concatenate(seg_panel, axis=0)
    n_segments = seg_start.shape[0]

    incidence = np.zeros((n_segments, n_panels))
    incidence[np.arange(n_segments), seg_panel] = 1.0

    n_copies = 1 if copy_axis is None else int(copies)
    angles = [TWO_PI * m / n_copies for m in range(n_copies)] if n_copies > 1 else [0.0]

    # Near-field regularisation: a trailing leg sits half a strip width from
    # the collocation point, so a core that only follows the segment length
    # makes the normal wash grow like 1/span and the lattice picks up the
    # odd-even (sawtooth) mode as the radial mesh is refined.  Tying the core
    # to the strip width keeps the near field bounded at every refinement.
    seg_core = np.maximum(core_ratio * np.linalg.norm(seg_end - seg_start, axis=-1),
                          0.5 * np.asarray(spans, dtype=float)[seg_panel])

    influence = np.zeros((n_panels, n_panels))
    for angle in angles:
        if angle == 0.0:
            s_a, s_b = seg_start, seg_end
        else:
            # rotate the vortices only: the control point stays put, so the
            # influence of the m-th copy is n . u(R_m seg -> cp), not the
            # trivially equal rotated-point version of the m = 0 system.
            rotate = _rotation(copy_axis, angle)
            s_a, s_b = rotate(seg_start), rotate(seg_end)
        velocities = segment_induced_velocity(
            control_points, s_a, s_b, core_ratio=seg_core
        )
        wash = np.einsum("qsk,qk->qs", velocities, normals)
        influence += wash @ incidence

    rhs = -np.einsum("pk,pk->p", w_inf, normals)
    circulation = np.linalg.solve(influence, rhs)

    # --- induced velocity at the bound-vortex midpoints --------------------
    midpoints = 0.5 * (bound_a + bound_b)
    strength = incidence @ circulation  # per-segment circulation
    velocity_at_mid = np.zeros_like(midpoints)
    for angle in angles:
        if angle == 0.0:
            s_a, s_b = seg_start, seg_end
        else:
            rotate = _rotation(copy_axis, angle)
            s_a, s_b = rotate(seg_start), rotate(seg_end)
        velocities = segment_induced_velocity(
            midpoints, s_a, s_b, core_ratio=seg_core
        )
        velocity_at_mid += np.einsum("qsk,s->qk", velocities, strength)

    return LatticeSolution(
        circulation=circulation,
        bound_a=bound_a,
        bound_b=bound_b,
        midpoints=midpoints,
        control_points=control_points,
        normals=normals,
        chords=chords,
        spans=spans,
        induced_velocity=velocity_at_mid,
        influence=influence,
    )


def kutta_joukowski_force(solution: LatticeSolution, total_velocity: FloatArray) -> FloatArray:
    """``F = rho Gamma (W x dl)`` on every bound vortex (``(P, 3)``)."""

    total_velocity = np.asarray(total_velocity, dtype=float).reshape(-1, 3)
    bound = solution.bound_vectors
    lift = np.cross(total_velocity, bound)
    return RHO * solution.circulation[:, None] * lift


# ---------------------------------------------------------------------------
# Blade geometry
# ---------------------------------------------------------------------------


def _cosine_nodes(start: float, stop: float, count: int) -> FloatArray:
    """``count + 1`` nodes on ``[start, stop]`` clustered at both ends."""

    if count < 1:
        raise ValueError("count must be >= 1")
    beta = np.linspace(0.0, math.pi, count + 1)
    nodes = 0.5 * (1.0 - np.cos(beta))
    return start + (stop - start) * nodes


@dataclass(frozen=True)
class BladeGeometry:
    """Parametric reconstruction of a propeller blade.

    Only the published main particulars are used: ``Z`` blades, diameter,
    ``Ae/Ao``, hub ratio, mean ``P/D`` and NACA66 sections (KP505, Tokyo 2015,
    as tabulated in Iftikhar & Takinaci 2023, Table 1).  Two quantities are not
    published there and are therefore declared modelling assumptions, so the
    study can report their effect:

    * chord distribution ``c/R = A (1 + w u) (1 - u)**0.45`` with ``u = (r - r_h)
      /(R - r_h)``; ``A`` follows from ``Ae/Ao`` exactly (closed-form below), and
      ``w = root_chord_weight`` sets how much chord the root carries relative to
      mid span.
    * radial pitch distribution: constant at the mean ``P/D`` (the real KP505
      pitch rises to a maximum near ``0.6 R``; that shape is not published in
      the source table, so it is not invented here).
    * no skew and no rake (both zero-mean effects in steady open water).
    """

    diameter: float = 1.0
    blades: int = 5
    hub_ratio: float = 0.18
    pitch_ratio: float = 0.95
    expanded_area_ratio: float = 0.80
    root_chord_offset: float = 0.12
    root_exponent: float = 0.80
    hump_exponent: float = 0.50
    thickness_ratio: float = 0.06
    n_radial: int = 12
    n_chord: int = 1
    radial_spacing: str = "cosine"

    # -- dimensionless radii -------------------------------------------------
    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def hub_radius(self) -> float:
        return self.hub_ratio * self.radius

    def chord_shape(self, r: FloatArray) -> FloatArray:
        """``(u + root_offset)^root_exponent (1 - u)^tip_exponent``, ``u`` in [0, 1].

        The root taper is not cosmetic: a full-chord root section wraps more than
        one blade pitch in azimuth (``c cos(beta)/r > 2 pi / Z``) and the
        neighbouring blades then touch, which destroys the lattice.  A root
        taper keeps ``c cos(beta)/r`` below one blade pitch, as on a real blade.
        """

        r = np.asarray(r, dtype=float)
        u = np.clip((r - self.hub_radius) / (self.radius - self.hub_radius), 0.0, 1.0)
        return (u + self.root_chord_offset) ** self.root_exponent * (
            1.0 - u
        ) ** self.hump_exponent

    def area_integral(self) -> float:
        """``int_{r_h}^{R} shape(r) dr`` by Gauss-Legendre quadrature."""

        nodes, weights = np.polynomial.legendre.leggauss(64)
        half = 0.5 * (self.radius - self.hub_radius)
        centre = 0.5 * (self.radius + self.hub_radius)
        r = centre + half * nodes
        return float(half * np.sum(weights * self.chord_shape(r)))

    def chord_scale(self) -> float:
        """Leading constant ``A`` such that ``Ae/Ao`` is matched exactly.

        ``Ae = Z int c dr``, ``Ao = pi D^2 / 4``, ``c = A * shape``.
        """

        disc_area = math.pi * self.diameter**2 / 4.0
        return (
            self.expanded_area_ratio
            * disc_area
            / (self.blades * self.area_integral())
        )

    def chord(self, r: FloatArray) -> FloatArray:
        return self.chord_scale() * self.chord_shape(r)

    def geometric_pitch_angle(self, r: FloatArray) -> FloatArray:
        """Angle of the chord line from the plane of rotation."""

        r = np.maximum(np.asarray(r, dtype=float), 1e-9)
        return np.arctan(self.pitch_ratio * self.diameter / (TWO_PI * r))

    def section_point(self, r: FloatArray, s: FloatArray) -> FloatArray:
        """Blade-surface point ``(x, y, z)`` at radius ``r``, chordwise ``s``.

        ``s`` is the arc length along the chord measured from mid-chord,
        positive towards the trailing edge.  The section sits on the helix of
        pitch ``P = (P/D) D``, so a chord run advances axially by
        ``s sin(beta)`` and tangentially by ``s cos(beta)``.
        """

        r = np.asarray(r, dtype=float)
        s = np.asarray(s, dtype=float)
        r, s = np.broadcast_arrays(r, s)
        beta = self.geometric_pitch_angle(r)
        theta = -s * np.cos(beta) / np.maximum(r, 1e-9)
        x = s * np.sin(beta)
        y = r * np.cos(theta)
        z = r * np.sin(theta)
        return np.stack([x, y, z], axis=-1)

    # -- lattice -------------------------------------------------------------
    def radial_nodes(self) -> FloatArray:
        if self.radial_spacing == "cosine":
            # full cosine clustering degenerates at a fine tip strip (the strip
            # width collapses to ~1e-3 D and the lattice becomes near singular),
            # so the clustering is damped: still finer at both ends, but with a
            # bounded strip-width ratio.
            beta = np.linspace(0.0, math.pi, self.n_radial + 1)
            shift = 0.5 * (1.0 - np.cos(beta))
            blend = 0.35 * shift + 0.65 * np.linspace(0.0, 1.0, self.n_radial + 1)
            return self.hub_radius + (self.radius - self.hub_radius) * blend
        if self.radial_spacing == "uniform":
            return np.linspace(self.hub_radius, self.radius, self.n_radial + 1)
        raise ValueError(f"unknown radial_spacing {self.radial_spacing!r}")

    def panels(self):
        """Return the lattice arrays for one blade (no wake)."""

        nodes = self.radial_nodes()
        n_r = self.n_radial
        n_c = self.n_chord
        chords = np.zeros((n_r, 1))
        mid_radius = 0.5 * (nodes[:-1] + nodes[1:])
        for j in range(n_r):
            chords[j, 0] = self.chord(np.array([mid_radius[j]]))[0]

        # chordwise nodes per strip, in fractions of the local chord
        fractions = np.linspace(-0.5, 0.5, n_c + 1)

        bound_a = []
        bound_b = []
        cps = []
        normals = []
        panel_chords = []
        panel_spans = []

        for j in range(n_r):
            r_in, r_out = nodes[j], nodes[j + 1]
            r_mid = 0.5 * (r_in + r_out)
            c = chords[j, 0]
            span_width = abs(self.section_point(r_out, 0.0)[1] - self.section_point(r_in, 0.0)[1])
            for i in range(n_c):
                s_qc = fractions[i] + 0.25 * (fractions[i + 1] - fractions[i])
                s_cp = fractions[i] + 0.75 * (fractions[i + 1] - fractions[i])
                s_lo, s_hi = fractions[i], fractions[i + 1]
                a_point = self.section_point(r_in, s_qc * c)
                b_point = self.section_point(r_out, s_qc * c)
                cp = self.section_point(r_mid, s_cp * c)
                span_tangent = self.section_point(r_out, s_cp * c) - self.section_point(r_in, s_cp * c)
                eps = 1e-4 * c
                chord_tangent = (
                    self.section_point(r_mid, s_cp * c + eps)
                    - self.section_point(r_mid, s_cp * c - eps)
                )
                normal = np.cross(span_tangent, chord_tangent)
                norm = np.linalg.norm(normal)
                normal = normal / norm if norm > 0 else np.array([1.0, 0.0, 0.0])
                bound_a.append(a_point)
                bound_b.append(b_point)
                cps.append(cp)
                normals.append(normal)
                panel_chords.append(c * (s_hi - s_lo))
                panel_spans.append(span_width)

        return {
            "bound_a": np.asarray(bound_a),
            "bound_b": np.asarray(bound_b),
            "control_points": np.asarray(cps),
            "normals": np.asarray(normals),
            "chords": np.asarray(panel_chords),
            "spans": np.asarray(panel_spans),
            "radial_nodes": nodes,
        }


def helical_wake(
    points: FloatArray,
    tan_beta: FloatArray,
    length: float,
    steps: int,
) -> FloatArray:
    """Helical trailing-leg polylines leaving ``points`` (``(P, 3)``).

    The helix keeps the radius of the attachment point and advances
    ``length`` diameters downstream; ``tan_beta`` is the local helix angle
    measured from the plane of rotation.  Positive advance means the azimuth
    decreases (the fluid moves in ``-theta`` relative to the blade).

    Returns ``(P, M, 3)`` with ``M = steps + 1``.
    """

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    tan_beta = np.asarray(tan_beta, dtype=float).reshape(-1)
    r = np.hypot(points[:, 1], points[:, 2])
    theta0 = np.arctan2(points[:, 2], points[:, 1])
    tau = np.linspace(0.0, float(length), int(steps) + 1)
    dtheta = -tau[None, :] / (np.maximum(r, 1e-9)[:, None] * np.maximum(tan_beta, 1e-9)[:, None])
    theta = theta0[:, None] + dtheta
    x = points[:, 0][:, None] + tau[None, :]
    y = r[:, None] * np.cos(theta)
    z = r[:, None] * np.sin(theta)
    return np.stack([x, y, z], axis=-1)


# ---------------------------------------------------------------------------
# Open-water analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenWaterResult:
    """Non-dimensional open-water quantities at one advance ratio."""

    advance_ratio: float
    thrust: float  # K_T  = T / (rho n^2 D^4)
    torque: float  # K_Q  = Q / (rho n^2 D^5)
    efficiency: float  # eta_0 = J K_T / (2 pi K_Q)
    thrust_inviscid: float
    torque_inviscid: float
    viscous_drag: float  # sum of the section drag forces / (rho n^2 D^4)
    momentum_bound: float  # 2 / (1 + sqrt(1 + C_T))
    wake_model: str
    n_panels: int

    @property
    def ten_kq(self) -> float:
        return 10.0 * self.torque

    def as_dict(self) -> dict:
        return {
            "J": self.advance_ratio,
            "KT": self.thrust,
            "KQ": self.torque,
            "10KQ": self.ten_kq,
            "eta0": self.efficiency,
            "KT_inviscid": self.thrust_inviscid,
            "KQ_inviscid": self.torque_inviscid,
            "viscous_drag_nd": self.viscous_drag,
            "eta0_momentum_bound": self.momentum_bound,
            "wake_model": self.wake_model,
            "n_panels": self.n_panels,
        }


def inflow_velocity(advance_ratio: float, angular_rate: float = TWO_PI):
    """``W_inf(P) = J x_hat - omega r (x_hat x e_r)`` in the blade-fixed frame."""

    def inflow(points: FloatArray) -> FloatArray:
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        r = np.hypot(points[:, 1], points[:, 2])
        # tangential unit vector (x_hat x e_r) for a blade turning in +theta
        e_theta = np.stack(
            [
                np.zeros_like(r),
                -points[:, 2] / np.maximum(r, 1e-12),
                points[:, 1] / np.maximum(r, 1e-12),
            ],
            axis=-1,
        )
        return np.stack(
            [
                np.full_like(r, advance_ratio),
                np.zeros_like(r),
                np.zeros_like(r),
            ],
            axis=-1,
        ) - angular_rate * r[:, None] * e_theta

    return inflow


def _wake_legs(
    geometry: BladeGeometry,
    attachment_a: FloatArray,
    attachment_b: FloatArray,
    wake_model: str,
    advance_ratio: float,
    wake_length: float,
    wake_steps: int,
    induced_velocity: FloatArray | None = None,
) -> FloatArray:
    r_a = np.hypot(attachment_a[:, 1], attachment_a[:, 2])
    r_b = np.hypot(attachment_b[:, 1], attachment_b[:, 2])
    if wake_model == "geometric":
        tan_a = np.tan(geometry.geometric_pitch_angle(r_a))
        tan_b = np.tan(geometry.geometric_pitch_angle(r_b))
    elif wake_model == "kinematic":
        tan_a = advance_ratio / (TWO_PI * np.maximum(r_a, 1e-9))
        tan_b = advance_ratio / (TWO_PI * np.maximum(r_b, 1e-9))
    else:
        raise ValueError(
            f"unknown wake model {wake_model!r}: the induced helix is built in "
            "open_water_point from a converged kinematic solution"
        )

    leg_a = helical_wake(attachment_a, tan_a, wake_length, wake_steps)
    leg_b = helical_wake(attachment_b, tan_b, wake_length, wake_steps)
    return np.stack([leg_a, leg_b], axis=1)


def section_drag_coefficient(reynolds: FloatArray, thickness_ratio: float) -> FloatArray:
    """ITTC-1978 section drag law (see module docstring for provenance)."""

    re = np.maximum(np.asarray(reynolds, dtype=float), 1.0)
    return 2.0 * (1.0 + 2.0 * thickness_ratio) * (
        0.044 / re ** (1.0 / 6.0) - 5.0 / re ** (2.0 / 3.0)
    )


def open_water_point(
    geometry: BladeGeometry,
    advance_ratio: float,
    *,
    wake_model: str = "kinematic",
    wake_length: float = 2.0,
    wake_steps: int = 24,
    rev_per_second: float = 10.0,
    viscosity: float = WATER_NU,
    viscous: bool = True,
    core_ratio: float = 0.05,
) -> OpenWaterResult:
    """Compute ``K_T``, ``K_Q`` and ``eta_0`` at one advance ratio."""

    if advance_ratio < 0.0:
        raise ValueError("advance_ratio must be non-negative")

    lattice = geometry.panels()
    first_model = "kinematic" if wake_model == "induced" else wake_model
    wake_legs = _wake_legs(
        geometry,
        lattice["bound_a"],
        lattice["bound_b"],
        first_model,
        advance_ratio,
        wake_length,
        wake_steps,
    )
    inflow = inflow_velocity(advance_ratio)
    solution = solve_lifting_surface(
        lattice["bound_a"],
        lattice["bound_b"],
        lattice["control_points"],
        lattice["normals"],
        wake_legs,
        inflow,
        chords=lattice["chords"],
        spans=lattice["spans"],
        copies=geometry.blades,
        copy_axis=np.array([1.0, 0.0, 0.0]),
        core_ratio=core_ratio,
    )

    if wake_model == "induced":
        # One force-free alignment step: rebuild the helix from the local total
        # velocity of the converged kinematic solution, then re-solve.
        r_mid = np.hypot(solution.midpoints[:, 1], solution.midpoints[:, 2])
        total = inflow(solution.midpoints) + solution.induced_velocity
        tan_local = np.abs(total[:, 0] / np.maximum(TWO_PI * r_mid, 1e-9))
        legs_a = helical_wake(lattice["bound_a"], tan_local, wake_length, wake_steps)
        legs_b = helical_wake(lattice["bound_b"], tan_local, wake_length, wake_steps)
        wake_legs = np.stack([legs_a, legs_b], axis=1)
        solution = solve_lifting_surface(
            lattice["bound_a"],
            lattice["bound_b"],
            lattice["control_points"],
            lattice["normals"],
            wake_legs,
            inflow,
            chords=lattice["chords"],
            spans=lattice["spans"],
            copies=geometry.blades,
            copy_axis=np.array([1.0, 0.0, 0.0]),
            core_ratio=core_ratio,
        )

    total_velocity = inflow(solution.midpoints) + solution.induced_velocity
    inviscid_force = kutta_joukowski_force(solution, total_velocity)
    force = inviscid_force.copy()

    drag_total = 0.0
    if viscous:
        speed = np.linalg.norm(total_velocity, axis=-1)
        # physical scale: D in metres, n in rev/s, W_nd = W/(n D)
        # physical chord: the lattice is in diameter units, so Re = W c / nu
        # needs both W and c converted (W_phys = speed * n * D, c_phys = c * D)
        w_phys = speed * rev_per_second * geometry.diameter
        reynolds = w_phys * solution.chords * geometry.diameter / viscosity
        cd = section_drag_coefficient(reynolds, geometry.thickness_ratio)
        area = solution.chords * solution.spans
        drag_magnitude = 0.5 * RHO * speed**2 * cd * area  # per panel, non-dim
        # viscous drag acts *along* the relative flow, i.e. it costs torque and
        # eats thrust; the opposite sign would make the section drag drive the
        # propeller (see tests: the drag torque must add to the induced torque)
        direction = total_velocity / np.maximum(speed, 1e-12)[:, None]
        drag_force = drag_magnitude[:, None] * direction
        force = force + drag_force
        drag_total = float(np.sum(drag_magnitude))

    thrust = -float(np.sum(force[:, 0])) * geometry.blades
    torque = float(
        np.sum(solution.midpoints[:, 1] * force[:, 2] - solution.midpoints[:, 2] * force[:, 1])
    ) * geometry.blades
    thrust_inv = -float(np.sum(inviscid_force[:, 0])) * geometry.blades
    torque_inv = float(
        np.sum(
            solution.midpoints[:, 1] * inviscid_force[:, 2]
            - solution.midpoints[:, 2] * inviscid_force[:, 1]
        )
    ) * geometry.blades

    if abs(torque) < 1e-14:
        efficiency = float("nan")
        bound = float("nan")
    else:
        efficiency = advance_ratio * thrust / (TWO_PI * torque)
        ct = 8.0 * thrust / (math.pi * max(advance_ratio, 1e-12) ** 2)
        bound = 2.0 / (1.0 + math.sqrt(max(1.0 + ct, 0.0)))

    return OpenWaterResult(
        advance_ratio=float(advance_ratio),
        thrust=thrust,
        torque=torque,
        efficiency=efficiency,
        thrust_inviscid=thrust_inv,
        torque_inviscid=torque_inv,
        viscous_drag=drag_total,
        momentum_bound=bound,
        wake_model=wake_model,
        n_panels=int(lattice["bound_a"].shape[0]),
    )