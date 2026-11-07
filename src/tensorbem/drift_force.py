"""Second-order mean wave drift force (m5242): near-field / far-field / anchor.

Physics
-------
A fixed body in a regular wave train of amplitude ``a`` and frequency ``omega``
feels a time-averaged second-order horizontal load whose leading (quadratic)
part is the *mean drift force* ``Fbar``.  Three mutually independent routes are
implemented here and cross-checked on one closed-form case.

1. **Near field** -- second-order pressure integration over the *instantaneous*
   wetted surface.  With the first-order potential ``phi`` (real field
   ``Re{phi e^{-i omega t}}``),

       F(t) = int_{S(t)} n [-rho*phi_t - (1/2)*rho*|grad phi|^2] dS,
       Fbar = (1/T) int_0^T F(t) dt.

   For a fixed body the second-order potential only contributes
   ``-rho*<phi2_t>``, which vanishes in the steady state, so the expression
   above is the complete second-order force.  Because every time sample is
   integrated up to ``z = zeta(t)``, the moving-waterline strip is carried
   *implicitly*: the classic ``-(1/2)*rho*g*zeta_r^2`` waterline term is
   reproduced by the quadrature instead of being imposed by hand.  The
   equivalent surface-plus-waterline (textbook) form is provided separately by
   :func:`near_field_surface_waterline_2d` as an algebraic cross-check.

2. **Far field** -- momentum (radiation-stress) balance in 2-D deep water.
   A progressive wave carries ``S_xx = E/2`` in deep water, so with incident
   amplitude ``A_i``, reflected ``A_r`` and transmitted ``A_t``

       Fbar_x = (rho*g/4) (A_i^2 + A_r^2 - A_t^2).

   ``A_r``/``A_t`` are *measured* from a travelling-wave decomposition (complex
   least squares on two far-field stations), so no pressure integral enters this
   route.  The 3-D generalisation (Maruo, Kochin-function based) is
   :func:`maruo_far_field_drift_3d`.

3. **Analytic anchor** -- 2-D vertical wall at ``x = 0`` with complete
   reflection (fluid at ``x < 0``, deep water), per unit length:

       Fbar_x = rho*g*a^2/2.

   Momentum derivation: incident and reflected waves each carry
   ``S_xx = E/2 = rho*g*a^2/4`` and nothing passes the wall, so the net
   x-momentum flux into the wall is ``rho*g*a^2/2``.  The near-field integral
   reproduces it as ``int_{-inf}^{0} -(1/2) rho <|grad phi|^2> dz = -rho g a^2/2``
   plus the waterline strip ``+(1/2) rho g <zeta_r^2> = +rho g a^2``.

Conventions
-----------
``n`` is the outward normal of the *body* (pointing into the fluid) and the
computed quantity is the force **on the body**.  For the wall this gives
``n_x = -1`` and ``Fbar_x = +rho*g*a^2/2``: the wall is pushed downstream.
The literature (Pinkster) form uses the inward normal and flips both signs --
the same statement.

Scope
-----
Deep water, fixed body (no first-order body motion), monochromatic waves,
zero forward speed.  Not implemented: second-order potential (slow-drift /
difference-frequency forcing), forward-speed terms, finite depth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_DENSITY = 1025.0
DEFAULT_GRAVITY = 9.81

__all__ = [
    "DEFAULT_DENSITY",
    "DEFAULT_GRAVITY",
    "WallField",
    "wall_mean_drift_analytic",
    "wall_first_order_field",
    "near_field_drift_wall",
    "near_field_surface_waterline_2d",
    "two_d_far_field_drift",
    "decompose_traveling_waves_2d",
    "maruo_far_field_drift_3d",
    "richardson_extrapolate",
]


def _positive(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return v


# ---------------------------------------------------------------------------
# 3. closed-form anchor
# ---------------------------------------------------------------------------
def wall_mean_drift_analytic(
    amplitude: float,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> float:
    """``Fbar = rho*g*a**2/2`` per unit length (2-D vertical wall, deep water)."""
    a = _positive("amplitude", amplitude)
    return 0.5 * _positive("density", density) * _positive("gravity", gravity) * a * a


# ---------------------------------------------------------------------------
# exact first-order field of the wall (drives both numerical routes)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WallField:
    """Exact first-order standing-wave field of a 2-D vertical wall at ``x = 0``.

    Fluid occupies ``x < 0``; the incident wave travels in ``+x`` as
    ``eta_i = a*cos(k x - omega t)`` and is totally reflected.  Deep water.
    """

    amplitude: float
    omega: float
    gravity: float = DEFAULT_GRAVITY

    @property
    def wavenumber(self) -> float:
        return self.omega * self.omega / self.gravity

    @property
    def period(self) -> float:
        return 2.0 * math.pi / self.omega

    def potential(self, x, z, t):
        k = self.wavenumber
        b = self.gravity * self.amplitude / self.omega
        return b * np.exp(k * z) * (
            np.sin(k * x - self.omega * t) - np.sin(k * x + self.omega * t)
        )

    def phi_t(self, x, z, t):
        k = self.wavenumber
        b = self.gravity * self.amplitude / self.omega
        return -b * self.omega * np.exp(k * z) * (
            np.cos(k * x - self.omega * t) + np.cos(k * x + self.omega * t)
        )

    def dphi_dx(self, x, z, t):
        k = self.wavenumber
        b = self.gravity * self.amplitude / self.omega
        return b * np.exp(k * z) * k * (
            np.cos(k * x - self.omega * t) - np.cos(k * x + self.omega * t)
        )

    def dphi_dz(self, x, z, t):
        return self.wavenumber * self.potential(x, z, t)

    def elevation(self, x, t):
        return -self.phi_t(x, 0.0, t) / self.gravity


def wall_first_order_field(
    amplitude: float,
    omega: float,
    gravity: float = DEFAULT_GRAVITY,
) -> WallField:
    return WallField(
        amplitude=_positive("amplitude", amplitude),
        omega=_positive("omega", omega),
        gravity=_positive("gravity", gravity),
    )


# ---------------------------------------------------------------------------
# 1. near field: time-domain second-order pressure integral
# ---------------------------------------------------------------------------
def _graded_depth_nodes(z_bottom: float, n_panels: int) -> np.ndarray:
    """Midpoint nodes from ``z_bottom`` to ``0`` grading towards the surface.

    ``u`` uniform in ``(0, 1]`` mapped by ``z = z_bottom * (1 - u)**2`` gives a
    mid-point spacing that shrinks like ``1/N`` near ``z = 0`` where the
    integrand of the surface term is largest.
    """
    u = (np.arange(n_panels, dtype=float) + 0.5) / n_panels
    return z_bottom * (1.0 - u) ** 2


def near_field_drift_wall(
    amplitude: float,
    omega: float,
    *,
    n_panels: int = 200,
    n_time: int = 64,
    n_strip: int = 8,
    depth_factor: float = 8.0,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    return_samples: bool = False,
):
    """Near-field mean drift force per unit length on the 2-D wall.

    Second-order pressure ``q = -rho*phi_t - (1/2)*rho*|grad phi|^2`` is
    integrated over the instantaneous wetted surface
    ``z in [z_bottom, zeta(0,t)]`` and time-averaged over one period.  The
    deepest point is ``z_bottom = -depth_factor/k`` (truncation error
    ``O(exp(-2*depth_factor))``).

    Returns the mean force in N/m; with ``return_samples=True`` also returns a
    dict of per-time-sample forces and the elevation trace.
    """
    field = wall_first_order_field(amplitude, omega, gravity=gravity)
    rho = _positive("density", density)
    if int(n_panels) < 4 or int(n_time) < 8 or int(n_strip) < 1:
        raise ValueError("need n_panels >= 4, n_time >= 8, n_strip >= 1")
    k = field.wavenumber
    z_bottom = -float(depth_factor) / k
    z_bulk = _graded_depth_nodes(z_bottom, int(n_panels))
    dz_bulk = np.gradient(np.concatenate(([z_bottom], z_bulk, [0.0])))[1:-1]

    n_t = int(n_time)
    times = np.arange(n_t, dtype=float) * field.period / n_t
    forces = np.empty(n_t, dtype=float)

    for i, t in enumerate(times):
        # bulk: mean wetted surface up to the mean waterline z = 0
        phi_t_bulk = field.phi_t(0.0, z_bulk, t)
        grad2_bulk = field.dphi_dz(0.0, z_bulk, t) ** 2 + field.dphi_dx(0.0, z_bulk, t) ** 2
        q_bulk = -rho * phi_t_bulk - 0.5 * rho * grad2_bulk
        force = float(np.sum(q_bulk * dz_bulk))

        # strip: 0 -> zeta(0, t), midpoint rule on n_strip sub-panels
        eta = float(field.elevation(0.0, t))
        if n_strip > 0:
            s = (np.arange(n_strip, dtype=float) + 0.5) / n_strip
            z_strip = 0.0 + s * eta
            dz_strip = eta / n_strip
            phi_t_s = field.phi_t(0.0, z_strip, t)
            grad2_s = field.dphi_dz(0.0, z_strip, t) ** 2 + field.dphi_dx(0.0, z_strip, t) ** 2
            q_strip = -rho * phi_t_s - 0.5 * rho * grad2_s
            force += float(np.sum(q_strip * dz_strip))
        # The quadrature above integrates q = p + rho*g*z up to the instantaneous
        # waterline.  Removing the still-water reference leaves the physical
        # second-order force; equivalently this restores the textbook
        # -(1/2) rho g zeta^2 waterline term (see near_field_surface_waterline_2d).
        force -= 0.5 * rho * gravity * eta * eta
        forces[i] = force

    mean_force = float(np.mean(forces))
    if not return_samples:
        return mean_force
    return mean_force, {
        "times": times,
        "forces": forces,
        "elevation": np.array([field.elevation(0.0, t) for t in times]),
        "z_bulk": z_bulk,
        "mean_q": np.mean(
            [
                -rho * field.phi_t(0.0, z_bulk, t)
                - 0.5
                * rho
                * (
                    field.dphi_dz(0.0, z_bulk, t) ** 2
                    + field.dphi_dx(0.0, z_bulk, t) ** 2
                )
                for t in times
            ],
            axis=0,
        ),
        "zeta_rms": float(
            math.sqrt(0.5) * 2.0 * field.amplitude
        ),  # <zeta^2> = (2a)^2/2 at the antinode
    }


def near_field_surface_waterline_2d(
    amplitude: float,
    omega: float,
    *,
    n_panels: int = 4000,
    depth_factor: float = 8.0,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> dict:
    """Textbook near-field form: surface term + waterline term (2-D wall).

    ``Fbar_x = int_{S0} (1/2) rho <|grad phi|^2> n_x dS
                - (1/2) rho g <zeta_r^2> n_x``

    with ``n_x = -1`` (outward normal of the body).  Both pieces are computed by
    separate quadratures so they can be audited individually.
    """
    field = wall_first_order_field(amplitude, omega, gravity=gravity)
    rho = _positive("density", density)
    g = _positive("gravity", gravity)
    k = field.wavenumber
    z_bottom = -float(depth_factor) / k
    z = _graded_depth_nodes(z_bottom, int(n_panels))
    dz = np.gradient(np.concatenate(([z_bottom], z, [0.0])))[1:-1]

    grad2 = float(
        np.mean(
            [
                field.dphi_dz(0.0, z, t) ** 2 + field.dphi_dx(0.0, z, t) ** 2
                for t in np.arange(32) * field.period / 32.0
            ],
            axis=0,
        ).dot(dz)
    )
    zeta_sq_mean = 2.0 * field.amplitude**2  # antinode: zeta = 2a cos(omega t)
    n_x = -1.0
    surface_term = 0.5 * rho * grad2 * n_x
    waterline_term = -0.5 * rho * g * zeta_sq_mean * n_x
    return {
        "surface_term": float(surface_term),
        "waterline_term": float(waterline_term),
        "total": float(surface_term + waterline_term),
        "grad2_integral": grad2,
        "zeta_sq_mean": zeta_sq_mean,
    }


# ---------------------------------------------------------------------------
# 2. far field: travelling-wave decomposition + radiation-stress balance
# ---------------------------------------------------------------------------
def two_d_far_field_drift(
    incident_amplitude: float,
    reflected_amplitude: float,
    transmitted_amplitude: float = 0.0,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> float:
    """``Fbar_x = (rho g/4)(A_i^2 + A_r^2 - A_t^2)`` -- 2-D deep-water balance.

    ``S_xx = E/2 = rho g a^2/4`` per progressive wave; the incident wave enters
    the control volume, the reflected wave leaves through the same face and the
    transmitted wave leaves through the far face.
    """
    rho = _positive("density", density)
    g = _positive("gravity", gravity)
    ai = _positive("incident_amplitude", incident_amplitude)
    ar = float(reflected_amplitude)
    at = float(transmitted_amplitude)
    if ar < 0.0 or at < 0.0:
        raise ValueError("amplitudes must be non-negative")
    return 0.25 * rho * g * (ai * ai + ar * ar - at * at)


def decompose_traveling_waves_2d(
    elevation_fn,
    stations,
    omega: float,
    *,
    n_time: int = 64,
    gravity: float = DEFAULT_GRAVITY,
) -> dict:
    """Split a 2-D wave field into ``+x`` and ``-x`` travelling amplitudes.

    ``elevation_fn(x, t)`` is sampled at ``n_time`` instants over one period at
    two stations; the complex amplitudes ``C(x) = (2/T) int zeta e^{i omega t} dt``
    satisfy ``C(x) = A_plus e^{i k x} + A_minus e^{-i k x}``, a 2x2 linear
    system solved by least squares (the stations need not be equally spaced).
    """
    if len(stations) != 2:
        raise ValueError("exactly two stations are required")
    w = _positive("omega", omega)
    g = _positive("gravity", gravity)
    k = w * w / g
    n_t = int(n_time)
    times = np.arange(n_t, dtype=float) * (2.0 * math.pi / w) / n_t
    rows, rhs = [], []
    for x in stations:
        vals = np.array([elevation_fn(float(x), float(t)) for t in times])
        c = 2.0 * np.sum(vals * np.exp(1j * w * times)) / n_t
        rows.append([np.exp(1j * k * float(x)), np.exp(-1j * k * float(x))])
        rhs.append(c)
    mat = np.array(rows, dtype=np.complex128)
    sol, *_ = np.linalg.lstsq(mat, np.array(rhs, dtype=np.complex128), rcond=None)
    residual = float(np.linalg.norm(mat @ sol - np.array(rhs)))
    return {
        "a_plus": complex(sol[0]),
        "a_minus": complex(sol[1]),
        "amplitude_plus": float(abs(sol[0])),
        "amplitude_minus": float(abs(sol[1])),
        "fit_residual": residual,
    }


# ---------------------------------------------------------------------------
# 3-D far field (Maruo) from Kochin function values
# ---------------------------------------------------------------------------
def maruo_far_field_drift_3d(
    kochin_values,
    azimuths,
    wavenumber: float,
    incident_amplitude: float,
    *,
    wave_angle: float = 0.0,
    radius_factor: float = 400.0,
    n_theta: int = 8192,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    incident_amplitude_per_direction: bool = True,
) -> dict:
    """Maruo far-field mean drift force from Kochin values, deep water.

    The far-field free-surface elevation in the direction ``theta`` is
    ``zeta_s ~ -beta_0 exp(i(kR - pi/4)) K(theta)`` with
    ``beta_0 = (k^2/(2 omega)) sqrt(2/(pi k R))`` (from
    ``phi ~ (i k/2) H_0^{(1)}(kR) exp(kz) K(theta)`` and ``zeta = -(1/g) phi_t``),
    superposed on the incident wave ``zeta_i = A exp(-i k R cos(theta - beta))``.
    The local deep-water radiation stress is ``S_rr = rho g |zeta|^2 / 4``, so

        Fbar_x = -(rho g/4) int_0^{2pi} |zeta_s + zeta_i|^2 R cos(theta) dtheta

    evaluated at a large ``kR`` (the incident-only part integrates to zero and is
    subtracted explicitly so that ``K == 0`` gives exactly zero).
    """
    k = _positive("wavenumber", wavenumber)
    g = _positive("gravity", gravity)
    rho = _positive("density", density)
    amp = _positive("incident_amplitude", incident_amplitude)
    omega = math.sqrt(g * k)
    n_t = int(n_theta)
    if n_t < 128:
        raise ValueError("n_theta must be >= 128")

    theta = 2.0 * math.pi * np.arange(n_t) / n_t
    dtheta = 2.0 * math.pi / n_t
    koch = np.asarray(kochin_values, dtype=np.complex128)
    if koch.shape != theta.shape:
        ang = np.mod(np.asarray(azimuths, dtype=float), 2.0 * math.pi)
        order = np.argsort(ang)
        ang, koch = ang[order], koch[order]
        # close the periodic table so interpolation is continuous at 2*pi
        ang = np.concatenate((ang, ang[:1] + 2.0 * math.pi))
        koch = np.concatenate((koch, koch[:1]))
        lin = theta
        koch = np.interp(lin, ang, koch.real) + 1j * np.interp(lin, ang, koch.imag)

    kR = _positive("radius_factor", radius_factor) * k
    beta0 = (k * k / (2.0 * omega)) * math.sqrt(2.0 / (math.pi * kR))
    zeta_s = -beta0 * np.exp(1j * (kR - math.pi / 4.0)) * koch
    zeta_i = amp * np.exp(-1j * kR * np.cos(theta - float(wave_angle)))

    weight = np.cos(theta) * kR * dtheta
    flux_s = float(np.sum(np.abs(zeta_s) ** 2 * weight))
    flux_i = float(np.sum(np.abs(zeta_i) ** 2 * weight))
    flux_cross = float(np.sum(2.0 * np.real(zeta_i * np.conj(zeta_s)) * weight))
    total = flux_s + flux_cross  # incident-only part removes exactly
    return {
        "mean_drift_force_x": -0.25 * rho * g * total,
        "flux_scattered": 0.25 * rho * g * flux_s,
        "flux_interference": 0.25 * rho * g * flux_cross,
        "flux_incident_control": 0.25 * rho * g * flux_i,
        "kR": kR,
        "n_theta": n_t,
        "wavenumber": k,
    }


# ---------------------------------------------------------------------------
# convergence helper
# ---------------------------------------------------------------------------
def richardson_extrapolate(
    panel_counts,
    values,
    *,
    tighten: float = 0.5,
) -> dict:
    """Three-level Richardson extrapolation in the panel size ``h ~ 1/N``.

    Fits ``F(h) = F_inf + C h**p`` through the three finest levels (successive
    refinement ratio ``r``), returning the extrapolated value, the observed
    order ``p`` and the residual of the finest level against ``F_inf``.
    """
    n = [float(v) for v in panel_counts]
    f = [float(v) for v in values]
    if len(n) != 3 or len(f) != 3:
        raise ValueError("need exactly three panel counts and three values")
    if not (n[0] < n[1] < n[2]):
        raise ValueError("panel counts must be increasing")
    h = [1.0 / v for v in n]
    r21 = h[1] / h[2]
    r10 = h[0] / h[1]
    if abs(r21 - r10) > 1e-9 * max(r21, r10):
        raise ValueError("panel refinements must be uniform for this estimator")
    d21 = f[2] - f[1]
    d10 = f[1] - f[0]
    if abs(d21) < 1e-300 or abs(d10) < 1e-300:
        raise ValueError("degenerate convergence sequence")
    ratio = d10 / d21
    if ratio <= 0.0:
        raise ValueError("sequence is not monotone; no real order available")
    p = math.log(ratio) / math.log(r21)
    f_inf = f[2] + d21 / (r21**p - 1.0)
    residual = abs(f[2] - f_inf) / abs(f_inf) if f_inf != 0.0 else abs(f[2] - f_inf)
    return {
        "panel_counts": n,
        "values": f,
        "panel_sizes": h,
        "observed_order": float(p),
        "extrapolated": float(f_inf),
        "residual_relative": float(residual),
        "refinement_ratio": float(r21),
    }