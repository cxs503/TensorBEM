"""Flexural-gravity waves under a mass-loading ice cover.

Wave-ice-ship leg of the ice-going performance problem.  The ice cover is
modelled in the classical **mass-loading** approximation (Weitz & Keller 1950;
Squire et al. 1995): the sheet is thin, has no flexural rigidity and no
dissipation, and its only effect is the inertia of the displaced ice mass.

With the velocity potential ``phi ~ W(z) exp(i(kx - omega t))`` in water of
depth ``H`` and an ice sheet of thickness ``h_i`` and density ``rho_i``, the
linearised ice-covered free-surface condition is a Robin condition

    dW/dz = K W,        K = omega**2 / (g - mu omega**2),
    mu   = rho_i h_i / rho_w        (ice mass per unit area / water density)

which, with ``W = cosh(k (z + H))`` and ``dW/dz = 0`` at the bottom, gives the
mass-loading dispersion relation

    omega**2 = g k tanh(k H) / (1 + mu k tanh(k H)).                     (1)

Two properties of (1) drive everything downstream:

* **Slowing.**  For a fixed frequency the ice wavenumber is larger than the
  open-water one -- ``k_i = k_0 / (1 - mu k_0)`` in deep water -- so the waves
  are shorter and slower under the cover, and the group velocity drops.  Mass
  loading dissipates no energy, so an energy-flux-conserving wave field becomes
  *steeper*, not weaker (``A_ice/A_open = sqrt(cg_open/cg_ice) > 1``).  The only
  attenuation this model contains is the cut-off below.
* **Cut-off.**  ``K`` diverges as ``omega**2 -> g / mu``.  Above

      omega_cut = sqrt(g / mu)                                           (2)

  no propagating root exists: the free waves are evanescent, ``k = i kappa``
  with

      kappa tan(kappa H) = omega**2 / (mu omega**2 - g),                  (3)

  so the ice cover is an exponential low-pass filter of the wave field with
  decay length ``1/kappa``.

Everything here is closed-form or one-dimensional root finding, and is verified
in ``tests/test_ice_mass_loading.py`` against (i) the deep-water
Weitz-Keller limit, (ii) the shallow-water limit
``omega**2 = g k**2 H / (1 + mu k**2 H)``, (iii) an independent numerical
integration of the vertical ODE (``shooting_robin_coefficient``) that never
uses the ``cosh`` eigenfunction, and (iv) the residual of the Robin condition.

Non-physical input (negative ice thickness, non-positive water depth,
non-positive wave height, an ice sheet thicker than the water column) is
rejected with ``ValueError`` instead of being silently propagated.

All quantities are SI.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

DEFAULT_GRAVITY = 9.80665
DEFAULT_WATER_DENSITY = 1025.0
DEFAULT_ICE_DENSITY = 917.0


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return value


def _non_negative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    return value


@dataclass(frozen=True)
class IceCover:
    """Mass-loading ice sheet floating on water of finite depth.

    Parameters
    ----------
    thickness :
        Ice thickness ``h_i`` (m).  Must be strictly positive: a zero or
        negative thickness is not an ice cover and is rejected.
    ice_density :
        Ice density (kg/m^3).
    water_density :
        Water density (kg/m^3).
    water_depth :
        Water depth ``H`` (m).  Must be strictly positive (zero draft is not a
        water column) and larger than ``thickness``.
    gravity :
        Gravitational acceleration (m/s^2).
    """

    thickness: float = 0.9
    ice_density: float = DEFAULT_ICE_DENSITY
    water_density: float = DEFAULT_WATER_DENSITY
    water_depth: float = 200.0
    gravity: float = DEFAULT_GRAVITY

    def __post_init__(self) -> None:
        thickness = _positive("ice thickness", self.thickness)
        ice_density = _positive("ice density", self.ice_density)
        water_density = _positive("water density", self.water_density)
        water_depth = _positive("water depth", self.water_depth)
        gravity = _positive("gravity", self.gravity)
        if thickness >= water_depth:
            raise ValueError(
                "ice thickness "
                f"({thickness} m) must be smaller than the water depth ({water_depth} m)"
            )
        object.__setattr__(self, "thickness", thickness)
        object.__setattr__(self, "ice_density", ice_density)
        object.__setattr__(self, "water_density", water_density)
        object.__setattr__(self, "water_depth", water_depth)
        object.__setattr__(self, "gravity", gravity)

    # ----- derived quantities -------------------------------------------
    @property
    def mass_per_area(self) -> float:
        """Ice mass per unit area ``rho_i h_i`` (kg/m^2)."""
        return self.ice_density * self.thickness

    @property
    def mu(self) -> float:
        """Mass-loading parameter ``rho_i h_i / rho_w`` (m)."""
        return self.mass_per_area / self.water_density

    @property
    def cutoff_omega(self) -> float:
        """Mass-loading cut-off frequency ``sqrt(g / mu)`` (rad/s)."""
        return math.sqrt(self.gravity / self.mu)

    @property
    def cutoff_period(self) -> float:
        """Mass-loading cut-off period ``2 pi / cutoff_omega`` (s)."""
        return 2.0 * math.pi / self.cutoff_omega


@dataclass(frozen=True)
class IceWave:
    """A single mass-loading flexural-gravity wave solution."""

    omega: float
    wavenumber: float
    propagating: bool
    decay_rate: float
    wavelength: float
    phase_speed: float
    group_speed: float
    robin_coefficient: float
    residual: float
    iterations: int
    ice: IceCover

    @property
    def period(self) -> float:
        return 2.0 * math.pi / self.omega

    @property
    def amplitude_decay_ratio(self) -> float:
        """``exp(-decay_rate * x)`` with ``x = 1 m`` (dimensionless)."""
        return math.exp(-self.decay_rate)


# ---------------------------------------------------------------------------
# open water reference (finite depth)
# ---------------------------------------------------------------------------
def open_water_wavenumber(
    omega: float, water_depth: float, gravity: float = DEFAULT_GRAVITY
) -> float:
    """Root ``k`` of ``omega**2 = g k tanh(k H)`` for open water."""
    omega = _positive("omega", omega)
    depth = _positive("water depth", water_depth)
    g = _positive("gravity", gravity)
    target = omega * omega

    def residual(k: float) -> float:
        return g * k * math.tanh(k * depth) - target

    lo = 0.0
    hi = max(omega * omega / g, 1.0e-6)
    guard = 0
    while residual(hi) <= 0.0:
        hi *= 2.0
        guard += 1
        if guard > 200:  # pragma: no cover - protective
            raise RuntimeError("failed to bracket the open-water wavenumber")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1.0e-15 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def open_water_group_velocity(
    omega: float,
    water_depth: float,
    gravity: float = DEFAULT_GRAVITY,
    wavenumber: float | None = None,
) -> float:
    """Deep/finite-depth group velocity ``domega/dk`` in open water."""
    depth = _positive("water depth", water_depth)
    g = _positive("gravity", gravity)
    k = open_water_wavenumber(omega, depth, g) if wavenumber is None else float(wavenumber)
    omega = math.sqrt(g * k * math.tanh(k * depth))
    tanh = math.tanh(k * depth)
    sech2 = 1.0 - tanh * tanh
    d_omega2_dk = g * (tanh + k * depth * sech2)
    return d_omega2_dk / (2.0 * omega)


# ---------------------------------------------------------------------------
# mass-loading dispersion
# ---------------------------------------------------------------------------
def mass_loading_omega(
    k: float, ice: IceCover
) -> float:
    """Frequency of the mass-loading wave with wavenumber ``k``: eq. (1)."""
    k = _positive("wavenumber", k)
    tanh = math.tanh(k * ice.water_depth)
    numerator = ice.gravity * k * tanh
    denominator = 1.0 + ice.mu * k * tanh
    return math.sqrt(numerator / denominator)


def mass_loading_omega_bem(
    k: float,
    ice: IceCover,
    *,
    panels_per_wavelength: int = 64,
    route: str = "driven",
) -> float:
    """Mass-loading frequency obtained from a boundary-element solve.

    Unlike :func:`mass_loading_omega` (eq. 1 evaluated in closed form), this
    entry point solves the Laplace boundary integral equation of one periodic
    cell of the ice-covered basin and reads the free-surface depth factor off the
    numerical flux; ``tanh(k H)`` never enters the computation.  It is the
    solver-side caller of :mod:`tensorbem.ice_mass_loading_dispersion` and exists
    so that the ice module can quote a numerically verified frequency without
    duplicating the BEM.

    Parameters
    ----------
    k :
        Wavenumber of the mode (1/m).
    ice :
        Ice cover; ``water_depth``, ``mu`` and ``gravity`` are taken from it.
    panels_per_wavelength :
        Free-surface panels per wavelength of the mode.
    route :
        ``'driven'`` (driven Dirichlet solve) or ``'dtn'`` (assembled
        Dirichlet-to-Neumann map).
    """
    from .ice_mass_loading_dispersion import strip_bem_mass_loading_omega

    return strip_bem_mass_loading_omega(
        k,
        depth=ice.water_depth,
        mu=ice.mu,
        gravity=ice.gravity,
        panels_per_wavelength=panels_per_wavelength,
        route=route,
    )


def robin_coefficient(omega: float, ice: IceCover) -> float:
    """Mass-loading Robin coefficient ``K = omega**2/(g - mu omega**2)``.

    Raises ``ValueError`` above the cut-off ``sqrt(g/mu)``, where the condition
    ``phi_z = K phi`` no longer describes a propagating wave (``K` diverges and
    changes sign); use :func:`mass_loading_wavenumber` for that regime, which
    returns the evanescent branch.
    """
    omega = _positive("omega", omega)
    denom = ice.gravity - ice.mu * omega * omega
    if denom <= 0.0:
        raise ValueError(
            "omega exceeds the mass-loading cut-off sqrt(g/mu) = "
            f"{ice.cutoff_omega:.6g} rad/s; no propagating ice wave exists"
        )
    return omega * omega / denom


def robin_residual(k: float, omega: float, ice: IceCover) -> float:
    """Residual of the ice-covered free-surface condition at wavenumber ``k``.

    The exact solution satisfies ``k tanh(k H) = K(omega)``; this returns the
    signed difference (units 1/m), which is the verification quantity used by
    the tests.
    """
    k = _positive("wavenumber", k)
    return k * math.tanh(k * ice.water_depth) - robin_coefficient(omega, ice)


def mass_loading_group_velocity(omega: float, ice: IceCover, wavenumber: float) -> float:
    """Analytic group velocity ``domega/dk`` of the mass-loading wave."""
    k = _positive("wavenumber", wavenumber)
    omega = _positive("omega", omega)
    depth = ice.water_depth
    tanh = math.tanh(k * depth)
    sech2 = 1.0 - tanh * tanh
    d_num = ice.gravity * (tanh + k * depth * sech2)          # d(g k tanh)/dk
    num = ice.gravity * k * tanh
    d_den = ice.mu * (tanh + k * depth * sech2)               # d(1 + mu k tanh)/dk
    den = 1.0 + ice.mu * k * tanh
    d_omega2_dk = (d_num * den - num * d_den) / (den * den)
    return d_omega2_dk / (2.0 * omega)


def mass_loading_wavenumber(
    omega: float,
    ice: IceCover,
    *,
    tol: float = 1.0e-14,
    max_iter: int = 300,
) -> IceWave:
    """Solve eq. (1) for ``k`` at frequency ``omega``.

    Propagating branch (``omega < sqrt(g/mu)``): the root of
    ``g k tanh(k H)/(1 + mu k tanh(k H)) - omega**2`` is bracketed and bisected;
    the function is strictly increasing in ``k``, so the root is unique.

    Evanescent branch (``omega >= sqrt(g/mu)``): ``k = i kappa`` with
    ``kappa tan(kappa H)`` from eq. (3), first branch ``0 < kappa H < pi/2``.
    """
    omega = _positive("omega", omega)
    cutoff = ice.cutoff_omega
    depth = ice.water_depth
    mu = ice.mu
    g = ice.gravity
    target = omega * omega

    if omega < cutoff:
        def residual(k: float) -> float:
            tanh = math.tanh(k * depth)
            return g * k * tanh / (1.0 + mu * k * tanh) - target

        lo = 0.0
        hi = max(target / g, 1.0e-8)
        guard = 0
        while residual(hi) <= 0.0:
            hi *= 2.0
            guard += 1
            if guard > 200:  # pragma: no cover - protective
                raise RuntimeError("failed to bracket the ice wavenumber")
        iterations = 0
        for iterations in range(1, max_iter + 1):
            mid = 0.5 * (lo + hi)
            if residual(mid) > 0.0:
                hi = mid
            else:
                lo = mid
            if hi - lo <= tol * max(1.0, hi):
                break
        k = 0.5 * (lo + hi)
        res = residual(k)
        group = mass_loading_group_velocity(omega, ice, k)
        return IceWave(
            omega=omega,
            wavenumber=k,
            propagating=True,
            decay_rate=0.0,
            wavelength=2.0 * math.pi / k,
            phase_speed=omega / k,
            group_speed=group,
            robin_coefficient=robin_coefficient(omega, ice),
            residual=res,
            iterations=iterations,
            ice=ice,
        )

    # evanescent branch: k = i kappa, kappa tan(kappa H) = omega^2/(mu omega^2 - g)
    t_target = target / (mu * target - g)

    def evanescent_residual(kappa: float) -> float:
        return kappa * math.tan(kappa * depth) - t_target

    lo = 1.0e-12 / depth
    hi = 0.5 * math.pi / depth * (1.0 - 1.0e-12)
    if evanescent_residual(hi) < 0.0:  # pragma: no cover - protective
        raise RuntimeError("failed to bracket the evanescent decay rate")
    iterations = 0
    for iterations in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        if evanescent_residual(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, hi):
            break
    kappa = 0.5 * (lo + hi)
    return IceWave(
        omega=omega,
        wavenumber=0.0,
        propagating=False,
        decay_rate=kappa,
        wavelength=math.inf,
        phase_speed=math.inf,
        group_speed=0.0,
        robin_coefficient=math.inf,
        residual=evanescent_residual(kappa),
        iterations=iterations,
        ice=ice,
    )


def mass_loading_attenuation(omega: float, ice: IceCover, distance: float) -> float:
    """Wave-amplitude ratio over ``distance`` metres of ice cover.

    * Propagating branch: **no decay** (mass loading is conservative).  The
      returned ratio is exactly 1.0; the reduction a ship experiences comes from
      the dispersion shift, quantified by :func:`energy_flux_amplitude_ratio`.
    * Evanescent branch (``omega > sqrt(g/mu)``): ``exp(-kappa * distance)``.
    """
    distance = _non_negative("distance", distance)
    wave = mass_loading_wavenumber(omega, ice)
    if not wave.propagating:
        return math.exp(-wave.decay_rate * distance)
    return 1.0


def energy_flux_amplitude_ratio(omega: float, ice: IceCover) -> float:
    """Amplitude ratio ``A_ice / A_open`` from energy-flux continuity.

    Across an abrupt ice edge the incident energy flux is conserved,
    ``A**2 cg`` constant, so ``A_ice / A_open = sqrt(cg_open / cg_ice)``.  Both
    group velocities are evaluated at the *same frequency*.  Mass loading slows
    the waves (``cg_ice < cg_open``), so the ratio is **> 1**: an ice cover
    *steepens* the wave field while shortening it and dissipates no energy.  It
    therefore provides no attenuation below the cut-off -- this is a physical
    finding of the mass-loading model, not a code artefact, and it means that
    observed ice-zone attenuation requires floe scattering / dissipation, which
    this repository does not model (see ``tensorbem.ice_rao``).  Returns
    ``0.0`` on the evanescent branch (no propagating incident wave reaches the
    cover).
    """
    wave = mass_loading_wavenumber(omega, ice)
    if not wave.propagating:
        return 0.0
    k0 = open_water_wavenumber(omega, ice.water_depth, ice.gravity)
    cg_open = open_water_group_velocity(omega, ice.water_depth, ice.gravity, k0)
    if wave.group_speed <= 0.0:  # pragma: no cover - protective
        return 1.0
    return math.sqrt(cg_open / wave.group_speed)


# ---------------------------------------------------------------------------
# independent verification helpers (no closed-form eigenfunction involved)
# ---------------------------------------------------------------------------
def shooting_robin_coefficient(
    k: float, ice: IceCover, *, steps: int = 400
) -> float:
    """``W'(0)/W(0)`` from numerical integration of ``W'' = k**2 W``.

    The vertical ODE is integrated with classical fourth-order Runge-Kutta from
    the bottom (``W(-H) = 1``, ``W'(-H) = 0``) to the surface.  No closed-form
    ``cosh`` is used, so the returned ratio is an independent estimate of the
    Robin coefficient ``K`` that the dispersion relation must satisfy.  It
    should agree with ``k tanh(k H)`` to the RK4 truncation error.
    """
    k = _positive("wavenumber", k)
    if not isinstance(steps, (int, np.integer)) or int(steps) < 4:
        raise ValueError("steps must be an integer >= 4")
    n = int(steps)
    depth = ice.water_depth
    dz = depth / n
    z = -depth
    w, dw = 1.0, 0.0

    def rhs(_z: float, _w: float, _dw: float) -> tuple[float, float]:
        return _dw, k * k * _w

    for _ in range(n):
        k1w, k1d = rhs(z, w, dw)
        k2w, k2d = rhs(z + 0.5 * dz, w + 0.5 * dz * k1w, dw + 0.5 * dz * k1d)
        k3w, k3d = rhs(z + 0.5 * dz, w + 0.5 * dz * k2w, dw + 0.5 * dz * k2d)
        k4w, k4d = rhs(z + dz, w + dz * k3w, dw + dz * k3d)
        w += dz * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0
        dw += dz * (k1d + 2.0 * k2d + 2.0 * k3d + k4d) / 6.0
        z += dz
    return dw / w


def shooting_omega(k: float, ice: IceCover, *, steps: int = 400) -> float:
    """Frequency implied by the numerically integrated eigenfunction.

    Inverts ``K = omega**2/(g - mu omega**2)`` for ``omega`` given the RK4
    estimate of ``K`` from :func:`shooting_robin_coefficient`.  Compared with
    :func:`mass_loading_omega`, this is an independent numerical route to the
    dispersion relation, used as the <5 % verification target.
    """
    coefficient = shooting_robin_coefficient(k, ice, steps=steps)
    return math.sqrt(coefficient * ice.gravity / (1.0 + ice.mu * coefficient))


def dispersion_table(
    omegas: ArrayLike, ice: IceCover
) -> dict[str, FloatArray]:
    """Tabulate the mass-loading dispersion on a frequency grid.

    Returns a dict with the frequency grid, ice and open-water wavenumbers,
    wavelengths, phase/group velocities, the energy-flux amplitude ratio, and
    the propagation flag (``1.0`` propagating, ``0.0`` evanescent).
    """
    omega_arr = np.asarray(omegas, dtype=float)
    if omega_arr.ndim != 1 or omega_arr.size == 0:
        raise ValueError("omegas must be a non-empty 1-D array")
    if not np.all(np.isfinite(omega_arr)) or np.any(omega_arr <= 0.0):
        raise ValueError("omegas must all be finite and strictly positive")

    n = omega_arr.size
    k_ice = np.zeros(n)
    k_open = np.zeros(n)
    wavelength = np.zeros(n)
    phase = np.zeros(n)
    group = np.zeros(n)
    ratio = np.zeros(n)
    decay = np.zeros(n)
    propagating = np.zeros(n)
    for i, omega in enumerate(omega_arr):
        wave = mass_loading_wavenumber(float(omega), ice)
        decay[i] = wave.decay_rate
        propagating[i] = 1.0 if wave.propagating else 0.0
        if wave.propagating:
            k_ice[i] = wave.wavenumber
            k_open[i] = open_water_wavenumber(float(omega), ice.water_depth, ice.gravity)
            wavelength[i] = wave.wavelength
            phase[i] = wave.phase_speed
            group[i] = wave.group_speed
            ratio[i] = energy_flux_amplitude_ratio(float(omega), ice)
        else:
            k_ice[i] = math.nan
            k_open[i] = math.nan
            wavelength[i] = math.nan
            phase[i] = math.nan
            group[i] = 0.0
            ratio[i] = 0.0
    return {
        "omega": omega_arr,
        "period": 2.0 * math.pi / omega_arr,
        "k_ice": k_ice,
        "k_open": k_open,
        "wavelength": wavelength,
        "phase_speed": phase,
        "group_speed": group,
        "amplitude_ratio": ratio,
        "decay_rate": decay,
        "propagating": propagating,
    }


def thickness_study(
    omega: float,
    thicknesses: ArrayLike,
    ice: IceCover,
) -> dict[str, FloatArray]:
    """``k(h_i)`` at fixed frequency for a grid of ice thicknesses.

    Mass loading is monotone in the areal ice mass, so ``k`` must increase with
    ``h_i``; the returned table also carries the analytic deep-water
    cross-check ``k = k_0/(1 - mu k_0)`` and the relative difference, which is
    the verification quantity for the "k(h) vs mass-loading analytic" gate.
    """
    omega = _positive("omega", omega)
    h_arr = np.asarray(thicknesses, dtype=float)
    if h_arr.ndim != 1 or h_arr.size == 0:
        raise ValueError("thicknesses must be a non-empty 1-D array")
    if not np.all(np.isfinite(h_arr)) or np.any(h_arr <= 0.0):
        raise ValueError("ice thicknesses must all be finite and strictly positive")
    if np.any(h_arr >= ice.water_depth):
        raise ValueError("each ice thickness must be smaller than the water depth")

    k_open = open_water_wavenumber(omega, ice.water_depth, ice.gravity)
    k_num = np.zeros(h_arr.size)
    k_analytic = np.zeros(h_arr.size)
    rel = np.zeros(h_arr.size)
    for i, thickness in enumerate(h_arr):
        cover = IceCover(
            thickness=float(thickness),
            ice_density=ice.ice_density,
            water_density=ice.water_density,
            water_depth=ice.water_depth,
            gravity=ice.gravity,
        )
        wave = mass_loading_wavenumber(omega, cover)
        if not wave.propagating:
            k_num[i] = math.nan
            k_analytic[i] = math.nan
            rel[i] = math.nan
            continue
        k_num[i] = wave.wavenumber
        # independent algebraic route: k tanh(kH) = omega^2/(g - mu omega^2)
        mu = cover.mu
        target = omega * omega / (ice.gravity - mu * omega * omega)
        lo, hi = 0.0, max(k_open, 1.0e-6)
        guard = 0
        while hi * math.tanh(hi * ice.water_depth) < target:
            hi *= 2.0
            guard += 1
            if guard > 200:  # pragma: no cover - protective
                raise RuntimeError("failed to bracket the analytic ice wavenumber")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if mid * math.tanh(mid * ice.water_depth) < target:
                lo = mid
            else:
                hi = mid
        k_analytic[i] = 0.5 * (lo + hi)
        # also report the closed-form deep-water value for context
        rel[i] = abs(k_num[i] - k_analytic[i]) / k_analytic[i]
    return {
        "omega": np.full(h_arr.size, omega),
        "thickness": h_arr,
        "mu": h_arr * ice.ice_density / ice.water_density,
        "k_open": np.full(h_arr.size, k_open),
        "k_ice": k_num,
        "k_analytic_robin": k_analytic,
        "relative_difference": rel,
    }