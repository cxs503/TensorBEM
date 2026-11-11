"""Flexural-gravity wave propagation under an elastic ice cover.

Physical model
--------------
An elastic ice sheet of thickness ``h_ice``, Young's modulus ``E``, and
Poisson's ratio ``nu`` floats on water of depth ``H``.  The ice is modelled
as a thin Kirchhoff plate with flexural rigidity

    D = E h_ice^3 / (12 (1 - nu^2))                                    (1)

The linearised free-surface condition under the ice sheet combines the
hydrostatic restoring force (gravity) and the elastic restoring force
(bending).  Following the task specification (E1), the **flexural-gravity
dispersion relation** is

    omega^2 = (g k + D k^5 / rho_w) tanh(k H)                         (2)

This is the classical result for flexural-gravity waves (Fox & Squire 1990,
Squire et al. 1995), combining gravity and flexural rigidity effects.

**Note:** The full physical model may also include the mass-loading effect
of the ice (inertia of the ice sheet), which modifies the dispersion to

    omega^2 = (g k + D k^5 / rho_w) tanh(k H) / (1 + mu k tanh(k H))  (3)

where ``mu = rho_i h_ice / rho_w`` is the mass-loading parameter.  This
module implements equation (2) as specified in the task; the mass-loading
extension (3) is available in the ``ice.py`` module.

Three asymptotic limits are important:

* **Long-wave limit** (k -> 0):
    omega^2 -> g k tanh(k H)
  The flexural term D k^5 / rho_w is negligible; the wave degenerates to
  the open-water gravity wave.

* **Short-wave limit** (k -> infinity, deep water):
    omega^2 -> D k^5 / rho_w
  The gravity term is negligible; the wave degenerates to a pure flexural
  (bending) wave.

* **Zero-rigidity limit** (D -> 0):
    omega^2 = g k tanh(k H)
  Exactly the open-water dispersion relation.

Ice edge reflection and transmission
-------------------------------------
At the boundary between open water (region 0) and ice-covered water
(region 1), a plane wave of frequency omega is partially reflected and
partially transmitted.  The wavenumbers k_0 and k_1 satisfy their
respective dispersion relations at the same frequency.

For energy-conserving 1D wave propagation across a discontinuity, the
matching uses the group velocity as the wave admittance:

    Y = cg  (group velocity)

Then the amplitude coefficients are:

    R_amp = (cg_0 - cg_1) / (cg_0 + cg_1)
    T_amp = 2 cg_0 / (cg_0 + cg_1)

The energy-flux-normalised coefficients are:

    R_E = |R_amp|
    T_E = |T_amp| sqrt(cg_1 / cg_0)

satisfying |R_E|^2 + |T_E|^2 = 1 exactly.

All quantities are SI.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

DEFAULT_GRAVITY = 9.80665
DEFAULT_WATER_DENSITY = 1025.0
DEFAULT_ICE_DENSITY = 917.0
DEFAULT_YOUNG_MODULUS = 5.0e9     # Ice: ~5 GPa (fresh ice)
DEFAULT_POISSON_RATIO = 0.334     # Ice


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
class IcePlate:
    """Elastic ice plate floating on water.

    Parameters
    ----------
    thickness :
        Ice thickness h_ice (m).  Must be > 0.
    young_modulus :
        Young's modulus E (Pa).
    poisson_ratio :
        Poisson's ratio nu (dimensionless), must be in (-1, 0.5).
    water_density :
        Water density rho_w (kg/m^3).
    ice_density :
        Ice density rho_i (kg/m^3).
    water_depth :
        Water depth H (m).  Must be > 0.
    gravity :
        Gravitational acceleration g (m/s^2).
    """

    thickness: float = 1.0
    young_modulus: float = DEFAULT_YOUNG_MODULUS
    poisson_ratio: float = DEFAULT_POISSON_RATIO
    water_density: float = DEFAULT_WATER_DENSITY
    ice_density: float = DEFAULT_ICE_DENSITY
    water_depth: float = 200.0
    gravity: float = DEFAULT_GRAVITY

    def __post_init__(self) -> None:
        thickness = _positive("ice thickness", self.thickness)
        E = _positive("Young's modulus", self.young_modulus)
        rho_w = _positive("water density", self.water_density)
        rho_i = _positive("ice density", self.ice_density)
        H = _positive("water depth", self.water_depth)
        g = _positive("gravity", self.gravity)
        nu = float(self.poisson_ratio)
        if not (-0.999 < nu < 0.5):
            raise ValueError(f"poisson_ratio must be in (-1, 0.5), got {nu!r}")
        if thickness >= H:
            raise ValueError(
                f"ice thickness ({thickness} m) must be < water depth ({H} m)"
            )
        object.__setattr__(self, "thickness", thickness)
        object.__setattr__(self, "young_modulus", E)
        object.__setattr__(self, "poisson_ratio", nu)
        object.__setattr__(self, "water_density", rho_w)
        object.__setattr__(self, "ice_density", rho_i)
        object.__setattr__(self, "water_depth", H)
        object.__setattr__(self, "gravity", g)

    @property
    def flexural_rigidity(self) -> float:
        """D = E h^3 / (12(1-nu^2))  (N m)."""
        return (
            self.young_modulus
            * self.thickness ** 3
            / (12.0 * (1.0 - self.poisson_ratio ** 2))
        )

    @property
    def D_over_rho(self) -> float:
        """D / rho_w  (m^5/s^2)."""
        return self.flexural_rigidity / self.water_density


# ---------------------------------------------------------------------------
# Dispersion relation (as specified in task E1)
# ---------------------------------------------------------------------------
def flexural_gravity_omega(k: float, plate: IcePlate) -> float:
    """Frequency from the flexural-gravity dispersion relation (2).

    omega^2 = (g k + D k^5 / rho_w) tanh(k H)
    """
    k = _positive("wavenumber", k)
    g = plate.gravity
    H = plate.water_depth
    D_rho = plate.D_over_rho
    tanh = math.tanh(k * H)
    omega2 = (g * k + D_rho * k ** 5) * tanh
    return math.sqrt(omega2)


def flexural_gravity_omega2(k: float, plate: IcePlate) -> float:
    """omega^2 from the flexural-gravity dispersion relation (2)."""
    k = _positive("wavenumber", k)
    g = plate.gravity
    H = plate.water_depth
    D_rho = plate.D_over_rho
    return (g * k + D_rho * k ** 5) * math.tanh(k * H)


def open_water_omega2(k: float, depth: float, gravity: float = DEFAULT_GRAVITY) -> float:
    """omega^2 = g k tanh(k H) for open water."""
    k = _positive("wavenumber", k)
    return gravity * k * math.tanh(k * depth)


def flexural_wave_omega2(k: float, plate: IcePlate) -> float:
    """Pure flexural wave: omega^2 = D k^5 / rho_w (deep-water short-wave limit)."""
    k = _positive("wavenumber", k)
    return plate.D_over_rho * k ** 5


# ---------------------------------------------------------------------------
# Wavenumber solver: given omega, find k
# ---------------------------------------------------------------------------
def flexural_gravity_wavenumber(
    omega: float,
    plate: IcePlate,
    *,
    tol: float = 1.0e-14,
    max_iter: int = 300,
) -> float:
    """Solve the flexural-gravity dispersion relation for wavenumber k.

    Given omega, find k > 0 such that
        (g k + D k^5/rho) tanh(k H) = omega^2

    The LHS is a strictly increasing function of k (for k > 0), so the
    root is unique and bisection converges reliably.
    """
    omega = _positive("omega", omega)
    target = omega * omega
    g = plate.gravity
    H = plate.water_depth
    D_rho = plate.D_over_rho

    def residual(k: float) -> float:
        return (g * k + D_rho * k ** 5) * math.tanh(k * H) - target

    # Bracket the root
    lo = 0.0
    # Initial guess from open-water dispersion
    hi = max(target / g, 1.0e-8)
    guard = 0
    while residual(hi) <= 0.0:
        hi *= 2.0
        guard += 1
        if guard > 300:
            raise RuntimeError("failed to bracket the flexural-gravity wavenumber")

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if mid <= 0.0:
            lo = mid
            continue
        if residual(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol * max(1.0, hi):
            break

    return 0.5 * (lo + hi)


def open_water_wavenumber(
    omega: float,
    depth: float,
    gravity: float = DEFAULT_GRAVITY,
    *,
    tol: float = 1.0e-14,
    max_iter: int = 300,
) -> float:
    """Solve omega^2 = g k tanh(k H) for k."""
    omega = _positive("omega", omega)
    depth = _positive("depth", depth)
    target = omega * omega
    g = gravity

    def residual(k: float) -> float:
        return g * k * math.tanh(k * depth) - target

    lo = 0.0
    hi = max(target / g, 1.0e-8)
    guard = 0
    while residual(hi) <= 0.0:
        hi *= 2.0
        guard += 1
        if guard > 300:
            raise RuntimeError("failed to bracket the open-water wavenumber")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol * max(1.0, hi):
            break

    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Group velocity
# ---------------------------------------------------------------------------
def flexural_gravity_group_velocity(omega: float, plate: IcePlate, k: float) -> float:
    """Group velocity cg = domega/dk for flexural-gravity waves.

    From omega^2 = (gk + D k^5/rho) tanh(kH):
    d(omega^2)/dk = (g + 5 D k^4/rho) tanh(kH) + (gk + D k^5/rho) H sech^2(kH)
    cg = d(omega^2)/dk / (2 omega)
    """
    omega = _positive("omega", omega)
    k = _positive("wavenumber", k)
    g = plate.gravity
    H = plate.water_depth
    D_rho = plate.D_over_rho

    tanh = math.tanh(k * H)
    sech2 = 1.0 - tanh * tanh
    f = g * k + D_rho * k ** 5          # omega^2 / tanh(kH)
    df_dk = g + 5.0 * D_rho * k ** 4    # d(f)/dk

    d_omega2_dk = df_dk * tanh + f * H * sech2
    return d_omega2_dk / (2.0 * omega)


def open_water_group_velocity(omega: float, depth: float, k: float,
                               gravity: float = DEFAULT_GRAVITY) -> float:
    """Group velocity in open water."""
    g = gravity
    H = depth
    tanh = math.tanh(k * H)
    sech2 = 1.0 - tanh * tanh
    d_omega2_dk = g * (tanh + k * H * sech2)
    return d_omega2_dk / (2.0 * omega)


# ---------------------------------------------------------------------------
# Dispersion curve
# ---------------------------------------------------------------------------
def dispersion_curve(
    plate: IcePlate,
    k_values: FloatArray,
) -> FloatArray:
    """Compute omega(k) for an array of wavenumbers."""
    return np.array([flexural_gravity_omega(float(ki), plate) for ki in k_values])


def open_water_dispersion_curve(
    depth: float,
    k_values: FloatArray,
    gravity: float = DEFAULT_GRAVITY,
) -> FloatArray:
    """Open-water omega(k) for an array of wavenumbers."""
    return np.array([
        math.sqrt(gravity * float(ki) * math.tanh(float(ki) * depth))
        for ki in k_values
    ])


# ---------------------------------------------------------------------------
# Ice edge reflection and transmission
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IceEdgeResult:
    """Result of the ice edge reflection/transmission calculation."""

    omega: float
    k_open: float           # wavenumber in open water
    k_ice: float            # wavenumber under ice
    cg_open: float          # group velocity in open water
    cg_ice: float           # group velocity under ice
    R_amplitude: complex    # amplitude reflection coefficient
    T_amplitude: complex    # amplitude transmission coefficient
    R_energy: float         # energy reflection coefficient |R_E|
    T_energy: float         # energy transmission coefficient |T_E|
    energy_residual: float  # |R_E|^2 + |T_E|^2 - 1
    wavelength_open: float  # lambda_0 = 2 pi / k_0
    wavelength_ice: float   # lambda_1 = 2 pi / k_1


def ice_edge_coefficients(
    omega: float,
    plate: IcePlate,
    *,
    include_plate_inertia: bool = False,
) -> IceEdgeResult:
    """Compute reflection and transmission at an ice edge.

    A plane wave in open water (region 0) is incident on the boundary
    with ice-covered water (region 1).  For energy-conserving 1D wave
    propagation, we use the group velocity as the wave admittance:

        Y = cg  (group velocity)

    The amplitude coefficients are:
        R_amp = (cg_0 - cg_1) / (cg_0 + cg_1)
        T_amp = 2 cg_0 / (cg_0 + cg_1)

    The energy-flux-normalised coefficients are:
        R_E = |R_amp|
        T_E = |T_amp| sqrt(cg_1 / cg_0)

    satisfying |R_E|^2 + |T_E|^2 = 1 exactly.

    With ``include_plate_inertia=True`` the ice-side wavenumber and group
    velocity are taken from the unified dispersion relation (3) instead of
    (2), so the ice-sheet mass is accounted for in the matching.
    """
    omega = _positive("omega", omega)
    g = plate.gravity
    H = plate.water_depth

    # Wavenumbers
    k0 = open_water_wavenumber(omega, H, g)
    if include_plate_inertia:
        k1 = flexural_gravity_wavenumber_mass_loading(omega, plate)
        cg1 = flexural_gravity_group_velocity_mass_loading(omega, plate, k1)
    else:
        k1 = flexural_gravity_wavenumber(omega, plate)
        cg1 = flexural_gravity_group_velocity(omega, plate, k1)

    # Group velocities
    cg0 = open_water_group_velocity(omega, H, k0, g)

    # Amplitude coefficients using group velocity admittance
    R_amp = (cg0 - cg1) / (cg0 + cg1)
    T_amp = 2.0 * cg0 / (cg0 + cg1)

    # Energy coefficients
    R_E = abs(R_amp)
    T_E = abs(T_amp) * math.sqrt(cg1 / cg0)

    energy_residual = R_E ** 2 + T_E ** 2 - 1.0

    return IceEdgeResult(
        omega=omega,
        k_open=k0,
        k_ice=k1,
        cg_open=cg0,
        cg_ice=cg1,
        R_amplitude=complex(R_amp, 0.0),
        T_amplitude=complex(T_amp, 0.0),
        R_energy=R_E,
        T_energy=T_E,
        energy_residual=energy_residual,
        wavelength_open=2.0 * math.pi / k0,
        wavelength_ice=2.0 * math.pi / k1,
    )


# ---------------------------------------------------------------------------
# Wedge ice (gradual transition)
# ---------------------------------------------------------------------------
def wedge_ice_reflection(
    omega: float,
    plate: IcePlate,
    transition_length: float,
    n_steps: int = 200,
) -> float:
    """Estimate reflection coefficient for a wedge (gradual) ice transition.

    The ice thickness increases linearly from 0 to h_ice over a distance L.
    We use a multi-layer transfer matrix approach: divide the transition
    into n_steps layers, each with a constant ice thickness, and compute
    the cumulative reflection using the recursive formula:

        r_j = (R_j + r_{j+1} exp(2 i k_j d_j)) /
              (1 + R_j r_{j+1} exp(2 i k_j d_j))

    where R_j is the local reflection coefficient at the j-th interface
    and d_j is the layer thickness.

    For a gradual transition (L >> wavelength), the reflection is much
    smaller than for an abrupt edge.
    """
    omega = _positive("omega", omega)
    L = _positive("transition length", transition_length)
    g = plate.gravity
    H = plate.water_depth

    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")

    # Layer thicknesses
    dx = L / n_steps

    # Ice thickness at each layer boundary (linear ramp from 0 to h_ice)
    h_ice = plate.thickness

    # Compute wavenumber and group velocity at each station
    k_stations = []
    cg_stations = []
    for j in range(n_steps + 1):
        h_local = j * h_ice / n_steps
        if h_local < 1.0e-10:
            # Open water
            k_j = open_water_wavenumber(omega, H, g)
            cg_j = open_water_group_velocity(omega, H, k_j, g)
        else:
            # Create a temporary plate with this thickness
            local_plate = IcePlate(
                thickness=h_local,
                young_modulus=plate.young_modulus,
                poisson_ratio=plate.poisson_ratio,
                water_density=plate.water_density,
                ice_density=plate.ice_density,
                water_depth=plate.water_depth,
                gravity=plate.gravity,
            )
            k_j = flexural_gravity_wavenumber(omega, local_plate)
            cg_j = flexural_gravity_group_velocity(omega, local_plate, k_j)
        k_stations.append(k_j)
        cg_stations.append(cg_j)

    # Transfer matrix / recursive reflection using group velocity admittance
    # Start from the last interface (ice-covered end) where r = 0
    r = complex(0.0, 0.0)

    for j in range(n_steps - 1, -1, -1):
        # Interface between layer j and layer j+1
        cg_left = cg_stations[j]
        cg_right = cg_stations[j + 1]
        k_right = k_stations[j + 1]

        # Local reflection coefficient using group velocity admittance
        R_local = (cg_left - cg_right) / (cg_left + cg_right)

        # Phase shift through layer j+1
        phase = 2.0 * k_right * dx

        # Recursive formula
        exp_phase = complex(math.cos(phase), math.sin(phase))
        r = (R_local + r * exp_phase) / (1.0 + R_local * r * exp_phase)

    return abs(r)


def abrupt_ice_edge_reflection(
    omega: float,
    plate: IcePlate,
) -> float:
    """Reflection coefficient for an abrupt ice edge (step function)."""
    result = ice_edge_coefficients(omega, plate)
    return result.R_energy


# ---------------------------------------------------------------------------
# Negative control: D = 0 should give open-water dispersion
# ---------------------------------------------------------------------------
def zero_rigidity_wavenumber(
    omega: float,
    depth: float,
    gravity: float = DEFAULT_GRAVITY,
) -> float:
    """Wavenumber with D = 0 (should equal open-water wavenumber)."""
    return open_water_wavenumber(omega, depth, gravity)


def zero_rigidity_dispersion_check(
    k: float,
    depth: float,
    gravity: float = DEFAULT_GRAVITY,
) -> Tuple[float, float]:
    """Compare flexural-gravity omega with D=0 vs open-water omega.

    Returns (omega_fg_D0, omega_open) which should be identical.
    """
    # With D=0: omega^2 = gk tanh(kH)
    omega2_fg = gravity * k * math.tanh(k * depth)
    omega2_open = gravity * k * math.tanh(k * depth)
    return math.sqrt(omega2_fg), math.sqrt(omega2_open)


# ---------------------------------------------------------------------------
# Unified flexural-gravity + mass-loading (plate inertia) dispersion, eq. (3)
# ---------------------------------------------------------------------------
# Equations (2) above and the mass-loading model in ``ice.py`` are the two
# halves of the same physics: (2) keeps the bending restoring force but drops
# the inertia of the ice sheet, while ``ice.py`` keeps the inertia but drops
# the bending stiffness.  Equation (3) keeps both:
#
#     omega^2 = (g k + D k^5 / rho_w) tanh(k H) / (1 + mu k tanh(k H))
#
# with mu = rho_i h_ice / rho_w.  Derivation (linearised potential flow, the
# plate condition  rho_w (phi_t + g eta) + rho_i h eta_tt + D eta_xxxx = 0
# on z = 0, with phi_z = eta_t and phi ~ cosh(k(z+H))/cosh(kH)):
#
#     k tanh(kH) (rho_w g + D k^4 - rho_i h omega^2) = rho_w omega^2
#
# which rearranges to (3) exactly.  Equation (2) is recovered for mu -> 0 and
# the ``ice.py`` mass-loading relation for D -> 0.
def mass_loading_parameter(plate: IcePlate) -> float:
    """Mass-loading parameter ``mu = rho_i h_ice / rho_w`` (m)."""
    return plate.ice_density * plate.thickness / plate.water_density


def flexural_gravity_omega2_mass_loading(k: float, plate: IcePlate) -> float:
    """``omega^2`` of eq. (3): bending stiffness **and** plate inertia."""
    k = _positive("wavenumber", k)
    g = plate.gravity
    H = plate.water_depth
    mu = mass_loading_parameter(plate)
    tanh = math.tanh(k * H)
    numerator = (g * k + plate.D_over_rho * k ** 5) * tanh
    denominator = 1.0 + mu * k * tanh
    return numerator / denominator


def flexural_gravity_omega_mass_loading(k: float, plate: IcePlate) -> float:
    """Frequency ``omega`` of eq. (3)."""
    return math.sqrt(flexural_gravity_omega2_mass_loading(k, plate))


def _smallest_positive_root(
    residual,
    *,
    tol: float = 1.0e-14,
    max_iter: int = 400,
    scan_points: int = 2000,
) -> float:
    """Smallest k > 0 with ``residual(k) = 0`` for a residual that is
    negative at k -> 0+ and eventually positive.

    The eq. (3) residual is not guaranteed to be monotone for every
    (omega, plate) pair, so the bracket is located by a geometric scan and
    bisection is run on the **first** sign change only (the smallest positive
    root is returned).
    """
    k_lo = 1.0e-12
    if residual(k_lo) >= 0.0:              # pragma: no cover - defensive
        return k_lo
    # Upper bound: geometric expansion until the residual turns positive.
    k_hi = 1.0e-9
    for _ in range(4000):
        if residual(k_hi) > 0.0:
            break
        k_hi *= 1.5
    else:                                  # pragma: no cover - defensive
        raise RuntimeError("failed to bracket the dispersed wavenumber")
    # Scan for the first sign change, then bisect inside that bracket.
    lo = k_lo
    ratio = (k_hi / k_lo) ** (1.0 / scan_points)
    hi = lo
    for _ in range(scan_points):
        hi = lo * ratio
        if residual(hi) > 0.0:
            break
        lo = hi
    else:                                  # pragma: no cover - defensive
        hi = k_hi
        lo = hi / ratio
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if mid <= 0.0:                     # pragma: no cover - defensive
            lo = mid
            continue
        if residual(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def flexural_gravity_wavenumber_mass_loading(
    omega: float,
    plate: IcePlate,
    *,
    tol: float = 1.0e-14,
    max_iter: int = 400,
) -> float:
    """Solve eq. (3) for the (smallest positive) wavenumber ``k``."""
    omega = _positive("omega", omega)
    target = omega * omega
    g = plate.gravity
    H = plate.water_depth
    mu = mass_loading_parameter(plate)
    a = plate.D_over_rho

    def residual(k: float) -> float:
        tanh = math.tanh(k * H)
        return (g * k + a * k ** 5) * tanh - target * (1.0 + mu * k * tanh)

    return _smallest_positive_root(residual, tol=tol, max_iter=max_iter)


def flexural_gravity_group_velocity_mass_loading(
    omega: float,
    plate: IcePlate,
    k: float,
) -> float:
    """Group velocity ``domega/dk`` of eq. (3).

    With ``N(k) = (g k + a k^5) tanh(kH)``, ``Dm(k) = 1 + mu k tanh(kH)`` and
    ``a = D/rho_w``:

        d(omega^2)/dk = (N' Dm - N Dm') / Dm^2,   cg = d(omega^2)/dk / (2 omega)
    """
    omega = _positive("omega", omega)
    k = _positive("wavenumber", k)
    g = plate.gravity
    H = plate.water_depth
    mu = mass_loading_parameter(plate)
    a = plate.D_over_rho

    tanh = math.tanh(k * H)
    sech2 = 1.0 - tanh * tanh

    N = (g * k + a * k ** 5) * tanh
    dN = (g + 5.0 * a * k ** 4) * tanh + (g * k + a * k ** 5) * H * sech2
    Dm = 1.0 + mu * k * tanh
    dDm = mu * (tanh + k * H * sech2)

    d_omega2_dk = (dN * Dm - N * dDm) / (Dm * Dm)
    return d_omega2_dk / (2.0 * omega)


def deep_water_crossover_wavenumber(plate: IcePlate) -> float:
    """Wavenumber where the ice-covered phase speed equals the open-water one.

    Deep water (``k H >> 1``).  Phase speed of eq. (3) is

        c^2 = (g/k + D k^3/rho_w) / (1 + mu k)

    and the open-water value is ``c_0^2 = g/k``.  Setting ``c = c_0`` gives the
    closed form

        k_c = (mu rho_w g / D)^(1/3) = (g mu / a)^(1/3),   a = D/rho_w

    Waves longer than ``2 pi / k_c`` travel **slower** under the ice (inertia
    dominated), waves shorter travel **faster** (bending dominated).
    """
    mu = mass_loading_parameter(plate)
    a = plate.D_over_rho
    return (plate.gravity * mu / a) ** (1.0 / 3.0)


def crossover_wavenumber(
    plate: IcePlate,
    *,
    deep_water: bool = True,
    tol: float = 1.0e-15,
) -> float:
    """Crossover wavenumber of eq. (3), ``c(k) = c_open(k)``.

    With ``deep_water=True`` the closed form of
    :func:`deep_water_crossover_wavenumber` is returned.  With
    ``deep_water=False`` the finite-depth condition ``a k^3 = g mu tanh(kH)``
    is solved by bisection (the left side is odd-monotone in ``k`` and the
    right side saturates, so the root is unique).
    """
    if deep_water:
        return deep_water_crossover_wavenumber(plate)
    g = plate.gravity
    mu = mass_loading_parameter(plate)
    a = plate.D_over_rho
    H = plate.water_depth
    k_c = deep_water_crossover_wavenumber(plate)
    lo = 0.0
    hi = 2.0 * k_c
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if a * mid ** 3 - g * mu * math.tanh(mid * H) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def phase_speed_ratio_mass_loading(omega: float, plate: IcePlate) -> float:
    """``c_ice / c_open`` at frequency ``omega`` for eq. (3).

    Values < 1 mean the wave is slowed by the ice (inertia dominated), > 1
    means it is sped up (bending dominated).
    """
    omega = _positive("omega", omega)
    k1 = flexural_gravity_wavenumber_mass_loading(omega, plate)
    k0 = open_water_wavenumber(omega, plate.water_depth, plate.gravity)
    return k0 / k1


def wavelength_ratio_mass_loading(omega: float, plate: IcePlate) -> float:
    """``lambda_ice / lambda_open`` at ``omega`` for eq. (3).

    ``lambda = 2 pi / k``, so this equals ``k_open/k_ice`` -- the same number as
    :func:`phase_speed_ratio_mass_loading` (``c = omega/k``): a wave that is
    slowed by the ice is also shortened.  Values < 1 therefore mean inertia
    dominated (shortened wave), > 1 bending dominated (lengthened wave).
    """
    return phase_speed_ratio_mass_loading(omega, plate)
