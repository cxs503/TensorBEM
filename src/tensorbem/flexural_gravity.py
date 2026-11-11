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
    """
    omega = _positive("omega", omega)
    g = plate.gravity
    H = plate.water_depth

    # Wavenumbers
    k0 = open_water_wavenumber(omega, H, g)
    k1 = flexural_gravity_wavenumber(omega, plate)

    # Group velocities
    cg0 = open_water_group_velocity(omega, H, k0, g)
    cg1 = flexural_gravity_group_velocity(omega, plate, k1)

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
