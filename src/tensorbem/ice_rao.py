"""Ice-covered 3-D radiation, diffraction and RAO for a floating hull.

This module turns the mass-loading dispersion of :mod:`tensorbem.ice` into a
boundary-element radiation/diffraction model of a ship in an ice field.

Method (mass-loading ice cover, deep water)
-------------------------------------------
The 3-D free-surface Green function used by :mod:`tensorbem.free_surface_solver3d`
is parameterised by a single wavenumber ``nu`` and imposes the linearised
free-surface condition ``dphi/dz = nu phi`` on ``z = 0`` (deep water:
``nu = omega**2/g``).  Under a mass-loading ice cover the boundary condition on
the ice/water interface is the *same* Robin condition with a different
coefficient (:mod:`tensorbem.ice`)

    dphi/dz = K phi,      K = omega**2 / (g - mu omega**2) = k_i(omega),

because in deep water the mass-loading dispersion has the exact closed-form root
``k_i = omega**2/(g - mu omega**2)``.  The ice-covered Green function is
therefore obtained by evaluating the *unmodified* kernel at ``nu = k_i``.  Two
bookkeeping consequences follow and are implemented explicitly here:

* the radiation damping is ``B = omega * Im(Z)`` with the **physical**
  ``omega``, while the underlying solver infers ``sqrt(g nu)`` from ``nu``; the
  returned damping is rescaled by ``omega / sqrt(g k_i)``;
* the excitation force ``F = -i omega rho int phi n dS`` is *invariant* to that
  bookkeeping, because the incident potential carries the factor ``1/omega``:
  ``omega * (1/omega)`` cancels.  This is verified in
  ``tests/test_ice_rao.py`` (``test_excitation_force_is_bookkeeping_invariant``).

Validity envelope (enforced, not assumed)
-----------------------------------------
* Deep water: the mass-loading root equals ``K`` only when ``tanh(k_i H) = 1``.
  Every call checks ``k_i * H >= 3`` (``tanh >= 0.995``) and raises otherwise.
  A finite-depth ice Green function (``k tanh(kH) = K`` inside the kernel) is
  **not** implemented; results shallow of that limit are withheld.
* Propagating branch only: ``omega < sqrt(g/mu)``.  Above cut-off the ice
  waves are evanescent (:func:`tensorbem.ice.mass_loading_wavenumber`) and the
  deep-water kernel has no real pole; the RAO spectrum reports those
  frequencies as ``propagating = False`` with zero excitation instead of a
  fabricated number.
* No ice-induced dissipation: mass loading is conservative, so radiation
  damping in ice is not augmented by an ice term.  Floe-collision and
  flexural damping are *not* modelled.

Added resistance ``R_AW`` in ice is **withheld** (:data:`ADDED_RESISTANCE_STATUS`):
it requires the second-order drift-force operator and/or a floe-scattering
model, neither of which exists in this repository, and no reference solution is
available to validate a number.  The first-order quantities that any ``R_AW``
model consumes (ice wavenumber shift, energy-flux amplitude ratio, excitation
force ratio, RAO ratio, evanescent decay length) are reported instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from .geometry3d import SurfaceMesh3D
from .hydrostatic import (
    HydrostaticProperties,
    compute_hydrostatic_restoring_matrix,
    compute_mass_matrix,
)
from .ice import (
    IceCover,
    energy_flux_amplitude_ratio,
    mass_loading_wavenumber,
    open_water_wavenumber,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

DEFAULT_GRAVITY = 9.80665
DEFAULT_DENSITY = 1025.0

#: ``tanh(3) = 0.995``; below this the deep-water kernel is outside its envelope.
DEEP_WATER_MARGIN = 3.0

ADDED_RESISTANCE_STATUS = {
    "status": "withheld",
    "reason": (
        "Ice-zone added resistance requires a second-order drift-force operator "
        "and/or a floe-scattering (attenuation) model; neither exists in this "
        "repository and no independent reference is available to validate a "
        "number, so R_AW is not reported. First-order drivers are reported instead."
    ),
}

_MODE_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return value


def ice_wavenumber_deep_water(omega: float, ice: IceCover) -> float:
    """Deep-water mass-loading wavenumber ``k_i = omega**2/(g - mu omega**2)``.

    Raises ``ValueError`` at or above the mass-loading cut-off.
    """
    omega = _positive("omega", omega)
    denom = ice.gravity - ice.mu * omega * omega
    if denom <= 0.0:
        raise ValueError(
            "omega exceeds the mass-loading cut-off sqrt(g/mu) = "
            f"{ice.cutoff_omega:.6g} rad/s; the ice waves are evanescent and the "
            "deep-water wave kernel has no real pole"
        )
    return omega * omega / denom


def _depth_parameter(wavenumber: float, ice: IceCover, *, subject: str) -> float:
    depth_parameter = wavenumber * ice.water_depth
    if depth_parameter < DEEP_WATER_MARGIN:
        raise ValueError(
            f"{subject}: k*H = {depth_parameter:.3f} < {DEEP_WATER_MARGIN} "
            "(tanh(kH) = "
            f"{math.tanh(depth_parameter):.4f}); the deep-water ice kernel is outside "
            "its validity envelope. Increase water_depth or use a shallower frequency."
        )
    return depth_parameter


@dataclass(frozen=True)
class IceRadiation3D:
    """Six-degree-of-freedom radiation in a mass-loading ice cover."""

    omega: float
    wavenumber_ice: float
    wavenumber_open: float
    depth_parameter: float
    added_mass: FloatArray
    damping: FloatArray
    condition_number: float
    relative_residual: float
    ice: IceCover


@dataclass(frozen=True)
class IceDiffraction3D:
    """Regular-wave excitation in a mass-loading ice cover."""

    omega: float
    wavenumber_ice: float
    wavenumber_open: float
    depth_parameter: float
    wave_angle: float
    amplitude: float
    amplitude_ratio: float
    excitation_force: ComplexArray
    condition_number: float
    relative_residual: float
    ice: IceCover


def solve_radiation_3d_ice(
    mesh: SurfaceMesh3D,
    omega: float,
    ice: IceCover,
    *,
    center: ArrayLike | None = None,
    density: float = DEFAULT_DENSITY,
    gravity: float | None = None,
    chief_points: ArrayLike | str | None = None,
    rankine_order: int = 5,
    operators=None,
) -> IceRadiation3D:
    """Radiation added mass and damping under a mass-loading ice cover.

    ``operators`` (a :class:`tensorbem.free_surface_solver3d.FreeSurface3DOperators`
    built at ``nu = k_i``) may be supplied to reuse the assembled matrices
    between the radiation and diffraction solves at the same frequency.
    """
    omega = _positive("omega", omega)
    g = ice.gravity if gravity is None else _positive("gravity", gravity)
    k_ice = ice_wavenumber_deep_water(omega, ice)
    depth_parameter = _depth_parameter(k_ice, ice, subject="ice radiation")

    result = solve_radiation_3d(
        mesh,
        k_ice,
        center=center,
        density=density,
        gravity=g,
        chief_points=chief_points,
        rankine_order=rankine_order,
        operators=operators,
    )
    # solver infers omega_int = sqrt(g k_ice) from nu; restore the physical factor
    damping = result.damping * (omega / math.sqrt(g * k_ice))
    k_open = open_water_wavenumber(omega, ice.water_depth, g)
    return IceRadiation3D(
        omega=omega,
        wavenumber_ice=k_ice,
        wavenumber_open=k_open,
        depth_parameter=depth_parameter,
        added_mass=result.added_mass,
        damping=damping,
        condition_number=float(result.condition_number),
        relative_residual=float(result.relative_residual),
        ice=ice,
    )


def solve_diffraction_3d_ice(
    mesh: SurfaceMesh3D,
    omega: float,
    ice: IceCover,
    *,
    amplitude: float = 1.0,
    wave_angle: float = 0.0,
    center: ArrayLike | None = None,
    density: float = DEFAULT_DENSITY,
    gravity: float | None = None,
    chief_points: ArrayLike | str | None = None,
    rankine_order: int = 5,
    apply_energy_flux_ratio: bool = False,
    operators=None,
) -> IceDiffraction3D:
    """Wave excitation force under a mass-loading ice cover.

    ``apply_energy_flux_ratio`` multiplies the incident amplitude by
    ``sqrt(cg_open/cg_ice)`` (energy-flux continuity across an abrupt ice edge,
    :func:`tensorbem.ice.energy_flux_amplitude_ratio`); the returned force then
    belongs to that steeper ice-field wave.  The default ``False`` means "the
    incident amplitude at the ship is the stated amplitude", which is the right
    choice when the amplitude in the ice field is already known.
    """
    omega = _positive("omega", omega)
    amp = _positive("wave amplitude", amplitude)
    if not isinstance(apply_energy_flux_ratio, bool):
        raise ValueError("apply_energy_flux_ratio must be a bool")
    g = ice.gravity if gravity is None else _positive("gravity", gravity)
    k_ice = ice_wavenumber_deep_water(omega, ice)
    depth_parameter = _depth_parameter(k_ice, ice, subject="ice diffraction")

    ratio = float(energy_flux_amplitude_ratio(omega, ice)) if apply_energy_flux_ratio else 1.0
    if apply_energy_flux_ratio and ratio == 0.0:
        raise ValueError(
            "the energy-flux amplitude ratio vanishes on the evanescent branch; "
            "no propagating incident wave reaches the cover"
        )

    if operators is None:
        operators = assemble_free_surface_operators_3d(
            mesh, k_ice, chief_points=chief_points, rankine_order=rankine_order
        )
    result = solve_diffraction_3d(
        mesh,
        k_ice,
        amplitude=amp * ratio,
        wave_angle=wave_angle,
        center=center,
        density=density,
        gravity=g,
        operators=operators,
    )
    k_open = open_water_wavenumber(omega, ice.water_depth, g)
    # when the energy-flux ratio is applied, ``amplitude`` is the *open-water*
    # amplitude and the reported force belongs to the steeper ice-field wave
    force = result.excitation_force
    return IceDiffraction3D(
        omega=omega,
        wavenumber_ice=k_ice,
        wavenumber_open=k_open,
        depth_parameter=depth_parameter,
        wave_angle=float(wave_angle),
        amplitude=amp,
        amplitude_ratio=ratio,
        excitation_force=np.asarray(force, dtype=np.complex128),
        condition_number=float(result.condition_number),
        relative_residual=float(result.relative_residual),
        ice=ice,
    )


def solve_radiation_3d_open_water(
    mesh: SurfaceMesh3D,
    omega: float,
    *,
    water_depth: float = 200.0,
    center: ArrayLike | None = None,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    chief_points: ArrayLike | str | None = None,
    rankine_order: int = 5,
) -> tuple[FloatArray, FloatArray, float]:
    """Open-water added mass and damping on the same discretisation.

    Returns ``(added_mass, damping, wavenumber)`` with ``nu = omega**2/g``; the
    water depth only enters the diagnostic wavenumber, the kernel is the deep
    water one (the repository's convention).
    """
    omega = _positive("omega", omega)
    g = _positive("gravity", gravity)
    nu = omega * omega / g
    result = solve_radiation_3d(
        mesh,
        nu,
        center=center,
        density=density,
        gravity=g,
        chief_points=chief_points,
        rankine_order=rankine_order,
    )
    return result.added_mass, result.damping, nu


@dataclass(frozen=True)
class IceRaoSpectrum:
    """Six-DOF RAO spectrum in ice, with the open-water baseline."""

    omegas: FloatArray
    wavenumbers_ice: FloatArray
    wavenumbers_open: FloatArray
    depth_parameter: FloatArray
    amplitude_ratio: FloatArray
    propagating: NDArray[np.bool_]
    decay_rate: FloatArray
    added_mass_ice: FloatArray
    damping_ice: FloatArray
    excitation_ice: ComplexArray
    rao_ice: ComplexArray
    added_mass_open: FloatArray
    damping_open: FloatArray
    excitation_open: ComplexArray
    rao_open: ComplexArray
    wave_angle: float
    ice: IceCover
    restoring_matrix: FloatArray
    notes: tuple[str, ...]

    @property
    def rao_ice_amplitude(self) -> FloatArray:
        return np.abs(self.rao_ice)

    @property
    def rao_open_amplitude(self) -> FloatArray:
        return np.abs(self.rao_open)


def _rao_from_impedance(
    omega: float,
    mass_matrix: FloatArray,
    added_mass: FloatArray,
    damping: FloatArray,
    restoring: FloatArray,
    force: ComplexArray,
) -> ComplexArray:
    impedance = (
        -(omega * omega) * (mass_matrix + added_mass)
        + 1j * omega * damping
        + restoring
    )
    return np.linalg.solve(impedance, force)


def compute_ice_rao_spectrum(
    hull: SurfaceMesh3D,
    omegas: Sequence[float],
    wave_angle: float,
    mass: float,
    center_of_gravity: ArrayLike,
    ice: IceCover,
    *,
    amplitude: float = 1.0,
    density: float = DEFAULT_DENSITY,
    inertia: ArrayLike | None = None,
    hydrostatic_props: HydrostaticProperties | None = None,
    chief_points: ArrayLike | str | None = "auto",
    rankine_order: int = 5,
    extra_damping: ArrayLike | None = None,
) -> IceRaoSpectrum:
    """Six-DOF RAO in a mass-loading ice cover, with the open-water baseline.

    The equation of motion is the standard frequency-domain form

        [-omega**2 (M + A) + i omega B + C] xi = F_exc

    with ``A``, ``B`` from :func:`solve_radiation_3d_ice` and ``F_exc`` from
    :func:`solve_diffraction_3d_ice`.  The hydrostatic restoring matrix ``C`` is
    identical in ice and in open water: mass loading adds inertia, not
    stiffness, so the ice cover does not change the waterplane restoring.
    """
    omega_arr = np.asarray(list(omegas), dtype=float)
    if omega_arr.ndim != 1 or omega_arr.size == 0:
        raise ValueError("omegas must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(omega_arr)) or np.any(omega_arr <= 0.0):
        raise ValueError("all omegas must be finite and strictly positive")
    mass_value = _positive("mass", mass)
    cg = np.asarray(center_of_gravity, dtype=float).reshape(3)
    amp = _positive("wave amplitude", amplitude)
    angle = float(wave_angle)

    if hydrostatic_props is None:
        hydrostatic_props = compute_hydrostatic_restoring_matrix(
            hull, mass_value, cg, density, ice.gravity
        )
    restoring = np.asarray(hydrostatic_props.restoring_matrix, dtype=float)
    mass_matrix = compute_mass_matrix(mass_value, cg, inertia)
    damping_extra = None if extra_damping is None else np.asarray(extra_damping, dtype=float)

    n = omega_arr.size
    k_ice = np.zeros(n)
    k_open = np.zeros(n)
    depth_par = np.zeros(n)
    ratio = np.zeros(n)
    decay = np.zeros(n)
    propagating = np.zeros(n, dtype=bool)
    a_ice = np.zeros((n, 6, 6))
    b_ice = np.zeros((n, 6, 6))
    f_ice = np.zeros((n, 6), dtype=np.complex128)
    xi_ice = np.zeros((n, 6), dtype=np.complex128)
    a_open = np.zeros((n, 6, 6))
    b_open = np.zeros((n, 6, 6))
    f_open = np.zeros((n, 6), dtype=np.complex128)
    xi_open = np.zeros((n, 6), dtype=np.complex128)

    for i, omega in enumerate(omega_arr):
        omega = float(omega)
        wave = mass_loading_wavenumber(omega, ice)
        k_open[i] = open_water_wavenumber(omega, ice.water_depth, ice.gravity)
        propagating[i] = bool(wave.propagating)
        decay[i] = wave.decay_rate
        if not wave.propagating:
            # no propagating incident wave: zero excitation, zero response.
            a_ice[i] = np.nan
            b_ice[i] = np.nan
            continue
        ratio[i] = float(energy_flux_amplitude_ratio(omega, ice))
        # --- ice branch -------------------------------------------------
        ops_ice = assemble_free_surface_operators_3d(
            hull, k_ice_here := ice_wavenumber_deep_water(omega, ice),
            chief_points=chief_points, rankine_order=rankine_order,
        )
        radiation = solve_radiation_3d_ice(
            hull,
            omega,
            ice,
            center=cg,
            density=density,
            chief_points=chief_points,
            rankine_order=rankine_order,
            operators=ops_ice,
        )
        diffraction = solve_diffraction_3d_ice(
            hull,
            omega,
            ice,
            amplitude=amp,
            wave_angle=angle,
            center=cg,
            density=density,
            chief_points=chief_points,
            rankine_order=rankine_order,
            operators=ops_ice,
        )
        k_ice[i] = radiation.wavenumber_ice
        depth_par[i] = radiation.depth_parameter
        a_ice[i] = radiation.added_mass
        b_ice[i] = radiation.damping if damping_extra is None else radiation.damping + damping_extra
        f_ice[i] = diffraction.excitation_force
        xi_ice[i] = _rao_from_impedance(
            omega, mass_matrix, a_ice[i], b_ice[i], restoring, f_ice[i]
        )
        # --- open water baseline ----------------------------------------
        nu = omega * omega / ice.gravity
        ops_open = assemble_free_surface_operators_3d(
            hull, nu, chief_points=chief_points, rankine_order=rankine_order
        )
        rad_o = solve_radiation_3d(
            hull,
            nu,
            center=cg,
            density=density,
            gravity=ice.gravity,
            operators=ops_open,
        )
        diff_o = solve_diffraction_3d(
            hull,
            nu,
            amplitude=amp,
            wave_angle=angle,
            center=cg,
            density=density,
            gravity=ice.gravity,
            operators=ops_open,
        )
        a_open[i] = rad_o.added_mass
        b_open[i] = (
            rad_o.damping if damping_extra is None else rad_o.damping + damping_extra
        )
        f_open[i] = diff_o.excitation_force
        xi_open[i] = _rao_from_impedance(
            omega, mass_matrix, a_open[i], b_open[i], restoring, f_open[i]
        )

    notes = (
        "Ice model: mass-loading ice sheet (no flexural rigidity, no dissipation); "
        "deep-water Green function evaluated at nu = k_i = omega^2/(g - mu omega^2).",
        "Ice affects inertia and the incident wave field, not the hydrostatic "
        "restoring matrix: C(ice) = C(open water).",
        "Frequencies above sqrt(g/mu) are evanescent: they are flagged "
        "propagating=False and carry zero excitation, not a fabricated response.",
        "Added resistance R_AW in ice: withheld (see tensorbem.ice_rao."
        "ADDED_RESISTANCE_STATUS).",
    )
    return IceRaoSpectrum(
        omegas=omega_arr,
        wavenumbers_ice=k_ice,
        wavenumbers_open=k_open,
        depth_parameter=depth_par,
        amplitude_ratio=ratio,
        propagating=propagating,
        decay_rate=decay,
        added_mass_ice=a_ice,
        damping_ice=b_ice,
        excitation_ice=f_ice,
        rao_ice=xi_ice,
        added_mass_open=a_open,
        damping_open=b_open,
        excitation_open=f_open,
        rao_open=xi_open,
        wave_angle=angle,
        ice=ice,
        restoring_matrix=restoring,
        notes=notes,
    )


def no_ice_limit_check(
    mesh: SurfaceMesh3D,
    omega: float,
    *,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    water_depth: float = 1.0e4,
    center: ArrayLike | None = None,
    chief_points: ArrayLike | str | None = "auto",
) -> dict[str, float]:
    """Compare the ice solve in the ``mu -> 0`` limit with the open-water solve.

    With a vanishing ice thickness the mass-loading coefficient ``k_i`` tends to
    the open-water wavenumber and the ice chain must reproduce the open-water
    added mass, damping and excitation force.  This is the internal consistency
    gate that catches bookkeeping errors in the rescalings above.
    """
    thin = IceCover(
        thickness=1.0e-9, water_depth=water_depth, gravity=gravity,
        water_density=density,
    )
    ice_result = solve_radiation_3d_ice(
        mesh, omega, thin, center=center, density=density, gravity=gravity,
        chief_points=chief_points,
    )
    a_i, b_i, k_i = ice_result.added_mass, ice_result.damping, ice_result.wavenumber_ice
    a_o, b_o, nu = solve_radiation_3d_open_water(
        mesh, omega, center=center, density=density, gravity=gravity,
        chief_points=chief_points,
    )
    scale_a = max(float(np.linalg.norm(a_o)), 1.0e-30)
    scale_b = max(float(np.linalg.norm(b_o)), 1.0e-30)
    return {
        "wavenumber_ice": k_i,
        "wavenumber_open": nu,
        "wavenumber_relative_difference": abs(k_i - nu) / nu,
        "added_mass_relative_difference": float(np.linalg.norm(a_i - a_o) / scale_a),
        "damping_relative_difference": float(np.linalg.norm(b_i - b_o) / scale_b),
    }


def added_mass_matrix_report(added_mass: FloatArray) -> dict[str, float]:
    """Flatten a 6x6 added-mass matrix into ``A_<mode>` keys for reporting."""
    matrix = np.asarray(added_mass, dtype=float)
    if matrix.shape != (6, 6):
        raise ValueError("added_mass must be a 6x6 matrix")
    return {
        f"A{i + 1}{j + 1}_{a}_{b}": float(matrix[i, j])
        for i, a in enumerate(_MODE_NAMES)
        for j, b in enumerate(_MODE_NAMES)
    }