"""Frequency-domain RAO (Response Amplitude Operator) solver.

The RAO describes the linear response of a floating body to regular waves.
The equation of motion in frequency domain is:

    [-ω²(M + A(ω)) + iω B(ω) + C] ξ(ω) = F_exc(ω)

where:
    M     = rigid-body mass matrix (6×6)
    A(ω)  = frequency-dependent added mass matrix (6×6)
    B(ω)  = frequency-dependent damping matrix (6×6)
    C     = hydrostatic restoring matrix (6×6)
    F_exc = wave excitation force vector (6×1)
    ξ(ω)  = complex RAO vector (6×1)

The transfer function H(ω) is:
    H(ω) = [-ω²(M + A(ω)) + iω B(ω) + C]^(-1)

And the RAO is:
    ξ(ω) = H(ω) F_exc(ω)

For each degree of freedom, |ξ_i(ω)| gives the amplitude of motion per unit
wave amplitude at frequency ω.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .free_surface_solver3d import (
    Diffraction3DResult,
    Radiation3DResult,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from .geometry3d import WettedHullMesh
from .hydrostatic import (
    HydrostaticProperties,
    compute_hydrostatic_restoring_matrix,
    compute_mass_matrix,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_DEFAULT_GRAVITY = 9.80665
_DEFAULT_DENSITY = 1025.0

# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------
#
# The impedance algebra below is silently *garbage-in / garbage-out*: a negative
# body mass, a negative wave frequency (``_omega_to_nu`` squares ``omega``, so
# the sign vanishes), a NaN centre of gravity or an asymmetric inertia tensor
# all produce a finite, plausible-looking RAO vector.  The checks here follow
# the convention used by the rest of the package (``tensorbem.ice``,
# ``tensorbem.time_domain``, ``tensorbem.drift_force``): a non-physical input
# raises ``ValueError`` instead of propagating into the hydrodynamics.


def _positive(name: str, value: object) -> float:
    """Return ``value`` as a finite positive float or raise ``ValueError``."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number, got {value!r}") from None
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {number!r}")
    return number


def _finite_scalar(name: str, value: object) -> float:
    """Return ``value`` as a finite float (any sign) or raise ``ValueError``."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {number!r}")
    return number


def _real_vector(name: str, values: object, size: int) -> FloatArray:
    """Return ``values`` as a finite length-``size`` float vector."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a sequence of {size} real numbers") from None
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite, got {array!r}")
    return array


def _symmetric_matrix(
    name: str,
    values: object,
    size: int,
    *,
    non_negative_diagonal: bool = False,
) -> FloatArray:
    """Return ``values`` as a finite symmetric ``size x size`` float matrix.

    ``non_negative_diagonal`` is used for the extra damping matrix: a negative
    diagonal entry removes energy from the system (a radiation *source*), which
    makes the impedance matrix non-passive and the RAO meaningless.
    """
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a {size}x{size} real matrix") from None
    if array.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite, got a non-finite entry")
    scale = max(1.0, float(np.max(np.abs(array))))
    asymmetry = float(np.max(np.abs(array - array.T)))
    if asymmetry > 1.0e-8 * scale:
        raise ValueError(
            f"{name} must be symmetric; max asymmetry {asymmetry:.3e} exceeds "
            f"1e-8 * {scale:.3e}"
        )
    symmetric = 0.5 * (array + array.T)
    if non_negative_diagonal:
        diagonal = np.diag(symmetric)
        if np.any(diagonal < 0.0):
            worst = float(np.min(diagonal))
            raise ValueError(
                f"{name} diagonal must be non-negative: a negative damping "
                f"coefficient ({worst:.3e}) is an energy source, not a damper"
            )
    return symmetric


def _omega_sweep(omegas: object) -> FloatArray:
    """Return ``omegas`` as a finite, strictly positive 1-D array."""
    try:
        array = np.asarray(omegas, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("omegas must be a sequence of real frequencies") from None
    if array.ndim != 1:
        raise ValueError(f"omegas must be one-dimensional, got shape {array.shape}")
    if array.size and not np.all(np.isfinite(array)):
        raise ValueError("omegas must be finite")
    if array.size and np.any(array <= 0.0):
        worst = float(np.min(array))
        raise ValueError(
            f"omegas must be strictly positive: omega**2 enters through the "
            f"dispersion relation, so a negative frequency is indistinguishable "
            f"from its magnitude (got {worst!r})"
        )
    return array


def _validated_common(
    *,
    mass: object,
    density: object,
    gravity: object,
    amplitude: object,
    center_of_gravity: object,
    inertia: object,
    extra_damping: object,
) -> tuple[float, float, float, float, FloatArray, FloatArray | None, FloatArray | None]:
    """Validate the inputs shared by the single-frequency and sweep entry points."""
    mass_value = _positive("mass", mass)
    density_value = _positive("density", density)
    gravity_value = _positive("gravity", gravity)
    amplitude_value = _positive("amplitude", amplitude)
    cg = _real_vector("center_of_gravity", center_of_gravity, 3)
    inertia_value = (
        None if inertia is None else _symmetric_matrix("inertia", inertia, 3)
    )
    extra = (
        None
        if extra_damping is None
        else _symmetric_matrix(
            "extra_damping", extra_damping, 6, non_negative_diagonal=True
        )
    )
    return mass_value, density_value, gravity_value, amplitude_value, cg, inertia_value, extra



@dataclass(frozen=True)
class RAOResult:
    """Result of RAO computation at a single frequency.

    Attributes
    ----------
    omega :
        Wave circular frequency (rad/s).
    nu :
        Wave number (rad/m).
    rao :
        Complex RAO vector (6,) for surge, sway, heave, roll, pitch, yaw.
        Units: m/m for translation, rad/m for rotation.
    rao_amplitude :
        Absolute value of RAO (|ξ|).
    excitation_force :
        Complex excitation force vector (6,).
    added_mass :
        Added mass matrix A(ω) at this frequency (6×6).
    damping :
        Damping matrix B(ω) at this frequency (6×6).
    impedance_residual :
        Residual of the impedance equation (should be ~0 if converged).
    """

    omega: float
    nu: float
    rao: ComplexArray
    rao_amplitude: FloatArray
    excitation_force: ComplexArray
    added_mass: FloatArray
    damping: FloatArray
    impedance_residual: float


@dataclass(frozen=True)
class RAOSpectrum:
    """RAO spectrum over a range of frequencies.

    Attributes
    ----------
    omegas :
        Array of wave circular frequencies (rad/s).
    nus :
        Array of wave numbers (rad/m).
    rao_amplitudes :
        Array of RAO amplitudes, shape (n_freq, 6).
    excitation_forces :
        Array of excitation forces, shape (n_freq, 6).
    added_masses :
        Array of added mass matrices, shape (n_freq, 6, 6).
    dampings :
        Array of damping matrices, shape (n_freq, 6, 6).
    """

    omegas: FloatArray
    nus: FloatArray
    rao_amplitudes: FloatArray
    excitation_forces: ComplexArray
    added_masses: FloatArray
    dampings: FloatArray


def _omega_to_nu(omega: float, gravity: float) -> float:
    """Convert angular frequency to wavenumber (deep water dispersion)."""
    return omega * omega / gravity


def compute_rao_at_frequency(
    hull: WettedHullMesh,
    omega: float,
    wave_angle: float,
    mass: float,
    center_of_gravity: FloatArray,
    hydrostatic_props: HydrostaticProperties | None = None,
    amplitude: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    inertia: FloatArray | None = None,
    extra_damping: FloatArray | None = None,
    chief_points: int | str | None = "auto",
) -> RAOResult:
    """Compute RAO at a single frequency.

    Parameters
    ----------
    hull :
        Wetted hull mesh.
    omega :
        Wave circular frequency (rad/s). Must be strictly positive and finite:
        only ``omega**2`` enters the dispersion relation, so a negative value
        would silently be solved as ``|omega|``.
    wave_angle :
        Wave direction (radians). 0 = head seas (propagating in +x direction).
    mass :
        Body mass (kg). Must be strictly positive and finite.
    center_of_gravity :
        Center of gravity coordinates (m), shape (3,), finite.
    hydrostatic_props :
        Pre-computed hydrostatic properties. If None, computed from hull.
    amplitude :
        Wave amplitude (m). Default: 1.0.
    density :
        Fluid density (kg/m³).
    gravity :
        Gravitational acceleration (m/s²).
    inertia :
        Inertia tensor about the origin (3×3), finite and symmetric. If None,
        approximated.
    extra_damping :
        Additional damping matrix (6×6), finite, symmetric, non-negative
        diagonal (e.g. viscous damping).
    chief_points :
        Number of CHIEF points for irregular frequency suppression.

    Returns
    -------
    result :
        RAO result at this frequency.

    Raises
    ------
    ValueError
        If any input is non-physical: non-positive or non-finite ``mass``,
        ``omega``, ``density``, ``gravity`` or ``amplitude``; malformed
        ``center_of_gravity``; a non-symmetric / non-finite ``inertia``; a
        non-symmetric ``extra_damping`` or one with a negative diagonal entry.

    Notes
    -----
    Froude-Krylov-only bodies are *not* checked for Archimedes equilibrium here
    (``mass`` versus ``rho`` times the displaced volume): the historical test
    cases and several benchmark drivers deliberately use a mass that differs
    from the displaced mass, and the restoring matrix is built from the mesh
    alone.  The mismatch shows up physically as a long-wave heave RAO that does
    not tend to 1 (see ``tests/test_hydrostatic_analytic_benchmark.py``).
    """
    omega_value = _positive("omega", omega)
    (
        mass_value,
        density_value,
        gravity_value,
        amplitude_value,
        cg,
        inertia_value,
        extra,
    ) = _validated_common(
        mass=mass,
        density=density,
        gravity=gravity,
        amplitude=amplitude,
        center_of_gravity=center_of_gravity,
        inertia=inertia,
        extra_damping=extra_damping,
    )
    return _assemble_rao_at_frequency(
        hull,
        omega_value,
        _finite_scalar("wave_angle", wave_angle),
        mass_value,
        cg,
        hydrostatic_props,
        amplitude_value,
        density_value,
        gravity_value,
        inertia_value,
        extra,
        chief_points,
    )


def _assemble_rao_at_frequency(
    hull: WettedHullMesh,
    omega: float,
    wave_angle: float,
    mass: float,
    center_of_gravity: FloatArray,
    hydrostatic_props: HydrostaticProperties | None = None,
    amplitude: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    inertia: FloatArray | None = None,
    extra_damping: FloatArray | None = None,
    chief_points: int | str | None = "auto",
) -> RAOResult:
    """Compute RAO at a single frequency.

    Parameters
    ----------
    hull :
        Wetted hull mesh.
    omega :
        Wave circular frequency (rad/s).
    wave_angle :
        Wave direction (radians). 0 = head seas (propagating in +x direction).
    mass :
        Body mass (kg).
    center_of_gravity :
        Center of gravity coordinates (m).
    hydrostatic_props :
        Pre-computed hydrostatic properties. If None, computed from hull.
    amplitude :
        Wave amplitude (m). Default: 1.0.
    density :
        Fluid density (kg/m³).
    gravity :
        Gravitational acceleration (m/s²).
    inertia :
        Inertia tensor about the origin (3×3). If None, approximated.
    extra_damping :
        Additional damping matrix (6×6) to add (e.g., viscous damping).
    chief_points :
        Number of CHIEF points for irregular frequency suppression.

    Returns
    -------
    result :
        RAO result at this frequency.

    Notes
    -----
    **No input validation.**  This is the raw impedance assembly; it is kept
    separate so that ``benchmarks/rao_input_validation.py`` can demonstrate what
    a non-physical input used to do (finite, plausible, wrong output) against the
    public entry point that now rejects it.  Call
    :func:`compute_rao_at_frequency` instead.
    """
    omega = float(omega)
    nu = _omega_to_nu(omega, gravity)

    # Compute hydrostatic properties if not provided
    if hydrostatic_props is None:
        hydrostatic_props = compute_hydrostatic_restoring_matrix(
            hull, mass, center_of_gravity, density, gravity
        )

    # Get restoring matrix
    c = hydrostatic_props.restoring_matrix

    # Compute mass matrix
    m = compute_mass_matrix(mass, center_of_gravity, inertia)

    # Solve radiation problem to get added mass and damping
    rad_result: Radiation3DResult = solve_radiation_3d(
        hull, nu, density=density, gravity=gravity, chief_points=chief_points
    )

    # Solve diffraction problem to get excitation force
    diff_result: Diffraction3DResult = solve_diffraction_3d(
        hull, nu, amplitude=amplitude, wave_angle=wave_angle,
        density=density, gravity=gravity, chief_points=chief_points
    )

    # Extract added mass and damping matrices
    added_mass = rad_result.added_mass
    damping = rad_result.damping

    # Add extra damping if provided
    if extra_damping is not None:
        damping = damping + np.asarray(extra_damping)

    # Build impedance matrix
    # Z(ω) = -ω²(M + A(ω)) + iω B(ω) + C
    omega_sq = omega * omega
    z = -omega_sq * (m + added_mass) + 1j * omega * damping + c

    # Excitation force
    f_exc = diff_result.excitation_force

    # Solve for RAO
    # Z ξ = F_exc  =>  ξ = Z^(-1) F_exc
    rao = np.linalg.solve(z, f_exc)

    # Compute residual
    residual_vec = z @ rao - f_exc
    impedance_residual = float(np.linalg.norm(residual_vec))

    return RAOResult(
        omega=omega,
        nu=nu,
        rao=rao,
        rao_amplitude=np.abs(rao),
        excitation_force=f_exc,
        added_mass=added_mass,
        damping=damping,
        impedance_residual=impedance_residual,
    )


def compute_rao_spectrum(
    hull: WettedHullMesh,
    omegas: Sequence[float],
    wave_angle: float,
    mass: float,
    center_of_gravity: FloatArray,
    hydrostatic_props: HydrostaticProperties | None = None,
    amplitude: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    inertia: FloatArray | None = None,
    extra_damping: FloatArray | None = None,
    chief_points: int | str | None = "auto",
) -> RAOSpectrum:
    """Compute RAO spectrum over a range of frequencies.

    Parameters
    ----------
    hull :
        Wetted hull mesh.
    omegas :
        Sequence of wave circular frequencies (rad/s).
    wave_angle :
        Wave direction (radians).
    mass :
        Body mass (kg).
    center_of_gravity :
        Center of gravity coordinates (m).
    hydrostatic_props :
        Pre-computed hydrostatic properties. If None, computed from hull.
    amplitude :
        Wave amplitude (m).
    density :
        Fluid density (kg/m³).
    gravity :
        Gravitational acceleration (m/s²).
    inertia :
        Inertia tensor about the origin (3×3).
    extra_damping :
        Additional damping matrix (6×6).
    chief_points :
        Number of CHIEF points.

    Returns
    -------
    spectrum :
        RAO spectrum over the frequency range.
    """
    omegas = _omega_sweep(omegas)
    (
        mass_value,
        density_value,
        gravity_value,
        amplitude_value,
        cg,
        inertia_value,
        extra,
    ) = _validated_common(
        mass=mass,
        density=density,
        gravity=gravity,
        amplitude=amplitude,
        center_of_gravity=center_of_gravity,
        inertia=inertia,
        extra_damping=extra_damping,
    )
    mass = mass_value
    density = density_value
    gravity = gravity_value
    amplitude = amplitude_value
    center_of_gravity = cg
    inertia = inertia_value
    extra_damping = extra
    wave_angle = _finite_scalar("wave_angle", wave_angle)
    n_freq = len(omegas)
    nus = np.array([_omega_to_nu(w, gravity) for w in omegas])

    rao_amplitudes = np.zeros((n_freq, 6), dtype=float)
    excitation_forces = np.zeros((n_freq, 6), dtype=complex)
    added_masses = np.zeros((n_freq, 6, 6), dtype=float)
    dampings = np.zeros((n_freq, 6, 6), dtype=float)

    # Compute hydrostatic properties once
    if hydrostatic_props is None:
        hydrostatic_props = compute_hydrostatic_restoring_matrix(
            hull, mass, center_of_gravity, density, gravity
        )

    for i, omega in enumerate(omegas):
        rao_result = compute_rao_at_frequency(
            hull=hull,
            omega=omega,
            wave_angle=wave_angle,
            mass=mass,
            center_of_gravity=center_of_gravity,
            hydrostatic_props=hydrostatic_props,
            amplitude=amplitude,
            density=density,
            gravity=gravity,
            inertia=inertia,
            extra_damping=extra_damping,
            chief_points=chief_points,
        )

        rao_amplitudes[i] = rao_result.rao_amplitude
        excitation_forces[i] = rao_result.excitation_force
        added_masses[i] = rao_result.added_mass
        dampings[i] = rao_result.damping

    return RAOSpectrum(
        omegas=omegas,
        nus=nus,
        rao_amplitudes=rao_amplitudes,
        excitation_forces=excitation_forces,
        added_masses=added_masses,
        dampings=dampings,
    )


def estimate_natural_frequency(
    mass: float,
    added_mass_inf: FloatArray,
    restoring_matrix: FloatArray,
    dof: int,
    *,
    mass_matrix: FloatArray | None = None,
) -> float:
    """Estimate natural frequency for a single DOF.

    The natural frequency is:
        ω_n = sqrt(C[i,i] / (M[i,i] + A_∞[i,i]))

    For translational DOFs (i = 0, 1, 2) ``M[i,i]`` is simply the body mass.
    For rotational DOFs (i = 3, 4, 5) ``M[i,i]`` is a moment of inertia, so a
    scalar mass is dimensionally invalid: pass the 6×6 ``mass_matrix`` from
    :func:`tensorbem.hydrostatic.compute_mass_matrix` (its diagonal already
    includes the parallel-axis transfer to the reference point).

    Parameters
    ----------
    mass :
        Body mass (kg).
    added_mass_inf :
        Added mass matrix at infinite frequency (6×6).
    restoring_matrix :
        Hydrostatic restoring matrix (6×6).
    dof :
        Degree of freedom index (0-5).

    Returns
    -------
    omega_n :
        Natural circular frequency (rad/s).
    """
    if isinstance(dof, (bool, np.bool_)) or not isinstance(dof, (int, np.integer)):
        raise ValueError("dof must be an integer in [0, 5]")
    dof = int(dof)
    if dof < 0 or dof > 5:
        raise ValueError("dof must be an integer in [0, 5]")

    if mass_matrix is not None:
        mm = np.asarray(mass_matrix, dtype=float)
        if mm.shape != (6, 6):
            raise ValueError("mass_matrix must have shape (6, 6)")
        inertial = float(mm[dof, dof])
    elif dof < 3:
        inertial = float(mass)
    else:
        raise ValueError(
            "rotational DOF requires the 6x6 mass_matrix keyword; a scalar mass "
            "is dimensionally invalid for roll/pitch/yaw natural frequencies"
        )

    added = np.asarray(added_mass_inf, dtype=float)
    if added.shape != (6, 6):
        raise ValueError(
            f"added_mass_inf must have shape (6, 6), got {added.shape}"
        )
    restoring = np.asarray(restoring_matrix, dtype=float)
    if restoring.shape != (6, 6):
        raise ValueError(
            f"restoring_matrix must have shape (6, 6), got {restoring.shape}"
        )
    if not np.all(np.isfinite(added)) or not np.all(np.isfinite(restoring)):
        raise ValueError("added_mass_inf and restoring_matrix must be finite")

    m_eff = inertial + added[dof, dof]
    c = restoring[dof, dof]

    if m_eff <= 0 or c <= 0:
        return 0.0

    return math.sqrt(c / m_eff)
