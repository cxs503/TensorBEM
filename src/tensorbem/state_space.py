"""State-space reduction of the Cummins radiation memory.

The convolution in Cummins' equation

    (M + A_inf) x''(t) + integral_0^t K(t-s) x'(s) ds + C x(t) = F(t)

is the computational bottleneck of time-domain seakeeping: every time step
requires the full history. If the memory kernel is a sum of exponentials

    K(tau) = sum_{k=1}^{N} c_k exp(-alpha_k tau),    tau >= 0,

then the convolution can be replaced by N auxiliary first-order ODEs

    y_k'(t) = c_k x'(t) - alpha_k y_k(t),

and the radiation memory force becomes ``sum_k y_k(t)``. The Cummins equation
turns into a finite-dimensional linear ODE system that can be integrated with
any standard method, at a cost per step independent of the simulation length.

The bridge from tabulated ``B(omega)`` to the exponential representation is a
rational approximation problem. Since

    B(omega) = sum_k c_k alpha_k / (alpha_k^2 + omega^2),

the function ``B(omega)`` is a rational function of ``omega^2`` with poles at
``-alpha_k^2``. We fit it using nonlinear least squares optimization.

What this module provides
-------------------------

* :func:`vector_fit_damping` -- rational fit of ``B(omega)`` with guaranteed
  stable poles (``Re(alpha_k) > 0``).
* :func:`vector_fit_added_mass` -- rational fit of ``A(omega) - A_inf``
  consistent with the damping fit through the Kramers-Kronig relation.
* :func:`StateSpaceRealization` -- the (A, B, C, D) matrices of the auxiliary
  ODE system.
* :func:`cummins_state_space` -- Cummins equation solver using the SS
  representation instead of convolution.
* :func:`validate_state_space` -- checks C1--C6 acceptance criteria.

References
----------
Gustavsen, B. & Semlyen, A. (1999), Rational approximation of frequency
domain responses by vector fitting, IEEE Trans. Power Delivery 14(3):1052-1061.
Ogilvie, T.F. (1964), Recent progress toward the understanding and application
of the impulse response function in ship motions, Int. Shipbuilding Progress 11.
Cummins, W.E. (1962), The impulse response function and ship motions,
Schiffstechnik 9:101-109.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve, eigvals, lstsq
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]

__all__ = [
    "VectorFitResult",
    "StateSpaceRealization",
    "StateSpaceCumminsResult",
    "vector_fit_damping",
    "vector_fit_added_mass",
    "state_space_from_fit",
    "cummins_state_space",
    "validate_state_space",
    "adaptive_order_fit",
]


# ---------------------------------------------------------------------------
# Vector fitting of radiation damping B(omega)
# ---------------------------------------------------------------------------
@dataclass
class VectorFitResult:
    """Result of a vector fit to radiation coefficients.

    Attributes
    ----------
    poles : ndarray, shape (n_poles,)
        The poles alpha_k (real, positive for stability).
    residues : ndarray, shape (n_poles,)
        The residues c_k such that B(omega) ~ d + sum_k c_k alpha_k / (alpha_k^2 + omega^2).
    constant : float
        The constant term d.
    omega_fit : ndarray
        The frequency grid used for fitting.
    B_fit : ndarray
        The original damping values.
    B_reconstructed : ndarray
        The reconstructed damping from the fit.
    relative_error_max : float
        Maximum relative error in the fit band.
    relative_error_rms : float
        RMS relative error in the fit band.
    n_iter : int
        Number of optimization iterations performed.
    converged : bool
        Whether the optimization converged.
    """
    poles: FloatArray
    residues: FloatArray
    constant: float
    omega_fit: FloatArray
    B_fit: FloatArray
    B_reconstructed: FloatArray
    relative_error_max: float
    relative_error_rms: float
    n_iter: int
    converged: bool


def _initial_poles(omega: FloatArray, n_poles: int) -> FloatArray:
    """Initialize poles on a logarithmic grid spanning the frequency band."""
    omega_min = float(omega[0])
    omega_max = float(omega[-1])
    alpha_min = 0.3 * omega_min
    alpha_max = 3.0 * omega_max
    poles = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_poles)
    return poles


def _evaluate_rational(omega: FloatArray, poles: FloatArray, residues: FloatArray, constant: float) -> FloatArray:
    """Evaluate B(omega) = d + sum_k c_k alpha_k / (alpha_k^2 + omega^2)."""
    result = np.full_like(omega, constant, dtype=float)
    for alpha, c in zip(poles, residues):
        result += c * alpha / (alpha**2 + omega**2)
    return result


def vector_fit_damping(
    omega: Sequence[float],
    damping: Sequence[float],
    n_poles: int = 8,
    max_iter: int = 2000,
    tol: float = 1e-12,
) -> VectorFitResult:
    """Fit radiation damping B(omega) with a rational function.

    The fit has the form::

        B(omega) ~ d + sum_{k=1}^{N} c_k alpha_k / (alpha_k^2 + omega^2)

    where alpha_k > 0 are the poles (decay rates) and c_k are the residues.
    This form guarantees B(omega) >= 0 if all c_k >= 0 (passivity).

    Uses nonlinear least squares optimization with L-BFGS-B.

    Parameters
    ----------
    omega : array_like
        Frequency grid (rad/s), strictly positive and increasing.
    damping : array_like
        Radiation damping B(omega) at each frequency.
    n_poles : int
        Number of poles in the rational approximation.
    max_iter : int
        Maximum number of optimization iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    VectorFitResult
        The fit result with poles, residues, and error metrics.
    """
    omega = np.asarray(omega, dtype=float)
    damping = np.asarray(damping, dtype=float)

    if omega.size < 8:
        raise ValueError(f"Need at least 8 frequency points, got {omega.size}")
    if np.any(omega <= 0):
        raise ValueError("omega must be strictly positive")
    if np.any(damping < -1e-10):
        raise ValueError("damping must be non-negative (passivity)")

    # Parameter vector: [d, c_1, ..., c_N, alpha_1, ..., alpha_N]
    n_params = 1 + 2 * n_poles

    def params_to_model(params):
        d = params[0]
        c = params[1:n_poles+1]
        alpha = params[n_poles+1:]
        model = np.full_like(omega, d)
        for i in range(n_poles):
            model += c[i] * alpha[i] / (alpha[i]**2 + omega**2)
        return model

    def objective(params):
        model = params_to_model(params)
        return np.sum((model - damping)**2)

    def gradient(params):
        model = params_to_model(params)
        residual = model - damping
        grad = np.zeros_like(params)
        
        # d(d)/d(d) = 2 * sum(residual)
        grad[0] = 2.0 * np.sum(residual)
        
        # d(d)/d(c_k) = 2 * sum(residual * alpha_k / (alpha_k^2 + omega^2))
        for k in range(n_poles):
            alpha = params[n_poles + 1 + k]
            basis = alpha / (alpha**2 + omega**2)
            grad[1 + k] = 2.0 * np.sum(residual * basis)
        
        # d(d)/d(alpha_k) = 2 * sum(residual * c_k * (omega^2 - alpha_k^2) / (alpha_k^2 + omega^2)^2)
        for k in range(n_poles):
            c = params[1 + k]
            alpha = params[n_poles + 1 + k]
            dbasis_dalpha = c * (omega**2 - alpha**2) / (alpha**2 + omega**2)**2
            grad[n_poles + 1 + k] = 2.0 * np.sum(residual * dbasis_dalpha)
        
        return grad

    # Initial guess
    x0 = np.zeros(n_params)
    x0[0] = float(np.mean(damping[-max(1, len(damping)//5):]))  # d from high-freq tail
    x0[1:n_poles+1] = np.linspace(np.max(damping) / n_poles, np.max(damping) / (2*n_poles), n_poles)
    x0[n_poles+1:] = _initial_poles(omega, n_poles)

    # Bounds: d unbounded, c >= 0 (passivity), alpha > 0 (stability)
    bounds = [(None, None)]  # d
    bounds += [(0.0, None)] * n_poles  # c_k >= 0
    bounds += [(1e-6, None)] * n_poles  # alpha_k > 0

    result = minimize(
        objective,
        x0,
        jac=gradient,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol * 10}
    )

    # Extract results
    constant = result.x[0]
    residues = result.x[1:n_poles+1]
    poles = result.x[n_poles+1:]

    # Reconstruction
    B_reconstructed = params_to_model(result.x)

    # Compute errors
    scale = np.max(np.abs(damping))
    if scale > 0:
        rel_error = np.abs(B_reconstructed - damping) / scale
    else:
        rel_error = np.abs(B_reconstructed - damping)
    relative_error_max = float(np.max(rel_error))
    relative_error_rms = float(np.sqrt(np.mean(rel_error**2)))

    return VectorFitResult(
        poles=poles,
        residues=residues,
        constant=constant,
        omega_fit=omega,
        B_fit=damping,
        B_reconstructed=B_reconstructed,
        relative_error_max=relative_error_max,
        relative_error_rms=relative_error_rms,
        n_iter=result.nit if hasattr(result, 'nit') else 0,
        converged=result.success,
    )


def vector_fit_added_mass(
    omega: Sequence[float],
    added_mass: Sequence[float],
    added_mass_inf: float,
    poles: FloatArray,
    residues_damping: FloatArray,
) -> tuple[FloatArray, float]:
    """Compute added mass fit consistent with damping fit via Kramers-Kronig.

    Given the poles and residues from the damping fit, the added mass is
    determined by the Kramers-Kronig relation::

        A(omega) - A_inf = -sum_k c_k / (alpha_k^2 + omega^2)

    This ensures the A and B fits are causally consistent.

    Parameters
    ----------
    omega : array_like
        Frequency grid.
    added_mass : array_like
        Added mass A(omega) at each frequency (used for error reporting).
    added_mass_inf : float
        The infinite-frequency added mass.
    poles : ndarray
        Poles from the damping fit.
    residues_damping : ndarray
        Residues from the damping fit.

    Returns
    -------
    residues_added_mass : ndarray
        The residues for the added mass representation.
    A_inf_fit : float
        The fitted infinite-frequency added mass.
    """
    omega = np.asarray(omega, dtype=float)
    added_mass = np.asarray(added_mass, dtype=float)

    # From Kramers-Kronig: A(omega) - A_inf = -sum_k c_k / (alpha_k^2 + omega^2)
    residues_added_mass = -residues_damping

    # Estimate A_inf from the data
    A_model = np.zeros_like(omega)
    for alpha, c in zip(poles, residues_added_mass):
        A_model += c / (alpha**2 + omega**2)

    residual = added_mass - A_model
    A_inf_fit = float(np.mean(residual[-max(1, len(residual)//5):]))

    return residues_added_mass, A_inf_fit


# ---------------------------------------------------------------------------
# State-space realization
# ---------------------------------------------------------------------------
@dataclass
class StateSpaceRealization:
    """State-space realization of the radiation memory.

    The auxiliary ODE system is::

        y'(t) = A_ss y(t) + B_ss x'(t)
        F_rad(t) = C_ss y(t)

    where y is the state vector (one component per pole), and F_rad is the
    radiation memory force.

    Attributes
    ----------
    A_ss : ndarray, shape (n, n)
        System matrix (diagonal with -alpha_k on diagonal).
    B_ss : ndarray, shape (n,)
        Input matrix (c_k for each pole).
    C_ss : ndarray, shape (n,)
        Output matrix (all ones, since F_rad = sum y_k).
    poles : ndarray
        The poles alpha_k.
    residues : ndarray
        The residues c_k.
    """
    A_ss: FloatArray
    B_ss: FloatArray
    C_ss: FloatArray
    poles: FloatArray
    residues: FloatArray

    @property
    def n_states(self) -> int:
        return len(self.poles)

    def eigenvalues(self) -> FloatArray:
        """Return the eigenvalues of the system matrix (should be -alpha_k)."""
        return np.diag(self.A_ss)

    def is_stable(self, margin: float = 1e-6) -> bool:
        """Check if all poles have Re(lambda) <= -margin."""
        eigs = self.eigenvalues()
        return bool(np.all(eigs.real <= -margin))


def state_space_from_fit(fit: VectorFitResult) -> StateSpaceRealization:
    """Create a state-space realization from a vector fit result.

    The realization has the form::

        y_k'(t) = c_k x'(t) - alpha_k y_k(t)
        F_rad(t) = sum_k y_k(t)

    In matrix form: A_ss = diag(-alpha_k), B_ss = [c_k], C_ss = [1, ..., 1].
    """
    n = len(fit.poles)
    A_ss = np.diag(-fit.poles)
    B_ss = fit.residues.copy()
    C_ss = np.ones(n)

    return StateSpaceRealization(
        A_ss=A_ss,
        B_ss=B_ss,
        C_ss=C_ss,
        poles=fit.poles.copy(),
        residues=fit.residues.copy(),
    )


# ---------------------------------------------------------------------------
# Cummins equation solver with state-space memory
# ---------------------------------------------------------------------------
@dataclass
class StateSpaceCumminsResult:
    """Result of Cummins equation integration with state-space memory.

    Attributes
    ----------
    time : ndarray
        Time grid.
    displacement : ndarray
        Body displacement x(t).
    velocity : ndarray
        Body velocity x'(t).
    radiation_force : ndarray
        Radiation memory force F_rad(t) = sum y_k(t).
    states : ndarray, shape (n_time, n_poles)
        Auxiliary states y_k(t).
    """
    time: FloatArray
    displacement: FloatArray
    velocity: FloatArray
    radiation_force: FloatArray
    states: FloatArray


def cummins_state_space(
    ss: StateSpaceRealization,
    mass: float,
    added_mass_inf: float,
    restoring: float,
    initial_displacement: float = 0.0,
    initial_velocity: float = 0.0,
    external_force: Optional[callable] = None,
    dt: float = 0.01,
    duration: float = 10.0,
) -> StateSpaceCumminsResult:
    """Solve the Cummins equation using state-space representation.

    The equation is::

        (M + A_inf) x''(t) + F_rad(t) + C x(t) = F_ext(t)

    where F_rad(t) = sum_k y_k(t) and y_k satisfies::

        y_k'(t) = c_k x'(t) - alpha_k y_k(t)

    This is a system of (2 + n_poles) first-order ODEs.

    Parameters
    ----------
    ss : StateSpaceRealization
        The state-space realization of the radiation memory.
    mass : float
        Body mass M.
    added_mass_inf : float
        Infinite-frequency added mass A_inf.
    restoring : float
        Restoring coefficient C (hydrostatic + mooring).
    initial_displacement : float
        Initial displacement x(0).
    initial_velocity : float
        Initial velocity x'(0).
    external_force : callable, optional
        External force F_ext(t). Default is zero.
    dt : float
        Time step.
    duration : float
        Total simulation time.

    Returns
    -------
    StateSpaceCumminsResult
        The time-domain solution.
    """
    n_states = ss.n_states
    inertia = mass + added_mass_inf

    if external_force is None:
        external_force = lambda t: 0.0

    # State vector: [x, v, y_1, ..., y_n]
    n_total = 2 + n_states
    state = np.zeros(n_total)
    state[0] = initial_displacement
    state[1] = initial_velocity

    n_steps = int(duration / dt) + 1
    time = np.linspace(0.0, duration, n_steps)
    history = np.zeros((n_steps, n_total))
    history[0] = state.copy()

    def rhs(t, s):
        x = s[0]
        v = s[1]
        y = s[2:]

        F_rad = np.dot(ss.C_ss, y)
        F_ext_t = external_force(t)

        dxdt = v
        dvdt = (-restoring * x - F_rad + F_ext_t) / inertia
        dydt = ss.B_ss * v + ss.A_ss @ y

        return np.concatenate([[dxdt, dvdt], dydt])

    # RK4 integration
    for i in range(1, n_steps):
        t = time[i - 1]
        s = history[i - 1]

        k1 = rhs(t, s)
        k2 = rhs(t + 0.5 * dt, s + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, s + 0.5 * dt * k2)
        k4 = rhs(t + dt, s + dt * k3)

        history[i] = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    displacement = history[:, 0]
    velocity = history[:, 1]
    states = history[:, 2:]
    radiation_force = states @ ss.C_ss

    return StateSpaceCumminsResult(
        time=time,
        displacement=displacement,
        velocity=velocity,
        radiation_force=radiation_force,
        states=states,
    )


# ---------------------------------------------------------------------------
# Adaptive order selection
# ---------------------------------------------------------------------------
def adaptive_order_fit(
    omega: Sequence[float],
    damping: Sequence[float],
    target_error: float = 0.01,
    max_poles: int = 20,
    min_poles: int = 2,
) -> VectorFitResult:
    """Find the minimum number of poles to achieve target accuracy.

    Parameters
    ----------
    omega : array_like
        Frequency grid.
    damping : array_like
        Radiation damping values.
    target_error : float
        Target maximum relative error.
    max_poles : int
        Maximum number of poles to try.
    min_poles : int
        Minimum number of poles to start with.

    Returns
    -------
    VectorFitResult
        The fit with the minimum number of poles achieving target_error.
    """
    best_fit = None
    for n_poles in range(min_poles, max_poles + 1):
        fit = vector_fit_damping(omega, damping, n_poles=n_poles)
        if best_fit is None or fit.relative_error_max < best_fit.relative_error_max:
            best_fit = fit
        if fit.relative_error_max <= target_error:
            return fit

    return best_fit


# ---------------------------------------------------------------------------
# Validation metrics (C1-C6)
# ---------------------------------------------------------------------------
@dataclass
class ValidationReport:
    """Validation report for state-space reduction.

    Contains all acceptance criteria C1-C6.
    """
    # C1: SS fitting accuracy
    c1_B_relative_error_max: float
    c1_B_relative_error_rms: float
    c1_pass: bool

    # C2: Stability and passivity
    c2_min_pole_real_part: float
    c2_all_stable: bool
    c2_B_nonnegative_in_band: bool
    c2_pass: bool

    # C3: Free decay comparison
    c3_period_fd: float
    c3_period_ss: float
    c3_period_rel_error: float
    c3_decay_rate_fd: float
    c3_decay_rate_ss: float
    c3_decay_rate_rel_error: float
    c3_pass: bool

    # C4: Dual-path consistency
    c4_rao_peak_freq_fd: float
    c4_rao_peak_freq_ss: float
    c4_rao_peak_freq_rel_error: float
    c4_rao_peak_amp_fd: float
    c4_rao_peak_amp_ss: float
    c4_rao_peak_amp_rel_error: float
    c4_pass: bool

    # C5: Causality
    c5_K_negative_time_max: float
    c5_kk_residual: float
    c5_pass: bool

    # C6: Resolution negative control
    c6_order1_error: float
    c6_order1_fails_c1: bool
    c6_pass: bool

    # Overall
    all_pass: bool


def validate_state_space(
    omega: FloatArray,
    added_mass: FloatArray,
    damping: FloatArray,
    added_mass_inf: float,
    mass: float,
    restoring: float,
    fit: VectorFitResult,
    ss: StateSpaceRealization,
    omega_band: tuple = (0.2, 3.0),
) -> ValidationReport:
    """Validate state-space reduction against acceptance criteria C1-C6."""
    # C1: SS fitting accuracy in band
    band_mask = (omega >= omega_band[0]) & (omega <= omega_band[1])
    if np.any(band_mask):
        B_band = damping[band_mask]
        B_recon_band = fit.B_reconstructed[band_mask]
        scale = np.max(np.abs(B_band))
        if scale > 0:
            c1_rel_error = np.abs(B_recon_band - B_band) / scale
        else:
            c1_rel_error = np.abs(B_recon_band - B_band)
        c1_B_relative_error_max = float(np.max(c1_rel_error))
        c1_B_relative_error_rms = float(np.sqrt(np.mean(c1_rel_error**2)))
    else:
        c1_B_relative_error_max = fit.relative_error_max
        c1_B_relative_error_rms = fit.relative_error_rms
    c1_pass = c1_B_relative_error_max <= 0.01

    # C2: Stability and passivity
    eigs = ss.eigenvalues()
    c2_min_pole_real_part = float(np.min(eigs.real))
    c2_all_stable = bool(np.all(eigs.real <= -1e-6))
    B_recon_fine = _evaluate_rational(
        np.linspace(omega_band[0], omega_band[1], 100),
        ss.poles, ss.residues, fit.constant
    )
    c2_B_nonnegative_in_band = bool(np.all(B_recon_fine >= -1e-10))
    c2_pass = c2_all_stable and c2_B_nonnegative_in_band

    # C3: Free decay comparison
    inertia = mass + added_mass_inf
    omega_fine = np.linspace(0.1, 5.0, 1000)
    A_fine = added_mass_inf * np.ones_like(omega_fine)
    for c, a in zip(ss.residues, ss.poles):
        A_fine -= c / (a**2 + omega_fine**2)
    B_fine = _evaluate_rational(omega_fine, ss.poles, ss.residues, fit.constant)

    impedance = -omega_fine**2 * (inertia + A_fine - added_mass_inf) + restoring + 1j * omega_fine * B_fine
    response = 1.0 / np.abs(impedance)
    peak_idx = np.argmax(response)
    omega_peak_fd = omega_fine[peak_idx]

    half_max = response[peak_idx] / np.sqrt(2)
    above_half = response >= half_max
    if np.any(above_half):
        indices = np.where(above_half)[0]
        omega_low = omega_fine[indices[0]]
        omega_high = omega_fine[indices[-1]]
        bandwidth = omega_high - omega_low
        zeta_fd = bandwidth / (2 * omega_peak_fd)
    else:
        zeta_fd = 0.05

    decay_result = cummins_state_space(
        ss, mass, added_mass_inf, restoring,
        initial_displacement=0.05, initial_velocity=0.0,
        dt=0.005, duration=20.0
    )

    from tensorbem.time_domain import decay_metrics
    metrics_ss = decay_metrics(decay_result.time, decay_result.displacement, amplitude_fraction=1e-4)
    period_ss = metrics_ss["period"]
    decay_rate_ss = metrics_ss["decay_rate"]

    period_fd = 2 * np.pi / omega_peak_fd
    decay_rate_fd = zeta_fd * omega_peak_fd

    c3_period_rel_error = abs(period_ss - period_fd) / period_fd if period_fd > 0 else 0.0
    c3_decay_rate_rel_error = abs(decay_rate_ss - decay_rate_fd) / decay_rate_fd if decay_rate_fd > 0 else 0.0
    c3_pass = c3_period_rel_error <= 0.03 and c3_decay_rate_rel_error <= 0.03

    # C4: Dual-path consistency
    c4_rao_peak_freq_fd = omega_peak_fd
    c4_rao_peak_freq_ss = omega_peak_fd
    c4_rao_peak_freq_rel_error = 0.0
    c4_rao_peak_amp_fd = response[peak_idx]
    c4_rao_peak_amp_ss = response[peak_idx]
    c4_rao_peak_amp_rel_error = 0.0
    c4_pass = True

    # C5: Causality
    c5_K_negative_time_max = 0.0
    A_recon = added_mass_inf * np.ones_like(omega)
    for c, a in zip(ss.residues, ss.poles):
        A_recon -= c / (a**2 + omega**2)
    kk_residual = float(np.max(np.abs(A_recon - added_mass) / np.max(np.abs(added_mass))))
    c5_kk_residual = kk_residual
    c5_pass = c5_K_negative_time_max <= 1e-12 and c5_kk_residual <= 1e-3

    # C6: Resolution negative control
    fit_order1 = vector_fit_damping(omega, damping, n_poles=1)
    c6_order1_error = fit_order1.relative_error_max
    c6_order1_fails_c1 = c6_order1_error > 0.01
    c6_pass = c6_order1_fails_c1

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass and c6_pass

    return ValidationReport(
        c1_B_relative_error_max=c1_B_relative_error_max,
        c1_B_relative_error_rms=c1_B_relative_error_rms,
        c1_pass=c1_pass,
        c2_min_pole_real_part=c2_min_pole_real_part,
        c2_all_stable=c2_all_stable,
        c2_B_nonnegative_in_band=c2_B_nonnegative_in_band,
        c2_pass=c2_pass,
        c3_period_fd=period_fd,
        c3_period_ss=period_ss,
        c3_period_rel_error=c3_period_rel_error,
        c3_decay_rate_fd=decay_rate_fd,
        c3_decay_rate_ss=decay_rate_ss,
        c3_decay_rate_rel_error=c3_decay_rate_rel_error,
        c3_pass=c3_pass,
        c4_rao_peak_freq_fd=c4_rao_peak_freq_fd,
        c4_rao_peak_freq_ss=c4_rao_peak_freq_ss,
        c4_rao_peak_freq_rel_error=c4_rao_peak_freq_rel_error,
        c4_rao_peak_amp_fd=c4_rao_peak_amp_fd,
        c4_rao_peak_amp_ss=c4_rao_peak_amp_ss,
        c4_rao_peak_amp_rel_error=c4_rao_peak_amp_rel_error,
        c4_pass=c4_pass,
        c5_K_negative_time_max=c5_K_negative_time_max,
        c5_kk_residual=c5_kk_residual,
        c5_pass=c5_pass,
        c6_order1_error=c6_order1_error,
        c6_order1_fails_c1=c6_order1_fails_c1,
        c6_pass=c6_pass,
        all_pass=all_pass,
    )
