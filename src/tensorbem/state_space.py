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
    relative_error_max_pointwise: float = float("nan")
    weight_floor: float = 1e-3


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


def _design_matrix(omega: FloatArray, poles: FloatArray) -> FloatArray:
    """Columns ``[1, alpha_k / (alpha_k^2 + omega^2)]`` -- linear in (d, c)."""
    o2 = omega**2
    a = poles[:, None]
    return np.concatenate([np.ones((omega.size, 1)), (a / (a * a + o2)).T], axis=1)


def _weights(omega: FloatArray, damping: FloatArray, weight_floor: float) -> FloatArray:
    """Relative-error weights, floored at ``weight_floor * max|B|``.

    Fitting a radiation-damping curve that spans several decades (the Hulme
    hemisphere heave curve spans ``1.7e3 : 2.1e-2``, four decades) with plain
    absolute residuals lets the peak dominate the objective and leaves the
    low- and high-frequency ends unconstrained.  Weighting each residual by
    ``1/max(|B|, floor * max|B|)`` makes the objective an error *relative to
    the local value*, with a floor that keeps the near-zero tail from
    dominating in turn.
    """
    scale = float(np.max(np.abs(damping)))
    if scale <= 0.0:
        return np.ones_like(damping)
    return 1.0 / np.maximum(np.abs(damping), weight_floor * scale)


def _solve_linear(
    omega: FloatArray,
    damping: FloatArray,
    weights: FloatArray,
    poles: FloatArray,
) -> tuple[FloatArray, float]:
    """Solve the weighted linear least squares for (d, c) at fixed poles.

    Returns the coefficient vector ``[d, c_1 ... c_N]`` and the weighted
    residual sum of squares.  ``d`` (the ``omega -> infinity`` limit of
    ``B``) is held non-negative: a negative asymptote would be a negative
    damping at high frequency, i.e. an active system.
    """
    A = _design_matrix(omega, poles)
    Aw = A * weights[:, None]
    bw = damping * weights
    coef = np.linalg.lstsq(Aw, bw, rcond=None)[0]
    if coef[0] < 0.0:
        # Re-solve with the constant pinned at the physical bound d = 0.
        coef_rest = np.linalg.lstsq(Aw[:, 1:], bw, rcond=None)[0]
        coef = np.concatenate([[0.0], coef_rest])
    residual = (A @ coef - damping) * weights
    return coef, float(residual @ residual)


def vector_fit_damping(
    omega: Sequence[float],
    damping: Sequence[float],
    n_poles: int = 8,
    max_iter: int = 400,
    tol: float = 1e-12,
    weight_floor: float = 1e-3,
) -> VectorFitResult:
    """Fit ``B(omega) ~ d + sum_{k=1}^{N} c_k alpha_k / (alpha_k^2 + omega^2)``.

    Parameters
    ----------
    omega : array_like
        Frequency grid (rad/s), strictly positive and increasing.
    damping : array_like
        Radiation damping B(omega) at each frequency.
    n_poles : int
        Number of poles in the rational approximation.
    max_iter : int
        Maximum number of outer (pole-position) optimisation iterations.
    tol : float
        Convergence tolerance of the outer optimisation (ftol).
    weight_floor : float
        Relative weight floor, see :func:`_weights`.

    Returns
    -------
    VectorFitResult
        The fit result with poles, residues, and error metrics.

    Notes
    -----
    ``B`` is a rational function of ``omega^2`` whose poles are the decay
    rates ``alpha_k``, and for a *fixed* pole set the residues ``c_k`` and the
    constant ``d`` enter linearly.  The fit is therefore done as a separable
    (variable-projection) weighted least-squares problem: the outer
    optimisation moves only the ``log alpha_k`` (``n_poles`` variables, bounded
    by the frequency band) and the inner problem solves (d, c) exactly with a
    linear least-squares solve.  The objective is the weighted residual norm
    with relative weighting (:func:`_weights`), which is what makes a
    four-decade damping curve fittable at all.

    This is a separable-least-squares relative of the pole-relocation idea of
    Sanathanan & Koerner (1963) and Gustavsen & Semlyen (1999) -- *not* the
    Gustavsen iteration itself.  Residues are left unconstrained in sign: the
    radiation memory kernel of a real hull oscillates, so a fit restricted to
    ``c_k >= 0`` cannot represent it (measured on the Hulme hemisphere curve:
    the best non-negative-residue fit stalls at 100 % peak error for every
    order from 3 to 15).
    """
    omega = np.asarray(omega, dtype=float)
    damping = np.asarray(damping, dtype=float)

    if omega.size < 8:
        raise ValueError(f"Need at least 8 frequency points, got {omega.size}")
    if np.any(omega <= 0):
        raise ValueError("omega must be strictly positive")
    if np.any(damping < -1e-10):
        raise ValueError("damping must be non-negative (passivity)")
    if n_poles < 1:
        raise ValueError(f"n_poles must be >= 1, got {n_poles}")

    weights = _weights(omega, damping, weight_floor)
    omega_min, omega_max = float(omega[0]), float(omega[-1])
    # Poles are searched in log space and kept inside a band that brackets the
    # fitted frequency window; without the bounds the optimiser happily parks
    # poles at 1e-12 / 1e12 where they only cancel each other numerically.
    lower = math.log(1e-3 * omega_min)
    upper = math.log(1e3 * omega_max)
    bounds = [(lower, upper)] * n_poles

    def objective(log_poles: FloatArray) -> float:
        return _solve_linear(omega, damping, weights, np.exp(log_poles))[1]

    # Multi-start: the objective has local minima, and a single log-spaced
    # start converges to a fit that misses the ends of the band (the pre-m14767
    # failure mode).  Keep the best of three deterministic starts.
    starts = [
        np.log(_initial_poles(omega, n_poles)),                                  # 0.3..3 x band
        np.linspace(math.log(omega_min), math.log(omega_max), n_poles),          # strictly in band
        np.linspace(lower, upper, n_poles),                                      # bracketing band
    ]

    best: Optional[tuple[float, FloatArray, FloatArray, int, bool]] = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": tol, "gtol": tol * 10},
        )
        poles = np.exp(result.x)
        coef, cost = _solve_linear(omega, damping, weights, poles)
        if best is None or cost < best[0]:
            best = (cost, poles, coef, int(getattr(result, "nit", 0)), bool(result.success))

    assert best is not None
    _, poles, coef, n_iter, converged = best
    constant = float(coef[0])
    residues = coef[1:]

    B_reconstructed = _evaluate_rational(omega, poles, residues, constant)

    scale = float(np.max(np.abs(damping)))
    if scale > 0:
        rel_error = np.abs(B_reconstructed - damping) / scale
        pointwise = np.abs(B_reconstructed - damping) / np.maximum(
            np.abs(damping), weight_floor * scale
        )
    else:
        rel_error = np.abs(B_reconstructed - damping)
        pointwise = rel_error
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
        n_iter=n_iter,
        converged=converged,
        relative_error_max_pointwise=float(np.max(pointwise)),
        weight_floor=weight_floor,
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
    D_ss : float
        Feedthrough of the fitted constant term: ``F_rad = C_ss y + D_ss x'``.
        ``D_ss`` is the ``omega -> infinity`` limit of the fitted ``B`` and is
        the coefficient of the delta function in the memory kernel
        ``K(tau) = sum_k c_k exp(-alpha_k tau) + D_ss delta(tau)``.  Dropping it
        silently (the pre-m14767 behaviour) makes the realized model radiate
        less than the fit it claims to represent.
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
    D_ss: float = 0.0

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
        D_ss=float(fit.constant),
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
    substeps : int
        Number of RK4 substeps actually taken per output interval, set by the
        explicit-integrator stability guard (``1`` when ``dt`` is already fine
        enough).
    """
    time: FloatArray
    displacement: FloatArray
    velocity: FloatArray
    radiation_force: FloatArray
    states: FloatArray
    substeps: int = 1


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

        F_rad = float(np.dot(ss.C_ss, y)) + ss.D_ss * v
        F_ext_t = external_force(t)

        dxdt = v
        dvdt = (-restoring * x - F_rad + F_ext_t) / inertia
        dydt = ss.B_ss * v + ss.A_ss @ y

        return np.concatenate([[dxdt, dvdt], dydt])

    # Explicit RK4 is only conditionally stable.  The auxiliary eigenvalues are
    # -alpha_k (real, |lambda| = alpha_max) and the body pair is nearly
    # imaginary (omega_n), so a step larger than the RK4 stability boundary
    # blows up instead of merely losing accuracy -- which is exactly what the
    # pre-m14767 fits (poles parked on the 1e-6 lower bound with residues of
    # ~7.5e3) did on the Hulme hemisphere data: overflow to NaN at dt = 0.01.
    # Substep internally rather than silently returning garbage.
    omega_n = math.sqrt(max(restoring / inertia, 0.0))
    rate_max = max(float(np.max(np.abs(ss.poles))) if ss.n_states else 0.0, omega_n)
    h_stable = 2.0 / rate_max if rate_max > 0.0 else dt
    substeps = max(1, int(math.ceil(dt / h_stable)))
    if substeps > 20000:
        raise ValueError(
            f"dt={dt:g} s needs {substeps} RK4 substeps for stability "
            f"(fastest rate {rate_max:g} 1/s); pass a smaller dt"
        )
    h = dt / substeps

    for i in range(1, n_steps):
        t = time[i - 1]
        s = history[i - 1]

        for _ in range(substeps):
            k1 = rhs(t, s)
            k2 = rhs(t + 0.5 * h, s + 0.5 * h * k1)
            k3 = rhs(t + 0.5 * h, s + 0.5 * h * k2)
            k4 = rhs(t + h, s + h * k3)
            s = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h

        history[i] = s

    displacement = history[:, 0]
    velocity = history[:, 1]
    states = history[:, 2:]
    radiation_force = states @ ss.C_ss + ss.D_ss * velocity

    return StateSpaceCumminsResult(
        time=time,
        displacement=displacement,
        velocity=velocity,
        radiation_force=radiation_force,
        states=states,
        substeps=substeps,
    )



def _harmonic_memory_response(
    ss: StateSpaceRealization,
    omega: float,
    velocity_amplitude: float = 1.0,
    settle_time_factor: float = 7.0,
    samples_per_period: int = 120,
) -> complex:
    """Complex response ``G(i omega) = F_rad / x'`` of the realization.

    The auxiliary ODEs are integrated with a prescribed harmonic velocity
    ``v(t) = v0 sin(omega t)`` (explicit RK4 with the same stability guard as
    :func:`cummins_state_space`), the transient is allowed to die out, and the
    complex Fourier coefficient of the last four periods is extracted.  This is
    the *time-domain* path; the frequency-domain counterpart is

        ``G(i omega) = D_ss + sum_k c_k / (alpha_k + i omega)``,

    whose real part is the fitted damping and whose imaginary part is
    ``omega * (A(omega) - A_inf)``.  Comparing the two is what C4 is supposed
    to test; before m14767 it compared a frequency-domain number with itself
    and returned ``pass = True``.
    """
    period = 2.0 * math.pi / omega
    h = period / samples_per_period
    alpha_min = float(np.min(np.abs(ss.poles))) if ss.n_states else omega
    rate_max = max(float(np.max(np.abs(ss.poles))) if ss.n_states else 0.0, omega)
    n_sub = max(1, int(math.ceil(h * rate_max / 2.0)))
    hs = h / n_sub

    settle = max(settle_time_factor / alpha_min if alpha_min > 0 else 0.0, 4.0 * period)
    n_settle = int(math.ceil(settle / h))
    n_window = int(4 * samples_per_period)
    n_total = n_settle + n_window

    def velocity(tt: float) -> float:
        return velocity_amplitude * math.sin(omega * tt)

    y = np.zeros(ss.n_states)
    t = 0.0
    window_t = np.empty(n_window)
    window_f = np.empty(n_window)
    for step in range(n_total):
        for _ in range(n_sub):
            k1 = ss.B_ss * velocity(t) + ss.A_ss @ y
            tm = t + 0.5 * hs
            k2 = ss.B_ss * velocity(tm) + ss.A_ss @ (y + 0.5 * hs * k1)
            k3 = ss.B_ss * velocity(tm) + ss.A_ss @ (y + 0.5 * hs * k2)
            t4 = t + hs
            k4 = ss.B_ss * velocity(t4) + ss.A_ss @ (y + hs * k3)
            y = y + (hs / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t = t4
        if step >= n_settle:
            j = step - n_settle
            window_t[j] = t
            window_f[j] = float(np.dot(ss.C_ss, y)) + ss.D_ss * velocity(t)

    coefficient = np.mean(window_f * np.exp(-1j * omega * window_t))
    return complex(2j * coefficient / velocity_amplitude)


def radiation_response(ss: StateSpaceRealization, omega: float) -> complex:
    """Frequency-domain response ``G(i omega) = B(omega) + i omega (A - A_inf)``."""
    omega = float(omega)
    acc = ss.D_ss + 0.0j
    for c, a in zip(ss.residues, ss.poles):
        acc += c / (float(a) + 1j * omega)
    return complex(acc)


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

    # C4: Dual-path consistency (time-domain realization vs frequency domain)
    c4_memory_amp_rel_error_max: float
    c4_memory_phase_error_max: float
    c4_pass: bool

    # C5: Kramers-Kronig consistency of the damping fit (causality is structural)
    c5_kk_residual: float
    c5_min_pole: float
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

    # C4: dual-path consistency of the *realization* -- integrate the auxiliary
    # ODEs under a prescribed harmonic velocity and compare with the analytic
    # complex response of the same poles/residues/feedthrough.
    band_edges = (max(omega_band[0], float(omega[0])), min(omega_band[1], float(omega[-1])))
    test_omegas = np.unique(
        np.concatenate([
            np.linspace(band_edges[0], band_edges[1], 3),
            np.array([omega_peak_fd]),
        ])
    )
    amp_errors, phase_errors = [], []
    for om in test_omegas:
        analytic = radiation_response(ss, om)
        numeric = _harmonic_memory_response(ss, float(om))
        amp_errors.append(abs(abs(numeric) - abs(analytic)) / abs(analytic))
        phase_errors.append(abs(float(np.angle(numeric / analytic))))
    c4_memory_amp_rel_error_max = float(np.max(amp_errors))
    c4_memory_phase_error_max = float(np.max(phase_errors))
    c4_pass = c4_memory_amp_rel_error_max <= 0.01 and c4_memory_phase_error_max <= 0.01

    # C5: Kramers-Kronig consistency between the fitted B and the measured A.
    # Causality itself is structural for Re(alpha_k) > 0 (the kernel is a sum of
    # decaying exponentials, identically zero for tau < 0); what is *not*
    # structural is whether the damping fit reproduces the added mass, so that
    # is what is measured here.  The pre-m14767 report asserted
    # ``c5_K_negative_time_max = 0.0`` -- a number by construction.
    A_recon = added_mass_inf * np.ones_like(omega)
    for c, a in zip(ss.residues, ss.poles):
        A_recon -= c / (a**2 + omega**2)
    kk_residual = float(np.max(np.abs(A_recon - added_mass) / np.max(np.abs(added_mass))))
    c5_kk_residual = kk_residual
    c5_min_pole = float(np.min(ss.poles)) if ss.n_states else float("nan")
    c5_pass = c5_min_pole > 0.0 and c5_kk_residual <= 0.03

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
        c4_memory_amp_rel_error_max=c4_memory_amp_rel_error_max,
        c4_memory_phase_error_max=c4_memory_phase_error_max,
        c4_pass=c4_pass,
        c5_kk_residual=c5_kk_residual,
        c5_min_pole=c5_min_pole,
        c5_pass=c5_pass,
        c6_order1_error=c6_order1_error,
        c6_order1_fails_c1=c6_order1_fails_c1,
        c6_pass=c6_pass,
        all_pass=all_pass,
    )
