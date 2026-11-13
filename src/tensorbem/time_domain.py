"""Cummins-equation impulse response of the radiation memory force.

Every frequency-domain radiation solution in this repository (panel solutions,
Hulme's semi-analytical hemisphere, tabulated data) is a *steady-state*
description of the fluid force at one frequency. A seakeeping simulation needs
the transient: the force that still acts on the body after the motion that
created it has stopped. The bridge between the two is Cummins' (1962) equation

    (M + A_inf) x''(t) + integral_0^t K(t - s) x'(s) ds + C x(t) = F(t),

whose memory kernel is the real, even, decaying impulse response

    K(tau) = (2/pi) integral_0^inf B(omega) cos(omega tau) domega ,           (1)
    B(omega)          = integral_0^inf K(tau) cos(omega tau) dtau ,           (2)
    A(omega) - A_inf  = -(1/omega) integral_0^inf K(tau) sin(omega tau) dtau . (3)

Eqs. (1)-(3) are one statement in three places: the same memory kernel must
reproduce *both* radiation coefficients. A kernel that reproduces ``B`` but not
``A`` (or whose ``A_inf`` is inconsistent) makes the time-domain and the
frequency-domain description of the same body disagree, and that disagreement
is invisible in either description alone. The checks here are therefore a
two-way round trip plus a time-domain/frequency-domain cross-check, not a
plausibility test on a single curve.

What is verified
----------------

* **Decay and parity of K** (:func:`kernel_diagnostics`,
  :func:`two_sided_kernel`). K is the cosine transform of a real function: it
  is even in ``tau``, its slope at ``tau = 0`` vanishes, it is bounded by
  ``K(0) = (2/pi) integral B domega`` and it must decay. A kernel built with
  ``sin`` instead of ``cos``, or from a sign-flipped damping sweep, fails these
  checks.
* **Fourier reciprocity inside the sampled band**
  (:func:`fourier_round_trip`). ``B`` is transformed to K by an *exact*
  piecewise-linear quadrature of (1) and transformed back by Simpson
  quadratures of (2)-(3) on a different grid; K then has to reproduce ``A``
  through (3) with the same ``A_inf``. The forward and inverse legs share
  neither a quadrature rule nor a sample set, so the residual measures the
  transform. Band edges are handled explicitly: below ``omega_min`` the sweep
  is extended with the long-wave model ``B -> c omega^3``, above ``omega_max``
  it is truncated with a reported bound.
* **Time domain vs frequency domain** (:func:`cummins_free_decay`,
  :func:`decay_metrics`, :func:`frequency_domain_impedance`,
  :func:`resonance_properties`). The Cummins equation is integrated with the
  same kernel and its free-decay period and decay rate are compared with the
  resonance and the half-power bandwidth of the mechanical response function
  ``1/kappa(omega)``, which is built from the sweep alone and never sees the
  time integration. Two frequency-domain damping estimates are reported: the
  energy identity ``B(omega_n)/(2 (M + A(omega_n)) omega_n)`` and the
  amplitude-based half-power bandwidth. The bandwidth estimate is the one a
  decay test measures, and the gap between the two is the size of the memory
  effect -- the free decay samples a band while the energy identity samples a
  frequency.

Sampling contract
-----------------

Both directions approximate an integral over an unbounded domain and both have
a hard aliasing limit, which this module *refuses* instead of silently
resampling:

* a kernel sampled with step ``dtau`` cannot represent frequencies above
  ``pi/dtau``, so ``dtau <= pi/omega_max`` is required;
* a kernel whose grid cannot resolve a requested frequency
  (``dtau > pi/omega_query_max``) is rejected as well.

A radiation sweep with *negative* damping is rejected too: ``B(omega) >= 0`` is
the passivity of the radiation problem (a passive body cannot radiate energy it
does not receive); a negative value would make the memory kernel grow.

Limits
------

Linear, deep-water, zero-forward-speed radiation in a single degree of freedom
(the caller supplies one ``(omega, A, B)`` triple per mode). The kernel is
truncated at ``tau_max`` and the sweep at ``omega_max``. Nonlinear roll
damping, parametric roll and forward speed are not modelled here; nothing in
this module solves a boundary-value problem -- it consumes coefficients from a
solver or a reference and verifies the frequency-to-time mapping.

References
----------

Cummins, W.E. (1962), *The impulse response function and ship motions*,
Schiffstechnik 9:101-109. Ogilvie, T.F. (1964), *Recent progress toward the
understanding and application of the impulse response function in ship
motions*, Int. Shipbuilding Progress 11. Kotik & Mangulis (1962) for the
Kramers-Kronig pairing behind eq. (3). Hulme, A. (1982), J. Fluid Mech.
121:443-463 for the floating-hemisphere coefficients used as the worked case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from tensorbem.causality import fit_added_mass_asymptote

FloatArray = NDArray[np.float64]

__all__ = [
    "ConstantAddedMassEnvelope",
    "FourierRoundTrip",
    "FreeDecayResult",
    "MemoryKernel",
    "added_mass_from_memory_kernel",
    "check_frequency_sampling",
    "check_time_sampling",
    "constant_added_mass_error_envelope",
    "cummins_free_decay",
    "damping_from_memory_kernel",
    "decay_metrics",
    "frequency_domain_impedance",
    "fourier_round_trip",
    "half_power_damping",
    "harmonic_response",
    "implied_added_mass_inf",
    "kernel_diagnostics",
    "radiation_memory_kernel",
    "resonance_properties",
    "two_sided_kernel",
    "validate_radiation_sweep",
]

_MINIMUM_POINTS = 8


# ---------------------------------------------------------------------------
# input validation and the sampling contract
# ---------------------------------------------------------------------------
def _as_float_array(name: str, values: Sequence[float]) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    return array


def validate_radiation_sweep(
    omega: Sequence[float],
    added_mass: Sequence[float],
    damping: Sequence[float],
    *,
    damping_relative_tolerance: float = 1.0e-10,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``(omega, added_mass, damping)`` as validated float arrays.

    Rejections (all ``ValueError``, with the offending quantity named):

    * fewer than eight tabulated frequencies -- too short to resolve a kernel;
    * non-finite values, or ``omega`` that is not strictly increasing and
      positive;
    * ``damping < -tol * scale``. Radiation damping is passive; a negative
      value would make the memory kernel grow and any downstream decay study
      meaningless, so the sweep is rejected rather than silently used.
    """
    w = _as_float_array("omega", omega)
    a = _as_float_array("added_mass", added_mass)
    b = _as_float_array("damping", damping)
    if not (w.size == a.size == b.size):
        raise ValueError(
            "omega, added_mass and damping must have the same length "
            f"(got {w.size}, {a.size}, {b.size})"
        )
    if w.size < _MINIMUM_POINTS:
        raise ValueError(
            f"a radiation sweep needs at least {_MINIMUM_POINTS} points, got {w.size}"
        )
    if np.any(w <= 0.0):
        raise ValueError("omega must be strictly positive")
    if np.any(np.diff(w) <= 0.0):
        raise ValueError("omega must be strictly increasing")
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    negative = b < -damping_relative_tolerance * max(scale, 1.0)
    if np.any(negative):
        index = int(np.argmin(b))
        raise ValueError(
            "negative radiation damping is not physical (energy would be "
            f"created, not radiated): min damping = {b[index]:.6g} at "
            f"omega = {w[index]:.6g}; B(omega) >= 0 is required"
        )
    return w, a, b


def check_time_sampling(dt: float, omega_max: float) -> float:
    """Reject a kernel time step that aliases the sampled frequency band.

    A grid with step ``dt`` represents frequencies up to ``pi/dt``. If the
    sweep extends beyond that, the kernel is aliased and eqs. (1)-(2) no longer
    describe one physical memory: ``omega + 2*pi/dt`` folds onto ``omega``.
    Returns the Nyquist frequency when the sampling is acceptable.
    """
    dt = float(dt)
    omega_max = float(omega_max)
    if not (dt > 0.0 and math.isfinite(dt)):
        raise ValueError(f"time step must be positive and finite, got {dt!r}")
    if not (omega_max > 0.0 and math.isfinite(omega_max)):
        raise ValueError(f"omega_max must be positive and finite, got {omega_max!r}")
    nyquist = math.pi / dt
    if omega_max > nyquist * (1.0 + 1.0e-12):
        raise ValueError(
            "sampling does not satisfy Nyquist: kernel step "
            f"dt = {dt:.6g} resolves frequencies up to pi/dt = {nyquist:.6g} "
            f"rad/s, but the radiation sweep reaches omega_max = {omega_max:.6g} "
            f"rad/s; use dt <= {math.pi / omega_max:.6g}"
        )
    return nyquist


def check_frequency_sampling(
    omega_query_max: float, tau_max: float, *, dtau: float
) -> None:
    """Reject a kernel grid that cannot represent the requested frequencies."""
    dtau = float(dtau)
    tau_max = float(tau_max)
    omega_query_max = float(omega_query_max)
    if not (tau_max > 0.0 and math.isfinite(tau_max)):
        raise ValueError(f"tau_max must be positive and finite, got {tau_max!r}")
    if not (omega_query_max > 0.0 and math.isfinite(omega_query_max)):
        raise ValueError(
            f"omega_query_max must be positive and finite, got {omega_query_max!r}"
        )
    if omega_query_max * dtau > math.pi * (1.0 + 1.0e-12):
        raise ValueError(
            "sampling does not satisfy Nyquist: the kernel step "
            f"dtau = {dtau:.6g} resolves frequencies up to pi/dtau = "
            f"{math.pi / dtau:.6g} rad/s, but omega = {omega_query_max:.6g} "
            f"rad/s was requested; use dtau <= {math.pi / omega_query_max:.6g}"
        )

# ---------------------------------------------------------------------------
# the impulse response K(tau)
# ---------------------------------------------------------------------------
def _piecewise_linear(damping: FloatArray, omega: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Coefficients ``(c0, c1)`` of the linear interpolant of ``B``."""
    a = omega[:-1]
    slope = np.diff(damping) / np.diff(omega)
    return damping[:-1] - a * slope, slope


def _cosine_moments(omega: FloatArray, tau: FloatArray) -> tuple[FloatArray, FloatArray]:
    """``int_a^b cos(w t) dw`` and ``int_a^b w cos(w t) dw`` for each interval.

    Both are written with :func:`numpy.sinc` so the ``tau -> 0`` limit is
    reached without cancellation; the closed forms are

        int_a^b cos(w t) dw   = b sinc(bt/pi) - a sinc(at/pi)
        int_a^b w cos(w t) dw = b^2 sinc(bt/pi) - a^2 sinc(at/pi)
                                + (a^2-b^2)/2 sinc((a+b)t/2pi) sinc((b-a)t/2pi).
    """
    a = omega[:-1]
    b = omega[1:]
    t = np.asarray(tau, dtype=float)[:, None]
    sa = np.sinc(a * t / math.pi)
    sb = np.sinc(b * t / math.pi)
    order_zero = b * sb - a * sa
    order_one = (
        b * b * sb
        - a * a * sa
        + 0.5 * (a * a - b * b) * np.sinc((a + b) * t / (2.0 * math.pi)) * np.sinc((b - a) * t / (2.0 * math.pi))
    )
    return order_zero, order_one


def _long_wave_damping_scale(omega: FloatArray, damping: FloatArray) -> float:
    """Fit ``B(omega) ~ c omega^3`` on the lowest decade of the sweep."""
    if omega.size < 2 or damping[0] <= 0.0:
        return 0.0
    limit = min(omega[0] * 10.0, omega[-1])
    mask = (omega <= limit) & (omega > 0.0)
    if np.count_nonzero(mask) < 2:
        mask = np.zeros_like(omega, dtype=bool)
        mask[: min(3, omega.size)] = True
    return float(np.mean(damping[mask] / omega[mask] ** 3))


def _long_wave_cosine(scale: float, omega_low: float, tau: FloatArray) -> FloatArray:
    """``int_0^omega_low c w^3 cos(w t) dw`` in closed form, stable at ``t -> 0``."""
    tau = np.asarray(tau, dtype=float)
    if scale == 0.0:
        return np.zeros_like(tau)
    x = omega_low * tau
    bracket = (
        x**3 * np.sin(x)
        + 3.0 * x * x * np.cos(x)
        - 6.0 * x * np.sin(x)
        + 12.0 * np.sin(0.5 * x) ** 2
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        values = scale * bracket / tau**4
    return np.where(np.abs(x) < 1.0e-8, scale * omega_low**4 / 4.0, values)


@dataclass(frozen=True)
class MemoryKernel:
    """A sampled radiation memory kernel ``K(tau)``.

    ``tau`` is uniform and starts at zero; ``values`` holds ``K`` there.
    Evaluation is piecewise linear inside the grid, **zero for negative
    times** (causality: the fluid cannot react before the body moves) and zero
    beyond ``tau_max`` (truncation, reported by :attr:`truncation`).
    """

    tau: FloatArray
    values: FloatArray
    omega_max: float = 0.0
    damping_scale: float = 0.0
    low_frequency_model: str = "cubic"
    high_frequency_bound: float = 0.0
    low_frequency_bound: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        tau = np.asarray(self.tau, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if tau.ndim != 1 or values.ndim != 1 or tau.size != values.size:
            raise ValueError("tau and values must be one-dimensional and equal in length")
        if tau.size < 2:
            raise ValueError("a memory kernel needs at least two samples")
        if tau[0] != 0.0:
            raise ValueError("the memory kernel grid must start at tau = 0")
        steps = np.diff(tau)
        if np.any(steps <= 0.0):
            raise ValueError("the memory kernel grid must be strictly increasing")
        if np.ptp(steps) > 1.0e-9 * float(np.mean(steps)):
            raise ValueError("the memory kernel grid must be uniform")
        if not np.all(np.isfinite(values)):
            raise ValueError("the memory kernel contains a non-finite value")
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "values", values)

    @property
    def dt(self) -> float:
        """Uniform sample step of the kernel grid."""
        return float(self.tau[1] - self.tau[0])

    @property
    def span(self) -> float:
        """``tau_max``: the longest memory time represented."""
        return float(self.tau[-1])

    @property
    def value_at_zero(self) -> float:
        return float(self.values[0])

    @property
    def nyquist_omega(self) -> float:
        """Highest frequency the grid can represent, ``pi/dt``."""
        return math.pi / self.dt

    def __call__(self, tau: float | Sequence[float]) -> FloatArray:
        """Evaluate ``K``; exactly zero for ``tau < 0`` (causality)."""
        query = np.asarray(tau, dtype=float)
        linear = np.interp(query, self.tau, self.values, left=0.0, right=0.0)
        return np.where(query < 0.0, 0.0, linear)

    @property
    def truncation(self) -> dict:
        """Size of the discarded low- and high-frequency bands."""
        scale = float(np.max(np.abs(self.values))) if self.values.size else 0.0
        return {
            "tau_max": self.span,
            "omega_max": self.omega_max,
            "low_frequency_model": self.low_frequency_model,
            "damping_scale": self.damping_scale,
            "high_frequency_band_bound": self.high_frequency_bound,
            "low_frequency_band_bound": self.low_frequency_bound,
            "relative_high_frequency_bound": (
                self.high_frequency_bound / scale if scale else 0.0
            ),
            "relative_low_frequency_bound": (
                self.low_frequency_bound / scale if scale else 0.0
            ),
        }


def radiation_memory_kernel(
    omega: Sequence[float],
    damping: Sequence[float],
    tau: Sequence[float],
    *,
    low_frequency_model: str = "cubic",
) -> MemoryKernel:
    """Memory kernel ``K(tau) = (2/pi) int_0^inf B(omega) cos(omega tau) domega``.

    ``B`` is integrated *exactly* as a piecewise-linear function of ``omega``,
    so no FFT leakage or band-edge aliasing enters the forward leg. The band
    ``[0, omega_min]`` is closed with the long-wave model ``B ~ c omega^3``
    (``low_frequency_model="cubic"``) or dropped (``"zero"``); above
    ``omega_max`` the sweep is truncated and the resulting bound is stored on
    the returned kernel.

    ``tau`` must be uniform, start at zero, and satisfy
    ``dtau <= pi/omega_max`` (:func:`check_time_sampling`).
    """
    w, _, b = validate_radiation_sweep(omega, np.zeros_like(np.asarray(damping, dtype=float)), damping)
    grid = np.asarray(tau, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("tau must be a one-dimensional grid with at least two samples")
    if grid[0] != 0.0:
        raise ValueError("tau must start at 0")
    steps = np.diff(grid)
    if np.ptp(steps) > 1.0e-9 * float(np.mean(steps)):
        raise ValueError("tau must be uniformly spaced")
    if low_frequency_model not in ("cubic", "zero"):
        raise ValueError(
            f"low_frequency_model must be 'cubic' or 'zero', got {low_frequency_model!r}"
        )
    check_time_sampling(float(steps[0]), float(w[-1]))

    c0, c1 = _piecewise_linear(b, w)
    order_zero, order_one = _cosine_moments(w, grid)
    values = float(2.0 / math.pi) * (order_zero @ c0 + order_one @ c1)

    scale = _long_wave_damping_scale(w, b) if low_frequency_model == "cubic" else 0.0
    if scale:
        values = values + _long_wave_cosine(scale, float(w[0]), grid)

    # Truncation bounds: the missing high band contributes at most
    # (2/pi) * int_{omega_max}^inf B domega <= (2/pi) B(omega_max) omega_max, and
    # the missing low band at most (2/pi) * c * omega_min^4 / 4.
    high_bound = float(2.0 / math.pi) * float(b[-1]) * float(w[-1])
    low_bound = float(2.0 / math.pi) * scale * float(w[0]) ** 4 / 4.0
    return MemoryKernel(
        tau=grid,
        values=values,
        omega_max=float(w[-1]),
        damping_scale=scale,
        low_frequency_model=low_frequency_model,
        high_frequency_bound=high_bound,
        low_frequency_bound=low_bound,
        metadata={"omega_min": float(w[0]), "damping_peak": float(np.max(b))},
    )


def two_sided_kernel(kernel: MemoryKernel) -> tuple[FloatArray, FloatArray]:
    """Even two-sided extension ``K(-tau) = K(tau)`` on ``[-tau_max, tau_max]``."""
    tau = np.concatenate([-kernel.tau[:0:-1], kernel.tau])
    values = np.concatenate([kernel.values[:0:-1], kernel.values])
    return tau, values


def kernel_diagnostics(
    kernel: MemoryKernel, omega: Sequence[float], damping: Sequence[float]
) -> dict:
    """Decay and parity properties of the impulse response.

    ``value_at_zero``
        ``K(0) = (2/pi) int B domega``; positive for a passive mode.
    ``relative_slope_at_zero``
        ``K'(0) * dtau / max|K|``; a cosine transform of a real function has
        ``K'(0) = 0``, so a value of order one flags a sine/phase error.
    ``even_residual``
        ``max|K(tau) - K(-tau)| / max|K|`` on the two-sided extension.
    ``tail_relative``, ``decay_rate``, ``memory_time_scale``
        how much of the kernel is left in the last half of the record, the
        log-envelope decay rate of ``|K|`` between 50% and 1% of its peak, and
        its reciprocal. ``decay_time_to_one_percent`` is the time at which the
        envelope first falls below 1% of the peak, the practical memory
        truncation length.
    ``first_zero``, ``points_per_shortest_period``
        the memory's oscillation scale and the grid resolution against it.
    """
    w, _, b = validate_radiation_sweep(
        omega, np.zeros_like(np.asarray(damping, dtype=float)), damping
    )
    values = kernel.values
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        raise ValueError("the memory kernel is identically zero")
    tau = kernel.tau
    _, two_values = two_sided_kernel(kernel)
    even_residual = float(np.max(np.abs(two_values - two_values[::-1])) / scale)

    slope = (float(values[1]) - float(values[0])) / kernel.dt

    signs = np.signbit(values)
    crossings = np.nonzero(signs[1:] != signs[:-1])[0]
    first_zero = float(tau[crossings[0] + 1]) if crossings.size else math.inf

    late = tau >= 0.5 * kernel.span
    tail_relative = float(np.max(np.abs(values[late])) / scale) if np.any(late) else 1.0

    # Decay rate from the *envelope* of |K|, averaged over one period of the
    # highest frequency in the sweep. Two things make the raw samples useless
    # here: |K| oscillates (a log of a value crossing zero is roundoff), and the
    # far tail of the kernel is at the level of the interpolation error of the
    # sweep, so a fit there measures the sweep spacing, not the physics. The fit
    # therefore uses the envelope between 50% and 1% of its peak -- the region
    # where the kernel is still three decades above the noise -- and the
    # remaining tail is reported separately as ``tail_relative``.
    omega_shortest = kernel.omega_max if kernel.omega_max > 0.0 else kernel.nyquist_omega
    samples_per_period = max(1, int(round(2.0 * math.pi / (omega_shortest * kernel.dt))))
    blocks = values.size // samples_per_period
    envelope = np.zeros(blocks)
    times = np.zeros(blocks)
    for block in range(blocks):
        start_index = block * samples_per_period
        stop_index = start_index + samples_per_period
        envelope[block] = float(np.max(np.abs(values[start_index:stop_index])))
        times[block] = float(tau[start_index:stop_index].mean())
    usable = (envelope <= 0.5 * scale) & (envelope >= 0.01 * scale)
    if np.count_nonzero(usable) >= 3:
        fit = np.polyfit(times[usable], np.log(envelope[usable]), 1)
        decay_rate = float(-fit[0])
        envelope_window = (
            float(times[usable][0]),
            float(times[usable][-1]),
            int(np.count_nonzero(usable)),
        )
    elif np.count_nonzero(usable) == 2:
        pair = np.log(envelope[usable])
        decay_rate = float(-(pair[1] - pair[0]) / (times[usable][1] - times[usable][0]))
        envelope_window = (float(times[usable][0]), float(times[usable][-1]), 2)
    else:
        decay_rate = math.inf
        envelope_window = (0.0, 0.0, 0)
    below_one_percent = np.nonzero(envelope < 0.01 * scale)[0]
    decay_time = (
        float(times[below_one_percent[0]]) if below_one_percent.size else math.inf
    )

    return {
        "value_at_zero": float(values[0]),
        "peak_abs": scale,
        "relative_slope_at_zero": slope * kernel.dt / scale,
        "absolute_slope_at_zero": slope,
        "even_residual": even_residual,
        "first_zero": first_zero,
        "damping_peak": float(np.max(b)),
        "tail_relative": tail_relative,
        "decay_rate": decay_rate,
        "memory_time_scale": (1.0 / decay_rate) if decay_rate > 0 else math.inf,
        "decay_envelope_window": envelope_window,
        "decay_time_to_one_percent": decay_time,
        "points_per_shortest_period": (
            math.pi / (kernel.dt * float(w[-1])) if w[-1] else math.inf
        ),
        "truncation": kernel.truncation,
    }


# ---------------------------------------------------------------------------
# inverse relations (kernel -> B, A)
# ---------------------------------------------------------------------------
def _simpson_weights(count: int) -> FloatArray:
    """Composite Simpson weights for a uniform grid of ``count`` points.

    The step is left out so the same weights serve any sample spacing. An even
    count falls back to Simpson on the leading odd block plus a trapezoid
    panel at the end.
    """
    if count < 2:
        raise ValueError("at least two samples are required for quadrature")
    if count % 2 == 1:
        weights = np.ones(count)
        weights[1:-1:2] = 4.0
        weights[2:-1:2] = 2.0
        return weights / 3.0
    weights = np.zeros(count)
    weights[: count - 1] = _simpson_weights(count - 1)
    weights[-1] += 0.5
    weights[-2] += 0.5
    return weights


def _memory_quadrature(
    kernel: MemoryKernel, omega_query: Sequence[float], function: str
) -> FloatArray:
    query = np.asarray(omega_query, dtype=float)
    if query.ndim != 1:
        raise ValueError("omega_query must be one-dimensional")
    if query.size == 0:
        return np.zeros(0)
    if np.any(query <= 0.0):
        raise ValueError("omega_query must be strictly positive")
    check_frequency_sampling(float(np.max(query)), kernel.span, dtau=kernel.dt)
    weights = _simpson_weights(kernel.tau.size) * kernel.dt
    trig = np.cos if function == "cos" else np.sin
    return trig(np.outer(query, kernel.tau)) @ (weights * kernel.values)


def damping_from_memory_kernel(
    kernel: MemoryKernel, omega_query: Sequence[float]
) -> FloatArray:
    """``B(omega) = int_0^tau_max K(tau) cos(omega tau) dtau`` (eq. 2)."""
    return _memory_quadrature(kernel, omega_query, "cos")


def added_mass_from_memory_kernel(
    kernel: MemoryKernel, omega_query: Sequence[float], added_mass_inf: float
) -> FloatArray:
    """``A(omega) = A_inf - (1/omega) int_0^tau_max K(tau) sin(omega tau) dtau``."""
    query = np.asarray(omega_query, dtype=float)
    sine = _memory_quadrature(kernel, query, "sin")
    return float(added_mass_inf) - sine / query


def implied_added_mass_inf(
    kernel: MemoryKernel, omega: Sequence[float], added_mass: Sequence[float]
) -> dict:
    """Invert eq. (3) for ``A_inf`` at every tabulated frequency.

    Rearranged, eq. (3) gives ``A_inf = A(omega) + (1/omega) int K sin``, which
    must be the *same number at every frequency* if the sweep and the kernel
    describe one causal system. The scatter is therefore a consistency residual
    that does not presuppose ``A_inf``.
    """
    w = np.asarray(omega, dtype=float)
    a = np.asarray(added_mass, dtype=float)
    values = a + _memory_quadrature(kernel, w, "sin") / w
    mean = float(np.mean(values))
    scale = float(np.max(np.abs(a))) if a.size else 0.0
    return {
        "mean": mean,
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "relative_spread": float(np.ptp(values) / abs(mean)) if mean else math.inf,
        "spread_over_added_mass_scale": (
            float(np.ptp(values) / scale) if scale else math.inf
        ),
    }


@dataclass(frozen=True)
class FourierRoundTrip:
    """Result of :func:`fourier_round_trip`."""

    omega_query: FloatArray
    damping_exact: FloatArray
    damping_round_trip: FloatArray
    added_mass_exact: FloatArray
    added_mass_round_trip: FloatArray
    implied_added_mass_inf: dict
    band: tuple[float, float]
    damping_relative_error_max: float
    added_mass_relative_error_max: float
    kernel: MemoryKernel
    diagnostics: dict
    tau_max: float
    dt: float

    @property
    def worst_relative_error(self) -> float:
        """Largest of the two in-band relative errors."""
        return max(self.damping_relative_error_max, self.added_mass_relative_error_max)


def fourier_round_trip(
    omega: Sequence[float],
    added_mass: Sequence[float],
    damping: Sequence[float],
    added_mass_inf: float,
    *,
    dt: float,
    tau_max: float | None = None,
    band: tuple[float, float] | None = None,
    samples: int = 60,
) -> FourierRoundTrip:
    """Transform ``B -> K -> (B, A)`` and report the in-band error.

    The forward leg is an exact piecewise-linear quadrature of eq. (1) on the
    tabulated sweep; the inverse legs are Simpson quadratures of eqs. (2)-(3) on
    a uniform ``tau`` grid. Nothing but the kernel is shared between the legs.

    ``band`` defaults to ``[max(omega_min, 4 pi/tau_max), min(0.6 omega_max,
    pi/(2 dt))]``. The low edge keeps at least two memory periods inside the
    window (below it the sine quadrature of eq. (3) is truncated mid-period and
    the error grows like ``1/omega``), while the upper edge stays clear of the
    truncated high band and of the kernel grid's Nyquist frequency. Both edges
    are properties of the sampling, not of the case, and are written into the
    returned record.
    """
    w, a, b = validate_radiation_sweep(omega, added_mass, damping)
    if not math.isfinite(float(added_mass_inf)):
        raise ValueError("added_mass_inf must be finite")
    step = min(float(dt), math.pi / float(w[-1]))
    span = (
        float(tau_max)
        if tau_max is not None
        else math.pi / float(np.min(np.diff(w)))
    )
    if not (span > 0.0 and math.isfinite(span)):
        raise ValueError("tau_max must be positive and finite")
    count = int(math.ceil(span / step)) + 1
    tau = np.arange(count, dtype=float) * step
    kernel = radiation_memory_kernel(w, b, tau)

    low, high = (
        band
        if band is not None
        else (
            max(float(w[0]), 4.0 * math.pi / span),
            min(0.6 * float(w[-1]), 0.5 * math.pi / step),
        )
    )
    if not (0.0 < low < high):
        raise ValueError("band must satisfy 0 < low < high")
    query = np.linspace(low, high, int(samples))

    round_b = damping_from_memory_kernel(kernel, query)
    round_a = added_mass_from_memory_kernel(kernel, query, added_mass_inf)
    exact_b = np.interp(query, w, b)
    exact_a = np.interp(query, w, a)

    peak_b = float(np.max(b))
    peak_a = float(np.max(np.abs(a)))
    damping_error = float(np.max(np.abs(round_b - exact_b)) / peak_b) if peak_b else math.inf
    added_error = float(np.max(np.abs(round_a - exact_a)) / peak_a) if peak_a else math.inf

    return FourierRoundTrip(
        omega_query=query,
        damping_exact=exact_b,
        damping_round_trip=round_b,
        added_mass_exact=exact_a,
        added_mass_round_trip=round_a,
        implied_added_mass_inf=implied_added_mass_inf(kernel, w, a),
        band=(low, high),
        damping_relative_error_max=damping_error,
        added_mass_relative_error_max=added_error,
        kernel=kernel,
        diagnostics=kernel_diagnostics(kernel, w, b),
        tau_max=span,
        dt=step,
    )


# ---------------------------------------------------------------------------
# frequency-domain response of the same memory system
# ---------------------------------------------------------------------------
def frequency_domain_impedance(
    omega_query: Sequence[float],
    omega: Sequence[float],
    added_mass: Sequence[float],
    damping: Sequence[float],
    *,
    mass: float,
    restoring: float,
) -> NDArray[np.complex128]:
    """Mechanical impedance ``kappa(omega) = C - (M + A(omega)) omega^2 + i omega B(omega)``.

    The response to a harmonic force ``F0 exp(i omega t)`` is ``F0 / kappa``.
    This is the exact harmonic form of the Cummins equation with the same
    ``(A, B)`` sweep: on the imaginary axis ``Khat(i omega) = B + i omega (A - A_inf)``,
    so the memory term plus the instantaneous added mass combine to
    ``omega^2 A(omega)`` (the Kramers-Kronig pairing).
    """
    w = np.asarray(omega_query, dtype=float)
    if np.any(w <= 0.0):
        raise ValueError("omega_query must be strictly positive")
    grid, a, b = validate_radiation_sweep(omega, added_mass, damping)
    added = np.interp(w, grid, a)
    damp = np.interp(w, grid, b)
    return float(restoring) - (float(mass) + added) * w * w + 1.0j * w * damp


def resonance_properties(
    omega: Sequence[float],
    added_mass: Sequence[float],
    damping: Sequence[float],
    *,
    mass: float,
    restoring: float,
    samples: int = 4000,
) -> dict:
    """Resonance and equivalent damping of the frequency-domain response.

    Solves ``Re kappa(omega_n) = 0`` (the point where the inertia of the body
    plus the frequency-dependent added mass is balanced by the restoring
    stiffness), then reports

    * ``zeta`` from the energy identity ``zeta = B/(2 (M + A) omega_n)``;
    * ``decay_rate = zeta * omega_n`` and ``damped_period = 2 pi/(omega_n sqrt(1-zeta^2))``;
    * ``zeta_bandwidth`` from the half-power bandwidth of ``|1/kappa|``, an
      amplitude-based estimate that does not assume the energy identity.

    Both estimates are reported because a decay test measures the amplitude
    envelope (bandwidth-like) whereas the energy identity is what a linear
    seakeeping code would print; the gap between them is the size of the
    memory effect.
    """
    grid, a, b = validate_radiation_sweep(omega, added_mass, damping)
    low, high = float(grid[0]), float(grid[-1])
    scan = np.geomspace(max(low, 1.0e-3), high, int(samples))
    real = frequency_domain_impedance(
        scan, grid, a, b, mass=mass, restoring=restoring
    ).real
    crossings = np.nonzero(np.signbit(real[1:]) != np.signbit(real[:-1]))[0]
    if crossings.size == 0:
        raise ValueError(
            "no resonance inside the tabulated band: Re kappa does not change sign"
        )
    index = crossings[0]
    lo, hi = float(scan[index]), float(scan[index + 1])
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        value = frequency_domain_impedance(
            [mid], grid, a, b, mass=mass, restoring=restoring
        ).real[0]
        if np.signbit(value) == np.signbit(
            frequency_domain_impedance([lo], grid, a, b, mass=mass, restoring=restoring).real[0]
        ):
            lo = mid
        else:
            hi = mid
    omega_n = 0.5 * (lo + hi)
    added_n = float(np.interp(omega_n, grid, a))
    damp_n = float(np.interp(omega_n, grid, b))
    inertia = float(mass) + added_n
    zeta = damp_n / (2.0 * inertia * omega_n)

    amplitude = np.abs(
        1.0
        / frequency_domain_impedance(scan, grid, a, b, mass=mass, restoring=restoring)
    )
    peak = int(np.argmax(amplitude))
    peak_amplitude = float(amplitude[peak])
    half = np.nonzero(amplitude >= peak_amplitude / math.sqrt(2.0))[0]
    if half.size >= 2:
        omega_lo, omega_hi = float(scan[half[0]]), float(scan[half[-1]])
        zeta_bandwidth = (omega_hi - omega_lo) / (2.0 * omega_n)
    else:
        omega_lo = omega_hi = omega_n
        zeta_bandwidth = float("nan")
    damped = omega_n * math.sqrt(max(1.0 - zeta * zeta, 0.0))
    return {
        "omega_n": omega_n,
        "added_mass_at_resonance": added_n,
        "damping_at_resonance": damp_n,
        "inertia": inertia,
        "zeta": zeta,
        "decay_rate": zeta * omega_n,
        "damped_period": (2.0 * math.pi / damped) if damped > 0.0 else math.inf,
        "rao_peak_omega": float(scan[peak]),
        "rao_peak_amplitude": peak_amplitude,
        "half_power_omega_low": omega_lo,
        "half_power_omega_high": omega_hi,
        "zeta_bandwidth": zeta_bandwidth,
        "resonance_scan_range": (low, high),
    }


# ---------------------------------------------------------------------------
# time-domain integration of the Cummins equation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FreeDecayResult:
    """Sampled free decay of the Cummins equation."""

    time: FloatArray
    displacement: FloatArray
    velocity: FloatArray
    dt: float
    added_mass_inf: float
    mass: float
    restoring: float
    natural_frequency: float
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return float(self.time[-1])


def _integrate_cummins(
    kernel: MemoryKernel,
    *,
    mass: float,
    restoring: float,
    added_mass_inf: float,
    initial_displacement: float,
    initial_velocity: float,
    dt: float,
    duration: float,
    force: tuple[float, float] | None = None,
    sample_stride: int = 1,
) -> FreeDecayResult:
    """Classical RK4 on the Cummins equation with a trapezoidal memory integral.

    The memory integral is accumulated on the uniform step grid; the four RK4
    stages need the kernel at half steps, so the kernel is pre-sampled on a
    ``dt/2`` grid and each stage memory sum is a strided slice. Cost is
    ``O(N^2)`` in the number of steps, which is the price of not fitting a
    rational approximation to ``Khat(s)``: the convolution is evaluated as
    written, so an error in the kernel cannot hide behind a fitted state-space
    model.

    ``force`` optionally supplies ``(amplitude, omega)`` of ``F(t) = A cos(omega t)``.
    """
    if not (dt > 0.0):
        raise ValueError("dt must be positive")
    if not (duration > 0.0):
        raise ValueError("duration must be positive")
    check_time_sampling(dt, kernel.omega_max if kernel.omega_max else kernel.nyquist_omega)
    inertia = float(mass) + float(added_mass_inf)
    if inertia <= 0.0:
        raise ValueError("mass + added_mass_inf must be positive")

    steps = int(round(duration / dt))
    if steps < 8:
        raise ValueError("duration/dt must give at least 8 integration steps")
    half = kernel(np.arange(0.0, steps * dt + dt, 0.5 * dt))
    if half.size < 2 * steps + 3:
        pad = np.zeros(2 * steps + 3)
        pad[: half.size] = half
        half = pad
    h0 = float(half[0])

    velocity = np.zeros(steps + 1)
    displacement = np.zeros(steps + 1)
    displacement[0] = float(initial_displacement)
    velocity[0] = float(initial_velocity)
    amplitude, drive = force if force is not None else (0.0, 0.0)

    def memory(index: int, stage: int, stage_velocity: float) -> float:
        """Trapezoidal memory integral at ``t = index*dt`` advanced by ``stage*dt/2``."""
        base = 2 * index + stage
        values = half[base::-2][: index + 1]
        total = 0.0
        if index > 0:
            total = 0.5 * values[0] * velocity[0]
            if index > 1:
                total += float(values[1:index] @ velocity[1:index])
            total += 0.5 * values[index] * velocity[index]
            total *= dt
        partial = 0.5 * stage * dt * (values[index] * velocity[index] + h0 * stage_velocity)
        return total + partial

    for index in range(steps):
        t = index * dt
        external = amplitude * math.cos(drive * t) if amplitude else 0.0
        a1 = (external - memory(index, 0, velocity[index]) - restoring * displacement[index]) / inertia
        v1 = velocity[index] + 0.5 * dt * a1
        z1 = displacement[index] + 0.5 * dt * velocity[index]
        external_mid = amplitude * math.cos(drive * (t + 0.5 * dt)) if amplitude else 0.0
        a2 = (external_mid - memory(index, 1, v1) - restoring * z1) / inertia
        v2 = velocity[index] + 0.5 * dt * a2
        z2 = displacement[index] + 0.5 * dt * v1
        a3 = (external_mid - memory(index, 1, v2) - restoring * z2) / inertia
        v3 = velocity[index] + dt * a3
        z3 = displacement[index] + dt * v2
        external_end = amplitude * math.cos(drive * (t + dt)) if amplitude else 0.0
        a4 = (external_end - memory(index, 2, v3) - restoring * z3) / inertia
        displacement[index + 1] = displacement[index] + dt / 6.0 * (
            velocity[index] + 2.0 * v1 + 2.0 * v2 + v3
        )
        velocity[index + 1] = velocity[index] + dt / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)

    return FreeDecayResult(
        time=np.arange(steps + 1, dtype=float) * dt,
        displacement=displacement,
        velocity=velocity,
        dt=float(dt),
        added_mass_inf=float(added_mass_inf),
        mass=float(mass),
        restoring=float(restoring),
        natural_frequency=math.sqrt(float(restoring) / inertia),
        metadata={"kernel_tau_max": kernel.span, "steps": steps},
    )


def cummins_free_decay(
    kernel: MemoryKernel,
    *,
    mass: float,
    restoring: float,
    added_mass_inf: float,
    initial_displacement: float = 1.0,
    dt: float,
    duration: float,
) -> FreeDecayResult:
    """Free decay of the Cummins equation from rest at ``z(0) = initial_displacement``."""
    return _integrate_cummins(
        kernel,
        mass=mass,
        restoring=restoring,
        added_mass_inf=added_mass_inf,
        initial_displacement=initial_displacement,
        initial_velocity=0.0,
        dt=dt,
        duration=duration,
    )


def harmonic_response(
    kernel: MemoryKernel,
    *,
    mass: float,
    restoring: float,
    added_mass_inf: float,
    force_amplitude: float,
    omega: float,
    dt: float,
    duration: float,
    warmup_fraction: float = 0.6,
) -> dict:
    """Steady-state response to ``F(t) = force_amplitude cos(omega t)``.

    Returns the amplitude and phase of the last ``1 - warmup_fraction`` of the
    record, measured by projecting onto ``cos``/``sin`` over an integer number
    of drive periods. This is the time-domain counterpart of
    ``force_amplitude / kappa(omega)`` and therefore closes the loop between the
    convolution and the frequency-domain impedance.
    """
    if not (0.0 <= warmup_fraction < 1.0):
        raise ValueError("warmup_fraction must lie in [0, 1)")
    if not (omega > 0.0 and force_amplitude != 0.0):
        raise ValueError("omega and force_amplitude must be non-zero")
    result = _integrate_cummins(
        kernel,
        mass=mass,
        restoring=restoring,
        added_mass_inf=added_mass_inf,
        initial_displacement=0.0,
        initial_velocity=0.0,
        dt=dt,
        duration=duration,
        force=(float(force_amplitude), float(omega)),
    )
    period = 2.0 * math.pi / omega
    window = result.time >= warmup_fraction * duration
    t = result.time[window]
    z = result.displacement[window]
    if t.size < 8:
        raise ValueError("the steady-state window is too short")
    # Least-squares projection on [cos(omega t), sin(omega t), 1, t - t0]: the
    # constant and the linear term absorb whatever is left of the transient (and
    # the algebraic tail of the memory), so a fractional number of cycles in the
    # window does not leak into the amplitude or the phase.
    tau = t - t[0]
    basis = np.column_stack(
        [np.cos(omega * t), np.sin(omega * t), tau, np.ones_like(tau)]
    )
    coefficients, *_ = np.linalg.lstsq(basis, z, rcond=None)
    cosine, sine = float(coefficients[0]), float(coefficients[1])
    amplitude = math.hypot(cosine, sine)
    phase = math.atan2(-sine, cosine)
    modelled = basis @ coefficients
    span = float(t[-1] - t[0])
    return {
        "amplitude": amplitude,
        "phase": phase,
        "omega": float(omega),
        "cycles": span / period,
        "window": (float(t[0]), float(t[-1])),
        "fit_residual_over_amplitude": float(
            np.max(np.abs(modelled - z)) / amplitude
        ) if amplitude else math.inf,
        "result": result,
    }


def decay_metrics(
    time: Sequence[float],
    signal: Sequence[float],
    *,
    amplitude_fraction: float = 1.0e-3,
    minimum_peaks: int = 3,
) -> dict:
    """Period and decay rate of a damped oscillation.

    The period is taken from upward zero crossings (linear interpolation), the
    decay rate from a least-squares fit of ``log|extremum|`` against time over
    the extrema that survive the amplitude floor -- extrema below the floor are
    roundoff, and fitting them would report the noise floor as a decay rate.
    """
    t = np.asarray(time, dtype=float)
    z = np.asarray(signal, dtype=float)
    if t.size != z.size or t.size < 4:
        raise ValueError("time and signal must have equal length >= 4")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("time must be strictly increasing")
    peak = float(np.max(np.abs(z)))
    if peak == 0.0:
        raise ValueError("the signal is identically zero")
    floor = amplitude_fraction * peak

    upward = np.nonzero((z[:-1] < 0.0) & (z[1:] >= 0.0))[0]
    if upward.size >= 2:
        crossings = t[upward] - z[upward] * (t[1] - t[0]) / (z[upward + 1] - z[upward])
        period = float(np.mean(np.diff(crossings)))
    else:
        period = math.inf

    extrema = [
        index
        for index in range(1, z.size - 1)
        if abs(z[index]) > abs(z[index - 1]) and abs(z[index]) >= abs(z[index + 1])
    ]
    kept = [i for i in extrema if abs(z[i]) > floor and t[i] > 0.0]
    if len(kept) < minimum_peaks:
        raise ValueError(
            f"only {len(kept)} extrema above the amplitude floor; cannot fit a decay"
        )
    times = t[kept]
    magnitudes = np.abs(z[kept])
    fit = np.polyfit(times, np.log(magnitudes), 1)
    predicted = np.polyval(fit, times)
    residual = np.log(magnitudes) - predicted
    total = np.log(magnitudes) - np.mean(np.log(magnitudes))
    r_squared = 1.0 - float(np.sum(residual**2) / np.sum(total**2))

    return {
        "period": period,
        "frequency": (2.0 * math.pi / period) if math.isfinite(period) else math.nan,
        "decay_rate": float(-fit[0]),
        "log_amplitude_intercept": float(fit[1]),
        "extrema_used": len(kept),
        "extrema_total": len(extrema),
        "amplitude_ratio_first_last": float(magnitudes[0] / magnitudes[-1]),
        "fit_r_squared": r_squared,
        "amplitude_fraction_floor": amplitude_fraction,
        "signal_peak": peak,
    }


# ---------------------------------------------------------------------------
# resolved half-power damping and the constant-added-mass error envelope
# ---------------------------------------------------------------------------
def _local_response_scan(
    omega: Sequence[float],
    added_mass: Sequence[float],
    damping: Sequence[float],
    *,
    mass: float,
    restoring: float,
    centre: float,
    half_width: float,
    samples: int,
) -> tuple[FloatArray, FloatArray]:
    """``|1/kappa|`` on a uniform grid of ``samples`` points around ``centre``."""
    grid, a, b = validate_radiation_sweep(omega, added_mass, damping)
    low, high = float(grid[0]), float(grid[-1])
    width = float(half_width)
    if not (width > 0.0 and math.isfinite(width)):
        raise ValueError(f"half_width must be positive and finite, got {half_width!r}")
    count = max(int(samples), 9)
    scan = np.linspace(centre - width, centre + width, count)
    if scan[0] <= 0.0 or scan[-1] > high:
        raise ValueError(
            "the local scan leaves the tabulated band "
            f"[{low:.6g}, {high:.6g}] rad/s; narrow half_width={width:.6g} "
            f"around omega={centre:.6g}"
        )
    kappa = frequency_domain_impedance(
        scan, grid, a, b, mass=mass, restoring=restoring
    )
    return scan, np.abs(1.0 / kappa)


def _half_power_crossing(
    scan: FloatArray, amplitude: FloatArray, level: float, *, rising: bool
) -> float:
    """Frequency where ``amplitude`` crosses ``level``, linearly interpolated.

    A crossing has to be *bracketed inside the scan window*: the previous
    implementation clamped a crossing that fell outside the window to
    ``scan[0]`` on the rising side (a silently low ``zeta``) and raised a bare
    ``IndexError`` on the descending side (``index 0 is out of bounds for axis 0
    with size 0``, which is exactly the verdict the Cummins/Wigley case recorded
    for its G4 gate).  Both cases now raise a ``ValueError`` that names the
    window, so the caller can widen the window instead of reporting a number.
    """
    values = np.asarray(amplitude, dtype=float)
    excess = values - float(level)
    if rising:
        index = int(np.nonzero(excess >= 0.0)[0][0]) if np.any(excess >= 0.0) else -1
        if index < 0:
            raise ValueError(
                f"the half-power level {level:.6g} is never reached on the "
                f"rising side of the scan (peak {float(np.max(values)):.6g})")
        if index == 0:
            raise ValueError(
                f"the lower half-power point lies below the scan window: the "
                f"first sample {float(values[0]):.6g} is already at or above "
                f"level {level:.6g}")
    else:
        index = int(np.nonzero(excess < 0.0)[0][0]) if np.any(excess < 0.0) else -1
        if index < 0:
            raise ValueError(
                f"the half-power level {level:.6g} is never crossed on the "
                f"descending side of the scan (last sample "
                f"{float(values[-1]):.6g})")
        if index == 0:
            raise ValueError(
                f"the upper half-power point lies above the scan window: the "
                f"first sample {float(values[0]):.6g} is already below level "
                f"{level:.6g}")
    left, right = index - 1, index
    weight = excess[left] / (excess[left] - excess[right])
    return float(scan[left] + weight * (scan[right] - scan[left]))


def half_power_damping(
    omega: Sequence[float],
    added_mass: Sequence[float],
    damping: Sequence[float],
    *,
    mass: float,
    restoring: float,
    omega_n: float | None = None,
    width_scale: float = 40.0,
    samples: int = 4001,
) -> dict:
    """Half-power damping ratio of ``1/kappa`` from a *locally refined* scan.

    :func:`resonance_properties` scans the whole tabulated band on a geometric
    grid, which is enough when ``zeta`` is of order 0.1 but *under-resolves* the
    half-power width when the resonance is lightly damped: the width
    ``2 zeta omega_n`` can span only a handful of scan points, and the reported
    ``zeta_bandwidth`` then comes out systematically low (the two crossings are
    quantised to the scan spacing). The estimator here builds a uniform grid of
    ``samples`` points over ``omega_n (1 +- width_scale*zeta_0)``, with ``zeta_0``
    the energy-identity estimate, and interpolates both crossings, so the answer
    no longer depends on the global scan.

    Returns the crossing frequencies, the local peak, ``zeta`` from the
    half-power width, and ``zeta_energy`` from ``B/(2 (M + A) omega)`` at the
    same frequency. Both are reported: a decay test measures the former and a
    linear seakeeping code prints the latter.
    """
    grid, a, b = validate_radiation_sweep(omega, added_mass, damping)
    high = float(grid[-1])
    if omega_n is None:
        omega_n = float(
            resonance_properties(grid, a, b, mass=mass, restoring=restoring)["omega_n"]
        )
    centre = float(omega_n)
    if not (0.0 < centre <= high):
        raise ValueError("omega_n must lie inside the tabulated band")
    added = float(np.interp(centre, grid, a))
    damp = float(np.interp(centre, grid, b))
    inertia = float(mass) + added
    if not (inertia > 0.0 and centre > 0.0):
        raise ValueError("mass + added_mass must be positive")
    zeta_guess = damp / (2.0 * inertia * centre)

    base_width = float(width_scale) * zeta_guess * centre
    if not (base_width > 0.0):
        raise ValueError("the energy-identity damping is zero; no half-power width")
    cap = min(0.9 * centre, 0.9 * (high - centre))
    width = min(base_width, cap)
    # The window is sized by the *energy-identity* zeta, which is itself read off
    # the tabulated sweep.  Where the sweep under-resolves the damping peak (a
    # light resonance sampled by a coarse frequency grid), that guess is too
    # small and the half-power points fall outside the window.  Widen by x4 and
    # retry instead of reporting a clipped crossing; the number of widenings is
    # part of the answer, because it is a direct measure of how badly the sweep
    # resolved the width.
    widenings = 0
    attempts: list[dict] = []
    while True:
        scan, amplitude = _local_response_scan(
            grid, a, b, mass=mass, restoring=restoring,
            centre=centre, half_width=width, samples=samples,
        )
        peak_index = int(np.argmax(amplitude))
        peak_amplitude = float(amplitude[peak_index])
        peak_omega = float(scan[peak_index])
        level = peak_amplitude / math.sqrt(2.0)
        try:
            lower = _half_power_crossing(
                scan[: peak_index + 1], amplitude[: peak_index + 1], level, rising=True)
            upper = _half_power_crossing(
                scan[peak_index:], amplitude[peak_index:], level, rising=False)
            break
        except ValueError as exc:
            attempts.append({"scan_half_width": width, "error": str(exc)})
            if width >= cap * (1.0 - 1e-12):
                raise ValueError(
                    "the half-power points are not bracketed even with the widest "
                    f"admissible window {width:.6g} rad/s around omega_n="
                    f"{centre:.6g} (band [{float(grid[0]):.6g}, {high:.6g}]); "
                    f"attempts={attempts}") from None
            widenings += 1
            width = min(width * 4.0, cap)
    zeta = (upper - lower) / (2.0 * peak_omega)
    added_peak = float(np.interp(peak_omega, grid, a))
    damp_peak = float(np.interp(peak_omega, grid, b))
    return {
        "omega_n": centre,
        "peak_omega": peak_omega,
        "peak_amplitude": peak_amplitude,
        "half_power_omega_low": lower,
        "half_power_omega_high": upper,
        "zeta": zeta,
        "zeta_energy": damp_peak / (2.0 * (float(mass) + added_peak) * peak_omega),
        "half_power_ratio": level / peak_amplitude,
        "scan_half_width": width,
        "scan_width_over_half_power_width": width / max(upper - lower, 1e-30),
        "scan_widenings": widenings,
        "scan_attempts": attempts,
        "samples": int(max(int(samples), 9)),
        "added_mass_at_peak": added_peak,
        "damping_at_peak": damp_peak,
    }


@dataclass(frozen=True)
class ConstantAddedMassEnvelope:
    """Error of the constant-added-mass model against the full memory kernel.

    ``ratios`` is ``omega_p / omega_n``: the damping-peak frequency of the
    radiation sweep over the oscillator's natural frequency. ``omega_n`` is
    built from ``A_inf`` only, so the sweep isolates what the memory does to an
    oscillator whose resonance is placed at a prescribed distance from the
    radiation band.
    """

    ratios: FloatArray
    natural_frequencies: FloatArray
    peak_amplitude_error: FloatArray
    peak_frequency_error: FloatArray
    phase_error_degrees: FloatArray
    damping_ratio_error: FloatArray
    added_mass_relative_change: FloatArray
    zeta_exact: FloatArray
    zeta_model: FloatArray
    exact_peak_amplitude: FloatArray
    exact_peak_omega: FloatArray
    approximate_peak_amplitude: FloatArray
    omega_p: float
    added_mass_inf: float
    mass: float

    def worst(self) -> dict:
        """Worst-case error and the ratio at which it occurs."""
        def _worst(values: FloatArray) -> tuple[float, float]:
            index = int(np.argmax(np.abs(values)))
            return float(np.abs(values[index])), float(self.ratios[index])

        amplitude, ratio_amplitude = _worst(self.peak_amplitude_error)
        phase, ratio_phase = _worst(self.phase_error_degrees)
        frequency, ratio_frequency = _worst(self.peak_frequency_error)
        return {
            "peak_amplitude_error": amplitude,
            "peak_amplitude_error_ratio": ratio_amplitude,
            "phase_error_degrees": phase,
            "phase_error_ratio": ratio_phase,
            "peak_frequency_error": frequency,
            "peak_frequency_error_ratio": ratio_frequency,
        }


def constant_added_mass_error_envelope(
    omega: Sequence[float],
    added_mass: Sequence[float],
    damping: Sequence[float],
    *,
    mass: float,
    ratios: Sequence[float],
    added_mass_inf: float | None = None,
    samples: int = 4001,
) -> ConstantAddedMassEnvelope:
    """Peak and phase error of the ``A = A_inf``, ``B = B(omega_n)`` model.

    The cheapest time-domain radiation model replaces the convolution by
    ``A_inf x'' + B(omega_n) x'``: it keeps one added mass and one damping
    coefficient for the whole record. That model is exact only if ``A`` is
    constant and ``B`` is flat across the response band, so its error is a
    function of where the oscillator sits relative to the radiation sweep --
    which is what this envelope measures.

    For each ratio ``omega_p/omega_n`` (``omega_p`` = frequency of the peak of
    ``B``) the restoring stiffness is chosen as ``(M + A_inf) omega_n^2``, so
    ``omega_n`` is the model's *own* natural frequency and the approximation
    error is not contaminated by a stiffness mistuning. Then, on a locally
    refined scan:

    * ``peak_amplitude_error = |A_approx - A_exact| / A_exact`` at the response
      peak,
    * ``peak_frequency_error`` = distance between the exact resonance and
      ``omega_n`` (the model pins its peak at ``omega_n``),
    * ``phase_error_degrees`` = phase difference at ``omega_n``,
    * ``damping_ratio_error`` = ``|zeta_exact - zeta_model| / zeta_exact``.

    The envelope is a property of the sweep, not of one body: a sweep whose
    damping peak sits far from the oscillator frequency is safe to collapse to
    constants, one that straddles it is not.
    """
    grid, a, b = validate_radiation_sweep(omega, added_mass, damping)
    high = float(grid[-1])
    target = np.asarray(ratios, dtype=float).reshape(-1)
    if target.size == 0:
        raise ValueError("at least one frequency ratio is required")
    if np.any(target <= 0.0) or not np.all(np.isfinite(target)):
        raise ValueError("ratios must be positive and finite")
    if added_mass_inf is None:
        added_mass_inf = fit_added_mass_asymptote(grid, a).added_mass_inf
    added_mass_inf = float(added_mass_inf)
    if not math.isfinite(added_mass_inf):
        raise ValueError("added_mass_inf must be finite")
    mass = float(mass)
    if not (mass + added_mass_inf > 0.0):
        raise ValueError("mass + added_mass_inf must be positive")

    peak_damping_index = int(np.argmax(b))
    omega_p = float(grid[peak_damping_index])
    if omega_p <= 0.0:
        raise ValueError("the damping sweep has no positive peak")

    natural = np.sort(omega_p / target)
    ratios_sorted = omega_p / natural
    amplitudes = np.zeros(natural.size)
    phases = np.zeros(natural.size)
    frequencies = np.zeros(natural.size)
    zeta_errors = np.zeros(natural.size)
    change = np.zeros(natural.size)
    approximate = np.zeros(natural.size)
    zeta_exact_values = np.zeros(natural.size)
    zeta_model_values = np.zeros(natural.size)

    for index, omega_n in enumerate(natural):
        restoring = (mass + added_mass_inf) * omega_n ** 2
        added = float(np.interp(omega_n, grid, a))
        damp = float(np.interp(omega_n, grid, b))
        change[index] = abs(added - added_mass_inf) / abs(added_mass_inf)
        zeta_exact = damp / (2.0 * (mass + added) * omega_n)
        zeta_model = damp / (2.0 * (mass + added_mass_inf) * omega_n)
        zeta_errors[index] = (
            abs(zeta_model - zeta_exact) / zeta_exact if zeta_exact else math.inf
        )
        zeta_exact_values[index] = zeta_exact
        zeta_model_values[index] = zeta_model
        width = max(25.0 * zeta_exact, 1.0e-4) * omega_n
        width = min(width, 0.5 * omega_n, 0.9 * (high - omega_n))
        if width <= 0.0:
            raise ValueError(
                f"omega_n={omega_n:.6g} leaves no room for a local scan inside the band"
            )
        scan, response = _local_response_scan(
            grid, a, b, mass=mass, restoring=restoring,
            centre=omega_n, half_width=width, samples=samples,
        )
        peak = int(np.argmax(response))
        amplitudes[index] = float(response[peak])
        frequencies[index] = float(scan[peak])
        approximate[index] = 1.0 / (omega_n * damp) if damp > 0.0 else math.inf
        kappa = frequency_domain_impedance(
            [omega_n], grid, a, b, mass=mass, restoring=restoring
        )[0]
        phases[index] = math.degrees(math.atan2(-kappa.imag, kappa.real))

    if np.any(approximate == math.inf):
        raise ValueError("the damping vanishes at a requested natural frequency")
    return ConstantAddedMassEnvelope(
        ratios=ratios_sorted,
        natural_frequencies=natural,
        peak_amplitude_error=(approximate - amplitudes) / amplitudes,
        peak_frequency_error=(frequencies - natural) / frequencies,
        phase_error_degrees=phases + 90.0,
        damping_ratio_error=zeta_errors,
        added_mass_relative_change=change,
        zeta_exact=zeta_exact_values,
        zeta_model=zeta_model_values,
        exact_peak_amplitude=amplitudes,
        exact_peak_omega=frequencies,
        approximate_peak_amplitude=approximate,
        omega_p=omega_p,
        added_mass_inf=added_mass_inf,
        mass=mass,
    )
