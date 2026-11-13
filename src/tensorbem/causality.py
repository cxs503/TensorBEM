"""Kramers-Kronig (causality) consistency of radiation coefficients.

Every frequency-domain radiation solution is a causal linear system: the force
acting on a body at time ``t`` can depend only on its motion at times ``<= t``.
Causality is therefore not an extra physical assumption to be checked against
an experiment; it is a structural property that *any* correct solution must
obey, and it is the only cheap check that ties a whole frequency sweep
together. A single added-mass or damping value can be wrong in the third digit
and still look plausible; a wrong Green function, a mis-set radiation
condition, or a solution contaminated by an irregular frequency shows up as a
*pair* of curves that cannot both come from one causal impulse response.

Formulation
-----------

Let the radiation force on a body in harmonic motion of velocity
``Re(V exp(i*omega*t))`` be ``-(B(omega) + i*omega*A(omega)) V``, with the
memory-function convention

    F(t) = -d/dt integral_0^inf K(s) x(t-s) ds,     K real, K(s) = 0 for s < 0.

The transform ``Khat(omega) = K_r + i*K_i`` of a real, causal, decaying memory
kernel has the cosine/sine representation

    K_r(omega) = B(omega)                                    = int_0^inf K(s) cos(omega s) ds
    K_i(omega) = omega * (A(omega) - A_inf)                   = -int_0^inf K(s) sin(omega s) ds

so ``K_r`` is even, ``K_i`` is odd, and ``Khat`` is analytic in the lower
half-plane with ``Khat -> 0`` at infinity (the instantaneous part of the added
mass is the delta-function contribution that is removed by ``A_inf``). Closing
the contour there for real ``omega`` gives

    PV integral_{-inf}^{inf} K_i(s)/(s-omega) ds = -pi * K_r(omega) .

Reducing the half-line integrand with the parity of ``K_i``,

    integral_{-inf}^{inf} K_i(s)/(s-omega) ds = integral_0^inf K_i(s) [1/(s-omega) + 1/(s+omega)] ds
                                              = integral_0^inf K_i(s) * 2s/(s^2-omega^2) ds ,

so with ``K_i = omega*(A - A_inf)``:

    B(omega) = -(2/pi) PV integral_0^inf s^2 (A(s)-A_inf) / (s^2-omega^2) ds        (1)

    A(omega) - A_inf = (2/pi) PV integral_0^inf B(s) / (s^2-omega^2) ds             (2)

Eq. (1) predicts the damping from the added-mass sweep; eq. (2) predicts the
added mass from the damping sweep. They are the hydrodynamic Kramers-Kronig
relations (Kotik & Mangulis 1962; Newman 1977, *Marine Hydrodynamics*, ch. 6)
and are the two checks reported by :func:`kkr_check`.

Both relations were validated against exact causal pairs before being applied
to panel solutions: for a memory kernel that is a sum of exponentials,
``K(s) = sum_j c_j exp(-a_j s)``, the coefficients are
``B = sum_j c_j a_j/(a_j^2+omega^2)`` and ``A - A_inf = -sum_j c_j/(a_j^2+omega^2)``
and both relations hold identically (the ``h_a(s) = 1/(a^2+s^2)`` family is
closed under these kernels). ``tests/test_causality.py`` locks that identity
down, so a residual reported for a panel solution is a statement about the
solution, not about the transform.

Numerics
--------

The sweep is treated as a tabulated function and interpolated linearly between
the tabulated frequencies; the principal-value integrals are then evaluated
*exactly* for that piecewise-linear interpolant, interval by interval. On an
interval ``[s1, s2]`` where ``g(s) = a + b*s``,

    g(s)/(s^2-omega^2) = (a+b*s)/(s^2-omega^2)

has the elementary antiderivative used below, and the two logarithmic terms it
produces are individually singular when the pole ``s = omega`` falls inside the
interval while their sum is finite -- the piecewise-linear model has a jump in
slope at the pole and the divergences cancel exactly in the given
combination. That cancellation is why a sample frequency must fall strictly
*inside* a grid interval: with the pole exactly at a node the cancellation is
incomplete and leaves a residual of order ``omega*(b_left-b_right)*ln(eps)``.
Passing such a frequency raises ``ValueError`` rather than returning a wrong
number.

The finite ends of the sweep are closed analytically. Above the last node the
two-term asymptote ``A(s) - A_inf -> c/s^2`` is used (``A_inf`` and ``c`` are
fitted by least squares on the upper tail,
:func:`fit_added_mass_asymptote`); below the first node the tabulated value is
held constant. For a body whose coefficients decay *exponentially* (any deeply
submerged section, where the free-surface term behaves like ``exp(-2*k*f)``)
the fit returns ``c ~ 0`` and ``A_inf`` equal to the flat plateau, which is the
correct behaviour, and both truncations then contribute far below the
discretisation error. The one genuinely accuracy-limiting case is a low sample
frequency: the omitted band ``s < s_min`` contributes about
``g(s_min)/(omega^2) * s_min`` to (1), so the sweep must reach well below the
lowest frequency of interest.

Nothing here assembles a matrix or shares code with the solvers: the module
takes a tabulated ``(omega, A, B)`` triple from any source -- a panel solution,
an analytical reference, a measurement -- and reports how far that triple is
from being causal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

__all__ = [
    "AsymptoteFit",
    "KKRCheck",
    "fit_added_mass_asymptote",
    "kkr_added_mass",
    "kkr_check",
    "kkr_damping",
    "principal_value_integral",
    "PowerTailFit",
    "fit_damping_power_tail",
    "power_tail_principal_value",
    "KKRClosure",
    "kk_added_mass_closure",
]

_MINIMUM_POINTS = 8


def _as_sweep(
    omegas: Sequence[float] | FloatArray,
    values: Sequence[float] | FloatArray,
    name: str,
) -> tuple[FloatArray, FloatArray]:
    """Validate a tabulated frequency sweep; return float copies."""
    grid = np.asarray(omegas, dtype=float).reshape(-1)
    data = np.asarray(values, dtype=float).reshape(-1)
    if grid.size != data.size:
        raise ValueError(f"omegas and {name} must have the same length")
    if grid.size < _MINIMUM_POINTS:
        raise ValueError(f"at least {_MINIMUM_POINTS} frequencies are required")
    if not np.all(np.isfinite(grid)) or not np.all(np.isfinite(data)):
        raise ValueError(f"omegas and {name} must be finite")
    if np.any(grid <= 0.0):
        raise ValueError("omegas must be strictly positive")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("omegas must be strictly increasing")
    return grid, data


@dataclass(frozen=True)
class AsymptoteFit:
    """Least-squares fit of ``A(s) = A_inf + c/s^2`` over the upper tail."""

    added_mass_inf: float
    tail_coefficient: float
    points: int
    relative_residual: float


def fit_added_mass_asymptote(
    omegas: Sequence[float] | FloatArray,
    added_mass: Sequence[float] | FloatArray,
    *,
    tail_fraction: float = 0.25,
) -> AsymptoteFit:
    """Fit the two-term high-frequency asymptote of the added mass.

    ``A_inf`` is the one piece of information the damping prediction (1) needs
    but a frequency sweep does not contain directly, so it is extrapolated from
    the upper tail. ``c`` is the coefficient of the ``1/s^2`` term; it either
    carries the true algebraic decay or absorbs an exponential tail as
    ``c ~ 0``.
    """
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    grid, data = _as_sweep(omegas, added_mass, "added_mass")
    count = max(4, int(round(float(tail_fraction) * grid.size)))
    count = min(count, grid.size)
    tail_grid = grid[-count:]
    tail_data = data[-count:]

    design = np.column_stack((np.ones_like(tail_grid), 1.0 / tail_grid**2))
    coefficients, *_ = np.linalg.lstsq(design, tail_data, rcond=None)
    residual = design @ coefficients - tail_data
    scale = max(float(np.max(np.abs(tail_data))), 1e-300)
    return AsymptoteFit(
        added_mass_inf=float(coefficients[0]),
        tail_coefficient=float(coefficients[1]),
        points=count,
        relative_residual=float(np.max(np.abs(residual)) / scale),
    )


def _pole_interval(grid: FloatArray, omega: float) -> int:
    """Index of the interval that contains ``omega`` strictly in its interior."""
    if not math.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be positive and finite")
    index = int(np.searchsorted(grid, omega))
    if index == 0 or index >= grid.size:
        raise ValueError("sample frequency lies outside the tabulated sweep")
    width = float(grid[index] - grid[index - 1])
    gap = min(abs(float(grid[index]) - omega), abs(omega - float(grid[index - 1])))
    if gap <= 1e-9 * width:
        raise ValueError(
            "sample frequency coincides with a tabulated node; the "
            "piecewise-linear principal-value integral needs an interior pole"
        )
    return index - 1


def _level_terms(grid: FloatArray, data: FloatArray) -> FloatArray:
    """Per-interval trap ``integral_{s1}^{s2} g(s) ds``, exact for linear ``g``."""
    return 0.5 * (data[:-1] + data[1:]) * np.diff(grid)


def _pv_terms(grid: FloatArray, data: FloatArray, omega: float) -> FloatArray:
    """Per-interval ``integral_{s1}^{s2} g(s)/(s^2-omega^2) ds``, linear ``g``.

    Both logarithmic terms are individually singular when the pole lies inside
    the interval; their combination is finite (see the module docstring), and
    the caller guarantees an interior pole.
    """
    width = np.diff(grid)
    left = grid[:-1]
    right = grid[1:]
    slope = np.diff(data) / width
    intercept = data[:-1] - slope * left

    log_square = np.log(np.abs(right**2 - omega**2)) - np.log(
        np.abs(left**2 - omega**2)
    )
    log_ratio = (
        np.log(np.abs(right - omega))
        - np.log(np.abs(right + omega))
        - np.log(np.abs(left - omega))
        + np.log(np.abs(left + omega))
    )
    return 0.5 * slope * log_square + 0.5 * intercept / omega * log_ratio


def principal_value_integral(
    omegas: Sequence[float] | FloatArray,
    values: Sequence[float] | FloatArray,
    omega: float,
    *,
    lower_tail: bool = True,
) -> float:
    """``PV integral_0^inf values(s)/(s^2-omega^2) ds`` over the tabulated sweep.

    Exact for the piecewise-linear interpolant of ``values``. ``omega`` must
    fall strictly between two tabulated nodes. With ``lower_tail=True`` the
    band below the first node is closed by holding the tabulated value
    constant, which is the leading behaviour for an even function near zero
    frequency; the band above the last node is *not* closed here because its
    model depends on the caller's asymptote.
    """
    grid, data = _as_sweep(omegas, values, "values")
    sample = float(omega)
    _pole_interval(grid, sample)
    total = float(np.sum(_pv_terms(grid, data, sample)))
    if lower_tail:
        first = float(grid[0])
        total += 0.5 * float(data[0]) / sample * math.log(
            abs(first - sample) / (first + sample)
        )
    return total


def _level_integral(
    omegas: Sequence[float] | FloatArray,
    values: Sequence[float] | FloatArray,
    *,
    lower_tail: bool = True,
) -> float:
    """``integral_0^inf values(s) ds`` for the piecewise-linear interpolant."""
    grid, data = _as_sweep(omegas, values, "values")
    total = float(np.sum(_level_terms(grid, data)))
    if lower_tail:
        total += 0.5 * float(data[0]) * float(grid[0])
    return total


def kkr_damping(
    omegas: Sequence[float] | FloatArray,
    added_mass: Sequence[float] | FloatArray,
    omega: float,
    *,
    fit: AsymptoteFit | None = None,
    added_mass_inf: float | None = None,
    tail_coefficient: float | None = None,
    tail_fraction: float = 0.25,
) -> float:
    """Damping at ``omega`` from the added-mass sweep, eq. (1).

    Uses ``s^2/(s^2-omega^2) = 1 + omega^2/(s^2-omega^2)`` so that both pieces
    are evaluated with the same piecewise-linear machinery: the first one is a
    plain integral of ``A - A_inf`` and converges because that difference
    decays, the second one is the principal-value kernel.
    """
    grid, data = _as_sweep(omegas, added_mass, "added_mass")
    if fit is None and (added_mass_inf is None or tail_coefficient is None):
        fit = fit_added_mass_asymptote(grid, data, tail_fraction=tail_fraction)
    if fit is not None:
        if added_mass_inf is None:
            added_mass_inf = fit.added_mass_inf
        if tail_coefficient is None:
            tail_coefficient = fit.tail_coefficient
    sample = float(omega)
    deficit = data - float(added_mass_inf)

    level = _level_integral(grid, deficit)
    principal = principal_value_integral(grid, deficit, sample)

    # Upper tail: A - A_inf = c/s^2 beyond the last node.
    last = float(grid[-1])
    coefficient = float(tail_coefficient)
    if coefficient != 0.0 and sample < last:
        level += coefficient / last
        principal += coefficient / sample**2 * (
            0.5 / sample * math.log((last + sample) / (last - sample)) - 1.0 / last
        )

    return -2.0 / math.pi * (level + sample**2 * principal)


def kkr_added_mass(
    omegas: Sequence[float] | FloatArray,
    damping: Sequence[float] | FloatArray,
    omega: float,
    *,
    added_mass_inf: float,
) -> float:
    """Added mass at ``omega`` from the damping sweep, eq. (2).

    The band above the last node contributes nothing because the damping
    vanishes there; that is the radiation condition, and it holds
    exponentially fast for every case exercised here.
    """
    grid, data = _as_sweep(omegas, damping, "damping")
    if not math.isfinite(float(added_mass_inf)):
        raise ValueError("added_mass_inf must be finite")
    return float(added_mass_inf) + 2.0 / math.pi * principal_value_integral(
        grid, data, float(omega)
    )


@dataclass(frozen=True)
class KKRCheck:
    """Causality consistency of a tabulated radiation-coefficient pair."""

    sample_omegas: FloatArray
    measured_damping: FloatArray
    predicted_damping: FloatArray
    damping_relative_error: FloatArray
    measured_added_mass: FloatArray
    predicted_added_mass: FloatArray
    added_mass_relative_error: FloatArray
    added_mass_inf: float
    tail_coefficient: float
    asymptote_relative_residual: float
    damping_scale: float
    added_mass_scale: float

    @property
    def max_damping_relative_error(self) -> float:
        return float(np.max(self.damping_relative_error))

    @property
    def max_added_mass_relative_error(self) -> float:
        return float(np.max(self.added_mass_relative_error))


def _default_samples(grid: FloatArray, target: int = 12) -> FloatArray:
    """Geometric midpoints of an evenly spaced subset of the grid intervals."""
    intervals = grid.size - 1
    stride = max(1, intervals // max(1, target))
    # Start a little inside the sweep: at the first interval the band below the
    # grid (closed here by holding the value constant) is comparable to omega
    # itself, and the constant extension is no longer a good model there.
    index = np.arange(max(1, intervals // 10), intervals, stride)
    return np.sqrt(grid[index] * grid[index + 1])


def kkr_check(
    omegas: Sequence[float] | FloatArray,
    added_mass: Sequence[float] | FloatArray,
    damping: Sequence[float] | FloatArray,
    *,
    sample_omegas: Sequence[float] | FloatArray | None = None,
    tail_fraction: float = 0.25,
    added_mass_inf: float | None = None,
) -> KKRCheck:
    """Compare a tabulated ``(A, B)`` pair against the KK relations (1)/(2).

    Errors are normalised by the largest tabulated magnitude of the respective
    quantity, so the reported number says how badly the sweep violates
    causality on the scale of the coefficients themselves. Sample frequencies
    default to geometric midpoints of every ``stride``-th interval, which keeps
    them away from the nodes (see the module docstring).
    """
    grid, mass = _as_sweep(omegas, added_mass, "added_mass")
    _, damper = _as_sweep(omegas, damping, "damping")
    if added_mass_inf is None:
        fit = fit_added_mass_asymptote(grid, mass, tail_fraction=tail_fraction)
    else:
        fit = AsymptoteFit(
            added_mass_inf=float(added_mass_inf),
            tail_coefficient=0.0,
            points=0,
            relative_residual=0.0,
        )

    samples = (
        _default_samples(grid)
        if sample_omegas is None
        else np.asarray(sample_omegas, dtype=float).reshape(-1)
    )
    if samples.size == 0:
        raise ValueError("at least one sample frequency is required")
    for sample in samples:
        _pole_interval(grid, float(sample))

    predicted_damping = np.array(
        [
            kkr_damping(
                grid,
                mass,
                float(sample),
                added_mass_inf=fit.added_mass_inf,
                tail_coefficient=fit.tail_coefficient,
            )
            for sample in samples
        ]
    )
    predicted_added_mass = np.array(
        [
            kkr_added_mass(
                grid, damper, float(sample), added_mass_inf=fit.added_mass_inf
            )
            for sample in samples
        ]
    )
    measured_damping = np.interp(samples, grid, damper)
    measured_mass = np.interp(samples, grid, mass)

    mass_scale = max(float(np.max(np.abs(mass))), 1e-300)
    damper_scale = max(float(np.max(np.abs(damper))), 1e-300)
    return KKRCheck(
        sample_omegas=samples,
        measured_damping=measured_damping,
        predicted_damping=predicted_damping,
        damping_relative_error=np.abs(predicted_damping - measured_damping) / damper_scale,
        measured_added_mass=measured_mass,
        predicted_added_mass=predicted_added_mass,
        added_mass_relative_error=np.abs(predicted_added_mass - measured_mass) / mass_scale,
        added_mass_inf=fit.added_mass_inf,
        tail_coefficient=fit.tail_coefficient,
        asymptote_relative_residual=fit.relative_residual,
        damping_scale=damper_scale,
        added_mass_scale=mass_scale,
    )

# ---------------------------------------------------------------------------
# Band-limited closure with an explicit high-frequency damping tail
# ---------------------------------------------------------------------------
#
# The tabulated relations above close the sweep at the *lower* end (the band
# ``s < s_min`` is held constant, which is the leading even behaviour) but at
# the *upper* end ``kkr_added_mass`` assumes the damping has already vanished at
# the last node.  That assumption is safe for a deeply submerged body (the wave
# kernel decays like ``exp(-2 k f)``) and it is *not* safe for a
# surface-piercing hull whose damping peak sits inside or just above the
# computed band: the omitted ``s > s_max`` band then contributes a bias to every
# reconstructed added-mass value, which shows up as a spurious causality
# violation of the whole sweep.
#
# Two ingredients remove that artefact:
#
# ``fit_damping_power_tail``
#     fits ``B(s) = C s^-p`` on the upper tail of the sweep.  For a
#     surface-piercing body the high-frequency damping is algebraic
#     (the Kochin integral is analytic in ``1/sqrt(nu)`` away from the
#     waterline singularity), so a power law is the right model, and ``p``
#     comes out of the fit rather than being assumed.
#
# ``power_tail_principal_value``
#     evaluates the omitted ``PV integral_{s_max}^inf C s^-p/(s^2-omega^2) ds``
#     in closed form.  With ``s = omega u`` the integrand becomes
#     ``C omega^(-p-1) u^-p/(u^2-1)`` and, because ``u >= a = s_max/omega > 1``,
#     the expansion ``u^-p/(u^2-1) = sum_k u^(-p-2k-2)`` converges
#     geometrically: the interval integral is
#     ``sum_k a^(-(p+2k+1))/(p+2k+1)``, summed here to machine precision.
#
# ``kk_added_mass_closure``
#     is the band-limited counterpart of eq. (2).  ``A_inf`` cannot be read off
#     a truncated sweep reliably (it is a constant that the KK relation itself
#     fixes only up to the closure), so when ``added_mass_inf="closure"`` it is
#     estimated as the least-squares offset between the measured added mass and
#     the reconstructed one -- one free parameter, never more -- and the
#     residual reported is the *shape* residual.  Passing a number instead keeps
#     the value fixed and reports the full residual.


@dataclass(frozen=True)
class PowerTailFit:
    """Fit of ``B(s) = C s^-p`` over the upper tail of a damping sweep."""

    coefficient: float
    exponent: float
    points: int
    relative_residual: float
    valid: bool
    reason: str


def fit_damping_power_tail(
    omegas: Sequence[float] | FloatArray,
    damping: Sequence[float] | FloatArray,
    *,
    tail_fraction: float = 0.25,
    min_exponent: float = 3.0,
    max_relative_residual: float = 0.25,
) -> PowerTailFit:
    """Least-squares power-law fit of the damping tail, with validity guards.

    The tail integral only converges for ``p > 2`` and a fit whose residual is
    comparable to the data carries no information, so both are treated as
    *invalid* and reported as such: the caller then falls back to the
    zero-tail relation instead of silently adding a wrong number.
    """
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    grid, data = _as_sweep(omegas, damping, "damping")
    count = max(3, int(round(float(tail_fraction) * grid.size)))
    count = min(count, grid.size)
    tail_grid = grid[-count:]
    tail_data = data[-count:]
    if np.any(tail_data <= 0.0):
        return PowerTailFit(0.0, 0.0, count, float("inf"), False,
                            "non-positive damping in the tail band")
    slope, intercept = np.polyfit(np.log(tail_grid), np.log(tail_data), 1)
    exponent = float(-slope)
    coefficient = float(math.exp(intercept))
    model = coefficient * tail_grid ** (-exponent)
    scale = float(np.max(tail_data))
    residual = float(np.max(np.abs(model - tail_data)) / scale)
    if exponent < float(min_exponent):
        return PowerTailFit(coefficient, exponent, count, residual, False,
                            f"fitted exponent {exponent:.3g} < {min_exponent:g}: "
                            "the sweep has not reached the algebraic tail")
    if residual > float(max_relative_residual):
        return PowerTailFit(coefficient, exponent, count, residual, False,
                            f"tail fit residual {residual:.3g} > "
                            f"{max_relative_residual:g}")
    return PowerTailFit(coefficient, exponent, count, residual, True, "ok")


def power_tail_principal_value(
    omega: float,
    coefficient: float,
    exponent: float,
    start: float,
) -> float:
    """``PV integral_start^inf C s^-p/(s^2-omega^2) ds`` for ``p > 2``, ``start > omega``."""
    sample = float(omega)
    upper = float(start)
    power = float(exponent)
    if not (sample > 0.0 and math.isfinite(sample)):
        raise ValueError("omega must be positive and finite")
    if not upper > sample:
        raise ValueError("the tail must start above the sample frequency")
    if power <= 2.0:
        raise ValueError("the power tail must decay faster than s^-2")
    ratio = upper / sample
    total = 0.0
    for k in range(0, 400):
        m = power + 2.0 * k + 1.0
        term = ratio ** (-m) / m
        total += term
        if term <= 1e-17 * abs(total):
            break
    return float(coefficient) * sample ** (-power - 1.0) * total


@dataclass(frozen=True)
class KKRClosure:
    """Band-limited Kramers-Kronig closure of an added-mass sweep."""

    sample_omegas: FloatArray
    measured_added_mass: FloatArray
    predicted_added_mass: FloatArray
    residual: FloatArray
    relative_residual: FloatArray
    added_mass_inf: float
    added_mass_inf_source: str
    damping_scale: float
    added_mass_scale: float
    tail: PowerTailFit | None

    @property
    def max_relative_residual(self) -> float:
        finite = self.relative_residual[np.isfinite(self.relative_residual)]
        return float(np.max(finite)) if finite.size else float("inf")


def kk_added_mass_closure(
    omegas: Sequence[float] | FloatArray,
    added_mass: Sequence[float] | FloatArray,
    damping: Sequence[float] | FloatArray,
    *,
    sample_omegas: Sequence[float] | FloatArray | None = None,
    added_mass_inf: float | Literal["closure"] = "closure",
    damping_tail: PowerTailFit | None = None,
    tail_fit: bool = True,
    tail_fraction: float = 0.25,
) -> KKRClosure:
    """Reconstruct ``A(omega)`` from a band-limited ``B(omega)`` sweep.

    The reconstruction is eq. (2) closed at both ends of the computed band:
    below ``s_min`` the damping is held at its first tabulated value and above
    ``s_max`` an explicit power tail is integrated in closed form (see
    :func:`power_tail_principal_value`).  ``added_mass_inf="closure"`` estimates
    the one undetermined constant by least squares on the residual.
    """
    grid, mass = _as_sweep(omegas, added_mass, "added_mass")
    _, damper = _as_sweep(omegas, damping, "damping")
    tail = damping_tail
    if tail is None and tail_fit:
        tail = fit_damping_power_tail(grid, damper, tail_fraction=tail_fraction)
    if tail is not None and not tail.valid:
        tail = None
    if sample_omegas is None:
        samples = _default_samples(grid)
    else:
        samples = np.asarray(list(sample_omegas), dtype=float).reshape(-1)
    raw = np.empty(samples.size, dtype=float)
    for index, sample in enumerate(samples):
        value = principal_value_integral(grid, damper, float(sample))
        if tail is not None:
            value += power_tail_principal_value(
                float(sample), tail.coefficient, tail.exponent, float(grid[-1])
            )
        raw[index] = 2.0 / math.pi * value
    measured = np.interp(samples, grid, mass)
    shape = measured - raw
    if isinstance(added_mass_inf, str):
        if added_mass_inf != "closure":
            raise ValueError("added_mass_inf must be a number or 'closure'")
        offset = float(np.mean(shape))
        source = "closure (least-squares offset)"
    else:
        offset = float(added_mass_inf)
        source = "supplied"
    predicted = raw + offset
    residual = measured - predicted
    scale = max(float(np.max(np.abs(mass))), 1e-300)
    return KKRClosure(
        sample_omegas=samples,
        measured_added_mass=measured,
        predicted_added_mass=predicted,
        residual=residual,
        relative_residual=np.abs(residual) / scale,
        added_mass_inf=offset,
        added_mass_inf_source=source,
        damping_scale=max(float(np.max(np.abs(damper))), 1e-300),
        added_mass_scale=scale,
        tail=tail,
    )
