"""Coupled multi-degree-of-freedom radiation memory kernel matrix ``K_ij(t)``.

``tensorbem.time_domain`` closes the frequency-to-time bridge for **one** degree
of freedom: a single ``(omega, A, B)`` triple in, one kernel
``K(tau) = (2/pi) int_0^inf B(omega) cos(omega tau) domega`` out. A ship is not
one degree of freedom. Sway, heave and pitch radiate into each other, so the
Cummins equation of a real body carries a *kernel matrix*,

    (M + A_inf) x''(t) + int_0^t K(t - s) x'(s) ds + C x(t) = F(t),              (1)

with the symmetric matrix of impulse responses

    K_ij(tau) = (2/pi) int_0^inf B_ij(omega) cos(omega tau) domega ,             (2)
    B_ij(omega) = int_0^inf K_ij(tau) cos(omega tau) dtau ,                      (3)
    A_ij(omega) - A_ij(inf) = -(1/omega) int_0^inf K_ij(tau) sin(omega tau) dtau .  (4)

Why this is not the single-DOF case applied to each entry
---------------------------------------------------------
The off-diagonal entries break every shortcut the scalar path relies on.

* ``time_domain.validate_radiation_sweep`` **rejects** negative damping, and
  correctly: on the diagonal ``B_ii >= 0`` is the passivity of the radiation
  problem. A cross coefficient ``B_ij`` (``i != j``) is not a radiated power and
  is routinely negative. A matrix kernel therefore cannot be built by feeding
  each entry through the scalar path; the entry-wise sign rule is replaced here
  by the honest matrix statement, positive semidefiniteness ``x^T B x >= 0``.
* ``K_ij(0+) = (2/pi) int_0^inf B_ij domega`` is not signed by passivity either.
  On the diagonal it must be positive; off the diagonal it can be either sign
  and has to be *predicted from the sweep*, not assumed.
* The scalar decay anchor ``zeta = B(omega_n) / (2 omega_n (M + A))`` becomes a
  quadratic form: a coupled oscillator samples the damping matrix along its mode
  shape, ``zeta_k = v_k^T B(omega_k) v_k / (2 omega_k v_k^T (M + A(omega_k)) v_k)``.
  A kernel matrix with the right diagonals and a wrong ``B_ij`` passes every
  scalar test and still gets the coupled decay wrong.
* Reciprocity ``A_ij = A_ji``, ``B_ij = B_ji`` is a property of the radiated
  field, not of the code that produced the coefficients. It is gated here
  because a kernel matrix built from an asymmetric sweep is not a physical one.

What is provided
----------------
``check_alias_budget`` implements the sampling criterion of the campaign that
produced this module: with the kernel sampled every ``dt`` the discrete spectrum
is periodic with ``2 pi / dt``, so damping at ``omega + 2 pi n / dt`` folds onto
``omega``. The budget ``eps_tau = max_omega sum_{n>=1} |B(omega + 2 pi n / dt)|
/ max|B|`` must stay under one percent; ``choose_time_step`` picks the coarsest
``dt`` that satisfies it, and the report carries the sweep-tail share
separately -- the tail share is what a sweep that stops too early would leave
behind as an assumption.

``coupled_memory_kernels`` evaluates (2) entry by entry as an *exact*
piecewise-linear quadrature on the tabulated sweep, closing the missing low band
with the long-wave model ``B ~ c omega^3`` (signed, so it also works for cross
entries) and carrying the high-band truncation bound per entry.

``matrix_round_trip`` transforms back with Simpson quadratures on a different
grid, so (3)-(4) are not checked against the rule that produced the kernel.
``harmonic_radiation_force`` is the strongest of the round trips: it marches the
*time-domain* convolution under a prescribed multi-mode harmonic motion and
requires the force waveform to match ``[omega^2 A_ij(omega) - i omega
B_ij(omega)] X_j e^{i omega t}`` point by point. ``coupled_free_decay`` closes
the loop through the equation of motion, and ``constant_matrix_envelope`` prices
the cheap ``A = A_inf``, ``B = B(omega_n)`` substitute against the full kernel.

Limits
------
Linear, deep-water, zero-forward-speed radiation; ``n`` rigid modes with one
``(N, n, n)`` sweep. Every quadrature here consumes coefficients; nothing solves
a boundary-value problem. The oscillator used by the decay and envelope checks is
a test stand: ``C`` is supplied by the caller and need not be a hydrostatic
stiffness. Coefficients outside the tabulated band are not extrapolated beyond
the long-wave model on the low side, and are truncated (with a reported bound)
on the high side.

References
----------
Cummins (1962); Ogilvie (1964); Kotik & Mangulis (1962) for the Kramers-Kronig
pairing (4); Wehausen & Laitone (1960) for the symmetry of the added-mass and
damping matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from tensorbem.time_domain import (
    _cosine_moments,
    _long_wave_cosine,
    _piecewise_linear,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

__all__ = [
    "CoupledKernelMatrix",
    "alias_budget",
    "check_alias_budget",
    "choose_time_step",
    "complex_resonance",
    "complex_resonances",
    "constant_matrix_envelope",
    "coupled_free_decay",
    "coupled_memory_kernels",
    "decay_ratio_from_series",
    "decay_ratio_richardson",
    "frequency_domain_force",
    "harmonic_radiation_force",
    "kernel_resonances",
    "kernel_truncation",
    "matrix_round_trip",
    "mode_shape",
    "truncate_to",
    "validate_coupled_sweep",
]


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------
def _as_matrix_sweep(name: str, values: Sequence, count: int) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (N, n, n)")
    if array.shape[1] != array.shape[2]:
        raise ValueError(f"{name} must be square in its last two axes")
    if array.shape[0] != count:
        raise ValueError(f"{name} has {array.shape[0]} frequencies, omega has {count}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    return array


def validate_coupled_sweep(
    omega: Sequence[float],
    added_mass: Sequence[Sequence[Sequence[float]]],
    damping: Sequence[Sequence[Sequence[float]]],
    *,
    symmetry_tolerance: float = 5.0e-3,
    passivity_tolerance: float = 1.0e-6,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Validate a coupled radiation sweep ``(omega, A[N,n,n], B[N,n,n])``.

    Rejections, each naming the offending quantity:

    * fewer than eight frequencies, non-finite values, ``omega`` not strictly
      increasing and positive;
    * ``A`` or ``B`` not symmetric to ``symmetry_tolerance`` (relative to the
      largest entry): the radiated field is reciprocal, so an asymmetric sweep is
      refused instead of silently symmetrised;
    * ``B`` not positive semidefinite at some frequency (beyond
      ``passivity_tolerance`` times the scale). This is the matrix form of the
      scalar ``B >= 0`` rule: a passive body cannot radiate energy it never
      received. Individual off-diagonal entries may be negative, and usually are.
    """
    w = np.asarray(omega, dtype=float)
    if w.ndim != 1:
        raise ValueError("omega must be one-dimensional")
    if w.size < 8:
        raise ValueError(f"a radiation sweep needs at least 8 points, got {w.size}")
    if not np.all(np.isfinite(w)):
        raise ValueError("omega contains a non-finite value")
    if np.any(w <= 0.0):
        raise ValueError("omega must be strictly positive")
    if np.any(np.diff(w) <= 0.0):
        raise ValueError("omega must be strictly increasing")
    a = _as_matrix_sweep("added_mass", added_mass, w.size)
    b = _as_matrix_sweep("damping", damping, w.size)

    scale_a = max(float(np.max(np.abs(a))), 1e-300)
    scale_b = max(float(np.max(np.abs(b))), 1e-300)
    asymmetry_a = float(np.max(np.abs(a - np.swapaxes(a, 1, 2)))) / scale_a
    asymmetry_b = float(np.max(np.abs(b - np.swapaxes(b, 1, 2)))) / scale_b
    if asymmetry_a > symmetry_tolerance:
        raise ValueError(
            "added-mass matrix is not reciprocal: max |A_ij - A_ji| / max|A| = "
            f"{asymmetry_a:.3g} > {symmetry_tolerance:.3g}"
        )
    if asymmetry_b > symmetry_tolerance:
        raise ValueError(
            "damping matrix is not reciprocal: max |B_ij - B_ji| / max|B| = "
            f"{asymmetry_b:.3g} > {symmetry_tolerance:.3g}"
        )

    smallest = float(np.min(np.linalg.eigvalsh(b)))
    if smallest < -passivity_tolerance * max(scale_b, 1.0):
        raise ValueError(
            "damping matrix is not positive semidefinite: min eigenvalue = "
            f"{smallest:.6g}; a passive body cannot radiate energy it does not "
            "receive (off-diagonal entries may be negative, the matrix may not)"
        )
    return w, a, b


# ---------------------------------------------------------------------------
# sampling contract: the aliasing budget of the discrete-time kernel
# ---------------------------------------------------------------------------
def _alias_images(
    grid: FloatArray, values: FloatArray, omega_query: FloatArray, dt: float, images: int
) -> FloatArray:
    """``sum_{n>=1} |B(omega + 2 pi n / dt)|`` by linear interpolation.

    Outside the tabulated band an entry is taken as zero. That assumption is
    reported separately by :func:`check_alias_budget` as ``tail_share``: a sweep
    that stops while ``|B|`` is still a sizable fraction of its peak hides budget
    in the assumption; one that stops after the coefficients have decayed by
    decades does not.
    """
    shift = 2.0 * math.pi / dt
    total = np.zeros_like(omega_query)
    for n in range(1, int(images) + 1):
        total = total + np.abs(
            np.interp(omega_query + n * shift, grid, values, left=0.0, right=0.0)
        )
    return total


def check_alias_budget(
    omega: Sequence[float],
    damping: Sequence[Sequence[Sequence[float]]],
    dt: float,
    *,
    images: int = 4,
    queries: int = 96,
) -> dict:
    """``eps_tau`` for a kernel sampled every ``dt``, per entry and worst case.

    ``eps_tau = max_{i,j,omega} sum_{n>=1} |B_ij(omega + 2 pi n / dt)| /
    max|B_ij|``, evaluated on the tabulated grid plus a uniform sub-grid so a
    peak between two nodes is not missed. Cross entries are normalised by their
    own peak: a small cross coefficient that folds completely is still a failure
    of the sampling, not a rounding effect.
    """
    w = np.asarray(omega, dtype=float)
    b = np.asarray(damping, dtype=float)
    if b.ndim != 3 or b.shape[0] != w.size or b.shape[1] != b.shape[2]:
        raise ValueError("damping must have shape (N, n, n) matching omega")
    dt = float(dt)
    if not (dt > 0.0 and math.isfinite(dt)):
        raise ValueError("dt must be positive and finite")
    modes = b.shape[1]
    query = np.unique(np.concatenate([w, np.linspace(w[0], w[-1], int(queries))]))
    per_entry: dict[tuple[int, int], dict] = {}
    worst_value = -1.0
    worst_entry = (0, 0)
    worst_omega = float(w[0])
    for i in range(modes):
        for j in range(modes):
            entry = b[:, i, j]
            scale = float(np.max(np.abs(entry)))
            if scale == 0.0:
                per_entry[(i, j)] = {
                    "epsilon_tau": 0.0,
                    "worst_omega": float(w[0]),
                    "tail_share": 0.0,
                }
                continue
            normalised = _alias_images(w, entry, query, dt, images) / scale
            index = int(np.argmax(normalised))
            value = float(normalised[index])
            top = w >= 0.5 * w[-1]
            tail_scale = float(np.max(np.abs(entry[top]))) if np.any(top) else 0.0
            per_entry[(i, j)] = {
                "epsilon_tau": value,
                "worst_omega": float(query[index]),
                "tail_share": tail_scale / scale,
            }
            if value > worst_value:
                worst_value = value
                worst_entry = (i, j)
                worst_omega = float(query[index])
    return {
        "dt": dt,
        "nyquist": math.pi / dt,
        "omega_min": float(w[0]),
        "omega_max": float(w[-1]),
        "images": int(images),
        "epsilon_tau": float(max(worst_value, 0.0)),
        "worst_entry": worst_entry,
        "worst_omega": worst_omega,
        "per_entry": per_entry,
        "sweep_covers_nyquist": bool(w[-1] >= math.pi / dt),
    }


def alias_budget(
    omega: Sequence[float],
    damping: Sequence[Sequence[Sequence[float]]],
    dt: float,
    *,
    threshold: float = 1.0e-2,
) -> dict:
    """Alias budget report with an explicit pass/fail against ``threshold``."""
    report = check_alias_budget(omega, damping, dt)
    report["threshold"] = float(threshold)
    report["passed"] = bool(report["epsilon_tau"] <= threshold)
    return report


def choose_time_step(
    omega: Sequence[float],
    damping: Sequence[Sequence[Sequence[float]]],
    *,
    threshold: float = 1.0e-2,
    dt_min: float | None = None,
    dt_max: float | None = None,
    steps: int = 30,
) -> tuple[float, dict]:
    """Coarsest ``dt`` whose alias budget stays under ``threshold``.

    The budget grows with ``dt`` (a coarser grid folds more of the band), so a
    bisection on ``log(dt)`` finds the answer in a handful of evaluations. ``dt``
    is bounded above by ``pi / omega_max`` so the grid's Nyquist frequency still
    covers the tabulated band, and below by ``dt_min``.
    """
    w = np.asarray(omega, dtype=float)
    upper = math.pi / float(w[-1])
    if dt_max is not None:
        upper = min(upper, float(dt_max))
    lower = float(dt_min) if dt_min is not None else upper / 1000.0
    upper_report = check_alias_budget(w, damping, upper)
    if upper_report["epsilon_tau"] <= threshold:
        return upper, upper_report
    low, high = lower, upper
    best = upper_report
    for _ in range(int(steps)):
        middle = math.sqrt(low * high)
        report = check_alias_budget(w, damping, middle)
        if report["epsilon_tau"] <= threshold:
            low, best = middle, report
        else:
            high = middle
    return low, best

# ---------------------------------------------------------------------------
# the kernel matrix
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CoupledKernelMatrix:
    """``K_ij(tau)`` on a uniform grid, with per-entry truncation bounds."""

    tau: FloatArray
    values: FloatArray
    omega_max: float
    damping_scale: FloatArray
    high_frequency_bound: FloatArray
    low_frequency_bound: FloatArray
    metadata: dict = field(default_factory=dict)

    @property
    def dt(self) -> float:
        return float(self.tau[1] - self.tau[0])

    @property
    def span(self) -> float:
        return float(self.tau[-1])

    @property
    def modes(self) -> int:
        return int(self.values.shape[1])

    def entry(self, i: int, j: int) -> FloatArray:
        return self.values[:, i, j]

    def at(self, tau: float | Sequence[float]) -> NDArray:
        """Kernel matrix at arbitrary ``tau``; zero for ``tau < 0`` and beyond the span.

        Causality is a property of the *kernel*, not of the caller: ``K(t<0) = 0``
        is what makes the convolution in (1) a convolution and not a correlation,
        so the query returns zero there instead of clamping to ``K(0+)``.
        """
        query = np.asarray(tau, dtype=float)
        flat = np.atleast_1d(query).reshape(-1)
        out = np.empty((flat.size, self.modes, self.modes))
        for i in range(self.modes):
            for j in range(self.modes):
                out[:, i, j] = np.interp(
                    flat, self.tau, self.values[:, i, j], left=0.0, right=0.0
                )
        return out.reshape(query.shape + (self.modes, self.modes))

    @property
    def peak(self) -> FloatArray:
        return np.max(np.abs(self.values), axis=0)


def coupled_memory_kernels(
    omega: Sequence[float],
    damping: Sequence[Sequence[Sequence[float]]],
    tau: Sequence[float],
    *,
    low_frequency_model: str = "cubic",
) -> CoupledKernelMatrix:
    """Kernel matrix ``K_ij(tau) = (2/pi) int_0^inf B_ij(omega) cos(omega tau) domega``.

    Each entry is integrated *exactly* over the tabulated band as a
    piecewise-linear function of ``omega`` (no FFT, no leakage); the band
    ``[0, omega_min]`` is closed with a signed long-wave model ``B ~ c omega^3``
    and the band above ``omega_max`` is truncated with the bound
    ``(2/pi) |B(omega_max)| omega_max`` stored per entry.

    The passivity rule is not applied entry by entry: cross coefficients may be
    negative. Symmetry and positive semidefiniteness are gated by
    :func:`validate_coupled_sweep`.
    """
    w = np.asarray(omega, dtype=float)
    b = np.asarray(damping, dtype=float)
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
    if b.ndim != 3 or b.shape[0] != w.size or b.shape[1] != b.shape[2]:
        raise ValueError("damping must have shape (N, n, n) matching omega")
    modes = b.shape[1]
    values = np.zeros((grid.size, modes, modes))
    scale = np.zeros((modes, modes))
    high = np.zeros((modes, modes))
    low = np.zeros((modes, modes))
    order_zero, order_one = _cosine_moments(w, grid)
    for i in range(modes):
        for j in range(modes):
            entry = b[:, i, j]
            c0, c1 = _piecewise_linear(entry, w)
            values[:, i, j] = float(2.0 / math.pi) * (order_zero @ c0 + order_one @ c1)
            if low_frequency_model == "cubic":
                limit = min(w[0] * 10.0, w[-1])
                mask = (w <= limit) & (w > 0.0)
                if np.count_nonzero(mask) < 2:
                    mask = np.zeros_like(w, dtype=bool)
                    mask[: min(3, w.size)] = True
                # signed scale: a cross entry may decay from a negative value
                scale[i, j] = float(np.mean(entry[mask] / w[mask] ** 3))
                if scale[i, j]:
                    values[:, i, j] += _long_wave_cosine(scale[i, j], float(w[0]), grid)
            high[i, j] = float(2.0 / math.pi) * abs(float(entry[-1])) * float(w[-1])
            low[i, j] = (
                float(2.0 / math.pi) * abs(float(scale[i, j])) * float(w[0]) ** 4 / 4.0
            )
    return CoupledKernelMatrix(
        tau=grid,
        values=values,
        omega_max=float(w[-1]),
        damping_scale=scale,
        high_frequency_bound=high,
        low_frequency_bound=low,
        metadata={
            "omega_min": float(w[0]),
            "low_frequency_model": low_frequency_model,
            "damping_peak": float(np.max(np.abs(b))),
            # kept so ``_refine_kernel`` can rebuild the transform on a finer
            # grid instead of interpolating it (interpolation smears the K(0+)
            # jump and biases the decay by a few percent)
            "sweep_omega": w,
            "sweep_damping": b,
        },
    )


def kernel_truncation(
    kernel: CoupledKernelMatrix,
    *,
    relative: float = 5.0e-3,
    frequencies: Sequence[float] | None = None,
) -> dict:
    """First ``tau`` after which ``|K_ij| / max|K_ij|`` stays below ``relative``.

    The kernel is cut at the largest of the per-entry truncation times, so every
    entry obeys the same tail criterion. Two truncation errors are reported:

    * ``worst_tail_fraction_after_cut`` -- the largest remaining ``|K|/max|K|``
      on the retained window (below the threshold by construction);
    * ``tail_transfer_fraction`` -- if ``frequencies`` is given, the largest
      change the cut makes to the *frequency-domain damping*,
      ``max |int_{tau_max}^inf K_ij cos(omega tau) dtau| / max|B_ij|``. This is
      the truncation error that matters, and it is not the same as the share of
      ``int |K| dtau``: a kernel that keeps oscillating carries a large absolute
      area in its tail while its cosine transform has already converged, so an
      area-based number overstates the error by two orders of magnitude.
    """
    if not (0.0 < float(relative) < 1.0):
        raise ValueError("relative must lie in (0, 1)")
    worst_index = 0
    per_entry: dict[tuple[int, int], dict] = {}
    for i in range(kernel.modes):
        for j in range(kernel.modes):
            entry = np.abs(kernel.entry(i, j))
            peak = float(np.max(entry))
            if peak == 0.0:
                per_entry[(i, j)] = {"tau": 0.0, "tail_fraction": 0.0}
                continue
            above = np.nonzero(entry / peak >= relative)[0]
            index = int(above[-1]) + 1 if above.size else 0
            index = min(index, kernel.tau.size - 1)
            worst_index = max(worst_index, index)
            per_entry[(i, j)] = {
                "tau": float(kernel.tau[index]),
                "tail_fraction": float(entry[index] / peak),
            }
    after = 0.0
    transfer = 0.0
    for i in range(kernel.modes):
        for j in range(kernel.modes):
            peak = float(kernel.peak[i, j])
            entry = np.abs(kernel.entry(i, j))
            if peak > 0.0:
                after = max(after, float(entry[worst_index]) / peak)
    if frequencies is not None:
        query = np.asarray(frequencies, dtype=float)
        for i in range(kernel.modes):
            for j in range(kernel.modes):
                tail = kernel.values[worst_index:, i, j]
                times = kernel.tau[worst_index:]
                if tail.size < 2:
                    continue
                for value in query:
                    projected = float(np.trapezoid(tail * np.cos(value * times), times))
                    scale = float(np.max(np.abs(kernel.entry(i, j))))
                    if scale > 0.0:
                        transfer = max(transfer, abs(projected) / scale)
    return {
        "relative_threshold": float(relative),
        "tau_max": float(kernel.tau[worst_index]),
        "index": int(worst_index),
        "sampled_fraction": float(kernel.tau[worst_index] / kernel.span),
        "worst_tail_fraction_after_cut": float(after),
        "tail_transfer_fraction": float(transfer),
        "frequencies_used_for_transfer": None if frequencies is None else [float(x) for x in frequencies],
        "per_entry": per_entry,
    }


def truncate_to(kernel: CoupledKernelMatrix, tau_max: float) -> CoupledKernelMatrix:
    """Return the same kernel cut at ``tau_max`` (inclusive)."""
    if not (0.0 < float(tau_max) <= kernel.span):
        raise ValueError("tau_max must lie inside the built grid")
    count = int(math.floor(float(tau_max) / kernel.dt + 1e-9)) + 1
    return CoupledKernelMatrix(
        tau=kernel.tau[:count],
        values=kernel.values[:count],
        omega_max=kernel.omega_max,
        damping_scale=kernel.damping_scale,
        high_frequency_bound=kernel.high_frequency_bound,
        low_frequency_bound=kernel.low_frequency_bound,
        metadata=dict(kernel.metadata),
    )


def _refine_kernel(kernel: CoupledKernelMatrix, dt_target: float) -> CoupledKernelMatrix:
    """Rebuild (or, if the sweep is gone, interpolate) on a finer time grid.

    When the sweep that produced the kernel is still attached, the kernel is
    *recomputed* on the finer grid: the cosine quadrature is exact for a
    piecewise-linear sweep, whereas linear interpolation of the kernel smears the
    ``K(0+)`` jump over one grid cell and biases the decay rate by a few percent
    (measured: 11% on the synthetic Kramers-Kronig pair at a factor-eight
    refinement). Interpolation is kept only as a fallback for kernels built by
    hand.
    """
    if dt_target >= kernel.dt:
        return kernel
    factor = int(math.ceil(kernel.dt / dt_target))
    dt_new = kernel.dt / factor
    tau = np.arange(kernel.tau.size * factor + 1) * dt_new
    omega = kernel.metadata.get("sweep_omega")
    damping = kernel.metadata.get("sweep_damping")
    if omega is not None and damping is not None:
        rebuilt = coupled_memory_kernels(
            omega,
            damping,
            tau,
            low_frequency_model=kernel.metadata.get("low_frequency_model", "cubic"),
        )
        values = rebuilt.values
    else:
        values = np.empty((tau.size, kernel.modes, kernel.modes))
        for i in range(kernel.modes):
            for j in range(kernel.modes):
                values[:, i, j] = np.interp(tau, kernel.tau, kernel.values[:, i, j], right=0.0)
    return CoupledKernelMatrix(
        tau=tau,
        values=values,
        omega_max=kernel.omega_max,
        damping_scale=kernel.damping_scale,
        high_frequency_bound=kernel.high_frequency_bound,
        low_frequency_bound=kernel.low_frequency_bound,
        metadata=dict(kernel.metadata),
    )


# ---------------------------------------------------------------------------
# inverse legs: B and A back out of the kernel
# ---------------------------------------------------------------------------
def _simpson_weights(count: int) -> FloatArray:
    weights = np.ones(count)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return weights / 3.0


def _inverse_legs(
    kernel: CoupledKernelMatrix, omega_query: FloatArray
) -> tuple[NDArray, NDArray]:
    """``B_ij(omega)`` and ``int K_ij sin(omega tau) dtau`` on the kernel grid."""
    count = kernel.tau.size
    if count % 2 == 0:
        count -= 1
    if count < 3:
        raise ValueError("the kernel grid is too short for a Simpson quadrature")
    tau = kernel.tau[:count]
    weights = _simpson_weights(count) * kernel.dt
    cosine = np.cos(np.outer(omega_query, tau))
    sine = np.sin(np.outer(omega_query, tau))
    b_rt = np.empty((omega_query.size, kernel.modes, kernel.modes))
    q_rt = np.empty_like(b_rt)
    for i in range(kernel.modes):
        for j in range(kernel.modes):
            weighted = kernel.values[:count, i, j] * weights
            b_rt[:, i, j] = cosine @ weighted
            q_rt[:, i, j] = sine @ weighted
    return b_rt, q_rt


def matrix_round_trip(
    omega: Sequence[float],
    added_mass: Sequence[Sequence[Sequence[float]]],
    damping: Sequence[Sequence[Sequence[float]]],
    added_mass_inf: Sequence[Sequence[float]],
    kernel: CoupledKernelMatrix,
    *,
    band: tuple[float, float] | None = None,
    samples: int = 60,
) -> dict:
    """``B -> K -> (B, A)`` for every entry, against the two-percent gate.

    The inverse legs are Simpson quadratures on the kernel grid; the forward leg
    was an exact piecewise-linear quadrature of the sweep. They share neither a
    rule nor a sample set, so the residual measures the transform.

    ``band`` defaults to ``[max(omega_min, 4 pi/tau_max), min(0.6 omega_max,
    pi/(2 dt))]``: below the low edge the sine quadrature of (4) is truncated
    mid-period, above the high edge the sweep itself is truncated. Each entry is
    normalised by its own peak, because a cross entry that is small everywhere
    would otherwise be judged on rounding noise.
    """
    w, a, b = validate_coupled_sweep(omega, added_mass, damping)
    infinity = np.asarray(added_mass_inf, dtype=float)
    if infinity.shape != (b.shape[1], b.shape[2]):
        raise ValueError("added_mass_inf must have shape (n, n)")
    low = max(float(w[0]), 4.0 * math.pi / kernel.span)
    high = min(0.6 * float(w[-1]), math.pi / (2.0 * kernel.dt))
    if band is not None:
        low, high = float(band[0]), float(band[1])
    if not (0.0 < low < high):
        raise ValueError(
            f"empty analysis band [{low:.6g}, {high:.6g}]: sweep and kernel grid are incompatible"
        )
    query = np.linspace(low, high, int(samples))
    b_rt, q_rt = _inverse_legs(kernel, query)
    a_exact = np.empty((query.size, b.shape[1], b.shape[2]))
    b_exact = np.empty_like(a_exact)
    for i in range(b.shape[1]):
        for j in range(b.shape[2]):
            a_exact[:, i, j] = np.interp(query, w, a[:, i, j])
            b_exact[:, i, j] = np.interp(query, w, b[:, i, j])
    a_rt = infinity[None, :, :] - q_rt / query[:, None, None]

    per_entry: dict[tuple[int, int], dict] = {}
    worst = 0.0
    worst_entry = (0, 0)
    for i in range(b.shape[1]):
        for j in range(b.shape[2]):
            denom_b = max(float(np.max(np.abs(b_exact[:, i, j]))), 1e-300)
            denom_a = max(float(np.max(np.abs(a_exact[:, i, j]))), 1e-300)
            err_b = float(np.max(np.abs(b_rt[:, i, j] - b_exact[:, i, j])) / denom_b)
            err_a = float(np.max(np.abs(a_rt[:, i, j] - a_exact[:, i, j])) / denom_a)
            per_entry[(i, j)] = {
                "damping_relative_error": err_b,
                "added_mass_relative_error": err_a,
                "damping_scale": denom_b,
                "added_mass_scale": denom_a,
            }
            gate = max(err_b, err_a)
            if gate > worst:
                worst, worst_entry = gate, (i, j)
    return {
        "band": (float(low), float(high)),
        "samples": int(samples),
        "per_entry": per_entry,
        "worst_entry": worst_entry,
        "worst_relative_error": float(worst),
        "passed": bool(worst <= 0.02),
    }


# ---------------------------------------------------------------------------
# prescribed-motion round trip: time-domain convolution against the sweep
# ---------------------------------------------------------------------------
def frequency_domain_force(
    omega: float,
    added_mass: Sequence[Sequence[float]],
    damping: Sequence[Sequence[float]],
    amplitude: Sequence[float],
    phase: Sequence[float] | None,
    time: Sequence[float],
) -> ComplexArray:
    """``F_i(t) = sum_j [omega^2 A_ij - i omega B_ij] X_j e^{i(omega t + phi_j)}``.

    Returned as the complex analytic signal; the physical force is its real part
    (time factor ``exp(i omega t)``, matching the ``B -> K`` convention of
    :mod:`tensorbem.time_domain`).
    """
    a = np.atleast_2d(np.asarray(added_mass, dtype=float))
    b = np.atleast_2d(np.asarray(damping, dtype=float))
    x = np.asarray(amplitude, dtype=float)
    phi = np.zeros_like(x) if phase is None else np.asarray(phase, dtype=float)
    t = np.asarray(time, dtype=float)
    matrix = omega ** 2 * a - 1j * omega * b
    modal = matrix @ (x * np.exp(1j * phi))
    return modal[None, :] * np.exp(1j * omega * t)[:, None]


def _trapezoid_memory(kernel_values: FloatArray, signal: FloatArray, dt: float) -> FloatArray:
    """``int_0^{t_k} K(t_k - s) g(s) ds`` on the kernel grid (trapezoid in ``s``).

    The kernel is zero beyond the built span, so a record longer than the kernel
    is padded with zeros rather than wrapped: past ``tau_max`` the convolution
    becomes a finite weighted average of the signal, which is the truncation the
    kernel was accepted with.
    """
    count = signal.size
    values = np.zeros(count)
    kept = min(count, kernel_values.size)
    values[:kept] = kernel_values[:kept]
    product = np.convolve(values, signal)[:count] * dt
    correction = 0.5 * dt * (values[0] * signal + values * signal[0])
    return product - correction


def harmonic_radiation_force(
    kernel: CoupledKernelMatrix,
    added_mass_inf: Sequence[Sequence[float]],
    *,
    omega: float,
    amplitude: Sequence[float],
    added_mass: Sequence[Sequence[float]],
    damping: Sequence[Sequence[float]],
    phase: Sequence[float] | None = None,
    warmup: float | None = None,
    cycles: int = 4,
    samples_per_period: float = 60.0,
) -> dict:
    """March the convolution under prescribed motion and compare with the sweep.

    With ``x_j(t) = X_j cos(omega t + phi_j)`` (the ``Re{X e^{i omega t}}``
    convention of the frequency-domain form) the Cummins radiation force is

        F_i(t) = sum_j omega^2 A_ij(inf) X_j cos(omega t + phi_j)
                 + sum_j omega X_j int_0^t K_ij(t - s) sin(omega s + phi_j) ds,

    which, once the transient memory has died out, must equal the frequency domain
    ``Re sum_j [omega^2 A_ij(omega) - i omega B_ij(omega)] X_j e^{i(omega t +
    phi_j)}``. The comparison is pointwise,

        max_t |F_i(t) - Re F_i^fd(t)| / |sum_j [omega^2 A_ij - i omega B_ij] X_j|,

    i.e. against the amplitude of that force component, so a zero crossing does
    not dominate the metric. The march uses the kernel on its own grid, a
    *prescribed* motion (the equation of motion is not integrated) and the
    trapezoid memory sum: it tests the kernel and the ``A(inf)`` split, not an
    integrator.

    ``added_mass`` and ``damping`` are the sweep values at ``omega`` (the
    reference side of the comparison), passed explicitly so the check cannot
    silently compare the kernel with the sweep at the wrong frequency.
    """
    omega = float(omega)
    if not (omega > 0.0 and math.isfinite(omega)):
        raise ValueError("omega must be positive and finite")
    x = np.asarray(amplitude, dtype=float)
    modes = x.size
    if modes != kernel.modes:
        raise ValueError(f"amplitude has {modes} entries, kernel has {kernel.modes} modes")
    phi = np.zeros(modes) if phase is None else np.asarray(phase, dtype=float)
    if phi.size != modes:
        raise ValueError("phase must match amplitude")
    infinity = np.atleast_2d(np.asarray(added_mass_inf, dtype=float))
    reference_a = np.atleast_2d(np.asarray(added_mass, dtype=float))
    reference_b = np.atleast_2d(np.asarray(damping, dtype=float))

    period = 2.0 * math.pi / omega
    steady = float(warmup) if warmup is not None else max(3.0 * kernel.span, 2.0 * period)
    work = _refine_kernel(kernel, period / float(samples_per_period))
    dt = work.dt
    steps = int(round((steady + cycles * period) / dt))
    start = int(round(steady / dt))
    if steps < 8 or start >= steps - 4:
        raise ValueError("the requested window is shorter than the kernel grid allows")
    time = np.arange(steps + 1) * dt

    force = np.zeros((time.size, modes))
    memory_share = np.zeros((time.size, modes))
    for i in range(modes):
        integral = np.zeros(time.size)
        inertia = np.zeros(time.size)
        for j in range(modes):
            # x = X cos(omega t + phi)  =>  xdot = -omega X sin,  xddot = -omega^2 X cos
            velocity_signal = -omega * x[j] * np.sin(omega * time + phi[j])
            integral += _trapezoid_memory(work.entry(i, j), velocity_signal, dt)
            inertia += -infinity[i, j] * (-omega ** 2 * x[j] * np.cos(omega * time + phi[j]))
        memory_share[:, i] = integral
        force[:, i] = inertia - integral

    reference = frequency_domain_force(omega, reference_a, reference_b, x, phi, time).real
    window = slice(start, time.size)
    per_mode: dict[int, dict] = {}
    worst = 0.0
    for i in range(modes):
        modal = (omega ** 2 * reference_a[i, :] - 1j * omega * reference_b[i, :]) * (
            x * np.exp(1j * phi)
        )
        normalisation = float(abs(np.sum(modal)))
        if normalisation == 0.0:
            raise ValueError(
                f"the frequency-domain force of mode {i} vanishes at omega={omega:.6g}"
            )
        residual = float(np.max(np.abs(force[window, i] - reference[window, i])))
        error = residual / normalisation
        per_mode[i] = {
            "relative_error": float(error),
            "absolute_residual": residual,
            "force_amplitude": normalisation,
            "memory_share_of_amplitude": float(
                np.max(np.abs(memory_share[window, i])) / normalisation
            ),
        }
        worst = max(worst, error)
    return {
        "omega": omega,
        "dt": dt,
        "period": period,
        "steady_start": float(time[start]),
        "window": (float(time[start]), float(time[-1])),
        "per_mode": per_mode,
        "worst_relative_error": float(worst),
        "passed": bool(worst <= 0.02),
        "time": time,
        "force": force,
        "reference": reference,
    }


# ---------------------------------------------------------------------------
# free decay of the coupled Cummins equation
# ---------------------------------------------------------------------------
def mode_shape(
    mass: Sequence[Sequence[float]],
    added_mass: Sequence[Sequence[float]],
    stiffness: Sequence[Sequence[float]],
) -> tuple[FloatArray, FloatArray]:
    """Undamped modes of ``(M + A) x'' + C x = 0`` for one frozen ``A``.

    Returns ``(omega_n, vectors)`` of the generalised problem ``C v = omega^2 (M +
    A) v`` with columns normalised so that ``v^T (M + A) v = 1``; a modal
    coordinate is then ``q = v^T (M + A) x``.
    """
    m = np.atleast_2d(np.asarray(mass, dtype=float))
    a = np.atleast_2d(np.asarray(added_mass, dtype=float))
    c = np.atleast_2d(np.asarray(stiffness, dtype=float))
    if m.shape != a.shape or m.shape != c.shape or m.shape[0] != m.shape[1]:
        raise ValueError("mass, added_mass and stiffness must be square and of equal size")
    total = 0.5 * ((m + a) + (m + a).T)
    eigenvalues_m, vectors_m = np.linalg.eigh(total)
    if np.any(eigenvalues_m <= 0.0):
        raise ValueError("mass + added mass must be positive definite")
    inverse_root = (vectors_m * (1.0 / np.sqrt(eigenvalues_m))) @ vectors_m.T
    whitened = inverse_root @ c @ inverse_root
    eigenvalues, vectors = np.linalg.eigh(0.5 * (whitened + whitened.T))
    frequency = np.sqrt(np.clip(eigenvalues, 0.0, None))
    order = np.argsort(frequency)
    return frequency[order], inverse_root @ vectors[:, order]


def modal_damping_ratio(
    omega: float,
    mass: Sequence[Sequence[float]],
    added_mass: Sequence[Sequence[float]],
    damping: Sequence[Sequence[float]],
    shape: Sequence[float],
) -> dict:
    """``zeta = v^T B v / (2 omega v^T (M + A) v)`` for one mode shape.

    The scalar anchor generalised to a matrix: a coupled oscillator samples the
    damping matrix along its mode shape, so ``B_ij`` and ``A_ij`` enter as
    quadratic forms and not as diagonal entries.
    """
    v = np.asarray(shape, dtype=float)
    inertia = np.atleast_2d(np.asarray(mass, dtype=float)) + np.atleast_2d(
        np.asarray(added_mass, dtype=float)
    )
    damp = np.atleast_2d(np.asarray(damping, dtype=float))
    omega = float(omega)
    if not (omega > 0.0):
        raise ValueError("omega must be positive")
    numerator = float(v @ damp @ v)
    denominator = 2.0 * omega * float(v @ inertia @ v)
    if denominator == 0.0:
        raise ValueError("the modal inertia vanishes for this shape")
    return {
        "zeta": numerator / denominator,
        "damping_form": numerator,
        "inertia_form": float(v @ inertia @ v),
        "omega": omega,
    }


def coupled_free_decay(
    kernel: CoupledKernelMatrix,
    added_mass_inf: Sequence[Sequence[float]],
    *,
    mass: Sequence[Sequence[float]],
    stiffness: Sequence[Sequence[float]],
    initial_displacement: Sequence[float],
    initial_velocity: Sequence[float] | None = None,
    duration: float,
) -> dict:
    """Integrate ``(M + A_inf) x'' + int K(t-s) x' ds + C x = 0`` from rest.

    Classical RK4 on the equation of motion, with the memory integral evaluated
    as a trapezoid sum on the kernel grid (no rational fit, no state-space
    reduction), so the kernel enters exactly as built: the same construction as
    ``time_domain._integrate_cummins``, extended to a matrix. The cost is
    ``O(steps^2)`` per entry, which makes this a verification tool rather than a
    production seakeeper.
    """
    n = kernel.modes
    inertia = np.atleast_2d(np.asarray(mass, dtype=float)) + np.atleast_2d(
        np.asarray(added_mass_inf, dtype=float)
    )
    stiff = np.atleast_2d(np.asarray(stiffness, dtype=float))
    if inertia.shape != (n, n) or stiff.shape != (n, n):
        raise ValueError(f"mass, added_mass_inf and stiffness must all be ({n}, {n})")
    x0 = np.asarray(initial_displacement, dtype=float)
    v0 = np.zeros(n) if initial_velocity is None else np.asarray(initial_velocity, dtype=float)
    if x0.size != n or v0.size != n:
        raise ValueError("initial conditions must match the number of modes")
    dt = kernel.dt
    steps = int(round(float(duration) / dt))
    if steps < 8:
        raise ValueError("duration must span at least 8 kernel steps")
    time = np.arange(steps + 1) * dt
    displacement = np.zeros((steps + 1, n))
    velocity = np.zeros((steps + 1, n))
    displacement[0] = x0
    velocity[0] = v0
    inverse = np.linalg.inv(0.5 * (inertia + inertia.T))
    # The stage quadrature reads the kernel on a half-step grid; it is
    # interpolated there (and zero beyond the built span) rather than required to
    # have been built at half steps, so the integration step and the kernel grid
    # stay independent.
    half_tau = np.arange(2 * steps + 1) * (0.5 * dt)
    half_values = np.empty((half_tau.size, n, n))
    for i in range(n):
        for j in range(n):
            half_values[:, i, j] = np.interp(
                half_tau, kernel.tau, kernel.values[:, i, j], left=0.0, right=0.0
            )
    half = CoupledKernelMatrix(
        tau=half_tau,
        values=half_values,
        omega_max=kernel.omega_max,
        damping_scale=kernel.damping_scale,
        high_frequency_bound=kernel.high_frequency_bound,
        low_frequency_bound=kernel.low_frequency_bound,
        metadata=dict(kernel.metadata),
    )
    h0 = half.values[0]

    def memory(index: int, stage: int, stage_velocity: FloatArray) -> FloatArray:
        """Memory force at ``t = index*dt + stage*dt/2``, trapezoid in the history."""
        total = np.zeros(n)
        base = 2 * index + stage
        history = half.values[base::-2][: index + 1]  # spacing dt in tau
        if index > 0:
            contribution = 0.5 * np.einsum("ij,j->i", history[0], velocity[0])
            if index > 1:
                contribution = contribution + np.einsum(
                    "sij,sj->i", history[1:index], velocity[1:index]
                )
            contribution = contribution + 0.5 * np.einsum(
                "ij,j->i", history[index], velocity[index]
            )
            total = dt * contribution
        partial = 0.5 * stage * dt * (
            np.einsum("ij,j->i", history[index], velocity[index])
            + np.einsum("ij,j->i", h0, stage_velocity)
        )
        return total + partial

    for index in range(steps):
        x_n = displacement[index]
        v_n = velocity[index]
        a1 = inverse @ (-stiff @ x_n - memory(index, 0, v_n))
        v1 = v_n + 0.5 * dt * a1
        x1 = x_n + 0.5 * dt * v_n
        a2 = inverse @ (-stiff @ x1 - memory(index, 1, v1))
        v2 = v_n + 0.5 * dt * a2
        x2 = x_n + 0.5 * dt * v1
        a3 = inverse @ (-stiff @ x2 - memory(index, 1, v2))
        v3 = v_n + dt * a3
        x3 = x_n + dt * v2
        a4 = inverse @ (-stiff @ x3 - memory(index, 2, v3))
        displacement[index + 1] = x_n + dt / 6.0 * (v_n + 2.0 * v1 + 2.0 * v2 + v3)
        velocity[index + 1] = v_n + dt / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
    return {
        "time": time,
        "displacement": displacement,
        "velocity": velocity,
        "dt": float(dt),
        "duration": float(duration),
        "steps": int(steps),
    }


# ---------------------------------------------------------------------------
# decay measurement on a single modal coordinate
# ---------------------------------------------------------------------------
def _dominant_frequency(time: FloatArray, signal: FloatArray) -> float | None:
    """Frequency of the largest oscillation in ``signal`` (FFT peak + parabolic refine)."""
    if signal.size < 32:
        return None
    step = float(time[1] - time[0])
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    frequencies = np.fft.rfftfreq(signal.size, d=step) * 2.0 * math.pi
    spectrum[0] = 0.0
    index = int(np.argmax(spectrum))
    if index == 0:
        return None
    offset = 0.0
    if 0 < index < spectrum.size - 1:
        left, centre, right = spectrum[index - 1], spectrum[index], spectrum[index + 1]
        denominator = left - 2.0 * centre + right
        if denominator != 0.0:
            offset = 0.5 * (left - right) / denominator
    step_frequency = frequencies[1] - frequencies[0]
    return float(frequencies[index] + offset * step_frequency)


def _envelope_peaks(time: FloatArray, signal: FloatArray) -> NDArray:
    """Times and magnitudes of the local extrema of ``signal`` (envelope samples)."""
    if signal.size < 5:
        return np.empty((0, 2))
    interior = np.arange(1, signal.size - 1)
    left = signal[interior - 1]
    centre = signal[interior]
    right = signal[interior + 1]
    is_peak = ((centre > left) & (centre >= right)) | ((centre < left) & (centre <= right))
    indices = interior[is_peak]
    return np.column_stack((time[indices], np.abs(signal[indices])))


def _fit_log_envelope(peaks: NDArray) -> float | None:
    """Least-squares slope of ``ln|peak|`` against time, i.e. ``-gamma``."""
    if peaks.shape[0] < 3:
        return None
    magnitude = peaks[:, 1]
    positive = magnitude > 0.0
    if np.count_nonzero(positive) < 3:
        return None
    times = peaks[positive, 0]
    values = np.log(magnitude[positive])
    design = np.column_stack((times, np.ones_like(times)))
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return float(-coefficients[0])


def _fit_decaying_sinusoid(
    time: FloatArray,
    signal: FloatArray,
    *,
    omega_guess: float,
    gamma_guess: float,
    omega_span: float = 0.06,
    gamma_bounds: tuple[float, float] | None = None,
) -> dict:
    """Joint least-squares fit of ``exp(-gamma t)(a cos + b sin)``.

    ``omega`` and ``gamma`` are the nonlinear parameters; for each trial pair the
    two amplitudes are solved exactly by linear least squares and the residual is
    returned to a Nelder-Mead simplex. The decay rate is *fitted*, not read off a
    log-envelope: successive envelope extrema are biased as soon as the record is
    not a perfect exponential, and that bias is several times the five percent
    this module has to resolve. (A hand-rolled grid search was tried first and
    was caught by the negative control: on a pure undamped sinusoid it reported
    ``zeta = 0.004`` because a slightly mistuned frequency can be traded against a
    spurious decay in the residual.)
    """
    from scipy.optimize import minimize

    omega_guess = float(omega_guess)
    gamma_guess = float(gamma_guess)
    low, high = omega_guess * (1.0 - omega_span), omega_guess * (1.0 + omega_span)
    if gamma_bounds is None:
        gamma_bounds = (0.0, max(4.0 * gamma_guess, 0.02 * omega_guess))
    span = float(np.max(np.abs(signal)))
    if span == 0.0:
        raise ValueError("the signal is identically zero")

    def amplitudes(omega: float, gamma: float) -> tuple[FloatArray, float]:
        envelope = np.exp(-gamma * time)
        basis = np.column_stack((np.cos(omega * time), np.sin(omega * time))) * envelope[:, None]
        coefficients, *_ = np.linalg.lstsq(basis, signal, rcond=None)
        residual = basis @ coefficients - signal
        return coefficients, float(residual @ residual) / (signal.size * span ** 2)

    def objective(parameters: Sequence[float]) -> float:
        omega, gamma = float(parameters[0]), float(parameters[1])
        if not (low <= omega <= high) or not (gamma_bounds[0] <= gamma <= gamma_bounds[1]):
            return 1.0e6
        return amplitudes(omega, gamma)[1]

    start = np.array(
        [min(max(omega_guess, low), high), min(max(gamma_guess, gamma_bounds[0]), gamma_bounds[1])]
    )
    result = minimize(
        objective,
        start,
        method="Nelder-Mead",
        bounds=[(low, high), gamma_bounds],
        options={"xatol": 1.0e-10, "fatol": 1.0e-16, "maxiter": 20000},
    )
    omega, gamma = float(result.x[0]), float(result.x[1])
    coefficients, residual = amplitudes(omega, gamma)
    return {
        "omega": omega,
        "gamma": max(gamma, 0.0),
        "cosine": float(coefficients[0]),
        "sine": float(coefficients[1]),
        "residual_over_amplitude": float(math.sqrt(max(residual, 0.0))),
        "iterations": int(result.nit),
        "converged": bool(result.success),
    }


def decay_ratio_from_series(
    time: FloatArray,
    signal: FloatArray,
    *,
    window: tuple[float, float] | None = None,
    omega_guess: float | None = None,
) -> dict:
    """Decay ratio ``zeta = gamma / omega`` of one modal coordinate trace.

    ``omega`` and ``gamma`` are fitted jointly (:func:`_fit_decaying_sinusoid`)
    on the requested window, starting from the spectral peak and the
    log-envelope slope. The normalised fit residual is reported next to the
    numbers, because a two-mode record fitted by one mode is exactly the
    situation in which a measured ``zeta`` means nothing.
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if window is None:
        window = (float(time[0]), float(time[-1]))
    mask = (time >= window[0]) & (time <= window[1])
    t = time[mask] - window[0]
    y = signal[mask]
    if t.size < 32:
        raise ValueError("the decay window is too short")
    frequency = float(omega_guess) if omega_guess is not None else _dominant_frequency(t, y)
    if frequency is None:
        raise ValueError("the record does not contain a resolvable oscillation")
    envelope = _fit_log_envelope(_envelope_peaks(t, y))
    gamma_guess = envelope if envelope is not None and envelope > 0.0 else 0.02 * frequency
    fit = _fit_decaying_sinusoid(t, y, omega_guess=frequency, gamma_guess=gamma_guess)
    return {
        "omega": fit["omega"],
        "gamma": fit["gamma"],
        "zeta": float(fit["gamma"] / fit["omega"]) if fit["omega"] else 0.0,
        "omega_initial": frequency,
        "gamma_initial": float(gamma_guess),
        "fit_residual_over_amplitude": float(fit["residual_over_amplitude"]),
        "window": (float(window[0]), float(window[1])),
        "samples": int(t.size),
    }


# ---------------------------------------------------------------------------
# the cheap substitute: A = A_inf, B = B(omega_n)
# ---------------------------------------------------------------------------
def constant_matrix_envelope(
    omega: Sequence[float],
    added_mass: Sequence[Sequence[Sequence[float]]],
    damping: Sequence[Sequence[Sequence[float]]],
    added_mass_inf: Sequence[Sequence[float]],
    mass: Sequence[Sequence[float]],
    *,
    ratios: Sequence[float],
    targets: Sequence[float],
    mode: int = 0,
    samples: int = 801,
) -> dict:
    """Peak/phase error of the constant-coefficient model versus ``omega_p/omega_n``.

    The cheap time-domain radiation model keeps one added mass and one damping
    coefficient for the whole record, ``A = A(omega_n)``, ``B = B(omega_n)``. It
    is exact only if both are flat across the response band, so its error is a
    function of where the oscillator sits relative to the sweep -- which is what
    this envelope measures.

    The oscillator is set up **in modal coordinates** of the coupled system,
    because that is the only way the question "where does the resonance sit"
    has a single answer when the added-mass matrix has off-diagonal entries:

    * ``M_tot = M + A(omega_ref)`` is whitened (``y = root^{-1} x``), the
      stiffness is ``C = root diag(omega_k^2) root`` and the mode shape is
      ``phi_k = root e_k`` (so ``phi^T M_tot phi = 1``);
    * the force is the modal force ``f = M_tot phi_k`` and the observation is
      the modal coordinate ``q = f^T x``, giving a unit-modal-force,
      unit-modal-receptance oscillator;
    * for each ratio the driven mode's target is moved to ``omega_n =
      omega_p / ratio`` (the other mode stays at its own target), so the model's
      *own* natural frequency is ``omega_n`` and the comparison is not
      contaminated by a stiffness mistuning.

    Reported per ratio: ``peak_amplitude_error`` (at the exact resonance),
    ``peak_frequency_error``, ``phase_error_degrees`` (the model is pinned at
    -90 deg at ``omega_n``, so the reported number is the exact phase plus 90),
    ``damping_ratio_error`` between the frozen and the frequency-dependent
    quadratic forms, and the same peak amplitude error seen on the physical
    degree of freedom ``mode`` (``dof_amplitude_error``), which is what a design
    engineer would actually read off.
    """
    w, a, b = validate_coupled_sweep(omega, added_mass, damping)
    infinity = np.asarray(added_mass_inf, dtype=float)
    structural = np.atleast_2d(np.asarray(mass, dtype=float))
    modes = b.shape[1]
    if structural.shape != (modes, modes) or infinity.shape != (modes, modes):
        raise ValueError("mass and added_mass_inf must be (n, n)")
    targets = np.asarray(targets, dtype=float).reshape(-1)
    if targets.size != modes:
        raise ValueError("targets must give one natural frequency per mode")
    ratio_list = np.asarray(ratios, dtype=float).reshape(-1)
    if ratio_list.size == 0 or np.any(ratio_list <= 0.0):
        raise ValueError("ratios must be a non-empty array of positive numbers")

    def added_at(value: float) -> FloatArray:
        return np.array([[np.interp(value, w, a[:, i, j]) for j in range(modes)] for i in range(modes)])

    def damp_at(value: float) -> FloatArray:
        return np.array([[np.interp(value, w, b[:, i, j]) for j in range(modes)] for i in range(modes)])

    reference = added_at(0.5 * (targets[0] + targets[-1]))
    total = 0.5 * ((structural + reference) + (structural + reference).T)
    eigenvalues_m, vectors_m = np.linalg.eigh(total)
    if np.any(eigenvalues_m <= 0.0):
        raise ValueError("mass + added mass must be positive definite")
    root = (vectors_m * np.sqrt(eigenvalues_m)) @ vectors_m.T
    inverse_root = (vectors_m * (1.0 / np.sqrt(eigenvalues_m))) @ vectors_m.T

    peak_index = int(np.argmax(np.abs(b[:, mode, mode])))
    omega_p = float(w[peak_index])
    rows = []
    for ratio in np.sort(ratio_list)[::-1]:
        natural = float(omega_p / ratio)
        desired = targets.copy()
        desired[mode] = natural
        stiffness = root @ np.diag(desired ** 2) @ root
        # phi_k = M^{-1/2} e_k is the mode shape normalised so that
        # phi^T M phi = 1; the unit-norm modal force keeps the reported
        # receptance O(1) regardless of the mass-matrix scale.
        shape = inverse_root[:, mode]
        modal_force = total @ shape
        modal_force = modal_force / float(np.linalg.norm(modal_force))
        observation = modal_force
        scan = np.linspace(
            max(natural - 0.35 * natural, 1e-6), natural + 0.35 * natural, int(samples)
        )
        exact, exact_dof = [], []
        for value in scan:
            matrix = (
                -value ** 2 * (structural + added_at(value))
                + 1j * value * damp_at(value)
                + stiffness
            )
            solution = np.linalg.solve(matrix, modal_force)
            exact.append(complex(observation @ solution))
            exact_dof.append(complex(solution[mode]))
        exact = np.asarray(exact)
        exact_dof = np.asarray(exact_dof)
        peak = int(np.argmax(np.abs(exact)))
        peak_in_window = bool(0 < peak < exact.size - 1)
        exact_peak_amplitude = float(np.abs(exact[peak]))
        exact_peak_omega = float(scan[peak])
        at_natural = int(np.argmin(np.abs(scan - natural)))
        model_added = added_at(natural)
        model_damp = damp_at(natural)
        model_matrix = (
            -natural ** 2 * (structural + model_added) + 1j * natural * model_damp + stiffness
        )
        model_solution = np.linalg.solve(model_matrix, modal_force)
        model_amplitude = float(abs(complex(observation @ model_solution)))
        model_peak_matrix = (
            -exact_peak_omega ** 2 * (structural + model_added)
            + 1j * exact_peak_omega * model_damp
            + stiffness
        )
        model_at_peak = float(
            abs(complex(observation @ np.linalg.solve(model_peak_matrix, modal_force)))
        )
        model_dof_at_peak = float(abs(np.linalg.solve(model_peak_matrix, modal_force)[mode]))
        zeta_exact = modal_damping_ratio(
            natural, structural, model_added, model_damp, shape
        )["zeta"]
        zeta_model = modal_damping_ratio(
            natural, structural, infinity, model_damp, shape
        )["zeta"]
        rows.append(
            {
                "omega_p_over_omega_n": float(omega_p / natural),
                "omega_n": natural,
                "added_mass_relative_change": float(
                    abs(model_added[mode, mode] - infinity[mode, mode]) / abs(infinity[mode, mode])
                ),
                "exact_peak_amplitude": exact_peak_amplitude,
                "exact_peak_omega": exact_peak_omega,
                "peak_in_window": peak_in_window,
                "peak_amplitude_error": float(
                    abs(model_at_peak - exact_peak_amplitude) / exact_peak_amplitude
                )
                if peak_in_window
                else None,
                "natural_amplitude_error": float(
                    abs(model_amplitude - abs(exact[at_natural])) / abs(exact[at_natural])
                ),
                "amplitude_error_at_natural": float(
                    abs(model_amplitude - abs(exact[at_natural])) / abs(exact[at_natural])
                ),
                "dof_amplitude_error": float(
                    abs(model_dof_at_peak - abs(exact_dof[peak])) / abs(exact_dof[peak])
                )
                if peak_in_window
                else None,
                "dof_amplitude_error_at_natural": float(
                    abs(
                        abs(np.linalg.solve(model_matrix, modal_force)[mode])
                        - abs(exact_dof[at_natural])
                    )
                    / abs(exact_dof[at_natural])
                ),
                "peak_frequency_error": float((exact_peak_omega - natural) / exact_peak_omega),
                "phase_error_degrees": float(math.degrees(np.angle(exact[at_natural])) + 90.0),
                "damping_ratio_error": float(abs(zeta_model - zeta_exact) / abs(zeta_exact))
                if zeta_exact
                else math.inf,
                "zeta_exact": float(zeta_exact),
                "zeta_model": float(zeta_model),
                "resonance_shift_over_half_power_width": float(
                    abs(exact_peak_omega - natural) / (2.0 * zeta_exact * natural)
                )
                if zeta_exact
                else math.inf,
            }
        )
    def _worst(key: str) -> dict | None:
        usable = [row for row in rows if row[key] is not None]
        if not usable:
            return None
        return max(usable, key=lambda row: abs(row[key]))

    worst_amplitude = _worst("peak_amplitude_error")
    worst_natural = _worst("amplitude_error_at_natural")
    worst_phase = _worst("phase_error_degrees")
    return {
        "omega_p": omega_p,
        "mode": int(mode),
        "targets": targets.tolist(),
        "rows": rows,
        "worst_peak_amplitude_error": None
        if worst_amplitude is None
        else float(worst_amplitude["peak_amplitude_error"]),
        "worst_peak_amplitude_ratio": None
        if worst_amplitude is None
        else float(worst_amplitude["omega_p_over_omega_n"]),
        "worst_amplitude_error_at_natural": None
        if worst_natural is None
        else float(worst_natural["amplitude_error_at_natural"]),
        "worst_phase_error_degrees": None
        if worst_phase is None
        else float(abs(worst_phase["phase_error_degrees"])),
        "worst_phase_ratio": None
        if worst_phase is None
        else float(worst_phase["omega_p_over_omega_n"]),
    }


# ---------------------------------------------------------------------------
# the exact linear anchor: complex resonances of the frequency-domain system
# ---------------------------------------------------------------------------
def complex_resonances(
    omega: Sequence[float],
    added_mass: Sequence[Sequence[Sequence[float]]],
    damping: Sequence[Sequence[Sequence[float]]],
    added_mass_inf: Sequence[Sequence[float]],
    mass: Sequence[Sequence[float]],
    stiffness: Sequence[Sequence[float]],
    *,
    omega_window: tuple[float, float],
    gamma_window: tuple[float, float] = (0.0, 0.6),
    omega_samples: int = 80,
    gamma_samples: int = 24,
    iterations: int = 80,
) -> list[dict]:
    """All complex resonances in a window of the complex frequency plane.

    ``Omega = omega_r + i gamma`` with ``gamma > 0`` decays as ``exp(-gamma t)``,
    so ``zeta = gamma / |Omega|`` is the exact linear-theory damping ratio. The
    coefficients are evaluated at ``Re(Omega)``: ``A`` and ``B`` are real
    functions of a real argument, analytically continued (the standard
    complex-frequency convention for a causal real kernel).

    A single Newton iteration from a real guess cannot tell one root from
    another -- with strongly coupled coefficients it walks from the second guess
    onto the first root, which is how a "two mode" comparison silently becomes a
    one mode comparison. This scans ``|det|`` on ``omega_r x gamma``, takes the
    local minima as starts and polishes each with Newton, so "two modes" means
    two roots or an honest report that only one was found.
    """
    w, a, b = validate_coupled_sweep(omega, added_mass, damping)
    infinity = np.atleast_2d(np.asarray(added_mass_inf, dtype=float))
    structural = np.atleast_2d(np.asarray(mass, dtype=float))
    c = np.atleast_2d(np.asarray(stiffness, dtype=float))
    modes = b.shape[1]
    if structural.shape != (modes, modes) or c.shape != (modes, modes):
        raise ValueError("mass, added_mass_inf and stiffness must match the number of modes")
    if infinity.shape != (modes, modes):
        raise ValueError("added_mass_inf must have shape (n, n)")

    def coefficients(value: float) -> tuple[FloatArray, FloatArray]:
        added = np.array(
            [[np.interp(value, w, a[:, i, j]) for j in range(modes)] for i in range(modes)]
        )
        damp = np.array(
            [[np.interp(value, w, b[:, i, j]) for j in range(modes)] for i in range(modes)]
        )
        return added, damp

    def determinant(value: complex) -> complex:
        added, damp = coefficients(float(abs(value)))
        # ``added`` is the *total* added mass at this frequency; adding the
        # asymptotic part on top of it would count it twice.
        matrix = -value ** 2 * (structural + added) + 1j * value * damp + c
        return complex(np.linalg.det(matrix))

    low, high = float(omega_window[0]), float(omega_window[1])
    gamma_low, gamma_high = float(gamma_window[0]), max(float(gamma_window[1]), 1e-6)
    grid_real = np.linspace(low, high, int(omega_samples))
    grid_gamma = np.linspace(gamma_low, gamma_high, int(gamma_samples))
    magnitude = np.array(
        [[abs(determinant(complex(r, g))) for r in grid_real] for g in grid_gamma]
    )
    scale = float(np.max(magnitude)) if magnitude.size else 0.0
    if scale == 0.0:
        raise ValueError("the determinant vanishes on the whole scan window")
    order = np.argsort(magnitude.reshape(-1))
    # Starts must be spread over the window: the smallest values of |det| cluster
    # around one root, and taking them all would leave the second mode with no
    # starting point at all (which is how a "two mode" test silently becomes a
    # one mode test).
    spacing_real = 0.05 * (high - low)
    spacing_gamma = 0.05 * (gamma_high - gamma_low)
    starts: list[complex] = []
    for flat in order:
        row, column = np.unravel_index(int(flat), magnitude.shape)
        candidate = complex(grid_real[column], grid_gamma[row])
        if any(
            abs(candidate.real - other.real) < spacing_real
            and abs(candidate.imag - other.imag) < spacing_gamma
            for other in starts
        ):
            continue
        starts.append(candidate)
        if len(starts) >= 6 * modes:
            break
    found: list[dict] = []
    for root in starts:
        try:
            for _ in range(int(iterations)):
                step = 1.0e-6 * abs(root)
                derivative = (determinant(root + step) - determinant(root - step)) / (2.0 * step)
                if derivative == 0.0:
                    raise ValueError("degenerate derivative")
                increment = determinant(root) / derivative
                root = root - increment
                if abs(increment) <= 1.0e-13 * abs(root):
                    break
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not (0.0 < root.real < 4.0 * high):
            continue
        if abs(determinant(root)) > 1.0e-6 * scale:
            continue
        if any(abs(root - other["root"]) < 1.0e-3 * abs(root) for other in found):
            continue
        added, damp = coefficients(float(abs(root)))
        matrix = -root ** 2 * (structural + added) + 1j * root * damp + c
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        index = int(np.argmin(np.abs(eigenvalues)))
        shape = eigenvectors[:, index] / np.max(np.abs(eigenvectors[:, index]))
        found.append(
            {
                "root": complex(root),
                "omega": float(root.real),
                "gamma": float(root.imag),
                "zeta": float(abs(root.imag) / abs(root)),
                "residual_determinant": float(abs(determinant(root))),
                "relative_residual": float(abs(determinant(root)) / scale),
                "mode_shape": shape.tolist(),
                "singular_value": float(np.min(np.abs(eigenvalues))),
            }
        )
    found.sort(key=lambda item: item["omega"])
    return found


def complex_resonance(
    omega: Sequence[float],
    added_mass: Sequence[Sequence[Sequence[float]]],
    damping: Sequence[Sequence[Sequence[float]]],
    added_mass_inf: Sequence[Sequence[float]],
    mass: Sequence[Sequence[float]],
    stiffness: Sequence[Sequence[float]],
    *,
    guess: float,
    iterations: int = 80,
    tolerance: float = 1.0e-12,
) -> dict:
    """Complex root of ``det[-Omega^2 (M + A) + i Omega B + C] = 0`` near ``guess``.

    Newton with a numerical derivative, starting from ``(guess, 0.02 guess)``.
    For several roots at once -- and a check that the roots are distinct -- use
    :func:`complex_resonances`, which scans first.
    """
    w, a, b = validate_coupled_sweep(omega, added_mass, damping)
    infinity = np.atleast_2d(np.asarray(added_mass_inf, dtype=float))
    structural = np.atleast_2d(np.asarray(mass, dtype=float))
    c = np.atleast_2d(np.asarray(stiffness, dtype=float))
    modes = b.shape[1]
    if structural.shape != (modes, modes) or c.shape != (modes, modes):
        raise ValueError("mass, added_mass_inf and stiffness must match the number of modes")
    if infinity.shape != (modes, modes):
        raise ValueError("added_mass_inf must have shape (n, n)")

    def coefficients(value: float) -> tuple[FloatArray, FloatArray]:
        added = np.array(
            [[np.interp(value, w, a[:, i, j]) for j in range(modes)] for i in range(modes)]
        )
        damp = np.array(
            [[np.interp(value, w, b[:, i, j]) for j in range(modes)] for i in range(modes)]
        )
        return added, damp

    def determinant(value: complex) -> complex:
        added, damp = coefficients(float(abs(value)))
        matrix = -value ** 2 * (structural + added) + 1j * value * damp + c
        return complex(np.linalg.det(matrix))

    root = complex(float(guess), 0.02 * float(guess))
    used = 0
    for used in range(1, int(iterations) + 1):
        step = 1.0e-6 * abs(root)
        derivative = (determinant(root + step) - determinant(root - step)) / (2.0 * step)
        if derivative == 0.0:
            raise ValueError("the determinant derivative vanished; the guess is degenerate")
        increment = determinant(root) / derivative
        root = root - increment
        if abs(increment) <= tolerance * abs(root):
            break
    added, damp = coefficients(float(abs(root)))
    matrix = -root ** 2 * (structural + added) + 1j * root * damp + c
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    index = int(np.argmin(np.abs(eigenvalues)))
    shape = eigenvectors[:, index] / np.max(np.abs(eigenvectors[:, index]))
    return {
        "root": complex(root),
        "omega": float(root.real),
        "gamma": float(root.imag),
        "zeta": float(abs(root.imag) / abs(root)),
        "residual_determinant": float(abs(determinant(root))),
        "mode_shape": shape.tolist(),
        "singular_value": float(np.min(np.abs(eigenvalues))),
        "iterations_used": int(used),
    }


def kernel_resonances(
    kernel: CoupledKernelMatrix,
    added_mass_inf: Sequence[Sequence[float]],
    *,
    mass: Sequence[Sequence[float]],
    stiffness: Sequence[Sequence[float]],
    guesses: Sequence[float],
    iterations: int = 200,
) -> list[dict]:
    """Complex resonances *of the kernel itself*, without the sweep.

    The time-domain Cummins system is LTI, so an exponential
    ``x(t) = Re{v exp(i Omega t)}`` is a solution when

        [-Omega^2 (M + A_inf) + i Omega F(Omega) + C] v = 0 ,
        F(Omega) = int_0^inf K(tau) exp(-i Omega tau) dtau .

    ``F`` is evaluated directly from the sampled kernel (trapezoid), so this root
    is the decay rate the discretised kernel implies -- no reference to
    ``A(omega)``/``B(omega)`` on the real axis and no analytic continuation
    approximation. Comparing a time-marched decay rate with *this* root isolates
    quadrature error; comparing it with the single-frequency quadratic form
    ``v^T B v / (2 omega v^T (M + A) v)`` measures the approximation of the
    anchor itself. The two are different questions and the campaign asks both.
    """
    n = kernel.modes
    inertia = np.atleast_2d(np.asarray(mass, dtype=float)) + np.atleast_2d(
        np.asarray(added_mass_inf, dtype=float)
    )
    stiff = np.atleast_2d(np.asarray(stiffness, dtype=float))
    if inertia.shape != (n, n) or stiff.shape != (n, n):
        raise ValueError(f"mass, added_mass_inf and stiffness must all be ({n}, {n})")
    weights = np.full(kernel.tau.size, kernel.dt)
    weights[0] *= 0.5
    weights[-1] *= 0.5

    def transfer(value: complex) -> NDArray:
        phase = np.exp(-1j * value * kernel.tau) * weights
        return np.einsum("t,tij->ij", phase, kernel.values)

    def determinant(value: complex) -> complex:
        return complex(np.linalg.det(-value ** 2 * inertia + 1j * value * transfer(value) + stiff))

    found: list[dict] = []
    for guess in guesses:
        root = complex(float(guess), 0.05 * float(guess))
        for _ in range(int(iterations)):
            step = 1.0e-6 * abs(root)
            derivative = (determinant(root + step) - determinant(root - step)) / (2.0 * step)
            if derivative == 0.0:
                break
            increment = determinant(root) / derivative
            root = root - increment
            if abs(increment) <= 1.0e-14 * abs(root):
                break
        if not (root.real > 0.0):
            raise ValueError(f"the kernel root starting from {guess} left the physical half-plane")
        if any(abs(root - other["root"]) < 1.0e-4 * abs(root) for other in found):
            continue
        found.append(
            {
                "root": complex(root),
                "omega": float(root.real),
                "gamma": float(root.imag),
                "zeta": float(abs(root.imag) / abs(root)),
                "residual_determinant": float(abs(determinant(root))),
            }
        )
    found.sort(key=lambda item: item["omega"])
    return found


def decay_ratio_richardson(
    kernel: CoupledKernelMatrix,
    added_mass_inf: Sequence[Sequence[float]],
    *,
    mass: Sequence[Sequence[float]],
    stiffness: Sequence[Sequence[float]],
    initial_displacement: Sequence[float],
    initial_velocity: Sequence[float] | None = None,
    duration: float,
    dt_target: float,
    window: tuple[float, float] = (0.25, 0.98),
    omega_guess: float | None = None,
) -> dict:
    """Free-decay ratio with a Richardson extrapolation in the time step.

    The trapezoid memory sum has a *first-order* global error: each step's local
    ``O(dt^2)`` error is accumulated over ``t/dt`` steps, which biases the decay
    rate by a few percent at the steps a seakeeping run would use (measured: 11%
    on the synthetic Kramers-Kronig pair at ``period/120``, and the bias vanishes
    linearly as ``dt -> 0``). Reporting a single-step answer would therefore
    report the quadrature, not the physics, so the rate is measured at ``dt`` and
    ``2 dt`` and extrapolated linearly; the difference between the two is
    returned as the numerical error bar.
    """
    results = {}
    for label, step in (("fine", float(dt_target)), ("coarse", 2.0 * float(dt_target))):
        work = _refine_kernel(kernel, step)
        run = coupled_free_decay(
            work,
            added_mass_inf,
            mass=mass,
            stiffness=stiffness,
            initial_displacement=initial_displacement,
            initial_velocity=initial_velocity,
            duration=duration,
        )
        displacement = np.asarray(initial_displacement, dtype=float)
        inertia = np.atleast_2d(np.asarray(mass, dtype=float)) + np.atleast_2d(
            np.asarray(added_mass_inf, dtype=float)
        )
        normaliser = float(displacement @ inertia @ displacement)
        projection = (run["displacement"] @ inertia @ displacement) / normaliser
        measured = decay_ratio_from_series(
            run["time"],
            projection,
            window=(window[0] * duration, window[1] * duration),
            omega_guess=omega_guess,
        )
        measured["dt"] = float(work.dt)
        measured["time"] = run["time"]
        measured["projection"] = projection
        results[label] = measured
    fine, coarse = results["fine"], results["coarse"]
    extrapolated = 2.0 * fine["zeta"] - coarse["zeta"]
    return {
        "zeta": float(extrapolated),
        "omega": float(fine["omega"]),
        "gamma": float(fine["gamma"]),
        "zeta_at_dt": fine["zeta"],
        "zeta_at_2dt": coarse["zeta"],
        "dt": fine["dt"],
        "drift_over_zeta": float(abs(fine["zeta"] - coarse["zeta"]) / abs(fine["zeta"]))
        if fine["zeta"]
        else 0.0,
        "fit_residual_over_amplitude": fine["fit_residual_over_amplitude"],
        "time": fine["time"],
        "projection": fine["projection"],
    }
