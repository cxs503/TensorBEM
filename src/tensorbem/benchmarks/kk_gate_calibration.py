"""Error-injection calibration of the Kramers-Kronig gate (case C6b, "not certified").

Why this module exists
----------------------

The Cummins/Wigley case judges causality with gate ``C3`` (KK closure of the
added mass) and referees the *gate itself* with the negative control ``C6b``:
the damping shape is corrupted and the closure must then report a residual
above the 10 % bar.  ``C6b`` passes on the committed sweep, but the artifact
carries the honest label ``"pass, not certified"``: on the 0.2-12 rad/s band no
DOF's damping has entered algebraic decay, so the truncation budget of the
omitted ``s > s_max`` integral is unbounded and the corrupted residual
(20.6 % on surge) cannot be attributed to the corruption *from the band-limited
data alone*.  The same unbounded budget is what keeps surge/sway/yaw out of gate
``C3a``'s certified window, and what the extended-band sweep (20 rad/s) is meant
to fix.

A wider band is not the only way to certify a detector.  What ``C6b`` claims is
a statement about the *estimator*, not about the hull: "if the damping shape is
perturbed by a known amount, the closure reports a residual that (a) grows with
the perturbation, (b) crosses the criterion bar below the perturbation actually
injected, and (c) is not an artefact of the estimator's own numerical bias".
Those three claims can be measured on data whose answer is known exactly --
:class:`CausalPair` is an exact radiation pair with closed-form
``(A(omega), B(omega))`` satisfying the KK relations identically (a
two-exponential memory kernel), so a residual the estimator reports on it is the
estimator's, not physics'.  Injecting a controlled corruption into that pair and
re-running the very same closure (``kk_added_mass_closure``) gives a
*calibration curve* of reported residual versus injected error, hence a
detection threshold in the same units the gate reports, hence a finite statement
about ``C6b``'s detection power that no amount of un-solved band can take away.

Channels calibrated (each with an exactly known truth):

``damping_step``
    ``B -> B * (1 + delta)`` above ``omega_step``.  This is *verbatim* the
    corruption ``kk_gate(corrupt=True)`` applies to the case sweep
    (``delta = 0.30`` above 1 rad/s), so the curve predicts the residual
    ``C6b`` should report and the smallest ``delta`` it can see.

``added_mass_bump``
    ``A -> A + delta * scale * w(omega)`` for a smooth in-band bump ``w`` with
    ``max|w| = 1``.  This is the channel the gate *measures* directly, so it is
    the natural unit in which to express sensitivity.  It also exposes the one
    free parameter of the closure: the least-squares ``A_inf`` offset absorbs the
    mean of the injected error, so a one-signed bump is under-reported by its own
    mean; a zero-mean shape is reported alongside as the best case.

``damping_bump``
    ``B -> B * (1 + delta * w(omega))`` localised in frequency.  KK is an
    integral transform, so a narrow in-band corruption of the damping is
    *attenuated* in the added mass the gate inspects.  This channel quantifies
    the gate's blind spot, which is the honest reading of the case's rule that
    "the corruption has to be in shape": a shape error local in frequency is not
    what ``C6b`` catches.

Outputs are a JSON certificate plus a gate record shaped like the case's other
gates (``id``/``title``/``threshold``/``passed``/``values``), so the case verdict
can carry the calibration next to ``C6b`` instead of the bare "not certified"
label.

Honest boundaries
-----------------

* The calibration pair is a *transfer function*, not a hull.  It is a valid
  causal radiation pair with ``B > 0``, ``B ~ omega^2`` at low frequency and a
  hump at ``omega = 4`` rad/s inside the band, so the estimator sees a
  body-like sweep; its coefficients are not claimed as data for the Wigley hull.
* The calibration certifies the *detector*, not the case's data.  It says what
  residual a given shape corruption produces in this estimator on this grid; it
  does not turn the case's unbounded truncation budget into a bounded one.  The
  two statements are disjoint and both are needed.
* The pair's damping decays as ``omega^-2``, so the power-tail fit is rejected
  exactly as it is on the case sweep (the admissibility rule is ``p > 2.05``).
  The estimator therefore runs here in the same configuration as on the case: no
  tail model, truncation bias present and reported as the ``floor`` term.  A
  residual is only called a *detection* above that floor.

Entry point::

    python -m tensorbem.benchmarks.kk_gate_calibration --out <dir>
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.causality import fit_damping_power_tail, kk_added_mass_closure

__all__ = [
    "CALIBRATION_BAND",
    "CALIBRATION_N_OMEGA",
    "C6B_INJECTED_STEP",
    "C6B_STEP_OMEGA",
    "KK_BAR",
    "CausalPair",
    "CALIBRATION_PAIR",
    "calibration_grid",
    "closure_residual",
    "bump_shape",
    "corrupt_damping_step",
    "corrupt_damping_bump",
    "corrupt_added_mass_bump",
    "calibration_curve",
    "recovery_slope",
    "detection_threshold",
    "calibration_gate",
    "certificate",
    "render_markdown",
    "main",
]

#: Same band and node count as the committed case sweep, so the estimator runs on
#: the same grid it is asked about (``case_bem_cummins_wigley.BAND`` /
#: ``N_OMEGA``; duplicated as literals so this module does not import the
#: cluster-facing solver stack).
CALIBRATION_BAND = (0.2, 12.0)
CALIBRATION_N_OMEGA = 41

#: The corruption gate ``C6b`` actually applies to the case sweep, and where.
C6B_INJECTED_STEP = 0.30
C6B_STEP_OMEGA = 1.0

#: Case criterion the gate is judged against (C3/C6b share the 10 % bar).
KK_BAR = 0.10


@dataclass(frozen=True)
class CausalPair:
    """Exact radiation pair from a two-exponential memory kernel.

    ``K(s) = c1 exp(-a1 s) + c2 exp(-a2 s)`` gives (see the
    :mod:`tensorbem.causality` docstring and ``tests/test_causality.py``)

        B(omega)         = sum_j c_j a_j / (a_j^2 + omega^2)
        A(omega) - A_inf = -sum_j c_j / (a_j^2 + omega^2)

    identically, i.e. the pair satisfies the KK relations by construction and any
    residual a closure reports on it is numerical.
    """

    name: str
    coefficients: tuple[float, ...]
    rates: tuple[float, ...]
    added_mass_inf: float

    def damping(self, omega: Sequence[float] | np.ndarray) -> np.ndarray:
        w = np.asarray(omega, dtype=float)
        return sum(
            c * a / (a * a + w * w) for c, a in zip(self.coefficients, self.rates)
        )

    def added_mass(self, omega: Sequence[float] | np.ndarray) -> np.ndarray:
        w = np.asarray(omega, dtype=float)
        return self.added_mass_inf - sum(
            c / (a * a + w * w) for c, a in zip(self.coefficients, self.rates)
        )

    def damping_peak_omega(self) -> float:
        """Frequency of the damping hump, from the closed form of |B|.

        ``B(omega) = N(omega^2) / prod_j (a_j^2 + omega^2)`` with
        ``N(u) = u * sum_j c_j a_j * prod_{k != j} (a_k^2 + u)``; the hump is
        located by a fine scan (the closed form's derivative is a cubic in
        ``u`` and not worth solving in closed form for a diagnostic).
        """
        grid = np.logspace(math.log10(1e-3), math.log10(1e3), 20001)
        values = self.damping(grid)
        return float(grid[int(np.argmax(values))])

    def summary(self) -> dict[str, Any]:
        grid = calibration_grid()
        low, high = float(grid[0]), float(grid[-1])
        bump = self.damping_peak_omega()
        return {
            "name": self.name,
            "coefficients": list(self.coefficients),
            "rates": list(self.rates),
            "added_mass_inf": self.added_mass_inf,
            "kernel": "sum_j c_j exp(-a_j s)  (exact causal memory kernel)",
            "damping_at_band_ends": [
                float(self.damping(np.array([low]))[0]),
                float(self.damping(np.array([high]))[0]),
            ],
            "damping_hump_omega": bump,
            "damping_max_on_band": float(np.max(np.abs(self.damping(grid)))),
            "added_mass_amplitude_on_band": float(
                np.max(np.abs(self.added_mass(grid) - self.added_mass_inf))
            ),
            "damping_positive_on_band": bool(np.all(self.damping(grid) > 0.0)),
        }


#: ``c = (-8, 32)``, ``a = (2, 8)``: ``sum_j c_j/a_j = -4 + 4 = 0``, so
#: ``B(0) = 0`` and ``B(omega) = 240 omega^2 / ((omega^2+4)(omega^2+64)) > 0`` with
#: a hump at ``omega = 4`` rad/s and a ``omega^-2`` tail -- a body-like damping
#: curve on the case band, and the exact pair of the KK relations.
#: ``A(omega) - A_inf = (384 - 24 omega^2)/((omega^2+4)(omega^2+64))``.
CALIBRATION_PAIR = CausalPair(
    name="two-exponential (c=(-8,32), a=(2,8))",
    coefficients=(-8.0, 32.0),
    rates=(2.0, 8.0),
    added_mass_inf=3.0,
)


def calibration_grid(
    band: tuple[float, float] = CALIBRATION_BAND,
    n_omega: int = CALIBRATION_N_OMEGA,
) -> np.ndarray:
    """Log-spaced nodes with both band ends included, as in the case sweep."""
    low, high = (float(band[0]), float(band[1]))
    if not (0.0 < low < high):
        raise ValueError(f"band must satisfy 0 < low < high, got {band!r}")
    count = int(n_omega)
    if count < 3:
        raise ValueError("n_omega must be at least 3")
    grid = np.logspace(math.log10(low), math.log10(high), count)
    grid[0], grid[-1] = low, high
    return grid


def closure_residual(
    omega: Sequence[float] | np.ndarray,
    added_mass: Sequence[float] | np.ndarray,
    damping: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Run the case's KK closure and report the statistic the gate reports.

    ``max|A_measured - A_reconstructed| / max|A_measured|`` over the default
    sample frequencies -- the same normalisation as ``kk_gate`` (``C3``/``C6b``),
    so numbers here are directly comparable to the case artifact.  The tail model
    is fitted exactly as the case does it (and rejected when inadmissible).
    """
    grid = np.asarray(omega, dtype=float)
    mass = np.asarray(added_mass, dtype=float)
    damper = np.asarray(damping, dtype=float)
    tail = fit_damping_power_tail(grid, damper)
    closure = kk_added_mass_closure(grid, mass, damper, damping_tail=tail)
    offset = float(closure.added_mass_inf)
    amplitude = float(np.max(np.abs(mass - offset)))
    absolute = float(np.max(np.abs(closure.residual))) if closure.residual.size else 0.0
    return {
        "max_relative_residual": closure.max_relative_residual,
        "amplitude_normalised_residual": (
            absolute / amplitude if amplitude > 0.0 else float("inf")
        ),
        "added_mass_amplitude": amplitude,
        "dilution_amplitude_over_scale": (
            amplitude / float(closure.added_mass_scale)
            if closure.added_mass_scale > 0.0
            else float("nan")
        ),
        "added_mass_scale": float(closure.added_mass_scale),
        "added_mass_inf": float(closure.added_mass_inf),
        "sample_omegas": closure.sample_omegas.tolist(),
        "relative_residual": closure.relative_residual.tolist(),
        "damping_tail_exponent": float(tail.exponent),
        "damping_tail_valid": bool(tail.valid),
        "damping_tail_reason": tail.reason,
    }


def bump_shape(
    omega: Sequence[float] | np.ndarray,
    *,
    centre: float = 4.0,
    width: float = 0.6,
    zero_mean: bool = False,
) -> np.ndarray:
    """Smooth in-band bump in ``log(omega)`` with ``max|w| = 1``.

    ``zero_mean`` subtracts the (sample-weighted) mean so the injected error is
    invisible to the ``A_inf`` least-squares offset -- the best case for the
    detector, and the configuration that isolates the closure's *shape*
    sensitivity from its one free parameter.
    """
    w = np.asarray(omega, dtype=float)
    shape = np.exp(-0.5 * (np.log(w / float(centre)) / float(width)) ** 2)
    shape = shape / float(np.max(np.abs(shape)))
    if zero_mean:
        shape = shape - float(np.mean(shape))
        peak = float(np.max(np.abs(shape)))
        if peak > 0.0:
            shape = shape / peak
    return shape


def corrupt_damping_step(
    omega: Sequence[float] | np.ndarray,
    damping: Sequence[float] | np.ndarray,
    step: float,
    *,
    omega_step: float = C6B_STEP_OMEGA,
) -> np.ndarray:
    """``B * (1 + step)`` above ``omega_step`` -- verbatim the ``C6b`` corruption."""
    grid = np.asarray(omega, dtype=float)
    damper = np.asarray(damping, dtype=float)
    return damper * np.where(grid > float(omega_step), 1.0 + float(step), 1.0)


def corrupt_damping_bump(
    omega: Sequence[float] | np.ndarray,
    damping: Sequence[float] | np.ndarray,
    delta: float,
    *,
    centre: float = 4.0,
    width: float = 0.6,
) -> np.ndarray:
    """``B * (1 + delta * w)``: a corruption localised in frequency."""
    grid = np.asarray(omega, dtype=float)
    damper = np.asarray(damping, dtype=float)
    shape = bump_shape(grid, centre=centre, width=width)
    return damper * (1.0 + float(delta) * shape)


def corrupt_added_mass_bump(
    omega: Sequence[float] | np.ndarray,
    added_mass: Sequence[float] | np.ndarray,
    delta: float,
    *,
    scale: float,
    centre: float = 4.0,
    width: float = 0.6,
    zero_mean: bool = False,
) -> np.ndarray:
    """``A + delta * scale * w``: an added-mass error of known relative size."""
    grid = np.asarray(omega, dtype=float)
    mass = np.asarray(added_mass, dtype=float)
    shape = bump_shape(grid, centre=centre, width=width, zero_mean=zero_mean)
    return mass + float(delta) * float(scale) * shape


def calibration_curve(
    channel: str,
    deltas: Sequence[float],
    *,
    pair: CausalPair = CALIBRATION_PAIR,
    band: tuple[float, float] = CALIBRATION_BAND,
    n_omega: int = CALIBRATION_N_OMEGA,
    bar: float = KK_BAR,
) -> dict[str, Any]:
    """Sweep the injected ``delta`` through one channel and measure the residual.

    ``delta = 0`` is included whether or not it is asked for: it is the *floor*,
    i.e. the residual the estimator reports on an exact pair, and without it a
    small injected error cannot be told from estimator bias.
    """
    grid = calibration_grid(band, n_omega)
    truth_mass = pair.added_mass(grid)
    truth_damper = pair.damping(grid)
    scale = float(np.max(np.abs(truth_mass - pair.added_mass_inf)))

    points: list[dict[str, Any]] = []
    grid_deltas = sorted({0.0, *(float(d) for d in deltas)})
    for delta in grid_deltas:
        if channel == "damping_step":
            mass = truth_mass
            damper = corrupt_damping_step(grid, truth_damper, delta)
        elif channel == "damping_bump":
            mass = truth_mass
            damper = corrupt_damping_bump(grid, truth_damper, delta)
        elif channel == "added_mass_bump":
            mass = corrupt_added_mass_bump(grid, truth_mass, delta, scale=scale)
            damper = truth_damper
        elif channel == "added_mass_bump_zero_mean":
            mass = corrupt_added_mass_bump(
                grid, truth_mass, delta, scale=scale, zero_mean=True
            )
            damper = truth_damper
        else:
            raise ValueError(f"unknown channel {channel!r}")
        result = closure_residual(grid, mass, damper)
        points.append(
            {
                "delta": float(delta),
                "reported_residual": float(result["max_relative_residual"]),
                "amplitude_normalised_residual": float(
                    result["amplitude_normalised_residual"]
                ),
                "dilution_amplitude_over_scale": float(
                    result["dilution_amplitude_over_scale"]
                ),
                "detected": bool(result["max_relative_residual"] > float(bar)),
                "detected_amplitude_normalised": bool(
                    result["amplitude_normalised_residual"] > float(bar)
                ),
            }
        )
    floor = next(p["reported_residual"] for p in points if p["delta"] == 0.0)
    floor_amplitude = next(
        p["amplitude_normalised_residual"] for p in points if p["delta"] == 0.0
    )
    for point in points:
        point["above_floor"] = float(point["reported_residual"] - floor)
        point["above_floor_amplitude_normalised"] = float(
            point["amplitude_normalised_residual"] - floor_amplitude
        )
    return {
        "channel": channel,
        "band": [float(band[0]), float(band[1])],
        "n_omega": int(n_omega),
        "bar": float(bar),
        "floor": float(floor),
        "floor_amplitude_normalised": float(floor_amplitude),
        "normalisations": {
            "reported_residual": "max|A_measured - A_KK| / max|A_measured|  (case statistic)",
            "amplitude_normalised_residual": (
                "max|A_measured - A_KK| / max|A_measured - A_inf|, A_inf from the "
                "closure (invariant to the arbitrary instantaneous offset)"
            ),
        },
        "residual_scale": "max |A_measured - A_inf| on the band",
        "points": points,
        "damping_tail_valid": bool(result["damping_tail_valid"]),
        "damping_tail_exponent": float(result["damping_tail_exponent"]),
        "damping_tail_reason": result["damping_tail_reason"],
    }


def recovery_slope(curve: dict[str, Any]) -> dict[str, Any]:
    """Least-squares slope of ``residual - floor`` against ``delta``.

    A slope near 1 means the gate *reports the injected error as it is*; a slope
    well below 1 means it under-reports, which is the quantitative form of
    "passing C3 is not evidence that A is right to better than (slope x bar)".
    Only strictly positive deltas above the floor are used; monotonicity is
    checked separately because a non-monotone detector can pass a corruption
    sized between two points of the grid.
    """
    points = [p for p in curve["points"] if p["delta"] > 0.0]
    if len(points) < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "n_points": len(points)}
    x = np.array([p["delta"] for p in points], dtype=float)
    y = np.array([p["above_floor"] for p in points], dtype=float)
    design = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ np.array([slope, intercept])
    residual = float(np.max(np.abs(prediction - y))) if y.size else 0.0
    ordered = [p["reported_residual"] for p in sorted(points, key=lambda p: p["delta"])]
    monotone = all(b >= a for a, b in zip(ordered, ordered[1:]))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "n_points": int(len(points)),
        "max_abs_linear_residual": residual,
        "monotone_increasing": bool(monotone),
    }


def detection_threshold(
    curve: dict[str, Any], *, key: str = "reported_residual"
) -> dict[str, Any]:
    """Smallest injected ``delta`` whose residual crosses the criterion bar.

    Solved by linear interpolation between the bracketing grid points, which is
    the strongest statement the sampled curve supports; when no point crosses the
    bar the threshold is ``inf`` (the detector cannot see the bar at any injected
    size on this grid) rather than a guess.
    """
    bar = float(curve["bar"])
    points = sorted(curve["points"], key=lambda p: p["delta"])
    for lower, upper in zip(points, points[1:]):
        if lower[key] <= bar < upper[key]:
            span = upper[key] - lower[key]
            fraction = (bar - lower[key]) / span if span > 0 else 0.0
            delta = lower["delta"] + fraction * (upper["delta"] - lower["delta"])
            return {
                "delta": float(delta),
                "bracket": [lower["delta"], upper["delta"]],
                "bar": bar,
                "statistic": key,
                "source": "linear interpolation between bracketing grid points",
            }
    if points and points[0][key] > bar:
        return {
            "delta": 0.0,
            "bracket": [points[0]["delta"], points[0]["delta"]],
            "bar": bar,
            "statistic": key,
            "source": "the floor itself already exceeds the bar",
        }
    return {
        "delta": float("inf"),
        "bracket": None,
        "bar": bar,
        "statistic": key,
        "source": "no injected size on the grid crosses the bar",
    }


def _primary_curve(
    *, bar: float = KK_BAR
) -> dict[str, Any]:
    """Damping-step calibration wide enough to bracket the bar and the C6b step."""
    deltas = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.9)
    return calibration_curve("damping_step", deltas, bar=bar)


def localisation_sweep(
    *,
    widths: Sequence[float] = (0.15, 0.25, 0.4, 0.6, 1.0),
    deltas: Sequence[float] = (0.3, 0.5, 1.0, 2.0),
    bar: float = KK_BAR,
    pair: CausalPair = CALIBRATION_PAIR,
) -> dict[str, Any]:
    """How narrow may a damping corruption be and still pass the gate?

    ``C6b`` corrupts the damping above 1 rad/s, i.e. over most of the band.  The
    complementary question -- which the negative control does *not* ask -- is how
    a corruption confined to a narrow frequency window fares, because the KK
    transform integrates over frequency and a narrow bump contributes little.
    Each row is one ``(width, delta)`` pair; ``passes`` means the gate would have
    accepted the corrupted data.
    """
    grid = calibration_grid()
    truth_mass = pair.added_mass(grid)
    truth_damper = pair.damping(grid)
    rows: list[dict[str, Any]] = []
    for width in widths:
        for delta in deltas:
            damper = corrupt_damping_bump(grid, truth_damper, delta, width=float(width))
            result = closure_residual(grid, truth_mass, damper)
            rows.append(
                {
                    "log_width": float(width),
                    "delta": float(delta),
                    "reported_residual": float(result["max_relative_residual"]),
                    "amplitude_normalised_residual": float(
                        result["amplitude_normalised_residual"]
                    ),
                    "passes_bar": bool(result["max_relative_residual"] <= float(bar)),
                }
            )
    passing = [row for row in rows if row["passes_bar"]]
    worst_passing = (
        max(passing, key=lambda row: row["delta"]) if passing else None
    )
    return {
        "channel": "damping_bump_localisation",
        "bar": float(bar),
        "rows": rows,
        "worst_passing_corruption": worst_passing,
        "statement": (
            "a damping corruption confined to a narrow band can be large in "
            "relative terms and still pass the KK gate: the declared statistic "
            "integrates the corruption away"
        ),
    }


def certificate(*, bar: float = KK_BAR) -> dict[str, Any]:
    """Full calibration bundle: three channels, slopes, thresholds, C6b verdict."""
    primary = _primary_curve(bar=bar)
    added = calibration_curve(
        "added_mass_bump", (0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8), bar=bar
    )
    added_zm = calibration_curve(
        "added_mass_bump_zero_mean", (0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8), bar=bar
    )
    damping_local = calibration_curve(
        "damping_bump", (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2), bar=bar
    )
    channels = {
        "damping_step": primary,
        "added_mass_bump": added,
        "added_mass_bump_zero_mean": added_zm,
        "damping_bump": damping_local,
    }
    slopes = {name: recovery_slope(curve) for name, curve in channels.items()}
    thresholds = {
        name: {
            "reported": detection_threshold(curve, key="reported_residual"),
            "amplitude_normalised": detection_threshold(
                curve, key="amplitude_normalised_residual"
            ),
        }
        for name, curve in channels.items()
    }
    observed_c6b = float(
        next(
            p["reported_residual"]
            for p in primary["points"]
            if abs(p["delta"] - C6B_INJECTED_STEP) < 1e-12
        )
    )
    observed_c6b_amplitude = float(
        next(
            p["amplitude_normalised_residual"]
            for p in primary["points"]
            if abs(p["delta"] - C6B_INJECTED_STEP) < 1e-12
        )
    )
    reported_threshold = float(thresholds["damping_step"]["reported"]["delta"])
    amplitude_threshold = float(
        thresholds["damping_step"]["amplitude_normalised"]["delta"]
    )
    return {
        "case": "CASE-BEM-CUMMINS-WIGLEY",
        "purpose": (
            "quantify the detection power of the Kramers-Kronig gate that the "
            "negative control C6b exercises, on an exact causal pair, so that the "
            "'pass, not certified' label is replaced by a finite sensitivity "
            "statement that does not depend on the un-solved high band"
        ),
        "pair": CALIBRATION_PAIR.summary(),
        "criteria": {
            "kk_bar": float(bar),
            "tail_admissibility_rule": "power-tail fit p > 2.05, else no tail model",
            "case_c6b_injected_step": C6B_INJECTED_STEP,
            "case_c6b_step_omega": C6B_STEP_OMEGA,
        },
        "channels": channels,
        "localisation": localisation_sweep(bar=bar),
        "recovery_slope": slopes,
        "detection_threshold": thresholds,
        "predicted_c6b_residual": observed_c6b,
        "predicted_c6b_residual_amplitude_normalised": observed_c6b_amplitude,
        "c6b_detection_margin": (
            float(reported_threshold / C6B_INJECTED_STEP)
            if math.isfinite(reported_threshold) and C6B_INJECTED_STEP > 0
            else float("inf")
        ),
        "c6b_detection_margin_amplitude_normalised": (
            float(amplitude_threshold / C6B_INJECTED_STEP)
            if math.isfinite(amplitude_threshold) and C6B_INJECTED_STEP > 0
            else float("inf")
        ),
    }


def calibration_gate(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Gate record (``C6b_cal``) carrying the calibration into the case verdict.

    Three conjunctive claims, each independently checkable from the payload, and
    each stated *before* the numbers are read off:

    * **floor**: on an *exact* pair the estimator reports at most 25 % of the
      criterion -- otherwise no residual it reports is attributable to corruption
      at all, and a "detected" corruption is indistinguishable from bias;
    * **monotone and responsive**: on the damping-step channel, for injected
      sizes at or above 5 % (the region where the floor no longer dominates), the
      reported residual increases with the injected size and its least-squares
      slope is positive -- a flat, saturating or sign-reversing detector cannot be
      read at the size ``C6b`` injects;
    * **sees the control's corruption**: the injected size that crosses the bar
      on the *amplitude-normalised* statistic is at most the 0.30 step ``C6b``
      injects.  The amplitude normalisation is used here because the reported
      statistic divides by ``max|A|``, which carries the arbitrary ``A_inf`` and
      thereby dilutes the residual by ``max|A - A_inf| / max|A|``; the invariant
      statement is the one that transfers to another body.
    """
    data = certificate() if payload is None else payload
    primary = data["channels"]["damping_step"]
    floor = float(primary["floor"])
    slope = data["recovery_slope"]["damping_step"]
    thresholds = data["detection_threshold"]["damping_step"]
    reported_threshold = float(thresholds["reported"]["delta"])
    amplitude_threshold = float(thresholds["amplitude_normalised"]["delta"])
    floor_ok = floor <= 0.25 * float(data["criteria"]["kk_bar"])
    responsive_points = [
        p for p in primary["points"] if p["delta"] >= 0.05
    ]
    ordered = [p["reported_residual"] for p in responsive_points]
    responsive_ok = bool(
        len(ordered) >= 3
        and all(b > a for a, b in zip(ordered, ordered[1:]))
        and math.isfinite(slope["slope"])
        and slope["slope"] > 0.0
    )
    detection_ok = bool(
        math.isfinite(amplitude_threshold)
        and amplitude_threshold <= C6B_INJECTED_STEP
    )
    passed = bool(floor_ok and responsive_ok and detection_ok)
    return {
        "id": "C6b_cal",
        "title": (
            "negative control calibrated against an exact causal pair: the KK "
            "gate reports an injected damping-shape corruption, with a measured "
            "detection threshold"
        ),
        "threshold": (
            "floor <= 25% of the KK bar; damping-step residual strictly increasing "
            "for injected delta >= 0.05 with positive slope; amplitude-normalised "
            "detection threshold <= the 0.30 step C6b injects"
        ),
        "passed": passed,
        "values": {
            "floor_relative_residual": floor,
            "floor_over_bar": floor / float(data["criteria"]["kk_bar"]),
            "damping_step_slope": slope["slope"],
            "damping_step_monotone_over_0p05": responsive_ok,
            "detection_threshold_step_reported": (
                reported_threshold if math.isfinite(reported_threshold) else None
            ),
            "detection_threshold_step_amplitude_normalised": (
                amplitude_threshold if math.isfinite(amplitude_threshold) else None
            ),
            "injected_step_C6b": C6B_INJECTED_STEP,
            "predicted_residual_at_c6b_step": data["predicted_c6b_residual"],
            "predicted_residual_at_c6b_step_amplitude_normalised": data[
                "predicted_c6b_residual_amplitude_normalised"
            ],
            "detection_margin_reported": data["c6b_detection_margin"],
            "detection_margin_amplitude_normalised": data[
                "c6b_detection_margin_amplitude_normalised"
            ],
            "added_mass_bump_slope": data["recovery_slope"]["added_mass_bump"][
                "slope"
            ],
            "added_mass_bump_zero_mean_slope": data["recovery_slope"][
                "added_mass_bump_zero_mean"
            ]["slope"],
            "damping_bump_threshold_reported": (
                data["detection_threshold"]["damping_bump"]["reported"]["delta"]
                if math.isfinite(
                    float(
                        data["detection_threshold"]["damping_bump"]["reported"][
                            "delta"
                        ]
                    )
                )
                else None
            ),
            "localisation_worst_passing_corruption": data["localisation"][
                "worst_passing_corruption"
            ],
            "claims": {
                "floor_ok": bool(floor_ok),
                "responsive_ok": bool(responsive_ok),
                "detection_ok": bool(detection_ok),
            },
            "note": (
                "certifies the detector, not the sweep: the case's truncation "
                "budget on surge/sway/yaw stays unbounded until the 20 rad/s band "
                "artifact exists, which is what gates C8/C9 and C3a_extended "
                "address"
            ),
        },
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "inf"
    return f"{number:.{digits}g}"


def render_markdown(data: dict[str, Any], gate: dict[str, Any]) -> str:
    """Paper-grade summary of the calibration (tables generated, never typed)."""
    lines: list[str] = []
    pair = data["pair"]
    lines.append("# KK gate calibration (C6b): error-injection certificate\n")
    lines.append(
        "Exact causal pair -- two-exponential memory kernel -- on the case band "
        f"{_fmt(data['channels']['damping_step']['band'][0])}-"
        f"{_fmt(data['channels']['damping_step']['band'][1])} rad/s, "
        f"{data['channels']['damping_step']['n_omega']} nodes. "
        "The estimator is the case's own `kk_added_mass_closure`.\n"
    )
    lines.append("## 1 Pair (truth)\n")
    lines.append("| quantity | value |\n| --- | --- |")
    lines.append(f"| kernel | `{pair['kernel']}` |")
    lines.append(
        f"| coefficients | {', '.join(_fmt(c) for c in pair['coefficients'])} |"
    )
    lines.append(f"| rates | {', '.join(_fmt(a) for a in pair['rates'])} |")
    lines.append(f"| A_inf | {_fmt(pair['added_mass_inf'])} |")
    lines.append(f"| B hump at omega | {_fmt(pair['damping_hump_omega'])} rad/s |")
    lines.append(f"| max B on band | {_fmt(pair['damping_max_on_band'])} |")
    lines.append(
        "| max \\|A - A_inf\\| on band | "
        f"{_fmt(pair['added_mass_amplitude_on_band'])} |"
    )
    lines.append(f"| B > 0 on band | {_fmt(pair['damping_positive_on_band'])} |")
    lines.append("")
    tail = data["channels"]["damping_step"]
    lines.append("## 2 Estimator configuration (same as the case)\n")
    lines.append("| item | value |\n| --- | --- |")
    lines.append(f"| power-tail fit valid | {_fmt(tail['damping_tail_valid'])} |")
    lines.append(f"| fitted tail exponent | {_fmt(tail['damping_tail_exponent'])} |")
    lines.append(f"| tail rule | {tail['damping_tail_reason']} |")
    lines.append(f"| floor on the exact pair | {_fmt(tail['floor'], 3)} |")
    lines.append("")
    lines.append("## 3 Calibration curves (injected -> reported)\n")
    for name, curve in data["channels"].items():
        lines.append(f"### {name}\n")
        lines.append(
            "| injected delta | reported residual | above floor | over bar |\n"
            "| --- | --- | --- | --- |"
        )
        for point in curve["points"]:
            lines.append(
                f"| {_fmt(point['delta'], 3)} | "
                f"{_fmt(point['reported_residual'], 3)} | "
                f"{_fmt(point['above_floor'], 3)} | "
                f"{_fmt(point['detected'])} |"
            )
        slope = data["recovery_slope"][name]
        threshold = data["detection_threshold"][name]["reported"]
        threshold_amplitude = data["detection_threshold"][name]["amplitude_normalised"]
        lines.append(
            f"\nslope `residual - floor` vs delta = **{_fmt(slope['slope'], 3)}**, "
            f"monotone = {_fmt(slope['monotone_increasing'])}; bar crossed at "
            f"delta = **{_fmt(threshold['delta'], 3)}** of max|A|, "
            f"**{_fmt(threshold_amplitude['delta'], 3)}** of the added-mass "
            f"amplitude ({threshold['source']}).\n"
        )
    lines.append("## 3b Localisation blind spot (how narrow may a corruption be?)\n")
    local = data["localisation"]
    lines.append("| log-width | injected delta | reported residual | over bar |\n| --- | --- | --- | --- |")
    for row in local["rows"]:
        lines.append(
            f"| {_fmt(row['log_width'], 3)} | {_fmt(row['delta'], 3)} | "
            f"{_fmt(row['reported_residual'], 3)} | "
            f"{_fmt(not row['passes_bar'])} |"
        )
    worst = local["worst_passing_corruption"]
    lines.append("")
    if worst:
        lines.append(
            "Largest corruption that still passes the bar: a **"
            f"{_fmt(1.0 + worst['delta'], 3)}x** damping error confined to "
            f"d(log omega) = {_fmt(worst['log_width'], 3)} reports only "
            f"**{_fmt(worst['reported_residual'], 3)}** of max|A|.\n"
        )
    lines.append("## 4 Gate record\n")
    lines.append("| item | value |\n| --- | --- |")
    lines.append(f"| gate | {gate['id']} |")
    lines.append(f"| passed | {_fmt(gate['passed'])} |")
    for key, value in gate["values"].items():
        if isinstance(value, dict):
            continue
        lines.append(f"| {key} | {_fmt(value, 4)} |")
    for key, value in gate["values"]["claims"].items():
        lines.append(f"| claim {key} | {_fmt(value)} |")
    lines.append("")
    lines.append("## 5 What this does and does not certify\n")
    lines.append(
        "* **Does**: the KK gate reports an injected damping-shape corruption "
        "monotonically and with a measured slope, crosses the 10 % bar at a "
        "measurable fraction of the 0.30 step that `C6b` injects, and carries an "
        "estimator bias (floor) far below the bar on data whose answer is known."
    )
    lines.append(
        "* **Does not**: change the case's truncation budget. surge/sway/yaw keep "
        "`n_decidable = 0` at 12 rad/s; that is a property of the un-solved "
        "high-frequency band and needs the 20 rad/s artifact (C8/C9)."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="markdown file (default: <out>/kk_gate_calibration.md)",
    )
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    data = certificate()
    gate = calibration_gate(data)
    payload = {"certificate": data, "gate": gate}
    json_path = out / "kk_gate_calibration.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path = args.markdown or (out / "kk_gate_calibration.md")
    markdown_path.write_text(render_markdown(data, gate), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())