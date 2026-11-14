"""How much of the Cummins-Wigley KK residual can the *lower* band tail explain?

Rounds 2-5 of ``CASE-BEM-CUMMINS-WIGLEY`` kept recording 20-80 % Kramers-Kronig
residuals and kept attributing them to band truncation, most often to the
un-measured part of the band below the lowest solved frequency.  The closure in
:func:`tensorbem.causality.kk_added_mass_closure` does close that sub-band, by
holding ``B`` at its first tabulated value over ``(0, s_min)``.  For a floating
body whose damping vanishes at zero frequency that constant hold is the crudest
possible model, so "the lower tail is the culprit" is a plausible story.

This script decides the *size* of that effect on a case where the answer is
known exactly, instead of on the panel data where the answer is not.

Construction
------------
A memory kernel that is a sum of exponentials is closed under the Cummins and
Kramers-Kronig transforms (module docstring of :mod:`tensorbem.causality`)::

    K(t) = sum_j c_j exp(-a_j t)
    B(w) = sum_j c_j a_j / (a_j^2 + w^2)
    A(w) - A_inf = -sum_j c_j / (a_j^2 + w^2)

so ``(A, B)`` are known in closed form and the causal residual of *any* closure
of the truncated band is a statement about the closure, not about BEM.  The
coefficient vector is constrained by least squares to satisfy ``sum_j c_j/a_j =
0``, which makes ``B ~ -w^2 sum_j c_j/a_j^3`` -- the ``w^2`` low-frequency onset
a 3-D radiating body has -- and the band is the predeclared extended band
:data:`tensorbem.benchmarks.cummins_wigley_kk_causality.EXTENDED_GRID` that the
anchor actually solves.  The sample points are the closure's own default
samples (interior nodes only, as the principal-value quadrature requires).

What is reported
----------------
* ``closure_constant_tail``: the shipped estimator's relative residual and its
  error against the exact ``A`` (the truth the panel data does not provide);
* ``lower_tail_bound``: the same with the sub-band below ``s_min`` integrated
  exactly rather than held constant -- i.e. an upper bound on what any better
  lower tail could buy;
* ``scale``: ``max |B|`` and ``max |A|`` on the band, so the residual can be
  read as an absolute number as well as a relative one.

Usage::

    PYTHONPATH=src python scripts/cummins_wigley_kk_lowtail_bound.py \\
        --out docs/case_bem_cummins_wigley_lowtail_bound.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

from tensorbem.benchmarks import cummins_wigley_kk_causality as case
from tensorbem.causality import kk_added_mass_closure, principal_value_integral

#: Kernel poles (rad/s) and the low-frequency constraint on the coefficients.
POLES = (0.8, 2.2, 5.5, 14.0)
SECOND_MOMENT_TARGET = -0.5


def exact_pair(poles=POLES, second_moment=SECOND_MOMENT_TARGET):
    """Coefficients with ``sum c/a = 0`` (so ``B ~ w^2``) and the closed-form pair."""
    a = np.asarray(poles, dtype=float)
    # unknowns c_1..c_{n-1}, with c_0 = 1 fixed
    rows = np.vstack([1.0 / a, 1.0 / a**3])[:, 1:]
    rhs = np.array(
        [-1.0 / a[0], second_moment - 1.0 / a[0] ** 3]
    )
    sol, *_ = np.linalg.lstsq(rows, rhs, rcond=None)
    c = np.concatenate([[1.0], sol])

    def damping(omega):
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        return np.sum(c[:, None] * a[:, None] / (a[:, None] ** 2 + omega[None, :] ** 2), axis=0)

    def added_mass(omega):
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        return -np.sum(c[:, None] / (a[:, None] ** 2 + omega[None, :] ** 2), axis=0)

    return c, damping, added_mass


def lower_tail_integral(omega, coefficient, exponent, end):
    """``PV integral_0^end C s^q/(s^2-omega^2) ds`` for ``end < omega``, closed form."""
    sample, upper, power = float(omega), float(end), float(exponent)
    if not upper < sample:
        raise ValueError("the sub-band must lie below the sample frequency")
    total = 0.0
    for k in range(0, 400):
        term = upper ** (power + 2.0 * k + 1.0) / ((power + 2.0 * k + 1.0) * sample ** (2 * k + 2))
        total += term
        if term <= 1e-18 * abs(total):
            break
    return -float(coefficient) * total


def fit_low_exponent(grid, data, take=3):
    """Least-squares exponent of ``B ~ C s^q`` on the lowest ``take`` nodes."""
    n = min(int(take), np.size(grid))
    slope, intercept = np.polyfit(np.log(grid[:n]), np.log(np.abs(data[:n])), 1)
    return float(slope), float(math.exp(intercept))


def study():
    c, damping, added_mass = exact_pair()
    grid = np.asarray(case.EXTENDED_GRID, dtype=float)
    damper = damping(grid)
    mass = added_mass(grid)
    closure = kk_added_mass_closure(grid, mass, damper, tail_fit=False)
    samples = np.asarray(closure.sample_omegas, dtype=float)
    truth = added_mass(samples)
    scale_a = float(np.max(np.abs(mass)))

    # lower-tail bound: exact analytic sub-band integral with a fitted exponent
    exponent, coefficient = fit_low_exponent(grid, damper)
    raw = np.empty(samples.size)
    held = np.empty(samples.size)
    for i, sample in enumerate(samples):
        value = principal_value_integral(grid, damper, float(sample), lower_tail=False)
        held[i] = value + 0.5 * float(damper[0]) / sample * math.log(
            abs(float(grid[0]) - sample) / (float(grid[0]) + sample)
        )
        raw[i] = value + lower_tail_integral(sample, coefficient, exponent, float(grid[0]))
    offset_exact = float(np.mean(truth - 2.0 / math.pi * raw))
    predicted_exact = 2.0 / math.pi * raw + offset_exact

    def stats(pred):
        resid = np.abs(truth - pred) / scale_a
        return {
            "max_relative_residual": float(np.max(resid)),
            "rms_relative_residual": float(np.sqrt(np.mean(resid**2))),
        }

    return {
        "case": "CASE-BEM-CUMMINS-WIGLEY / KK lower-band-tail attribution",
        "kernel": {"poles_rad_s": list(POLES), "coefficients": c.tolist(),
                   "sum_c_over_a": float(np.sum(c / np.asarray(POLES)))},
        "band_rad_s": [float(grid[0]), float(grid[-1])],
        "band_nodes": int(grid.size),
        "sample_points": samples.tolist(),
        "scales": {
            "max_abs_B": float(np.max(np.abs(damper))),
            "max_abs_A": scale_a,
            "B_at_lowest_node": float(damper[0]),
        },
        "low_frequency_fit": {"exponent_q": exponent, "coefficient": coefficient},
        "closure_constant_tail": {
            **stats(closure.predicted_added_mass),
            "relative_residual_max": float(closure.max_relative_residual),
        },
        "lower_tail_bound": {**stats(predicted_exact),
                             "added_mass_inf": offset_exact},
        "exact_truth_max_abs_A": scale_a,
        "verdict": {
            "lower_tail_is_the_culprit": bool(
                stats(predicted_exact)["max_relative_residual"]
                < 0.5 * stats(closure.predicted_added_mass)["max_relative_residual"]
            ),
            "constant_tail_under_3_percent": bool(
                stats(closure.predicted_added_mass)["max_relative_residual"] < 0.03
            ),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/case_bem_cummins_wigley_lowtail_bound.json")
    args = parser.parse_args(argv)
    payload = study()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps(
        {
            "constant_tail": payload["closure_constant_tail"],
            "lower_tail_bound": payload["lower_tail_bound"],
            "exponent_q": payload["low_frequency_fit"]["exponent_q"],
            "verdict": payload["verdict"],
        }, indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())