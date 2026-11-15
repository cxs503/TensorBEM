"""Where does the ``B -> K -> (B, A)`` round trip (gate G5) lose its accuracy?

Question
--------
``CASE-BEM-CUMMINS-WIGLEY`` fails G5 (round trip ``B -> K -> (B, A)``, tol 5 %)
by 20-48 % on every DOF, and fails G3 for a related reason.  Candidate sources:

``inverse_legs``
    the Simpson quadratures of eqs. (2)-(3) in :mod:`tensorbem.time_domain`
    (``damping_from_memory_kernel`` / ``added_mass_from_memory_kernel``);
``forward_leg``
    :func:`tensorbem.time_domain.radiation_memory_kernel`, which builds
    ``K(tau) = (2/pi) int_0^omega_max B cos(omega tau) d(omega)`` and *drops* the
    band above ``omega_max``.  Dropping the band at a frequency where
    ``B(omega_max) != 0`` leaves a jump discontinuity in the tabulated
    ``B``; the resulting kernel has an algebraic ``sin(omega_max tau)/tau``
    tail, so the sine transform used for ``A`` (which is divided by
    ``omega``) never converges.  This is a *kernel-model* defect, not a
    property of the panel solution;
``band``
    the sweep's own ``[omega_min, omega_max]`` window and the query band inside
    it.

The script separates them with an arm whose answer is known analytically -- a
memory kernel that is a single exponential, ``K(tau) = c exp(-a tau)``, whose
pairs are exact::

    B(omega) = c a / (a^2 + omega^2),    A(omega) - A_inf = -c / (a^2 + omega^2).

If the inverse legs alone are fed the *exact* kernel they must return ``A`` and
``B`` to quadrature accuracy; any residual on that arm belongs to the forward
leg, because the forward leg is the only thing the two arms do not share.

Arms
----
``exact_kernel``    inverse legs on ``K = c exp(-a tau)`` sampled on the case's
                    own tau grid -> measures the inverse legs alone.
``table_kernel``    full round trip with the shipped (truncated) forward leg.
``tail_closed``     full round trip where the forward leg is closed above
                    ``omega_max`` with the algebraic tail
                    ``B = B(omega_max) (omega_max/omega)^2`` integrated in
                    closed form -> measures what the truncation costs.

Usage::

    PYTHONPATH=src python3 scripts/cummins_kernel_high_frequency_audit.py \\
        --sweep-npz docs/m18249_case_bem_cummins_wigley/sweep_coarse.npz \\
        --json docs/case_bem_cummins_wigley_kernel_hf_audit.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np
from scipy.integrate import quad
from scipy.special import sici

from tensorbem import time_domain as td
from tensorbem.benchmarks import cummins_wigley_kk_causality as case

DOF_LABELS = ("surge", "sway", "heave", "roll", "pitch", "yaw")


# ---------------------------------------------------------------------------
# high-frequency closure of the forward leg
# ---------------------------------------------------------------------------
def tail_cosine_power(amplitude: float, omega_high: float, exponent: float,
                      tau: np.ndarray) -> np.ndarray:
    """``int_omega_high^inf A (omega_high/omega)^p cos(omega tau) d omega``.

    ``exponent = 2`` has the closed form used below (``cos x / x - (pi/2 - Si x)``
    with a series for small argument); any other exponent is integrated
    numerically as a Fourier integral (``scipy.integrate.quad(weight="cos")``),
    which is the method that handles an oscillatory integrand on an infinite
    range without truncation.
    """
    tau = np.asarray(tau, dtype=float)
    if exponent == 2.0:
        x = omega_high * tau
        small = x < 1.0e-2
        xs = np.where(small, 1.0, x)
        f = np.cos(xs) / xs - (math.pi / 2.0 - sici(xs)[0])
        f = np.where(small, 1.0 / xs - math.pi / 2.0 + xs / 2.0, f)
        scale = omega_high ** 2 * np.where(small, tau, tau) * f
        scale = np.where(tau > 0.0, scale, omega_high)
        return amplitude * scale
    scale = amplitude * omega_high ** exponent
    out = np.empty_like(tau)
    for i, value in enumerate(tau):
        if value <= 0.0:
            out[i] = omega_high / (exponent - 1.0) if exponent > 1.0 else math.nan
        else:
            out[i] = quad(lambda s: s ** (-exponent), omega_high, np.inf,
                          weight="cos", wvar=value, limit=400)[0]
    return scale * out


def kernel_tail_closed(omega, damping, tau, *, exponent: float = 2.0,
                       low_frequency_model: str = "cubic"):
    """Shipped forward leg plus the analytic closure of the band above ``omega_max``."""
    base = td.radiation_memory_kernel(omega, damping, tau,
                                     low_frequency_model=low_frequency_model)
    grid = np.asarray(tau, dtype=float)
    omega_high = float(np.asarray(omega, dtype=float)[-1])
    amplitude = float(np.asarray(damping, dtype=float)[-1])
    values = np.asarray(base.values, dtype=float).copy()
    if amplitude > 0.0 and omega_high > 0.0:
        values += 2.0 / math.pi * tail_cosine_power(
            amplitude, omega_high, exponent, grid)
    return td.MemoryKernel(
        tau=grid, values=values, omega_max=omega_high,
        damping_scale=base.damping_scale, low_frequency_model=low_frequency_model,
        high_frequency_bound=0.0, low_frequency_bound=0.0, metadata={},
    )


# ---------------------------------------------------------------------------
# round-trip error, exactly as gate G5 computes it
# ---------------------------------------------------------------------------
def round_trip_errors(kernel, omega, added, damping, added_mass_inf,
                      band=(0.314, 7.2), samples: int = 60) -> dict:
    query = np.linspace(band[0], band[1], samples)
    round_b = np.asarray(td.damping_from_memory_kernel(kernel, query), dtype=float)
    round_a = np.asarray(td.added_mass_from_memory_kernel(kernel, query, added_mass_inf),
                         dtype=float)
    exact_b = np.interp(query, omega, damping)
    exact_a = np.interp(query, omega, added)
    damping_err = np.abs(round_b - exact_b) / max(float(np.max(np.abs(exact_b))), 1e-30)
    added_err = np.abs(round_a - exact_a) / max(
        float(np.max(np.abs(exact_a - added_mass_inf))), 1e-30)
    return {
        "band": [float(band[0]), float(band[1])],
        "damping_rel_err": float(np.max(damping_err)),
        "added_mass_rel_err": float(np.max(added_err)),
        "worst": float(max(np.max(damping_err), np.max(added_err))),
        "query": query.tolist(),
        "damping_profile": damping_err.tolist(),
        "added_mass_profile": added_err.tolist(),
    }


def synthetic_arm(omega: np.ndarray, tau: np.ndarray, c: float = 300.0,
                  a: float = 8.0, added_mass_inf: float = 50.0) -> dict:
    """One-exponential kernel: exact ``(A, B)`` known, so both arms are judgeable."""
    damping = c * a / (a * a + omega * omega)
    added = added_mass_inf - c / (a * a + omega * omega)
    exact_kernel = td.MemoryKernel(
        tau=tau, values=c * np.exp(-a * tau), omega_max=float(omega[-1]),
        damping_scale=0.0, low_frequency_model="zero",
        high_frequency_bound=0.0, low_frequency_bound=0.0, metadata={},
    )
    band = (float(max(omega[0], 0.3)), float(0.6 * omega[-1]))
    out = {
        "params": {"c": c, "a": a, "added_mass_inf": added_mass_inf},
        "exact_kernel": round_trip_errors(exact_kernel, omega, added, damping, added_mass_inf, band),
        "table_kernel": round_trip_errors(
            td.radiation_memory_kernel(omega, damping, tau), omega, added, damping,
            added_mass_inf, band),
        "tail_closed": round_trip_errors(
            kernel_tail_closed(omega, damping, tau), omega, added, damping,
            added_mass_inf, band),
    }
    return out


def panel_arm(npz: pathlib.Path, tau: np.ndarray) -> dict:
    data = np.load(npz)
    omega = np.asarray(data["omega"], dtype=float)
    band = (float(max(omega[0], 4.0 * math.pi / tau[-1])), float(0.6 * omega[-1]))
    rows = {}
    for dof, label in enumerate(DOF_LABELS):
        added = np.asarray(data["added_mass"], dtype=float)[:, dof, dof]
        damping = np.asarray(data["damping"], dtype=float)[:, dof, dof]
        a_inf = float(td.fit_added_mass_asymptote(omega, added).added_mass_inf)
        shipped = td.radiation_memory_kernel(omega, damping, tau)
        closed = kernel_tail_closed(omega, damping, tau)
        rows[label] = {
            "added_mass_inf": a_inf,
            "damping_at_omega_max": float(damping[-1]),
            "tail_kernel_at_zero": float(tail_cosine_power(
                float(damping[-1]), float(omega[-1]), 2.0, np.zeros(1))[0]),
            "shipped": round_trip_errors(shipped, omega, added, damping, a_inf, band),
            "tail_closed": round_trip_errors(closed, omega, added, damping, a_inf, band),
            "shipped_kernel_end": float(np.asarray(shipped.values)[-1]),
            "closed_kernel_end": float(np.asarray(closed.values)[-1]),
        }
    return {"npz": str(npz), "omega_min": float(omega[0]), "omega_max": float(omega[-1]),
            "n_omega": int(omega.size), "band": list(band), "dof": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-npz", type=pathlib.Path,
                        default=pathlib.Path("docs/m18249_case_bem_cummins_wigley/sweep_coarse.npz"))
    parser.add_argument("--json", type=pathlib.Path,
                        default=pathlib.Path("docs/case_bem_cummins_wigley_kernel_hf_audit.json"))
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--tau-max", type=float, default=40.0)
    args = parser.parse_args()

    tau = np.arange(int(round(args.tau_max / args.dt)) + 1, dtype=float) * args.dt
    data = np.load(args.sweep_npz)
    omega = np.asarray(data["omega"], dtype=float)

    payload = {
        "dt": args.dt,
        "tau_max": float(tau[-1]),
        "synthetic": synthetic_arm(omega, tau),
        "panel": panel_arm(args.sweep_npz, tau),
    }
    args.json.write_text(json.dumps(payload, indent=1))

    syn = payload["synthetic"]
    print("synthetic one-exponential pair (B, A known exactly)")
    for arm in ("exact_kernel", "table_kernel", "tail_closed"):
        print(f"  {arm:14s} damping {syn[arm]['damping_rel_err']:.4%}  "
              f"added {syn[arm]['added_mass_rel_err']:.4%}  worst {syn[arm]['worst']:.4%}")
    print("panel sweep (G5 statistic: max over the query band, G5 tol 5 %)")
    for label, row in payload["panel"]["dof"].items():
        print(f"  {label:6s} shipped {row['shipped']['worst']:.4%} -> "
              f"tail-closed {row['tail_closed']['worst']:.4%}")
    print(f"-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())