"""Reference-integrity checks for the Havelock submerged-sphere radiation case.

The numerical benchmark (:mod:`tensorbem.benchmarks.submerged_sphere_havelock`)
compares the 3-D free-surface BEM against
:func:`tensorbem.analytic.submerged_sphere_wave_multipole`.  That comparison is
only as trustworthy as the reference itself, so this module audits the
reference **before** it is used as a target.  Four checks, all of them
closed-form and independent of the panel solver:

1. **Truncation independence** -- the multipole system is truncated at degree
   ``n_terms``; raising it from 20 to 40 must not move ``A33' / B33' / A11' /
   B11'`` by more than round-off.  (``zero transcription``: the target values
   are *evaluated*, never copied out of a table.)
2. **Haskind identity** -- for the axisymmetric (heave) and antisymmetric
   (surge) multipole systems the Newman/Haskind relation
   ``|F_z|^2 = 2 rho g omega B33 / nu^2`` and ``|F_x|^2 = 4 rho g omega B11 / nu^2``
   holds exactly for the linearised free-surface problem, and the radiation and
   diffraction systems are solved independently here, so the residual measures
   the algebraic consistency of the two branches.
3. **Deep-submergence limit (handbook anchor)** -- as ``f/a -> inf`` the free
   surface decouples and the added mass must tend to Lamb's unbounded-fluid
   value ``A' = 0.5`` (i.e. ``C_M = 1 + A' = 1.5``), for *both* the heave and the
   surge mode of a sphere.  The amplitude of the residual decays like
   ``(a/f)**3`` (image dipole), which is checked as an observed order.
   ``0.5`` / ``1.5`` are closed-form classical constants, not transcribed
   digits: Lamb (1932) *Hydrodynamics*, 6th ed., art. 92 (sphere in an
   infinite fluid) and the Morison inertia coefficient ``C_M = 1 + C_a``.
4. **Low-frequency damping limit** -- ``B33', B11' -> 0`` as ``ka -> 0`` (no
   radiated energy in the zero-frequency limit).

Every quantity produced here is a *computed* number, so the reference for the
case carries no hand-copied digits at all.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from tensorbem.analytic import submerged_sphere_wave_multipole

#: Lamb (1932) *Hydrodynamics* 6th ed. art. 92: added mass of a sphere in an
#: unbounded inviscid fluid equals half the displaced mass, i.e. ``A' = 0.5``
#: in the ``A / (rho * volume)`` normalisation used throughout this repo.
UNBOUNDED_FLUID_ADDED_MASS_DIMLESS = 0.5
#: Morison inertia coefficient ``C_M = 1 + C_a`` for a sphere in unbounded fluid.
UNBOUNDED_FLUID_INERTIA_COEFFICIENT = 1.5

_MODES = (
    "heave_added_mass_dimless",
    "heave_damping_dimless",
    "surge_added_mass_dimless",
    "surge_damping_dimless",
)


def _rel(a: float, b: float) -> float:
    scale = max(abs(b), 1.0e-30)
    return abs(a - b) / scale


def run_truncation_study(
    ka_values: Sequence[float] = (0.1, 0.5, 1.0, 2.0, 5.0),
    radius: float = 1.0,
    submergence: float = 2.0,
    density: float = 1025.0,
    gravity: float = 9.80665,
    n_terms_low: int = 20,
    n_terms_high: int = 40,
) -> dict[str, Any]:
    """Multipole truncation independence and Haskind residuals."""
    rows = []
    worst_nterm = 0.0
    worst_haskind = 0.0
    for ka in ka_values:
        low = submerged_sphere_wave_multipole(
            float(ka), radius=radius, submergence=submergence,
            density=density, gravity=gravity, n_terms=n_terms_low,
        )
        high = submerged_sphere_wave_multipole(
            float(ka), radius=radius, submergence=submergence,
            density=density, gravity=gravity, n_terms=n_terms_high,
        )
        deltas = {name: _rel(getattr(low, name), getattr(high, name)) for name in _MODES}
        worst_nterm = max(worst_nterm, max(deltas.values()))
        haskind_z = float(low.haskind_error_z)
        haskind_x = float(low.haskind_error_x)
        worst_haskind = max(worst_haskind, haskind_z, haskind_x)
        rows.append(
            {
                "ka": float(ka),
                "n_terms_low": int(n_terms_low),
                "n_terms_high": int(n_terms_high),
                "a33_n20": low.heave_added_mass_dimless,
                "a33_n40": high.heave_added_mass_dimless,
                "b33_n20": low.heave_damping_dimless,
                "b33_n40": high.heave_damping_dimless,
                "a11_n20": low.surge_added_mass_dimless,
                "a11_n40": high.surge_added_mass_dimless,
                "b11_n20": low.surge_damping_dimless,
                "b11_n40": high.surge_damping_dimless,
                "max_truncation_rel_delta": max(deltas.values()),
                "truncation_rel_delta": deltas,
                "haskind_error_z": haskind_z,
                "haskind_error_x": haskind_x,
            }
        )
    return {
        "rows": rows,
        "worst_truncation_rel_delta": worst_nterm,
        "worst_haskind_error": worst_haskind,
        "n_terms_low": int(n_terms_low),
        "n_terms_high": int(n_terms_high),
    }


def run_deep_submergence_limit(
    submergence_ratios: Sequence[float] = (2.0, 3.0, 5.0, 10.0, 20.0),
    ka: float = 1.0e-3,
    radius: float = 1.0,
    n_terms: int = 30,
) -> dict[str, Any]:
    """``f/a -> inf``: ``A33'`` and ``A11'`` must approach Lamb's ``0.5``."""
    rows = []
    for ratio in submergence_ratios:
        res = submerged_sphere_wave_multipole(
            ka, radius=radius, submergence=float(ratio) * radius, n_terms=n_terms
        )
        rows.append(
            {
                "submergence_ratio": float(ratio),
                "a33": res.heave_added_mass_dimless,
                "a33_excess_over_lamb": res.heave_added_mass_dimless
                - UNBOUNDED_FLUID_ADDED_MASS_DIMLESS,
                "a11": res.surge_added_mass_dimless,
                "a11_excess_over_lamb": res.surge_added_mass_dimless
                - UNBOUNDED_FLUID_ADDED_MASS_DIMLESS,
                "inertia_coefficient_cm": 1.0 + res.heave_added_mass_dimless,
                "b33": res.heave_damping_dimless,
            }
        )
    orders = []
    for prev, cur in zip(rows[:-1], rows[1:]):
        e0 = abs(prev["a33_excess_over_lamb"])
        e1 = abs(cur["a33_excess_over_lamb"])
        r0 = prev["submergence_ratio"]
        r1 = cur["submergence_ratio"]
        if e0 > 0.0 and e1 > 0.0 and r1 > r0:
            orders.append(math.log(e0 / e1) / math.log(r1 / r0))
    return {
        "ka": float(ka),
        "rows": rows,
        "lamb_added_mass_dimless": UNBOUNDED_FLUID_ADDED_MASS_DIMLESS,
        "morison_inertia_coefficient": UNBOUNDED_FLUID_INERTIA_COEFFICIENT,
        "observed_excess_decay_orders": orders,
        "observed_excess_decay_order_last": orders[-1] if orders else None,
    }


def run_low_frequency_damping(
    ka_values: Sequence[float] = (1.0e-3, 1.0e-2, 5.0e-2),
    radius: float = 1.0,
    submergence: float = 2.0,
) -> list[dict[str, float]]:
    """``B33', B11' -> 0`` as ``ka -> 0`` (vanishing radiated energy)."""
    rows = []
    for ka in ka_values:
        res = submerged_sphere_wave_multipole(
            float(ka), radius=radius, submergence=submergence, n_terms=30
        )
        rows.append(
            {
                "ka": float(ka),
                "b33": res.heave_damping_dimless,
                "b11": res.surge_damping_dimless,
                "a33": res.heave_added_mass_dimless,
                "a11": res.surge_added_mass_dimless,
            }
        )
    return rows


def run(
    *,
    ka_values: Sequence[float] = (0.1, 0.5, 1.0, 2.0, 5.0),
    radius: float = 1.0,
    submergence: float = 2.0,
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> dict[str, Any]:
    """All reference-integrity checks (analytic only, no panel solve)."""
    truncation = run_truncation_study(
        ka_values=ka_values, radius=radius, submergence=submergence,
        density=density, gravity=gravity,
    )
    deep = run_deep_submergence_limit(radius=radius)
    low = run_low_frequency_damping(radius=radius, submergence=submergence)
    return {
        "benchmark": "submerged_sphere_havelock_reference_integrity",
        "geometry": {
            "radius": radius,
            "submergence": submergence,
            "submergence_ratio": submergence / radius,
            "density": density,
            "gravity": gravity,
        },
        "truncation": truncation,
        "deep_submergence_limit": deep,
        "low_frequency_damping": low,
    }


def main() -> None:  # pragma: no cover - thin CLI
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submergence", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(submergence=args.submergence)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text)


if __name__ == "__main__":  # pragma: no cover
    main()