#!/usr/bin/env python
"""Measure the capytaine reference's *own* error against an exact anchor.

Motivation
----------
`docs/xval_bem_finite_depth_crossval_plan.md` (mission m23018) bounds the
reference's remaining discretisation error by its last-rung change and finds
that bound (1.6--1.8 %) already eats the 1/3-of-tolerance budget.  A bound is not
a measurement: this script turns it into one, on every finite-depth grid cell
where an *exact* answer exists, without running any solver.

Why an exact answer exists on part of the grid
----------------------------------------------
For a submerged sphere the Thorne/Wang/Linton multipole solution is closed form
(:func:`tensorbem.analytic.submerged_sphere_wave_multipole`: zero quadrature, and
its far-field/Haskind identity closes to ~1e-16).  It is a **deep-water**
solution, so it is the exact finite-depth answer only where the seabed is
irrelevant.  The relevant small parameter is *not* ``exp(-2 k0 h)`` but the
clearance-based exponent

    ``exp(-2 k0 (h - f - a))``

with ``f`` the sphere centre depth and ``a`` its radius: the rigid-bottom image
sits one full clearance below the body, not one depth below the free surface.
At ``h/a = 3, k0 a = 2`` the clearance is 0.5 m and the exponent is only 2 --
a 13 % image contribution, so that cell is **excluded** here even though
``k0 h = 6`` looks deep.  Cells are admitted only when the exponent is at least
``CLEARANCE_EXPONENT`` (default 4 -> 3.4e-4 of image field).

Output
------
``benchmarks/verified/capytaine_finite_depth/reference_truth_audit.json``:

* ``cells``: per admitted cell, per quantity -- the exact value, every rung's
  relative error, the error of the rung the study actually reports, the
  last-rung-change proxy, and their ratio (how *optimistic* the proxy is);
* ``excluded``: every grid cell left out, with its exponent and the reason;
* ``summary``: worst measured reference error, worst proxy/measured ratio.

The consumer is :mod:`tensorbem.benchmarks.capytaine_finite_depth_crossval`,
which replaces its model bound by these measured values where they exist.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorbem.analytic import submerged_sphere_wave_multipole  # noqa: E402
from tensorbem.benchmarks import finite_depth_crossval_plan as plan  # noqa: E402

#: Minimum ``2 k0 (h - f - a)`` for a cell to be treated as effectively deep.
CLEARANCE_EXPONENT = 4.0

#: Multipole truncation used for the anchor, and the order used to prove it has
#: converged (the truncated series is checked, not trusted).
N_TERMS = 32
N_TERMS_CHECK = 24
N_TERMS_AGREEMENT = 1e-10

DOFS = ("heave", "surge")
QUANTITIES = ("added_mass", "damping")


def _truth(ka: float, h_over_a: float) -> dict[str, dict[str, float]]:
    """Exact deep-water multipole values, with a truncation-convergence check."""
    fine = submerged_sphere_wave_multipole(
        ka,
        radius=plan.SPHERE_RADIUS,
        submergence=plan.SPHERE_SUBMERGENCE,
        density=plan.DENSITY,
        gravity=plan.GRAVITY,
        n_terms=N_TERMS,
    )
    coarse = submerged_sphere_wave_multipole(
        ka,
        radius=plan.SPHERE_RADIUS,
        submergence=plan.SPHERE_SUBMERGENCE,
        density=plan.DENSITY,
        gravity=plan.GRAVITY,
        n_terms=N_TERMS_CHECK,
    )
    out: dict[str, dict[str, float]] = {}
    spread = 0.0
    for dof in DOFS:
        for quantity in QUANTITIES:
            value = float(getattr(fine, f"{dof}_{quantity}"))
            other = float(getattr(coarse, f"{dof}_{quantity}"))
            spread = max(spread, abs(value - other) / abs(value) if value else 0.0)
            out[f"{dof}.{quantity}"] = value
    if spread > N_TERMS_AGREEMENT:
        raise RuntimeError(
            f"multipole truncation not converged at ka={ka}: n_terms "
            f"{N_TERMS_CHECK} vs {N_TERMS} differ by {spread:.2e}"
        )
    haskind = max(abs(fine.haskind_error_x), abs(fine.haskind_error_z))
    if haskind > 1e-10:
        raise RuntimeError(f"multipole far-field identity violated: {haskind:.2e}")
    out["_haskind_error"] = haskind
    out["_truncation_spread"] = spread
    return out


def audit(reference: dict) -> dict:
    """Per-cell measured reference error, plus the cells that had to be dropped."""
    rows: list[dict] = []
    excluded: list[dict] = []
    for cell in reference["cells"]:
        h = float(cell["depth"])
        k0 = float(cell["k0"])
        clearance = h - (plan.SPHERE_SUBMERGENCE + plan.SPHERE_RADIUS)
        exponent = 2.0 * k0 * clearance
        record = {
            "cell": cell["cell"],
            "h_over_a": float(cell["h_over_a"]),
            "ka": float(cell["ka"]),
            "k0h": float(cell["k0h"]),
            "clearance": float(clearance),
            "clearance_exponent": float(exponent),
        }
        if exponent < CLEARANCE_EXPONENT:
            record["reason"] = (
                "clearance-based image exponent "
                f"2 k0 (h-f-a) = {exponent:.2f} < {CLEARANCE_EXPONENT:g}: the "
                "deep-water anchor is not the exact finite-depth answer here"
            )
            excluded.append(record)
            continue
        truth = _truth(record["ka"], record["h_over_a"])
        levels = sorted(cell["levels"], key=lambda rec: rec["panels"])
        record["rungs"] = [float(rec["panels"]) for rec in levels]
        record["haskind_error"] = truth["_haskind_error"]
        record["truncation_spread"] = truth["_truncation_spread"]
        record["quantities"] = {}
        for slot in (f"{dof}.{quantity}" for dof in DOFS for quantity in QUANTITIES):
            dof, quantity = slot.split(".")
            exact = truth[slot]
            values = [float(rec["values"][dof][quantity]) for rec in levels]
            errors = [(value - exact) / exact for value in values]
            finest_change = (
                abs(values[-1] - values[-2]) / abs(values[-1])
                if len(values) >= 2 and values[-1]
                else None
            )
            record["quantities"][slot] = {
                "exact": exact,
                "rungs": [float(rec["panels"]) for rec in levels],
                "values": values,
                "rung_errors": [float(e) for e in errors],
                "finest_error": float(errors[-1]),
                "finest_error_abs": abs(float(errors[-1])),
                "last_rung_change": finest_change,
                "proxy_optimism": (
                    None
                    if not finest_change
                    else float(abs(float(errors[-1])) / float(finest_change))
                ),
            }
        rows.append(record)
    measured = [
        q["finest_error_abs"] for row in rows for q in row["quantities"].values()
    ]
    optimism = [
        q["proxy_optimism"]
        for row in rows
        for q in row["quantities"].values()
        if q["proxy_optimism"] is not None
    ]
    worst_cell = max(
        (
            (row["cell"], slot, q["finest_error_abs"])
            for row in rows
            for slot, q in row["quantities"].items()
        ),
        key=lambda item: item[2],
        default=None,
    )
    return {
        "criterion": {
            "anchor": "tensorbem.analytic.submerged_sphere_wave_multipole "
                      f"(n_terms={N_TERMS}, checked at {N_TERMS_CHECK})",
            "admission": "2 k0 (h - f - a) >= CLEARANCE_EXPONENT",
            "clearance_exponent": CLEARANCE_EXPONENT,
            "geometry": {
                "radius": plan.SPHERE_RADIUS,
                "submergence": plan.SPHERE_SUBMERGENCE,
            },
            "note": "the deep-water anchor is exact for the finite-depth problem "
                    "only where the seabed image is negligible; the exponent is "
                    "clearance-based, not k0 h",
        },
        "cells": rows,
        "excluded": excluded,
        "summary": {
            "admitted_cells": len(rows),
            "excluded_cells": len(excluded),
            "measured_quantities": len(measured),
            "max_measured_reference_error": max(measured) if measured else None,
            "max_measured_at": worst_cell,
            "max_proxy_optimism": max(optimism) if optimism else None,
            "median_proxy_optimism": (
                sorted(optimism)[len(optimism) // 2] if optimism else None
            ),
            "proxy_underestimates_truth": (
                None
                if not optimism
                else sum(1 for value in optimism if value > 1.0) / len(optimism)
            ),
        },
    }


def measured_by_cell(report: dict) -> dict[str, dict[str, float]]:
    """``cell -> slot -> measured absolute relative error`` for the consumer."""
    return {
        row["cell"]: {
            slot: float(q["finest_error_abs"])
            for slot, q in row["quantities"].items()
        }
        for row in report["cells"]
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "verified"
        / "capytaine_finite_depth",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    reference = json.loads((args.root / "capytaine_ref.json").read_text())
    report = audit(reference)
    out = args.out or (args.root / "reference_truth_audit.json")
    out.write_text(json.dumps(report, indent=1) + "\n")
    summary = report["summary"]
    print(f"admitted cells: {summary['admitted_cells']} "
          f"(excluded {summary['excluded_cells']})")
    print(f"max measured reference error: "
          f"{100.0 * summary['max_measured_reference_error']:.2f} % "
          f"at {summary['max_measured_at']}")
    print(f"proxy optimism: max {summary['max_proxy_optimism']:.2f}x, "
          f"median {summary['median_proxy_optimism']:.2f}x, "
          f"underestimates in {100.0 * summary['proxy_underestimates_truth']:.0f} % "
          "of the measured quantities")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())