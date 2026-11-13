"""Conducting-sphere anchor for the linear cathodic-protection BEM.

Why this module exists
----------------------
:mod:`tensorbem.cathodic_protection` solves a *linear* mixed Dirichlet/Neumann
exterior Laplace problem on a ship hull: potentials are prescribed on every
coating holiday and on every anode, the coating is insulating, and the anode
current is the integrated normal current over the anode panels.  Such a model
has no polarization kinetics, so it cannot be validated against measured
protection potentials -- but it *can* be validated exactly, because a bare
metal sphere held at a uniform potential in an unbounded electrolyte is one of
the classical closed-form results of cathodic-protection theory:

    phi(r) = phi_a * a / r ,
    I      = -sigma * integral_S dphi/dn dS = 4 pi sigma a phi_a ,
    R      = 1 / (4 pi sigma a)        (spreading resistance of an isolated
                                        anode, DNV-RP-B401 design formula).

Prescribing ``phi = phi_a`` on *every* panel of an icosphere collapses the
mixed solver to the single-layer equation ``S q = (1/2 - H) phi_a``, so the
only remaining content is the physics and the numerics: the current integral
must reproduce ``4 pi sigma a phi_a`` and the exterior representation must
reproduce ``phi_a a / r``.

What is pinned here
-------------------
1. **Ohm's law scaling.**  The boundary integral equation returns the
   single-layer density ``q`` (V/m); the physical current density is
   ``j_n = sigma q``.  Without the conductivity factor the current integral is
   low by exactly ``1/sigma`` -- a 71.4 % error at ``sigma = 3.5 S/m``.  That
   defect is what this anchor exists to catch.
2. **Second-order convergence.**  The nested icosphere family has an exact
   refinement ratio ``r = (N_fine/N_coarse)**0.5 = 2``, so the observed order
   ``p = log(e_coarse/e_fine)/log(r)`` is measurable; the expected value is
   ``p = 2`` for the piecewise-constant single-layer representation used here.
3. **Exterior field.**  The representation formula
   (:func:`tensorbem.cathodic_protection.evaluate_exterior_potential`) must
   reproduce ``phi_a a / r`` outside the sphere.

Run:
    python -m tensorbem.benchmarks.cathodic_protection_sphere --subdivisions 1 2 3

The benchmark writes JSON to stdout; ``--output`` also persists it.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Sequence

from tensorbem.cathodic_protection import (
    SEAWATER_CONDUCTIVITY,
    verify_conducting_sphere,
    verify_conducting_sphere_routes,
)


def build_report(
    subdivisions: Sequence[int] = (1, 2, 3),
    radius: float = 1.0,
    potential: float = -1.0,
    conductivity: float = SEAWATER_CONDUCTIVITY,
    routes_subdivisions: int | None = None,
) -> dict[str, Any]:
    """Assemble the conducting-sphere benchmark report.

    ``routes_subdivisions`` additionally runs the two-route cross-check
    (:func:`tensorbem.cathodic_protection.verify_conducting_sphere_routes`) at
    that icosphere level: the surface panel integral and the far-field
    (Gauss-law) recovery of ``4 pi sigma a phi_a``.
    """
    report = verify_conducting_sphere(
        subdivisions=tuple(int(s) for s in subdivisions),
        radius=radius,
        potential=potential,
        conductivity=conductivity,
    )
    if routes_subdivisions is not None:
        report["routes"] = verify_conducting_sphere_routes(
            subdivisions=int(routes_subdivisions),
            radius=radius,
            potential=potential,
            conductivity=conductivity,
        )
    report["python"] = platform.python_version()
    report["platform"] = platform.platform()
    report["table"] = [
        {
            "subdivisions": level["subdivisions"],
            "panels": level["n_panels"],
            "current_A": level["current_A"],
            "current_rel_error_percent": 100.0 * level["current_rel_error"],
            "max_potential_rel_error_percent": 100.0 * level["max_potential_rel_error"],
        }
        for level in report["levels"]
    ]
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--potential", type=float, default=-1.0)
    parser.add_argument("--conductivity", type=float, default=SEAWATER_CONDUCTIVITY)
    parser.add_argument(
        "--routes-subdivisions",
        type=int,
        default=None,
        help="also run the surface-vs-far-field current cross-check at this level",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = build_report(
        subdivisions=args.subdivisions,
        radius=args.radius,
        potential=args.potential,
        conductivity=args.conductivity,
        routes_subdivisions=args.routes_subdivisions,
    )
    payload = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
        exact = report["exact_current_A"]
        finest = report["levels"][-1]
        print(f"wrote {args.output}")
        # A single refinement level has no observed-order pair; print a dash
        # instead of crashing on the None placeholder.
        order = report["observed_order_finest_pair"]
        order_text = f"{order:.2f}" if order is not None else "n/a (single level)"
        print(
            f"finest level: {finest['n_panels']} panels, "
            f"I = {finest['current_A']:.6f} A vs exact {exact:.6f} A "
            f"({100.0 * finest['current_rel_error']:.3f} %), "
            f"observed order {order_text}"
        )
        if "routes" in report:
            routes = report["routes"]
            print(
                "routes: surface {:.4f} % | far-field {:.4f} % | "
                "surface-vs-far-field {:.2e}".format(
                    100.0 * routes["surface_rel_error"],
                    100.0 * routes["far_field_rel_error"],
                    routes["far_field"]["far_field_vs_surface_rel_error"],
                )
            )
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())