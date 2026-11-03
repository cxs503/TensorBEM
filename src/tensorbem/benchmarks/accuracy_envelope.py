"""Per-quantity accuracy envelope of the 3-D seakeeping solver.

Motivation
----------
The published 3-D benchmark tables (``docs/marine_bem_benchmarks.md`` §4.10-§4.11)
quote the error of the *added mass*, of the *exciting force* and of the observed
order of the *radiation damping*, but never the damping error itself.  For a
seakeeping study that is exactly the wrong quantity to leave unstated: the
damping is what sets the response amplitude at resonance, and it converges with
a much larger error constant than the added mass at the same mesh.

This module turns "which error at which mesh" into a measured, reproducible map.
For two bodies with fully independent closed-form references it sweeps

* the mesh (submerged sphere: 20/80/320 panels; floating hemisphere: 16/64/256),
* the dimensionless frequency ``ka``, and

and reports, per quantity,

* the near-field error (the quantity integrated from the solved potentials),
* the far-field (Kochin) damping error at the *same* mesh, and
* the panels-per-wavelength ratio, so the result is a mesh-selection rule and
not a remark about one particular mesh.

References
----------
* submerged sphere: Thorne/Wang/Linton multipole system,
  :func:`tensorbem.analytic.submerged_sphere_wave_multipole` (closed-form
  factorials + exponential integrals, no numerical quadrature, no shared code
  with the BEM solver);
* floating hemisphere: Hulme (1982) spherical wave-free multipole,
  :func:`tensorbem.analytic.hulme_hemisphere_radiation`.

The near-field and far-field damping are two *different* discrete estimators of
the same continuum quantity, so their disagreement at fixed mesh is a direct
measurement of the slow-converging part of the damping error.

Run::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.accuracy_envelope \
        --output artifacts/accuracy_envelope/envelope.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.analytic import (
    hulme_hemisphere_radiation,
    submerged_sphere_wave_multipole,
)
from tensorbem.free_surface_solver3d import (
    far_field_damping_matrix_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh, submerged_sphere_mesh

DENSITY = 1025.0
GRAVITY = 9.80665
RADIUS = 1.0
SUBMERGENCE = 1.5
#: Default ka grid.  It brackets the band a full-scale ship occupies for
#: head-seas response (k*L/2 in 0.5-3) and runs into short waves where panels
#: per wavelength become the binding constraint.
DEFAULT_KA = (0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0)
SPHERE_SUBDIVISIONS = (0, 1, 2)
HEMISPHERE_SUBDIVISIONS = (0, 1, 2)

#: Error targets quoted by the project's acceptance standard.
TARGETS = (0.03, 0.05, 0.10)


def _rel(value: float, reference: float) -> float:
    return float(abs(value - reference) / abs(reference))


def _panels_per_wavelength(ka: float, radius: float, h_max: float) -> float:
    """Panel-edge resolution measured in deep-water wavelengths.

    A panel edge ``h`` resolves ``lambda / h`` samples of a wave with
    ``lambda = 2 pi / k``; with ``k = ka / radius`` this is
    ``2 pi radius / (ka h)``.
    """
    if ka <= 0.0 or h_max <= 0.0:
        return float("inf")
    return float(2.0 * math.pi * radius / (ka * h_max))


def _order(coarse: float, fine: float, h_coarse: float, h_fine: float) -> float | None:
    if coarse <= 0.0 or fine <= 0.0 or h_coarse <= h_fine:
        return None
    return float(math.log(coarse / fine) / math.log(h_coarse / h_fine))


def run_sphere_envelope(
    subdivisions: Sequence[int] = SPHERE_SUBDIVISIONS,
    ka_values: Sequence[float] = DEFAULT_KA,
) -> list[dict[str, Any]]:
    """Sweep (mesh, ka) for the submerged sphere against the multipole reference."""
    rows: list[dict[str, Any]] = []
    for sub in subdivisions:
        mesh = submerged_sphere_mesh(
            subdivisions=int(sub), radius=RADIUS, submergence=SUBMERGENCE
        )
        h_max = float(mesh.h_max)
        for ka in ka_values:
            ka = float(ka)
            nu = ka / RADIUS
            start = time.time()
            radiation = solve_radiation_3d(
                mesh,
                nu,
                modes=(0, 2),
                center=(0.0, 0.0, -SUBMERGENCE),
                density=DENSITY,
                gravity=GRAVITY,
            )
            far = far_field_damping_matrix_3d(
                mesh, radiation, density=DENSITY, gravity=GRAVITY
            )
            reference = submerged_sphere_wave_multipole(
                ka, radius=RADIUS, submergence=SUBMERGENCE, density=DENSITY
            )
            rows.append(
                {
                    "body": "submerged_sphere",
                    "subdivision": int(sub),
                    "panels": int(mesh.panel_count),
                    "h_max": h_max,
                    "ka": ka,
                    "panels_per_wavelength": _panels_per_wavelength(ka, RADIUS, h_max),
                    "condition_number": float(radiation.condition_number),
                    "boundary_residual": float(radiation.relative_residual),
                    "a33": float(radiation.added_mass[1, 1]),
                    "a33_ref": float(reference.heave_added_mass),
                    "a33_err": _rel(float(radiation.added_mass[1, 1]), reference.heave_added_mass),
                    "a11": float(radiation.added_mass[0, 0]),
                    "a11_ref": float(reference.surge_added_mass),
                    "a11_err": _rel(float(radiation.added_mass[0, 0]), reference.surge_added_mass),
                    "b33_near": float(radiation.damping[1, 1]),
                    "b33_ref": float(reference.heave_damping),
                    "b33_near_err": _rel(float(radiation.damping[1, 1]), reference.heave_damping),
                    "b33_far": float(far[1, 1]),
                    "b33_far_err": _rel(float(far[1, 1]), reference.heave_damping),
                    "seconds": time.time() - start,
                }
            )
            print(
                f"sphere sub={sub} N={mesh.panel_count} ka={ka:.2f} "
                f"A33 {rows[-1]['a33_err']:.2e} B33near {rows[-1]['b33_near_err']:.2e} "
                f"B33far {rows[-1]['b33_far_err']:.2e} ({rows[-1]['seconds']:.1f}s)",
                flush=True,
            )
    return rows


def run_hemisphere_envelope(
    subdivisions: Sequence[int] = HEMISPHERE_SUBDIVISIONS,
    ka_values: Sequence[float] = DEFAULT_KA,
) -> list[dict[str, Any]]:
    """Sweep (mesh, ka) for the floating hemisphere against Hulme (1982)."""
    rows: list[dict[str, Any]] = []
    for sub in subdivisions:
        mesh = hemisphere_hull_mesh(subdivisions=int(sub), radius=RADIUS)
        h_max = float(mesh.h_max)
        for ka in ka_values:
            ka = float(ka)
            start = time.time()
            radiation = solve_radiation_3d(
                mesh,
                ka,  # radius = 1 so nu = ka
                modes=(0, 2),
                density=DENSITY,
                gravity=GRAVITY,
            )
            far = far_field_damping_matrix_3d(
                mesh, radiation, density=DENSITY, gravity=GRAVITY
            )
            reference = hulme_hemisphere_radiation(ka, radius=RADIUS, density=DENSITY)
            rows.append(
                {
                    "body": "floating_hemisphere",
                    "subdivision": int(sub),
                    "panels": int(mesh.panel_count),
                    "h_max": h_max,
                    "ka": ka,
                    "panels_per_wavelength": _panels_per_wavelength(ka, RADIUS, h_max),
                    "condition_number": float(radiation.condition_number),
                    "boundary_residual": float(radiation.relative_residual),
                    "a33": float(radiation.added_mass[1, 1]),
                    "a33_ref": float(reference.heave_added_mass),
                    "a33_err": _rel(float(radiation.added_mass[1, 1]), reference.heave_added_mass),
                    "a11": float(radiation.added_mass[0, 0]),
                    "a11_ref": float(reference.surge_added_mass),
                    "a11_err": _rel(float(radiation.added_mass[0, 0]), reference.surge_added_mass),
                    "b33_near": float(radiation.damping[1, 1]),
                    "b33_ref": float(reference.heave_damping),
                    "b33_near_err": _rel(float(radiation.damping[1, 1]), reference.heave_damping),
                    "b33_far": float(far[1, 1]),
                    "b33_far_err": _rel(float(far[1, 1]), reference.heave_damping),
                    "seconds": time.time() - start,
                }
            )
            print(
                f"hemisphere sub={sub} N={mesh.panel_count} ka={ka:.2f} "
                f"A33 {rows[-1]['a33_err']:.2e} B33near {rows[-1]['b33_near_err']:.2e} "
                f"B33far {rows[-1]['b33_far_err']:.2e} ({rows[-1]['seconds']:.1f}s)",
                flush=True,
            )
    return rows


def _mesh_ladder(rows: list[dict[str, Any]], ka: float) -> list[dict[str, Any]]:
    ladder = sorted(
        (row for row in rows if abs(row["ka"] - ka) < 1.0e-12), key=lambda r: r["panels"]
    )
    for coarse, fine in zip(ladder, ladder[1:]):
        for quantity in ("a33_err", "a11_err", "b33_near_err", "b33_far_err"):
            coarse[f"order_{quantity}"] = None
            fine[f"order_{quantity}"] = _order(
                coarse[quantity], fine[quantity], coarse["h_max"], fine["h_max"]
            )
    return ladder


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce the sweep to the questions a user actually asks.

    (1) For every quantity, which mesh is needed for the 3 % target at each ka?
    (2) How much better is the far-field damping estimator at fixed mesh?
    """
    by_body: dict[str, Any] = {}
    for body in sorted({row["body"] for row in rows}):
        body_rows = [row for row in rows if row["body"] == body]
        ka_values = sorted({row["ka"] for row in body_rows})
        fine = max(row["panels"] for row in body_rows)

        worst_fine = {
            quantity: max(
                row[quantity] for row in body_rows if row["panels"] == fine
            )
            for quantity in ("a33_err", "a11_err", "b33_near_err", "b33_far_err")
        }
        # largest ka still inside each target at the finest mesh
        band: dict[str, float | None] = {}
        for quantity in ("a33_err", "a11_err", "b33_near_err", "b33_far_err"):
            inside = [
                row["ka"]
                for row in sorted(body_rows, key=lambda r: r["ka"])
                if row["panels"] == fine and row[quantity] <= 0.03
            ]
            band[quantity] = max(inside) if inside else None

        far_vs_near = []
        for ka in ka_values:
            ladder = _mesh_ladder(body_rows, ka)
            far_vs_near.append(
                {
                    "ka": ka,
                    "panels": [row["panels"] for row in ladder],
                    "b33_near_err": [row["b33_near_err"] for row in ladder],
                    "b33_far_err": [row["b33_far_err"] for row in ladder],
                    "far_better": all(
                        row["b33_far_err"] < row["b33_near_err"] for row in ladder
                    ),
                }
            )

        by_body[body] = {
            "finest_panels": fine,
            "worst_error_at_finest_mesh": worst_fine,
            "largest_ka_with_error_le_3pct_at_finest_mesh": band,
            "far_field_vs_near_field_damping": far_vs_near,
            "ka_ladders": {f"{ka:.2f}": _mesh_ladder(body_rows, ka) for ka in ka_values},
        }
    return by_body


def _environment() -> dict[str, str]:
    import numpy

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "platform": platform.platform(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/accuracy_envelope/envelope.json"),
    )
    parser.add_argument(
        "--ka",
        type=float,
        nargs="+",
        default=list(DEFAULT_KA),
        help="dimensionless frequencies to sweep",
    )
    parser.add_argument(
        "--bodies",
        nargs="+",
        default=["sphere", "hemisphere"],
        choices=["sphere", "hemisphere"],
    )
    parser.add_argument(
        "--sphere-subdivisions", type=int, nargs="+", default=list(SPHERE_SUBDIVISIONS)
    )
    parser.add_argument(
        "--hemisphere-subdivisions",
        type=int,
        nargs="+",
        default=list(HEMISPHERE_SUBDIVISIONS),
    )
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    if "sphere" in args.bodies:
        rows += run_sphere_envelope(args.sphere_subdivisions, args.ka)
    if "hemisphere" in args.bodies:
        rows += run_hemisphere_envelope(args.hemisphere_subdivisions, args.ka)

    payload = {
        "environment": _environment(),
        "settings": {
            "density": DENSITY,
            "gravity": GRAVITY,
            "radius": RADIUS,
            "submergence": SUBMERGENCE,
            "ka": list(args.ka),
            "targets": list(TARGETS),
        },
        "rows": rows,
        "summary": summarise(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())