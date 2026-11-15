#!/usr/bin/env python3
"""High-``ka`` mesh refinement of the TensorBEM arm for CASE-BEM-HULME-HEMISPHERE.

Why this file exists
--------------------
The anchor's evidence chain leaves three questions *open*, and all three are on
the TensorBEM side of the comparison, not the referee side:

1. ``radiation_capytaine_band_xval.json`` refused to attribute ``ka = 2.58, 3.0,
   4.0`` because the *referee* (capytaine) had not self-converged there
   (``G2`` FAIL, drift up to 37 %).  What is left unstated is the complementary
   fact: at ``ka >= 3`` the TensorBEM arm is also under-resolved *at the case
   mesh* (1024 constant panels), so nobody knows whether its 10-40 % band error
   is a **discretisation limit** or a **method defect**.
2. ``band_extension/report.md`` declares the extension band on a single mesh.
   A method that is only mesh-limited must show the error *decreasing* under
   refinement; a method that is defective does not.
3. The ``numpy`` 2.0.1 defect (m20910/m21563) was worked around upstream by
   upgrading the node environment.  The evidence that the *current* environment
   produces a mesh-converging sequence -- rather than one healthy-looking point
   -- has never been recorded.

This driver therefore re-solves the high-``ka`` radiation band on a panel
ladder ``1024 -> 4096`` (subdivisions 4 -> 5) in the current environment and
records, per cell, everything needed to tell the two hypotheses apart:

* ``added_mass``/``damping`` non-dimensionalised as ``A/(rho*nabla)`` and
  ``B/(rho*omega*nabla)`` with ``nabla = 2*pi*a^3/3``, so the numbers are
  comparable with ``band_extension/`` and with the Hulme (1982) series;
* the solver-stage guard output verbatim (``relative_residual``,
  ``condition_number``, ``solve_warnings``, ``chief_points_used``), because the
  m20910 defect produced full-rank, healthy-conditioned, wrong solutions;
* geometry pre-flight (wetted area, waterplane area, displacement) against the
  analytic hemisphere, so a mesh-generation regression cannot be mistaken for
  a hydrodynamics result;
* optional CHIEF arm (``chief_points="auto"`` / explicit interior panels) on the
  cell that m21334 flagged: CHIEF *on* was 18.5 % worse than *off* at
  ``ka = 2.58`` under the broken numpy.

Usage (one shard per invocation; ``--shard``/``--nshards`` stride the cell list)::

    PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_hemisphere/\\
        high_ka_mesh_refinement.py --shard 0 --nshards 8 --out-dir <dir>

The analysis/merge step is ``high_ka_mesh_refinement_analysis.py``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

RHO = 1025.0
GRAVITY = 9.80665
RADIUS = 1.0
NABLA = 2.0 / 3.0 * math.pi * RADIUS**3
WETTED_AREA = 2.0 * math.pi * RADIUS**2
WATERPLANE_AREA = math.pi * RADIUS**2

#: Case mesh (1024 panels) and the first refinement rung (4096 panels).  A third
#: rung (16384) is out of scope: the Kelvin assembly is O(N^2) and the 4096 rung
#: already takes ~17 min per cell on one core of the 702 cluster.
SUBDIVISIONS = (4, 5)

#: Radiation band this driver is responsible for: ``ka <= 2`` is the accepted
#: case band and is not re-run here, ``3.5`` and ``4.0`` are the cells where the
#: capytaine referee could not vote.
KA_GRID = (2.58, 3.0, 3.5, 4.0)

MAGNITUDES = ("A11", "B11", "A33", "B33")
MODES = (0, 2)  # surge, heave


def cells() -> list[dict[str, Any]]:
    """Cell list: the (ka, subdivisions) ladder plus the CHIEF control arm."""
    out: list[dict[str, Any]] = []
    for ka in KA_GRID:
        for sub in SUBDIVISIONS:
            out.append(
                {
                    "id": f"ka{ka:g}_sub{sub}",
                    "ka": float(ka),
                    "subdivisions": int(sub),
                    "chief": "off",
                }
            )
    # CHIEF arm on the case mesh only (the mesh where m21334 saw the anomaly).
    out.append({"id": "ka2.58_sub4_chief_auto", "ka": 2.58, "subdivisions": 4,
                "chief": "auto"})
    out.append({"id": "ka3_sub4_chief_auto", "ka": 3.0, "subdivisions": 4,
                "chief": "auto"})
    return out


def _mesh( subdivisions: int):
    from tensorbem.geometry3d import hemisphere_hull_mesh

    return hemisphere_hull_mesh(int(subdivisions), radius=RADIUS)


def geometry_preflight(mesh) -> dict[str, Any]:
    """Compare the panel mesh against the analytic hemisphere, before any solve."""
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tri = verts[faces]
    # Wetted area by Heron on each triangle.
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    wetted = float(area.sum())
    # Waterplane area: assembled from the waterline edges (z == 0) of the hull.
    # The hull is star-shaped about the origin, so the closed volume follows from
    # the divergence theorem on the wetted surface alone: V = int (x.n) dS / 3.
    centroids = tri.mean(axis=1)
    normals = cross / np.maximum(np.linalg.norm(cross, axis=1)[:, None], 1e-300)
    volume = float((area * np.einsum("ij,ij->i", centroids, normals)).sum() / 3.0)
    # Waterplane: the projected area of the waterline ring, |z| <= 1e-12.
    on_wl = np.isclose(verts[:, 2], 0.0, atol=1e-12)
    wp_radius = float(np.max(np.hypot(verts[on_wl, 0], verts[on_wl, 1]))) if on_wl.any() else float("nan")
    wp_area = math.pi * wp_radius**2
    out = {
        "panels": int(faces.shape[0]),
        "vertices": int(verts.shape[0]),
        "waterline_vertices": int(on_wl.sum()),
        "wetted_area_m2": wetted,
        "wetted_area_exact_m2": WETTED_AREA,
        "wetted_area_rel_error": (wetted - WETTED_AREA) / WETTED_AREA,
        "displacement_m3": volume,
        "displacement_exact_m3": math.pi * 2.0 / 3.0 * RADIUS**3,
        "displacement_rel_error": (volume - math.pi * 2.0 / 3.0 * RADIUS**3)
        / (math.pi * 2.0 / 3.0 * RADIUS**3),
        "waterplane_radius": wp_radius,
        "waterplane_area_m2": wp_area,
        "waterplane_area_exact_m2": WATERPLANE_AREA,
        "waterplane_area_rel_error": (wp_area - WATERPLANE_AREA) / WATERPLANE_AREA,
        "all_vertices_wet": bool(np.all(verts[:, 2] <= 1e-12)),
    }
    # An inscribed polyhedron *under*-estimates both area and volume by O(h^2),
    # so the gate is a facet-resolution gate, not an exactness gate: at 1024
    # panels the volume deficit is already 0.59 % (and it halves by 4x per rung
    # of refinement).  The waterplane radius is exact by construction.
    out["ok"] = bool(
        out["all_vertices_wet"]
        and abs(out["wetted_area_rel_error"]) < 1.0e-2
        and abs(out["displacement_rel_error"]) < 1.0e-2
        and abs(out["waterplane_area_rel_error"]) < 1.0e-2
        and abs(out["waterplane_radius"] - RADIUS) < 1.0e-12
    )
    return out


def solve_cell(cell: dict[str, Any]) -> dict[str, Any]:
    from tensorbem.free_surface_solver3d import solve_radiation_3d

    ka = float(cell["ka"])
    sub = int(cell["subdivisions"])
    mesh = _mesh(sub)
    geom = geometry_preflight(mesh)
    chief = None if cell.get("chief", "off") == "off" else cell["chief"]
    t0 = time.time()
    res = solve_radiation_3d(mesh, nu=ka / RADIUS, modes=list(MODES),
                             center=(0.0, 0.0, 0.0), density=RHO, gravity=GRAVITY,
                             chief_points=chief)
    wall = time.time() - t0
    omega = math.sqrt(GRAVITY * ka / RADIUS)
    # res.added_mass / res.damping are indexed by (mode_i, mode_j) over MODES.
    a = np.asarray(res.added_mass, dtype=float)
    b = np.asarray(res.damping, dtype=float)
    scale_a = RHO * NABLA
    scale_b = RHO * omega * NABLA
    out: dict[str, Any] = {
        "id": cell["id"],
        "ka": ka,
        "subdivisions": sub,
        "panels": geom["panels"],
        "chief": cell.get("chief", "off"),
        "chief_points_used": int(np.asarray(res.chief_points_used).shape[0])
        if res.chief_points_used is not None
        else 0,
        "A11": float(a[0, 0]) / scale_a,
        "B11": float(b[0, 0]) / scale_b,
        "A33": float(a[1, 1]) / scale_a,
        "B33": float(b[1, 1]) / scale_b,
        "added_mass_SI": a.tolist(),
        "damping_SI": b.tolist(),
        "condition_number": float(res.condition_number),
        "relative_residual": float(res.relative_residual),
        "solve_warnings": [str(w) for w in getattr(res, "warnings", ()) or ()],
        "wall_clock_s": wall,
        "geometry": geom,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "host": platform.node(),
        },
    }
    out["ok"] = bool(
        geom["ok"]
        and math.isfinite(out["relative_residual"])
        and out["relative_residual"] < 1.0e-2
        and math.isfinite(out["condition_number"])
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--only", type=str, default=None,
                    help="comma separated cell ids to run instead of striding")
    args = ap.parse_args(argv)

    all_cells = cells()
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        todo = [c for c in all_cells if c["id"] in want]
    else:
        todo = all_cells[args.shard :: max(1, args.nshards)]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for cell in todo:
        t0 = time.time()
        rec = solve_cell(cell)
        records.append(rec)
        print(json.dumps({k: rec[k] for k in
                          ("id", "panels", "A11", "B11", "A33", "B33",
                           "relative_residual", "wall_clock_s", "ok")}),
              flush=True)
        print(f"  solved {cell['id']} in {time.time() - t0:.1f}s", flush=True)

    payload = {
        "driver": "high_ka_mesh_refinement",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "shard": args.shard,
        "nshards": args.nshards,
        "cells": records,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
        },
    }
    path = args.out_dir / f"high_ka_mesh_refinement_shard{args.shard}.json"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)
    return 0 if all(r["ok"] for r in records) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())