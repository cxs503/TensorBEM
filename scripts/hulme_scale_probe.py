#!/usr/bin/env python3
"""Probe why the in-house hemisphere cell degrades at ``subdivisions = 5``.

Context (dev report m20234 §5, adjudicated here): the anchor
``BEM-HULME-HEMISPHERE`` grades the panel solver at ``subdivisions = 4``
(1024 panels) and the refinement ladder's next rung (``subdivisions = 5``,
4096 panels) returned ``A33 = 0.2079`` against a reference of ``0.3884``
(-46 %) while the *reported* operator condition number stayed at 2.4 --
i.e. the solve degrades silently, with no ill-conditioning to blame.

This probe separates the candidate mechanisms without repeating a full case
run.  For one ``(ka, subdivisions)`` cell it records

* the closed-form reference and the solved dimensionless coefficients;
* the assembled operator conditioning (:func:`numpy.linalg.cond`) *and* the
  least-squares **consistency** residual of the CHIEF-augmented system, which
  ``numpy.linalg.cond`` cannot see (the rectangular matrix has 5 more rows than
  columns, so a small ``cond`` says nothing about whether those rows agree);
* the per-CHIEF-row residual, i.e. which interior point (if any) is the one
  whose null-field equation the body solution cannot satisfy;
* the singular-value spectrum of the augmented matrix (the CHIEF rows only
  *add* singular values, so a healthy ``cond`` is expected even when the rows
  are nearly dependent);
* the same cell solved with ``chief=False`` (square system) as the control.

Usage
-----
    PYTHONPATH=src python scripts/hulme_scale_probe.py --ka 2.0 --subdivisions 3 4
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import numpy as np

from tensorbem.benchmarks.hulme_hemisphere_case import (
    DENSITY,
    GRAVITY,
    RADIUS,
    reference,
)
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    rigid_body_normals_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh

QUANTITIES = ("A11_dimless", "B11_dimless", "A33_dimless", "B33_dimless")


def _dimensionless(rad: Any, ka: float) -> dict[str, float]:
    vol = (2.0 / 3.0) * math.pi * RADIUS**3
    norm_a = DENSITY * vol
    norm_b = DENSITY * math.sqrt(GRAVITY * (ka / RADIUS)) * vol
    return {
        "A11_dimless": float(rad.added_mass[0, 0]) / norm_a,
        "A33_dimless": float(rad.added_mass[1, 1]) / norm_a,
        "B11_dimless": float(rad.damping[0, 0]) / norm_b,
        "B33_dimless": float(rad.damping[1, 1]) / norm_b,
    }


def probe(ka: float, subdivisions: int) -> dict[str, Any]:
    t0 = time.time()
    mesh = hemisphere_hull_mesh(subdivisions, radius=RADIUS)
    nu = ka / RADIUS
    out: dict[str, Any] = {
        "ka": ka,
        "subdivisions": subdivisions,
        "panels": mesh.panel_count,
        "mesh_volume": float(mesh.volume),
        "mesh_waterplane_area": float(mesh.waterplane_area),
    }

    # --- CHIEF-augmented assembly (the graded configuration) ----------------
    ops = assemble_free_surface_operators_3d(mesh, nu, chief_points="auto")
    out["condition_number"] = float(ops.condition_number)
    out["chief_points"] = np.asarray(ops.chief_points, dtype=float).tolist()
    out["chief_point_pair_distances"] = _pair_distances(ops.chief_points)

    sv = np.linalg.svd(ops.system_matrix, compute_uv=False)
    out["sv_max"] = float(sv[0])
    out["sv_min"] = float(sv[-1])
    out["sv_ratio"] = float(sv[0] / sv[-1])
    out["sv_tail"] = [float(x) for x in sv[-6:]]
    out["augmented_rows"] = int(ops.system_matrix.shape[0])
    out["augmented_cols"] = int(ops.system_matrix.shape[1])

    # Heave (mode 2, unit velocity, no external forcing): the RHS the solver
    # actually builds is S_body @ q_ext, so reproduce it exactly.
    q_ext = (-rigid_body_normals_3d(mesh)[:, [2]]).astype(np.complex128)
    rhs_body = ops.single_layer @ q_ext
    rhs = np.vstack((rhs_body, ops.chief_single_layer @ q_ext))
    sol, *_ = np.linalg.lstsq(ops.system_matrix, rhs, rcond=None)
    resid = ops.system_matrix @ sol - rhs
    denom = max(float(np.linalg.norm(rhs)), 1.0e-30)
    out["lstsq_rel_residual"] = float(np.linalg.norm(resid) / denom)
    chief_resid = np.linalg.norm(resid[mesh.panel_count:], axis=0)
    chief_rhs = np.linalg.norm(rhs[mesh.panel_count:], axis=0)
    out["chief_row_rel_residual"] = [
        float(a / max(b, 1.0e-30)) for a, b in zip(chief_resid, chief_rhs)
    ]
    body_resid = np.linalg.norm(resid[: mesh.panel_count], axis=0)
    body_rhs = np.linalg.norm(rhs[: mesh.panel_count], axis=0)
    out["body_row_rel_residual"] = float(
        np.linalg.norm(body_resid) / max(float(np.linalg.norm(body_rhs)), 1.0e-30)
    )

    # --- graded solves -----------------------------------------------------
    rad_chief = solve_radiation_3d(
        mesh, nu, modes=(0, 2), density=DENSITY, gravity=GRAVITY, operators=ops
    )
    out["with_chief"] = _dimensionless(rad_chief, ka)
    out["with_chief_reported_rel_residual"] = float(rad_chief.relative_residual)

    ops_nc = assemble_free_surface_operators_3d(mesh, nu, chief_points=None)
    out["no_chief_condition_number"] = float(ops_nc.condition_number)
    rad_nc = solve_radiation_3d(
        mesh, nu, modes=(0, 2), density=DENSITY, gravity=GRAVITY, operators=ops_nc
    )
    out["no_chief"] = _dimensionless(rad_nc, ka)
    out["no_chief_reported_rel_residual"] = float(rad_nc.relative_residual)

    ref = reference(ka)
    out["reference"] = {k: ref[k] for k in QUANTITIES}
    for tag in ("with_chief", "no_chief"):
        out[f"rel_error_{tag}"] = {
            k: (out[tag][k] - ref[k]) / abs(ref[k]) for k in QUANTITIES
        }
    out["wall_seconds"] = time.time() - t0
    return out


def _pair_distances(points: np.ndarray) -> list[float]:
    pts = np.asarray(points, dtype=float)
    dists: list[float] = []
    for i in range(pts.shape[0]):
        for j in range(i + 1, pts.shape[0]):
            dists.append(float(np.linalg.norm(pts[i] - pts[j])))
    return dists


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ka", type=float, default=2.0)
    ap.add_argument("--subdivisions", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    rows = [probe(args.ka, sd) for sd in args.subdivisions]
    payload = {"ka": args.ka, "rows": rows}
    text = json.dumps(payload, indent=1)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())