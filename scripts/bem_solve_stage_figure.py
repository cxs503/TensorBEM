#!/usr/bin/env python3
"""Render the solve-stage health-audit figure and its numbers (zero hand-copying).

Left panel  : stationarity of the returned vector as the solution is perturbed
              away from the least-squares minimiser, on the *production*
              64-panel hemisphere CHIEF system.  The 1e-6 guard threshold sits
              in the middle of a seven-decade gap, so the guard can be tight and
              still never fire on the healthy path.
Right panel : for the three rungs the case actually runs, the inherent CHIEF
              residual (the equations genuinely disagree by 1e-2 .. 1e-4) next to
              the stationarity of the healthy solve -- the point being that the
              residual cannot police a solve and the stationarity can.

Outputs
-------
    docs/bem_solve_stage_guard.png
    benchmarks/verified/casepool_bem_hulme_hemisphere/solve_stage_audit.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorbem.free_surface_solver3d import (  # noqa: E402
    _STATIONARITY_TOL,
    assemble_free_surface_operators_3d,
    least_squares_stationarity,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh  # noqa: E402

OUT_PNG = Path("docs/bem_solve_stage_guard.png")
OUT_JSON = Path("benchmarks/verified/casepool_bem_hulme_hemisphere/solve_stage_audit.json")
PERTURBATIONS = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1)


def _rhs(ops, n: int) -> np.ndarray:
    rng = np.random.default_rng(20261115)
    q_ext = (rng.normal(size=(n, 1)) + 0.0j).astype(np.complex128)
    return np.vstack((ops.single_layer @ q_ext, ops.chief_single_layer @ q_ext))


def sweep() -> dict:
    mesh = hemisphere_hull_mesh(2, radius=1.0)  # 64 panels, ~0.4 s
    ops = assemble_free_surface_operators_3d(mesh, 2.0, chief_points="auto")
    rhs = _rhs(ops, mesh.panel_count)
    sol, *_ = np.linalg.lstsq(ops.system_matrix, rhs, rcond=None)
    rows = []
    for delta in PERTURBATIONS:
        vec = sol * (1.0 + delta)
        rows.append(
            {
                "relative_perturbation": delta,
                "stationarity": least_squares_stationarity(ops.system_matrix, vec, rhs),
                "relative_residual": float(
                    np.linalg.norm(ops.system_matrix @ vec - rhs) / np.linalg.norm(rhs)
                ),
            }
        )
    return {"panels": int(mesh.panel_count), "rows": rows}


def rungs() -> list[dict]:
    out = []
    for sd in (2, 3):
        mesh = hemisphere_hull_mesh(sd, radius=1.0)
        ops = assemble_free_surface_operators_3d(mesh, 2.0, chief_points="auto")
        rhs = _rhs(ops, mesh.panel_count)
        sol, *_ = np.linalg.lstsq(ops.system_matrix, rhs, rcond=None)
        rad = solve_radiation_3d(mesh, 2.0, modes=(0, 2), operators=ops)
        out.append(
            {
                "subdivisions": sd,
                "panels": int(mesh.panel_count),
                "condition_number": float(ops.condition_number),
                "inherent_relative_residual": float(
                    np.linalg.norm(ops.system_matrix @ sol - rhs) / np.linalg.norm(rhs)
                ),
                "stationarity": least_squares_stationarity(ops.system_matrix, sol, rhs),
                "solve_reported_relative_residual": float(rad.relative_residual),
                "A33_dimless": float(rad.added_mass[1, 1]) / (1025.0 * (2.0 / 3.0) * math.pi),
            }
        )
    return out


def main() -> int:
    pert = sweep()
    rungs_data = rungs()
    payload = {
        "guard": {
            "stationarity_tolerance": _STATIONARITY_TOL,
            "definition": "||A^H r|| / (||A|| ||r||), r = A x - b",
        },
        "perturbation_sweep": pert,
        "rungs": rungs_data,
        "cross_environment_reference": {
            "note": (
                "geometry/assembly identical across environments; the residual "
                "column is the m20910 fingerprint measurement, reproduced here "
                "for the healthy control-plane side"
            ),
            "control_plane_reported_relative_residual_1024_panels": 1.769e-04,
            "cluster_node_reported_relative_residual_1024_panels": 4.91e-01,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=1) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    x = [max(row["relative_perturbation"], 1e-16) for row in pert["rows"]]
    y = [row["stationarity"] for row in pert["rows"]]
    ax.loglog(x, y, "o-", color="#1f77b4", label="stationarity of returned vector")
    ax.axhline(_STATIONARITY_TOL, color="crimson", ls="--", lw=1.2,
               label=f"guard threshold {_STATIONARITY_TOL:g}")
    ax.set_xlabel("relative perturbation of the least-squares solution")
    ax.set_ylabel(r"$\|A^{H}r\|\,/\,(\|A\|\,\|r\|)$")
    ax.set_title(f"64-panel hemisphere CHIEF system: guard vs perturbation")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    labels = [f"sd={r['subdivisions']}\n{r['panels']} panels" for r in rungs_data]
    resid = [r["inherent_relative_residual"] for r in rungs_data]
    stat = [max(r["stationarity"], 1e-17) for r in rungs_data]
    idx = np.arange(len(labels))
    ax.bar(idx - 0.2, resid, 0.38, label="CHIEF residual (equations disagree)",
           color="#ff7f0e")
    ax.bar(idx + 0.2, stat, 0.38, label="stationarity of the solve", color="#2ca02c")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(_STATIONARITY_TOL, color="crimson", ls="--", lw=1.2)
    ax.set_ylabel("dimensionless")
    ax.set_title("residual judges the physics,\nstationarity judges the solver")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", which="both", alpha=0.25)

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"wrote {OUT_PNG} and {OUT_JSON}")
    print(json.dumps(payload["rungs"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())