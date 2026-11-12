#!/usr/bin/env python3
"""Freeze the capytaine side of the cross-solver regression.

Only this script imports capytaine; the numbers, the mesh arrays and the
provenance are written into
``benchmarks/verified/cross_solver_regression/capytaine_ref.json`` so that
``tensorbem.benchmarks.cross_solver_regression`` and its test run without
capytaine installed.

The meshes are built by the *same* builders the TensorBEM side uses
(``cross_solver_regression.CASES``), and the vertex/face arrays are stored in the
artifact: the comparison is therefore on identical panels, and the stored
``mesh_signature`` is re-checked on the TensorBEM side at compare time.

Run::

    python scripts/cross_solver_regression_ref.py --out benchmarks/verified/cross_solver_regression/capytaine_ref.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorbem.benchmarks.cross_solver_regression import (  # noqa: E402
    CASES,
    DENSITY,
    GRAVITY,
    NU_GRID,
    build_case,
    mesh_signature,
)

OUT_DEFAULT = Path("benchmarks/verified/cross_solver_regression/capytaine_ref.json")


def build() -> dict:
    import capytaine as cpt

    solver = cpt.BEMSolver()
    payload: dict = {
        "metadata": {
            "tool": "capytaine",
            "version": getattr(cpt, "__version__", "unknown"),
            "module_path": str(Path(cpt.__file__).resolve()),
            "solver": type(solver).__name__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "density": DENSITY,
            "gravity": GRAVITY,
            "water_depth": "infinite",
            "dof": "heave",
            "lid": "none (frequency band kept below the first irregular frequency)",
            "meshes": "identical vertex/face arrays as the TensorBEM builders; arrays stored below",
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        },
        "nu_grid": list(NU_GRID),
        "cases": [],
    }
    for case in CASES:
        case_out = {"name": case["name"], "title": case["title"], "levels": []}
        for level in case["levels"]:
            hull = build_case(case, level)
            vertices = np.asarray(hull.vertices, dtype=float)
            faces = np.asarray(hull.faces, dtype=np.int64)
            mesh = cpt.Mesh(vertices=vertices, faces=faces)
            body = cpt.FloatingBody(mesh=mesh)
            body.add_all_rigid_body_dofs()
            coeffs: dict[str, dict[str, float]] = {}
            for nu in NU_GRID:
                res = solver.solve(
                    cpt.RadiationProblem(
                        body=body,
                        omega=math.sqrt(nu * GRAVITY),
                        radiating_dof="Heave",
                        water_depth=np.inf,
                        rho=DENSITY,
                        g=GRAVITY,
                    )
                )
                coeffs[f"{nu:.3f}"] = {
                    "added_mass": float(res.added_mass["Heave"]),
                    "damping": float(res.radiation_damping["Heave"]),
                }
            case_out["levels"].append(
                {
                    "level": level,
                    "label": case["level_label"](level),
                    "panels": int(mesh.nb_faces),
                    "volume": float(mesh.volume),
                    "mesh_signature": mesh_signature(vertices, faces),
                    "vertices": [[round(float(x), 12) for x in row] for row in vertices],
                    "faces": [[int(i) for i in row] for row in faces],
                    "capytaine": coeffs,
                }
            )
            print(
                f"{case['name']} level {level}: {mesh.nb_faces} panels, volume {mesh.volume:.6f} m^3",
                flush=True,
            )
        payload["cases"].append(case_out)
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args(argv)
    payload = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
