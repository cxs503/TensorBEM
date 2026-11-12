"""Freeze the independent-code reference for the Wigley III recalled-value audit.

Run ONCE on a machine that has capytaine installed:

    python scripts/wigley3_recalled_ref_capytaine.py --out benchmarks/verified/wigley3_recalled_ref/capytaine_ref.json

capytaine is deliberately never imported inside ``src/`` or ``tests/`` (same
policy as ``scripts/cross_solver_regression_ref.py``): the study re-reads the
frozen numbers and the mesh signatures, so a silent drift in either mesh
builder turns the audit red.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorbem import wigley3_freq as wf  # noqa: E402
from tensorbem.wigley3_hull import (  # noqa: E402
    wigley3_displaced_volume,
    wigley3_wetted_hull_mesh,
)

DENSITY = 1025.0
GRAVITY = 9.80665
OMEGA = 1.0
DOFS = ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")
CENTER = (0.0, 0.0, -0.09375)

BUILDERS = {
    "wigley3_hull": [
        ("nx=10,nz=5", lambda: wigley3_wetted_hull_mesh(nx=10, nz=5)),
        ("nx=20,nz=10", lambda: wigley3_wetted_hull_mesh(nx=20, nz=10)),
        ("nx=40,nz=20", lambda: wigley3_wetted_hull_mesh(nx=40, nz=20)),
    ],
    "wigley3_freq": [
        ("n_sections=10,n_levels=5", lambda: wf.wigley3_wetted_hull_mesh(n_sections=10, n_levels=5)),
        ("n_sections=20,n_levels=10", lambda: wf.wigley3_wetted_hull_mesh(n_sections=20, n_levels=10)),
        ("n_sections=40,n_levels=20", lambda: wf.wigley3_wetted_hull_mesh(n_sections=40, n_levels=20)),
    ],
}


def signature(vertices: np.ndarray, faces: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(vertices, dtype="<f8").tobytes())
    h.update(np.ascontiguousarray(faces, dtype="<i8").tobytes())
    return h.hexdigest()


def build() -> dict:
    import capytaine as cpt

    volume = wigley3_displaced_volume()
    solver = cpt.BEMSolver()
    payload: dict = {
        "metadata": {
            "tool": "capytaine",
            "version": getattr(cpt, "__version__", "unknown"),
            "module_path": str(Path(cpt.__file__).resolve()),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "density": DENSITY,
            "gravity": GRAVITY,
            "water_depth": "infinite",
            "omega": OMEGA,
            "rotation_center": list(CENTER),
            "displaced_volume": volume,
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        },
        "builders": {},
    }
    for builder, levels in BUILDERS.items():
        rows = []
        for label, make in levels:
            hull = make()
            verts = np.asarray(hull.vertices, dtype=float)
            faces = np.asarray(hull.faces, dtype=np.int64)
            mesh = cpt.Mesh(vertices=verts, faces=faces)
            body = cpt.FloatingBody(mesh=mesh)
            body.add_all_rigid_body_dofs(rotation_center=np.array(CENTER))
            added, damping = {}, {}
            for dof in DOFS:
                res = solver.solve(
                    cpt.RadiationProblem(
                        body=body,
                        omega=OMEGA,
                        radiating_dof=dof,
                        water_depth=np.inf,
                        rho=DENSITY,
                        g=GRAVITY,
                    )
                )
                added[dof] = float(res.added_mass[dof])
                damping[dof] = float(res.radiation_damping[dof])
            rows.append(
                {
                    "level": label,
                    "panels": int(mesh.nb_faces),
                    "mesh_volume": float(mesh.volume),
                    "mesh_volume_rel_error": float(mesh.volume / volume - 1.0),
                    "mesh_signature": signature(verts, faces),
                    "added_mass": added,
                    "damping": damping,
                }
            )
            print(f"{builder} {label}: {mesh.nb_faces} panels, A_heave/rhoV = "
                  f"{added['Heave'] / (DENSITY * volume):.4f}", flush=True)
        payload["builders"][builder] = rows
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/verified/wigley3_recalled_ref/capytaine_ref.json"),
    )
    args = ap.parse_args(argv)
    payload = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())