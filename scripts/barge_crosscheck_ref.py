#!/usr/bin/env python3
"""Produce the independent capytaine reference for the wall-sided barge case.

Why this file exists
--------------------
``benchmarks/verified/capytaine_crosscheck/capytaine_ref.json`` stores the
wall-sided barge as an *excluded negative control*, on the ground that
"neither capytaine nor the low-order TensorBEM assembly converges (spurious
irregular frequencies at the sharp bilge corners)".  That claim does not
survive inspection:

* the barge mesh in the old producer is
  ``mesh_parallelepiped(size=(2, 1, 1), center=(0, 0, -0.5)).immersed_part()``,
  i.e. a box spanning ``z in [-1, 0]``.  ``immersed_part()`` only clips faces
  *above* the waterline, so the **top face at z = 0 survives**: the "barge" is
  a sealed box and its waterplane is a Neumann wall lying exactly on the free
  surface.  The reported negative heave damping (B33 < 0) is the signature of
  that degenerate panel, not of an irregular frequency.
* the same box has ``volume = 2.0 m^3`` for a case labelled ``T = 0.5``
  (draft would be 1.0 m), so the label and the geometry disagree as well.

This producer builds the intended geometry -- the wetted surface of a
``L x B x T = 2 x 1 x 0.5`` barge, *no* waterplane face -- from
``tensorbem.geometry3d.box_hull_mesh``, converts it to a capytaine ``Mesh``
(same vertices, same triangles, so the two codes are compared on identical
panels) and solves heave radiation with and without an interior lid.  It also
*reproduces* the sealed-box construction so the root cause is recorded as
measured evidence instead of prose.

Only this script imports capytaine; the numbers are frozen into
``benchmarks/verified/barge_crosscheck/capytaine_ref.json`` together with the
provenance.  Run::

    python scripts/barge_crosscheck_ref.py --out benchmarks/verified/barge_crosscheck/capytaine_ref.json
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

from tensorbem.geometry3d import box_hull_mesh  # noqa: E402

DENSITY = 1025.0
GRAVITY = 9.80665
NU_GRID = (0.5, 1.0, 2.0)
LENGTH, BEAM, DRAFT = 2.0, 1.0, 0.5
LEVELS = ((4, 2, 1), (8, 4, 2), (16, 8, 4))


def _solve(cpt, solver, mesh, lid, dofs=("Heave",)):
    body = cpt.FloatingBody(mesh=mesh, lid_mesh=lid)
    body.add_all_rigid_body_dofs()
    out = {}
    for nu in NU_GRID:
        per_dof = {}
        for dof in dofs:
            res = solver.solve(
                cpt.RadiationProblem(
                    body=body,
                    omega=math.sqrt(nu * GRAVITY),
                    radiating_dof=dof,
                    water_depth=np.inf,
                    rho=DENSITY,
                    g=GRAVITY,
                )
            )
            per_dof[dof.lower()] = {
                "added_mass": float(res.added_mass[dof]),
                "damping": float(res.radiation_damping[dof]),
            }
        out[f"{nu:.3f}"] = per_dof
    return out


def build(out_path: Path) -> dict:
    import capytaine as cpt

    solver = cpt.BEMSolver()
    payload = {
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
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "meshes": "identical triangulated panels as tensorbem.geometry3d.box_hull_mesh",
        },
        "nu_grid": list(NU_GRID),
        "geometry": {"length": LENGTH, "beam": BEAM, "draft": DRAFT},
        "levels": [],
    }
    for res in LEVELS:
        hull = box_hull_mesh(LENGTH, BEAM, DRAFT, n_length=res[0], n_beam=res[1], n_draft=res[2])
        mesh = cpt.Mesh(vertices=hull.vertices, faces=hull.faces)
        lid = cpt.mesh_rectangle(
            size=(LENGTH, BEAM), center=(0.0, 0.0, 0.0), resolution=(res[0], res[1])
        )
        payload["levels"].append(
            {
                "resolution": list(res),
                "panels": int(mesh.nb_faces),
                "volume": float(mesh.volume),
                "bare": _solve(cpt, solver, mesh, None),
                "lid": _solve(cpt, solver, mesh, lid),
            }
        )

    # Root-cause reproduction of the old negative control: a box that spans
    # exactly z in [-draft, 0] keeps its waterplane face through immersed_part().
    sealed = cpt.mesh_parallelepiped(
        size=(2.0, 1.0, 1.0), center=(0.0, 0.0, -0.5), resolution=(4, 4, 2)
    ).immersed_part()
    faces_on_waterline = int(np.count_nonzero(np.isclose(sealed.faces_centers[:, 2], 0.0)))
    payload["sealed_box_reproduction"] = {
        "construction": "mesh_parallelepiped(size=(2,1,1), center=(0,0,-0.5), resolution=(4,4,2)).immersed_part()",
        "panels": int(sealed.nb_faces),
        "volume": float(sealed.volume),
        "faces_lying_on_waterline": faces_on_waterline,
        "values": _solve(cpt, solver, sealed, None),
        "note": (
            "the waterplane face survives clipping and lies exactly on the free "
            "surface; the resulting negative heave damping is the artefact the "
            "old reference file recorded as an irregular-frequency phenomenon"
        ),
    }

    a, b = 1.0 / BEAM**2, 1.0 / LENGTH**2
    k = math.pi * math.sqrt(a + b)
    omega_irr = math.sqrt(math.pi * GRAVITY * math.sqrt(a + b) / math.tanh(k * DRAFT))
    payload["irregular_frequency_estimate"] = {
        "formula": "capytaine FloatingBody.first_irregular_frequency_estimate (parallelepiped bound)",
        "omega_irr": omega_irr,
        "nu_irr": omega_irr**2 / GRAVITY,
        "note": "first irregular frequency lies above the tested band, so the case is usable without a lid",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/verified/barge_crosscheck/capytaine_ref.json"),
    )
    args = ap.parse_args(argv)
    payload = build(args.out)
    print(f"wrote {args.out}")
    print("nu_irr =", round(payload["irregular_frequency_estimate"]["nu_irr"], 4))
    for row in payload["levels"]:
        print(row["resolution"], row["panels"], {k: (round(v["heave"]["added_mass"], 2), round(v["heave"]["damping"], 2)) for k, v in row["bare"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
