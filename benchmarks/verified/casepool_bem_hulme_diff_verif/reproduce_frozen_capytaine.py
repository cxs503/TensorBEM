"""Reproduce the *frozen* capytaine artefact of the Hulme diffraction case.

The delivered case reads ``capytaine_scattered.json`` -- a frozen foreign-code
result.  A frozen number is evidence only if it can be reproduced, so this
script re-runs the same capytaine problems live (at the base-case band and at
the mesh resolution the frozen file used as its finest) and writes the
cell-by-cell comparison to ``frozen_reproduction.json``.

It is deliberately small: no TensorBEM solver is involved, so it checks the
foreign route's provenance and conventions (scattered-only force, open wetted
surface, ``exp(-i omega t)``) independently of everything else in the case.

Usage::

    PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_diff_verif/reproduce_frozen_capytaine.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "capytaine_scattered.json"
OUT = HERE / "frozen_reproduction.json"

DENSITY = 1025.0
GRAVITY = 9.80665
RADIUS = 1.0
#: the frozen artefact stores forces normalised by rho g a^2 (A = 1 m); the
#: live capytaine result is dimensional, so un-normalise before comparing.
SCALE = DENSITY * GRAVITY * RADIUS**2


def _rel(got: complex, frozen: complex) -> float:
    return abs(got - frozen) / abs(frozen)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="relative tolerance for 'reproduced' (capytaine is deterministic; "
        "only a different build/machine should move it)",
    )
    args = parser.parse_args(argv)

    import capytaine as cpt

    cpt.set_logging("ERROR")
    solver = cpt.BEMSolver()
    frozen = json.loads(FROZEN.read_text())
    # the frozen artefact's own finest resolution (last ladder rung).
    finest = frozen["mesh_ladder"][-1]
    resolution = tuple(finest["resolution"])
    mesh = cpt.mesh_sphere(
        radius=1.0, center=(0.0, 0.0, 0.0), resolution=resolution
    ).immersed_part()
    body = cpt.FloatingBody(mesh=mesh)
    body.add_all_rigid_body_dofs()

    rows = {}
    for key, values in finest["values"].items():
        ka = float(key)
        res = solver.solve(
            cpt.DiffractionProblem(
                body=body,
                omega=math.sqrt(ka * GRAVITY),
                wave_direction=0.0,
                water_depth=np.inf,
                rho=DENSITY,
                g=GRAVITY,
            )
        )
        row = {}
        for dof in ("surge", "heave"):
            got = complex(np.asarray(res.forces[dof.capitalize()]))
            ref = SCALE * complex(values[dof]["re"], values[dof]["im"])
            row[dof] = {
                "live": {"re": got.real, "im": got.imag},
                "frozen": values[dof],
                "frozen_dimensional": {"re": ref.real, "im": ref.imag},
                "relative": _rel(got, ref),
            }
        rows[key] = row

    worst = max(
        (row[dof]["relative"], key, dof)
        for key, row in rows.items()
        for dof in ("surge", "heave")
    )
    payload = {
        "kind": "frozen-artefact reproduction",
        "frozen_file": FROZEN.name,
        "resolution": list(resolution),
        "faces_frozen": int(finest["n_faces"]),
        "faces_live": int(mesh.nb_faces),
        "tolerance": args.tolerance,
        "worst_relative": worst[0],
        "worst_cell": {"ka": float(worst[1]), "dof": worst[2]},
        "verdict": "REPRODUCED" if worst[0] <= args.tolerance else "NOT_REPRODUCED",
        "rows": rows,
        "environment": {
            "capytaine": getattr(cpt, "__version__", "unknown"),
            "numpy": np.__version__,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"faces frozen/live: {payload['faces_frozen']}/{payload['faces_live']}")
    print(f"worst relative: {worst[0]:.3e} at (ka={worst[1]}, {worst[2]}) -> "
          f"{payload['verdict']}")
    return 0 if payload["verdict"] == "REPRODUCED" else 1


if __name__ == "__main__":
    raise SystemExit(main())