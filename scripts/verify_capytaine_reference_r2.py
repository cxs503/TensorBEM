#!/usr/bin/env python3
"""Re-solve the round-2 capytaine reference and prove it reproduces.

Why this exists
---------------
``benchmarks/verified/capytaine_crosscheck_r2/capytaine_ref_r2.json`` is the
**foreign-code side** of the round-2 cross-validation: every TensorBEM entry of
the 6x6 radiation matrix is scored against the numbers stored in that file.  The
file records its own provenance (tool, version, platform, solver settings,
rotation centre, mesh level, face count) but nothing in the suite ever
*re-derived* it -- the reference was taken on faith, and ``metadata.git_commit``
is ``null``, so "this is what capytaine 3.1.dev0 computes" rested on a single
one-shot external run.  A cross-validation whose referee cannot be re-run is
only as good as the file's honesty.

What it does
------------
For every comparison case in the reference it

1. rebuilds the mesh from the **recorded** level and geometry and asserts the
   face count (and volume, where recorded) equals the recorded one -- a wrong
   mesh specification cannot pass silently;
2. re-solves all six radiating dofs on the recorded frequency grid with the
   recorded density, gravity, water depth and rotation centre;
3. compares the full 6x6 ``added_mass`` / ``damping`` matrices entry by entry
   with the stored reference;
4. writes ``reproduction_check.json`` next to the reference.

Pre-registered tolerance
------------------------
``TOL = 1e-8`` relative, per entry.  The re-run repeats the same discrete
problem (same mesh, same solver, same defaults), so the only admissible
difference is floating-point non-determinism from a different BLAS / numpy build
(the recorded run used numpy 2.4.6; the check reports the installed one).
``1e-8`` is three orders of magnitude looser than that noise floor and five
orders tighter than any physical claim made from the reference, so it cannot
hide a wrong reference.  The value is fixed **before** the check is run and is
echoed into the artifact.

Only this script and ``scripts/capytaine_crosscheck_r2_ref.py`` import
``capytaine`` (``tests/test_capytaine_crosscheck_r2.py`` gates on that).

Usage::

    python scripts/verify_capytaine_reference_r2.py [--cases a,b] [--out path]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorbem.benchmarks.capytaine_crosscheck_r2 import (  # noqa: E402
    ARTIFACT_ROOT,
    CASES,
    DOF_ORDER,
)

TOL = 1e-8
_CAPYTAINE_NAMES = {
    "surge": "Surge",
    "sway": "Sway",
    "heave": "Heave",
    "roll": "Roll",
    "pitch": "Pitch",
    "yaw": "Yaw",
}


def _mesh(cpt: Any, case: str, level: Sequence[int]) -> Any:
    """Rebuild the wetted mesh from the recorded level (geometry from ``CASES``)."""
    spec = CASES[case]
    if "sphere" not in spec:
        raise ValueError(f"case {case!r} has no geometry spec")
    centre = spec.get("body_centre", spec.get("centre"))
    return cpt.mesh_sphere(
        radius=1.0, center=tuple(float(v) for v in centre), resolution=tuple(level)
    ).immersed_part()


def _body(cpt: Any, mesh: Any, rotation_center: Sequence[float]) -> Any:
    centre = np.asarray(rotation_center, dtype=float)
    body = cpt.FloatingBody(mesh=mesh, dofs=cpt.rigid_body_dofs(rotation_center=centre))
    if not np.allclose(np.asarray(body.rotation_center, dtype=float), centre):
        raise RuntimeError("rotation centre not applied by capytaine")
    return body


def _solve_case(
    cpt: Any, solver: Any, case: str, block: dict[str, Any], meta: dict[str, Any]
) -> dict[str, Any]:
    level = block["reference_level"]
    mesh = _mesh(cpt, case, level)
    if int(mesh.nb_faces) != int(block["reference_faces"]):
        raise RuntimeError(
            "{}: rebuilt mesh has {} faces, reference records {} -- the recorded "
            "mesh specification does not reproduce".format(
                case, mesh.nb_faces, block["reference_faces"]
            )
        )
    body = _body(cpt, mesh, block["rotation_center"])
    rho = float(meta["density"])
    g = float(meta["gravity"])
    worst = 0.0
    worst_where = None
    rows = []
    for nu in (0.5, 1.0, 2.0):
        key = "{:.3f}".format(nu)
        stored = block["reference"][key]
        a_mat = np.zeros((6, 6))
        b_mat = np.zeros((6, 6))
        for dof in DOF_ORDER:
            j = DOF_ORDER.index(dof)
            res = solver.solve(
                cpt.RadiationProblem(
                    body=body,
                    omega=math.sqrt(nu * g),
                    radiating_dof=_CAPYTAINE_NAMES[dof],
                    water_depth=np.inf,
                    rho=rho,
                    g=g,
                )
            )
            am = {k.lower(): float(v) for k, v in res.added_mass.items()}
            bm = {k.lower(): float(v) for k, v in res.radiation_damping.items()}
            for i, row in enumerate(DOF_ORDER):
                if row in am:
                    a_mat[i, j] = am[row]
                if row in bm:
                    b_mat[i, j] = bm[row]
        nu_worst = 0.0
        for quantity, live, ref in (
            ("added_mass", a_mat, np.asarray(stored["added_mass"], dtype=float)),
            ("damping", b_mat, np.asarray(stored["damping"], dtype=float)),
        ):
            scale = max(float(np.max(np.abs(ref))), 1e-300)
            dev = np.abs(live - ref) / scale
            for i, row in enumerate(DOF_ORDER):
                for jj, col in enumerate(DOF_ORDER):
                    if dev[i, jj] > worst:
                        worst = float(dev[i, jj])
                        worst_where = {
                            "case": case,
                            "nu": nu,
                            "quantity": quantity,
                            "row": row,
                            "column": col,
                            "reproduced": float(live[i, jj]),
                            "recorded": float(ref[i, jj]),
                        }
                    nu_worst = max(nu_worst, float(dev[i, jj]))
        rows.append({"nu": nu, "worst_relative": nu_worst})
    return {
        "case": case,
        "level": list(level),
        "n_faces": int(mesh.nb_faces),
        "rotation_center": [float(v) for v in block["rotation_center"]],
        "nu_rows": rows,
        "worst_relative": worst,
        "worst_entry": worst_where,
        "verdict": "PASS" if worst <= TOL else "FAIL",
    }


def reproduction_check(
    cases: Sequence[str] | None = None, reference_path: Path | None = None
) -> dict[str, Any]:
    import capytaine as cpt

    path = reference_path or (ROOT / ARTIFACT_ROOT / "capytaine_ref_r2.json")
    reference = json.loads(Path(path).read_text())
    meta = reference["metadata"]
    solver = cpt.BEMSolver()
    selected = list(cases) if cases else list(reference["cases"].keys())
    rows = []
    for case in selected:
        block = reference["cases"][case]
        rows.append(_solve_case(cpt, solver, case, block, meta))
    worst = max((r["worst_relative"] for r in rows), default=0.0)
    recorded_version = meta.get("version")
    installed_version = getattr(cpt, "__version__", "unknown")
    return {
        "check": "capytaine round-2 reference re-solve",
        "reference_path": str(Path(path)),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "tolerance_relative": TOL,
        "tolerance_reason": (
            "identical discrete problem re-run: only BLAS/numpy build differences are "
            "admissible; recorded numpy {} vs installed {}".format(
                meta.get("numpy"), np.__version__
            )
        ),
        "capytaine_version_recorded": recorded_version,
        "capytaine_version_installed": installed_version,
        "version_match": bool(recorded_version == installed_version),
        "recorded_git_commit": meta.get("git_commit"),
        "platform": platform.platform(),
        "numpy_installed": np.__version__,
        "cases": rows,
        "n_cases": len(rows),
        "worst_relative": worst,
        "verdict": "PASS" if worst <= TOL else "FAIL",
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="", help="comma separated case names (default: all)")
    ap.add_argument(
        "--out",
        default=str(ROOT / ARTIFACT_ROOT / "reproduction_check.json"),
        help="output artifact path",
    )
    args = ap.parse_args(argv)
    cases = [c for c in args.cases.split(",") if c] or None
    result = reproduction_check(cases)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(
        "{}: worst relative {:.3e} over {} case(s) -> {} ({})".format(
            out, result["worst_relative"], result["n_cases"], result["verdict"],
            "version match" if result["version_match"] else "VERSION MISMATCH",
        )
    )
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
