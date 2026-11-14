#!/usr/bin/env python3
"""Produce the *round 2* capytaine reference: the full 6x6 radiation matrix.

Round 1 (``scripts/capytaine_crosscheck_ref.py``) cross-checked only the
**diagonal** heave and surge entries of two smooth bodies.  Everything else in
the radiation matrix -- the rotational block (roll / pitch / yaw) and the
translation-rotation coupling entries -- was never compared against a foreign
code, and the frozen file was never re-run to show that it still reproduces.

This producer closes both holes on the referee side:

* it solves **every radiating dof** and records the **whole column**
  ``A[i, j]`` / ``B[i, j]`` (capytaine returns the full column in
  ``result.added_mass`` / ``result.radiation_damping``), so off-diagonal
  entries become comparable;
* the rotation centre is set **explicitly** and recorded.  In capytaine 3.1 the
  default ``FloatingBody.rotation_center`` is ``nan`` and setting it on the body
  is refused -- it must be baked into the dofs, which is exactly the bug class
  a silent default would hide;
* the round-1 frozen reference is **re-produced live** at the round-1 meshes,
  and the deviation is written to ``round1_reproduction``;
* a deep-submergence sphere ladder is recorded so the analytic (unbounded
  fluid) Lamb / added-inertia limits can be scored on both codes.

Only this producer imports ``capytaine`` (see the optional-dependency gate in
``tests/test_capytaine_crosscheck_r2.py``).

Usage::

    python scripts/capytaine_crosscheck_r2_ref.py --out <artifact>.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

DENSITY = 1025.0
GRAVITY = 9.80665
NU_GRID = (0.5, 1.0, 2.0)
DOF_ORDER = ("surge", "sway", "heave", "roll", "pitch", "yaw")
_CAPYTAINE_NAMES = {
    "surge": "Surge",
    "sway": "Sway",
    "heave": "Heave",
    "roll": "Roll",
    "pitch": "Pitch",
    "yaw": "Yaw",
}

#: TensorBEM side uses ``hemisphere_hull_mesh(sub)`` / ``floating_sphere_hull_mesh(sub)``.
#: Here the level is the (n_theta, n_phi) pair handed to ``capytaine.mesh_sphere``.
HEMISPHERE_LEVELS = ((8, 16), (16, 32), (24, 48), (28, 56))
SPHERE_LEVELS = ((12, 24), (20, 40), (28, 56))

#: Meshes used by round 1 as its "finest coarser level"; the frozen reference
#: in ``capytaine_ref.json`` was produced at exactly these.
ROUND1_HEMISPHERE_LEVEL = (36, 72)
ROUND1_SPHERE_LEVEL = (28, 56)

#: Deep-submergence ladder (sphere radius 1, centre z = -depth) for the
#: unbounded-fluid analytic limits.
DEEP_DEPTHS = (3.0, 5.0, 8.0)
DEEP_NU = (0.5, 1.0)
DEEP_LEVELS = ((20, 40), (28, 56))


def _provenance(module: Any, source: Path | None) -> dict[str, Any]:
    commit = None
    if source is not None and (source / ".git").exists():
        try:
            commit = subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:  # pragma: no cover - provenance only
            commit = None
    return {
        "tool": "capytaine",
        "version": getattr(module, "__version__", "unknown"),
        "git_commit": commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "density": DENSITY,
        "gravity": GRAVITY,
        "water_depth": "infinite",
        "dof_order": list(DOF_ORDER),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "solver": "capytaine.BEMSolver() default settings, no lid mesh, "
        "no special quadrature override",
        "note": (
            "round 2 of the cross-code axis: full 6x6 matrix, rotational block, "
            "explicit rotation centre, live re-production of round 1"
        ),
    }


def _mesh(cpt: Any, kind: str, level: tuple[int, int]) -> Any:
    if kind == "hemisphere_r1":
        return cpt.mesh_sphere(
            radius=1.0, center=(0.0, 0.0, 0.0), resolution=level
        ).immersed_part()
    if kind == "sphere_r1_z-0.5":
        return cpt.mesh_sphere(
            radius=1.0, center=(0.0, 0.0, -0.5), resolution=level
        ).immersed_part()
    if kind == "sphere_deep":
        raise ValueError("sphere_deep is built by _mesh_deep")
    raise ValueError(f"unknown case {kind!r}")


def _mesh_deep(cpt: Any, depth: float, level: tuple[int, int]) -> Any:
    return cpt.mesh_sphere(
        radius=1.0, center=(0.0, 0.0, -float(depth)), resolution=level
    ).immersed_part()


def _body(cpt: Any, mesh: Any, rotation_center: tuple[float, float, float]) -> Any:
    """Body whose six dofs rotate about ``rotation_center``.

    The centre has to be baked into the dofs (capytaine 3.1 refuses to set it on
    the body afterwards and defaults it to ``nan``); the assertion keeps a
    silent-default regression from producing a comparison of the wrong matrix.
    """
    body = cpt.FloatingBody(
        mesh=mesh,
        dofs=cpt.rigid_body_dofs(rotation_center=np.asarray(rotation_center, dtype=float)),
    )
    centre = np.asarray(body.rotation_center, dtype=float)
    if not np.allclose(centre, np.asarray(rotation_center, dtype=float)):
        raise RuntimeError(f"rotation centre not applied: {centre!r}")
    return body


def _columns(
    cpt: Any,
    solver: Any,
    body: Any,
    dofs: tuple[str, ...],
    nu_grid: tuple[float, ...],
) -> dict[str, Any]:
    """Full ``A[:, j]`` / ``B[:, j]`` for every radiating dof ``j``."""
    out: dict[str, Any] = {}
    for nu in nu_grid:
        omega = math.sqrt(nu * GRAVITY)
        a_mat = np.zeros((6, 6))
        b_mat = np.zeros((6, 6))
        for dof in dofs:
            # the column index is the *global* dof index: a partial dof list
            # (e.g. the deep-submergence ladder solves only surge and roll) must
            # not pack its columns at the front of the matrix.
            j = DOF_ORDER.index(dof) if dof in DOF_ORDER else None
            if j is None:
                raise ValueError(f"unknown dof {dof!r}")
            res = solver.solve(
                cpt.RadiationProblem(
                    body=body,
                    omega=omega,
                    radiating_dof=_CAPYTAINE_NAMES[dof],
                    water_depth=np.inf,
                    rho=DENSITY,
                    g=GRAVITY,
                )
            )
            am = {k.lower(): float(v) for k, v in res.added_mass.items()}
            bm = {k.lower(): float(v) for k, v in res.radiation_damping.items()}
            for i, row in enumerate(DOF_ORDER):
                if row in am:
                    a_mat[i, j] = am[row]
                if row in bm:
                    b_mat[i, j] = bm[row]
        out[f"{nu:.3f}"] = {
            "added_mass": a_mat.tolist(),
            "damping": b_mat.tolist(),
        }
    return out


def _study(cpt, solver, body_factory, levels, dofs, nu_grid) -> dict[str, Any]:
    rows = []
    for level in levels:
        body = body_factory(level)
        rows.append(
            {
                "level": list(level),
                "n_faces": int(body.mesh.nb_faces),
                "volume": float(body.mesh.volume),
                "matrices": _columns(cpt, solver, body, dofs, nu_grid),
            }
        )
    return {
        "levels": rows,
        "reference": rows[-1]["matrices"],
        "reference_level": rows[-1]["level"],
        "reference_faces": rows[-1]["n_faces"],
    }


def _round1_reproduction(cpt, solver, frozen_path: Path) -> dict[str, Any]:
    """Re-run round 1's own meshes live and compare with the frozen numbers."""
    if not frozen_path.exists():
        return {"available": False, "reason": f"{frozen_path} missing"}
    frozen = json.loads(frozen_path.read_text())
    out: dict[str, Any] = {"available": True, "frozen_path": str(frozen_path),
                           "entries": [], "worst_relative": 0.0}
    specs = (
        ("hemisphere_r1", (0.0, 0.0, 0.0), ROUND1_HEMISPHERE_LEVEL, ("heave", "surge")),
        ("sphere_r1_z-0.5", (0.0, 0.0, -0.5), ROUND1_SPHERE_LEVEL, ("heave",)),
    )
    for case, centre, level, dofs in specs:
        mesh = _mesh(cpt, case, level)
        body = _body(cpt, mesh, centre)
        live = _columns(cpt, solver, body, dofs, NU_GRID)
        ref = frozen["cases"][case]["reference"]
        for nu in NU_GRID:
            key = f"{nu:.3f}"
            for dof in dofs:
                j = DOF_ORDER.index(dof)
                for comp, block in (("added_mass", "added_mass"),
                                    ("damping", "damping")):
                    base = float(ref[key][dof][comp])
                    got = float(live[key][block][j][j])
                    rel = (got - base) / abs(base) if base else float("inf")
                    out["entries"].append(
                        {
                            "case": case, "nu": nu, "dof": dof, "quantity": comp,
                            "frozen": base, "live": got, "relative": rel,
                        }
                    )
                    out["worst_relative"] = max(out["worst_relative"], abs(rel))
    return out


def build(out_path: Path, capytaine_source: Path | None) -> dict[str, Any]:
    import capytaine as cpt

    solver = cpt.BEMSolver()
    payload: dict[str, Any] = {
        "metadata": _provenance(cpt, capytaine_source),
        "nu_grid": list(NU_GRID),
        "cases": {},
    }
    payload["cases"]["hemisphere_r1"] = {
        "kind": "surface-piercing hemisphere, radius 1.0, centre (0,0,0)",
        "rotation_center": [0.0, 0.0, 0.0],
        "dofs": list(DOF_ORDER),
        **_study(
            cpt,
            solver,
            lambda lvl: _body(
                cpt, _mesh(cpt, "hemisphere_r1", lvl), (0.0, 0.0, 0.0)
            ),
            HEMISPHERE_LEVELS,
            DOF_ORDER,
            NU_GRID,
        ),
    }
    payload["cases"]["sphere_r1_z-0.5"] = {
        "kind": "floating sphere, radius 1.0, centre (0,0,-0.5)",
        "rotation_center": [0.0, 0.0, -0.5],
        "dofs": list(DOF_ORDER),
        **_study(
            cpt,
            solver,
            lambda lvl: _body(
                cpt, _mesh(cpt, "sphere_r1_z-0.5", lvl), (0.0, 0.0, -0.5)
            ),
            SPHERE_LEVELS,
            DOF_ORDER,
            NU_GRID,
        ),
    }

    # ------------------------------------------------------------------------
    # Non-degenerate rotation.  Round 2's own two bodies put the rotation centre
    # on the sphere centre of curvature, where capytaine's ``(r - centre) x n``
    # vanishes pointwise -- the rotational block is an exact zero and a relative
    # comparison of it measures the discretisation residual of a zero.  These
    # two bodies keep the geometry and move the rotation centre, which is the
    # case that actually exercises the rotational generalised normals.
    # ------------------------------------------------------------------------
    payload["cases"]["hemisphere_r1_cz-0.5"] = {
        "kind": (
            "surface-piercing hemisphere, radius 1.0, centre (0,0,0), "
            "rotation centre (0,0,-0.5)"
        ),
        "rotation_center": [0.0, 0.0, -0.5],
        "dofs": list(DOF_ORDER),
        **_study(
            cpt,
            solver,
            lambda lvl: _body(
                cpt, _mesh(cpt, "hemisphere_r1", lvl), (0.0, 0.0, -0.5)
            ),
            HEMISPHERE_LEVELS,
            DOF_ORDER,
            NU_GRID,
        ),
    }
    payload["cases"]["sphere_r1_z-0.5_c0"] = {
        "kind": (
            "floating sphere, radius 1.0, centre (0,0,-0.5), "
            "rotation centre at the waterline origin (0,0,0)"
        ),
        "rotation_center": [0.0, 0.0, 0.0],
        "dofs": list(DOF_ORDER),
        **_study(
            cpt,
            solver,
            lambda lvl: _body(
                cpt, _mesh(cpt, "sphere_r1_z-0.5", lvl), (0.0, 0.0, 0.0)
            ),
            SPHERE_LEVELS,
            DOF_ORDER,
            NU_GRID,
        ),
    }

    deep: dict[str, Any] = {"depths": list(DEEP_DEPTHS), "nu_grid": list(DEEP_NU),
                            "dofs": ["surge", "roll"], "cases": {}}
    for depth in DEEP_DEPTHS:
        centre = (0.0, 0.0, -float(depth))
        deep["cases"][f"depth_{depth:g}"] = {
            "kind": f"submerged sphere, radius 1.0, centre z = -{depth:g}",
            "rotation_center": list(centre),
            **_study(
                cpt,
                solver,
                lambda lvl, c=centre, d=depth: _body(cpt, _mesh_deep(cpt, d, lvl), c),
                DEEP_LEVELS,
                ("surge", "roll"),
                DEEP_NU,
            ),
        }
    payload["deep_submergence"] = deep

    frozen = out_path.parent.parent / "capytaine_crosscheck" / "capytaine_ref.json"
    payload["round1_reproduction"] = _round1_reproduction(cpt, solver, frozen)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/verified/capytaine_crosscheck_r2/capytaine_ref_r2.json"),
    )
    parser.add_argument(
        "--capytaine-source",
        type=Path,
        default=None,
        help="optional git checkout of capytaine, recorded as provenance",
    )
    args = parser.parse_args(argv)
    payload = build(args.out, args.capytaine_source)
    print(f"wrote {args.out}")
    print(
        f"capytaine {payload['metadata']['version']} "
        f"commit={payload['metadata']['git_commit']} "
        f"round1 reproduction {payload['round1_reproduction'].get('worst_relative')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())