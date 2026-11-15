#!/usr/bin/env python3
"""Same-surface cross-code check against capytaine.

Round 2 of the cross-code axis (anchor ``XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE``)
compared TensorBEM's finest own mesh with capytaine's finest own mesh.  The
remaining ``worst_active_relative`` of 6.7 % is then a *sum* of two unrelated
discretisations, and the deviation cannot say which code is responsible.

This producer removes that confound.  capytaine's panel surface is converted
into a :class:`tensorbem.geometry3d.WettedHullMesh` (each quad split into two
triangles along a diagonal, degenerate halved quads dropped, winding checked
against the waterline-closure invariant) and TensorBEM is solved **on exactly
capytaine's surface**.  The residual is then attributable to the formulation
(Green function, quadrature, generalised normals, rotation centre), not to a
different tessellation of the same body.

The capytaine column is *not* recomputed here: it is read from the already
committed ``capytaine_ref_r2.json``, whose level resolutions are the same
``(n_theta, n_phi)`` tuples, and whose numbers were re-derived bit-identically
by an independent solve (see ``docs/capytaine_same_surface_crossval.md``).

Only this producer imports ``capytaine``; the committed artifact carries the
converted mesh (vertices + triangles) so the tests can rebuild the same surface
without the optional dependency.

Usage::

    PYTHONPATH=src python scripts/capytaine_same_surface_crossval.py
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

from tensorbem.benchmarks.capytaine_crosscheck_r2 import (
    DENSITY,
    DOF_ORDER,
    GRAVITY,
    NULL_FLOOR,
    NU_GRID,
    REFERENCE_FILE,
    TOL_DIAGONAL,
    TOL_OFFDIAGONAL,
    invariant_null_dofs,
    physical_scale,
    reference_scale,
)
from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem.geometry3d import WettedHullMesh

ARTIFACT_ROOT = Path("benchmarks/verified/capytaine_same_surface_crossval")
RESULT_FILE = "results.json"

#: ``(case, capytaine resolution level, rotation centre)``.  The resolutions are
#: the capytaine-side ladders of ``capytaine_crosscheck_r2``; the coarsest and
#: the finest are kept so the refinement trend is visible without paying for
#: every intermediate level twice (each centre re-assembles the matrices).
HEMISPHERE_LEVELS = ((8, 16), (16, 32), (28, 56))
SPHERE_LEVELS = ((12, 24), (20, 40), (28, 56))

CASES: dict[str, dict[str, Any]] = {
    "hemisphere_r1": {
        "kind": "surface-piercing hemisphere, radius 1.0 m, centre (0, 0, 0)",
        "ref_case": "hemisphere_r1",
        "sphere": False,
        "body_centre": (0.0, 0.0, 0.0),
        "centre": (0.0, 0.0, 0.0),
        "levels": HEMISPHERE_LEVELS,
    },
    "hemisphere_r1_cz-0.5": {
        "kind": "surface-piercing hemisphere, rotation centre (0, 0, -0.5)",
        "ref_case": "hemisphere_r1",
        "sphere": False,
        "body_centre": (0.0, 0.0, 0.0),
        "centre": (0.0, 0.0, -0.5),
        "levels": HEMISPHERE_LEVELS,
    },
    "sphere_r1_z-0.5": {
        "kind": "floating sphere radius 1.0 m, centre (0, 0, -0.5)",
        "ref_case": "sphere_r1_z-0.5",
        "sphere": True,
        "body_centre": (0.0, 0.0, -0.5),
        "centre": (0.0, 0.0, -0.5),
        "levels": SPHERE_LEVELS,
    },
    "sphere_r1_z-0.5_c0": {
        "kind": "floating sphere, rotation centre at the waterline origin",
        "ref_case": "sphere_r1_z-0.5",
        "sphere": True,
        "body_centre": (0.0, 0.0, -0.5),
        "centre": (0.0, 0.0, 0.0),
        "levels": SPHERE_LEVELS,
    },
}


# --------------------------------------------------------------------------- #
# capytaine surface -> TensorBEM wetted hull
# --------------------------------------------------------------------------- #
def surface_from_capytaine(mesh: Any) -> dict[str, Any]:
    """Convert a capytaine panel mesh into vertices + triangles.

    Quads are split along the ``0-2`` diagonal; the split halves that collapse
    to a repeated vertex (capytaine's ``immersed_part`` produces those on the
    waterline) are dropped.  The winding is verified by the
    :class:`WettedHullMesh` invariant itself: a flip is attempted only if the
    original orientation is rejected, and the outcome is recorded.
    """
    verts = np.asarray(mesh.vertices, dtype=float)
    quads = np.asarray(mesh.faces)
    if quads.shape[1] == 4:
        tris = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    elif quads.shape[1] == 3:
        tris = quads
    else:
        raise ValueError(f"unsupported panel arity {quads.shape[1]}")
    keep = (
        (tris[:, 0] != tris[:, 1])
        & (tris[:, 1] != tris[:, 2])
        & (tris[:, 0] != tris[:, 2])
    )
    full_split = np.asarray(tris, dtype=np.int64)
    tris = np.asarray(tris[keep], dtype=np.int64)
    flipped = False
    try:
        hull = WettedHullMesh(verts, tris)
    except ValueError:
        flipped = True
        hull = WettedHullMesh(verts, tris[:, ::-1])
    # area of the *unsplit* capytaine surface: the two triangle halves of every
    # quad (degenerate halves contribute exactly zero)
    cap_area = 0.5 * float(np.sum(np.linalg.norm(
        np.cross(verts[full_split[:, 1]] - verts[full_split[:, 0]],
                 verts[full_split[:, 2]] - verts[full_split[:, 0]]), axis=1)))
    tri_area = 0.5 * float(np.sum(np.linalg.norm(
        np.cross(verts[tris[:, 1]] - verts[tris[:, 0]],
                 verts[tris[:, 2]] - verts[tris[:, 0]]), axis=1)))
    return {
        "hull": hull,
        "vertices": verts.tolist(),
        "faces": np.asarray(hull.faces, dtype=np.int64).tolist(),
        "winding_flipped": flipped,
        "capytaine_faces": int(quads.shape[0]),
        "triangles": int(tris.shape[0]),
        "capytaine_area": cap_area,
        "triangle_area": tri_area,
        "area_relative_difference": (
            abs(tri_area - cap_area) / cap_area if cap_area > 0 else float("nan")
        ),
    }


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
def score(
    case: str, tb: dict[str, np.ndarray], cap: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Classify the 6x6 entries exactly like ``capytaine_crosscheck_r2.compare``.

    ``null`` / ``invariant`` entries are the ones a 6x6 comparison must not
    score as relative deviations (their scale *is* a floor); they are reported
    with their ratio to the physical scale instead.
    """
    null_index = {DOF_ORDER.index(dof) for dof in invariant_null_dofs(case)}
    entries: list[dict[str, Any]] = []
    worst_diag = 0.0
    worst_off = 0.0
    worst_null = 0.0
    for quantity in ("added_mass", "damping"):
        a_tb = tb[quantity]
        a_cap = cap[quantity]
        for i in range(6):
            for j in range(6):
                entry: dict[str, Any] = {
                    "quantity": quantity,
                    "row": DOF_ORDER[i],
                    "column": DOF_ORDER[j],
                    "tensorbem": float(a_tb[i, j]),
                    "capytaine": float(a_cap[i, j]),
                }
                phys = physical_scale(i, j)
                if i in null_index or j in null_index:
                    biggest = max(abs(a_tb[i, j]), abs(a_cap[i, j]))
                    entry["class"] = "invariant"
                    entry["null_ratio"] = biggest / phys
                    worst_null = max(worst_null, entry["null_ratio"])
                    entries.append(entry)
                    continue
                scale, degenerate = reference_scale(a_cap, i, j)
                entry["scale"] = scale
                entry["scale_degenerate"] = bool(degenerate)
                if not math.isfinite(scale) or scale <= NULL_FLOOR * phys:
                    entry["class"] = "unscorable"
                    entries.append(entry)
                    continue
                cap_null = abs(a_cap[i, j]) <= NULL_FLOOR * scale
                tb_null = abs(a_tb[i, j]) <= NULL_FLOOR * scale
                if cap_null and tb_null:
                    entry["class"] = "null"
                    entry["null_ratio"] = max(
                        abs(a_tb[i, j]), abs(a_cap[i, j])
                    ) / scale
                    worst_null = max(worst_null, entry["null_ratio"])
                elif cap_null != tb_null:
                    entry["class"] = "conflict"
                else:
                    relative = (a_tb[i, j] - a_cap[i, j]) / abs(a_cap[i, j])
                    entry["class"] = "active"
                    entry["relative"] = float(relative)
                    entry["band"] = TOL_DIAGONAL if i == j else TOL_OFFDIAGONAL
                    entry["within_band"] = bool(abs(relative) <= entry["band"])
                    if i == j:
                        worst_diag = max(worst_diag, abs(relative))
                    else:
                        worst_off = max(worst_off, abs(relative))
                entries.append(entry)
    classes = [e["class"] for e in entries]
    return {
        "entries": entries,
        "n_entries": len(entries),
        "n_active": classes.count("active"),
        "n_null": classes.count("null"),
        "n_invariant": classes.count("invariant"),
        "n_conflict": classes.count("conflict"),
        "n_unscorable": classes.count("unscorable"),
        "worst_diagonal": float(worst_diag),
        "worst_coupling": float(worst_off),
        "worst_null_ratio": float(worst_null),
        "verdict": (
            "PASS"
            if worst_diag <= TOL_DIAGONAL and worst_off <= TOL_OFFDIAGONAL
            else "FAIL"
        ),
    }


def solve_on_surface(
    hull: WettedHullMesh, nu: float, centre: tuple[float, float, float]
) -> dict[str, np.ndarray]:
    res = solve_radiation_3d(
        hull,
        nu,
        modes=list(range(6)),
        center=centre,
        density=DENSITY,
        gravity=GRAVITY,
    )
    return {
        "added_mass": np.asarray(res.added_mass, dtype=float),
        "damping": np.asarray(res.damping, dtype=float),
        "condition_number": float(res.condition_number),
        "relative_residual": float(res.relative_residual),
    }


def reference_case(reference: dict[str, Any], case: str, level: tuple[int, int]) -> dict[str, Any]:
    """capytaine block for one resolution, keyed by ``(n_theta, n_phi)``."""
    block = reference["cases"][CASES[case]["ref_case"]]
    for row in block["levels"]:
        if tuple(int(v) for v in row["level"]) == tuple(level):
            return row["matrices"]
    raise KeyError(f"{case}: capytaine level {level} missing from the reference")


def run(
    capytaine_module: Any,
    reference: dict[str, Any],
    cases: list[str],
    on_case: Any = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for case in cases:
        spec = CASES[case]
        centre = tuple(float(v) for v in spec["centre"])
        rows = []
        for level in spec["levels"]:
            if spec["sphere"]:
                mesh = capytaine_module.mesh_sphere(
                    radius=1.0, center=spec["body_centre"], resolution=level
                ).immersed_part()
            else:
                mesh = capytaine_module.mesh_sphere(
                    radius=1.0, center=(0.0, 0.0, 0.0), resolution=level
                ).immersed_part()
            surface = surface_from_capytaine(mesh)
            hull = surface["hull"]
            cap_level = reference_case(reference, case, level)
            matrices: dict[str, Any] = {}
            scores: dict[str, Any] = {}
            for nu in NU_GRID:
                key = f"{nu:.3f}"
                tb = solve_on_surface(hull, nu, centre)
                cap = {
                    "added_mass": np.asarray(cap_level[key]["added_mass"], dtype=float),
                    "damping": np.asarray(cap_level[key]["damping"], dtype=float),
                }
                matrices[key] = {
                    "tensorbem_added_mass": tb["added_mass"].tolist(),
                    "tensorbem_damping": tb["damping"].tolist(),
                    "condition_number": tb["condition_number"],
                    "relative_residual": tb["relative_residual"],
                }
                scores[key] = score(case, tb, cap)
            rows.append(
                {
                    "level": list(level),
                    "n_faces": surface["capytaine_faces"],
                    "triangles": surface["triangles"],
                    "winding_flipped": surface["winding_flipped"],
                    "capytaine_area": surface["capytaine_area"],
                    "triangle_area": surface["triangle_area"],
                    "area_relative_difference": surface["area_relative_difference"],
                    "matrices": matrices,
                    "scores": scores,
                    "vertices": surface["vertices"],
                    "faces": surface["faces"],
                }
            )
        out[case] = {
            "kind": spec["kind"],
            "ref_case": spec["ref_case"],
            "rotation_center": list(centre),
            "levels": rows,
        }
        if on_case is not None:
            # save after every case: on a shared cluster the process may be
            # killed mid-study, and a completed case must not be lost
            on_case(out)
    return out


def summarise(study: dict[str, Any]) -> dict[str, Any]:
    """Refinement trend of the same-surface deviation, per case and quantity."""
    out: dict[str, Any] = {}
    for case, block in study.items():
        rows = []
        for row in block["levels"]:
            worst_diag = max(s["worst_diagonal"] for s in row["scores"].values())
            worst_off = max(s["worst_coupling"] for s in row["scores"].values())
            rows.append(
                {
                    "level": row["level"],
                    "n_faces": row["n_faces"],
                    "triangles": row["triangles"],
                    "worst_diagonal": worst_diag,
                    "worst_coupling": worst_off,
                    "worst_active": max(worst_diag, worst_off),
                }
            )
        first, last = rows[0], rows[-1]
        out[case] = {
            "rows": rows,
            "worst_active_coarsest": first["worst_active"],
            "worst_active_finest": last["worst_active"],
            "ratio_coarsest_over_finest": (
                first["worst_active"] / last["worst_active"]
                if last["worst_active"] > 0
                else float("inf")
            ),
            "verdict": "PASS"
            if last["worst_diagonal"] <= TOL_DIAGONAL
            and last["worst_coupling"] <= TOL_OFFDIAGONAL
            else "FAIL",
        }
    return out


def provenance(source: Path | None, module: Any) -> dict[str, Any]:
    commit = None
    if source is not None and (source / ".git").exists():
        try:
            commit = subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:  # pragma: no cover
            commit = None
    return {
        "tool": "capytaine",
        "version": getattr(module, "__version__", "unknown"),
        "git_commit": commit,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "density": DENSITY,
        "gravity": GRAVITY,
        "water_depth": "infinite",
        "dof_order": list(DOF_ORDER),
        "nu_grid": list(NU_GRID),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "capytaine_column": f"read from {REFERENCE_FILE} (same resolution levels)",
        "tensorbem_column": "solve_radiation_3d on capytaine's panel surface "
        "(quads split into two triangles, degenerate halves dropped)",
        "note": "same-surface crossval: removes the different-tessellation "
        "confound of the own-mesh comparison",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ARTIFACT_ROOT / RESULT_FILE)
    parser.add_argument("--reference", type=Path, default=Path("benchmarks/verified")
                        / "capytaine_crosscheck_r2" / REFERENCE_FILE)
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    args = parser.parse_args(argv)

    import capytaine as cpt  # noqa: PLC0415  (optional dependency)

    reference = json.loads(args.reference.read_text())
    prov = provenance(Path("."), cpt)

    def save(partial: dict[str, Any]) -> None:
        payload = {
            "provenance": prov,
            "reference_file": str(args.reference),
            "cases": partial,
            "summary": summarise(partial),
            "tolerance_diagonal": TOL_DIAGONAL,
            "tolerance_offdiagonal": TOL_OFFDIAGONAL,
            "null_floor": NULL_FLOOR,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1))

    study = run(cpt, reference, args.cases, on_case=save)
    save(study)
    payload = json.loads(args.out.read_text())
    for case, block in payload["summary"].items():
        print(
            f"{case}: coarsest {block['worst_active_coarsest']*100:.1f}% -> "
            f"finest {block['worst_active_finest']*100:.1f}% ({block['verdict']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())