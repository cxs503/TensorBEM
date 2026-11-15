#!/usr/bin/env python3
"""Case-mesh (1280-panel) computation of the Havelock submerged-sphere band rows.

Motivation
----------
``CASE-BEM-SUBMERGED-SPHERE-HAVELOCK`` gates the full band ``ka = 0.1 ... 5.0``
on the 1280-panel case mesh with a 5 % relative tolerance.  Rows ``ka = 4.0``
and ``ka = 5.0`` in ``docs/case_bem_submerged_sphere_havelock/numbers.json``
were, until this script ran, **GCI-scaled estimates** derived from the
320-panel economy mesh (``est = ref + (econ - ref) / r**p``, ``r = 2``,
``p = 1.91`` taken from the ``ka = 0.8`` convergence ladder), with a placeholder
``condition_number = 2.05``.  The round-4 adversarial audit (m21580) flagged
exactly that as a major finding: a 5 % PASS over the full band cannot rest on
two extrapolated rows when the claim is "every item is computed".

What this script does
---------------------
1. **Route-identity control** -- re-runs ``ka = 3.0`` through *exactly* the code
   path used for the rest of the case-mesh table
   (:func:`tensorbem.benchmarks.submerged_sphere_havelock.run_frequency_sweep`,
   ``subdivision = 3`` -> 1280 panels, flat-panel measure, the committed
   production route) and compares the result digit-by-digit with the committed
   row.  Only if that reproduces does the new ``ka = 4,5`` numbers belong to the
   same table.
2. **Computation** -- same call, ``ka = 4.0`` and ``ka = 5.0``.
3. **Geometry / conditioning audit** -- panel count, wetted-area deficit against
   ``4 pi a^2``, mean panel size, full-precision ``condition_number`` and
   ``relative_residual`` of the assembled system.
4. **Provenance** -- git HEAD/branch/dirty flag, package versions, wall time, so
   the artefact can be tied to an exact commit (no hand-copied numbers).

Output: a single JSON artefact consumed by ``scripts/havelock_ka45_apply.py``.

Usage
-----
    PYTHONPATH=src python3 scripts/havelock_ka45_case_mesh_compute.py \
        --out docs/case_bem_submerged_sphere_havelock/ka45_case_mesh_computed.json
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tensorbem.benchmarks import submerged_sphere_havelock as library
from tensorbem.geometry3d import submerged_sphere_mesh

RADIUS = 1.0
SUBMERGENCE = 2.0
DENSITY = 1025.0
GRAVITY = 9.80665
SUBDIVISION = 3                 # 4 * 4**3 = 1280 panels (case mesh)
CONTROL_KA = 3.0                # already computed on this mesh -> route control
TARGET_KA = (4.0, 5.0)          # the rows that must be replaced
COEFFICIENTS = ("a33", "b33", "a11", "b11")
NUMBERS = Path("docs/case_bem_submerged_sphere_havelock/numbers.json")
ROUTE_CONTROL_RTOL = 1.0e-12    # digit-identity between two runs of the same path


def _git(*args: str) -> str:
    try:
        return subprocess.run(("git",) + args, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:                                    # pragma: no cover
        return ""


def _provenance() -> dict:
    return {
        "git_head": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def _geometry_audit() -> dict:
    """Panel count / wetted area / panel size of the case mesh."""
    mesh = submerged_sphere_mesh(SUBDIVISION, radius=RADIUS, submergence=SUBMERGENCE)
    # submerged_sphere_mesh already centres the sphere at z = -submergence
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tri_area = 0.5 * np.linalg.norm(
        np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                 verts[faces[:, 2]] - verts[faces[:, 0]]), axis=1)
    poly_area = float(tri_area.sum())
    sphere_area = 4.0 * np.pi * RADIUS**2
    centre = np.array([0.0, 0.0, -SUBMERGENCE])
    radii = np.linalg.norm(verts - centre, axis=1)
    return {
        "subdivision": SUBDIVISION,
        "n_panels": int(mesh.panel_count),
        "h_mean": float(mesh.h_mean),
        "polyhedral_area": poly_area,
        "sphere_area": sphere_area,
        "area_deficit": 1.0 - poly_area / sphere_area,
        "max_radial_error": float(np.max(np.abs(radii - RADIUS))),
        "verdict": "PASS" if abs(poly_area / sphere_area - 1.0) < 0.02 else "FAIL",
    }


def _compare(row: dict, committed: dict) -> dict:
    """Relative difference between a freshly computed row and the committed one."""
    diffs = {}
    for key in COEFFICIENTS:
        for suffix in ("bem", "exact"):
            field = f"{key}_{suffix}"
            a, b = float(row[field]), float(committed[field])
            diffs[field] = abs(a - b) / max(abs(b), 1.0e-30)
    return {
        "ka": float(row["ka"]),
        "max_rel_diff": max(diffs.values()),
        "per_field_rel_diff": diffs,
        "committed_condition_number": committed.get("condition_number"),
        "recomputed_condition_number": row.get("condition_number"),
        "route_identical": max(diffs.values()) < ROUTE_CONTROL_RTOL,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path,
                        default=Path("docs/case_bem_submerged_sphere_havelock/ka45_case_mesh_computed.json"))
    parser.add_argument("--numbers", type=Path, default=NUMBERS)
    parser.add_argument("--ka", type=str, default=None,
                        help="comma-separated subset of the (control + target) wavenumbers; "
                             "allows the run to be split into bounded chunks")
    parser.add_argument("--merge", type=Path, nargs="+", default=None,
                        help="merge chunk artefacts instead of computing")
    args = parser.parse_args(argv)
    if args.merge:
        return _merge(args.merge, args.out)

    committed = json.loads(args.numbers.read_text())
    committed_rows = {float(r["ka"]): r for r in committed["band_sweep_case"]}

    geometry = _geometry_audit()
    provenance = _provenance()

    ka_list = ((CONTROL_KA,) + TARGET_KA) if args.ka is None else tuple(
        float(x) for x in args.ka.split(","))
    started = time.perf_counter()
    rows = library.run_frequency_sweep(
        ka_values=ka_list,
        radius=RADIUS, submergence=SUBMERGENCE, subdivision=SUBDIVISION,
        density=DENSITY, gravity=GRAVITY,
    )
    elapsed = time.perf_counter() - started

    control = None
    targets = list(rows)
    if CONTROL_KA in ka_list:
        control = _compare(rows[0], committed_rows[CONTROL_KA])
        targets = [r for r in rows if float(r["ka"]) != CONTROL_KA]

    # How far were the GCI estimates from the computed values?  (audit of the
    # claim "estimates are within 0.2 % of the computed values")
    estimate_vs_computed = []
    for row in targets:
        old = committed_rows[float(row["ka"])]
        entry = {"ka": float(row["ka"]), "quantities": {}}
        for key in COEFFICIENTS:
            est = float(old[f"{key}_bem"])
            got = float(row[f"{key}_bem"])
            entry["quantities"][key] = {
                "gci_estimate": est,
                "computed": got,
                "rel_diff": abs(est - got) / max(abs(got), 1.0e-30),
                "gci_estimate_rel_error": float(old[f"{key}_rel_error"]),
                "computed_rel_error": float(row[f"{key}_rel_error"]),
            }
        estimate_vs_computed.append(entry)

    payload = {
        "case": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
        "purpose": "replace GCI-scaled ka=4,5 rows by 1280-panel BEM computations",
        "parameters": {
            "radius_m": RADIUS, "submergence_m": SUBMERGENCE,
            "density_kg_m3": DENSITY, "gravity_m_s2": GRAVITY,
            "subdivision": SUBDIVISION, "area_measure": "flat",
            "route": "tensorbem.benchmarks.submerged_sphere_havelock.run_frequency_sweep",
        },
        "provenance": provenance,
        "elapsed_seconds": round(elapsed, 2),
        "geometry_audit": geometry,
        "route_control": control,
        "ka_values": list(ka_list),
        "results": targets,
        "estimate_vs_computed": estimate_vs_computed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    if control is not None:
        print(f"route control ka={CONTROL_KA}: max rel diff {control['max_rel_diff']:.3e} "
              f"-> {'IDENTICAL' if control['route_identical'] else 'MISMATCH'}")
    else:
        print(f"no route control in this chunk (ka list {ka_list}); "
              "merge with the control chunk before applying")
    for entry in estimate_vs_computed:
        print(f"ka={entry['ka']:g}: " + ", ".join(
            f"{k} err {v['gci_estimate_rel_error']*100:.3f}% -> {v['computed_rel_error']*100:.3f}% "
            f"(|est-computed|/computed {v['rel_diff']*100:.3f}%)"
            for k, v in entry["quantities"].items()))
    print(f"elapsed {elapsed:.1f}s; wrote {args.out}")
    return 0


def _merge(parts: list[Path], out: Path) -> int:
    """Merge chunk artefacts into one payload (used when the run is split)."""
    merged: dict = {"results": [], "estimate_vs_computed": [], "elapsed_seconds": 0.0}
    for part in parts:
        data = json.loads(part.read_text())
        merged["provenance"] = data["provenance"]
        merged["parameters"] = data["parameters"]
        merged["case"] = data["case"]
        merged["purpose"] = data["purpose"]
        merged["geometry_audit"] = data["geometry_audit"]
        merged["elapsed_seconds"] = round(
            merged["elapsed_seconds"] + float(data.get("elapsed_seconds") or 0.0), 2)
        merged["results"].extend(data["results"])
        merged["estimate_vs_computed"].extend(data["estimate_vs_computed"])
        if data.get("route_control"):
            merged["route_control"] = data["route_control"]
    merged["results"].sort(key=lambda r: float(r["ka"]))
    merged["estimate_vs_computed"].sort(key=lambda e: float(e["ka"]))
    merged["ka_values"] = [float(r["ka"]) for r in merged["results"]]
    assert "route_control" in merged, "no chunk carried the ka=3.0 route control"
    assert len(merged["results"]) == len(TARGET_KA), merged["ka_values"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"merged {len(parts)} chunks -> {out} ({merged['ka_values']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())