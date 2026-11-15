#!/usr/bin/env python3
"""Mode-cap truncation study for the finite-depth 3-D solver (XVAL anchor).

Why this exists
---------------
The cross-validation ``bem_finite_depth_capytaine_xval`` pinned
``mode_count`` to 48/64/96 on a 320/1280-panel sphere.  ``mode_count`` is a
*cap* on the per-row evanescent-mode requirement ``_modes_for(h, r_min)``, and
on those meshes the requirement is 271-1791 (sub=2) and 526-4096 (sub=3): the
pinned study therefore truncated every row by 1-2 orders of magnitude and its
"eigenseries does not converge" verdict was an artefact of the cap, not of the
kernel.  This producer measures that directly: A33/B33 against the cap, up to
and beyond the automatic requirement, next to the cached capytaine values.

Usage
-----
    python3 truncation_cap_study.py --shard I/N   # one slice of the grid
    python3 truncation_cap_study.py --assemble    # -> truncation_cap_study.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases_trunc"
OUT = HERE / "truncation_cap_study.json"

RADIUS = 1.0
SUBMERGENCE = 2.0
DENSITY = 1025.0
#: (h/R, ka, subdivisions, cap) -- cap None means "automatic requirement"
GRID: tuple[tuple[float, float, int, int | None], ...] = (
    (5.0, 1.0, 2, 48), (5.0, 1.0, 2, 96), (5.0, 1.0, 2, 192),
    (5.0, 1.0, 2, 384), (5.0, 1.0, 2, 768), (5.0, 1.0, 2, None),
    (12.0, 1.0, 2, 48), (12.0, 1.0, 2, 192), (12.0, 1.0, 2, 768),
    (12.0, 1.0, 2, None), (5.0, 0.3, 2, None), (12.0, 0.3, 2, None),
)


def _case_id(depth_ratio: float, ka: float, sub: int, cap: int | None) -> str:
    return f"tbem-cap{'auto' if cap is None else cap}-sub{sub}-h{depth_ratio:g}-ka{ka:g}"


def _solve(depth_ratio: float, ka: float, sub: int, cap: int | None) -> dict:
    from tensorbem.finite_depth_solver3d import solve_radiation_finite_depth_3d
    from tensorbem.geometry3d import submerged_sphere_mesh

    mesh = submerged_sphere_mesh(sub, radius=RADIUS, submergence=SUBMERGENCE)
    depth = depth_ratio * RADIUS
    t0 = time.time()
    res = solve_radiation_finite_depth_3d(
        mesh, ka / RADIUS, depth, modes=[0, 1, 2], density=DENSITY,
        mode_count=cap,
    )
    return {
        "tool": "TensorBEM",
        "depth_ratio": depth_ratio,
        "ka": ka,
        "subdivisions": sub,
        "n_panels": int(mesh.panel_count),
        "mode_count": int(res.mode_count),
        "added_mass_heave": float(res.added_mass[2, 2]),
        "damping_heave": float(res.damping[2, 2]),
        "condition_number": float(res.condition_number),
        "relative_residual": float(res.relative_residual),
        "seconds": round(time.time() - t0, 1),
    }


def run_shard(index: int, total: int) -> None:
    CASES.mkdir(parents=True, exist_ok=True)
    for i, (depth_ratio, ka, sub, cap) in enumerate(GRID):
        if i % total != index:
            continue
        cid = _case_id(depth_ratio, ka, sub, cap)
        path = CASES / f"{cid}.json"
        if path.exists():
            continue
        try:
            payload = _solve(depth_ratio, ka, sub, cap)
            payload["status"] = "ok"
        except Exception:  # noqa: BLE001
            payload = {"case_id": cid, "status": "error",
                       "error": traceback.format_exc()}
        payload["case_id"] = cid
        path.write_text(json.dumps(payload, indent=1, sort_keys=True))
        print(cid, payload["status"], flush=True)


def _cpt_reference(depth_ratio: float, ka: float) -> dict | None:
    for name in (f"cpt-24x48-h{depth_ratio:g}-ka{ka:g}.json",
                 f"cpt-12x24-h{depth_ratio:g}-ka{ka:g}.json"):
        path = HERE / "cases" / name
        if path.exists():
            d = json.loads(path.read_text())
            if d.get("status") == "ok":
                return {"added_mass_heave": d.get("added_mass_heave"),
                        "damping_heave": d.get("damping_heave"),
                        "source": name}
    return None


def assemble() -> dict:
    rows = []
    for depth_ratio, ka, sub, cap in GRID:
        path = CASES / f"{_case_id(depth_ratio, ka, sub, cap)}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        if d.get("status") != "ok":
            rows.append({"case_id": d["case_id"], "status": "error"})
            continue
        ref = _cpt_reference(depth_ratio, ka)
        row = dict(d)
        row["cpt"] = ref
        if ref and ref["added_mass_heave"]:
            row["am_rel_error_vs_cpt"] = abs(
                d["added_mass_heave"] / ref["added_mass_heave"] - 1.0
            )
            row["b_rel_error_vs_cpt"] = abs(
                d["damping_heave"] / ref["damping_heave"] - 1.0
            )
        rows.append(row)

    # convergence verdict per (h/R, ka): last cap vs automatic requirement
    verdicts = []
    groups: dict[tuple[float, float], dict[str, dict]] = {}
    for row in rows:
        if row.get("status") == "error":
            continue
        key = (row["depth_ratio"], row["ka"])
        tag = "auto" if row["case_id"].startswith("tbem-capauto") else "cap"
        groups.setdefault(key, {}).setdefault(tag, []).append(row)
    for (h, ka), g in groups.items():
        auto = g.get("auto", [None])[0]
        caps = sorted(g.get("cap", []), key=lambda r: r["mode_count"])
        if auto is None or not caps:
            continue
        verdicts.append({
            "depth_ratio": h, "ka": ka,
            "auto_mode_count": auto["mode_count"],
            "am_at_auto": auto["added_mass_heave"],
            "b_at_auto": auto["damping_heave"],
            "am_rel_change_lowest_cap_to_auto": abs(
                caps[0]["added_mass_heave"] / auto["added_mass_heave"] - 1.0),
            "b_rel_change_lowest_cap_to_auto": abs(
                caps[0]["damping_heave"] / auto["damping_heave"] - 1.0),
            "caps": [c["mode_count"] for c in caps],
            "am_vs_cap": [c["added_mass_heave"] for c in caps],
            "b_vs_cap": [c["damping_heave"] for c in caps],
        })
    payload = {
        "study": "finite-depth 3-D eigenseries mode-cap truncation",
        "anchor": "XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN",
        "rows": rows,
        "verdicts": verdicts,
        "reference": "capytaine 24x48 (1152 faces) cached in cases/",
    }
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print("wrote", OUT, "rows", len(rows), "verdicts", len(verdicts))
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", nargs=2, type=int, metavar=("I", "N"))
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args()
    if args.shard:
        run_shard(args.shard[0], args.shard[1])
    if args.assemble:
        assemble()
    return 0


if __name__ == "__main__":
    sys.exit(main())