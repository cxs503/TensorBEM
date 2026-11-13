#!/usr/bin/env python3
"""Independent re-check of the CASE-BEM-HULME-HEMISPHERE numbers (m17235 audit).

Why this exists
---------------
Mission m17181 graded the floating-hemisphere case against
``tensorbem.analytic.hulme_hemisphere_radiation`` -- which its own
``results.json`` provenance records as a *re-derived* series, because the
published Hulme (1982) table digits were "NOT AVAILABLE in this sandbox".
An audit therefore cannot take the reference on trust.  This script measures
the same physical quantities with a third-party BEM code (capytaine, Rankine
panel method, installed as an optional dependency and never imported by
``src/``), at increasing panel counts, and compares the three numbers:

    got  <- the case's TensorBEM result      (m17181 results.json)
    ref  <- the reference the case trusted   (m17181 results.json)
    ind  <- capytaine, this script, resolution ladder

Usage
-----
    PYTHONPATH=src python3 scripts/independent_hemisphere_check.py \
        --case-dir /root/tp-agent/work/m17181/runs/casepool_bem_hulme_hemisphere \
        --out artifacts/independent_hemisphere_check.json

Read-only with respect to the audited mission: it only reads the m17181
artefacts and never writes into that worktree.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

DENSITY = 1025.0
GRAVITY = 9.80665
RADIUS = 1.0
VOLUME = (2.0 / 3.0) * math.pi * RADIUS**3
RESOLUTIONS = ((20, 40), (28, 56), (36, 72), (44, 88))
NU_GRID = (1.0, 2.0)
DOFS = ("Heave", "Surge")


def _load_case(case_dir: Path) -> dict:
    results = json.loads((case_dir / "results.json").read_text())
    ref = results["reference"]["values"] if "values" in results["reference"] else results["reference"]
    # normalise the reference block: {ka: {quantity: value}}
    ref_rows: dict[str, dict[str, float]] = {}
    for key, block in ref.items():
        if isinstance(block, dict) and "A11_dimless" in block:
            ref_rows[f"{float(key):.3f}"] = block
        elif isinstance(block, dict):
            for sub in block.values():
                if isinstance(sub, dict) and "A11_dimless" in sub:
                    ref_rows[f"{float(key):.3f}"] = sub
    got_rows: dict[str, dict[str, float]] = {}
    for row in results["rows"]:
        if row.get("kind") != "band" or int(row["subdivisions"]) != 3:
            continue
        got_rows[f"{float(row['ka']):.3f}"] = row
    return {"results": results, "ref": ref_rows, "got": got_rows}


def _capytaine_sweep() -> dict:
    import capytaine as cpt

    solver = cpt.BEMSolver()
    levels = []
    for resolution in RESOLUTIONS:
        mesh = cpt.mesh_sphere(
            radius=RADIUS, center=(0.0, 0.0, 0.0), resolution=resolution
        ).immersed_part()
        body = cpt.FloatingBody(mesh=mesh)
        body.add_all_rigid_body_dofs()
        per_freq = {}
        for nu in NU_GRID:
            omega = math.sqrt(nu * GRAVITY)
            block = {}
            for dof in DOFS:
                res = solver.solve(
                    cpt.RadiationProblem(
                        body=body,
                        omega=omega,
                        radiating_dof=dof,
                        water_depth=np.inf,
                        rho=DENSITY,
                        g=GRAVITY,
                    )
                )
                block[dof.lower()] = {
                    "added_mass": float(res.added_mass[dof]),
                    "damping": float(res.radiation_damping[dof]),
                }
            per_freq[f"{nu:.3f}"] = block
        levels.append(
            {
                "resolution": list(resolution),
                "n_faces": int(mesh.nb_faces),
                "mesh_volume": float(mesh.volume),
                "values": per_freq,
            }
        )
        print(
            f"[capytaine] resolution={resolution} faces={mesh.nb_faces} "
            f"volume={mesh.volume:.6f} done",
            flush=True,
        )
    return {
        "metadata": {
            "tool": "capytaine",
            "version": getattr(cpt, "__version__", "unknown"),
            "python": sys.version.split()[0],
            "machine": platform.machine(),
            "numpy": np.__version__,
            "rho": DENSITY,
            "g": GRAVITY,
            "water_depth": "infinite",
            "body": "surface-piercing hemisphere R=1, centre (0,0,0), flat face at z=0",
            "resolutions": [list(r) for r in RESOLUTIONS],
        },
        "levels": levels,
    }


def _compare(case: dict, independent: dict) -> list[dict]:
    """got/ref/independent, all non-dimensionalised identically."""
    rows = []
    finest = independent["levels"][-1]
    for nu_key, block in finest["values"].items():
        ka = float(nu_key)
        ref = case["ref"].get(f"{ka:.3f}", {})
        got = case["got"].get(f"{ka:.3f}", {})
        omega = math.sqrt(ka * GRAVITY)
        rho_vol = DENSITY * VOLUME
        rho_omega_vol = DENSITY * omega * VOLUME
        for dof, suffix in (("heave", "33"), ("surge", "11")):
            ind = block[dof]
            ind_a = ind["added_mass"] / rho_vol
            ind_b = ind["damping"] / rho_omega_vol
            ref_a = ref.get(f"A{suffix}_dimless")
            ref_b = ref.get(f"B{suffix}_dimless")
            got_a = got.get(f"A{suffix}_dimless")
            got_b = got.get(f"B{suffix}_dimless")
            for name, ref_v, got_v, ind_v in (
                ("added_mass", ref_a, got_a, ind_a),
                ("damping", ref_b, got_b, ind_b),
            ):
                rows.append(
                    {
                        "ka": ka,
                        "dof": dof,
                        "quantity": name,
                        "ref_selfderived": ref_v,
                        "got_tensorbem": got_v,
                        "ind_capytaine": ind_v,
                        "err_got_vs_ref": None
                        if (ref_v in (None, 0) or got_v is None)
                        else (got_v - ref_v) / ref_v,
                        "err_got_vs_ind": None
                        if (ref_v is None or got_v is None)
                        else (got_v - ind_v) / ind_v,
                        "err_ref_vs_ind": None
                        if (ref_v is None or ref_v == 0)
                        else (ref_v - ind_v) / ind_v,
                        "ind_convergence": _convergence(independent, ka, dof, name, ind_v),
                    }
                )
    return rows


def _convergence(independent: dict, ka: float, dof: str, quantity: str, finest: float) -> dict:
    key = f"{ka:.3f}"
    prev = None
    ladder = []
    for level in independent["levels"]:
        val = level["values"][key][dof][quantity]
        ladder.append({"n_faces": level["n_faces"], "value": val})
    if len(ladder) >= 2:
        prev = ladder[-2]["value"]
    return {
        "ladder": ladder,
        "finest_vs_previous_rel": None if not prev else (finest - prev) / prev,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-dir",
        default="/root/tp-agent/work/m17181/runs/casepool_bem_hulme_hemisphere",
    )
    parser.add_argument("--out", default="artifacts/independent_hemisphere_check.json")
    args = parser.parse_args(argv)

    case = _load_case(Path(args.case_dir))
    print(f"[case] ref ka keys: {sorted(case['ref'])}", flush=True)
    print(f"[case] got ka keys (band, subdivisions=3): {sorted(case['got'])}", flush=True)
    independent = _capytaine_sweep()
    rows = _compare(case, independent)
    payload = {
        "case": "CASE-BEM-HULME-HEMISPHERE / BEM-HULME-HEMISPHERE (case_pool id 138)",
        "audited_mission": 17181,
        "case_dir_read": str(args.case_dir),
        "comparison": rows,
        "independent": independent,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    for row in rows:
        def pct(value):
            return "     n/a" if value is None else f"{100 * value:+7.2f}%"

        print(
            f"ka={row['ka']:.1f} {row['dof']:5s} {row['quantity']:10s} "
            f"ref={pct(row['err_got_vs_ref'])} got_vs_ind={pct(row['err_got_vs_ind'])} "
            f"ref_vs_ind={pct(row['err_ref_vs_ind'])} "
            f"cap_tail={pct(row['ind_convergence']['finest_vs_previous_rel'])}"
        )
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())