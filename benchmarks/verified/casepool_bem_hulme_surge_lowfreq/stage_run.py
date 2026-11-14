#!/usr/bin/env python3
"""Staged local driver for CASE-BEM-HULME-SURGE-LOWFREQ.

The full package needs reference ladder + seven 3-D solves, which together exceed a
single short foreground command, so it is split into cacheable stages (the compute
gate owns anything longer).  Each stage writes into ``_stage_cache.json``:

    python stage_run.py reference     # R3 series ladder (cheap)
    python stage_run.py mesh          # R4 mesh ladder at ka = 1e-3
    python stage_run.py ka            # R4 ka ladder (reuses the ka=1e-3, sub=3 point)
    python stage_run.py assemble      # gates + results.json + report.md + figure.png

``assemble`` reuses :func:`limits.run_case` so there is exactly one code path for the
gates and the artifact rendering.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import limits  # noqa: E402

CACHE = HERE / "_stage_cache.json"


def _load() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def _save(data: dict) -> None:
    CACHE.write_text(json.dumps(data, indent=1) + "\n")


def stage_reference() -> None:
    data = _load()
    data["reference_ladder"] = limits.reference_ladder()
    data["extrapolation"] = limits.extrapolate_limit(data["reference_ladder"])
    _save(data)
    print(json.dumps({
        "stage": "reference",
        "rows": len(data["reference_ladder"]),
        "a0_dimless": data["extrapolation"]["a0_dimless"],
        "rel_error_vs_exact": data["extrapolation"]["rel_error_vs_exact"],
    }))


def stage_mesh() -> None:
    data = _load()
    rows = [_solve_with_cache(data, sub, limits.MESH_LADDER_KA) for sub in limits.MESH_LEVELS]
    data["solver_mesh_ladder"] = rows
    _save(data)
    print(json.dumps({
        "stage": "mesh",
        "rows": [(r["panels"], r["rel_error_vs_exact"]) for r in rows],
    }))


def stage_ka() -> None:
    data = _load()
    rows = []
    for ka in limits.KA_LADDER:
        rows.append(_solve_with_cache(data, limits.KA_LADDER_SUB, float(ka)))
    data["solver_ka_ladder"] = rows
    _save(data)
    print(json.dumps({
        "stage": "ka",
        "rows": [(r["ka"], r["rel_error_vs_exact"]) for r in rows],
    }))


def _solve_with_cache(data: dict, sub: int, ka: float) -> dict:
    key = f"solve:{int(sub)}:{float(ka):.6e}"
    done = data.setdefault("solves", {})
    if key in done:
        return done[key]
    row = limits._solve_surge(int(sub), float(ka))
    done[key] = row
    _save(data)
    return row


def stage_assemble() -> None:
    data = _load()
    for key in ("reference_ladder", "extrapolation", "solver_mesh_ladder", "solver_ka_ladder"):
        if key not in data:
            raise SystemExit(f"cache incomplete: {key} missing (run the earlier stages)")

    limits.reference_ladder = lambda *a, **k: data["reference_ladder"]
    limits.extrapolate_limit = lambda *a, **k: data["extrapolation"]
    limits.solver_ladders = lambda *a, **k: (data["solver_mesh_ladder"],
                                             data["solver_ka_ladder"])
    payload = limits.run_case(write=True)
    print(json.dumps({
        "stage": "assemble",
        "verdict": payload["verdict"],
        "failing": payload["failing"],
        "extrapolated": payload["extrapolation"]["a0_dimless"],
        "solver_finest": payload["solver_mesh_ladder"][-1]["added_mass_dimless"],
    }))


STAGES = {
    "reference": stage_reference,
    "mesh": stage_mesh,
    "ka": stage_ka,
    "assemble": stage_assemble,
}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "assemble"
    STAGES[name]()