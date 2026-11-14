#!/usr/bin/env python3
"""Extend the capytaine ladder beyond the frozen 1600-panel rung, on a time budget.

The control host forbids long runs, and the cluster nodes do not have capytaine
installed (job cj2583 died with ModuleNotFoundError), so the extension runs here
in one bounded pass: the four published rungs are carried over verbatim from
``capytaine_excitation.json`` (marked ``reused-frozen`` in ``levels_provenance``)
and only the new rungs are computed, newest-last, stopping before the wall-clock
budget so that a partial extension still produces a valid artifact.

Usage: python scripts/capytaine_hulme_ladder_extend_partial.py --budget 95
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("_ext", HERE / "capytaine_hulme_ladder_extended.py")
EXT = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = EXT; SPEC.loader.exec_module(EXT)
BASE = EXT._BASE

CASE_DIR = HERE.parent / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere"
FROZEN = CASE_DIR / "capytaine_excitation.json"
OUT = CASE_DIR / "capytaine_ladder_extended.json"
NEW_LEVELS = ((48, 96), (56, 112), (64, 128))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=float, default=95.0)
    args = ap.parse_args(argv)
    import capytaine as cpt
    frozen = json.loads(FROZEN.read_text())
    levels = [{**lvl, "levels_provenance": "reused-frozen"} for lvl in frozen["levels"]]
    solver = cpt.BEMSolver()
    started = time.time()
    for resolution in NEW_LEVELS:
        if time.time() - started > args.budget:
            print(f"budget spent, stopping before resolution {resolution}", flush=True)
            break
        t0 = time.time()
        mesh = cpt.mesh_sphere(radius=BASE.RADIUS, center=(0.0, 0.0, 0.0),
                               resolution=resolution).immersed_part()
        level = {"resolution": list(resolution), **BASE._sweep(cpt, solver, mesh),
                 "levels_provenance": "computed-this-run"}
        level["wall_seconds"] = time.time() - t0
        levels.append(level)
        print(f"level {level['n_faces']:5d} panels t={level['wall_seconds']:.1f}s", flush=True)
    levels.sort(key=lambda lvl: lvl["n_faces"])
    payload = {
        "metadata": BASE._provenance(cpt),
        "case_code": "BEM-HULME-HEMISPHERE",
        "purpose": ("extended refinement ladder of the anchor's independent code; rungs up to 1600 "
                    "panels are the frozen published values, later rungs computed this run"),
        "geometry": frozen["geometry"],
        "nabla_exact": frozen["nabla_exact"],
        "levels": levels,
        "convergence": BASE._convergence(levels),
        "levels_provenance": {str(lvl["n_faces"]): lvl["levels_provenance"] for lvl in levels},
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(OUT), "panels": [l["n_faces"] for l in levels],
                      "elapsed_s": round(time.time() - started, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
