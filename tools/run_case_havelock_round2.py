"""Run CASE-BEM-SUBMERGED-SPHERE-HAVELOCK round 2 and write the case artifact.

Primary surface measure = exact spherical patch areas (round 2); the round-1
flat-triangle sweep is kept as the control in the same artifact.

    PYTHONPATH=src python3 tools/run_case_havelock_round2.py \
        --output runs/m21069/case_havelock_round2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorbem.benchmarks.case_havelock_submerged_sphere import run_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = run_case(quick=args.quick)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=float))
    prod = result["judgement"]["production_mesh"]
    ctrl = result["judgement"]["control_production_mesh"]
    print(json.dumps({
        "output": str(args.output),
        "primary_area_measure": result["surface_measure"]["primary"],
        "production_mesh_max_error": prod["max_rel_error"],
        "production_mesh_passed": prod["passed"],
        "production_mesh_n_failures": prod["n_failures"],
        "control_area_measure": result["surface_measure"]["control"],
        "control_max_error": None if ctrl is None else ctrl["max_rel_error"],
        "control_passed": None if ctrl is None else ctrl["passed"],
        "all_items_passed": result["judgement"]["all_items"]["passed"],
        "all_items_n_failures": result["judgement"]["all_items"]["n_failures"],
    }, indent=2))


if __name__ == "__main__":
    main()