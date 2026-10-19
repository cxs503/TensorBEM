"""Run a reproducible refinement study for the currently supported 2-D cases.

This study deliberately covers only the unit-disk manufactured solution and
the translating-cylinder boundary-trace/added-mass check. It does not claim a
general exterior-domain formulation or 3-D ship hydrodynamics.

Example:
    python -m tensorbem.benchmarks.refinement_study --panels 16 32 64 128 --output artifacts/refinement.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from tensorbem.benchmarks.cylinder_added_mass import run as run_cylinder
from tensorbem.benchmarks.unit_disk import run as run_unit_disk


def run_study(panel_counts: list[int]) -> dict[str, Any]:
    """Run both benchmark families and return JSON-serializable evidence."""
    if not panel_counts or any(n < 3 for n in panel_counts):
        raise ValueError("panel_counts must contain integers >= 3")
    if len(set(panel_counts)) != len(panel_counts):
        raise ValueError("panel_counts must be unique")
    panel_counts = sorted(panel_counts)
    unit_disk = [run_unit_disk(n) for n in panel_counts]
    cylinder = [run_cylinder(n) for n in panel_counts]

    errors = [float(row["added_mass_relative_error"]) for row in cylinder]
    trends = {
        "cylinder_added_mass_error_decreases": all(
            later < earlier for earlier, later in zip(errors, errors[1:])
        ),
        "all_outputs_finite": all(
            math.isfinite(float(value))
            for row in unit_disk + cylinder
            for value in row.values()
            if isinstance(value, (float, int))
        ),
    }
    return {
        "schema_version": 1,
        "study": "tensorbem_2d_refinement",
        "panel_counts": panel_counts,
        "unit_disk_manufactured_solution": unit_disk,
        "translating_cylinder_added_mass": cylinder,
        "checks": trends,
        "scope": (
            "Constant-panel, midpoint-collocation 2-D Laplace prototype. "
            "These cases are narrow verification tests and do not validate "
            "general exterior flow, free-surface conditions, or 3-D ship flow."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--output", type=Path, help="also write machine-readable JSON")
    args = parser.parse_args()
    result = run_study(args.panels)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
