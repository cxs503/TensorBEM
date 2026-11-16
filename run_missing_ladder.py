#!/usr/bin/env python3
"""Compute the missing TensorBEM ladder levels for the finite-depth crossval.

Resumes from the existing tensor_ladder.json, computes only missing (cell, level)
pairs, and writes back the updated cache.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from tensorbem.benchmarks import finite_depth_crossval_plan as plan
from tensorbem.benchmarks.capytaine_finite_depth_crossval import build_ladder

CACHE = Path("benchmarks/verified/capytaine_finite_depth/tensor_ladder.json")

def main():
    # Load existing cache to resume
    levels = [0, 1, 2]
    jobs = 8  # parallel processes

    print(f"Building ladder with levels={levels}, jobs={jobs}")
    print(f"Resuming from {CACHE}")

    result = build_ladder(
        cells=plan.cells(),
        levels=levels,
        cache_path=CACHE,
        jobs=jobs,
    )

    # Summary
    cells_data = result.get("cells", {})
    complete = sum(
        1 for entry in cells_data.values()
        if all(str(l) in entry.get("levels", {}) for l in levels)
    )
    print(f"\nDone: {complete}/{len(cells_data)} cells complete with all 3 levels")

if __name__ == "__main__":
    main()
