#!/usr/bin/env python3
"""Re-execution check for the m17181 case driver (m17235 audit).

Runs nothing itself: it diffs the per-cell worst relative errors printed by a
fresh execution of

    PYTHONPATH=src python3 benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py \
        --out /tmp/repro_m17235 --shard 0 --shards 1

against the frozen ``runs/casepool_bem_hulme_hemisphere/results.json`` of
mission m17181, so "repro_cmd is executable and deterministic" is a measured
statement instead of a claim.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

CASE_DIR = Path("/root/tp-agent/work/m17181/runs/casepool_bem_hulme_hemisphere")
LOG = Path("/tmp/repro_m17235.log")
OUT = Path(__file__).resolve().parents[1] / "artifacts/repro_check.json"
PANELS_TO_SUBDIVISIONS = {64: 2, 256: 3, 1024: 4}


def main() -> int:
    results = json.loads((CASE_DIR / "results.json").read_text())
    frozen = {}
    for row in results["rows"]:
        if row.get("kind") != "band":
            continue
        worst = max(abs(row[k]) for k in row if k.startswith("rel_error_"))
        frozen[(round(float(row["ka"]), 3), int(row["subdivisions"]))] = worst

    cells = []
    for line in LOG.read_text().splitlines():
        m = re.search(r"band ka=([\d.]+) panels=(\d+) worst=([\d.]+)%", line)
        if not m:
            continue
        ka = round(float(m.group(1)), 3)
        panels = int(m.group(2))
        rerun = float(m.group(3)) / 100.0
        subdivisions = PANELS_TO_SUBDIVISIONS[panels]
        expected = frozen.get((ka, subdivisions))
        cells.append(
            {
                "ka": ka,
                "panels": panels,
                "subdivisions": subdivisions,
                "worst_rel_error_rerun": rerun,
                "worst_rel_error_frozen": expected,
                "abs_diff": None if expected is None else abs(rerun - expected),
            }
        )
    diffs = [c["abs_diff"] for c in cells if c["abs_diff"] is not None]
    payload = {
        "audited_mission": 17181,
        "repro_cmd": (
            "PYTHONPATH=src python3 benchmarks/verified/casepool_bem_hulme_hemisphere/"
            "run_case.py --out /tmp/repro_m17235 --shard 0 --shards 1"
        ),
        "rerun_log": str(LOG),
        "frozen_artifacts": str(CASE_DIR / "results.json"),
        "cells_compared": len(cells),
        "max_abs_diff": max(diffs) if diffs else None,
        "matches_within_log_precision": bool(diffs) and max(diffs) < 1e-6,
        "cells": cells,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    print(json.dumps({k: payload[k] for k in
                      ("cells_compared", "max_abs_diff", "matches_within_log_precision")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())