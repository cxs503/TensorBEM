"""Verify the new 3-D free-surface registry groups and accumulate the JSON.

Usage::

    PYTHONPATH=src python3 outputs/m3638/verify_groups.py hulme submerged_sphere
    PYTHONPATH=src python3 outputs/m3638/verify_groups.py hulme_chief

Each invocation runs only the requested registry groups through the same
:func:`tensorbem.benchmarks.declared_numbers.check` path the CI gate uses and
merges the rows into ``outputs/m3638/group_checks.json``.  Splitting by group
keeps every process well inside the sandbox's 120 s wall clock.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS, check

OUT = Path("outputs/m3638/group_checks.json")


def main(argv: list[str]) -> int:
    groups = argv[1:]
    if not groups:
        raise SystemExit("usage: verify_groups.py GROUP [GROUP ...]")
    wanted = {g for g in groups}
    entries = tuple(e for e in DECLARED_NUMBERS if e.group in wanted)
    unknown = wanted - {e.group for e in entries}
    if unknown:
        raise SystemExit(f"no registry entries for {sorted(unknown)}")

    start = time.perf_counter()
    report = check(tiers=("standard",), entries=entries)
    elapsed = time.perf_counter() - start

    payload = {"groups": sorted(wanted), "elapsed_s": round(elapsed, 2),
               "summary": report["summary"], "checks": report["checks"]}
    if OUT.exists():
        acc = json.loads(OUT.read_text())
    else:
        acc = {"runs": []}
    acc["runs"] = [r for r in acc.get("runs", []) if r["groups"] != sorted(wanted)]
    acc["runs"].append(payload)
    rows = [row for run in acc["runs"] for row in run["checks"]]
    acc["all_checks"] = rows
    acc["summary"] = {
        "groups": sorted({g for run in acc["runs"] for g in run["groups"]}),
        "checked": len(rows),
        "passed": sum(1 for r in rows if r["verdict"] == "PASS"),
        "drifted": sum(1 for r in rows if r["verdict"] != "PASS"),
        "worst_relative_deviation": max(
            (abs(r["relative_deviation"]) for r in rows), default=0.0
        ),
        "total_elapsed_s": round(sum(r["elapsed_s"] for r in acc["runs"]), 2),
    }
    OUT.write_text(json.dumps(acc, indent=1, sort_keys=True) + "\n")

    print(f"groups={sorted(wanted)} elapsed={elapsed:.1f}s "
          f"checked={payload['summary']['checked']} "
          f"drifted={payload['summary']['drifted']}")
    for row in report["checks"]:
        if row["verdict"] != "PASS":
            print("  DRIFT", row["key"], row["declared"], row["measured"])
    return 0 if payload["summary"]["drifted"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))