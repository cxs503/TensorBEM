"""Recompute the 3-D free-surface tables of docs/marine_bem_benchmarks.md.

Runs only the new registry groups (the rest of the heavy tier is unchanged) and
writes one strict-JSON report plus a per-cell table, so that every number in the
documentation report is injected from the file rather than typed.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS, check

GROUPS = ("hulme3d", "hulme3d_chief", "wang_linton3d")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    entries = tuple(e for e in DECLARED_NUMBERS if e.group in GROUPS)
    started = time.time()
    report = check(tiers=("heavy",), entries=entries)
    report["elapsed_seconds"] = time.time() - started
    report["python"] = platform.python_version()
    report["note"] = (
        "heavy tier, 3-D deep-water Kelvin Green function; "
        "every row of a table shares one solve"
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    summary = report["summary"]
    print(
        f"checked {summary['checked']} passed {summary['passed']} "
        f"drifted {summary['drifted']} in {report['elapsed_seconds']:.1f}s"
    )
    for row in report["checks"]:
        flag = "OK " if row["verdict"] == "PASS" else "BAD"
        print(
            f"{flag} {row['key']:52s} declared {row['declared']:.6g} "
            f"measured {row['measured']:.6g} dev {row['relative_deviation']:+.2e}"
        )
    return 0 if summary["drifted"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())