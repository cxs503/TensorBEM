"""Re-verify the registry rows that PR #128 (agent/m3181) contributed.

The healing merge renamed one symbol and re-pointed two import aliases inside
the dispatch branch that PR #128 added.  That is exactly the kind of edit that
can silently break a *quantity key* (the row keeps a declared number but reads a
value the runner no longer emits).  This script therefore re-runs the three
groups the PR registers -- ``hulme3d``, ``hulme3d_chief``, ``wang_linton3d`` --
against the merged tree and reports one row per quoted cell.

Run:  PYTHONPATH=src python3 docs/m3754/verify_pr_registry.py
Writes docs/m3754/artifacts/pr128_registry_recheck.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS, check  # noqa: E402

GROUPS = ("hulme3d", "hulme3d_chief", "wang_linton3d")


def main() -> int:
    selected = tuple(entry for entry in DECLARED_NUMBERS if entry.group in GROUPS)
    start = time.time()
    report = check(tiers=("heavy",), entries=selected)
    report["elapsed_s"] = round(time.time() - start, 1)
    report["groups"] = list(GROUPS)
    out = ROOT / "docs" / "m3754" / "artifacts" / "pr128_registry_recheck.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    summary = report["summary"]
    print(json.dumps({"groups": list(GROUPS), **summary, "elapsed_s": report["elapsed_s"]}, indent=2))
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())