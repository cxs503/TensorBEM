"""Run the coupled Cummins kernel-matrix campaign and write report + figures.

Usage (from the repository root)::

    PYTHONPATH=src python docs/cummins_matrix_closure/run_campaign.py [--refresh]

``--refresh`` re-solves the boundary-element sweep instead of loading the cached
one; the cache (``radiation_sweep.json``) is committed so the campaign is
reproducible without re-running 56 BEM solves.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorbem.benchmarks import cummins_matrix_closure as campaign  # noqa: E402


def main(argv: list[str]) -> int:
    refresh = "--refresh" in argv
    report = campaign.run_campaign(refresh=refresh, keep_traces=True)
    report_path = campaign.write_report(report, HERE / "campaign.json")
    figures = campaign.render_figures(report)
    manifest = {
        "campaign": report["campaign"],
        "all_passed": report["all_passed"],
        "verdicts": [
            {"id": verdict["id"], "passed": verdict["passed"], "threshold": verdict["threshold"]}
            for verdict in report["verdicts"]
        ],
        "report": str(report_path),
        "figures": [{"name": item["name"], "caption": item["caption"], "path": item["path"]} for item in figures],
        "kernel_grid": report["kernel_grid"],
        "case": report["case"]["provenance"],
        "sweep_points": report["case"]["sweep_points"],
    }
    campaign.write_report(manifest, HERE / "manifest.json")
    print(json.dumps(manifest["verdicts"], indent=1))
    print("all_passed:", report["all_passed"])
    print("report:", report_path)
    for item in figures:
        print("figure:", item["path"])
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))