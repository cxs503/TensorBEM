"""Run the declared-numbers test module on the merged tree and on the baseline.

The merge report quotes "passed / skipped / failed" per tree; those numbers come
from here, parsed out of pytest's own summary line, never typed by hand.

Usage (from the repository root)::

    PYTHONPATH=src python3 docs/m3686/run_tests.py            # run both trees
    PYTHONPATH=src python3 docs/m3686/run_tests.py --from-logs /tmp/merged.log /tmp/base.log
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TARGET = "tests/test_declared_numbers.py"
SUMMARY = re.compile(r"(\d+) (passed|failed|skipped|error)")


def _parse(log: str) -> dict:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    for value, kind in SUMMARY.findall(log):
        counts[kind] = counts.get(kind, 0) + int(value)
    seconds = 0.0
    hit = re.search(r"in ([\d.]+)s", log)
    if hit:
        seconds = float(hit.group(1))
    return counts, seconds


def _archive(ref: str) -> Path:
    target = Path(tempfile.mkdtemp(prefix="tensorbem_"))
    archive = subprocess.run(
        ["git", "archive", ref], cwd=ROOT, check=True, capture_output=True
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(target)], input=archive, check=True)
    return target


def _run(tree: Path, log_path: Path) -> Path:
    started = time.time()
    with log_path.open("wb") as handle:
        subprocess.run(
            [sys.executable, "-m", "pytest", TARGET, "-q"],
            cwd=tree,
            env={"PYTHONPATH": str(tree / "src"), "PATH": "/usr/bin:/bin"},
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    print("{} took {:.1f} s".format(tree, time.time() - started), file=sys.stderr)
    return log_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="b420b2e", help="baseline revision to compare against")
    parser.add_argument("--from-logs", nargs=2, metavar=("MERGED", "BASELINE"))
    args = parser.parse_args(argv)

    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    if args.from_logs:
        merged_log, baseline_log = (Path(p) for p in args.from_logs)
        merged_text = merged_log.read_text(encoding="utf-8")
        baseline_text = baseline_log.read_text(encoding="utf-8")
        shutil.copyfile(merged_log, logs / "pytest_merged.log")
        shutil.copyfile(baseline_log, logs / "pytest_baseline.log")
    else:
        baseline_tree = _archive(args.ref)
        _run(baseline_tree, logs / "pytest_baseline.log")
        _run(ROOT, logs / "pytest_merged.log")
        merged_text = (logs / "pytest_merged.log").read_text(encoding="utf-8")
        baseline_text = (logs / "pytest_baseline.log").read_text(encoding="utf-8")

    merged_counts, merged_seconds = _parse(merged_text)
    baseline_counts, baseline_seconds = _parse(baseline_text)
    evidence = {
        "target": TARGET,
        "baseline_ref": args.ref,
        "merged": dict(merged_counts, seconds=round(merged_seconds, 1)),
        "baseline": dict(baseline_counts, seconds=round(baseline_seconds, 1)),
    }
    assert merged_counts["failed"] == 0 and merged_counts["error"] == 0, evidence
    assert merged_counts["passed"] >= baseline_counts["passed"], evidence
    (HERE / "tests_evidence.json").write_text(json.dumps(evidence, indent=1), encoding="utf-8")
    print(json.dumps(evidence, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())