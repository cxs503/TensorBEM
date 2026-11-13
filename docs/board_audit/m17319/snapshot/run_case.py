"""Case-pool driver for CASE-BEM-HULME-HEMISPHERE (tensor_case_pool.id = 138).

The physics, the predeclared gates and the report/figure writers live in the
library module :mod:`tensorbem.benchmarks.hulme_hemisphere_case`; this driver
only adds the two things a case-pool run needs on top of it:

* **shard orchestration** -- ``--shards N --parallel P`` fans the deterministic
  task list of the library module out over ``P`` worker processes (each worker
  is ``python -m tensorbem.benchmarks.hulme_hemisphere_case --shard i --shards N``)
  and then merges the shard files, so a wall-clock-bound rerun does not have to
  be serial while producing byte-identical ``results.json``;
* **exit-code contract** -- the driver exits non-zero when the merged verdict is
  FAIL, so a case-pool scheduler can treat the case as a signal.

One-shot reproduction of the committed artefacts::

    python benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py \\
        --out benchmarks/verified/casepool_bem_hulme_hemisphere --parallel 8

The same numbers can be produced without this file by
``python -m tensorbem.benchmarks.hulme_hemisphere_case --out <dir>``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from tensorbem.benchmarks import hulme_hemisphere_case as case

REPO = Path(__file__).resolve().parents[2]


def _worker(shard: int, shards: int, out_dir: Path, log_dir: Path) -> subprocess.Popen:
    """Start one shard as a child process of the library module CLI."""
    log_path = log_dir / f"worker_{shard}.log"
    handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "tensorbem.benchmarks.hulme_hemisphere_case",
         "--shard", str(shard), "--shards", str(shards), "--out", str(out_dir)],
        stdout=handle,
        stderr=subprocess.STDOUT,
        cwd=REPO,
    )
    proc._case_log = handle  # type: ignore[attr-defined]  # closed on wait
    return proc


def run_sharded(out_dir: Path, shards: int, parallel: int) -> int:
    """Run ``shards`` shards, at most ``parallel`` at a time, then merge."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    pending = list(range(shards))
    running: list[subprocess.Popen] = []
    failures: list[int] = []
    while pending or running:
        while pending and len(running) < max(1, parallel):
            running.append(_worker(pending.pop(0), shards, out_dir, log_dir))
        for proc in list(running):
            if proc.poll() is None:
                continue
            running.remove(proc)
            proc._case_log.close()  # type: ignore[attr-defined]
            if proc.returncode != 0:
                failures.append(proc.returncode)
    if failures:
        print(f"FAIL: {len(failures)} shard worker(s) exited non-zero", file=sys.stderr)
        return 2
    return case.main(["--merge", "--out", str(out_dir)])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="benchmarks/verified/casepool_bem_hulme_hemisphere")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=1,
                        help="worker processes started at the same time")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--spec", action="store_true",
                        help="print the deterministic task list and exit")
    args = parser.parse_args(argv)

    out_dir = Path(args.out).resolve()
    if args.spec:
        print(json.dumps({"tasks": case.build_tasks(),
                          "band_ka": list(case.BAND_KA),
                          "case_subdivisions": case.CASE_SUBDIVISIONS,
                          "gate": case.GATE,
                          "geometry_gate": case.GEOM_GATE}, indent=2))
        return 0
    if args.merge_only:
        return case.main(["--merge", "--out", str(out_dir)])
    return run_sharded(out_dir, max(1, args.shards), max(1, args.parallel))


if __name__ == "__main__":
    raise SystemExit(main())