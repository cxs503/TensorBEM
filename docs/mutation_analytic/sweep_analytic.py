"""Resumable driver for the ``tensorbem.analytic`` mutation sweep.

The batch runner in ``tensorbem.benchmarks.mutation_audit`` assumes it owns the
whole wall-clock window.  In this worktree the shell harness reaps the process
group after 120 s, so a 24-mutant sweep has to be chunked.  This driver reuses
the audit engine's own enumeration / shadow-tree / verdict functions unchanged
and simply persists one JSON row per finished mutant, so successive invocations
resume where the previous one stopped.

Usage::

    BUDGET=95 python3 docs/mutation_analytic/sweep_analytic.py \
        docs/mutation_analytic/before/results.json            # mapped tests only
    BUDGET=95 python3 docs/mutation_analytic/sweep_analytic.py \
        docs/mutation_analytic/after/results.json tests/test_mutation_gaps_analytic.py

``N``/``SEED``/``JOBS``/``BUDGET`` come from the environment and reproduce the
same mutant sample every time (sampling mirrors ``mutation_audit.sweep``).
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorbem.benchmarks import mutation_audit as ma  # noqa: E402

MODULE = "tensorbem.analytic"


def _target(extra: tuple[str, ...]) -> ma.Target:
    base = next(t for t in ma.TARGETS if t.module == MODULE)
    return ma.Target(base.module, base.source, base.unit_tests + extra)


def main(argv: list[str]) -> int:
    state_path = Path(argv[1])
    extra = tuple(argv[2:])
    n = int(os.environ.get("N", "24"))
    seed = int(os.environ.get("SEED", "20261104"))
    jobs = int(os.environ.get("JOBS", "6"))
    budget = float(os.environ.get("BUDGET", "95"))
    timeout = float(os.environ.get("MUT_TIMEOUT", "300"))

    target = _target(extra)
    source = target.source.read_text(encoding="utf-8")
    everything = ma.enumerate_mutants(target.module, source, target.source)
    if len(everything) > n:
        sample = sorted(
            random.Random(seed).sample(everything, n), key=lambda m: (m.lineno, m.col)
        )
    else:  # pragma: no cover - only for a tiny module
        sample = everything

    state_path.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict] = {}
    if state_path.exists():
        rows = {r["key"]: r for r in json.loads(state_path.read_text())["mutants"]}
    todo = [m for m in sample if m.key not in rows]

    started = time.time()
    if todo:
        staging = Path(tempfile.mkdtemp(prefix="tensorbem-mutation-"))
        try:
            ma._shadow_package(staging)  # noqa: SLF001 - reuse the audited isolation
            ma._check_isolation(staging / "src")  # noqa: SLF001
            # Sequential on purpose: the harness reaps the process group at 120 s,
            # and a thread pool cannot be abandoned mid-run (atexit joins it).
            for mutant in todo:
                if (time.time() - started) >= budget:
                    break
                row = ma._run_one(target, mutant, staging, timeout, True)
                rows[row["key"]] = row
                print(
                    f"{row['verdict']:<9} {row['key']} {row['summary'][:90]}",
                    flush=True,
                )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    ordered = [rows[m.key] for m in sample if m.key in rows]
    payload = {
        "module": MODULE,
        "seed": seed,
        "max_per_module": n,
        "extra_tests": list(extra),
        "unit_tests": list(target.unit_tests),
        "mutants": ordered,
        "score": {
            k: sum(1 for r in ordered if r["verdict"] == k)
            for k in ("killed", "survived", "invalid", "timeout")
        },
    }
    payload["score"]["selected"] = len(sample)
    payload["score"]["done"] = len(ordered)
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"done {len(ordered)}/{len(sample)} in {time.time() - started:.1f}s -> {state_path}"
    )
    return 0 if len(ordered) == len(sample) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))