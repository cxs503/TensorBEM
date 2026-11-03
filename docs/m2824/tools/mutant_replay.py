"""Replay the four coverage-gap mutants from ``docs/mutation_audit_tensorbem.md``.

The mutation audit samples mutants from a seeded AST walk, so a re-run is not
guaranteed to regenerate the *same* four survivors that motivated the
backfilled tests.  This script replays them literally: each historical mutant is
written into a private copy of ``src/tensorbem`` (the worktree is never touched),
and the mapped unit test plus the backfilled file are run against it.

A mutant that makes the suite fail is *killed*; the run is only evidence if the
unmutated baseline is green in the same harness.

Usage::

    PYTHONPATH=src python docs/m2824/tools/mutant_replay.py --out docs/m2824/outputs
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
BACKFILL = "tests/test_mutation_gap_closure.py"

#: The four survivors classified as genuine coverage gaps (audit §5).
MUTANTS: tuple[dict, str] = (
    {
        "id": "hydrostatic:236 x_moment /6.0 -> *6.0",
        "module": "hydrostatic",
        "original": "x_moment += sign * np.sum((x + x_next) * cross) / 6.0",
        "replacement": "x_moment += sign * np.sum((x + x_next) * cross) * 6.0",
        "unit_test": "tests/test_hydrostatic_analytic_benchmark.py",
    },
    {
        "id": "hydrostatic empty-waterline guard 0.0 -> 1.0",
        "module": "hydrostatic",
        "original": (
            "        # No waterplane (fully submerged or no intersection)\n"
            "        return 0.0, np.zeros(2), np.zeros(3)"
        ),
        "replacement": (
            "        # No waterplane (fully submerged or no intersection)\n"
            "        return 1.0, np.zeros(2), np.zeros(3)"
        ),
        "unit_test": "tests/test_hydrostatic_analytic_benchmark.py",
    },
    {
        "id": "hydrostatic zero-total-area guard 0.0 -> 1.0",
        "module": "hydrostatic",
        "original": (
            "    if total_area < 1.0e-14:\n"
            "        return 0.0, np.zeros(2), np.zeros(3)"
        ),
        "replacement": (
            "    if total_area < 1.0e-14:\n"
            "        return 1.0, np.zeros(2), np.zeros(3)"
        ),
        "unit_test": "tests/test_hydrostatic_analytic_benchmark.py",
    },
    {
        "id": "hydrostatic:314 zero-volume np.zeros(3) -> np.zeros(4)",
        "module": "hydrostatic",
        "original": "    if abs(total_volume) < 1e-14:\n        return 0.0, np.zeros(3)",
        "replacement": "    if abs(total_volume) < 1e-14:\n        return 0.0, np.zeros(4)",
        "unit_test": "tests/test_hydrostatic_analytic_benchmark.py",
    },
    {
        "id": "free_surface:243 depth term 2.0 * (...) -> 2.0 / (...)",
        "module": "free_surface",
        "original": "    df_d_depth = -depth / r_image2 + 2.0 * (elementary_cos + frequency * cos_integral)",
        "replacement": "    df_d_depth = -depth / r_image2 + 2.0 / (elementary_cos + frequency * cos_integral)",
        "unit_test": "tests/test_free_surface_kernel.py",
    },
)


def _shadow(tmp: Path) -> Path:
    """Copy the package into ``tmp`` so the worktree is never modified."""
    shadow = tmp / "src"
    shutil.copytree(SRC_ROOT / "tensorbem", shadow / "tensorbem")
    return shadow


def _run_tests(shadow: Path, tests: list[str], timeout: float) -> tuple[bool, str]:
    env = {
        "PYTHONPATH": str(shadow),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/root"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", *tests],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    return proc.returncode == 0, "\n".join(tail[-4:])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "m2824" / "outputs")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="tensorbem-gap-replay-") as tmp_name:
        tmp = Path(tmp_name)
        shadow = _shadow(tmp)

        # Baseline: unmutated shadow through the same harness.
        baseline_ok, baseline_tail = _run_tests(
            shadow, [BACKFILL], args.timeout
        )
        print(f"baseline (backfill only): {'green' if baseline_ok else 'RED'}")

        for mutant in MUTANTS:
            target = shadow / "tensorbem" / f"{mutant['module']}.py"
            pristine = target.read_text(encoding="utf-8")
            count = pristine.count(mutant["original"])
            if count != 1:
                rows.append(
                    {
                        **{k: mutant[k] for k in ("id", "module", "unit_test")},
                        "applicable": False,
                        "verdict": f"ERROR: pattern matched {count} times",
                    }
                )
                continue
            target.write_text(
                pristine.replace(mutant["original"], mutant["replacement"]),
                encoding="utf-8",
            )
            try:
                pre_ok, pre_tail = _run_tests(shadow, [mutant["unit_test"]], args.timeout)
                post_ok, post_tail = _run_tests(
                    shadow, [mutant["unit_test"], BACKFILL], args.timeout
                )
            finally:
                target.write_text(pristine, encoding="utf-8")
            rows.append(
                {
                    **{k: mutant[k] for k in ("id", "module", "unit_test")},
                    "applicable": True,
                    "pre_suite_green": pre_ok,
                    "post_suite_green": post_ok,
                    "verdict": (
                        "killed by the backfilled test"
                        if pre_ok and not post_ok
                        else "killed by the pre-existing suite"
                        if not pre_ok
                        else "SURVIVES"
                    ),
                    "backfill_tail": post_tail,
                    "pre_tail": pre_tail,
                }
            )
            print(f"{mutant['id']}: {rows[-1]['verdict']}")

    payload = {
        "generated_by": "docs/m2824/tools/mutant_replay.py",
        "worktree": str(REPO_ROOT),
        "backfill": BACKFILL,
        "baseline_green": baseline_ok,
        "baseline_tail": baseline_tail,
        "mutants": rows,
        "killed": sum(1 for r in rows if r.get("verdict", "").startswith("killed")),
        "total": len(rows),
    }
    (args.out / "mutant_replay.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{payload['killed']}/{payload['total']} killed -> {args.out / 'mutant_replay.json'}")
    return 0 if payload["killed"] == len(rows) and baseline_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())