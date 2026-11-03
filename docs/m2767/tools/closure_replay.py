"""Re-apply the four mutants recorded as open items by the m2582 mutation audit.

`docs/mutation_audit_tensorbem.md` (m2582, 2026-11-03) closed 7 of 14 sampled
survivors and registered the remaining 4 as "real or suspected coverage gaps":

  1. ``hydrostatic.py``  ``x_moment += sign * np.sum((x + x_next) * cross) / 6.0``
     swapped to ``* 6.0`` -- survived because every existing waterplane case is
     centred on the origin, so the first moment is identically zero;
  2. ``hydrostatic.py``  the zero-area early return ``0.0`` -> ``1.0``;
  3. ``hydrostatic.py``  the zero-volume early return ``np.zeros(3)`` -> ``np.zeros(4)``;
  4. ``free_surface.py`` ``2.0 * (elementary_cos + frequency * cos_integral)`` in
     ``df_d_depth`` swapped to ``2.0 / ...`` -- suspected dead code or a missing
     finite-difference guard.

This script re-injects each of them into a private shadow package and runs the
suite twice: with the pre-existing tests only, and with the back-filled tests
added.  A mutant is *closed* when it survives the first run and is killed by the
second.  Numbers go to ``artifacts/hydrostatic_eccentric/closure.json``.

Run:
    PYTHONPATH=src python3 docs/m2767/tools/closure_replay.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

BASELINE_TESTS = [
    "tests/test_mutation_gaps.py",           # m2582 back-fill
]
NEW_TESTS = [
    "tests/test_hydrostatic_eccentric_waterplane.py",
    "tests/test_free_surface_source_depth_derivative.py",
]

# (label, module file, unit test file, literal old snippet, literal new snippet)
MUTANTS = [
    (
        "hydrostatic:236 first-moment /6.0 -> *6.0",
        "src/tensorbem/hydrostatic.py",
        "tests/test_hydrostatic_analytic_benchmark.py",
        "(x + x_next) * cross) / 6.0",
        "(x + x_next) * cross) * 6.0",
    ),
    (
        "hydrostatic:255 zero-area return 0.0 -> 1.0",
        "src/tensorbem/hydrostatic.py",
        "tests/test_hydrostatic_analytic_benchmark.py",
        "    if total_area < 1.0e-14:\n        return 0.0, np.zeros(2), np.zeros(3)",
        "    if total_area < 1.0e-14:\n        return 1.0, np.zeros(2), np.zeros(3)",
    ),
    (
        "hydrostatic:314 zero-volume return zeros(3) -> zeros(4)",
        "src/tensorbem/hydrostatic.py",
        "tests/test_hydrostatic_analytic_benchmark.py",
        "    if abs(total_volume) < 1e-14:\n        return 0.0, np.zeros(3)",
        "    if abs(total_volume) < 1e-14:\n        return 0.0, np.zeros(4)",
    ),
    (
        "free_surface:243 df_d_depth 2.0* -> 2.0/",
        "src/tensorbem/free_surface.py",
        "tests/test_free_surface_kernel.py",
        "df_d_depth = -depth / r_image2 + 2.0 * (",
        "df_d_depth = -depth / r_image2 + 2.0 / (",
    ),
]


def _shadow(staging: Path) -> Path:
    shadow = staging / "tensorbem"
    shutil.copytree(REPO / "src" / "tensorbem", shadow)
    return shadow


def _run(tests: list[str], shadow: Path) -> tuple[bool, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(shadow.parent)
    env["BLAS_NUM_THREADS"] = env["OMP_NUM_THREADS"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return proc.returncode == 0, tail


def main() -> None:
    rows = []
    staging = Path(tempfile.mkdtemp(prefix="closure-replay-"))
    try:
        clean = _shadow(staging / "clean")
        ok, tail = _run([*BASELINE_TESTS, *NEW_TESTS], clean)
        print(f"unmutated shadow: {'green' if ok else 'RED'} ({tail})")
        results = []
        for label, source_rel, unit_test, old, new in MUTANTS:
            case_dir = staging / label.split(":")[0].replace(" ", "_")
            source = (REPO / source_rel).read_text(encoding="utf-8")
            if old not in source:
                results.append({"mutant": label, "baseline": "snippet-missing", "after": None})
                print(f"{label}: SNIPPET MISSING")
                continue
            shadow = _shadow(case_dir)
            target = shadow / source_rel.split("/")[-1]
            target.write_text(source.replace(old, new, 1), encoding="utf-8")
            before_ok, before_tail = _run([*BASELINE_TESTS, unit_test], shadow)
            after_ok, after_tail = _run([*BASELINE_TESTS, unit_test, *NEW_TESTS], shadow)
            row = {
                "mutant": label,
                "baseline": "survived" if before_ok else "killed",
                "after": "survived" if after_ok else "killed",
                "baseline_tail": before_tail,
                "after_tail": after_tail,
            }
            results.append(row)
            print(f"{label}: baseline={row['baseline']:<9} after={row['after']:<9}")
            print(f"    baseline: {before_tail}")
            print(f"    after   : {after_tail}")
            shutil.rmtree(case_dir, ignore_errors=True)
        rows = results
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    out = REPO / "artifacts" / "hydrostatic_eccentric" / "closure.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mutants": rows}, indent=2), encoding="utf-8")
    closed = sum(1 for r in rows if r["baseline"] == "survived" and r["after"] == "killed")
    print(f"\n{closed}/{len(rows)} mutants closed (survived before, killed after) -> {out}")


if __name__ == "__main__":
    main()