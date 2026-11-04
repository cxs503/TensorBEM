"""Targeted re-run of the four mutation survivors of ``mutation_audit_tensorbem.md``.

The full audit (``python -m tensorbem.benchmarks.mutation_audit``) samples eight
mutants per module; it is the right tool for *finding* blind spots, but a poor
tool for *closing* them, because a re-sweep does not guarantee it samples the
same mutants again.  This script re-runs an explicit, named set of mutants so the
before/after verdict is exact rather than statistical:

* **legacy source** (``git show HEAD:src/tensorbem/<mod>.py``) reproduces the
  audit's "survived" verdict on the pre-fix tree;
* **current source** with the pre-existing test files shows whether the source
  fix alone closes the blind spot;
* **current source + ``tests/test_mutation_gaps_hydrostatic.py``** shows what the
  backfilled guards add.

Each mutant is applied as a minimal textual edit at an exact AST node location
(``mutation_audit.enumerate_mutants`` / ``apply_mutant``), and every run happens
in a private copy of ``src/tensorbem`` so the worktree is never touched.

Usage::

    PYTHONPATH=src python3 docs/m2970/tools/targeted_mutation.py --out docs/m2970/outputs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorbem.benchmarks import mutation_audit as ma  # noqa: E402

NEW_TESTS = "tests/test_mutation_gaps_hydrostatic.py"

BASELINE_TESTS = {
    "tensorbem.hydrostatic": (
        "tests/test_hydrostatic_analytic_benchmark.py",
        "tests/test_waterplane_multi_loop.py",
        "tests/test_hydrostatic_convergence.py",
    ),
    "tensorbem.free_surface": (
        "tests/test_free_surface_kernel.py",
        "tests/test_ursell_benchmark.py",
    ),
}

#: (module, function, operator, source flavour, audit line, audit column, marker)
#: The line/column pair is the node the audit recorded; the marker identifies the
#: node's source line when the fix has moved it.
PROBES = (
    ("tensorbem.hydrostatic", "compute_waterplane_properties", "swap_binop", "current",
     236, 20, "x_moment +="),
    ("tensorbem.hydrostatic", "compute_waterplane_properties", "shift_constant", "current",
     255, 15, "np.zeros(2)"),
    ("tensorbem.hydrostatic", "compute_waterplane_properties", "shift_constant", "current",
     161, 15, "np.zeros(2)"),
    ("tensorbem.hydrostatic", "compute_displaced_volume_and_buoyancy", "shift_constant",
     "current", 314, 29, "np.zeros(3)"),
    ("tensorbem.free_surface", "green_gradient", "swap_binop", "legacy",
     243, 37, "df_d_depth ="),
    ("tensorbem.free_surface", "green_gradient", "swap_binop", "current",
     243, 37, "df_d_depth ="),
)


def _legacy_source(module: str) -> str:
    rel = "src/" + module.replace(".", "/") + ".py"
    proc = subprocess.run(
        ["git", "show", f"HEAD:{rel}"], cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git show HEAD:{rel} failed: {proc.stderr.strip()}")
    return proc.stdout


def _pick_mutant(
    source: str,
    path: Path,
    module: str,
    function: str,
    operator: str,
    lineno: int,
    col: int,
    marker: str,
) -> ma.Mutant:
    lines = source.splitlines()
    candidates = [
        m
        for m in ma.enumerate_mutants(module, source, path)
        if m.function.endswith(function) and m.operator == operator
    ]
    if not candidates:
        raise RuntimeError(f"no {operator} mutant of {module}:{function} in this source")
    exact = [
        m
        for m in candidates
        if m.lineno == lineno and m.col == col and marker in lines[m.lineno - 1]
    ]
    if len(exact) == 1:
        return exact[0]
    fuzzy = [m for m in candidates if marker in lines[m.lineno - 1]]
    if len(fuzzy) == 1:
        return fuzzy[0]
    listing = "; ".join(f"{m.lineno}:{m.col} {m.original!r}" for m in candidates)
    raise RuntimeError(
        f"cannot pin {module}:{function} {operator} to line {lineno} col {col} "
        f"({marker!r}): {listing}"
    )


def _run_case(
    module: str,
    mutant: ma.Mutant,
    source: str,
    flavour: str,
    extra_tests: tuple[str, ...],
    timeout: float,
) -> dict:
    staging = Path(tempfile.mkdtemp(prefix="m2970-"))
    try:
        shadow_src = ma._shadow_package(staging)
        ma._check_isolation(shadow_src)
        rel = Path(module.replace(".", "/") + ".py")
        shadow_file = shadow_src / rel
        shadow_file.write_text(ma.apply_mutant(source, mutant))
        tests = BASELINE_TESTS[module] + extra_tests
        started = time.time()
        returncode, output = ma.run_tests(shadow_src, tests, timeout)
        verdict, failing = ma._classify(returncode, output)
        return {
            "flavour": flavour,
            "tests": len(tests),
            "verdict": verdict,
            "failing_test": failing,
            "seconds": round(time.time() - started, 1),
        }
    finally:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "m2970" / "outputs")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args(argv)

    rows: list[dict] = []
    for module, function, operator, flavour, lineno, col, marker in PROBES:
        source = _legacy_source(module) if flavour == "legacy" else (
            REPO_ROOT / "src" / (module.replace(".", "/") + ".py")
        ).read_text()
        path = REPO_ROOT / "src" / (module.replace(".", "/") + ".py")
        mutant = _pick_mutant(
            source, path, module, function, operator, lineno, col, marker
        )
        cases = [
            _run_case(module, mutant, source, flavour, (), args.timeout),
            _run_case(module, mutant, source, flavour, (NEW_TESTS,), args.timeout),
        ]
        rows.append(
            {
                "mutant": mutant.key,
                "summary": mutant.summary,
                "module": module,
                "function": function,
                "operator": operator,
                "source": flavour,
                "lineno": mutant.lineno,
                "original": mutant.original,
                "replacement": mutant.replacement,
                "baseline": cases[0],
                "with_new_tests": cases[1],
            }
        )
        print(
            f"{mutant.key:48s} [{flavour:7s}] baseline={cases[0]['verdict']:9s}"
            f" +new={cases[1]['verdict']:9s}"
        )

    payload = {
        "generated_by": "docs/m2970/tools/targeted_mutation.py",
        "new_tests": NEW_TESTS,
        "baseline_tests": {k: list(v) for k, v in BASELINE_TESTS.items()},
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "targeted_mutation.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out / 'targeted_mutation.json'}")
    return payload


if __name__ == "__main__":
    main()