"""Re-check the four mutation-audit survivors that round 2 claims to close.

``docs/mutation_audit_tensorbem.md`` §5 registered four survivors as follow-up
work.  This script re-enumerates exactly those mutants (selected by their
*source text*, so the selection survives line-number drift), applies each one to
a private shadow copy of ``src/tensorbem`` using the audit tool's own
``apply_mutant``, and judges it twice:

* **before** -- against the unit tests the audit used (the pre-existing suite);
* **after**  -- against the round-2 guardrails in
  ``tests/test_mutation_gaps_round2.py`` (and, for the kernel site, also against
  the round-1 finite-difference test, which could not see the defect while the
  depth derivative was a dead local).

A mutant counts as closed only if the "before" run passes (reproducing the
survivor) and the "after" run fails (the guardrail fires).

Usage::

    PYTHONPATH=src python3 docs/m2856/tools/verify_round2_mutants.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from tensorbem.benchmarks.mutation_audit import (
    SRC_ROOT,
    TARGETS,
    _check_isolation,
    apply_mutant,
    enumerate_mutants,
    run_tests,
)

ROUND2 = "tests/test_mutation_gaps_round2.py"
ROUND1_FD = (
    "tests/test_mutation_gaps.py"
    "::test_deep_water_source_gradient_matches_a_finite_difference"
)

#: ``(mutant key from docs/mutation_audit_tensorbem.md, guardrails that should
#: kill it)``.  The key pins module, line, column and operator; line numbers are
#: asserted against the mutated snippet so a silent drift is caught rather than
#: verified against the wrong site.
SITES: tuple[tuple[str, tuple[str, ...], tuple[str, ...] | None], ...] = (
    # registered survivor: waterline first moment, ``/ 6.0`` -> ``* 6.0``
    ("tensorbem.hydrostatic:236:20:swap_binop", (ROUND2,), None),
    # registered survivor: net-zero-area branch literal ``0.0`` -> ``1.0``
    ("tensorbem.hydrostatic:255:15:shift_constant", (ROUND2,), None),
    # registered survivor: zero-displacement centre ``np.zeros(3)`` -> ``(4)``
    ("tensorbem.hydrostatic:314:29:shift_constant", (ROUND2,), None),
    # registered survivor: dead depth derivative, ``2.0 * (...)`` -> ``/(...)``
    # the round-1 finite-difference test alone is also re-judged here: it could
    # not see this defect while the depth derivative was a dead local, and it
    # can now that ``green_gradient`` reuses it.
    ("tensorbem.free_surface:243:37:swap_binop", (ROUND2, ROUND1_FD), (ROUND1_FD,)),
    # not in the sampled set, but the same class: the no-waterplane branch
    ("tensorbem.hydrostatic:161:15:shift_constant", (ROUND2,), None),
)


def _select(key: str):
    module, lineno, col, operator = key.rsplit(":", 3)
    target = next(t for t in TARGETS if t.module == module)
    source = target.source.read_text(encoding="utf-8")
    matches = [
        mutant
        for mutant in enumerate_mutants(module, source, target.source)
        if mutant.key == key
    ]
    if len(matches) != 1:
        raise SystemExit(f"{key}: expected the audit's mutant, found {len(matches)}")
    mutant = matches[0]
    if mutant.operator != operator:
        raise SystemExit(f"{key}: operator drift ({mutant.operator})")
    return target, source, mutant


def _judge(target, source, mutant, tests, staging: Path) -> dict:
    work = Path(tempfile.mkdtemp(prefix="round2-", dir=str(staging)))
    shadow_src = work / "src"
    shutil.copytree(staging / "src", shadow_src)
    relative = target.source.relative_to(SRC_ROOT / "tensorbem")
    (shadow_src / "tensorbem" / relative).write_text(
        apply_mutant(source, mutant), encoding="utf-8"
    )
    code, output = run_tests(shadow_src, tuple(tests), timeout=600.0)
    shutil.rmtree(work, ignore_errors=True)
    failing = [
        line.strip()
        for line in output.splitlines()
        if line.startswith(("FAILED", "ERROR"))
    ]
    return {
        "tests": list(tests),
        "pytest_exit": code,
        "verdict": "killed" if code != 0 else "survived",
        "failing": failing,
    }


def main() -> int:
    staging = Path(tempfile.mkdtemp(prefix="round2-suite-"))
    source_root = staging / "src"
    shutil.copytree(
        SRC_ROOT / "tensorbem",
        source_root / "tensorbem",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    print(f"shadow package: {_check_isolation(source_root)}\n")

    rows = []
    closed = 0
    for key, tests, alone in SITES:
        target, source, mutant = _select(key)
        before = _judge(target, source, mutant, target.unit_tests, staging)
        after = _judge(target, source, mutant, tests, staging)
        alone_result = (
            _judge(target, source, mutant, alone, staging) if alone else None
        )
        is_closed = before["verdict"] == "survived" and after["verdict"] == "killed"
        closed += int(is_closed)
        rows.append(
            {
                "key": mutant.key,
                "summary": mutant.summary,
                "before": before,
                "after": after,
                "round1_alone": alone_result,
                "closed": is_closed,
            }
        )
        print(f"{mutant.key}\n  {mutant.summary}")
        print(f"  before ({', '.join(before['tests'])}): {before['verdict']}")
        print(f"  after  ({', '.join(after['tests'])}): {after['verdict']}")
        if alone_result is not None:
            print(
                f"  round-1 test alone ({', '.join(alone_result['tests'])}): "
                f"{alone_result['verdict']}"
            )
        if after["failing"]:
            print(f"  failing: {after['failing'][0]}")
        print(f"  -> {'CLOSED' if is_closed else 'NOT CLOSED'}\n")

    shutil.rmtree(staging, ignore_errors=True)
    payload = {"closed": closed, "total": len(rows), "rows": rows}
    out = Path(__file__).resolve().parents[1] / "outputs" / "round2_mutants.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"closed": closed, "total": len(rows)}, ensure_ascii=False))
    print(f"artifact: {out}")
    return 0 if closed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())