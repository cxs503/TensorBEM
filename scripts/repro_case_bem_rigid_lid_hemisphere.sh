#!/usr/bin/env bash
# One-command reproduction of pool case CASE-BEM-RIGID-LID-HEMISPHERE (TensorBEM).
#
#   bash scripts/repro_case_bem_rigid_lid_hemisphere.sh [outdir]
#
# Steps
#   1. run the case driver (writes results.json / report.md / case.json / figures)
#      and fail on any violated predeclared gate (--check);
#   2. re-derive the verdict from the freshly written payload, independently of
#      the driver's exit code (belt and braces: a silently swallowed exception
#      cannot turn a FAIL into a PASS);
#   3. run the case tests, which re-solve one row and re-render the artifacts.
#
# Default outdir is the committed artifact directory; pass a scratch directory to
# reproduce without touching the work tree.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

outdir="${1:-benchmarks/verified/case_bem_rigid_lid_hemisphere}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

echo "== [1/3] case driver -> $outdir"
python -m tensorbem.benchmarks.case_bem_rigid_lid_hemisphere \
    --outdir "$outdir" --check

echo "== [2/3] verdict recomputed from the payload"
python - "$outdir" <<'PY'
import json
import pathlib
import sys

from tensorbem.benchmarks import case_bem_rigid_lid_hemisphere as case

payload = json.loads((pathlib.Path(sys.argv[1]) / "results.json").read_text())
recomputed = case.verdict(payload)
stored = payload["verdict"]
if recomputed != stored:
    raise SystemExit("stored verdict differs from the recomputed one: %r" % (stored,))
if not recomputed["passed"]:
    raise SystemExit("case FAILED: %r" % (recomputed["failed_checks"],))
print(
    "verdict PASS; reference A33/(rho*nabla) = %.7f (%.4f kg); %d checks"
    % (
        payload["reference"]["value_dimless"],
        payload["reference"]["value_kg"],
        len(recomputed["checks"]),
    )
)
PY

echo "== [3/3] case tests"
python -m pytest tests/test_case_bem_rigid_lid_hemisphere.py -q

echo "== done: $outdir/report.md"