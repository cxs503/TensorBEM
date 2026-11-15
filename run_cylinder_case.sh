#!/bin/bash
# Regenerate CASE-BEM-LAPLACE-CYLINDER full benchmark record with figures
set -euo pipefail
cd /root/tp-agent/work/m22173
export PYTHONPATH=src

OUTDIR="benchmarks/verified/case_bem_laplace_cylinder"
mkdir -p "$OUTDIR/figures"

echo "=== Running full case (ladder 32,64,128) ==="
PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream \
  --outdir "$OUTDIR" \
  --doc "docs/CASE-BEM-LAPLACE-CYLINDER.md" \
  --check

echo "=== Generating figures ==="
# The case module should have generated figures already via --check
# Let's verify
ls -la "$OUTDIR/figures/" 2>/dev/null || echo "No figures dir yet"
ls -la "$OUTDIR/" 

echo "=== Verifying results ==="
python3 -c "
import json
d = json.load(open('$OUTDIR/results.json'))
print('Total criteria:', len(d.get('criteria', [])))
print('Overall:', d.get('overall'))
print('Provenance generator:', d.get('provenance', {}).get('generator', 'NONE'))
print('Failed checks:', d.get('acceptance', {}).get('failed_checks', []))
"

echo "=== Running tests ==="
PYTHONPATH=src python3 -m pytest tests/test_case_cylinder_stream.py -v --tb=short -x 2>&1 | tail -40

echo "=== Done ==="
