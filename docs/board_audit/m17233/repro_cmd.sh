#!/usr/bin/env bash
# Audit m17233 / subject m17177 — CASE-BEM-ELLIPSOID-ADDED-MASS one-shot reproduction.
# Run from a TensorBEM checkout root: bash docs/board_audit/m17233/repro_cmd.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
OUT=docs/board_audit/m17233/rerun
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
for s in 0 1 2; do
  python -m tensorbem.benchmarks.ellipsoid_added_mass --subdivisions "$s" \
      --semi-axes 1.5 1.0 0.75 --density 1025.0 \
      --output "$OUT/ellipsoid_added_mass_sub${s}_rerun.json" > /dev/null
  echo "sub=$s written"
done
python docs/board_audit/m17233/independent_ref_check.py \
    --output "$OUT/independent_reference.json" > /dev/null
python - <<'PY'
import json
ref = json.load(open('docs/board_audit/m17233/rerun/independent_reference.json'))
got = json.load(open('docs/board_audit/m17233/rerun/ellipsoid_added_mass_sub2_rerun.json'))
bem = [row['added_mass'] for row in got['axes']]
print('ref (independent Lamb integral):', [round(x, 6) for x in ref['ref_added_mass']])
print('got (BEM, 320 panels):        ', [round(x, 6) for x in bem])
print('rel error %:                  ', [round(1e2 * abs(bem[i] - ref['ref_added_mass'][i]) / ref['ref_added_mass'][i], 4) for i in range(3)])
print('alpha sum residual:', ref['alpha_sum_identity_residual'])
print('PASS <5%:', ref['passes_5pct'], '| PASS <3%:', ref['passes_3pct'])
PY
