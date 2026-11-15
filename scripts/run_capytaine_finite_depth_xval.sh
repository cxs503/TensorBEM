#!/usr/bin/env bash
# Reproduce the *whole* finite-depth cross-validation grid in one command.
#
# The plan (:mod:`tensorbem.benchmarks.finite_depth_crossval_plan`) is 12 cells
# (h/a x k0*a) with a three-rung panel ladder on each side.
#
# Two stages, split by what they need:
#   * the capytaine reference needs `capytaine` (built from source on loongarch64,
#     installed on the control host but *not* on the cluster nodes) -- it is the
#     only part that imports it.  The stage is skipped when the frozen reference
#     already covers every planned cell, so a re-run resumes instead of
#     recomputing, and a node without capytaine can still run the ladder.
#   * the TensorBEM ladder and the consumer need nothing but this repo.  The
#     ladder is sharded per cell (index % shards == shard) and merged with the
#     producer's own completeness-checked merger; the consumer then writes
#     results.json + docs/capytaine_finite_depth_crossval.md + figures.
#
# Usage (inside a repo checkout):
#   bash scripts/run_capytaine_finite_depth_xval.sh            # 6 shards
#   SHARDS=4 bash scripts/run_capytaine_finite_depth_xval.sh
#   EXTRA_LEVEL=1 bash scripts/run_capytaine_finite_depth_xval.sh   # +4 cells at 1280 panels
#
# Env:
#   SHARDS       ladder shards (default 6; both sides are embarrassingly parallel
#                over cells, so this is roughly the core count used)
#   EXTRA_LEVEL  1 = also compute the 1280-panel confirmation rung on the four
#                EXTRA_LEVEL_CELLS (hours), 0 = skip it and let the study say so
#                (default 0)
#   WORK         scratch directory for shard files (default /tmp/xval)
set -euo pipefail

SHARDS="${SHARDS:-6}"
EXTRA_LEVEL="${EXTRA_LEVEL:-0}"
WORK="${WORK:-/tmp/xval}"
ROOT="benchmarks/verified/capytaine_finite_depth"
PY="${PYTHON:-python}"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

extra_flag=--no-extra-level
if [ "$EXTRA_LEVEL" = "1" ]; then
  extra_flag=--extra-level=3
fi
mkdir -p "$WORK" "$ROOT"

read -r PLANNED COVERED <<<"$("$PY" - <<'PY'
import json
from pathlib import Path
from tensorbem.benchmarks import finite_depth_crossval_plan as plan
planned = {plan.cell_id(c["h_over_a"], c["ka"]) for c in plan.cells()}
path = Path("benchmarks/verified/capytaine_finite_depth/capytaine_ref.json")
covered = set()
if path.exists():
    covered = {c["cell"] for c in json.loads(path.read_text()).get("cells", [])}
print(len(planned), len(planned & covered))
PY
)"

if [ "$COVERED" -ge "$PLANNED" ]; then
  echo "=== [1/4] capytaine reference: frozen file already covers ${COVERED}/${PLANNED} cells, skipped ==="
else
  if ! "$PY" -c "import capytaine" >/dev/null 2>&1; then
    cat >&2 <<'EOF'
capytaine is not importable on this host: the reference stage cannot run here.
It needs a host with capytaine installed (`pip install capytaine`, or from source
on loongarch64); the TensorBEM ladder and the consumer do not.  Produce the
reference there:

    PYTHONPATH=src python scripts/capytaine_finite_depth_ref.py \
        --out benchmarks/verified/capytaine_finite_depth/capytaine_ref.json

and re-run this driver (it skips the stage once the file covers the grid).
EOF
    exit 3
  fi
  echo "=== [1/4] capytaine reference: ${SHARDS} shards ==============================="
  rm -f "$WORK"/cp_ref_shard*.json
  for i in $(seq 0 $((SHARDS - 1))); do
    "$PY" scripts/capytaine_finite_depth_ref.py \
      --shard "$i" --shards "$SHARDS" --out "$WORK/cp_ref_shard${i}.json" &
  done
  wait

  echo "=== [2/4] merge the reference shards ========================================="
  merge_args=()
  for i in $(seq 0 $((SHARDS - 1))); do
    merge_args+=("$WORK/cp_ref_shard${i}.json")
  done
  "$PY" scripts/capytaine_finite_depth_ref.py \
    --merge "${merge_args[@]}" --out "$ROOT/capytaine_ref.json"
fi

echo "=== [3/4] TensorBEM ladder: ${SHARDS} shards (+ merge) ======================="
rm -f "$WORK"/tb_ladder_shard*.json
for i in $(seq 0 $((SHARDS - 1))); do
  "$PY" -m tensorbem.benchmarks.capytaine_finite_depth_crossval \
    --ladder-only --shard "$i" --shards "$SHARDS" \
    --levels 0,1,2 "$extra_flag" \
    --cache "$WORK/tb_ladder_shard${i}.json" &
done
wait

cache_args=()
for i in $(seq 0 $((SHARDS - 1))); do
  cache_args+=("$WORK/tb_ladder_shard${i}.json")
done
"$PY" -m tensorbem.benchmarks.capytaine_finite_depth_crossval \
  --merge-caches "${cache_args[@]}" --levels 0,1,2 \
  --cache "$ROOT/tensor_ladder.json"

echo "=== [4/4] consumer: comparison + gates + page + figures ====================="
"$PY" -m tensorbem.benchmarks.capytaine_finite_depth_crossval \
  --levels 0,1,2 "$extra_flag" --jobs 1 --write-doc

echo "=== done: $ROOT/results.json ================================================"