#!/usr/bin/env bash
# Per-file pytest driver for the #128 healing mission.
#
# Why per file: the full suite contains multi-minute benchmark studies, so a
# single pytest invocation is opaque while it runs and a single hang loses all
# evidence.  One pytest per test file gives a progressive, per-file verdict.
#
# The editable ``tensorbem`` install on this box points at *another* agent's
# worktree, therefore PYTHONPATH is pinned to <root>/src explicitly.
#
# usage: run_suite.sh <tree-root> <out-dir> [parallelism]
set -u
ROOT=${1:?tree root}
OUT=${2:?out dir}
PAR=${3:-10}
mkdir -p "$OUT"
cd "$ROOT" || exit 1
: > "$OUT/per_file.tsv"
export PYTHONPATH="$ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
ls tests/test_*.py | sed 's#^tests/##' | \
  xargs -P "$PAR" -I{} bash -c '
    f="{}"
    start=$(date +%s)
    out=$(python3 -m pytest -q -p no:cacheprovider -p no:randomly \
          "tests/$f" 2>&1)
    rc=$?
    dur=$(( $(date +%s) - start ))
    printf "%s\t%s\t%s\n" "$rc" "$dur" "$f" >> "'"$OUT"'/per_file.tsv"
    printf "%s\n" "$out" > "'"$OUT"'/log_${rc}_${f%.py}.txt"
  '
echo "DONE" > "$OUT/DONE"