#!/usr/bin/env bash
# Resumable reproduction of the CI evidence pipeline + full test suite.
# Each step is guarded by a sentinel in <state-dir> so the script can be
# re-invoked across tool-call time limits and continues where it stopped.
# Usage: run_ci_equivalent.sh <tree-dir> <log-file> <state-dir>
set -u
TREE="$1"; LOG="$2"; STATE="$3"
mkdir -p "$STATE"
cd "$TREE" || exit 2
export PYTHONPATH="$TREE/src"
export MPLBACKEND=Agg
touch "$LOG"
echo "### tree=$TREE" >>"$LOG"

step() {  # step <name> <cmd...>
  local name="$1"; shift
  if [ -f "$STATE/$name.done" ]; then return 0; fi
  echo "### $name: $*" >>"$LOG"
  "$@" >>"$LOG" 2>&1
  local rc=$?
  echo "### $name exit=$rc" >>"$LOG"
  if [ "$rc" -eq 0 ]; then : > "$STATE/$name.done"; else echo "### FAILED $name" >>"$LOG"; return "$rc"; fi
}

step unit_disk       python3 -m tensorbem.benchmarks.unit_disk --panels 64
step cyl_added       python3 -m tensorbem.benchmarks.cylinder_added_mass --panels 32
step refinement      python3 -m tensorbem.benchmarks.refinement_study --panels 16 32 64 128 --output artifacts/refinement-study.json
step analytical      python3 -m tensorbem.benchmarks.analytical_suite --panels 16 32 64 128 --output artifacts/analytical-study.json
step sphere_added    python3 -m tensorbem.benchmarks.sphere_added_mass --subdivisions 2
step sphere_velocity python3 -m tensorbem.benchmarks.sphere_velocity --subdivisions 2 --output artifacts/sphere-velocity.json
step surface_velocity python3 -m tensorbem.benchmarks.surface_velocity_study --levels 1 2 3 --output artifacts/surface-velocity-study.json
step three_dim       python3 -m tensorbem.benchmarks.three_dimensional_study --levels 0 1 2 3 --output artifacts/three-dimensional-study.json
step ellipse_x       python3 -m tensorbem.benchmarks.ellipse_added_mass --panels 64 --axis x
step ellipse_y       python3 -m tensorbem.benchmarks.ellipse_added_mass --panels 64 --axis y
step ellipsoid       python3 -m tensorbem.benchmarks.ellipsoid_added_mass --subdivisions 2
step ursell          python3 -m tensorbem.benchmarks.ursell_cylinder --panels 64 --ka 0.6 --output artifacts/ursell.json
step doc_claims      python3 -m tensorbem.doc_claims --write
step rigid_body      python3 -m tensorbem.benchmarks.rigid_body_added_mass --panels 16 32 64 128 --subdivisions 0 1 2 --output artifacts/rigid-body-added-mass.json
step dean            python3 -m tensorbem.benchmarks.dean_diffraction --panels 64 --ka 0.6 --output artifacts/dean-diffraction.json
step maccamy         python3 -m tensorbem.benchmarks.maccamy_fuchs --panels 16 32 64 128 --output artifacts/maccamy-fuchs.json
step hemisphere      python3 -m tensorbem.benchmarks.floating_hemisphere --subdivisions 1 2 3 --output artifacts/floating-hemisphere.json
step submerged_sphere python3 -m tensorbem.benchmarks.submerged_sphere_wave --subdivisions 0 1 2 --output artifacts/submerged-sphere-wave.json
step classical       python3 -m tensorbem.benchmarks.classical_study --panels 16 32 64 128 --subdivisions 0 1 2 3 --harmonic-levels 1 2 3 --output artifacts/classical-study.json

if [ -f "$STATE/pytest.done" ]; then echo "### all steps + pytest already done" >>"$LOG"; exit 0; fi
echo "### pytest" >>"$LOG"
python3 -m pytest -q -p no:cacheprovider >>"$LOG" 2>&1
rc=$?
echo "### pytest_exit=$rc" >>"$LOG"
: > "$STATE/pytest.done"
exit 0