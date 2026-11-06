#!/usr/bin/env bash
# Remaining CI steps, run in parallel (host is heavily loaded; serialising them
# costs more than the contention does). Temporary helper, deleted before commit.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
mkdir -p artifacts .scratch/logs

run_one() {
  local tag="$1"; shift
  if "$@" >".scratch/logs/$tag.log" 2>&1; then echo "OK    $tag"; else echo "FAIL  $tag"; fi
}

run_one surface_velocity python -m tensorbem.benchmarks.surface_velocity_study --levels 1 2 3 --output artifacts/surface-velocity-study.json &
run_one three_d python -m tensorbem.benchmarks.three_dimensional_study --levels 0 1 2 3 --output artifacts/three-dimensional-study.json &
run_one ellipse_x python -m tensorbem.benchmarks.ellipse_added_mass --panels 64 --axis x &
run_one ellipse_y python -m tensorbem.benchmarks.ellipse_added_mass --panels 64 --axis y &
run_one ellipsoid python -m tensorbem.benchmarks.ellipsoid_added_mass --subdivisions 2 &
run_one ursell python -m tensorbem.benchmarks.ursell_cylinder --panels 64 --ka 0.6 --output artifacts/ursell.json &
run_one docclaims python -m tensorbem.doc_claims --write &
run_one rigidbody python -m tensorbem.benchmarks.rigid_body_added_mass --panels 16 32 64 128 --subdivisions 0 1 2 --output artifacts/rigid-body-added-mass.json &
run_one dean python -m tensorbem.benchmarks.dean_diffraction --panels 64 --ka 0.6 --output artifacts/dean-diffraction.json &
run_one maccamy python -m tensorbem.benchmarks.maccamy_fuchs --panels 16 32 64 128 --output artifacts/maccamy-fuchs.json &
run_one hemisphere python -m tensorbem.benchmarks.floating_hemisphere --subdivisions 1 2 3 --output artifacts/floating-hemisphere.json &
run_one spherewave python -m tensorbem.benchmarks.submerged_sphere_wave --subdivisions 0 1 2 --output artifacts/submerged-sphere-wave.json &
run_one classical python -m tensorbem.benchmarks.classical_study --panels 16 32 64 128 --subdivisions 0 1 2 3 --harmonic-levels 1 2 3 --output artifacts/classical-study.json &
wait

echo "=== evidence done, starting pytest ==="
python -m pytest -q -p no:cacheprovider > .scratch/logs/pytest_full.log 2>&1
echo "PYTEST_EXIT=$?" >> .scratch/logs/pytest_full.log
echo DONE