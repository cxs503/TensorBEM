#!/usr/bin/env bash
# Drive the CI-equivalent pipeline for the baseline tree and the merged tree
# concurrently (both runners are resumable, so a killed wall-clock just means
# "invoke me again").
set -u
BASE_TREE="$1"; MERGED_TREE="$2"; DRIVER_DIR="$3"
bash "$DRIVER_DIR/run_ci_equivalent.sh" "$BASE_TREE" /tmp/m3796_base.log /tmp/m3796_base.state &
p1=$!
bash "$DRIVER_DIR/run_ci_equivalent.sh" "$MERGED_TREE" /tmp/m3796_merged.log /tmp/m3796_merged.state &
p2=$!
wait $p1; echo "baseline driver rc=$?"
wait $p2; echo "merged driver rc=$?"