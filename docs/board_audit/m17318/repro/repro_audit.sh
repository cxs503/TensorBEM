#!/usr/bin/env bash
# Auditor-side reproduction of the two numbers this audit quotes for
# CASE-BEM-SPHERE-ADDED-MASS (m17262). Reruns the *subject's* module
# (snapshot: ../references/sphere_added_mass_m17262.py, whose sha256 is checked
# below) through the production entry and compares the resulting relative
# errors with the auditor artifacts it claims to reproduce.
#
# Usage: bash docs/board_audit/m17318/repro/repro_audit.sh
# Exit 0 = reproduced; exit 3 = mismatch; exit 4 = subject src tree missing.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
AUDIT="$(cd "$HERE/.." && pwd)"
REF="$AUDIT/references"
ART="$AUDIT/artifacts"
SUBJ="${SUBJECT_SRC:-/root/tp-agent/work/m17262}"

# 1) snapshot integrity: the module we vouch for is the one we rerun
python3 - "$REF/sphere_added_mass_m17262.py" "$SUBJ/src/tensorbem/benchmarks/sphere_added_mass.py" <<'PY'
import hashlib, sys, pathlib
a, b = (pathlib.Path(p) for p in sys.argv[1:3])
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if b.exists() and h(a) != h(b):
    sys.exit("snapshot differs from subject file -- subject tree changed after audit")
print("snapshot sha256 ok:", h(a)[:16])
PY

# 2) the subject module needs the repo sources on PYTHONPATH
if [ ! -d "$SUBJ/src/tensorbem" ]; then
  echo "SKIP: subject src tree $SUBJ/src/tensorbem not present (worktree recycled?)" >&2
  exit 4
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
for s in 1 2; do
  PYTHONPATH="$SUBJ/src" python3 -m tensorbem.benchmarks.sphere_added_mass \
      --subdivisions "$s" --output "$TMP/sub$s.json" >/dev/null
done

# 3) compare against the recorded auditor artifacts (exact rel_err match)
python3 - "$TMP" "$ART" <<'PY'
import json, sys, pathlib
tmp, art = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
worst = 0.0
for s in (1, 2):
    got = json.loads((tmp / f"sub{s}.json").read_text())
    want = json.loads((art / f"auditor_rerun_sub{s}.json").read_text())
    assert got["panels"] == want["panels"], (got["panels"], want["panels"])
    dev = abs(got["added_mass_relative_error"] - want["added_mass_relative_error"])
    worst = max(worst, dev)
    print(f"sub{s} panels={got['panels']:>5} rel_err={got['added_mass_relative_error']:.12e} dev={dev:.1e}")
sys.exit(0 if worst < 1e-12 else f"MISMATCH: deviation {worst:.3e}")
PY
echo "PASS: sub1/sub2 reproduced to <1e-12"