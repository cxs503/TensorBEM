#!/usr/bin/env python3
"""Re-derive the union resolution from the three parents and prove it reproduces
the committed blob.

Inputs are the three stages git itself used:
    :1 = merge base  6025970:src/tensorbem/benchmarks/declared_numbers.py
    :2 = ours        main (origin/main)
    :3 = theirs      agent/m3236

`git merge-file --diff3` regenerates exactly the conflict markers the merge
produced; the resolver in resolve_union.py is then applied to that text and the
result is compared byte-for-byte with the file that was committed.  The same run
records the hunk accounting used in the report, so no number in the report is
hand-copied.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resolve_union import resolve  # noqa: E402

PATH = "src/tensorbem/benchmarks/declared_numbers.py"
OURS = "origin/main"
THEIRS = "agent/m3236"


def blob(rev: str) -> str:
    return subprocess.run(
        ["git", "show", f"{rev}:{PATH}"], capture_output=True, text=True, check=True
    ).stdout


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        base_f, ours_f, theirs_f = (tmpdir / n for n in ("base.py", "ours.py", "theirs.py"))
        base_f.write_text(blob("6025970"))
        ours_f.write_text(blob(OURS))
        theirs_f.write_text(blob(THEIRS))
        merged = subprocess.run(
            ["git", "merge-file", "-p", str(ours_f), str(base_f), str(theirs_f)],
            capture_output=True,
            text=True,
        )
    conflicted = merged.stdout
    if "<<<<<<<" not in conflicted:
        raise SystemExit("merge-file did not reproduce a conflict; assumptions broken")
    resolved, hunks = resolve(conflicted)
    committed = Path(PATH).read_text()

    def sha(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    report = {
        "path": PATH,
        "parents": {"base": "6025970", "ours": OURS, "theirs": THEIRS},
        "merge_file_conflict_hunks": len(hunks),
        "hunks": hunks,
        "sha256_resolved_from_parents": sha(resolved),
        "sha256_committed_blob": sha(committed),
        "identical": resolved == committed,
        "markers_remaining": committed.count("<<<<<<<") + committed.count(">>>>>>>"),
    }
    out = Path("docs/m3796/resolution_check.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("merge_file_conflict_hunks", "identical", "markers_remaining")}, indent=2))
    for h in hunks:
        print("  hunk", h)
    return 0 if report["identical"] and report["markers_remaining"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())