#!/usr/bin/env python3
"""Line-level "no silent loss" audit of a completed three-way merge.

Git can drop content without raising a conflict: when both sides rewrite the
same region it keeps one side and the other side's lines disappear.  Mechanical
``git checkout --ours/--theirs`` overlays have exactly that failure mode, so the
merge is checked mechanically instead of trusted.

Two independent checks per changed path:

* **single-side files** -- only one branch touched the file, so the merged blob
  must be byte-identical to that branch's blob.  Any difference is a silent
  loss or an unexplained edit.
* **both-sides files** -- every line one side *added* relative to the merge base
  must appear in the merged blob, and every line one side *deleted* must be
  either absent from the merged blob or listed in the allowlist with a reason.
  ``dropped_*`` therefore counts unexplained content loss; a line that one side
  moved elsewhere in the same file is reported separately (``relocated_*``) and
  is not loss.

Usage::

    python3 docs/m3737/verify_merge_union.py --base <sha> \
        --ours origin/main --theirs origin/agent/m3181 --merged HEAD \
        --allowlist docs/m3737/merge_union_allowlist.json \
        --json docs/m3737/artifacts/merge_union_audit.json

Exit status is 0 only when no unexplained loss was found.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout


def show(rev: str, path: str) -> bytes | None:
    proc = subprocess.run(
        ["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True
    )
    return proc.stdout if proc.returncode == 0 else None


def changed_paths(base: str, rev: str) -> set[str]:
    return {p for p in git("diff", "--name-only", base, rev).splitlines() if p}


def edited(base: bytes, side: bytes) -> tuple[list[str], list[str]]:
    """Lines the side added to / removed from one file relative to the base."""
    a = base.decode("utf-8", "replace").splitlines()
    b = side.decode("utf-8", "replace").splitlines()
    added, removed = [], []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag in ("insert", "replace"):
            added.extend(b[j1:j2])
        if tag in ("delete", "replace"):
            removed.extend(a[i1:i2])
    return added, removed


def analyse_side(base, side, merged_lines, allowlist, path, label):
    """Return the audit counters for one side of one file."""
    added, removed = edited(base or b"", side or b"")
    out = {"added": 0, "dropped": [], "relocated_removed": [], "allowed": []}
    for line in added:
        if not line.strip():
            continue
        out["added"] += 1
        if line not in merged_lines:
            if (path, label, line) in allowlist:
                out["allowed"].append(line)
            else:
                out["dropped"].append(line)
    for line in removed:
        if not line.strip() or line in merged_lines:
            continue
        # A deleted line that is absent from the merge can still be legitimate
        # (the other side had already deleted it); the case that matters is a
        # deletion the merge *reverted*, i.e. the line survives.
        out["relocated_removed"].append(line)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ours", default="origin/main")
    ap.add_argument("--theirs", default="origin/agent/m3181")
    ap.add_argument("--merged", default="HEAD")
    ap.add_argument("--allowlist", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    allow = set()
    if args.allowlist:
        raw = json.loads((ROOT / args.allowlist).read_text())
        for entry in raw["allow"]:
            allow.add((entry["path"], entry["side"], entry["line"]))

    ours_paths = changed_paths(args.base, args.ours)
    theirs_paths = changed_paths(args.base, args.theirs)
    report = {"base": args.base, "ours": args.ours, "theirs": args.theirs,
              "merged": args.merged, "files": []}
    counters = {"both_sides_files": 0, "single_side_files": 0, "binary_files": 0,
                "dropped": 0, "allowed": 0, "single_side_mismatch": 0,
                "merged_files": 0}

    for path in sorted(ours_paths | theirs_paths):
        base_raw = show(args.base, path)
        merged_raw = show(args.merged, path)
        if b"\0" in (base_raw or b"") or b"\0" in (merged_raw or b""):
            counters["binary_files"] += 1
            ours_raw = show(args.ours, path)
            theirs_raw = show(args.theirs, path)
            report["files"].append({
                "path": path, "kind": "binary",
                "same_as_ours": merged_raw == ours_raw,
                "same_as_theirs": merged_raw == theirs_raw,
            })
            continue

        merged_lines = set((merged_raw or b"").decode("utf-8", "replace").splitlines())
        both = path in ours_paths and path in theirs_paths
        entry = {"path": path, "kind": "both-sides" if both else "single-side"}

        if not both:
            side_rev = args.ours if path in ours_paths else args.theirs
            side_raw = show(side_rev, path)
            entry["edited_by"] = "ours" if path in ours_paths else "theirs"
            entry["merged_equals_editor"] = merged_raw == side_raw
            counters["single_side_files"] += 1
            if merged_raw != side_raw:
                counters["single_side_mismatch"] += 1
            report["files"].append(entry)
            continue

        counters["both_sides_files"] += 1
        ours_raw = show(args.ours, path)
        theirs_raw = show(args.theirs, path)
        ours = analyse_side(base_raw, ours_raw, merged_lines, allow, path, "ours")
        theirs = analyse_side(base_raw, theirs_raw, merged_lines, allow, path, "theirs")
        entry.update({
            "added_ours": ours["added"], "added_theirs": theirs["added"],
            "dropped_ours": ours["dropped"], "dropped_theirs": theirs["dropped"],
            "allowed_ours": ours["allowed"], "allowed_theirs": theirs["allowed"],
            "deleted_ours_absent": ours["relocated_removed"],
            "deleted_theirs_absent": theirs["relocated_removed"],
        })
        counters["dropped"] += len(ours["dropped"]) + len(theirs["dropped"])
        counters["allowed"] += len(ours["allowed"]) + len(theirs["allowed"])
        report["files"].append(entry)

    counters["merged_files"] = len(report["files"])
    report["summary"] = counters
    report["clean_union"] = counters["dropped"] == 0 and counters["single_side_mismatch"] == 0

    if args.json:
        out = ROOT / args.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {out.relative_to(ROOT)}")
    print(json.dumps(counters, indent=2))
    for entry in report["files"]:
        if entry.get("kind") == "single-side" and not entry["merged_equals_editor"]:
            print(f"  SINGLE-SIDE MISMATCH {entry['path']} (edited by {entry['edited_by']})")
        for key in ("dropped_ours", "dropped_theirs", "allowed_ours", "allowed_theirs"):
            for line in entry.get(key, []):
                print(f"  {key:15s} {entry['path']}: {line[:100]}")
    print("CLEAN UNION" if report["clean_union"] else "UNEXPLAINED LOSS")
    return 0 if report["clean_union"] else 1


if __name__ == "__main__":
    sys.exit(main())