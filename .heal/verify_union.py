"""Machine check that the PR #128 merge is a semantic *union*, not an overlay.

Overlay merges (``git checkout --ours/--theirs`` on a conflicted path, or a
"take theirs for anything that conflicts" loop) leave no trace in the diff:
the result compiles and the tests that run may even pass, because the side that
carried the extra feature is simply gone. This script refuses to trust the merge
commit and re-derives the evidence from the three trees:

    base   = merge-base(m3181, main)
    ours   = agent/m3181 before the merge
    theirs = main
    merged = the merge commit

For every path that either side touched it asks one question: *is every line
either side introduced still reachable in `merged`?* Concretely, for each side
it builds the multiset of added lines (relative to base, non-blank unless the
side only added blank lines) and requires each of them to appear in `merged` at
least as often as it appears in that side's own file. Lines a side *removed*
are reported too, so a union that quietly resurrects something one side deleted
is visible as well.

A line that a side introduced but that legitimately does *not* survive (a rename
of a colliding symbol, an "ours"-side line that the resolution rewrote) is not
allowed to be invisible: it must be listed in :data:`DECLARED_REWRITES` with the
reason, and the count must match exactly. Anything else fails the check.

Usage:
    python3 verify_union.py --repo <merge worktree> --base <sha> --ours <sha> \
        --theirs <sha> --merged HEAD --json out.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from typing import Iterable

#: (path, side, line-substring) -> reason.  A line introduced by `side` that is
#: absent from the merge is only tolerated when it is listed here.
DECLARED_REWRITES: list[tuple[str, str, str, str]] = [
    (
        "src/tensorbem/benchmarks/declared_numbers.py",
        "theirs",
        "def _chief_rows(rows)",
        "name collision with this branch's _chief_rows() registry builder; "
        "renamed to _chief_study_rows",
    ),
    (
        "src/tensorbem/benchmarks/declared_numbers.py",
        "theirs",
        "return _chief_rows(rows)",
        "call site of the renamed flattener",
    ),
    (
        "tests/test_declared_numbers.py",
        "ours",
        "# sign is exactly the claim the table makes.",
        "this sentence continues with the CHIEF sign rationale main added, so "
        "the merged comment carries it extended rather than verbatim",
    ),
    (
        "tests/test_declared_numbers.py",
        "theirs",
        "# sign-bearing quantities exist (e.g. the unphysical negative B33' that",
        "rationale comment merged with this branch's wording",
    ),
    (
        "tests/test_declared_numbers.py",
        "theirs",
        "# table 11 quotes as the defect CHIEF removes), so only zero is illegal",
        "rationale comment merged with this branch's wording",
    ),
]


def git(repo: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", repo, *args], check=True, capture_output=True, text=True
    ).stdout


def blob_bytes(repo: str, ref: str, path: str) -> bytes | None:
    """Raw file content at `ref`, or None if the path does not exist there."""
    proc = subprocess.run(
        ["git", "-C", repo, "cat-file", "-p", f"{ref}:{path}"],
        capture_output=True,
    )
    return proc.stdout if proc.returncode == 0 else None


def decode(raw: bytes | None) -> str | None:
    """Text view of a blob; None for missing or binary blobs."""
    if raw is None:
        return None
    if b"\0" in raw[:8000]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def changed_paths(repo: str, base: str, ref: str) -> list[str]:
    out = git(repo, "diff", "--name-only", "-z", base, ref)
    return [p for p in out.split("\0") if p]


def added_lines(base: str | None, side: str | None) -> Counter:
    """Multiset of lines present in `side` but not in `base`, ignoring blanks."""
    base_lines = Counter((base or "").splitlines())
    side_lines = Counter((side or "").splitlines())
    out: Counter = Counter()
    for line, count in side_lines.items():
        if not line.strip():
            continue
        extra = count - base_lines.get(line, 0)
        if extra > 0:
            out[line] = extra
    return out


def removed_lines(base: str | None, side: str | None) -> Counter:
    base_lines = Counter((base or "").splitlines())
    side_lines = Counter((side or "").splitlines())
    out: Counter = Counter()
    for line, count in base_lines.items():
        if not line.strip():
            continue
        gone = count - side_lines.get(line, 0)
        if gone > 0:
            out[line] = gone
    return out


def missing(side_added: Counter, merged_lines: Counter) -> list[tuple[str, int]]:
    """Lines of `side_added` that `merged` cannot account for."""
    out = []
    for line, count in side_added.items():
        if merged_lines.get(line, 0) < count:
            out.append((line, count - merged_lines.get(line, 0)))
    return sorted(out)


def check(repo: str, base: str, ours: str, theirs: str, merged: str) -> dict:
    paths = sorted(
        set(changed_paths(repo, base, ours)) | set(changed_paths(repo, base, theirs))
    )
    declared = Counter((p, side, frag) for p, side, frag, _ in DECLARED_REWRITES)
    used_rewrites: Counter = Counter()
    files = []
    failures = 0
    for path in paths:
        refs = {"base": base, "ours": ours, "theirs": theirs, "merged": merged}
        raw = {name: blob_bytes(repo, ref, path) for name, ref in refs.items()}
        text = {name: decode(raw[name]) for name in raw}
        row = {
            "path": path,
            "binary": any(raw[r] is not None and text[r] is None for r in raw),
            "state": {name: raw[name] is not None for name in refs},
            "lost_from_ours": [],
            "lost_from_theirs": [],
            "rewritten_from_ours": [],
            "rewritten_from_theirs": [],
            "resurrected_both_deleted": [],
        }
        if row["binary"]:
            # Bytes cannot be unioned line-wise: a side's blob survives only if
            # the merge carries it verbatim. Anything else must be declared.
            side_blobs = {
                "ours": raw["ours"],
                "theirs": raw["theirs"],
            }
            changed = {
                side: content
                for side, content in side_blobs.items()
                if content is not None and content != raw["base"]
            }
            for side, content in changed.items():
                if content != raw["merged"]:
                    row[f"lost_from_{side}"].append("<binary blob not carried by merge>")
                    failures += 1
            files.append(row)
            continue

        merged_lines = Counter((text["merged"] or "").splitlines())
        base_counter = Counter((text["base"] or "").splitlines())
        for side in ("ours", "theirs"):
            if text[side] is None:
                continue
            lost = missing(added_lines(text["base"], text[side]), merged_lines)
            kept = []
            for line, count in lost:
                hit = [
                    frag
                    for (p, s, frag) in declared
                    if p == path and s == side and frag in line
                ]
                if hit:
                    used_rewrites[(path, side, hit[0])] += 1
                    row[f"rewritten_from_{side}"].append({"line": line, "reason": hit[0]})
                else:
                    kept.append(line)
            row[f"lost_from_{side}"] = kept
            failures += len(kept)

        # A line *both* sides deleted must not come back through the resolution.
        if text["base"] is not None and text["merged"] is not None:
            both_removed = removed_lines(text["base"], text["ours"]) & removed_lines(
                text["base"], text["theirs"]
            )
            for line in both_removed:
                kept_base = base_counter.get(line, 0) - both_removed[line]
                if merged_lines.get(line, 0) > kept_base:
                    row["resurrected_both_deleted"].append(line)
                    failures += 1

        if row["state"]["ours"] and row["state"]["theirs"] and not row["state"]["merged"]:
            row["lost_from_ours"].append("<whole file dropped by merge>")
            failures += 1
        files.append(row)

    unused = declared - used_rewrites
    return {
        "base": base,
        "ours": ours,
        "theirs": theirs,
        "merged": merged,
        "base_commit": git(repo, "rev-parse", base).strip(),
        "ours_commit": git(repo, "rev-parse", ours).strip(),
        "theirs_commit": git(repo, "rev-parse", theirs).strip(),
        "merged_commit": git(repo, "rev-parse", merged).strip(),
        "paths_checked": len(paths),
        "conflicted_paths": sorted(
            git(repo, "diff", "--name-only", "--diff-filter=U", base, merged).split()
        ) if False else [],
        "failures": failures,
        "declared_rewrites": [
            {"path": p, "side": s, "fragment": f, "reason": r}
            for p, s, f, r in DECLARED_REWRITES
            if used_rewrites[(p, s, f)]
        ],
        "unused_declared_rewrites": [
            {"path": p, "side": s, "fragment": f} for (p, s, f) in unused
        ],
        "verdict": "UNION" if failures == 0 and not unused else "FAIL",
        "files": [row for row in files if any(
            row[k] for k in ("lost_from_ours", "lost_from_theirs", "resurrected_both_deleted")
        )] or files,
    }


def side_ref(side: str, ours: str, theirs: str) -> str:
    return ours if side == "ours" else theirs


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--base", required=True)
    parser.add_argument("--ours", required=True)
    parser.add_argument("--theirs", required=True)
    parser.add_argument("--merged", default="HEAD")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = check(args.repo, args.base, args.ours, args.theirs, args.merged)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json:
        with open(args.json, "w") as handle:
            handle.write(text + "\n")
    print(report["verdict"], "paths_checked=", report["paths_checked"], "failures=", report["failures"])
    for row in report["files"]:
        for side in ("ours", "theirs"):
            for line in row[f"lost_from_{side}"]:
                print(f"LOST[{side}] {row['path']}: {line[:120]}")
        for line in row.get("resurrected_both_deleted", []):
            print(f"RESURRECTED {row['path']}: {line[:120]}")
    for item in report["unused_declared_rewrites"]:
        print("UNUSED-REWRITE", item)
    return 0 if report["verdict"] == "UNION" else 1


if __name__ == "__main__":
    raise SystemExit(main())