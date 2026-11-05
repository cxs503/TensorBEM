#!/usr/bin/env python3
"""Audit a merge commit for silently dropped content.

A three-way merge is only a cure if the merged tree is a *union* of both sides.
A hand resolution (or a mechanical overlay) that drops one side's block looks
exactly like a clean merge from the outside -- the file parses and the tests
that survived are green -- so the property has to be checked against git, not
by reading the diff. This tool checks two layers:

1. **Tree and lines.** Every file added by either side exists in the merge and
   no file either side deleted comes back; every line either side inserted
   (relative to the merge base) is present in the merged file, verbatim or
   after a *declared* rename. A line that the merge legitimately rewrote (for
   example ``study = bench.run(...)`` becoming ``_shared_study(key, lambda: ...)``
   to share one solve between two readings of the same table) is discharged by a
   disposition that names a witness which must exist in the merged file. An
   undisposed line is a content loss and the verdict is ``DROP_DETECTED``.
2. **Registered numbers.** With ``--registry``, each side's registry is imported
   from its own tree and every ``DeclaredNumber`` the side registers must exist
   in the merged registry (key mapped through the declared group renames) with
   the same declared value. This is the invariant that matters for
   ``declared_numbers.py``: losing a *number* is the failure mode, not losing
   the exact source line that spelled it.

Usage::

    python tools/audit_union.py --merge cbcac1f --ours ecba798 --theirs 5b4dee7 \
        --dispositions tools/pr135_union_dispositions.json \
        --registry --json artifacts/pr135_union_audit.json

``verdict`` is ``PASS`` only when neither layer finds an undisposed loss.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    out = subprocess.run(
        ("git", *args),
        capture_output=True,
        text=True,
        errors="surrogateescape",
        check=False,
    )
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def changed_files(base: str, side: str) -> dict[str, str]:
    """``{path: status}`` for ``base..side`` (R100 is split by --no-renames)."""
    rows = {}
    for line in git("diff", "--name-status", "--no-renames", base, side).splitlines():
        status, path = line.split("\t", 1)
        rows[path] = status[0]
    return rows


def added_lines(base: str, side: str, path: str) -> list[str]:
    """Lines ``side`` inserted into ``path`` (no hunk headers, no context)."""
    diff = git("diff", "-U0", "--no-color", base, side, "--", path)
    return [
        line[1:].rstrip()
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def tree_files(rev: str) -> set[str]:
    return set(git("ls-tree", "-r", "--name-only", rev).splitlines())


def is_binary(rev: str, path: str) -> bool:
    """A binary blob has no lines to lose; it is audited at file level only."""
    blob = subprocess.run(
        ("git", "cat-file", "-p", f"{rev}:{path}"), capture_output=True, check=False
    )
    return b"\0" in blob.stdout[:8192]


def disposition_for(
    dispositions: list[dict], path: str, side: str, line: str
) -> dict | None:
    """The disposition that covers ``line``, if any (exact beats prefix)."""
    exact = [
        d
        for d in dispositions
        if d.get("path") == path
        and d.get("text") is not None
        and d.get("side", "any") in (side, "any")
        and d.get("match", "exact") == "exact"
        and d["text"] == line
    ]
    if exact:
        return exact[0]
    for d in dispositions:
        if (
            d.get("path") == path
            and d.get("text") is not None
            and d.get("side", "any") in (side, "any")
            and d.get("match") == "prefix"
            and line.startswith(d["text"])
        ):
            return d
    return None


def discharge(
    disposition: dict, line: str, merged_lines: set[str], merged_text: str
) -> tuple[bool, str]:
    """``(discharged, how)`` -- check the declared rename or witness."""
    if disposition["action"] in ("rename", "revalue"):
        renamed = line
        for old, new in disposition.get("rename", ()):
            renamed = renamed.replace(old, new)
        return (renamed in merged_lines, f"rename -> {renamed.strip()[:80]}")
    if disposition["action"] == "rewrite":
        missing = [w for w in disposition["witness"] if w not in merged_text]
        return (not missing, "witness" if not missing else f"witness missing: {missing}")
    raise SystemExit(f"unknown disposition action: {disposition['action']!r}")


def audit_lines(
    merge: str, ours: str, theirs: str, dispositions: list[dict]
) -> tuple[list[dict], list[dict]]:
    base = git("merge-base", ours, theirs).strip()
    sides = {"ours": ours, "theirs": theirs}
    files = {name: changed_files(base, rev) for name, rev in sides.items()}
    merged_files = tree_files(merge)
    merged_cache: dict[str, tuple[set[str], str]] = {}

    findings: list[dict] = []
    per_file: list[dict] = []

    # (1) file-level union
    for side, rows in files.items():
        for path, status in rows.items():
            if status == "A" and path not in merged_files:
                findings.append(
                    {"layer": "files", "kind": "file_added_by_side_missing",
                     "side": side, "path": path}
                )
            if status == "D" and path in merged_files:
                findings.append(
                    {"layer": "files", "kind": "file_deleted_by_side_resurrected",
                     "side": side, "path": path}
                )

    # (2) line-level union
    for path in sorted(set(files["ours"]) | set(files["theirs"])):
        if path not in merged_files or is_binary(merge, path):
            continue
        if path not in merged_cache:
            text = git("show", f"{merge}:{path}")
            merged_cache[path] = (
                {line.rstrip() for line in text.splitlines()},
                text,
            )
        merged_lines, merged_text = merged_cache[path]
        row = {"path": path, "sides": [], "inserted": {}, "undisposed": {}}
        for side, rev in sides.items():
            if path not in files[side]:
                continue
            row["sides"].append(side)
            inserted = [
                line
                for line in added_lines(base, rev, path)
                if len(line.strip()) >= 8  # blank/brace lines carry no content
            ]
            row["inserted"][side] = len(inserted)
            undisposed = []
            for line in inserted:
                if line in merged_lines:
                    continue
                disp = disposition_for(dispositions, path, side, line)
                ok, how = (False, "no disposition")
                if disp is not None:
                    ok, how = discharge(disp, line, merged_lines, merged_text)
                if ok:
                    row.setdefault("disposed", []).append(
                        {"line": line.strip()[:100], "how": how,
                         "reason": disp.get("reason", "")}
                    )
                else:
                    undisposed.append(line)
                    findings.append(
                        {"layer": "lines", "kind": "line_inserted_by_side_missing",
                         "side": side, "path": path, "line": line, "detail": how}
                    )
            row["undisposed"][side] = undisposed
        per_file.append(row)
    return findings, per_file


def import_registry(rev: str, src_prefix: str) -> dict[str, dict]:
    """Import ``DECLARED_NUMBERS`` from ``rev``'s own tree."""
    with tempfile.TemporaryDirectory() as tmp:
        archive = subprocess.run(
            ("git", "archive", rev, src_prefix),
            capture_output=True,
            check=False,
        )
        if archive.returncode != 0:
            raise SystemExit(f"git archive {rev} failed")
        subprocess.run(("tar", "-x", "-C", tmp), input=archive.stdout, check=True)
        code = (
            "import json;"
            "from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS as D;"
            "print(json.dumps({e.key: {'group': e.group, 'declared': e.declared,"
            " 'kind': e.kind, 'tier': e.tier} for e in D}))"
        )
        out = subprocess.run(
            (sys.executable, "-c", code),
            cwd=tmp,
            env={"PYTHONPATH": str(Path(tmp) / src_prefix), "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode != 0:
            raise SystemExit(f"import registry at {rev} failed: {out.stderr[-400:]}")
        return json.loads(out.stdout)


def revalue_for(dispositions: list[dict], side: str, key: str) -> dict | None:
    for d in dispositions:
        if d.get("action") == "revalue" and d.get("key") == key and d.get("side", "any") in (side, "any"):
            return d
    return None


def audit_registry(
    merge: str,
    ours: str,
    theirs: str,
    group_map: dict[str, str],
    src_prefix: str,
    dispositions: list[dict] | None = None,
) -> dict:
    revs = {"ours": ours, "theirs": theirs, "merge": merge}
    rows = {name: import_registry(rev, src_prefix) for name, rev in revs.items()}
    merged = rows["merge"]
    findings: list[dict] = []
    per_side: dict[str, dict] = {}
    for side in ("ours", "theirs"):
        lost, differing, corrections = [], [], []
        for key, row in rows[side].items():
            target = merged.get(key)
            if target is None:
                lost.append({"key": key, "group": row["group"]})
                findings.append(
                    {"layer": "registry", "kind": "registered_number_missing",
                     "side": side, "key": key}
                )
                continue
            if abs(target["declared"] - row["declared"]) > 1e-12 * max(
                1.0, abs(row["declared"])
            ):
                declared_revalue = revalue_for(dispositions or [], side, key)
                if declared_revalue is not None and abs(
                    target["declared"] - declared_revalue["to"]
                ) <= 1e-12 * max(1.0, abs(declared_revalue["to"])):
                    corrections.append(
                        {"key": key, "from": row["declared"],
                         "to": target["declared"], "reason": declared_revalue["reason"]}
                    )
                    continue
                differing.append(
                    {"key": key, "side": row["declared"], "merge": target["declared"]}
                )
                findings.append(
                    {"layer": "registry", "kind": "registered_number_changed",
                     "side": side, "key": key}
                )
        per_side[side] = {
            "rows": len(rows[side]),
            "missing": lost,
            "changed": differing,
            "declared_corrections": corrections,
        }
    groups = {}
    for row in merged.values():
        groups[row["group"]] = groups.get(row["group"], 0) + 1
    union_expected = len(set(rows["ours"]) | set(rows["theirs"]))
    return {
        "rows_per_side": {k: len(v) for k, v in rows.items()},
        "union_expected_rows": union_expected,
        "merged_rows": len(merged),
        "group_counts": groups,
        "group_map": group_map,
        "per_side": per_side,
        "findings": findings,
        "verdict": "PASS" if not findings else "DROP_DETECTED",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge", required=True, help="merge commit to audit")
    parser.add_argument("--ours", default=None, help="first-parent side (default merge^1)")
    parser.add_argument("--theirs", default=None, help="second-parent side (default merge^2)")
    parser.add_argument("--dispositions", default=None, help="renames/rewrites JSON")
    parser.add_argument("--registry", action="store_true", help="also audit the registry")
    parser.add_argument("--src-prefix", default="src")
    parser.add_argument("--json", default=None, help="write the report here")
    args = parser.parse_args(argv)

    ours = args.ours or f"{args.merge}^1"
    theirs = args.theirs or f"{args.merge}^2"
    dispositions = (
        json.loads(Path(args.dispositions).read_text()) if args.dispositions else []
    )
    group_map = (dispositions[-1].get("group_map", {}) if dispositions else {}) or {}

    line_findings, per_file = audit_lines(args.merge, ours, theirs, dispositions)
    report = {
        "merge": git("rev-parse", args.merge).strip(),
        "merge_parents": git("rev-list", "--parents", "-n", "1", args.merge).split()[1:],
        "ours": git("rev-parse", ours).strip(),
        "theirs": git("rev-parse", theirs).strip(),
        "merge_base": git("merge-base", ours, theirs).strip(),
        "layer_lines": {
            "dispositions": len(dispositions),
            "per_file": per_file,
            "findings": line_findings,
        },
    }
    if args.registry:
        report["layer_registry"] = audit_registry(
            args.merge, ours, theirs, group_map, args.src_prefix, dispositions
        )
    dropped = [f for f in line_findings if f["layer"] == "files"] + [
        f for f in line_findings if f["layer"] == "lines"
    ]
    if args.registry and report["layer_registry"]["findings"]:
        dropped += report["layer_registry"]["findings"]
    report["findings"] = dropped
    report["verdict"] = "PASS" if not dropped else "DROP_DETECTED"

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text + "\n")
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "findings": len(dropped),
                "files_audited": len(per_file),
                "registry": report.get("layer_registry", {}).get("verdict", "skipped"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())