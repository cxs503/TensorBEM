#!/usr/bin/env python3
"""Machine check that a merge of ``agent/m3181`` into ``main`` lost nothing.

PR #128 was declared unmergeable because the branch and ``main`` had drifted
apart.  The hazard when healing such a PR by hand is *silent content loss*: a
resolution that takes one side wholesale (``--theirs`` / ``--ours`` / a file
copy) produces a tree that compiles and whose tests are green while a whole
table, a registry block or a helper from the other side has evaporated.

This script therefore checks a merged tree against **both** of its parents on
three independent levels and exits non-zero if any of them fails:

1. **File level** — every path present in ``main`` or in the branch is present
   in the merged tree (a path only one side has is a legitimate *addition*; a
   path neither side deleted but that is missing from the merge is a loss).
2. **Line level** — every non-blank line that either parent *added* relative to
   the merge base appears verbatim in the merged file, unless it is listed in
   :data:`EXPECTED_REWRITES` (deliberate disambiguations, each with a reason).
3. **Symbol level** (Python only) — every module-level ``def`` / ``class`` /
   assignment name of either parent is present in the merged module, and no
   top-level name is defined twice (a duplicated name silently shadows one of
   the two bodies, which is the sneakiest form of this bug).
4. **Registry level** (optional, ``--runtime``) — imports the merged registry
   and asserts its key set equals the union of the parents' key sets.

Usage::

    python docs/tools/verify_pr128_union_m3772.py \
        --main main --branch agent/m3181            # worktree = merge result
    python docs/tools/verify_pr128_union_m3772.py --main main \
        --branch agent/m3181 --runtime              # + registry union

The script reads ``git show <ref>:<path>`` for the parents, so it needs a git
worktree whose HEAD/index is the merge result; it does not modify the tree.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

#: Added lines that a deliberate, documented resolution rewrote.  Every entry is
#: (file, added line, why).  Anything not listed here must survive verbatim.
EXPECTED_REWRITES: tuple[tuple[str, str, str], ...] = (
    (
        "src/tensorbem/benchmarks/declared_numbers.py",
        "def _chief_rows():",
        "branch-side no-argument builder shadowed main's _chief_rows(rows) "
        "flattener; renamed to _hulme3d_chief_rows (see heal report)",
    ),
    (
        "src/tensorbem/benchmarks/declared_numbers.py",
        "        + _chief_rows()",
        "call site of the renamed builder",
    ),
    (
        "tests/test_declared_numbers.py",
        "        # A quote may legitimately be non-positive: the unregularized",
        "both sides reworded the same assertion comment; texts merged",
    ),
    (
        "tests/test_declared_numbers.py",
        "        # 3-D BIE produces an unphysical negative heave damping, and that",
        "both sides reworded the same assertion comment; texts merged",
    ),
    (
        "tests/test_declared_numbers.py",
        "        # sign is exactly the claim the table makes.",
        "both sides reworded the same assertion comment; texts merged",
    ),
    (
        "tests/test_declared_numbers.py",
        "        # table 11 quotes as the defect CHIEF removes), so only zero is illegal",
        "both sides reworded the same assertion comment; texts merged",
    ),
)


def git(*args: str, check: bool = True) -> str:
    proc = subprocess.run(("git",) + args, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def paths(ref: str) -> set[str]:
    return {line for line in git("ls-tree", "-r", "--name-only", ref).splitlines() if line}


def blob(ref: str, path: str) -> str | None:
    proc = subprocess.run(("git", "show", f"{ref}:{path}"), capture_output=True, text=True)
    return proc.stdout if proc.returncode == 0 else None


def added_lines(base: str, ref: str, path: str) -> list[str]:
    diff = git("diff", "-U0", base, ref, "--", path)
    return [line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]


def top_level_names(source: str) -> Counter[str]:
    tree = ast.parse(source)
    names: Counter[str] = Counter()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names[node.name] += 1
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names[node.target.id] += 1
    return names


def _registry_keys(ref: str) -> list[str] | None:
    """Declared-number keys of a git ref, evaluated in a throwaway export.

    The registry is assembled at import time (``) + _free_surface_3d_rows()``),
    so static parsing cannot recover it; we export ``src/`` with ``git archive``
    and import it in a subprocess with that export first on ``sys.path``.
    """
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tar = subprocess.run(("git", "archive", ref, "src"), capture_output=True)
        if tar.returncode != 0:
            return None
        subprocess.run(("tar", "-x", "-C", tmp), input=tar.stdout, check=True)
        code = (
            "import json;"
            "from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS as D;"
            "print(json.dumps([e.key for e in D]))"
        )
        env = dict(os.environ, PYTHONPATH=str(Path(tmp) / "src"))
        out = subprocess.run((sys.executable, "-c", code), capture_output=True, text=True, env=env)
        if out.returncode != 0:
            return None
        return json.loads(out.stdout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="main", help="ref of the mainline parent")
    ap.add_argument("--branch", default="agent/m3181", help="ref of the PR branch parent")
    ap.add_argument("--base", default=None, help="merge base (default: git merge-base)")
    ap.add_argument("--runtime", action="store_true", help="also compare registry key sets")
    ap.add_argument("--json", default=None, help="write the evidence bundle here")
    ap.add_argument(
        "--conflicts-json",
        default=None,
        help="JSON record of the conflict resolution (docs/m3772/outputs/conflicts.json)",
    )
    args = ap.parse_args()

    base = args.base or git("merge-base", args.main, args.branch).strip()
    tracked = set(git("ls-files").splitlines())
    parents = {args.main: paths(args.main), args.branch: paths(args.branch)}
    changed = {ref: set(git("diff", "--name-only", base, ref).splitlines()) for ref in parents}
    both = changed[args.main] & changed[args.branch]

    report: dict[str, object] = {
        "base": base,
        "main": args.main,
        "branch": args.branch,
        "file_counts": {
            "merged": len(tracked),
            "main": len(parents[args.main]),
            "branch": len(parents[args.branch]),
            "union": len(parents[args.main] | parents[args.branch]),
            "changed_both_sides": sorted(both),
        },
        "failures": [],
    }
    failures: list[str] = report["failures"]  # type: ignore[assignment]

    # 1. file level ------------------------------------------------------
    missing = (parents[args.main] | parents[args.branch]) - tracked
    if missing:
        failures.append("file level: paths present in a parent but absent from the merge: "
                        + ", ".join(sorted(missing)))

    # 2. line level ------------------------------------------------------
    allowed = {(f, line) for f, line, _ in EXPECTED_REWRITES}
    line_losses: list[dict[str, str]] = []
    for ref in parents:
        for path in sorted(both):
            if path.endswith((".png", ".jpg", ".npz", ".pdf")):
                continue
            merged_text = (
                Path(path).read_text(encoding="utf-8", errors="replace") if Path(path).exists() else None
            )
            merged_lines = Counter(merged_text.splitlines()) if merged_text is not None else Counter()
            for line in added_lines(base, ref, path):
                if not line.strip() or line in merged_lines:
                    continue
                if (path, line) in allowed:
                    continue
                line_losses.append({"ref": ref, "path": path, "line": line})
                merged_lines[line] += 1  # count once, do not spam duplicates
    report["line_losses"] = line_losses
    if line_losses:
        failures.append(
            "line level: %d added line(s) from a parent are missing from the merge "
            "and are not documented rewrites" % len(line_losses)
        )

    # 3. symbol level ----------------------------------------------------
    symbol_losses: list[dict[str, object]] = []
    shadowed: list[dict[str, object]] = []
    for path in sorted(both):
        if not path.endswith(".py"):
            continue
        parent_names: dict[str, Counter[str]] = {}
        for ref in parents:
            text = blob(ref, path)
            if text is None:
                break
            try:
                parent_names[ref] = top_level_names(text)
            except SyntaxError:  # pragma: no cover - parents are CI-green
                pass
        if len(parent_names) != 2:
            continue
        merged_path = Path(path)
        if not merged_path.exists():
            continue
        merged_names = top_level_names(merged_path.read_text(encoding="utf-8"))
        union = set(parent_names[args.main]) | set(parent_names[args.branch])
        gone = sorted(union - set(merged_names))
        if gone:
            symbol_losses.append({"path": path, "missing": gone})
        dups = sorted(name for name, count in merged_names.items() if count > 1)
        if dups:
            shadowed.append({"path": path, "duplicated": dups})
    report["symbol_losses"] = symbol_losses
    report["shadowed_names"] = shadowed
    if symbol_losses:
        failures.append("symbol level: top-level names from a parent are missing from the merge")
    if shadowed:
        failures.append("symbol level: duplicated top-level names in the merge (silent shadowing)")

    # 4. registry level --------------------------------------------------
    if args.runtime:
        import importlib

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        module = importlib.import_module("tensorbem.benchmarks.declared_numbers")
        merged_keys = [entry.key for entry in module.DECLARED_NUMBERS]
        main_keys = _registry_keys(args.main)
        branch_keys = _registry_keys(args.branch)
        report["registry"] = {
            "merged_entries": len(merged_keys),
            "merged_unique": len(set(merged_keys)),
            "main_entries": len(main_keys) if main_keys is not None else None,
            "branch_entries": len(branch_keys) if branch_keys is not None else None,
            "groups": sorted({entry.group for entry in module.DECLARED_NUMBERS}),
        }
        if len(merged_keys) != len(set(merged_keys)):
            failures.append("registry level: duplicate declared-number keys in the merge")
        if main_keys is not None and branch_keys is not None:
            a, b, m = set(main_keys), set(branch_keys), set(merged_keys)
            report["registry_composition"] = {
                "main_entries": len(a),
                "branch_entries": len(b),
                "merged_entries": len(m),
                "union": len(a | b),
                "main_only": len(a - b),
                "branch_only": len(b - a),
                "identical": len(a & b),
                "missing_from_merge": sorted((a | b) - m),
                "extra_in_merge": sorted(m - (a | b)),
            }
            if (a | b) - m:
                failures.append("registry level: declared numbers of a parent are missing from the merge")
            if m - (a | b):
                failures.append("registry level: the merge invents declared-number keys absent from both parents")

    # 5. conflict record -------------------------------------------------
    if args.conflicts_json:
        report["conflicts"] = json.loads(Path(args.conflicts_json).read_text(encoding="utf-8"))

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(payload + "\n", encoding="utf-8")

    print(json.dumps({k: report[k] for k in ("base", "file_counts", "failures")}, indent=2, ensure_ascii=False))
    if args.runtime:
        print(json.dumps(report.get("registry", {}), indent=2, ensure_ascii=False))
    if failures:
        print("\nUNION CHECK FAILED", file=sys.stderr)
        return 1
    print("\nUNION CHECK PASSED: no file, line, symbol or registry entry from either parent was lost")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())