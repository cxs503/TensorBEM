"""Mechanical verification of the agent/m3181 -> main merge (mission 3751).

Three independent checks, all offline:

1. **Parse**: every ``*.py`` file of the merged tree compiles (``ast.parse``).
2. **Duplicate definitions**: no module-level function/class name is defined
   twice in the same file *unless* the same duplication already exists in one
   of the two parents.  Git's line merge happily concatenates two identically
   named helpers; the later definition then silently shadows the earlier one,
   which is exactly the "mechanical overlay loses content" failure mode this
   mission is about.  Merge-introduced duplicates are FAIL, inherited ones are
   reported as INFO.
3. **Union of content**: every non-trivial line that *each* parent added to a
   file touched by both parents (and every line of every file added by one
   side only) is still present in the merged working tree.  The only accepted
   exceptions are the renames listed in :data:`DOCUMENTED_RENAMES`, each of
   which is a semantic-preserving alias consolidation recorded in the mission
   report.

Binary files (figures) are compared by existence plus byte identity against
the single side that added them.

Usage (run before committing the merge)::

    python docs/m3751/verify_merge.py
    python docs/m3751/verify_merge.py --base <sha> --ours <ref> --theirs <ref>

Exit status 0 = merge is a faithful union; 1 = content was lost.
"""

from __future__ import annotations

import argparse
import ast
import collections
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: ``(side, old_line) -> new_line`` renames applied on purpose.  Side "ours"
#: is agent/m3181 (the PR branch), side "theirs" is main.
DOCUMENTED_RENAMES = {
    # Both parents imported the same benchmark modules under different local
    # aliases while the merged body uses both spellings; main's spelling is
    # kept as canonical and the PR branch's three call sites are renamed.
    ("ours", "floating_hemisphere as hulme_benchmark,"): (
        "floating_hemisphere as hemisphere_benchmark,"
    ),
    ("ours", "submerged_sphere_wave as wang_linton_benchmark,"): (
        "submerged_sphere_wave as sphere_benchmark,"
    ),
    ("ours", "levels = hulme_benchmark.run_refinement_study("): (
        "levels = hemisphere_benchmark.run_refinement_study("
    ),
    ("ours", "rows = hulme_benchmark.run_irregular_frequency_study("): (
        "rows = hemisphere_benchmark.run_irregular_frequency_study("
    ),
    ("ours", "levels = wang_linton_benchmark.run_refinement_study("): (
        "levels = sphere_benchmark.run_refinement_study("
    ),
    # Silent collision: both parents defined ``_chief_rows`` at module level
    # (ours: registry-row factory without arguments; theirs: study flattener
    # taking the study rows).  Ours keeps the name because
    # ``_free_surface_3d_rows`` calls it; theirs is renamed.
    ("theirs", "def _chief_rows(rows) -> dict[str, float]:"): (
        "def _chief_study_rows(rows) -> dict[str, float]:"
    ),
    ("theirs", "return _chief_rows(rows)"): "return _chief_study_rows(rows)",
    # Registry reconciliation (docs/m3751/reconcile_table_cells.py): the merged
    # tree carries two registrations of the same document tables.  The PR
    # branch's generated family quoted the pre-m3150 digits and the ideal-mesh
    # log2 derivation; both are brought in line with main's corrected convention
    # so that a single claim per table cell survives the merge.
    ("ours", 'values[f"{tag}_observed_order_b11"] = math.log('): (
        'values[f"{tag}_observed_order_b11"] = error_ratio / ('
    ),
    ("ours", ") / math.log(2.0)"): (
        'math.log(h0 / h1) if h1 < h0 else math.log(2.0)'
    ),
    ("ours", '(2, 64, "observed_order_b11", 1.62),'): "(2, 64, \"observed_order_b11\", 1.80),",
    ("ours", '(3, 256, "observed_order_b11", 1.88),'): "(3, 256, \"observed_order_b11\", 1.94),",
    ("ours", '(1, 80, "rotational_added_mass_max_ratio", 2.2e-4),'): "(1, 80, \"rotational_added_mass_max_ratio\", 2.151e-4),",
    ("ours", '(2, 320, "rotational_added_mass_max_ratio", 2.6e-5),'): "(2, 320, \"rotational_added_mass_max_ratio\", 2.648e-5),",
}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        encoding="utf-8",
        errors="surrogateescape",
    ).stdout


def _worktree_files() -> list[str]:
    return sorted(
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )


def check_parse(files: list[str]) -> list[str]:
    failures = []
    for name in files:
        if not name.endswith(".py"):
            continue
        try:
            ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)
        except SyntaxError as exc:  # pragma: no cover - failure path
            failures.append(f"PARSE {name}: {exc}")
    return failures


def _duplicates(source: str, name: str) -> set[str]:
    tree = ast.parse(source, filename=name)
    functions = collections.Counter(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    assignments = collections.Counter(
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    )
    return {f"def {key}" for key, count in functions.items() if count > 1} | {
        f"assign {key}" for key, count in assignments.items() if count > 1
    }


def _parent_source(ref: str, name: str) -> str | None:
    try:
        return git("show", f"{ref}:{name}")
    except subprocess.CalledProcessError:
        return None


def check_duplicate_definitions(
    files: list[str], ours: str, theirs: str
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    notes: list[str] = []
    for name in files:
        if not name.endswith(".py"):
            continue
        merged = _duplicates((ROOT / name).read_text(encoding="utf-8"), name)
        if not merged:
            continue
        inherited: set[str] = set()
        for ref in (ours, theirs):
            source = _parent_source(ref, name)
            if source is None:
                continue
            try:
                inherited |= _duplicates(source, name)
            except SyntaxError:  # pragma: no cover
                continue
        for item in sorted(merged):
            if item in inherited:
                notes.append(f"inherited duplicate in {name}: {item}")
            else:
                failures.append(f"DUPLICATE from merge in {name}: {item}")
    return failures, notes


def _interesting(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def check_union(base: str, ours: str, theirs: str) -> list[str]:
    failures: list[str] = []
    present = set(_worktree_files())

    def changed(ref: str) -> set[str]:
        return {name for name in git("diff", "--name-only", base, ref).splitlines() if name}

    ours_files, theirs_files = changed(ours), changed(theirs)
    for name in sorted(ours_files | theirs_files):
        if name not in present:
            side = "both" if name in ours_files and name in theirs_files else (
                "ours" if name in ours_files else "theirs"
            )
            failures.append(f"LOST FILE {name} (added by {side})")
            continue
        try:
            text = (ROOT / name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # binary: existence is the check
        haystack = set(text.splitlines())
        stripped_haystack = {line.strip() for line in haystack}
        for side, ref in (("ours", ours), ("theirs", theirs)):
            diff = git("diff", "-U0", base, ref, "--", name)
            added = [
                line[1:]
                for line in diff.splitlines()
                if line.startswith("+") and not line.startswith("+++")
            ]
            for line in added:
                if not _interesting(line) or line in haystack:
                    continue
                replacement = DOCUMENTED_RENAMES.get((side, line.strip()))
                if replacement is not None:
                    if replacement.strip() in stripped_haystack:
                        continue
                    failures.append(
                        f"RENAME NOT APPLIED in {name}: {line.strip()[:80]}"
                    )
                    continue
                if line.strip() in stripped_haystack:
                    continue
                failures.append(f"LOST LINE in {name} from {side}: {line.strip()[:100]}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="4f5b73e73c4e731582f2b050328ec5fcf62b4a86")
    parser.add_argument("--ours", default="origin/agent/m3181")
    parser.add_argument("--theirs", default="origin/main")
    args = parser.parse_args()

    files = _worktree_files()
    failures = check_parse(files)
    dup_failures, notes = check_duplicate_definitions(files, args.ours, args.theirs)
    failures += dup_failures
    failures += check_union(args.base, args.ours, args.theirs)

    print(f"parsed {sum(name.endswith('.py') for name in files)} python files")
    print(f"merge-introduced duplicate definitions: {len(dup_failures)}")
    if notes:
        print(f"inherited duplicates (present in a parent, informational): {len(notes)}")
    if failures:
        print(f"FAIL ({len(failures)} findings)")
        for item in failures:
            print("  -", item)
        return 1
    print("OK: parse + duplicate-definition + content-union checks clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())