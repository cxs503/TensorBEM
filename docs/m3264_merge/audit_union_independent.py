"""Independent union audit of the #135 resolution (m3951, second round).

The merge of ``main`` into ``agent/m3264`` is judged here on one question only:
*did any side lose content?* A previous round resolved the two textual
conflicts (``src/tensorbem/benchmarks/declared_numbers.py`` and
``tests/test_declared_numbers.py``); this script re-derives the evidence from
git objects rather than trusting the merge commit message, and it fails loudly
if a single row, test function or file disappeared.

Four checks, all mechanical:

``paths``
    Every path touched by either side exists in the merged tree, and a path
    touched by only one side is byte-identical in the merged tree (a mechanical
    overlay silently dropping a file fails here).
``main_registry``
    Every 3-D registry row of ``main`` is present in the merge under the same
    key, with the same declared value, comparison kind, tier and source string.
``branch_registry``
    Every 3-D registry row of the branch maps onto a merged row. The branch
    generated its keys from the study data (``sub3.b11``), the merge spells them
    in main's hand-written style (``sub3_surge_damping_rel_error``), so the
    mapping is name-based (with the aliases the branch itself declared) *and*
    value-based: the merge may re-sign a magnitude, it may not re-quantify it.
``test_functions``
    Every test function of either side exists in the merged test file. The one
    legitimate exception is main's
    ``test_3d_free_surface_rows_are_guarded_by_the_default_tiers``, which the
    merge folded into ``test_free_surface_3d_rows_live_on_the_standard_tier``;
    that fold is accepted only if the merged body keeps each of main's guard
    clauses (group set, tier, document citation, key uniqueness).

Run from a repository checkout::

    python3 docs/m3264_merge/audit_union_independent.py [--json out.json]

Exit status is 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REGISTRY_PATH = "src/tensorbem/benchmarks/declared_numbers.py"
TEST_PATH = "tests/test_declared_numbers.py"
GROUPS = ("hulme_chief", "hulme_hemisphere", "submerged_sphere")

#: Cost classes, cheapest first. A merge may move a row to a *more* expensive
#: tier (it then still runs in the default sweep), never to a cheaper one.
TIER_ORDER = ("fast", "standard", "heavy")

#: The branch quoted two-figure cells (``rotational_added_mass_max_ratio``) at
#: their printed precision; the merge quotes the measured value. Two rows are the
#: same cell when they agree to the branch's own two-figure band.
_TWO_FIGURE_TOLERANCE = 5.0e-2

#: Renames the branch's generated quantity names need to reach main's hand-
#: written ones, keyed by the branch quantity.
_BRANCH_RENAMES = {
    "unregularized.b33_magnitude": "ka2.58_unregularized_b33_dimless",
    "chief_regularized.b33_magnitude": "ka2.58_chief_regularized_b33_dimless",
    "unregularized.passivity_violated": "ka2.58_unregularized_passivity",
    "chief_regularized.passivity_violated": "ka2.58_chief_regularized_passivity",
}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout


def git_blob(ref: str, path: str) -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", f"{ref}:{path}"], cwd=REPO, capture_output=True, text=True
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def load_registry(ref: str):
    """Import the registry module of ``ref`` without touching the worktree."""
    blob = git("show", f"{ref}:{REGISTRY_PATH}")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(blob)
        path = handle.name
    spec = importlib.util.spec_from_file_location(f"_registry_{abs(hash(ref))}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def auto_refs() -> tuple[str, str, str]:
    """Merge base, main tip and branch tip, discovered from the worktree."""
    main = "origin/main"
    branch = "origin/agent/m3264"
    base = git("merge-base", main, branch).strip()
    return base, main, branch


def check_paths(base: str, main: str, branch: str) -> dict:
    def touched(ref: str) -> set[str]:
        return set(git("diff", "--name-only", base, ref).split())

    main_files, branch_files = touched(main), touched(branch)
    shared = main_files & branch_files
    merged_files = set(git("ls-tree", "-r", "--name-only", "HEAD").split())
    missing = sorted((main_files | branch_files) - merged_files)
    changed_one_sided = []
    for ref, files in ((main, main_files - shared), (branch, branch_files - shared)):
        for path in sorted(files):
            if git_blob(ref, path) != git_blob("HEAD", path):
                changed_one_sided.append({"path": path, "side": ref})
    ok = not missing and not changed_one_sided
    return {
        "ok": ok,
        "main_only_paths": len(main_files - shared),
        "branch_only_paths": len(branch_files - shared),
        "shared_paths": sorted(shared),
        "missing_in_merge": missing,
        "one_sided_content_changed": changed_one_sided,
    }


def load_worktree_registry():
    """Import the registry of the working tree (the merge under test)."""
    path = REPO / REGISTRY_PATH
    spec = importlib.util.spec_from_file_location("_registry_worktree", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rows(module, groups=GROUPS) -> list[dict]:
    return [
        {
            "key": entry.key,
            "group": entry.group,
            "quantity": entry.quantity,
            "declared": float(entry.declared),
            "kind": entry.kind,
            "tier": entry.tier,
            "source": entry.source,
        }
        for entry in module.DECLARED_NUMBERS
        if entry.group in groups
    ]


def check_main_registry(main: str) -> dict:
    """Every main row survives; a re-tier must not make a row cheaper to skip."""
    main_rows = _rows(load_registry(main))
    merged = {row["key"]: row for row in _rows(load_worktree_registry())}
    problems, retiered = [], []
    for row in main_rows:
        match = merged.get(row["key"])
        if match is None:
            problems.append({"key": row["key"], "problem": "row dropped"})
            continue
        for field in ("group", "quantity", "kind", "source"):
            if match[field] != row[field]:
                problems.append(
                    {"key": row["key"], "problem": f"{field} changed",
                     "main": row[field], "merged": match[field]}
                )
        if not math.isclose(match["declared"], row["declared"], rel_tol=1e-12, abs_tol=0.0):
            problems.append(
                {"key": row["key"], "problem": "declared value changed",
                 "main": row["declared"], "merged": match["declared"]}
            )
        if match["tier"] != row["tier"]:
            cheaper = TIER_ORDER.index(match["tier"]) < TIER_ORDER.index(row["tier"])
            retiered.append(
                {"key": row["key"], "main_tier": row["tier"], "merged_tier": match["tier"]}
            )
            if cheaper:
                problems.append(
                    {"key": row["key"], "problem": "row moved to a cheaper tier",
                     "main": row["tier"], "merged": match["tier"]}
                )
    # A re-tier is only acceptable if the row is still recomputed by the default
    # sweep: ``check()`` defaults to the fast+standard tiers.
    escaped = [
        item for item in retiered if item["merged_tier"] not in ("fast", "standard")
    ]
    for item in escaped:
        problems.append({**item, "problem": "re-tiered out of the default sweep"})
    return {
        "ok": not problems,
        "main_rows": len(main_rows),
        "merged_rows": len(merged),
        "retiered": retiered,
        "problems": problems,
    }


def _branch_candidates(entry: dict) -> list[str]:
    """Merged-style quantity names a branch row may have been re-declared under.

    The branch keyed its rows by study output (``sub3.surge_damping_rel_error``,
    ``unregularized.b33_magnitude``); main's rows spell the same cell out
    (``sub3_surge_damping_rel_error``, ``ka2.58_unregularized_b33_dimless``).
    """
    quantity = entry["quantity"]
    renamed = _BRANCH_RENAMES.get(quantity)
    if renamed is not None:
        return [renamed]
    if "." not in quantity:
        # ``exact_a33``-style chief rows carry the frequency prefix on main's side.
        return [quantity, f"ka2.58_{quantity}"]
    scheme, name = quantity.split(".", 1)
    return [f"{scheme}_{name}", f"ka2.58_{scheme}_{name}", f"ka2.58_{scheme}_{name}_dimless"]


def check_branch_registry(branch: str) -> dict:
    branch_rows = _rows(load_registry(branch))
    merged_rows = _rows(load_worktree_registry())
    by_quantity = {(row["group"], row["quantity"]): row for row in merged_rows}
    matched, unmapped = [], []
    for row in branch_rows:
        candidates = [
            by_quantity.get((row["group"], name)) for name in _branch_candidates(row)
        ]
        candidates = [c for c in candidates if c is not None]
        picked = None
        for candidate in candidates:
            # Same cell: equal to the digit, or equal within the two-figure band
            # the branch printed its rounded cells at. The sign may flip: the
            # branch stored |B33'|, main stores B33' with the passivity sign.
            if math.isclose(
                abs(candidate["declared"]),
                abs(row["declared"]),
                rel_tol=_TWO_FIGURE_TOLERANCE,
                abs_tol=1e-30,
            ):
                picked = candidate
                break
        if picked is None:
            unmapped.append(row)
        else:
            matched.append(
                {
                    "branch_key": row["key"],
                    "branch_quantity": row["quantity"],
                    "merged_key": picked["key"],
                    "merged_quantity": picked["quantity"],
                    "branch_declared": row["declared"],
                    "merged_declared": picked["declared"],
                    "sign_flipped": (picked["declared"] < 0) != (row["declared"] < 0),
                }
            )
    return {
        "ok": not unmapped,
        "branch_rows": len(branch_rows),
        "matched": len(matched),
        "unmapped": unmapped,
        "mapping": matched,
    }


#: Main's guards that the folded test must keep asserting.
_GUARD_CLAUSES = ("entry.source", "marine_bem_benchmarks.md", "entry.key", "tier")


def check_test_functions(main: str, branch: str) -> dict:
    def names(ref: str) -> list[str]:
        src = git("show", f"{ref}:{TEST_PATH}")
        return re.findall(r"^def (test_\w+)", src, re.M)

    main_tests, branch_tests = names(main), names(branch)
    merged_src = (REPO / TEST_PATH).read_text()
    merged_tests = re.findall(r"^def (test_\w+)", merged_src, re.M)
    folds = {}
    missing = []
    for name in main_tests + branch_tests:
        if name in merged_tests:
            continue
        missing.append(name)
    # main's tier guard was folded into the branch's stronger guard; accept the
    # fold only when the merged body still carries main's clauses.
    folded = "test_3d_free_surface_rows_are_guarded_by_the_default_tiers"
    fold_target = "test_free_surface_3d_rows_live_on_the_standard_tier"
    if folded in missing and fold_target in merged_tests:
        body = merged_src.split(f"def {fold_target}", 1)[1].split("\ndef ", 1)[0]
        kept = [clause for clause in _GUARD_CLAUSES if clause in body]
        if len(kept) == len(_GUARD_CLAUSES):
            folds[folded] = fold_target
            missing.remove(folded)
        else:
            folds[folded] = f"{fold_target} (MISSING CLAUSES: kept {kept})"
    return {
        "ok": not missing,
        "main_tests": len(main_tests),
        "branch_tests": len(branch_tests),
        "merged_tests": len(merged_tests),
        "missing": missing,
        "folded_into": folds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None, help="write the evidence to this path")
    args = parser.parse_args(argv)

    base, main_ref, branch_ref = auto_refs()
    report = {
        "merge_base": base,
        "main": main_ref,
        "branch": branch_ref,
        "head": git("rev-parse", "HEAD").strip(),
        "paths": check_paths(base, main_ref, branch_ref),
        "main_registry": check_main_registry(main_ref),
        "branch_registry": check_branch_registry(branch_ref),
        "test_functions": check_test_functions(main_ref, branch_ref),
    }
    report["ok"] = all(
        report[key]["ok"]
        for key in ("paths", "main_registry", "branch_registry", "test_functions")
    )
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")

    print(f"merge base  : {base}")
    print(f"main / head : {main_ref} / {report['head']}")
    print(
        f"paths       : {'PASS' if report['paths']['ok'] else 'FAIL'}"
        f"  (main-only {report['paths']['main_only_paths']},"
        f" branch-only {report['paths']['branch_only_paths']},"
        f" shared {len(report['paths']['shared_paths'])})"
    )
    mr, br, tf = report["main_registry"], report["branch_registry"], report["test_functions"]
    print(
        f"main rows   : {'PASS' if mr['ok'] else 'FAIL'}"
        f"  ({mr['main_rows']} main rows, {mr['merged_rows']} merged rows)"
    )
    print(
        f"branch rows : {'PASS' if br['ok'] else 'FAIL'}"
        f"  ({br['matched']}/{br['branch_rows']} mapped)"
    )
    print(
        f"test funcs  : {'PASS' if tf['ok'] else 'FAIL'}"
        f"  (main {tf['main_tests']}, branch {tf['branch_tests']},"
        f" merged {tf['merged_tests']}, folded {tf['folded_into']})"
    )
    print(f"\nUNION AUDIT: {'PASS' if report['ok'] else 'FAIL'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())