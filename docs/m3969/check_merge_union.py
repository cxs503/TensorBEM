"""Independent, mechanical check that merging ``main`` into ``agent/m3250`` lost nothing.

Run from the worktree root (the merged tree must be checked out)::

    PYTHONPATH=src python3 docs/m3969/check_merge_union.py

The cure of PR #136 rests on a claim that is easy to assert and hard to prove:
"the merge is a semantic union, neither side's content was dropped".  This
script decides it from the three git revisions rather than by eye, and is
deliberately *independent* of ``docs/m3250/merge_audit/coverage_audit.py``
(written by the merge round itself): it never imports that code and, for the
number registry, it compares *values* per family instead of re-using the
merge round's name mapping.

Three checks, one per deliverable that the two sides touched:

``R`` (registry, ``src/tensorbem/benchmarks/declared_numbers.py``)
    Load ``DECLARED_NUMBERS`` from the main revision, the branch revision and
    the merged working tree.  ``main``'s entries must survive *identically*
    (same group, quantity, declared value).  The branch registered the same
    physical tables under other group/quantity names, so its claims are matched
    by value: every branch claim must have a merged entry in its family whose
    declared value agrees to ``RTOL``.

``T`` (tests, ``tests/test_declared_numbers.py``)
    Every test function defined on either parent must still exist in the merged
    file, or be listed in ``RENAMED_TESTS`` with the function that inherited its
    assertions.

``D`` (documentation, ``docs/marine_bem_benchmarks.md``)
    Every line either parent *added* relative to the merge base must appear in
    the merged file, except the lines listed in ``SUPERSEDED`` (table cells the
    two sides corrected differently -- the merged file must then carry the
    corrected value, which is checked separately).

Exit status is 0 only if all three pass; ``union_report.json`` next to this
file records the per-item verdicts.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BASE = "4f5b73e73c4e731582f2b050328ec5fcf62b4a86"  # git merge-base main agent/m3250
MAIN = "origin/main"
BRANCH = "origin/agent/m3250"
REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

RTOL = 5.0e-2  # branch pinned 2-3 significant printed digits, 1.5-5 % bands

REGISTRY = "src/tensorbem/benchmarks/declared_numbers.py"
TESTS = "tests/test_declared_numbers.py"
DOC = "docs/marine_bem_benchmarks.md"

#: branch group -> merged group (same physical table, different naming round)
GROUP_MAP = {
    "hemisphere3d": "hulme_hemisphere",
    "hemisphere3d_chief": "hulme_chief",
    "submerged_sphere3d": "submerged_sphere",
}
#: branch "<panels>.<name>" -> merged "sub<level>_<name>"
PANEL_LEVEL = {16: 1, 64: 2, 256: 3, 20: 0, 80: 1, 320: 2}
#: quantity renames between the two naming rounds
QUANTITY_MAP = {"order_": "observed_order_", "b33_dimless_abs": "b33_dimless"}
#: the CHIEF family names its schemes instead of numbering panels
CHIEF_SCHEME = {"unregularized": "ka2.58_unregularized",
                "chief_regularized": "ka2.58_chief_regularized"}

#: test functions that were renamed, with the merged function carrying their
#: assertions (checked by reading the merged body; see the cure report).
RENAMED_TESTS = {
    "test_free_surface_3d_tables_are_reproduced":
        "test_3d_refinement_columns_are_monotone_and_inside_the_acceptance_band",
}
#: lines added by a parent that the merge must *not* keep verbatim, because the
#: two sides corrected the same printed cell differently; the merged file must
#: instead carry the corrected cell (checked in ``corrected_cells_present``).
SUPERSEDED = {
    "| 2 | 64 | 3.052": "table 10 surge-damping observed order: printed 1.62/1.88 "
                        "replaced by the h_mean-based 1.80/1.94",
    "| 3 | 256 | 2.783": "table 10 surge-damping observed order: printed 1.62/1.88 "
                         "replaced by the h_mean-based 1.80/1.94",
    "| 1 | 80 | 2.259": "table 12 rotational decoupling 2.2e-4 -> 2.15e-4",
    "| 2 | 320 | 2.188": "table 12 rotational decoupling 2.6e-5 -> 2.65e-5",
}


def git_show(rev: str, path: str) -> str:
    out = subprocess.run(
        ["git", "show", f"{rev}:{path}"], cwd=REPO, capture_output=True, text=True
    )
    if out.returncode != 0:
        raise SystemExit(f"git show {rev}:{path} failed: {out.stderr.strip()}")
    return out.stdout


def load_registry(source: str, tag: str):
    tmp = Path(tempfile.mkdtemp()) / f"{tag}.py"
    tmp.write_text(source)
    spec = importlib.util.spec_from_file_location(f"union_probe_{tag}", tmp)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return list(module.DECLARED_NUMBERS)


def mapped_key(entry):
    """branch entry -> (merged group, merged quantity), or "key" for identity.

    Three naming rounds have to be bridged, all of them observable from the two
    revisions alone:

    * the refinement families number panels (``16.a33`` -> ``sub1_a33``);
    * the CHIEF family names the scheme (``unregularized.a33`` ->
      ``ka2.58_unregularized_a33``);
    * every other family is untouched by the merge and is matched by its key,
      because there ``(group, quantity)`` is deliberately not unique -- the
      panel count lives in ``parameters``.
    """
    if entry.group in {"hulme_hemisphere", "hulme_chief", "submerged_sphere"}:
        return "key"
    group = GROUP_MAP.get(entry.group)
    if group is None:
        return "key"
    head, _, name = entry.quantity.partition(".")
    for old, new in QUANTITY_MAP.items():
        if name.startswith(old):
            name = name[len(old):] if old == "b33_dimless_abs" else new + name[len(old):]
            if old == "b33_dimless_abs":
                name = "b33_dimless"
            break
    if group == "hulme_chief":
        scheme = CHIEF_SCHEME.get(head)
        return (group, f"{scheme}_{name}") if scheme else None
    level = PANEL_LEVEL.get(int(head)) if head.isdigit() else None
    return (group, f"sub{level}_{name}") if level is not None else None


def check_registry() -> dict:
    main = load_registry(git_show(MAIN, REGISTRY), "main")
    branch = load_registry(git_show(BRANCH, REGISTRY), "branch")
    merged = load_registry((REPO / REGISTRY).read_text(), "merged")
    index = {(e.group, e.quantity): e for e in merged}
    by_key = {e.key: e for e in merged}
    if len(by_key) != len(merged):
        raise SystemExit("merged registry has duplicate keys -- reindex needed")

    main_lost, main_moved = [], []
    for entry in main:
        hit = by_key.get(entry.key)
        if hit is None:
            main_lost.append(entry.key)
        elif abs(hit.declared - entry.declared) > 1e-12:
            main_moved.append(
                {"key": entry.key, "main": entry.declared, "merged": hit.declared}
            )

    branch_lost = []
    worst = {"rel": 0.0, "claim": None, "mapped": None,
             "branch": None, "merged": None}
    for entry in branch:
        key = mapped_key(entry)
        hit = by_key.get(entry.key) if key == "key" else (index.get(key) if key else None)
        if hit is None:
            branch_lost.append({"claim": entry.key, "mapped": key})
            continue
        # the branch registered the passive-damping magnitudes (``..._abs``);
        # the merged registry keeps the signed value, so compare magnitudes
        measured = abs(hit.declared) if entry.quantity.endswith("_abs") else hit.declared
        ref = max(abs(entry.declared), abs(measured), 1e-30)
        rel = abs(measured - entry.declared) / ref
        if rel > RTOL:
            branch_lost.append(
                {"claim": entry.key, "mapped": key, "rel": rel,
                 "branch": entry.declared, "merged": hit.declared}
            )
        if rel > worst["rel"]:
            worst = {"rel": rel, "claim": entry.key, "mapped": key,
                     "branch": entry.declared, "merged": hit.declared}

    return {
        "main_entries": len(main),
        "branch_claims": len(branch),
        "merged_entries": len(merged),
        "main_lost": main_lost,
        "main_value_moved": main_moved,
        "branch_lost": branch_lost,
        "worst_branch_deviation": worst,
        "ok": not main_lost and not main_moved and not branch_lost,
    }


def test_functions(text: str) -> set[str]:
    return set(re.findall(r"^def (test_\w+)", text, re.M))


def check_tests() -> dict:
    main = test_functions(git_show(MAIN, TESTS))
    branch = test_functions(git_show(BRANCH, TESTS))
    merged = test_functions((REPO / TESTS).read_text())

    def missing(names):
        out = []
        for name in sorted(names - merged):
            heir = RENAMED_TESTS.get(name)
            if heir is None or heir not in merged:
                out.append(name)
        return out

    result = {
        "main_tests": len(main),
        "branch_tests": len(branch),
        "merged_tests": len(merged),
        "main_missing": missing(main),
        "branch_missing": missing(branch),
        "renamed": {k: v for k, v in RENAMED_TESTS.items() if k not in merged},
        "merged_only": sorted(merged - main - branch),
    }
    result["ok"] = not result["main_missing"] and not result["branch_missing"]
    return result


def added_lines(rev: str) -> set[str]:
    out = subprocess.run(
        ["git", "diff", "--unified=0", BASE, rev, "--", DOC],
        cwd=REPO, capture_output=True, text=True,
    ).stdout
    lines = set()
    for raw in out.splitlines():
        if raw.startswith("+") and not raw.startswith("+++"):
            line = raw[1:].strip()
            if line:
                lines.add(line)
    return lines


def check_doc() -> dict:
    merged_lines = {(ln.strip()) for ln in (REPO / DOC).read_text().splitlines()}
    merged_blob = "\n".join(merged_lines)

    def missing(rev):
        lost, superseded = [], []
        for line in sorted(added_lines(rev)):
            if line in merged_lines:
                continue
            marker = next((k for k in SUPERSEDED if line.startswith(k)), None)
            if marker:
                superseded.append({"line": line, "reason": SUPERSEDED[marker]})
            else:
                lost.append(line)
        return lost, superseded

    main_lost, main_super = missing(MAIN)
    branch_lost, branch_super = missing(BRANCH)
    # the corrections the merge must carry instead of those printed cells
    corrections = {
        "1.80 / 1.94": "1.80 / 1.94" in merged_blob,
        "2.15\\times10^{-4}": "2.15\\times10^{-4}" in merged_blob,
        "2.65\\times10^{-5}": "2.65\\times10^{-5}" in merged_blob,
    }
    result = {
        "main_added_lines": len(added_lines(MAIN)),
        "branch_added_lines": len(added_lines(BRANCH)),
        "main_missing": main_lost,
        "branch_missing": branch_lost,
        "main_superseded": main_super,
        "branch_superseded": branch_super,
        "corrected_cells_present": corrections,
    }
    result["ok"] = (
        not main_lost and not branch_lost and all(corrections.values())
    )
    return result


def main() -> int:
    report = {
        "base": BASE,
        "main": MAIN,
        "branch": BRANCH,
        "rtol": RTOL,
        "registry": check_registry(),
        "tests": check_tests(),
        "doc": check_doc(),
    }
    report["ok"] = all(report[k]["ok"] for k in ("registry", "tests", "doc"))
    (HERE / "union_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )

    reg, tst, doc = report["registry"], report["tests"], report["doc"]
    print(f"R registry : main {reg['main_entries']} / branch {reg['branch_claims']} "
          f"/ merged {reg['merged_entries']}  lost(main)={len(reg['main_lost'])} "
          f"lost(branch)={len(reg['branch_lost'])} "
          f"worst_branch_dev={reg['worst_branch_deviation']['rel']:.2e}")
    print(f"T tests    : main {tst['main_tests']} / branch {tst['branch_tests']} "
          f"/ merged {tst['merged_tests']}  missing(main)={tst['main_missing']} "
          f"missing(branch)={tst['branch_missing']} renamed={tst['renamed']}")
    print(f"D doc      : added(main)={doc['main_added_lines']} "
          f"added(branch)={doc['branch_added_lines']} "
          f"lost(main)={len(doc['main_missing'])} lost(branch)={len(doc['branch_missing'])} "
          f"superseded={len(doc['main_superseded'])+len(doc['branch_superseded'])} "
          f"corrected={doc['corrected_cells_present']}")
    print(f"UNION: {'PASS' if report['ok'] else 'FAIL'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())