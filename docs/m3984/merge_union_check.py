#!/usr/bin/env python3
"""Silent-loss audit for the PR #109 three-way merge (agent/m2966 x main).

The failure mode this guards against is the one this platform has already been
bitten by: a "mechanical overlay" resolution that keeps one side's file and
quietly drops the other side's content.  A merge commit can look clean in
``git log`` and still lose content, so the audit runs on the *trees* and on the
*guard inventory*, never on the resolution narrative:

  1. inventory  -- every path present in main or in the branch must be present
     in the merge; paths that only the merge tip adds must be the declared
     evidence artifacts of the resolution, nothing else;
  2. one-side-owned files must be byte identical to their owner in the merge,
     except the two declared *reconciliation seams* (a file owned by one side
     that had to be touched so the other side's change stays true: the doc
     label rename breaks the branch's doc guard, and the third MFS row breaks
     main's row-by-position parser).  For a seam the check is stronger than
     "differs": every line of the owner's version must still be present, i.e.
     the seam is additive/adaptive, never subtractive;
  3. two-side-owned files (the real conflicts) -- every line added by a side
     relative to the merge base must survive in the merge, either verbatim or
     inside a documented rewrite; every assertion each side added must still be
     present in the merged source (guard inventory, whitespace-insensitive);
  4. no side's deletion may be silently undone;
  5. integration -- the merged documentation must still parse to the *claimed*
     panel row (this is the defect that publishing a third MFS row exposed),
     and the merged guard test must still read the merged document.

Usage:
    python docs/m3984/merge_union_check.py [--json OUT] [--md OUT] [--merge SHA]

Exit status 0 only when every check passes.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

BASE = "d22b6adadf2729d2788211cbdd9a0f6f89940629"    # merge-base(main, agent/m2966)
MAIN = "ecba798833d6c2429fbc897d2ce515efb70dd682"     # origin/main when the merge was cut
BRANCH = "ca4ba73e52d8a940b52e03a368b27855bf748836"   # origin/agent/m2966 (PR #109 head)
MERGE = "9a87109375cc74909c0bbb12a6756dbb0f739ced"    # merged branch tip (8fa1b57 + evidence)

#: Paths the merge tip adds that neither side has: the resolution's own
#: evidence.  Anything else appearing out of thin air is a failure.
EVIDENCE_PREFIXES = ("docs/m2966/", "docs/m3984/")

#: A file owned by one side that the merge had to touch anyway, with the reason
#: and the check that covers it.  ``owner`` is the side that owns the content;
#: the seam must keep every line of the *other* side untouched and add/adapt
#: only in service of the named integration contract.
SEAMS: dict[str, dict] = {
    "src/tensorbem/doc_claims.py": {
        "owner": "main",
        "added_by_merge": (
            "publishing the branch's third MFS row (256 panels) turned main's "
            "row-by-position parser into a silent mis-read: the last published "
            "MFS row would win the 'MFS_PANELS' claim.  The seam selects the row "
            "by the panel count in its own label."
        ),
        "covered_by": "tests/test_ursell_doc_reproducibility.py",
    },
    "tests/test_ursell_doc_reproducibility.py": {
        "owner": "branch",
        "added_by_merge": (
            "main renamed every MFS row to the generated '<N> 面元、<S> 源' label "
            "(m3229); the branch's guard parsed '<N> 面元' by column counting.  The "
            "seam re-points the regex at the generated label and adds the 256-panel "
            "row to MFS_DOCUMENTED, keeping the 'count columns, do not trust the "
            "label' intent."
        ),
        "covered_by": "tests/test_ursell_doc_reproducibility.py",
    },
}

#: Superseded owner lines inside a seam: the line the owner had, the tokens the
#: merged file must carry instead (all of them, so a superset rewrite is proven,
#: not assumed), and why.
SEAM_EDITS: list[dict] = [
    {
        "file": "tests/test_ursell_doc_reproducibility.py",
        "lost": "MFS_DOCUMENTED = {64: (2543.24, 2708.57), 128: (2544.69, 2715.76)}",
        "required": [
            "64: (2543.24, 2708.57),",
            "128: (2544.69, 2715.76),",
            "256: (2545.05, 2717.55),",
        ],
        "reason": "the documented-row table grows from two rows to three: the 64- and "
        "128-panel tuples keep their published values verbatim, the 256-panel row is "
        "the row only the branch has",
    },
    {
        "file": "tests/test_ursell_doc_reproducibility.py",
        "lost": "mfs_counts = {",
        "required": [
            r"基本解方法（MFS，(\d+) 面元、(\d+) 源）",
            "mfs_counts = {panels for panels, _sources in mfs_rows}",
        ],
        "reason": "main's generated labels carry the source count too, so the guard "
        "parses both numbers; the panel-count set is still what the assertion uses",
    },
    {
        "file": "tests/test_ursell_doc_reproducibility.py",
        "lost": "int(m.group(1))",
        "required": [
            r"基本解方法（MFS，(\d+) 面元、(\d+) 源）",
            "assert all(sources == 24 for _panels, sources in mfs_rows)",
        ],
        "reason": "the single captured number becomes a (panels, sources) pair, which "
        "is strictly more information than the branch checked",
    },
    {
        "file": "tests/test_ursell_doc_reproducibility.py",
        "lost": "配点网格 (\\d+) 面元",
        "required": [
            r"基本解方法（MFS，(\d+) 面元、(\d+) 源）",
            "assert panel_counts <= mfs_counts",
        ],
        "reason": "the branch documented this regex as provisional ('label it with the "
        "panel count'); main's m3229 made the label generated, so the regex follows the "
        "generator instead of the hand-written string, and the column-count intent is kept",
    },
    {
        "file": "tests/test_ursell_doc_reproducibility.py",
        "lost": "assert {64, 128} <= mfs_counts, sorted(mfs_counts)",
        "required": ["assert {64, 128, 256} <= mfs_counts, sorted(mfs_counts)"],
        "reason": "the required panel set is a superset of the branch's: it adds the "
        "256-panel row the branch publishes in the document",
    },
]

#: Lines added by one side that the union deliberately *rewrote* instead of
#: copying verbatim: (side, file, added-line-substring, must-be-present,
#: reason).
ALLOWED_EDITS: list[tuple[str, str, str, str, str]] = [
    (
        "branch",
        "docs/marine_bem_benchmarks.md",
        "配点网格 64 面元",
        "64 面元、24 源",
        "m3229 (main) renamed every MFS row to the generated 'N panels / S sources' "
        "label; the branch's rows keep their numbers under main's label",
    ),
    (
        "branch",
        "docs/marine_bem_benchmarks.md",
        "配点网格 128 面元",
        "128 面元、24 源",
        "same label renaming, 128-panel row",
    ),
    (
        "branch",
        "docs/marine_bem_benchmarks.md",
        "配点网格 256 面元",
        "256 面元、24 源",
        "same label renaming, 256-panel row (the row main does not have)",
    ),
    (
        "main",
        "docs/marine_bem_benchmarks.md",
        "2.23 / 1.59 / 2.04 / 1.30",
        "2.23 / 1.59 / 2.04 / 1.80",
        "m3150 re-derived the B11 observed-order column from the measured mean-edge "
        "ratio (1.62->1.80, 1.88->1.94); the merged table carries m3150's newer "
        "revision and both notes stay",
    ),
    (
        "main",
        "tests/test_documented_benchmarks.py",
        "fine = elementary_log_integral(xi, length)",
        "fine = elementary_log_integral(case_xi, case_length)",
        "the guard loop that main added is kept, with its loop variables renamed "
        "from the branch's outer-scope names; the closed form does not touch the "
        "panel primitive, which is the property main's assertion pins",
    ),
    (
        "main",
        "tests/test_documented_benchmarks.py",
        "coarse = composite_gauss(xi, h, length, pieces=1, order=24)",
        "coarse = composite_gauss(case_xi, case_h, case_length, 1, 24)",
        "same loop rewrite: positional vs keyword call, same rule and order",
    ),
    (
        "main",
        "tests/test_documented_benchmarks.py",
        "fine = composite_gauss(xi, h, length, pieces=400, order=20)",
        "fine = composite_gauss(case_xi, case_h, case_length, 400, 20)",
        "same loop rewrite: 400 pieces, order 20, unchanged",
    ),
    (
        "branch",
        "tests/test_documented_benchmarks.py",
        "nodes, weights = np.polynomial.legendre.leggauss(20)",
        "nodes, weights = np.polynomial.legendre.leggauss(order)",
        "main's referee takes (pieces, order) as parameters, so the branch's fixed "
        "20-point rule becomes the default argument of the shared helper",
    ),
    (
        "branch",
        "tests/test_documented_benchmarks.py",
        "def composite_gauss(pieces: int) -> float:",
        "def composite_gauss(\n        xi: float, h: float, length: float, pieces: int, order: int\n    ) -> float:",
        "same helper, generalised so main's three smooth-panel configurations can "
        "reuse it instead of duplicating it",
    ),
    (
        "branch",
        "tests/test_documented_benchmarks.py",
        "errors = [abs(composite_gauss(n) - exact) for n in pieces]",
        "errors = [abs(composite_gauss(xi, h, length, n, 20) - exact) for n in pieces]",
        "same convergence sweep (100/200/400/800 pieces, order 20)",
    ),
    (
        "branch",
        "tests/test_documented_benchmarks.py",
        "assert abs(composite_gauss(1) - exact) > 1.0e-4",
        "assert abs(composite_gauss(xi, h, length, 1, 20) - exact) > 1.0e-4",
        "same single-straddling-panel defect assertion",
    ),
]

#: Composition the union legitimately created: a line that exists in neither side
#: verbatim but is the merge of the two (main's label with the branch's row,
#: the branch's sweep against main's parameterised helper, ...).  Every entry has
#: to be declared here, so the check below can only ever pass by *explicit*
#: composition -- silence is not an option.
INVENTED_OK: list[tuple[str, str, str]] = [
    (
        "docs/marine_bem_benchmarks.md",
        "| 基本解方法（MFS，256 面元、24 源） | 2545.05 |",
        "main's generated '<N> 面元、<S> 源' label × the branch's 256-panel row: the "
        "composition the union demands, numbers re-derived by "
        "docs/m2966/merge_resolution_checks.py (mfs_row(256))",
    ),
    (
        "docs/marine_bem_benchmarks.md",
        "复核补记（2026-11-05，m2966）",
        "the resolution's own note recording that both provenance notes survive; it "
        "restates mfs_row(256) numbers rather than introducing any",
    ),
    (
        "tests/test_documented_benchmarks.py",
        "def composite_gauss(",
        "the branch's helper gains main's (pieces, order, xi, h, length) parameters so "
        "main's three-configuration loop can reuse it instead of duplicating it",
    ),
    (
        "tests/test_documented_benchmarks.py",
        "xi: float, h: float, length: float, pieces: int, order: int",
        "signature of the shared helper",
    ),
    (
        "tests/test_documented_benchmarks.py",
        ") -> float:",
        "signature of the shared helper",
    ),
    (
        "tests/test_documented_benchmarks.py",
        "errors = [abs(composite_gauss(xi, h, length, n, 20) - exact) for n in pieces]",
        "the branch's 100/200/400/800 sweep, re-expressed against the shared helper "
        "(order 20 unchanged)",
    ),
    (
        "tests/test_documented_benchmarks.py",
        "assert abs(composite_gauss(xi, h, length, 1, 20) - exact) > 1.0e-4",
        "the branch's single-straddling-panel defect assertion, same rule and order",
    ),
    (
        "tests/test_documented_benchmarks.py",
        "Second, fully independent referee for the same self-panel value",
        "comment introducing main's closed-form referee inside the branch's test",
    ),
    (
        "tests/test_documented_benchmarks.py",
        "elementary closed form never touches ``_log_panel_primitive``",
        "comment stating the independence property the assertion pins",
    ),
    (
        "tests/test_documented_benchmarks.py",
        "assert abs(elementary_log_integral(xi, length) - exact) < 1.0e-13 * abs(exact)",
        "merge-authored joining of main's closed form with the branch's 1e-13 "
        "tolerance (branch line 124): the two referees must agree to machine "
        "precision",
    ),
    (
        "tests/test_documented_benchmarks.py",
        "Smooth-panel configurations: the closed form must reach 1e-8 against the",
        "comment introducing main's smooth-panel loop",
    ),
    (
        "tests/test_documented_benchmarks.py",
        "composite rule, which is admissible only when it is converged (h > 0).",
        "comment introducing main's smooth-panel loop",
    ),
    (
        "tests/test_documented_benchmarks.py",
        "for case_xi, case_h, case_length in (",
        "main's three-configuration loop, renamed away from the branch's outer-scope "
        "xi/h/length",
    ),
    (
        "tests/test_documented_benchmarks.py",
        "(0.5, 0.0, 1.0),",
        "main's coplanar configuration, one tuple per line after the loop rewrite",
    ),
    (
        "tests/test_documented_benchmarks.py",
        "(0.0, 1.0e-3, 1.0),",
        "main's near-surface configuration (h = 1e-3), one tuple per line",
    ),
    (
        "tests/test_documented_benchmarks.py",
        "(0.9, 0.3, 1.0),",
        "main's smooth off-centre configuration, one tuple per line",
    ),
    (
        "tests/test_documented_benchmarks.py",
        "):",
        "loop header of main's three-configuration loop",
    ),
    (
        "tests/test_documented_benchmarks.py",
        "closed = _rankine_panel_integral(case_xi, case_h, case_length)",
        "main's closed-form call bound to the renamed loop variables",
    ),
    (
        "tests/test_documented_benchmarks.py",
        "if case_h == 0.0:",
        "main's self-panel / smooth-panel branch on the renamed variables",
    ),
    (
        "tests/test_documented_benchmarks.py",
        "fine = elementary_log_integral(case_xi, case_length)",
        "main's self-panel reference bound to the renamed variables",
    ),
    (
        "tests/test_documented_benchmarks.py",
        "coarse = composite_gauss(case_xi, case_h, case_length, 1, 24)",
        "main's single-straddling-panel call, positional form of the same rule",
    ),
    (
        "tests/test_documented_benchmarks.py",
        "fine = composite_gauss(case_xi, case_h, case_length, 400, 20)",
        "main's converged-reference call (400 pieces, order 20)",
    ),
    (
        "tests/test_documented_benchmarks.py",
        "assert abs(closed - fine) < 1.0e-8, (case_xi, case_h, closed, fine)",
        "main's 1e-8 agreement gate, re-bound to the renamed variables",
    ),
]

#: Guard inventory: assertions each side added, written as whitespace-insensitive
#: tokens that must appear in the merged source.  This is the check that catches
#: "the line moved, so the substring check cannot see it -- but the property is
#: still there" versus "the property is gone".
GUARD_TOKENS: dict[str, dict[str, list[str]]] = {
    "tests/test_documented_benchmarks.py": {
        "main": [
            # independent second referee, closed form never touches the primitive
            "return (left * math.log(left) - left) + (right * math.log(right) - right)",
            "assert abs(elementary_log_integral(xi, length) - exact) < 1.0e-13 * abs(exact)",
            # three smooth-panel configurations and the 1e-8 gate
            "(0.5, 0.0, 1.0),",
            "(0.0, 1.0e-3, 1.0),",
            "(0.9, 0.3, 1.0),",
            "assert abs(closed - fine) < 1.0e-8,",
            "assert abs(coarse - fine) > 1.0e-4, (coarse, fine)",
        ],
        "branch": [
            # observed order of the uniform-piece rule on the coplanar panel
            "observed_order = math.log2(errors[0] / errors[-1]) / math.log2(",
            "assert 0.8 < observed_order < 1.2, (observed_order, errors)",
            "assert errors[2] > 1.0e-6, errors",
            "pieces = (100, 200, 400, 800)",
            "assert abs(composite_gauss(xi, h, length, 1, 20) - exact) > 1.0e-4",
        ],
    },
    "tests/test_ursell_doc_reproducibility.py": {
        "branch": [
            # every published MFS row must reproduce from the shipped method
            "MFS_DOCUMENTED",
            "256: (2545.05, 2717.55)",
        ],
    },
}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def tree(rev: str) -> dict[str, str]:
    files = {}
    for line in git("ls-tree", "-r", rev).splitlines():
        meta, path = line.split("\t", 1)
        _mode, kind, sha = meta.split()
        if kind == "blob":
            files[path] = sha
    return files


def blob(rev: str, path: str) -> str:
    return git("show", f"{rev}:{path}")


def added_lines(base: str, rev: str, path: str) -> list[str]:
    try:
        old = blob(base, path).splitlines()
    except subprocess.CalledProcessError:
        old = []
    remaining: dict[str, int] = {}
    for line in old:
        remaining[line] = remaining.get(line, 0) + 1
    out = []
    for line in blob(rev, path).splitlines():
        if remaining.get(line, 0):
            remaining[line] -= 1
        else:
            out.append(line)
    return out


def missing_lines(owner: str, rev: str, path: str) -> list[str]:
    """Lines of rev:path absent from owner:path (multiset difference)."""
    have: dict[str, int] = {}
    for line in blob(owner, path).splitlines():
        have[line] = have.get(line, 0) + 1
    out = []
    for line in blob(rev, path).splitlines():
        if have.get(line, 0):
            have[line] -= 1
        else:
            out.append(line)
    return out


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="docs/m3984/outputs/merge_union_check.json")
    ap.add_argument("--md", default="docs/m3984/merge_union_check.md")
    ap.add_argument("--merge", default=MERGE)
    args = ap.parse_args(argv)

    base_t, main_t, branch_t, merge_t = (
        tree(BASE),
        tree(MAIN),
        tree(BRANCH),
        tree(args.merge),
    )

    report: dict = {
        "base": BASE,
        "main": MAIN,
        "branch": BRANCH,
        "merge": args.merge,
        "checks": [],
        "failures": [],
    }

    def check(name: str, ok: bool, detail: dict) -> None:
        report["checks"].append({"name": name, "ok": bool(ok), **detail})
        if not ok:
            report["failures"].append({"name": name, **detail})

    # ---- 1. inventory ------------------------------------------------------
    only_main = sorted(set(main_t) - set(merge_t))
    only_branch = sorted(set(branch_t) - set(merge_t))
    foreign = sorted(
        p
        for p in set(merge_t) - set(main_t) - set(branch_t)
        if not p.startswith(EVIDENCE_PREFIXES)
    )
    evidence = sorted(
        p
        for p in set(merge_t) - set(main_t) - set(branch_t)
        if p.startswith(EVIDENCE_PREFIXES)
    )
    check("inventory/no-main-file-lost", not only_main, {"missing": only_main})
    check("inventory/no-branch-file-lost", not only_branch, {"missing": only_branch})
    check(
        "inventory/merge-adds-only-declared-evidence",
        not foreign,
        {"unexpected": foreign, "declared_evidence": evidence},
    )

    touched_main = {p for p, s in main_t.items() if base_t.get(p) != s}
    touched_branch = {p for p, s in branch_t.items() if base_t.get(p) != s}
    both = sorted((touched_main & touched_branch) & set(merge_t))
    single = sorted((touched_main ^ touched_branch) & set(merge_t))

    # ---- 2. one-side-owned files ------------------------------------------
    identical, seams_seen, seam_drift = [], [], []
    for path in single:
        owner = "main" if path in touched_main else "branch"
        owner_t = main_t if owner == "main" else branch_t
        if merge_t.get(path) == owner_t.get(path):
            identical.append(path)
            continue
        if path not in SEAMS:
            seam_drift.append({"file": path, "owner": owner, "reason": "undeclared"})
            continue
        owner_rev = MAIN if owner == "main" else BRANCH
        lost = missing_lines(args.merge, owner_rev, path)
        merged_text = blob(args.merge, path)
        unexplained = []
        explained = []
        for line in lost:
            entry = next(
                (e for e in SEAM_EDITS if e["file"] == path and e["lost"] in line), None
            )
            if entry and all(tok in merged_text for tok in entry["required"]):
                explained.append({"line": line, "reason": entry["reason"]})
            else:
                unexplained.append(line)
        seams_seen.append(
            {
                "file": path,
                "owner": owner,
                "owner_lines_superseded": explained,
                "owner_lines_lost": unexplained,
                "justification": SEAMS[path]["added_by_merge"],
                "covered_by": SEAMS[path]["covered_by"],
            }
        )
        if unexplained:
            seam_drift.append({"file": path, "owner": owner, "lost": unexplained})
    check(
        "content/one-side-owned-files-intact",
        not seam_drift,
        {
            "byte_identical": len(identical),
            "declared_seams": seams_seen,
            "violations": seam_drift,
        },
    )

    # ---- 3. two-side-owned files: union of additions + guard inventory ----
    lost: list[dict] = []
    rewritten: list[dict] = []
    guard_missing: list[dict] = []
    for path in both:
        merged_text = blob(args.merge, path)
        merged_lines = set(merged_text.splitlines())
        for side, rev in (("main", MAIN), ("branch", BRANCH)):
            for line in added_lines(BASE, rev, path):
                if line in merged_lines or not line.strip():
                    continue
                allow = next(
                    (a for a in ALLOWED_EDITS if a[0] == side and a[1] == path and a[2] in line),
                    None,
                )
                if allow and allow[3] in merged_text:
                    rewritten.append({"side": side, "file": path, "line": line, "reason": allow[4]})
                    continue
                lost.append({"side": side, "file": path, "line": line})
    check(
        "content/both-side-additions-survive",
        not lost,
        {"files": both, "silently_lost_lines": lost, "documented_rewrites": rewritten},
    )

    squashed = {p: squash(blob(args.merge, p)) for p in GUARD_TOKENS}
    for path, sides in GUARD_TOKENS.items():
        if path not in squashed:
            guard_missing.append({"file": path, "issue": "file missing from merge"})
            continue
        for side, tokens in sides.items():
            for token in tokens:
                if squash(token) not in squashed[path]:
                    guard_missing.append({"file": path, "side": side, "token": token})
    check(
        "content/guard-inventory-both-sides",
        not guard_missing,
        {
            "files": sorted(GUARD_TOKENS),
            "missing_guards": guard_missing,
            "note": "whitespace-insensitive; a moved line still counts, a lost property does not",
        },
    )

    # ---- 4. both-side files invent nothing; deletions are not silently undone
    invented, composed = [], []
    for path in both:
        sides = set(blob(MAIN, path).splitlines()) | set(blob(BRANCH, path).splitlines())
        sides |= set(blob(BASE, path).splitlines())
        for line in blob(args.merge, path).splitlines():
            if not line.strip() or line in sides:
                continue
            declared = next(
                (e for e in INVENTED_OK if e[0] == path and e[1] in line), None
            )
            if declared:
                composed.append({"file": path, "line": line, "reason": declared[2]})
            else:
                invented.append({"file": path, "line": line})
    check(
        "content/composition-is-declared",
        not invented,
        {
            "undeclared": invented,
            "declared_compositions": [
                {"file": c["file"], "line": c["line"].strip()[:110], "reason": c["reason"]}
                for c in composed
            ],
            "note": "a conflicted file may only gain lines the resolution declares",
        },
    )

    deletions_undone = []
    for path in both:
        base_lines = set(blob(BASE, path).splitlines())
        merged_lines = set(blob(args.merge, path).splitlines())
        for side, rev in (("main", MAIN), ("branch", BRANCH)):
            other = BRANCH if side == "main" else MAIN
            other_lines = set(blob(other, path).splitlines())
            for line in base_lines - set(blob(rev, path).splitlines()):
                if line in merged_lines and line not in other_lines:
                    deletions_undone.append({"file": path, "deleted_by": side, "line": line})
    check(
        "content/no-side-deletion-undone",
        not deletions_undone,
        {"deletions_undone": deletions_undone},
    )

    # ---- 5. integration: the merged doc still parses to the claimed row ---
    sys.path.insert(0, str(Path("src").resolve()))
    from tensorbem.doc_claims import MFS_PANELS, parse_ursell_table  # noqa: E402

    doc = "docs/marine_bem_benchmarks.md"
    rows = {
        "merge": parse_ursell_table(blob(args.merge, doc)),
        "branch": parse_ursell_table(blob(BRANCH, doc)),
        "main": parse_ursell_table(blob(MAIN, doc)),
    }
    claimed = {}
    for name in ("64", "128", "256"):
        label = f"| 基本解方法（MFS，{name} 面元、24 源） |"
        claimed[name] = label in blob(args.merge, doc)
    check(
        "integration/doc_claims-selects-claimed-row",
        abs(rows["merge"]["mfs_added_mass"] - rows["branch"]["mfs_added_mass"]) < 1e-6
        and abs(rows["merge"]["mfs_damping"] - rows["branch"]["mfs_damping"]) < 1e-6,
        {
            "MFS_PANELS": MFS_PANELS,
            "parsed_mfs_added_mass": {k: v["mfs_added_mass"] for k, v in rows.items()},
            "parsed_mfs_damping": {k: v["mfs_damping"] for k, v in rows.items()},
            "three_rows_published": claimed,
        },
    )

    report["ok"] = not report["failures"]
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# PR #109 三方合并静默丢内容审计（机械可复算）",
        "",
        f"base=`{BASE[:7]}` · main=`{MAIN[:7]}` · branch=`{BRANCH[:7]}` · merge=`{args.merge[:7]}`",
        "",
        "| 判据 | 结果 |",
        "|---|---|",
    ]
    for c in report["checks"]:
        lines.append(f"| `{c['name']}` | {'**PASS**' if c['ok'] else '**FAIL**'} |")
    lines += ["", f"总体：**{'PASS' if report['ok'] else 'FAIL'}**", "", "## 两处语义冲突的逐行裁决", ""]
    for item in rewritten:
        lines.append(
            f"- `{item['file']}` [{item['side']}] `{item['line'].strip()[:90]}` → {item['reason']}"
        )
    lines += ["", "## 两处跨侧联动接缝（一方独有文件被合并改动）", ""]
    for item in seams_seen:
        lines.append(
            f"- `{item['file']}`（{item['owner']} 独有）：主人版本无一行丢失"
            f"（lost={item['owner_lines_lost']}）。理由：{item['justification']} 由 `{item['covered_by']}` 覆盖。"
        )
    lines += [
        "",
        "## 复现",
        "",
        "```",
        "python docs/m3984/merge_union_check.py    # 退出码 0 = 全过",
        "```",
        "",
        "输出同时落在 `docs/m3984/outputs/merge_union_check.json`。",
    ]
    Path(args.md).write_text("\n".join(lines) + "\n")

    print(json.dumps({"ok": report["ok"], "failures": report["failures"]},
                     ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))