"""Three-way union resolution for PR #139 (agent/m3306 <- main), v2.

Why not resolve the git conflict markers in place: the two sides appended
parallel `DeclaredNumber(...)` blocks at the *same* anchors, so git's hunk
boundaries land inside an entry (head in one hunk, `declared=`/closing paren in
the common text).  A marker-driven overlay therefore truncates entries.

This script instead composes the union from the three revisions, with the two
content sets treated as first-class data:

    base   = 4f5b73e          (merge base, main before either side's work)
    ours   = 36ea849          (agent/m3306)
    theirs = origin/main      (ecba798)

For every conflicted file the rule is: start from the side that is the *target*
of the merge, then re-insert everything the other side added relative to base,
and where both sides changed the same thing, adjudicate with a measurement and
log the adjudication.  Nothing is dropped silently: `audit_union.py` re-derives
the entry/definition sets from the merged files and checks them against both
sides.
"""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE, OURS, THEIRS = "4f5b73e", "36ea849", "origin/main"
LOG: list[dict] = []


def show(rev: str, path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def registry_entries(src: str) -> dict[str, str]:
    """Map key -> source text of every DeclaredNumber(...) in DECLARED_NUMBERS."""
    tree = ast.parse(src)
    entries: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "DECLARED_NUMBERS":
            for elt in node.value.elts:
                key = None
                for kw in elt.keywords:
                    if kw.arg == "key":
                        key = ast.literal_eval(kw.value)
                assert key is not None, "entry without key"
                assert key not in entries, f"duplicate key {key}"
                entries[key] = ast.get_source_segment(src, elt) + ","
    return entries


def top_level_defs(src: str) -> dict[str, str]:
    """Map top-level name -> source text for functions/fixtures."""
    tree = ast.parse(src)
    lines = src.split("\n")
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            out[node.name] = "\n".join(lines[start : node.end_lineno])
    return out


# ------------------------------------------------------- declared_numbers.py --
REG = "src/tensorbem/benchmarks/declared_numbers.py"
base_entries = registry_entries(show(BASE, REG))
our_entries = registry_entries(show(OURS, REG))
their_entries = registry_entries(show(THEIRS, REG))
assert set(base_entries) <= set(our_entries)
assert set(base_entries) <= set(their_entries)
new_ours = [k for k in our_entries if k not in base_entries]
new_theirs = [k for k in their_entries if k not in base_entries]
assert not (set(new_ours) & set(new_theirs)), "both sides added the same key"

merged = show(THEIRS, REG)
their_tree = ast.parse(merged)
last_entry_end = 0
registry_close = None
for node in ast.walk(their_tree):
    if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "DECLARED_NUMBERS":
        last_entry_end = node.value.elts[-1].end_lineno
        registry_close = node.value.end_lineno  # line of the closing paren

mlines = merged.split("\n")
insert_at = last_entry_end  # 1-based: insert after this line
block = ["", "    # --- m3306 pins (PR #139), merged unchanged -------------------------"]
for key in new_ours:
    block.append(our_entries[key])
mlines[insert_at:insert_at] = block
merged = "\n".join(mlines)

# helper: keep main's _study_rows/_chief_rows and re-add m3306's _finest_level
FINEST = top_level_defs(show(OURS, REG))["_finest_level"]
anchor = "def _run_group("
assert merged.count(anchor) == 1
merged = merged.replace(anchor, FINEST + "\n\n\n" + anchor, 1)

# dispatch: both pin conventions must be served from one solve per setting
OLD_HEMI = '''    if group == "hulme_hemisphere":
        study = hemisphere_benchmark.run_refinement_study(
            subdivisions=tuple(int(n) for n in parameters.get("subdivisions", (1, 2, 3))),
            ka=float(parameters.get("ka", 1.0)),
        )
        return _study_rows(study)
    if group == "submerged_sphere":
        study = sphere_benchmark.run_refinement_study(
            subdivisions=tuple(int(n) for n in parameters.get("subdivisions", (0, 1, 2))),
            ka=float(parameters.get("ka", 0.8)),
            submergence=float(parameters.get("submergence", 1.5)),
        )
        return _study_rows(study)
'''
NEW_HEMI = '''    if group == "hulme_hemisphere":
        study = hemisphere_benchmark.run_refinement_study(
            subdivisions=tuple(int(n) for n in parameters.get("subdivisions", (1, 2, 3))),
            ka=float(parameters.get("ka", 1.0)),
        )
        # Two pin conventions survived the merge: main registers one row per
        # panel level (``sub<level>_<quantity>``), m3306 registers the finest
        # level plus its observed orders.  A single solve feeds both.
        values = _study_rows(study)
        values.update(
            _finest_level(
                study, extra_orders={"observed_order_b11": "surge_damping_rel_error"}
            )
        )
        return values
    if group == "submerged_sphere":
        study = sphere_benchmark.run_refinement_study(
            subdivisions=tuple(int(n) for n in parameters.get("subdivisions", (0, 1, 2))),
            ka=float(parameters.get("ka", 0.8)),
            submergence=float(parameters.get("submergence", 1.5)),
        )
        values = _study_rows(study)
        values.update(_finest_level(study))
        return values
'''
assert merged.count(OLD_HEMI) == 1
merged = merged.replace(OLD_HEMI, NEW_HEMI, 1)

# m3306's irregular-frequency group, re-added next to main's hulme_chief group.
IRREGULAR = '''    if group == "hulme_irregular":
        rows = hemisphere_benchmark.run_irregular_frequency_study(
            subdivision=int(parameters["subdivisions"]),
            ka_values=(float(parameters["ka"]),),
        )
        row = rows[0]
        unreg = row["unregularized"]
        chief = row["chief_regularized"]
        return {
            "condition_number": float(unreg["condition_number"]),
            "added_mass_dimless": float(unreg["a33_dimless"]),
            "damping_dimless": float(unreg["b33_dimless"]),
            # Registered as a magnitude: the registry only quotes positive
            # numbers, and the sign is the whole point of the claim anyway.
            "damping_magnitude": abs(float(unreg["b33_dimless"])),
            "passivity_violated": 1.0 if float(unreg["b33_dimless"]) < 0.0 else 0.0,
            "chief_condition_number": float(chief["condition_number"]),
            "chief_added_mass_dimless": float(chief["a33_dimless"]),
            "chief_damping_dimless": float(chief["b33_dimless"]),
            "chief_point_count": float(chief["chief_point_count"]),
            "exact_added_mass_dimless": float(row["exact_a33"]),
            "exact_damping_dimless": float(row["exact_b33"]),
        }
'''
chief_anchor = '    if group == "hulme_chief":\n'
assert merged.count(chief_anchor) == 1
merged = merged.replace(chief_anchor, IRREGULAR + chief_anchor, 1)

(ROOT / REG).write_text(merged)
LOG.append(
    {
        "file": REG,
        "strategy": (
            "main's file kept as the skeleton (canonical sub<level>_ pin convention, "
            "_study_rows/_chief_rows, hulme_chief handler); m3306's 37 added registry "
            "entries re-inserted unchanged; m3306's _finest_level + hulme_irregular "
            "handler re-added; the two 3-D handlers return the union of both "
            "flattenings so both conventions are recomputed from one solve"
        ),
        "entries_main": len(their_entries),
        "entries_m3306_added": len(new_ours),
        "entries_base": len(base_entries),
    }
)

# ------------------------------------------------------------- tests file ----
TESTS = "tests/test_declared_numbers.py"
base_defs = top_level_defs(show(BASE, TESTS))
our_defs = top_level_defs(show(OURS, TESTS))
their_defs = top_level_defs(show(THEIRS, TESTS))
assert set(base_defs) <= set(our_defs) and set(base_defs) <= set(their_defs)
new_defs = [name for name in our_defs if name not in base_defs]
mdefs = [name for name in their_defs if name not in base_defs]
assert not (set(new_defs) & set(mdefs))

txt = show(THEIRS, TESTS)
# union of the "known groups" tuple in the registry-shape test
OLD_GROUPS = '''        "hulme_chief",
        "hulme_hemisphere",
'''
NEW_GROUPS = '''        "hulme_chief",
        "hulme_hemisphere",
        "hulme_irregular",
'''
assert txt.count(OLD_GROUPS) == 1
txt = txt.replace(OLD_GROUPS, NEW_GROUPS, 1)
# re-insert m3306's tests (they run the standard tier, shared with main's)
txt = txt.rstrip("\n") + "\n\n\n" + "\n\n\n".join(our_defs[name] for name in new_defs) + "\n"
(ROOT / TESTS).write_text(txt)
LOG.append(
    {
        "file": TESTS,
        "strategy": (
            "main's tests kept; m3306's standard-tier fixture and its three "
            "3-D pinning tests re-appended; the known-group tuple of the "
            "registry-shape test widened to the union "
            "{hulme_chief, hulme_hemisphere, hulme_irregular, submerged_sphere}"
        ),
        "tests_readded": new_defs,
    }
)

# -------------------------------------------------------------- docs file ----
DOCS = "docs/marine_bem_benchmarks.md"
txt = show(OURS, DOCS)
# main's three cell corrections under the same h_mean convention; the 256-panel
# row already carries the identical 1.94 in m3306's tree, so only two differ.
FIXES = [
    (
        "| 2.23 / 1.59 / 2.04 / 1.62 / 1.77 |",
        "| 2.23 / 1.59 / 2.04 / 1.80 / 1.77 |",
        "table 10, row 2 (16->64 panels), B11' observed order: m3306 left 1.62 "
        "(implied h ratio 2), main recomputed it with the measured h_mean ratio "
        "1.87 -> 1.80, which is the convention m3306 itself argues for in the "
        "paragraph it added; the merged registry pins main's value",
    ),
    (
        r"| \(2.2\times10^{-4}\) |",
        r"| \(2.15\times10^{-4}\) |",
        "table 12, row 1, rotational coupling: main's reproducible 3-s.f. value",
    ),
    (
        r"| **\(2.6\times10^{-5}\)** |",
        r"| **\(2.65\times10^{-5}\)** |",
        "table 12, row 2, rotational coupling: main's reproducible 3-s.f. value",
    ),
]
for old, new, why in FIXES:
    assert txt.count(old) == 1, old
    txt = txt.replace(old, new, 1)

# main's m3150 paragraph sits before "## 5."; m3306's block ends right there too,
# so insert main's paragraph after m3306's block to keep both narratives.
main_docs = show(THEIRS, DOCS)
para_start = main_docs.index("**表 10 的 \\(B_{11}'\\) 观测阶列与表 12 转动耦合比列（m3150 复算订正）**")
para_end = main_docs.index("\n\n", para_start)
PARA = main_docs[para_start:para_end]
anchor_docs = "<!-- END m3306 declared-number pins -->"
assert txt.count(anchor_docs) == 1
txt = txt.replace(anchor_docs, anchor_docs + "\n\n" + PARA, 1)
(ROOT / DOCS).write_text(txt)
LOG.append(
    {
        "file": DOCS,
        "strategy": (
            "m3306's tree kept as the skeleton (it carries the 1.94 correction and "
            "the whole 'declared-number pins' section); main's three reproducible "
            "cell corrections re-applied; main's m3150 convention paragraph "
            "re-inserted after m3306's block so both narratives survive"
        ),
        "adjudications": [why for _, _, why in FIXES],
    }
)

out = ROOT / "outputs/verif_m3930/merge_resolution_log.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps(
        {
            "merge": f"{OURS} (agent/m3306) <- {THEIRS} (main), base {BASE}",
            "conflict_files": [REG, TESTS, DOCS],
            "files": LOG,
        },
        indent=2,
        ensure_ascii=False,
    )
    + "\n"
)
for entry in LOG:
    print(f"{entry['file']}: {entry['strategy'][:90]}...")
print(f"wrote {out.relative_to(ROOT)}")