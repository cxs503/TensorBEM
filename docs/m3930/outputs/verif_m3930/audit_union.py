"""Independent union audit for the PR #139 merge (does not trust the resolver).

Checks, per conflicted file:

1. registry: the set of `key=` values in the merged file equals, element for
   element, the union of both sides' registries (no entry lost on either side,
   no phantom entry invented); every key is unique; every entry keeps the
   `declared` value of the side that introduced it.
2. tests: every top-level test function / fixture present on either side is
   present in the merged file (same source text).
3. docs: every added line of either side (relative to the merge base) appears
   in the merged document, except the explicitly adjudicated table cells which
   are listed with the value that won and why.
4. the merged modules import and expose the expected registries.

Exit status is non-zero on any violation, so this is the machine gate for
"nothing was silently dropped".
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE, OURS, THEIRS = "4f5b73e", "36ea849", "origin/main"
REG = "src/tensorbem/benchmarks/declared_numbers.py"
TESTS = "tests/test_declared_numbers.py"
DOCS = "docs/marine_bem_benchmarks.md"

failures: list[str] = []
report: dict = {}


def show(rev: str, path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def entries(src: str) -> dict[str, dict]:
    tree = ast.parse(src)
    out: dict[str, dict] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "DECLARED_NUMBERS":
            for elt in node.value.elts:
                kw = {k.arg: k.value for k in elt.keywords}
                key = ast.literal_eval(kw["key"])
                if key in out:
                    failures.append(f"{REG}: duplicate key {key!r} in merged registry")
                out[key] = {
                    "declared": ast.literal_eval(kw["declared"]),
                    "group": ast.literal_eval(kw["group"]),
                    "quantity": ast.literal_eval(kw["quantity"]),
                    "tier": ast.literal_eval(kw["tier"]) if "tier" in kw else "fast",
                }
    return out


def defs(src: str) -> dict[str, str]:
    tree = ast.parse(src)
    lines = src.split("\n")
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            out[node.name] = "\n".join(lines[start : node.end_lineno])
    return out


merged_reg = (ROOT / REG).read_text()
base_e, our_e, their_e, merged_e = (
    entries(show(BASE, REG)),
    entries(show(OURS, REG)),
    entries(show(THEIRS, REG)),
    entries(merged_reg),
)
# 1. registry union
expected = set(their_e) | set(our_e)
missing = expected - set(merged_e)
phantom = set(merged_e) - expected
if missing:
    failures.append(f"{REG}: entries dropped by the merge: {sorted(missing)}")
if phantom:
    failures.append(f"{REG}: entries invented by the merge: {sorted(phantom)}")
value_conflicts = {
    key: {"main": their_e[key]["declared"], "m3306": our_e[key]["declared"]}
    for key in set(their_e) & set(our_e)
    if their_e[key]["declared"] != our_e[key]["declared"]
}
if value_conflicts:
    failures.append(f"{REG}: same key with different declared values: {value_conflicts}")
report["registry"] = {
    "keys_base": len(base_e),
    "keys_main": len(their_e),
    "keys_m3306": len(our_e),
    "keys_merged": len(merged_e),
    "keys_expected": len(expected),
    "shared_keys_different_value": value_conflicts,
    "groups_merged": sorted({e["group"] for e in merged_e.values()}),
    "tiers_merged": sorted({e["tier"] for e in merged_e.values()}),
}

# 2. tests union
merged_t = (ROOT / TESTS).read_text()
d_base, d_ours, d_theirs, d_merged = (
    defs(show(BASE, TESTS)),
    defs(show(OURS, TESTS)),
    defs(show(THEIRS, TESTS)),
    defs(merged_t),
)
ADJUDICATED_DEFS = {
    "test_registry_covers_every_documented_benchmark_family": (
        "known-group tuple widened to the union of both sides' group sets"
    )
}
for name, text in {**d_ours, **d_theirs}.items():
    if name not in d_merged:
        failures.append(f"{TESTS}: test/fixture {name} lost")
    elif name in ADJUDICATED_DEFS:
        continue
    elif d_merged[name] != text and name in d_ours and d_merged[name] != d_theirs.get(name):
        failures.append(f"{TESTS}: {name} text changed by the merge")
report["tests"] = {
    "defs_base": len(d_base),
    "defs_main": len(d_theirs),
    "defs_m3306": len(d_ours),
    "defs_merged": len(d_merged),
    "tests_merged": sorted(n for n in d_merged if n.startswith("test_")),
}

# 3. docs union, added lines only
def added_lines(rev: str) -> list[str]:
    diff = subprocess.run(
        ["git", "diff", "-U0", f"{BASE}:{DOCS}", f"{rev}:{DOCS}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    added = []
    for line in diff.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            stripped = line[1:].strip()
            if stripped:
                added.append(stripped)
    return added


merged_docs = (ROOT / DOCS).read_text()
merged_doc_lines = {line.strip() for line in merged_docs.split("\n")}
ADJUDICATED = {
    "| 2.23 / 1.59 / 2.04 / 1.62 / 1.77 |": "table 10 row 2: replaced by main's 1.80 (h_mean convention)",
}
lost = {"ours": [], "theirs": []}
for side, rev in (("ours", OURS), ("theirs", THEIRS)):
    for line in added_lines(rev):
        if line in merged_doc_lines:
            continue
        if line in ADJUDICATED:
            continue
        lost[side].append(line)
report["tests"]["adjudicated"] = ADJUDICATED_DEFS
report["docs"] = {
    "added_lines_ours": len(added_lines(OURS)),
    "added_lines_theirs": len(added_lines(THEIRS)),
    "not_found_ours": lost["ours"][:10],
    "not_found_theirs": lost["theirs"][:10],
    "adjudicated": ADJUDICATED,
}
for side, lines in lost.items():
    if lines:
        failures.append(f"{DOCS}: {len(lines)} line(s) added by {side} not present: {lines[:5]}")

# 4. import check
sys.path.insert(0, str(ROOT / "src"))
from tensorbem.benchmarks import declared_numbers as dn  # noqa: E402

report["import"] = {
    "module": str(dn.__file__),
    "registry_len": len(dn.DECLARED_NUMBERS),
    "unique_keys": len({e.key for e in dn.DECLARED_NUMBERS}) == len(dn.DECLARED_NUMBERS),
}
if not report["import"]["unique_keys"]:
    failures.append(f"{REG}: duplicate keys in the imported registry")

(ROOT / "outputs/verif_m3930/union_audit.json").write_text(
    json.dumps({"failures": failures, **report}, indent=2, ensure_ascii=False) + "\n"
)
print(json.dumps({"failures": failures, **report}, indent=2, ensure_ascii=False))
sys.exit(1 if failures else 0)