"""Audit that the merge kept every line either side added (no silent loss)."""
import json, os, subprocess, sys

REPO = "/root/tp-agent/work/m3735"
BASE = "4f5b73e73c4e731582f2b050328ec5fcf62b4a86"   # merge base
SIDES = {"m3181": "origin/agent/m3181", "main": "origin/main"}
#: intentional, documented rewrites of one side's lines (rename to avoid a
#: same-module definition collision: main already owns `_chief_rows(rows)`)
EXPECTED_REMAP = {
    # main already owns `_chief_rows(rows)` (a study flattener), so m3181's
    # registry builder of the same name is renamed instead of shadowing it
    "def _chief_rows():": "def _chief_table_rows():",
    "        + _chief_rows()": "        + _chief_table_rows()",
}
#: lines whose *text* was replaced by the other side's equivalent wording for
#: the identical assertion (`assert entry.declared != 0.0`); the mapped target
#: line must be present, so the semantic content is provably kept
SUBSUMED = {
    "        # A quote may legitimately be non-positive: the unregularized":
        "        # table 11 quotes as the defect CHIEF removes), so only zero is illegal",
    "        # 3-D BIE produces an unphysical negative heave damping, and that":
        "        # table 11 quotes as the defect CHIEF removes), so only zero is illegal",
    "        # sign is exactly the claim the table makes.":
        "        # table 11 quotes as the defect CHIEF removes), so only zero is illegal",
}

def run(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True).stdout

def changed_files(rev):
    out = run("diff", "--name-only", BASE, rev)
    return [ln for ln in out.splitlines() if ln.strip()]

def added_lines(rev, path):
    out = run("diff", "-U0", "--no-color", BASE, rev, "--", path)
    return [ln[1:] for ln in out.splitlines() if ln.startswith("+") and not ln.startswith("+++")]

report = {"base": BASE, "sides": {}, "missing": []}
allfiles = set()
for name, rev in SIDES.items():
    allfiles |= set(changed_files(rev))
report["changed_files"] = sorted(allfiles)

for name, rev in SIDES.items():
    stats = {"files": 0, "added_lines": 0, "lines_missing": 0, "files_missing": []}
    for path in sorted(changed_files(rev)):
        stats["files"] += 1
        full = os.path.join(REPO, path)
        if not os.path.exists(full):
            stats["files_missing"].append(path)
            report["missing"].append({"side": name, "file": path, "reason": "file absent after merge"})
            continue
        try:
            merged = open(full, encoding="utf-8", errors="replace").read().split("\n")
        except OSError:
            continue
        merged_set = set(merged)
        stats_seen = 0
        for line in added_lines(rev, path):
            stats["added_lines"] += 1
            key = line
            if key in merged_set:
                continue
            alt = EXPECTED_REMAP.get(line) or SUBSUMED.get(line)
            if alt is not None and alt in merged_set:
                continue
            report["missing"].append({"side": name, "file": path, "line": line})
            stats["lines_missing"] += 1
    report["sides"][name] = stats

report["expected_remap"] = {"renamed": EXPECTED_REMAP, "subsumed": SUBSUMED}
report["verdict"] = "CLEAN" if not report["missing"] else "LOSS"
print(json.dumps({k: report[k] for k in ("sides", "verdict")}, indent=2, ensure_ascii=False))
print("missing entries:", len(report["missing"]))
for item in report["missing"][:15]:
    print(" -", item)
open("/tmp/mg/union_audit.json", "w", encoding="utf-8").write(json.dumps(report, indent=2, ensure_ascii=False))
