"""Asset inventory + wiring scan for the 2026-11-11 TensorBEM quality audit (m14488).

Reads git history (first-parent merge commits ``Merge #N (...)``), classifies
every file added to ``main`` between a base merge and HEAD, and measures, for
every package module, which layer consumes it:

    core      - imported by another ``src/tensorbem`` module outside
                ``benchmarks/`` (i.e. inside the shipped solver/library surface)
    bench     - imported only by ``src/tensorbem/benchmarks`` (study code)
    tool      - imported only by ``scripts/`` / ``tools/`` / ``examples/``
    tests     - only referenced by ``tests/``
    orphan    - no reference outside itself

Usage::

    PYTHONPATH=src python3 docs/audit_m14488/inventory_scan.py \
        --base-ref b17b169 --out docs/audit_m14488/inventory.json

The script is read-only (git plumbing + AST), so it can be re-run by an
independent reviewer to reproduce every count in ``audit_report.md``.
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[2]
MERGE_RE = re.compile(r"^Merge #(\d+) \(([^)]*)\) via")

CATEGORY_RULES = (
    ("test", lambda p: p.startswith("tests/")),
    ("doc", lambda p: p.startswith(("docs/", "papers/")) or p.endswith(".md")),
    ("artifact", lambda p: p.startswith(("artifacts/", "benchmarks/verified/"))
     or p.endswith((".json", ".png", ".csv"))),
    ("core_module", lambda p: p.startswith("src/tensorbem/") and "/benchmarks/" not in p),
    ("benchmark_module", lambda p: p.startswith("src/tensorbem/benchmarks/")),
    ("tool", lambda p: p.startswith(("scripts/", "tools/", "examples/"))),
)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout


def classify(path: str) -> str:
    for name, rule in CATEGORY_RULES:
        if rule(path):
            return name
    return "other"


def added_lines(path: str, sha: str) -> int:
    try:
        return pathlib.Path(REPO, path).read_text(encoding="utf-8", errors="ignore").count("\n")
    except OSError:
        return 0


def scan_merges(base_ref: str) -> list[dict]:
    log = git("log", "--first-parent", "--format=%H%x00%s", f"{base_ref}..HEAD")
    merges = []
    for line in log.splitlines():
        if not line.strip():
            continue
        sha, subject = line.split("\x00")
        m = MERGE_RE.match(subject)
        if not m:
            continue
        pr, mission = int(m.group(1)), m.group(2).strip()
        name_status = git("diff", "--name-status", f"{sha}^1", sha).splitlines()
        files = []
        for row in name_status:
            parts = row.split("\t")
            status, path = parts[0], parts[-1]
            if status.startswith("D"):
                continue
            numstat = git("diff", "--numstat", f"{sha}^1", sha, "--", path).split()
            first = numstat[0] if numstat else "0"
            added = int(first) if first.isdigit() else 0
            files.append(
                {
                    "path": path,
                    "status": status[:1],
                    "category": classify(path),
                    "added": added,
                    "lines_now": added_lines(path, sha),
                }
            )
        merges.append({"pr": pr, "mission": mission, "sha": sha, "files": files})
    return merges


def _imported_leaves(path: pathlib.Path, leaves: set[str]) -> set[str]:
    """Return package-module leaves actually imported by ``path`` (AST-based)."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "tensorbem" and len(parts) > 1 and parts[1] in leaves:
                    found.add(parts[1])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            parts = mod.split(".") if mod else []
            if node.level and not parts:  # ``from . import x``
                for alias in node.names:
                    if alias.name in leaves:
                        found.add(alias.name)
                continue
            if parts and parts[0] == "tensorbem" and len(parts) > 1 and parts[1] in leaves:
                found.add(parts[1])
            if parts == ["tensorbem"]:  # ``from tensorbem import <leaf> as alias``
                for alias in node.names:
                    if alias.name in leaves:
                        found.add(alias.name)
            elif parts and parts[0] in leaves and (node.level or True):
                found.add(parts[0])
            if node.level and parts and parts[0] in leaves:  # ``from .nk_panel3d import x``
                found.add(parts[0])
            if node.level and parts and parts[0] == "benchmarks" and len(parts) > 1:
                pass
    return found


def module_consumers() -> dict[str, dict[str, list[str]]]:
    """Map every ``src/tensorbem/<x>.py`` leaf to the layers that import it."""

    src = REPO / "src" / "tensorbem"
    leaves = {p.stem for p in src.glob("*.py") if p.stem != "__init__"}
    bench = src / "benchmarks"
    bench_leaves = {p.stem for p in bench.glob("*.py")} if bench.is_dir() else set()
    consumers: dict[str, dict[str, list[str]]] = {
        leaf: {"core": [], "bench": [], "tool": [], "tests": [], "orphan": []} for leaf in leaves
    }

    def scan(paths, layer):
        for path in paths:
            for leaf in _imported_leaves(path, leaves | bench_leaves):
                if leaf not in consumers:
                    continue
                consumers[leaf][layer].append(str(path.relative_to(REPO)))

    scan([p for p in src.glob("*.py")], "core")
    scan([p for p in bench.glob("*.py")] if bench.is_dir() else [], "bench")
    tool_files = []
    for sub in ("scripts", "tools", "examples"):
        tool_files.extend((REPO / sub).rglob("*.py"))
    scan(tool_files, "tool")
    scan((REPO / "tests").rglob("*.py"), "tests")

    for leaf, bucket in consumers.items():
        if not any(bucket[k] for k in ("core", "bench", "tool", "tests")):
            bucket["orphan"].append(f"src/tensorbem/{leaf}.py")
    return consumers


def classify_wiring(bucket: dict[str, list[str]]) -> str:
    if bucket["core"]:
        return "core"
    if bucket["bench"]:
        return "bench"
    if bucket["tool"]:
        return "tool"
    if bucket["tests"]:
        return "tests_only"
    return "orphan"


def stub_scan() -> dict[str, list[str]]:
    """AST scan for placeholder bodies (``pass`` / ``...`` / ``raise NotImplementedError``)."""

    from tensorbem import capability_evidence as ce

    findings: dict[str, list[str]] = {}
    root = REPO / "src" / "tensorbem"
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = [n for n in node.body if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Constant)]
                if ce._body_is_stub(node) if hasattr(ce, "_body_is_stub") else False:
                    findings.setdefault(str(path.relative_to(REPO)), []).append(node.name)
                elif len(body) == 1 and isinstance(body[0], ast.Pass):
                    findings.setdefault(str(path.relative_to(REPO)), []).append(node.name)
    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ref", default="b17b169", help="last merge before the audited window")
    ap.add_argument("--out", default="docs/audit_m14488/inventory.json")
    args = ap.parse_args()

    merges = scan_merges(args.base_ref)
    consumers = module_consumers()
    wiring = {leaf: {"wiring": classify_wiring(b), **b} for leaf, b in consumers.items()}
    payload = {
        "base_ref": args.base_ref,
        "head": git("rev-parse", "HEAD").strip(),
        "merges": merges,
        "wiring": wiring,
        "stub_symbols": stub_scan(),
    }
    out = pathlib.Path(REPO, args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    totals: dict[str, int] = {}
    for merge in merges:
        for f in merge["files"]:
            totals[f["category"]] = totals.get(f["category"], 0) + f["added"]
    print(json.dumps({"merges": len(merges), "added_by_category": totals,
                      "orphan_modules": [k for k, v in wiring.items() if v["wiring"] == "orphan"],
                      "stub_symbols": payload["stub_symbols"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())