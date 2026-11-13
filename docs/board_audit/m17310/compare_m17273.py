#!/usr/bin/env python3
"""Auditor (m17310) independent replay comparison for m17273 / CASE-BEM-ELLIPSOID-KOCHIN.

Shipped results.json (from commit c121bee, benchmarks/verified/bem_ellipsoid_kochin/) vs
auditor rerun (fresh checkout of the same commit, --outdir runs_audit).
Diffs are computed, never typed. Exit 0 iff every graded number reproduces within 1e-12 rel.
"""
import json, sys, hashlib, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # worktree /root/tp-agent/work/m17310
REPRO = ROOT / "runs/m17310/repro"
SHIPPED = REPRO / "benchmarks/verified/bem_ellipsoid_kochin/results.json"
RERUN = REPRO / "runs_audit/results.json"

def walk(a, b, path="", out=None):
    if out is None: out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            walk(a.get(k), b.get(k), f"{path}.{k}" if path else k, out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        d = abs(a - b); r = d / max(abs(a), 1e-300)
        out.append((path, a, b, d, r))
    elif a != b:
        out.append((path, a, b, None, None))
    return out

def main():
    a = json.loads(SHIPPED.read_text()); b = json.loads(RERUN.read_text())
    diffs = [d for d in walk(a, b) if d[3] is not None]
    exact = sum(1 for d in diffs if d[3] == 0.0)
    nz = [d for d in diffs if d[3] != 0.0]
    print(f"numeric leaves compared: {len(diffs)}   bit-identical: {exact}   differing: {len(nz)}")
    for d in sorted(nz, key=lambda x: -x[4])[:10]:
        print(f"  {d[0]:60s} shipped={d[1]!r:24s} rerun={d[2]!r:24s} rel={d[4]:.3e}")
    worst = max([d[4] for d in nz], default=0.0)
    print(f"worst relative difference: {worst:.3e}")

    # criteria table, straight from the rerun
    rows = []
    print("\ncriteria (rerun, straight from results.json.criteria):")
    for name, c in b["criteria"].items():
        print(f"  {name:34s} value={c['value']:>22.10g} limit={c['limit']:>10.3g} {c['kind']} passed={c['passed']}")
        rows.append((name, c["value"], c["limit"], c["passed"]))
    print("\ndiagnostics (not part of the verdict):")
    for name, c in b["diagnostics"].items():
        print(f"  {name:34s} value={c['value']:.10g} limit={c['limit']:.3g} passed={c['passed']}")
    n_fail = sum(1 for r in rows if not r[3])
    print(f"\ncriteria graded: {len(rows)}   FAIL: {n_fail}   -> all_criteria_passed={b['all_criteria_passed']}")

    # grading tolerances: pool tolerance from the case module
    print("\npool_tolerance:", b.get("pool_tolerance"))
    print("all_criteria_passed:", b.get("all_criteria_passed"))
    print("runtime_seconds(shipped):", a.get("runtime_seconds"), " rerun:", b.get("runtime_seconds"))
    print("repro_cmd:", b.get("repro_cmd"))
    print("artifacts:", json.dumps(b.get("artifacts"), ensure_ascii=False)[:400])
    print("\nsha256 shipped results.json:", hashlib.sha256(SHIPPED.read_bytes()).hexdigest())
    print("sha256 rerun   results.json:", hashlib.sha256(RERUN.read_bytes()).hexdigest())

    # worst graded residual vs the 5 % court tolerance
    key = {r[0]: r[1] for r in rows}
    graded = {k: key[k] for k in ("kochin_panels_integrated", "kochin_panels_ordinate", "haskind_far_panels") if k in key}
    tol = b.get("pool_tolerance")
    print("\nworst graded error vs tolerance:", max(graded.values()), "tol:", tol)
    nondet = [d[4] for d in nz if d[0] != "runtime_seconds"]
    ok = ((max(nondet, default=0.0) < 1e-12)
          and bool(b.get("all_criteria_passed")) and max(graded.values()) < tol)
    print("REPRO_OK:", ok)
    return 0 if ok else 1

sys.exit(main())
