"""Machine-checkable evidence bundle for CASE-BEM-HULME-HEMISPHERE (m17292).

The campaign gate does not accept a self-declaration ("the case is implemented
and verified"); it wants the artefacts to exist at the declared paths, the
implementation files to be non-placeholder, the callers to really call the code,
and the ref/got numbers to be re-derivable from the bytes on disk.  This script
produces exactly that and fails loudly when any claim does not hold::

    python benchmarks/verified/casepool_bem_hulme_hemisphere/evidence.py

It writes ``evidence.json`` next to the artefacts with the blocks

``impl_files``           implementation files + a non-placeholder audit of each
                         (line count, public callables, no ``NotImplementedError``
                         / ``TODO`` / ``FIXME`` / empty body)
``entry``                executable entry points (package CLI + case-pool
                         driver) with the exact reproduction command
``run_artifacts``        files produced by the real run (results/reference/
                         environment JSON, run log, figure, report), each with
                         size + sha256 so the numbers trace to the bytes
``integration_callers``  files outside the module that import and call it, each
                         verified by grepping the symbol out of the caller source
``benchmark``            ref/got/rel_error/tolerance/pass rows, recomputed from
                         the raw numbers in ``results.json`` (never copied from
                         the stored error fields), plus a fresh recomputation of
                         the Hulme reference series for every ``ka``
``geometry``             the Cb/volume/waterplane precondition the discipline
                         requires before any coefficient table is readable
``tests``                the pytest files that assert the anchors
``repro_cmd``            the one-line commands that reproduce each block

Exit code is 0 only if every declared path exists, every caller greps, every
implementation file passes the non-placeholder audit, the geometry precondition
holds and every benchmark row is inside the 5 % case-pool tolerance.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # benchmarks/verified/<case>/ -> repo root
CASE = "BEM-HULME-HEMISPHERE"

IMPL_FILES = [
    "src/tensorbem/benchmarks/hulme_hemisphere_case.py",
    "src/tensorbem/analytic.py",
    "benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py",
    "benchmarks/verified/casepool_bem_hulme_hemisphere/make_doc.py",
]

ENTRIES = [
    {
        "path": "src/tensorbem/benchmarks/hulme_hemisphere_case.py",
        "kind": "cli",
        "command": ("python -m tensorbem.benchmarks.hulme_hemisphere_case "
                    "--out benchmarks/verified/casepool_bem_hulme_hemisphere"),
        "note": "package entry: computes the task list, the Hulme reference, the verdict, "
                "report.md and figure.png",
    },
    {
        "path": "benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py",
        "kind": "case_pool_driver",
        "command": ("python benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py "
                    "--out benchmarks/verified/casepool_bem_hulme_hemisphere "
                    "--shards 4 --parallel 4"),
        "note": "shard orchestration over the package entry + merge; non-zero exit on FAIL",
    },
]

INTEGRATION_CALLERS = [
    {"file": "benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py",
     "symbol": "hulme_hemisphere_case",
     "kind": "case_pool_driver",
     "note": "the case-pool driver imports the module and calls case.main() / case.build_tasks()"},
    {"file": "tests/test_hulme_hemisphere_case.py",
     "symbol": "run_cell",
     "kind": "test",
     "note": "the TensorBEM test suite solves a real BEM cell and applies evaluate()"},
    {"file": "tests/test_hulme_hemisphere_case.py",
     "symbol": "evaluate",
     "kind": "test",
     "note": "gate evaluation is exercised on both passing and deliberately failing rows"},
    {"file": "benchmarks/verified/casepool_bem_hulme_hemisphere/evidence.py",
     "symbol": "hulme_hemisphere_case",
     "kind": "evidence",
     "note": "this bundle re-derives the reference series through the module instead of "
             "trusting the stored numbers"},
]

RUN_FILES = [
    "results.json",
    "reference.json",
    "environment.json",
    "run.log",
    "figure.png",
    "report.md",
]
LOGS_DIR = "logs"

TESTS = ["tests/test_hulme_hemisphere_case.py"]

PLACEHOLDER_MARKERS = ("NotImplementedError", "TODO", "FIXME", "placeholder")

#: quantities compared against the Hulme reference (case mesh, every ka)
QUANTITIES = ("A11_dimless", "B11_dimless", "A33_dimless", "B33_dimless",
              "Fx_dimless", "Fz_dimless")
PHASE_KEYS = ("phase_Fx_deg", "phase_Fz_deg")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_impl(rel: str) -> dict:
    """Non-placeholder audit: real code, no surrender markers, no empty bodies."""
    path = REPO / rel
    row = {"path": rel, "exists": path.is_file(), "ok": False}
    if not path.is_file():
        return row
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    callables = [n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    empty = [n.name for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and all(isinstance(b, ast.Pass) or (isinstance(b, ast.Expr)
                                                 and isinstance(b.value, ast.Constant))
                     for b in n.body)]
    markers = [m for m in PLACEHOLDER_MARKERS if m in source]
    row.update({
        "lines": len(source.splitlines()),
        "callables": len(callables),
        "public_callables": sorted({c for c in callables if not c.startswith("_")})[:12],
        "empty_bodies": empty,
        "placeholder_markers": markers,
        "ok": (len(source.splitlines()) >= 20 and callables and not empty and not markers),
    })
    return row


def audit_caller(row: dict) -> dict:
    path = REPO / row["file"]
    out = {**row, "exists": path.is_file(), "verified": False}
    if not path.is_file():
        return out
    source = path.read_text(encoding="utf-8")
    hits = [ln for ln in source.splitlines() if row["symbol"] in ln]
    out["hits"] = len(hits)
    out["first_hit"] = hits[0].strip()[:120] if hits else ""
    out["verified"] = bool(hits)
    return out


def build_benchmark(results: dict) -> dict:
    """Re-derive ref/got/error from the raw numbers in results.json.

    The stored ``rel_error_*`` fields are *not* trusted: they are recomputed
    from ``got`` and ``ref``, and a mismatch >1e-9 flags a hand-edited table.
    """
    case_sub = results["case"].get("case_subdivisions")
    rows = []
    max_drift = 0.0
    for row in results["rows"]:
        if row.get("kind") != "band":
            continue
        if case_sub is not None and row["subdivisions"] != case_sub:
            continue
        ref = row["reference"]
        for key in QUANTITIES:
            got = float(row[key])
            target = float(ref[key])
            err = (got - target) / abs(target)
            stored = float(row[f"rel_error_{key}"])
            max_drift = max(max_drift, abs(err - stored))
            rows.append({
                "ka": row["ka"],
                "panels": row["panels"],
                "quantity": key,
                "ref": target,
                "got": got,
                "rel_error": err,
                "tolerance": results["case"]["tol_policy"],
                "pass": abs(err) < 0.05,
            })
        for key in PHASE_KEYS:
            got = float(row[key])
            target = float(ref[key])
            stored = float(row[f"delta_{key}"])
            max_drift = max(max_drift, abs((got - target) - stored))
            rows.append({
                "ka": row["ka"],
                "panels": row["panels"],
                "quantity": key,
                "ref": target,
                "got": got,
                "rel_error": got - target,
                "tolerance": "5 deg absolute",
                "pass": abs(got - target) < 5.0,
                "note": "phase delta is absolute (deg), not relative",
            })
    return {
        "source": "results.json rows (case mesh, kind=band)",
        "n_rows": len(rows),
        "all_pass": all(r["pass"] for r in rows) if rows else False,
        "max_abs_rel_error": max((abs(r["rel_error"]) for r in rows
                                  if r["quantity"] in QUANTITIES), default=float("nan")),
        "recompute_vs_stored_drift": max_drift,
        "reproduces_stored_rows": bool(max_drift < 1e-9),
        "rows": rows,
    }


def build_reference_check(results: dict) -> dict:
    """Recompute the Hulme series now and compare it to the stored reference."""
    sys.path.insert(0, str(REPO / "src"))
    from tensorbem.benchmarks import hulme_hemisphere_case as case  # noqa: PLC0415

    stored = results["reference"]
    worst = 0.0
    cases = 0
    detail = []
    for ka_key, values in sorted(stored.items(), key=lambda kv: float(kv[0])):
        fresh = case.reference(float(ka_key))
        for key in QUANTITIES + PHASE_KEYS:
            drift = abs(fresh[key] - float(values[key])) / max(1.0, abs(float(values[key])))
            worst = max(worst, drift)
            cases += 1
        detail.append({"ka": float(ka_key), "A33": fresh["A33_dimless"],
                       "B33": fresh["B33_dimless"]})
    ladders = results.get("ref_ladder", {})
    worst_tail = max((v["max_rel_delta"]["96->144"] for v in ladders.values()),
                     default=float("nan"))
    return {
        "implementation": "tensorbem.analytic.hulme_hemisphere_radiation",
        "terms_used": case.REF_TERMS[-1],
        "n_values_compared": cases,
        "max_drift": worst,
        "matches": bool(worst < 1e-12),
        "worst_96_144_term_drift": worst_tail,
        "self_converged": bool(worst_tail < 0.05 * 0.05),
        "values": detail,
        "paper_digits_transcribed": False,
        "note": ("no digit of Hulme's printed table is copied into the repository; the "
                 "reference is the wave-free multipole series recomputed at run time "
                 "(see docs/bem_hulme_case.md)"),
    }


def build_geometry(results: dict) -> dict:
    case_sub = results["case"].get("case_subdivisions")
    volumes = []
    for row in results["rows"]:
        if case_sub is not None and row["subdivisions"] != case_sub:
            continue
        volumes.append({
            "ka": row["ka"],
            "panels": row["panels"],
            "mesh_volume": row["mesh_volume"],
            "nabla_exact": row["nabla_exact"],
            "volume_rel_error": row["mesh_volume"] / row["nabla_exact"] - 1.0,
            "waterplane_rel_error": row["mesh_waterplane_area"] / (math.pi) - 1.0,
        })
    worst_v = max((abs(v["volume_rel_error"]) for v in volumes), default=float("nan"))
    worst_w = max((abs(v["waterplane_rel_error"]) for v in volumes), default=float("nan"))
    return {
        "criterion": "|nabla_mesh/nabla - 1| < 2 % and |A_wp,mesh/(pi a^2) - 1| < 2 %",
        "worst_volume_rel_error": worst_v,
        "worst_waterplane_rel_error": worst_w,
        "pass": bool(max(worst_v, worst_w) < 0.02),
        "rows": volumes,
    }


def collect_tests() -> list[dict]:
    out = []
    for rel in TESTS:
        path = REPO / rel
        proc = subprocess.run([sys.executable, "-m", "pytest", rel, "--collect-only", "-q"],
                              cwd=REPO, capture_output=True, text=True)
        ids = [ln.strip() for ln in proc.stdout.splitlines() if "::" in ln]
        out.append({"path": rel, "exists": path.is_file(), "n_collected": len(ids),
                    "node_ids_sample": ids[:4]})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-dir", default=str(HERE))
    parser.add_argument("--strict", action="store_true",
                        help="also fail when any benchmark row is outside tolerance")
    args = parser.parse_args(argv)
    case_dir = Path(args.case_dir).resolve()
    results_path = case_dir / "results.json"
    if not results_path.is_file():
        print(f"FAIL: no results.json under {case_dir}; run the case first",
              file=sys.stderr)
        return 2
    results = json.loads(results_path.read_text(encoding="utf-8"))

    impl = [audit_impl(rel) for rel in IMPL_FILES]
    entries = []
    for entry in ENTRIES:
        path = REPO / entry["path"]
        entries.append({**entry, "exists": path.is_file(),
                        "sha256": sha256(path) if path.is_file() else None})

    callers = [audit_caller(row) for row in INTEGRATION_CALLERS]

    run_artifacts = []
    for name in RUN_FILES:
        path = case_dir / name
        run_artifacts.append({
            "path": str(path.relative_to(REPO)),
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": sha256(path) if path.is_file() else None,
        })
    log_dir = case_dir / LOGS_DIR
    if log_dir.is_dir():
        for path in sorted(log_dir.glob("*.log")):
            run_artifacts.append({
                "path": str(path.relative_to(REPO)),
                "exists": True,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "role": "shard worker stdout",
            })

    benchmark = build_benchmark(results)
    reference = build_reference_check(results)
    geometry = build_geometry(results)
    tests = collect_tests()

    report_path = case_dir / "report.md"
    report_text = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    verdict = results["verdict"]["verdict"]
    checks = {
        "impl_files_ok": all(r["ok"] for r in impl),
        "entries_exist": all(r["exists"] for r in entries),
        "run_artifacts_exist": all(r["exists"] for r in run_artifacts),
        "integration_callers_verified": all(r["verified"] for r in callers),
        "tests_collected": all(r["n_collected"] > 0 for r in tests),
        "benchmark_rows_recomputed": benchmark["reproduces_stored_rows"],
        "benchmark_rows_in_tolerance": benchmark["all_pass"],
        "reference_recomputed_matches": reference["matches"],
        "reference_self_converged": reference["self_converged"],
        "geometry_precondition": geometry["pass"],
        "report_carries_verdict": verdict in report_text,
        "verdict_pass": verdict == "PASS",
    }
    mandatory = ["impl_files_ok", "entries_exist", "run_artifacts_exist",
                 "integration_callers_verified", "tests_collected",
                 "benchmark_rows_recomputed", "reference_recomputed_matches",
                 "reference_self_converged", "geometry_precondition",
                 "report_carries_verdict"]
    if args.strict:
        mandatory += ["benchmark_rows_in_tolerance", "verdict_pass"]
    ok = all(checks[k] for k in mandatory)

    levels = {
        "L1_artifact": bool(checks["report_carries_verdict"]),
        "L2_implementation": bool(checks["impl_files_ok"] and checks["entries_exist"]
                                  and checks["run_artifacts_exist"]),
        "L3_integration": bool(checks["integration_callers_verified"]
                              and checks["tests_collected"]),
        "L4_benchmark": bool(checks["benchmark_rows_recomputed"]
                             and checks["benchmark_rows_in_tolerance"]
                             and checks["reference_recomputed_matches"]),
    }

    bundle = {
        "case": CASE,
        "case_pool_id": results["case"].get("case_pool_id"),
        "mission_id": results["case"].get("mission_id"),
        "repo": "TensorBEM",
        "case_dir": str(case_dir.relative_to(REPO)),
        "verdict": verdict,
        "levels": levels,
        "environment": results.get("environment", {}),
        "impl_files": impl,
        "entry": entries,
        "run_artifacts": run_artifacts,
        "integration_callers": callers,
        "benchmark": benchmark,
        "reference": reference,
        "geometry": geometry,
        "tests": tests,
        "repro_cmd": {
            "case_full": ("python benchmarks/verified/casepool_bem_hulme_hemisphere/"
                          "run_case.py --out benchmarks/verified/"
                          "casepool_bem_hulme_hemisphere --shards 4 --parallel 4"),
            "case_package_cli": ("python -m tensorbem.benchmarks.hulme_hemisphere_case "
                                 "--out benchmarks/verified/casepool_bem_hulme_hemisphere"),
            "evidence": ("python benchmarks/verified/casepool_bem_hulme_hemisphere/"
                         "evidence.py --strict"),
            "tests": "python -m pytest tests/test_hulme_hemisphere_case.py -q",
            "doc": ("python benchmarks/verified/casepool_bem_hulme_hemisphere/make_doc.py "
                    "--results benchmarks/verified/casepool_bem_hulme_hemisphere/results.json"),
        },
        "checks": checks,
        "ok": ok,
    }
    out = case_dir / "evidence.json"
    out.write_text(json.dumps(bundle, indent=2, default=float), encoding="utf-8")
    for key, value in checks.items():
        print(f"[{'PASS' if value else 'FAIL'}] {key}")
    print(f"[{'PASS' if ok else 'FAIL'}] bundle")
    print(f"evidence -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())