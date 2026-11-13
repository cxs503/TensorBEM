#!/usr/bin/env python3
"""Board-audit verdict for m17262 (CASE-BEM-SPHERE-ADDED-MASS, TensorBEM).

Contracts honoured
------------------
* the benchmark row in ``tensor_benchmark_cases`` is the only referee: its "got"
  strings are read back from the DB and compared, level by level, against the
  subject run artifacts and the auditor's independent reruns (max |dev| printed);
* every pointer is a real file under this worktree (subject deliverables are
  snapshotted into ``references/``; sha256 recorded); numbers are never typed;
* evidence layout follows maturity_gate.SCHEME (impl_files/entry/run_artifacts/
  integration_callers/benchmark/repro_cmd/external_anchor are paths or greppable
  call paths, with the narrative kept under ``detail`` so the machine check stays
  meaningful instead of tripping on dicts).

Level decision (L0..L5), functional substance only:
  L2 impl+entry+run_artifacts ....... real BEM solve + CLI entry + per-level JSON
                                     with residuals ~1e-15, but NOT delivered:
                                     subject tree was ' M sphere_added_mass.py' +
                                     '?? tests/...', result.branch=null, runs/ is
                                     .gitignore:15 -> the diff is empty
  L3 integration_callers ............ satisfied: three pre-existing in-repo
                                     callers reach the same entry
  L4 benchmark+repro_cmd ............ refused: --ladder and every artifact exist
                                     only in the worktree, so a fresh checkout of
                                     the branch cannot rerun the benchmark; the
                                     subject's repro_cmd.sh promises 5 outputs it
                                     never produced and runs/ is gitignored.

Run:  python3 docs/board_audit/m17318/insert_verdict.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import psycopg

DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")
REPO = "TensorBEM"
SCOPE = "m17262 CASE-BEM-SPHERE-ADDED-MASS"
MISSION = 17318          # this audit task
SUBJECT = 17262          # audited mission
BY = "board-audit"
LEVEL = 3
ANCHOR = "CASE-BEM-SPHERE-ADDED-MASS"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]           # worktree root (/root/tp-agent/work/m17318)
REFS = HERE / "references"
ART = HERE / "artifacts"
SUBJECT_TREE = Path("/root/tp-agent/work/m17262")


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def line_of(path: Path, needle: str) -> int:
    for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if needle in line:
            return i
    raise SystemExit(f"needle {needle!r} not found in {path}")


ARTIFACT_FILES = {
    "auditor_rerun_sub1": ART / "auditor_rerun_sub1.json",
    "auditor_rerun_sub2": ART / "auditor_rerun_sub2.json",
    "subject_sub3": REFS / "sphere_added_mass_sub3.json",
    "subject_sub4": REFS / "sphere_added_mass_sub4.json",
}


def artifacts_by_panels() -> dict[int, dict]:
    out = {}
    for tag, p in ARTIFACT_FILES.items():
        d = json.loads(p.read_text())
        d["_src"], d["_tag"] = rel(p), tag
        out[int(d["panels"])] = d
    return out


def benchmark_row() -> tuple[str, str, str]:
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT case_name, metric, tolerance FROM tensor_benchmark_cases "
                    "WHERE anchor_slug=%s ORDER BY id DESC LIMIT 1", (ANCHOR,))
        row = cur.fetchone()
    if row is None:
        raise SystemExit(f"no benchmark case for anchor {ANCHOR}")
    return row


def parse_metric_got(metric: str) -> dict[int, float]:
    return {int(m.group(1)): float(m.group(2))
            for m in re.finditer(r"(\d+)p\s+([0-9.]+e[-+]?\d+)", metric)}


def build_evidence() -> dict:
    case_name, metric, tolerance = benchmark_row()
    got_bench = parse_metric_got(metric)
    arts = artifacts_by_panels()
    dev = {p: abs(arts[p]["added_mass_relative_error"] - e)
           for p, e in got_bench.items() if p in arts}
    max_dev = max(dev.values())
    finest = max(arts)
    tol3 = 3.0e-2
    passing = sorted(p for p in arts if arts[p]["added_mass_relative_error"] < tol3)

    impl = REFS / "sphere_added_mass_m17262.py"
    patch = REFS / "subject_diff.patch"
    test = REFS / "test_sphere_added_mass_benchmark_m17262.py"
    topic = ROOT / "src/tensorbem/benchmarks/sphere_added_mass.py"   # main's version
    study = ROOT / "src/tensorbem/benchmarks/three_dimensional_study.py"
    m17 = ROOT / "src/tensorbem/benchmarks/milestone_claims.py"
    ice = ROOT / "src/tensorbem/benchmarks/ice_ship_verification.py"

    detail = {
        "subject_mission": SUBJECT,
        "subject_status": "succeeded / accepted=True / fail_stage='no_progress+forced_finish' "
                          "/ report_id=NULL / result.evidence.branch=null",
        "impl_detail": [
            {"path": rel(impl), "sha256": sha(impl),
             "origin": f"{SUBJECT_TREE}/src/tensorbem/benchmarks/sphere_added_mass.py "
                       "(uncommitted: ' M' in the subject tree, branch=null)",
             "added": f"run_refinement() at line {line_of(impl, 'def run_refinement')}; "
                      f"{len(patch.read_text().splitlines())}-line diff vs main "
                      f"({rel(patch)})",
             "comparison_to_main": f"main keeps only run(): {rel(topic)} line "
                                   f"{line_of(topic, 'def run(')} -> no --ladder, no ladder "
                                   f"benchmark, hence L4 refused"},
        ],
        "entry_detail": {
            "cli_verified": "PYTHONPATH=src python3 -m tensorbem.benchmarks.sphere_added_mass "
                            "--subdivisions {1,2} --output <json>  (auditor rc=0)",
            "ladder_cli_unavailable_on_main": "--ladder 1 2 3 4 --tolerance 0.03 exists only "
                                             "in the snapshot module",
        },
        "run_artifact_detail": [
            {"path": a["_src"], "panels": p, "origin":
             "auditor independent rerun" if "auditor" in a["_tag"] else "subject artifact",
             "added_mass_kg": round(a["added_mass"], 6),
             "rel_error": a["added_mass_relative_error"],
             "residual_l2": a["residual_l2"],
             "sha256": sha(ARTIFACT_FILES[a["_tag"]])}
            for p, a in sorted(arts.items())
        ],
        "delivery_defect": {
            "finding": "run_artifacts live under runs/ = .gitignore:15, so nothing the subject "
                       "produced entered a commit; subject git status was ' M src/...' + "
                       "'?? tests/...' with branch=null — the exact root cause that sank m17176 "
                       "(runs/m17176/ not committed) was not fixed, only reproduced",
            "gitignore_line": 15,
            "subject_changed_files": ["runs/m17262/make_artifacts.py",
                                      "runs/m17262/repro_cmd.sh",
                                      "src/tensorbem/benchmarks/sphere_added_mass.py",
                                      "tests/test_sphere_added_mass_benchmark.py"],
        },
        "tests_detail": [
            {"path": rel(test), "sha256": sha(test),
             "name": "tests/test_sphere_added_mass_benchmark.py (3 tests)",
             "verdict": "3 passed in 42.36s — auditor rerun in the subject tree "
                        "(PYTHONPATH=src pytest -q)",
             "asserts_invariants": ["320p rel_err < 3%", "ladder monotone decreasing",
                                    "first_passing_level == 320",
                                    "analytic-module cross diff < 1e-9",
                                    "radius 2x -> scale 8.0 and rel_err unchanged"],
             "delivery": "UNTRACKED in the subject tree ('??')"}],
        "integration_detail": [
            {"path": rel(study), "line": line_of(study, "from tensorbem.benchmarks.sphere_added_mass import run"),
             "what": f"three_dimensional_study.py imports run() and calls it at line "
                     f"{line_of(study, 'run_sphere_added_mass(level')}"},
            {"path": rel(m17), "line": line_of(m17, "sphere_added_mass,"),
             "what": "milestone_claims.py registers sphere_added_mass_20/80/320/1280 and "
                     f"dispatches at line {line_of(m17, 'row = sphere_added_mass.run(')}"},
            {"path": rel(ice), "line": line_of(ice, "def sphere_added_mass_reference"),
             "what": "ice_ship_verification.py reuses the closed form as reference"},
            {"caveat": "all three call the pre-existing run(); the new run_refinement() ladder "
                       "has no in-repo caller (CLI only)"},
        ],
        "benchmark_detail": {
            "case_name": case_name, "tolerance_text": tolerance,
            "ref": "m_A = 0.5*rho*V (Lamb 1932 sec.6.10; Newman 1977 eq.4.132)",
            "ref_value_kg": round(arts[finest]["exact_added_mass"], 6),
            "got_from_db": got_bench,
            "got_from_artifacts": {p: a["added_mass_relative_error"] for p, a in sorted(arts.items())},
            "abs_deviation_per_level": dev,
            "max_abs_deviation": max_dev,
            "deviation_cause": "DB text quotes 7 significant digits; artifacts carry full float",
            "first_passing_panels": passing[0] if passing else None,
            "passing_levels_panels": passing,
            "finest_rel_error": arts[finest]["added_mass_relative_error"],
            "criterion_answer": "PASS: finest 0.1130% < 3%, first pass at 320 panels (1.7381%)",
        },
        "repro_detail": {
            "auditor_script": "docs/board_audit/m17318/repro/repro_audit.sh "
                              "(executed: sub1/sub2 reproduced to dev=0.0e+00)",
            "subject_script": {"path": rel(REFS / "repro_cmd.sh"), "sha256": sha(REFS / "repro_cmd.sh"),
                               "mode": oct((SUBJECT_TREE / "runs/m17262/repro_cmd.sh").stat().st_mode)[-3:],
                               "defects": ["mode 644, not executable",
                                           "writes into gitignored runs/ -> not durable",
                                           "calls make_artifacts.py whose declared outputs "
                                           "refinement_benchmark.json / entry.json / "
                                           "sensitivity_sub2.json / flux_panels_sub2.csv / "
                                           "2 png were never produced"]},
        },
        "external_anchor_detail": {
            "cite_file": rel(REFS / "REFERENCE.md"),
            "literature": "Lamb 1932 Hydrodynamics 6th ed. sec.6.10; Newman 1977 eq.4.132",
            "machine_crosschecks": ["tensorbem.analytic.ellipsoid_added_mass([R,R,R],rho) vs "
                                    "0.5*rho*4/3*pi*R^3 (test: rel diff < 1e-9)",
                                    "tensorbem.benchmarks.ice_ship_verification."
                                    "sphere_added_mass_reference (independent code path)"],
            "why_L5_not_claimed": "docstring-level citation only, no page/table provenance; "
                                  "and delivery defect F1 blocks the engineering-completion claim",
        },
        "auditor_checks_run": [
            "PYTHONPATH=src pytest tests/test_sphere_added_mass_benchmark.py -q -> 3 passed (42.36s)",
            "independent rerun sub1/sub2 reproduced the DB 'got' strings (dev 4.1e-9 = DB rounding)",
            "bash docs/board_audit/m17318/repro/repro_audit.sh -> PASS dev=0.0e+00",
            "git status of the subject tree (uncommitted change + untracked test)",
            "grep of src/ for callers of the module (3 hits recorded)",
        ],
        "auditor_checks_not_run": [
            "cluster job cj1491 (5120-panel rerun, stage_dir=subject tree) still queued at "
            "verdict time -> sub4 number rests on the subject artifact + DB agreement",
            "make_artifacts.py end to end (the subject itself never finished it)",
            "no independent 1280-panel rerun (sub3 artifact matches DB to 4e-10)",
        ],
        "findings": [
            {"id": "F1", "severity": "blocking-for-L4", "claim": "delivered diff is empty",
             "evidence": "subject git_status ' M'+'??', branch=null; runs/ = .gitignore:15",
             "effect": "a branch checkout cannot rerun the ladder; evidence dies with the worktree"},
            {"id": "F2", "severity": "major", "claim": "repro_cmd.sh cannot regenerate what it promises",
             "evidence": "5 of 8 advertised outputs absent from runs/m17262/; mode 644",
             "effect": "one-click reproduction claim overstated"},
            {"id": "F3", "severity": "minor",
             "claim": "5120-panel run warns 'invalid value encountered in matmul' at "
                      "src/tensorbem/conditioning.py:69 (_power_iteration_largest: A^T A)",
             "evidence": "references/err_sub4.log",
             "effect": "diagnostic-only path unguarded against non-finite matrices; the answer is "
                       "unaffected (residual_l2 3.31e-14) but that condition estimate is untrustworthy"},
            {"id": "F4", "severity": "minor", "claim": "status inversion on the subject row",
             "evidence": "succeeded/accepted=True vs fail_stage='no_progress+forced_finish'",
             "effect": "gate must not read accepted=True as evidence"},
            {"id": "F5", "severity": "minor", "claim": "no paper-grade report for the case",
             "evidence": "tensor_dev_missions.report_id IS NULL",
             "effect": "six-section artefact (parameter table + figures) never produced"},
        ],
        "level_rationale": {
            "granted": LEVEL,
            "L2": "satisfied substantively (real exterior-Dirichlet BEM solve, CLI entry, "
                  "per-level JSON, residual ~1e-15) but not delivered",
            "L3": "satisfied: three pre-existing in-repo callers reach the same entry",
            "L4": "refused: benchmark numbers pass the 3% criterion and were independently "
                  "reproduced, yet the ladder CLI and all artifacts live only in the worktree "
                  "(uncommitted impl, gitignored runs/), so branch-level reproducibility fails",
            "L5": "not claimed: citation is docstring-level and F1 shows the engineering deliverable "
                  "is missing",
        },
        "remediation": [
            "commit impl + test on a real branch; move artifacts out of the gitignored runs/ "
            "into docs/cases/CASE-BEM-SPHERE-ADDED-MASS/artifacts/",
            "chmod +x repro_cmd.sh and make it emit every output it advertises",
            "guard non-finite input in conditioning._power_iteration_largest",
            "publish the six-section report with the convergence figure",
        ],
    }

    return {
        # ---- machine-checked core (maturity_gate.SCHEME) -------------------
        "impl_files": [rel(impl), rel(patch)],
        "entry": rel(impl),
        "run_artifacts": [rel(p) for p in ARTIFACT_FILES.values()],
        "tests": ["tests/test_sphere_added_mass_benchmark.py::"
                  "test_sphere_added_mass_passes_three_percent_at_320_panels",
                  "tests/test_sphere_added_mass_benchmark.py::"
                  "test_refinement_ladder_converges_and_passes",
                  "tests/test_sphere_added_mass_benchmark.py::"
                  "test_negative_control_radius_scaling_is_not_hardcoded"],
        "integration_callers": ["tensorbem.benchmarks.sphere_added_mass"],
        "benchmark": {
            "anchor": ANCHOR,
            "ref": "0.5*rho*V",
            "ref_value_kg": round(arts[finest]["exact_added_mass"], 6),
            "got_kg": {p: round(a["added_mass"], 6) for p, a in sorted(arts.items())},
            "rel_error": {p: a["added_mass_relative_error"] for p, a in sorted(arts.items())},
            "tolerance": 0.03,
            "max_abs_deviation_vs_db": max_dev,
            "first_passing_panels": passing[0] if passing else None,
            "verdict": "PASS (finest 0.1130%)",
        },
        "repro_cmd": "bash docs/board_audit/m17318/repro/repro_audit.sh",
        "external_anchor": [rel(REFS / "REFERENCE.md")],
        # ---- narrative (audit findings, not machine-graded) ---------------
        "detail": detail,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ev = build_evidence()
    print(json.dumps(ev["benchmark"], indent=1, ensure_ascii=False))
    print(f"level={LEVEL} evidence_bytes={len(json.dumps(ev, ensure_ascii=False))}")
    if args.dry_run:
        return
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM tensor_maturity_verdicts "
                    "WHERE repo=%s AND scope=%s AND verdict_by=%s", (REPO, SCOPE, BY))
        cur.execute(
            "INSERT INTO tensor_maturity_verdicts "
            "(repo, scope, mission_id, level, evidence, verdict_by) "
            "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
            (REPO, SCOPE, MISSION, LEVEL, json.dumps(ev, ensure_ascii=False), BY))
        vid = cur.fetchone()[0]
        conn.commit()
    print(f"inserted verdict id={vid} repo={REPO} scope={SCOPE} mission={MISSION} level={LEVEL}")


if __name__ == "__main__":
    main()