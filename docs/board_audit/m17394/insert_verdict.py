#!/usr/bin/env python3
"""Board-audit verdict for m17334 (CASE-BEM-SUBMERGED-SPHERE-HAVELOCK, TensorBEM).

Auditor re-verification performed (this task m17394):
- tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-SUBMERGED-SPHERE-HAVELOCK' -> 0 rows
  (systemic registry gap, same class as m17174); tolerance falls back to mission goal text <5%.
- Independent re-run of the declared repro_cmd cheap stage (PYTHONPATH=src run_case.py
  --stage cheap): exit 0, economy-band got/ref values bit-identical to the committed
  smoke_economy_320 artifact (max abs deviation 0.0) -> repro_cmd executable, numbers real.
- table_band_case.csv ref column vs tensorbem.analytic.submerged_sphere_wave_multipole
  (n_terms=20) re-evaluated by the auditor: max rel deviation 0.0 -> zero hand transcription.
- case_mesh tier (1280 panels, the documented case mesh): 20/20 items <5%, max 1.43%
  (ka=0.5 b33: got 0.0834933 vs ref 0.0847055) -> PASS.
- economy_mesh tier (320 panels, committed): 7/28 items >=5%, max 5.59% -> honest FAIL.
- capytaine 3.1.dev0 cross-check worst |capy-ref| = 3.96% -> PASS (ref not self-proving).
- DEFECT: committed CI guard tests/test_case_pool_registry.py::
  test_every_registered_case_resolves_its_paths FAILS in the committed state: registry
  declares docs/case_bem_submerged_sphere_havelock/{CASE.md,fig1..fig4.png} which were
  never generated/committed (make_doc.py not run on final numbers.json).
- DEFECT: the PASSING case_mesh tier artifacts (top-level numbers.json/verdict.json/
  tiers.json/table_*.csv) are UNCOMMITTED (git status ??); commit d2815a5 carries only
  the FAILing economy tier. Report 16344 section 6 itself states the 1280-panel run
  was still running at submission.

Level: L3 (system integration). L4 refused: the only tier that passes the <5% gate is
uncommitted, the committed tier fails, and the committed CI guard is red.
"""
from __future__ import annotations

DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")
REPO = "TensorBEM"
SCOPE = "m17334 CASE-BEM-SUBMERGED-SPHERE-HAVELOCK"
MISSION = 17394
LEVEL = 3
BY = "board-audit"

import json
import psycopg

DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")
REPO = "TensorBEM"
SCOPE = "m17334 CASE-BEM-SUBMERGED-SPHERE-HAVELOCK"
MISSION = 17394
LEVEL = 3
BY = "board-audit"

EVIDENCE = {
    "impl_files": [
        "benchmarks/verified/casepool_bem_submerged_sphere_havelock/run_case.py",
        "benchmarks/verified/casepool_bem_submerged_sphere_havelock/make_doc.py",
        "src/tensorbem/benchmarks/case_havelock_submerged_sphere.py",
        "src/tensorbem/benchmarks/submerged_sphere_havelock.py",
        "src/tensorbem/analytic.py:1306 (submerged_sphere_wave_multipole, independent multipole ref)",
    ],
    "entry": "benchmarks/verified/casepool_bem_submerged_sphere_havelock/run_case.py (predeclared G1-G5 gates, staged runs, artifact emission)",
    "run_artifacts": [
        "docs/case_bem_submerged_sphere_havelock/smoke_economy_320/numbers.json (committed, economy tier FAIL 5.59%)",
        "docs/case_bem_submerged_sphere_havelock/smoke_economy_320/verdict.json",
        "docs/case_bem_submerged_sphere_havelock/capytaine_crosscheck.json (committed, worst 3.96% PASS)",
        "docs/case_bem_submerged_sphere_havelock/numbers.json (UNCOMMITTED, case_mesh tier PASS 1.43%)",
    ],
    "tests": [
        "tests/test_case_pool_registry.py (committed; test_every_registered_case_resolves_its_paths FAILS: registry points at missing CASE.md + fig1-4.png)",
        "tests/test_casepool_submerged_sphere_havelock.py (committed; requires CASE.md/numbers.json case_mesh tier, not generated at submission)",
    ],
    "integration_callers": [
        "src/tensorbem/benchmarks/case_pool_registry.py (anchor registry, missing_paths/artifact_verdict)",
        "scripts/case_pool_run.py (dispatch CLI, --list/--check)",
    ],
    "benchmark": (
        "tensor_benchmark_cases: 0 rows for anchor (judge row absent; tolerance <5% from mission goal). "
        "case_mesh 1280 panels: 20/20 items <5%, max 1.43% (ka=0.5 b33 got 0.0834933/ref 0.0847055) PASS but UNCOMMITTED; "
        "economy_mesh 320 panels (committed): 7/28 items >=5%, max 5.59% FAIL; "
        "capytaine 3.1.dev0 independent cross-check worst 3.96% PASS; "
        "auditor re-run cheap stage: economy got/ref bit-identical to committed artifact (dev 0.0); "
        "case-mesh ref column vs analytic re-eval: max dev 0.0"
    ),
    "repro_cmd": "PYTHONPATH=src python3 benchmarks/verified/casepool_bem_submerged_sphere_havelock/run_case.py --stage all --outdir docs/case_bem_submerged_sphere_havelock (auditor re-ran --stage cheap: exit 0, ~4 min)",
    "external_anchor": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK; Havelock 1955 / Thorne 1953 / Wang 1986 / Linton-McIver 1991; ref evaluated programmatically (no publisher table in sandbox), anchored by truncation 20->40 machine-zero + Haskind residual <1e-10 + Lamb deep-submergence limit",
    "defects": [
        "passing case_mesh tier artifacts uncommitted (git ?? in m17334 worktree; commit d2815a5 carries only the failing economy tier)",
        "committed CI guard red: registry declares CASE.md + fig1-4.png that do not exist",
        "report 16344 section 6 admits case-mesh run unfinished at submission; report economy table says 6/12 >=5% while final artifact says 7/28 (report predates final run)",
    ],
    "level_rationale": "L2 met (real impl+entry+committed run artifacts); L3 met (registry+CLI platform call surface, real callers); L4 refused: passing tier uncommitted, committed tier fails <5% gate, committed CI guard red. L4 becomes confirmable once case_mesh artifacts + generated CASE.md/figures are committed and the registry guard goes green.",
}


def main() -> None:
    conn = psycopg.connect(DSN, connect_timeout=15)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        RETURNING id
        """,
        (REPO, SCOPE, MISSION, LEVEL, json.dumps(EVIDENCE, ensure_ascii=False), BY),
    )
    row = cur.fetchone()
    if row is None:
        cur.execute(
            "SELECT id, level FROM tensor_maturity_verdicts "
            "WHERE repo=%s AND scope=%s AND mission_id=%s",
            (REPO, SCOPE, MISSION),
        )
        row = cur.fetchone()
        print("existing row:", row)
    else:
        print("inserted id:", row[0])
    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
