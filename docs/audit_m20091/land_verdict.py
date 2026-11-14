#!/usr/bin/env python3
"""Land the m20149 board-audit verdict for m20091 CASE-BEM-CRACK-SIF.

1) builds the evidence bundle with real pointers,
2) dry-runs the platform maturity gate (read-only import of /root/tp-agent/maturity_gate.py),
3) upserts the verdict row (idempotent on repo+scope+verdict_by).

Usage: python3 land_verdict.py [--write]
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

REPO = "TensorBEM"
SCOPE = "m20091 CASE-BEM-CRACK-SIF"
MISSION = 20149
LEVEL = 2
VERDICT_BY = "board-audit"
AUDITEE = "/root/tp-agent/work/m20091"
HERE = pathlib.Path(__file__).parent
WT = "/root/tp-agent/work/m20149"

EVIDENCE = {
    "audited_mission": "m20091 (TensorBEM, task_key=case-bem-crack-sif-rework1, status=succeeded)",
    "impl_files": [
        f"{AUDITEE}/src/tensorbem/crack_sif.py",
        "docs/audit_m20091/auditee_snapshot/crack_sif.py",
    ],
    "entry": [
        f"{AUDITEE}/src/tensorbem/benchmarks/case_bem_crack_sif.py",
        f"{AUDITEE}/run_benchmark_crack_sif.py",
        "docs/audit_m20091/auditee_snapshot/case_bem_crack_sif.py",
    ],
    "run_artifacts": [
        f"{AUDITEE}/artifacts/case_bem_crack_sif.json",
        f"{AUDITEE}/artifacts/cod_profile.png",
        f"{AUDITEE}/artifacts/convergence.png",
        f"{AUDITEE}/artifacts/sif_comparison.png",
        f"{AUDITEE}/artifacts/reproduce.sh",
        "docs/audit_m20091/probe_result.json",
        "docs/audit_m20091/audit_numbers.json",
        "docs/audit_m20091/fig1_error_vs_mesh.png",
        "docs/audit_m20091/fig2_sif_bars.png",
        "docs/audit_m20091/pytest_rerun.log",
        "docs/audit_m20091/reproduce_rerun.log",
        "docs/audit_m20091/snapshot_sha256.txt",
        "docs/audit_m20091/AUDIT_m20091.md",
    ],
    "tests": [
        "tests/test_crack_sif.py::TestChebyshevSolver::test_exact_sif",
        "tests/test_crack_sif.py::TestConstantDDSolver::test_sif_within_tolerance",
        "tests/test_crack_sif.py::TestConstantDDSolver::test_residual_small",
        "tests/test_crack_sif.py::TestConvergence::test_convergence_trend",
        "pytest tests/test_crack_sif.py -q @ " + f"{AUDITEE}" + " -> 25 passed in 1.15s (docs/audit_m20091/pytest_rerun.log)",
    ],
    "integration_callers": [],
    "integration_callers_absent_reason": (
        "grep -rn 'crack_sif' over the repo: only src/tensorbem/capability_evidence.py:1787 "
        "(declaration table), the module's own benchmark and its tests reference it; no main-solver "
        "or platform route imports it -> L3 not met"
    ),
    "benchmark": {
        "registry_row": "ABSENT: SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-CRACK-SIF' -> 0 rows",
        "ref_source": "Tada, Paris & Irwin (2000) Eq. 2.1a, K_I = sigma*sqrt(pi*a)",
        "ref_MPa_sqrt_m": 17.7245385,
        "got_delivered_MPa_sqrt_m_n640": 19.2739,
        "err_delivered_pct_n640": 8.742,
        "got_independent_replicate_MPa_sqrt_m_n5120": 19.3257,
        "err_independent_pct_n5120": 9.038,
        "tolerance_pct": 3.0,
        "gate": "FAIL (>3%): numeric route not convergent; delivered 0.0000% is an algebraic identity",
        "claimed_err_pct": 0.0,
        "claimed_err_is_tautology": {
            "random_cases": 120,
            "max_rel_diff_vs_closed_form": 1.87e-16,
            "k_got_spread_over_n_elements_4_to_1000_Pa": 0.0,
            "linalg_calls_in_chebyshev_path": "none (AST scan)",
        },
        "cod_field_err_pct_n640": 0.210,
        "root_cause": (
            "extract_sif (crack_sif.py:189-198) 5-tip-element mean; same DD field with a Griffith-shape "
            "least-squares extraction gives 0.108% at n=640 -> defect is post-processing, not the BIE solve"
        ),
    },
    "repro_cmd": f"cd {AUDITEE} && PYTHONPATH=src python3 artifacts/reproduce.sh",
    "repro_cmd_note": (
        "runs and prints 'Error = 0.0000e+00% / PASS' (reproduce_rerun.log), but `bash artifacts/reproduce.sh` "
        "fails: the file is Python named .sh (reproduce_bash_invocation.log)"
    ),
    "external_anchor": "Tada/Paris/Irwin 2000 Handbook Eq. 2.1a - closed-form theory only, not experiment/printed table -> caps at L4, never L5",
    "six_section_report": {
        "in_repo": "docs/case_bem_crack_sif.md sections 1-6 present (byte copy: docs/audit_m20091/auditee_snapshot/case_bem_crack_sif.md)",
        "tensor_reports_rows_for_m20091": 0,
    },
    "claimed_vs_found": {
        "claimed": "Chebyshev error <1e-10%, PASS, 25 tests, L4-style delivery",
        "found": "default route is an algebraic identity (no linear solve); the only numeric route is "
                 "non-convergent at 8.7-9.0% > 3%; no benchmark registry row; no integration caller; "
                 "report misattributes the drift to the discretisation instead of the SIF extraction",
        "honest_level": "L2 (real DD-BIE implementation, accuracy/integration/benchmark not met)",
    },
    "verdict_doc": "docs/audit_m20091/AUDIT_m20091.md",
    "probe": "docs/audit_m20091/probe_crack_sif.py",
    "checks_not_run": [
        "third-party BEM cross-check (the repo's Kelvin-BEM elasticity2d.py from m20003 is not on this branch or in main)",
        "pixel-level verification of the auditee PNGs (existence/size only)",
        "higher-order / Galerkin DD variants (constant-DD collocation scope only)",
    ],
    "independence": "read-only import of the auditee module; auditee tree untouched; independent vectorised re-implementation reproduced the DD field with max rel diff 0.0",
}


def gate_judge(level: int, evidence: dict) -> tuple[int, list[str]]:
    spec = importlib.util.spec_from_file_location("maturity_gate", "/root/tp-agent/maturity_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # module level: no side effects beyond env read
    return mod.judge(level, evidence, REPO, MISSION)


def sql(text: str) -> str:
    return subprocess.run(
        ["psql", "-h", "127.0.0.1", "-U", "postgres", "-d", "tensorplatform", "-At", "-c", text],
        capture_output=True, text=True, env=_env(),
    ).stdout.strip()


def _env() -> dict:
    import os
    e = dict(os.environ)
    e.setdefault("PGPASSWORD", os.environ.get("TP_PG_PASS", "postgres"))
    return e


def main() -> int:
    gl, why = gate_judge(LEVEL, EVIDENCE)
    print(json.dumps({"claimed": LEVEL, "gate_level": gl, "reasons": why}, ensure_ascii=False, indent=1))
    if gl < LEVEL:
        print("ABORT: gate would downgrade this verdict; fix evidence first", file=sys.stderr)
        return 1
    if "--write" not in sys.argv:
        print("dry-run only (pass --write to upsert)")
        return 0
    ev = json.dumps(EVIDENCE, ensure_ascii=False).replace("'", "''")
    stmt = (
        "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
        f"VALUES ('{REPO}', '{SCOPE}', {MISSION}, {LEVEL}, '{ev}'::jsonb, '{VERDICT_BY}') "
        f"ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET mission_id={MISSION}, level={LEVEL}, "
        "evidence=EXCLUDED.evidence, gate_state='pending', gate_level=NULL, gate_note=NULL "
        "RETURNING id, gate_state;"
    )
    print("upsert ->", sql(stmt))
    print(sql("SELECT id, repo, scope, mission_id, level, verdict_by, gate_state FROM tensor_maturity_verdicts "
              f"WHERE repo='{REPO}' AND scope='{SCOPE}' AND verdict_by='{VERDICT_BY}';"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())