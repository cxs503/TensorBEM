#!/usr/bin/env python3
"""Insert the m17429 board-audit verdict for m17373 CASE-BEM-ICE-DISPERSION.

Evidence keys follow the maturity_gate JUDGE_SCHEME contract
(impl_files / entry / run_artifacts / tests / integration_callers /
benchmark / repro_cmd / external_anchor).  Every pointer below was opened by
this audit; numbers come from docs/audit_m17429/audit_metrics.json.
"""
import json
import subprocess
import sys

sys.path.insert(0, "/root/tp-agent")
import maturity_gate  # noqa: E402  (read-only use of the judge, for karma check)

REPO = "TensorBEM"
SCOPE = "m17373 CASE-BEM-ICE-DISPERSION"
AUDIT_MID = 17429
WT_M = "/root/tp-agent/work/m17373"          # audited work tree (commit 26cbde4)
LEVEL = 3

EVIDENCE = {
    "audited_mission": 17373,
    "audited_commit": "26cbde4516fcf6756105ec6584dcd097b51b399d",
    "audited_branch": "agent/m17373",
    "impl_files": [
        f"{WT_M}/src/tensorbem/ice_mass_loading_dispersion.py",
        f"{WT_M}/src/tensorbem/benchmarks/ice_mass_loading_dispersion.py",
        f"{WT_M}/src/tensorbem/ice.py",
        f"{WT_M}/src/tensorbem/__init__.py",
    ],
    "impl_lines": {
        "bem_driver": f"{WT_M}/src/tensorbem/ice_mass_loading_dispersion.py:521-596 (strip_bem_depth_factor: mesh+BIE solve+Rayleigh quotient)",
        "matrix_assembly": f"{WT_M}/src/tensorbem/ice_mass_loading_dispersion.py:371-436 (assemble_strip_matrices 单/双层积分)",
        "periodic_ties": f"{WT_M}/src/tensorbem/ice_mass_loading_dispersion.py:436-481 (tie_operators / _tied_operator)",
        "dtn_route": f"{WT_M}/src/tensorbem/ice_mass_loading_dispersion.py:609-643,759-788",
        "solver_entry_wrapper": f"{WT_M}/src/tensorbem/ice.py:239-283 (mass_loading_omega_bem)",
        "benchmark_driver": f"{WT_M}/src/tensorbem/benchmarks/ice_mass_loading_dispersion.py:1-739",
    },
    "entry": f"{WT_M}/src/tensorbem/benchmarks/ice_mass_loading_dispersion.py",
    "run_artifacts": [
        f"{WT_M}/docs/case_bem_ice_dispersion/numbers.json",
        f"{WT_M}/docs/case_bem_ice_dispersion/table.md",
        f"{WT_M}/docs/case_bem_ice_dispersion/REPORT.md",
        f"{WT_M}/docs/case_bem_ice_dispersion/figs/fig_dispersion.png",
        f"{WT_M}/docs/case_bem_ice_dispersion/figs/fig_convergence.png",
        f"{WT_M}/docs/case_bem_ice_dispersion/figs/fig_field.png",
        f"{WT_M}/docs/case_bem_ice_dispersion/figs/fig_sensitivity.png",
        "docs/audit_m17429/audit_metrics.json",
        "docs/audit_m17429/numbers_regenerated.json",
        "docs/audit_m17429/logs/independent_check.log",
        "docs/audit_m17429/logs/repro_benchmark.log",
        "docs/audit_m17429/logs/repro_tests.log",
        "docs/audit_m17429/fig_ref_vs_got.png",
        "docs/audit_m17429/fig_convergence_independent.png",
        "docs/audit_m17429/fig_depth_factor.png",
        "docs/audit_m17429/AUDIT_REPORT.md",
    ],
    "tests": [
        "tests/test_ice_mass_loading_dispersion.py::test_depth_factor_reproduces_k_tanh",
        "tests/test_ice_mass_loading_dispersion.py::test_dtn_route_agrees_with_the_driven_route",
        "tests/test_ice_mass_loading_dispersion.py::test_boundary_values_off_the_free_surface_are_recovered",
        "tests/test_ice_mass_loading_dispersion.py::test_interior_field_matches_the_exact_mode",
        "tests/test_ice_mass_loading_dispersion.py::test_bem_frequency_matches_the_closed_form",
        "tests/test_ice_mass_loading_dispersion.py::test_refinement_reduces_the_depth_factor_error",
        "tests/test_ice_mass_loading_dispersion.py::test_ice_module_callers_use_the_bem",
        "tests/test_ice_mass_loading_dispersion.py::test_benchmark_driver_runs_and_stays_within_tolerance",
        "tests/test_ice_mass_loading_dispersion.py (whole file, 31 tests, auditor rerun: 31 passed in 15.1 s)",
    ],
    "integration_callers": ["mass_loading_omega_bem"],
    "integration_callers_note": (
        "solver-side wrapper tensorbem/ice.py:239-283 + package export "
        "src/tensorbem/__init__.py:230-244, exercised by "
        "tests/test_ice_mass_loading_dispersion.py::test_ice_module_callers_use_the_bem (passes). "
        "CAVEAT F1/F2: branch agent/m17373 is unmerged and has no remote ref (main fc205ca "
        "lacks the module), and no existing solver path calls it (ice_rao.py / hydroelastic.py "
        "still use the closed form) -> machine gate greps the main checkout and will demote to L2."
    ),
    "benchmark": {
        "registered_row": "tensor_benchmark_cases anchor_slug='CASE-BEM-ICE-DISPERSION' (1 row, tolerance <5%, prose metric, no per-item ref/got)",
        "auditor_independent_ref": "Weitz-Keller closed form recomputed in docs/audit_m17429/audit_m17373.py (not imported from mission code)",
        "omega_max_rel_error_pct": 0.3846,
        "omega_error_pct_by_row": {
            "deep_k0.02": 0.0698, "deep_k0.05": 0.0686, "deep_k0.1": 0.0663,
            "deep_k0.2": 0.0616, "deep_k0.4": 0.0537,
            "shallow_k0.02": 0.3846, "shallow_k0.05": 0.2209,
            "shallow_k0.1": 0.1131, "shallow_k0.2": 0.0601,
        },
        "depth_factor_rel_error_pct_at_64ppw": {"deep": 0.1446, "shallow_k0.04": 0.7712},
        "tolerance_pct": 5.0,
        "pass": True,
        "registered_metric_recomputed": {
            "deep_water_k_solver_agreement": 2.2e-13,
            "shallow_kH": 0.426472,
            "shallow_omega_recovered_rel_err": 3.14e-14,
            "shallow_long_wave_rel_err": 0.0413,
            "rk4_shooting_rel_err": 1.43e-13,
            "robin_residual_abs": 4.02e-15,
            "thickness_study_max_rel_diff": 1.1e-14,
        },
        "registered_metric_gap": "row entry 'shallow k rel err 2.87e-02' not reproducible (audit: 4.13% at the same kH=0.426; h_i scan 3.29-4.27%) - F4",
        "tautology_note": "omega_got and omega_ref share the analytic Robin map Lambda->omega (monotone), so the headline omega error re-expresses the Lambda error; the genuinely independent BEM prediction is Lambda = (dphi/dz)/phi vs k*tanh(kH) (F3)",
        "convergence_independent": {
            "deep": {"err_pct": [3.6334, 1.2941, 0.4301, 0.1446, 0.0517], "monotone": True},
            "intermediate": {"err_pct": [2.3101, 0.8688, 0.5178, 0.1370, 0.0491], "monotone": True},
            "shallow": {"err_pct": [5.9085, 2.9660, 0.9091, 0.2304, 0.0595], "monotone": True},
        },
    },
    "repro_cmd": (
        "cd /tmp/aud17373  # byte copy of m17373 @26cbde4 src/tests; auditor ran 2026-11-13 exit 0\n"
        "PYTHONPATH=src python -m pytest tests/test_ice_mass_loading_dispersion.py -q  # 31 passed, 15.1 s\n"
        "PYTHONPATH=src python -m tensorbem.benchmarks.ice_mass_loading_dispersion --figure  # 30.8 s, all criteria passed, numbers.json bit-identical\n"
        "cd /root/tp-agent/work/m17429 && python3 docs/audit_m17429/audit_m17373.py  # independent R1-R4 checks"
    ),
    "external_anchor": ["docs/audit_m17429/AUDIT_REPORT.md"],
    "external_anchor_note": (
        "WITHHELD: ref is the programmatic Weitz-Keller closed form (Fox & Squire 1994 eq.3 / "
        "Weitz & Keller 1950); the 1950 printed table page was not reachable offline, so no "
        "primary-source numeric anchor exists -> L5 not claimed. Mission report already self-declares this (REPORT.md:40-46)."
    ),
    "checks_run": [
        "auditor rerun of mission repro_cmd x2 in isolated copy: pytest 31 passed; benchmark driver exit 0 in 30.8 s; regenerated numbers.json field-identical to committed (sha a101fa49b9aa1cb6)",
        "independent R2: 9 dispersion rows re-adjudicated against a closed form recomputed by the auditor (max 0.3850% < 5%)",
        "independent R3: own ppw ladder 8..128 for deep/intermediate/shallow (monotone, orders 0.75-1.98)",
        "independent R1: registered benchmark-row quantities re-derived from tensorbem.ice (RK4 1.43e-13, Robin 4.02e-15, thickness 1.1e-14, shallow long-wave 4.13%)",
        "source read: strip_bem_depth_factor drives Dirichlet cos(kx) -> solves BIE -> Rayleigh quotient (no analytic shortcut); depth_factor_ref only used for comparison",
        "merge state: git branch -r has no agent/m17373; main fc205ca lacks src/tensorbem/ice_mass_loading_dispersion.py",
    ],
    "checks_not_run": [
        "second-order/higher-order panel formulation cross-check - not_run",
        "3-D ice sheet or oblique incidence - out of the module's scope, not_run",
        "primary-source numeric table of Weitz & Keller (1950) - not reachable offline (WITHHELD)",
        "cluster (loong64) rerun - not needed, case runs in ~31 s single core on the audit host",
    ],
    "level_downgrade_reason": (
        "claimed L4 ('L2-L4 evidence supplemented'). Kept at L3 because (F1) branch agent/m17373 is "
        "unmerged with no remote ref, so the integration caller does not exist in the ecosystem main "
        "checkout and no existing solver path (ice_rao/hydroelastic) invokes it; (F3) the benchmark's "
        "headline omega comparison shares the analytic Robin map with the reference, so the independent "
        "BEM prediction verified is the depth factor Lambda, not a BEM-predicted dispersion curve; "
        "(F5) the primary-source anchor is WITHHELD. Numerics are honest and strong (0.385% < 5%, "
        "monotone order ~1.5, byte-identical reproduction), so L4 is one merge + one external anchor away."
    ),
    "verdict_summary": (
        "Real 1047-line periodic-strip BEM (BIE assembly + tied mixed BC solve + Rayleigh-quotient "
        "depth factor), 31 tests pass on auditor rerun, benchmark verified independently at 0.385% "
        "max error vs a 5% tolerance, artifacts byte-reproducible. Level 3: integration is library-API "
        "only and lives in an unmerged branch; external anchor withheld."
    ),
}

DEVISE = json.dumps(EVIDENCE, ensure_ascii=False)

# karma check: what will the gate machine-judge from this evidence?
gate_level, reasons = maturity_gate.judge(LEVEL, EVIDENCE, REPO, AUDIT_MID)
print("declared level:", LEVEL, "-> gate_expected_level:", gate_level)
for r in reasons:
    print("   gate reason:", r)

sql = (
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES "
    f"('{REPO}', '{SCOPE}', {AUDIT_MID}, {LEVEL}, $j${DEVISE}$j$::jsonb, 'board-audit') "
    "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET "
    "mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, evidence = EXCLUDED.evidence, "
    "gate_state = 'pending', gate_level = NULL, gate_note = NULL, created_at = now() "
    "RETURNING id, gate_state;"
)
r = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform",
                    "-tA", "-F", "|", "-c", sql], capture_output=True, text=True)
print("insert:", r.returncode, r.stdout.strip(), r.stderr.strip()[:300])