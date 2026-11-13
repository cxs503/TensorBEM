#!/usr/bin/env python3
"""Insert board-audit verdict for m17205 (CASE-BEM-TWO-CYL-MUTUAL) into tensor_maturity_verdicts."""
import json
import subprocess

EVIDENCE = {
    "impl_files": [
        "src/tensorbem/benchmarks/two_cylinder_added_mass.py",
        "src/tensorbem/analytic.py",
    ],
    "entry": "src/tensorbem/benchmarks/two_cylinder_added_mass.py",
    "run_artifacts": [
        "/root/tp-agent/work/m17205/artifacts/two_cylinder_added_mass/benchmark.json",
        "/root/tp-agent/work/m17205/artifacts/two_cylinder_added_mass/figures_digest.json",
        "/root/tp-agent/work/m17205/artifacts/two_cylinder_added_mass/field_and_boundary.png",
        "/root/tp-agent/work/m17205/artifacts/two_cylinder_added_mass/interaction.png",
        "/root/tp-agent/work/m17205/artifacts/two_cylinder_added_mass/refinement.png",
    ],
    "tests": [
        "tests/test_two_cylinder_added_mass.py (audit run 2026-11-13 in m17205 worktree, PYTHONPATH=src: 18 passed in 49.93s)",
        "test_bem_matches_the_multipole_reference",
        "test_bem_converges_towards_the_reference",
        "test_bem_boundary_potential_matches_the_series_field",
        "test_injected_double_layer_sign_error_turns_the_checks_red",
        "test_decoupled_single_body_treatment_is_rejected",
    ],
    "integration_callers": [
        "from ..solver import assemble_matrices (src/tensorbem/benchmarks/two_cylinder_added_mass.py:66, invoked at :106; :65 MultiLoopMesh two-loop hole mesh - main-solver layer matrices, no new kernel)",
    ],
    "benchmark": {
        "anchor": "CASE-BEM-TWO-CYL-MUTUAL",
        "criterion_source": "tensor_benchmark_cases row id=41 registered by this audit (was 0 rows - systemic registry gap, cf. verdict#490 S2); tol_policy '<5%' from tensor_case_pool id=141 (BEM-TWO-CYL-MUTUAL) + mission 17205 goal; ecosystem line <3%",
        "ref_96": {"self": 3322.936921146587, "mutual_along": -730.5061072751591,
                    "mutual_transverse": 730.5061072751554, "in_phase_heave": 4053.443028421744},
        "got_96": {"self": 3322.8414061593007, "mutual_along": -730.1037484383877,
                    "mutual_transverse": 730.1037484383879, "in_phase_heave": 4052.9451545976904},
        "rel_error_96": {"self": 2.8744146985851748e-05, "mutual_along": 0.0005507946241165166,
                         "mutual_transverse": 0.0005507946241110725, "in_phase_heave": 0.0001228273891017764},
        "max_relative_error_96": 0.0005507946241165166,
        "ladder_max_err": {"48": 0.0022034075117916257, "64": 0.0012394885311633667,
                           "96": 0.0005507946241165166},
        "observed_orders": [1.9997986198424411, 2.000399234731374],
        "sweep": "d/a=3/5/10/20/40: mutual*d^2 -> 6440.27 ((a/d)^2 law), self -> rho*pi*a^2 (1+7.8e-7 at d/a=40)",
        "all_pass_3pct": True,
        "all_pass_5pct": True,
        "ref_impl": "src/tensorbem/analytic.py:1678 (independent of solver operators; BC residual 3.69e-11, reflection symmetry 2.5e-15, far-field 1+7.8e-7)",
    },
    "repro_cmd": ("cd /root/tp-agent/work/m17205 && PYTHONPATH=src python3 -m "
                  "tensorbem.benchmarks.two_cylinder_added_mass --panels 48 64 96 --json <out.json> "
                  "[audit rerun 2026-11-13: 282/282 scalar keys identical, benchmark.json sha256 "
                  "68e3568c26cfe7b24e65ab88fd95f629c0f488fdf3b87599229065948709b32b byte-identical]"),
    "external_anchor": ("closed-form multipole series re-implementation src/tensorbem/analytic.py:1678 "
                        "(independent of solver operators; self-verified BC residual 3.69e-11, far-field "
                        "1+7.8e-7) - L4 reference, NOT an L5 printed-table/experiment anchor (WITHHELD)"),
    "run_artifacts_sha256": {
        "benchmark.json": "68e3568c26cfe7b24e65ab88fd95f629c0f488fdf3b87599229065948709b32b",
        "figures_digest.json": "a59408db1b9035aa4875ca1780ed9e7e041afaabf60e1aaac8280017e9a5cf64",
        "field_and_boundary.png": "91785758dea2396ea0f232e8c8e0525ba093a6b6de4c89cbea63cdf563ab5094",
        "interaction.png": "59d9038bf3528bcd7a1ab8295b8745b6a1869e7e6194dd94eb71b9a081d0c95d",
        "refinement.png": "39fcdce20cf36f34fa365bfd385b8a004d34d298ce5d9d0f8f0539d4616fbfaa",
    },
    "audit": {
        "verdict_doc": "docs/board_audit/m17237/AUDIT_m17205.md",
        "report": "tensor_reports id=16206 (six sections complete, 3 figures)",
        "findings": [
            "F1 medium: tensor_benchmark_cases had 0 rows for anchor CASE-BEM-TWO-CYL-MUTUAL -> registered by this audit (row id=41, status=passed); tolerance taken from case pool tol_policy <5% + ecosystem <3%",
            "F2 medium: m17205 deliverables untracked in git (branch agent/m17205 has no unique commits, head_commit null; artifacts on-disk only) - substance recovered by auditor rerun (byte-identical); case code itself is in main via m4721 PR#315 (commit 1259bb5)",
            "F3 low: tensor_case_pool id=141 still 'dispatched' - left to mechanism (taskboard feature_ledger done -> case_feeder passed) after gate confirms",
            "F4 info: reference is in-repo closed-form re-implementation, no printed-table transcription (WITHHELD for L5)",
            "F5 info: mission changed_files only _post_m17205.py (untracked) - m17205 contribution = pool-case execution + artifacts + report + negative controls, per goal 'reuse existing capability'",
        ],
        "checks_not_run": [
            "PNG pixel-level content (backed by figures_digest.json + passing tests)",
            "second independent BEM implementation cross-check",
            "printed-table (Newman/Pinkham) numeric anchor",
        ],
        "level_rationale": ("L4: real BEM solve over two disjoint loops via main-solver assemble_matrices + "
                           "independent closed-form multipole anchor + per-quantity error table + O(h^2) "
                           "convergence (observed 1.9998/2.0004) + separation sweep + one-command repro "
                           "independently rerun byte-identical + 18 tests incl. injected-error negative controls; "
                           "no new physics beyond the verified exterior operator -> not L5"),
    },
}

ev = json.dumps(EVIDENCE, ensure_ascii=False)
sql = (
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
    "VALUES ('TensorBEM', 'm17205 CASE-BEM-TWO-CYL-MUTUAL', 17237, 4, "
    f"'{ev.replace(chr(39), chr(39)*2)}'::jsonb, 'board-audit') "
    "RETURNING id, gate_state"
)
r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-tA", "-c", sql],
                   capture_output=True, text=True)
print("rc:", r.returncode)
print("out:", r.stdout.strip())
print("err:", r.stderr.strip()[:500])
