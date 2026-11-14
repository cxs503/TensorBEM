#!/usr/bin/env python3
"""Board-audit verdict insertion for m20003 CASE-BEM-CRACK-SIF."""
import json
import subprocess

EVIDENCE = {
    "impl_files": [
        "src/tensorbem/elasticity2d.py (722 lines: plane-strain Kelvin BEM, constant elements, closed-form self term, dimensionless assembly, crack_quarter_mesh, opening_displacement, sif_from_profile, sif_from_opening, reference_center_crack_k, uniaxial_patch, lame_annulus)",
        "src/tensorbem/benchmarks/case_bem_crack_sif.py (736 lines: case-pool driver, 4-mesh ladder, 4-axis sensitivity sweep, 11 gates, programmatic REPORT.md + 4 figures)",
        "src/tensorbem/benchmarks/case_pool_registry.py (CASE-BEM-CRACK-SIF entry at line 121)",
    ],
    "entry": "src/tensorbem/benchmarks/case_bem_crack_sif.py (commit 4e310bc9)",
    "run_artifacts": {
        "numbers.json": "docs/case_bem_crack_sif/numbers.json (892 lines, 27410 bytes, all values programmatic)",
        "REPORT.md": "docs/case_bem_crack_sif/REPORT.md (166 lines, 7 sections)",
        "figures": [
            "docs/case_bem_crack_sif/crack_opening_profile.png (55316 bytes)",
            "docs/case_bem_crack_sif/crack_sif_extrapolation.png (42723 bytes)",
            "docs/case_bem_crack_sif/crack_sif_ladder.png (44236 bytes)",
            "docs/case_bem_crack_sif/crack_sif_truncation.png (49163 bytes)",
        ],
    },
    "tests": [
        "tests/test_elasticity2d_crack_sif.py (111 lines, 8 tests, all passed in 50.70s)",
        "test_rigid_body_row_sum_of_traction_kernel",
        "test_plane_strain_modulus",
        "test_reference_center_crack_k_is_the_closed_form",
        "test_near_tip_relation_recovers_k_on_the_exact_griffith_profile",
        "test_uniaxial_patch_test (includes O(h) order check)",
        "test_lame_annulus_against_closed_form",
        "test_crack_quarter_mesh_geometry",
        "test_crack_sif_coarse_mesh_lands_within_tolerance",
    ],
    "integration_callers": [
        "elasticity2d.py is a new module (no prior elasticity operator in TensorBEM); it is the foundation for the crack benchmark",
        "case_bem_crack_sif.py imports elasticity2d as e2 and calls: e2.reference_center_crack_k, e2.crack_quarter_mesh, e2.solve, e2.opening_displacement, e2.sif_from_profile, e2.sif_from_opening, e2.uniaxial_patch, e2.lame_annulus",
    ],
    "benchmark": {
        "anchor": "CASE-BEM-CRACK-SIF",
        "criterion_source": "case_pool_registry.py tolerance='<5%'; reference = closed-form K_I = sigma*sqrt(pi*a) evaluated by elasticity2d.reference_center_crack_k (no table transcription)",
        "reference": {
            "formula": "K_I = sigma * sqrt(pi * a)",
            "sigma_MPa": 100.0,
            "a_mm": 10.0,
            "K_ref_MPa_sqrt_m": 17.72453850905516,
            "citations": [
                "Irwin (1957) stress-intensity definition applied to the Westergaard (1939) centre-crack solution",
                "Tada, Paris & Irwin, The Stress Analysis of Cracks Handbook, 3rd ed. (2000), case 2.1",
                "Anderson, Fracture Mechanics, 3rd ed. (2005), Table 2.1",
            ],
        },
        "ladder_results": {
            "finest_mesh": {
                "tip_element_over_a": 0.00025,
                "segments": 274,
                "unknowns": 548,
                "condition_number": 2.18e6,
                "K_A_MPa_sqrt_m": 17.815306855785664,
                "K_B_MPa_sqrt_m": 17.725303482846,
                "K_C_MPa_sqrt_m": 17.861217863590824,
                "F_A": 1.0051210555741201,
                "F_B": 1.000043159024448,
                "F_C": 1.0077113068115053,
                "err_A_percent": 0.5121055574120126,
                "err_B_percent": 0.004315902444806952,
                "err_C_percent": 0.7711306811505336,
                "opening_profile_max_rel_error": 0.002214890,
            },
            "convergence_A": "1.1587% -> 0.7800% -> 0.6002% -> 0.5121% (monotone, ratio 0.442)",
            "convergence_C": "1.5249% -> 1.1020% -> 0.8810% -> 0.7711% (monotone)",
            "convergence_B": "0.0123% -> -0.0040% -> 0.0047% -> 0.0043% (near-zero, shape-consistent)",
        },
        "gates": {
            "V1_patch_displacement": {"value": 0.001997, "criterion": "< 2e-3", "pass": True},
            "V2_patch_force_balance": {"value": 2.93e-06, "criterion": "< 1e-4", "pass": True},
            "V3_patch_order": {"value": 4.107, "criterion": "3.0-5.5", "pass": True},
            "V4_lame_inner_u": {"value": 0.00936, "criterion": "< 1e-2", "pass": True},
            "V5_lame_refined": {"value": 0.00299, "criterion": "< 5e-3", "pass": True},
            "G1_K_near_tip": {"value": 0.00512, "criterion": "< 5%", "pass": True},
            "G2_K_profile": {"value": 4.32e-05, "criterion": "< 5%", "pass": True},
            "G3_K_traction": {"value": 0.00771, "criterion": "< 5%", "pass": True},
            "G4_refinement_trend_A": {"value": 0.442, "criterion": "< 1", "pass": True},
            "G5_truncation": {"value": 1.31e-04, "criterion": "< 1e-3", "pass": True},
            "G6_quadrature": {"value": 5.80e-06, "criterion": "< 1e-3", "pass": True},
        },
        "all_gates_pass": True,
        "max_relative_error_percent": 0.7711,
        "all_within_5_percent": True,
    },
    "repro_cmd": "PYTHONPATH=src python3 -m tensorbem.benchmarks.case_bem_crack_sif --output docs/case_bem_crack_sif/numbers.json --figures docs/case_bem_crack_sif",
    "external_anchor": "closed-form K_I = sigma*sqrt(pi*a) from Westergaard/Irwin theory (Tada 2000 case 2.1, Anderson 2005 Table 2.1); evaluated programmatically by elasticity2d.reference_center_crack_k — NOT a printed-table transcription, NOT an experiment; L4 reference (WITHHELD for L5)",
    "audit": {
        "verdict_doc": "this audit (m20065)",
        "report": "tensor_reports (six sections + figures, programmatic injection)",
        "findings": [
            "F1 info: TensorBEM had zero elasticity operators before this commit; elasticity2d.py is a genuine new module (722 lines) implementing 2-D plane-strain Kelvin BEM from scratch",
            "F2 info: three independent SIF routes (A: COD extrapolation, B: profile fit, C: traction extrapolation) all agree within 5%; route B is near-exact (0.004%) because it assumes the Griffith shape, routes A and C are genuine predictions (0.51% and 0.77%)",
            "F3 info: convergence is monotone but plateaus at ~0.5% for route A — constant elements cannot represent sqrt(r) crack-tip opening; this is honestly documented as a limitation, not hidden",
            "F4 info: reference is closed-form theory, not experiment or printed table — appropriate for L4 but not L5",
            "F5 info: 8 fast tests all pass, including negative controls (rigid-body row sum, exact-profile SIF recovery, coarse-mesh tolerance check)",
        ],
        "checks_not_run": [
            "Full benchmark rerun (timeout > 120s; numbers.json and tests independently verified)",
            "PNG pixel-level content verification (backed by file existence and sizes)",
            "Independent BEM implementation cross-check (no second elasticity BEM exists in TensorBEM)",
        ],
        "level_rationale": (
            "L4: real BEM solve (elasticity2d.py, 722 lines) + complete case-pool benchmark "
            "(4-mesh ladder, 4-axis sensitivity, 11 gates all pass) + error analysis "
            "(three independent SIF routes, convergence documented, truncation error quantified) "
            "+ one-command repro + 8 tests incl. negative controls + programmatic report with figures; "
            "reference is closed-form theory (not experiment/printed table) -> not L5. "
            "The implementation is a genuine new physics module (2-D elastostatic BEM) "
            "with proper verification (patch tests, Lame annulus) before the crack case."
        ),
    },
}

ev = json.dumps(EVIDENCE, ensure_ascii=False)
sql = (
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
    "VALUES ('TensorBEM', 'm20003 CASE-BEM-CRACK-SIF', 20065, 4, "
    f"'{ev.replace(chr(39), chr(39)*2)}'::jsonb, 'board-audit') "
    "RETURNING id, gate_state"
)
r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-tA", "-c", sql],
                   capture_output=True, text=True)
print("rc:", r.returncode)
print("out:", r.stdout.strip())
print("err:", r.stderr.strip()[:500])
