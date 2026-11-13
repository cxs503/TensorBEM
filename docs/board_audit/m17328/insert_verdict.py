#!/usr/bin/env python3
"""Board-audit verdict for m17277 (CASE-BEM-QTF-DRIFT) -> tensor_maturity_verdicts.

Idempotent: deletes any earlier verdict of the same (repo, scope, verdict_by)
before inserting. Also registers the missing referee row in
tensor_benchmark_cases (0 rows existed for this anchor) with status='failed'
(the measured non-diagonal error exceeds the only registered tolerance).

Run: python3 docs/board_audit/m17328/insert_verdict.py
"""
from __future__ import annotations

import json
import subprocess

REPO = "TensorBEM"
SCOPE = "m17277 CASE-BEM-QTF-DRIFT"
MISSION = 17328
LEVEL = 3
BY = "board-audit"
W = "/root/tp-agent/work/m17277"

STUDY = json.load(open(f"{W}/docs/case_qtf_drift_closed_form/study.json"))
MED = STUDY["levels"][1]
PAIRS = {f"({p['i']},{p['j']})": {"ka": [p["ka_i"], p["ka_j"]],
                                 "bem_abs": p["bem_abs"], "ref_abs": p["ref_abs"],
                                 "rel_error": p["abs_error"]}
         for p in MED["pairs"]}
AUDIT = json.load(open("/tmp/audit_m17328/independent_qtf.json"))

EVIDENCE = {
    "impl_files": [
        f"{W}/src/tensorbem/cylinder_qtf.py",
        f"{W}/src/tensorbem/benchmarks/cylinder_qtf_closed_form.py",
        f"{W}/tests/test_cylinder_qtf.py",
    ],
    "entry": f"{W}/src/tensorbem/benchmarks/cylinder_qtf_closed_form.py",
    "run_artifacts": [
        f"{W}/docs/case_qtf_drift_closed_form/study.json",
        f"{W}/docs/case_qtf_drift_closed_form/tables.md",
        f"{W}/docs/case_qtf_drift_closed_form/report.md",
        f"{W}/docs/case_qtf_drift_closed_form/qtf_bem_vs_closed_form.png",
        f"{W}/docs/case_qtf_drift_closed_form/qtf_closed_form_curves.png",
        f"{W}/docs/case_qtf_drift_closed_form/qtf_closed_form_map.png",
        f"{W}/docs/case_qtf_drift_closed_form/qtf_mesh_convergence.png",
    ],
    "tests": [
        "tests/test_cylinder_qtf.py -- auditor rerun 2026-11-13: 19 passed in 3.71 s (PYTHONPATH=src, -p no:cacheprovider)",
        "test_diagonal_reproduces_the_exact_mean_drift[ka=0.2..1.4] (G1 <1e-10)",
        "test_diagonal_reproduces_the_reduced_maruo_far_field[ka=0.2,0.5,0.9,1.4] (G2 <1e-3)",
        "test_long_wave_limit_matches_the_classical_cubic_law (G3)",
        "test_matrix_is_hermitian_and_diagonal_is_real (G4)",
        "test_azimuthal_quadrature_is_exact_for_the_finite_series",
        "test_off_diagonal_is_complex_and_continuous_across_the_diagonal",
    ],
    "integration_callers": [
        "quadratic_qtf",
        "src/tensorbem/benchmarks/cylinder_qtf_closed_form.py:bem_level -> quadratic_qtf(...) + solve_diffraction_3d(...) + bottom_mounted_cylinder_mesh(...) (main-solver stack; ECO main reference src/tensorbem/benchmarks/qtf_drift_study.py:217 calls the same assembler)",
        "closed_form_cylinder_qtf returns the solver's DriftQTF contract (src/tensorbem/qtf_drift.py:DriftQTF); no main-solver caller of the new module yet (harness-side only)",
    ],
    "benchmark": {
        "anchor": "CASE-BEM-QTF-DRIFT",
        "criterion_source": "tensor_benchmark_cases had 0 rows for this anchor -> referee row registered by this audit; tolerance '<10%' from tensor_case_pool id=176 tol_policy + mission 17277 goal",
        "metric": "per-pair |T_xx(ka_i,ka_j)| relative error vs the closed form, max over off-diagonal pairs, finest affordable mesh (medium=696 panels, ka=0.4/0.6/0.8, head seas, kH=12)",
        "got_vs_ref_medium": PAIRS,
        "offdiagonal_median_error": MED["offdiagonal_median_error"],
        "offdiagonal_max_error": MED["offdiagonal_max_error"],
        "diagonal_median_error": MED["diagonal_median_error"],
        "refinement": {"coarse_336": {"diag": 0.06592506677530524, "offdiag": 0.20410941678446287,
                                      "offdiag_max": 0.2688716584658148},
                       "medium_696": {"diag": MED["diagonal_median_error"], "offdiag": MED["offdiagonal_median_error"],
                                      "offdiag_max": MED["offdiagonal_max_error"]}},
        "within_10_percent": False,
        "verdict": "FAIL (honest): off-diagonal 20.62% median / 32.07% max vs the <10% criterion, and the off-diagonal error does not fall under refinement (20.41% -> 20.62%) while the diagonal does (6.59% -> 4.56%); diagonal FAILs nothing but the case criterion is not met",
        "ref_side_gates": {"G1_diagonal_identity": 4.2527895494136006e-15,
                          "G2_far_field_consistency": 1.417596516471196e-15,
                          "G3_long_wave_law": 0.001176596415473447,
                          "G4_hermitian": 2.449552155689319e-14},
        "auditor_independent_check": {
            "method": "own scipy MacCamy-Fuchs series (jv/jvp/hankel1/h1vp), own 4096-pt azimuthal quadrature; does not import tensorbem.cylinder_drift",
            "no_flux_bc_rel_residual_max": max(r["no_flux_bc_residual_rel"] for r in AUDIT["first_order_field_independent_check"]),
            "psi_vs_module_max_abs": max(r["max_abs_psi_diff_vs_module"] for r in AUDIT["first_order_field_independent_check"]),
            "azimuthal_integrals_max_rel_diff": max(max(r["I1_rel_diff"], r["I2_rel_diff"]) for r in AUDIT["azimuthal_integrals_independent_check"]),
            "diagonal_vs_audit_near_field_max_rel": max(r["rel_diff_vs_audit_near"] for r in AUDIT["diagonal_anchor_independent_check"]),
            "diagonal_vs_audit_maruo_far_field_max_rel": max(r["rel_diff_vs_audit_far"] for r in AUDIT["diagonal_anchor_independent_check"]),
            "long_wave_coefficient_at_ka0.002": [r for r in AUDIT["long_wave_asymptote_independent_check"] if r["ka"] == 0.002][0],
            "off_diagonal_independent_anchor": "NONE (near-field and Maruo routes yield the diagonal only) -> off-diagonal absolute value WITHHELD",
        },
    },
    "repro_cmd": ("cd <TensorBEM @ 3cd2c8d> && PYTHONPATH=src python3 -m "
                  "tensorbem.benchmarks.cylinder_qtf_closed_form --json <out>/study.json "
                  "--figures <out> --levels coarse medium --tables <out>/tables.md "
                  "[auditor rerun 2026-11-13 in /tmp: coarse level 9/9 pairs reproduce the "
                  "delivered study.json bit-for-bit; a real 336-panel diffraction solve "
                  "measures 26.8 s, so the delivered solve_seconds_total=0.235 s is a cache "
                  "reload - the delivered artifact was produced with --cache /tmp/qcache; "
                  "the documented command also omits --no-depth-sweep, so it emits a "
                  "superset of the delivered study.json (no depth_sensitivity key there)]"),
    "external_anchor": ("classical long-wave asymptote F_x = (5 pi^2/16) rho g A^2 a (ka)^3 "
                        "(Maruo/Vugts) reproduced at 6.7e-7 relative at ka=0.002 by the "
                        "delivered closed form and by the auditor's own MF series; NO printed "
                        "table/page numeric anchor was used (mission asked for one; report "
                        "declares the deviation honestly) -> external anchor WITHHELD for L5"),
    "run_artifacts_sha256": {
        "study.json": "6318f5216f70adc55be61b6afa0cd74cfd879e3172421cea07d57218a2107973",
        "tables.md": "d7c636e3f4fd5945524bc84abe1ad0d477699411a702d379be8ac283ae031196",
        "report.md": "b480990d0301a46aa65ca454a82ce145eb6429fe2edaa219d56b1b0aa3c505a3",
        "qtf_bem_vs_closed_form.png": "a9fe40213078a9377e04ae0a78f7c23a754e595c17049562ce9f63175434621a",
        "qtf_closed_form_curves.png": "41ed9da924fd7069684c703628fb9a67e57d7b0d389f631bb402059ab92bee4e",
        "qtf_closed_form_map.png": "feca2d0f54929bd57fb145863c6198dade2800515b39a89fa851f990259421c4",
        "qtf_mesh_convergence.png": "92e48e7a82b6acd87b55340a4fd853d10a9e580a00bf84dfad2766ba1e17a2fc",
    },
    "audit": {
        "verdict_doc": "docs/board_audit/m17328/AUDIT_m17277.md",
        "subject_commit": "3cd2c8d (branch agent/m17277 only; not in origin/main)",
        "subject_mission": 17277,
        "subject_report": "tensor_reports id=16291 (six sections + notes/repro, 4 embedded figures)",
        "disposition": "approve with notes",
        "findings": [
            "F1 high: case criterion FAILs honestly - off-diagonal 20.62% median / 32.07% max vs <10%, non-monotone under refinement; self-report matches measurement",
            "F2 medium: 'the gap is on the BEM side' is an inference, not a proof; no independent off-diagonal anchor exists (only Hermiticity 8.9e-16 + ka continuity) -> reference off-diagonal WITHHELD; capytaine three-way cross-check is the round-2 fix",
            "F3 medium: mission asked for a literature table/page ref; the delivery computes the reference closed form instead and declares it (report s1). Method deviation is honest but the external-anchor half of the requirement is unmet -> L5 denied",
            "F4 low: documented repro_cmd does not reproduce the delivered study.json byte-identically - no --cache (solve_seconds_total=0.235 s is a cache reload, real solve 26.8 s) and no --no-depth-sweep (documented command emits a superset)",
            "F5 low: study.json carries no git_head/BLAS/thread provenance",
            "F6 info: new module has no main-solver caller yet and lives only on agent/m17277; pre-push gate grep may downgrade L3->L2 (process fact, verdict 503 precedent)",
            "F7 info: DriftQTF.matrix is (n,n,n_components) on the BEM side vs (n,n) in the closed form; absorbed in the case via _component(), public API left unchanged",
            "F8 info: depth sweep not run; analytic bound e^{-kH}=6.1e-6 at kH=12 accepted, table left empty and labelled as not run",
        ],
        "checks_not_run": [
            "capytaine/third-party panel code cross-check of the off-diagonal QTF",
            "printed-table (Newman/Molins) numeric anchor - no network egress",
            "fine level (34x20) mesh ladder",
            "depth sweep (kH = 6/12/24)",
            "pixel-level PNG content (sizes + sha256 + regenerated tables only)",
        ],
        "level_rationale": ("L3: real module (225 lines, machine-precision analytic verification G1-G4 re-derived independently by the audit) + case entry driving the main-solver stack and returning the solver's DriftQTF contract; L4 denied because the only registered criterion (<10%) FAILs at 20.62%/32.07% and the off-diagonal does not converge, and the reference has no external printed anchor. Matches the m17242 precedent (partial FAIL -> L3)."),
    },
}


def psql(sql: str) -> subprocess.CompletedProcess:
    return subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-tA", "-c", sql],
                          capture_output=True, text=True)


def esc(s: str) -> str:
    return s.replace("'", "''")


def main() -> None:
    # 1) referee row (0 rows existed for this anchor)
    ref_row = psql("SELECT id FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-QTF-DRIFT'")
    if not ref_row.stdout.strip():
        vals = ("'CASE-BEM-QTF-DRIFT', 'BEM-QTF-DRIFT second-order difference-frequency QTF of a "
                "fixed bottom-mounted cylinder vs machine-precision closed form', "
                "'closed form in src/tensorbem/cylinder_qtf.py (MacCamy-Fuchs modes + Pinkster "
                "near field; anchored to near-field/Maruo mean drift and the 5 pi^2/16 long-wave "
                "law); no printed-table anchor (WITHHELD)', "
                "'per-pair |T_xx(ka_i,ka_j)| relative error, max over off-diagonal pairs, medium "
                "mesh 696 panels, ka=0.4/0.6/0.8 head seas kH=12', '<10%', 'TensorBEM', 'failed'")
        out = psql("INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, "
                   f"metric, tolerance, repo, status) VALUES ({vals}) RETURNING id")
        print("referee row inserted:", out.stdout.strip(), out.stderr.strip()[:200])
    else:
        print("referee row already present:", ref_row.stdout.strip())

    # 2) verdict
    ev = json.dumps(EVIDENCE, ensure_ascii=False)
    psql("DELETE FROM tensor_maturity_verdicts WHERE repo='TensorBEM' AND "
         f"scope='{esc(SCOPE)}' AND verdict_by='{BY}'")
    out = psql("INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, "
               f"verdict_by) VALUES ('{REPO}', '{esc(SCOPE)}', {MISSION}, {LEVEL}, "
               f"'{esc(ev)}'::jsonb, '{BY}') RETURNING id, gate_state")
    print("verdict:", out.stdout.strip(), out.stderr.strip()[:500])


if __name__ == "__main__":
    main()