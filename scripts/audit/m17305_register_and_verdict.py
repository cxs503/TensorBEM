#!/usr/bin/env python3
"""m17305 board-audit: enrol the benchmark row and write the maturity verdict.

1. tensor_benchmark_cases: register anchor CASE-BEM-NEUMANN-KELVIN (was 0 rows) from
   tensor_case_pool id=167 tol_policy, with the auditor's measured spread.
2. tensor_maturity_verdicts: insert the audited verdict (L4) with real evidence pointers.
3. Dry-run the platform gate (`maturity_gate.judge`) on that evidence and print the result.

Every number written here is read from runs/m17305/audit_numbers.json (no hand copy).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WT = Path("/root/tp-agent/work/m17305")
SUBJ = "/root/tp-agent/work/m17246"
AUD = json.loads((WT / "runs/m17305/audit_numbers.json").read_text())
ROW = [r for r in AUD["rows"]]


def psql(sql: str) -> str:
    r = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform",
                        "-tA", "-F", "\x01", "-c", sql], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:800])
    return r.stdout.strip()


def esc(s: str) -> str:
    return s.replace("'", "''")


def main() -> int:
    errs = ", ".join(f"Fr={r['Fr']:.2f}:{100*r['err_got_vs_independent_ref']:+.4f}%" for r in ROW)
    bench = {
        "anchor_slug": "CASE-BEM-NEUMANN-KELVIN",
        "ref": ("Michell thin-ship Cw(Fr), Phil. Mag. 45 (1898) 106-123, via the arbitrated "
                "tensorbem.wigley.michell_wave_resistance_canonical; referee row enrolled by this audit "
                "and independently re-derived by audit m17305 (own Gauss-Legendre quadrature in (x,z), "
                "no TensorBEM import)"),
        "tolerance": "<5% (tensor_case_pool id=167 tol_policy); ecosystem <3%",
        "max_rel_error": AUD["max_err_got_vs_independent_ref"],
        "max_rel_error_vs_delivered_ref": max(abs(r["err_got_vs_delivered_ref"]) for r in ROW),
        "rows": {f"Fr={r['Fr']:.2f}": {"ref": r["ref_delivered"], "ref_auditor": r["ref_independent"],
                                      "got": r["got"],
                                      "err_vs_auditor_ref": r["err_got_vs_independent_ref"]} for r in ROW},
        "ref_route_spread_max": AUD["max_ref_spread"],
        "route_a_all_passed": AUD["route_a_all_passed_delivered"],
        "route_b": {"max_abs_error": AUD["route_b"]["max_abs_error"],
                    "any_passed": AUD["route_b"]["any_passed"]},
        "artifact": "runs/m17305/audit_numbers.json, docs/board_audit/m17305/table_audit.md",
        "auditor_error_summary": errs,
    }
    ev = {
        "impl_files": [f"{SUBJ}/src/tensorbem/nk_panel3d.py",
                       f"{SUBJ}/scripts/case_bem_neumann_kelvin_michell.py"],
        "entry": f"{SUBJ}/scripts/case_bem_neumann_kelvin_michell.py",
        "run_artifacts": [f"{SUBJ}/docs/case_bem_neumann_kelvin/{f}" for f in
                          ("numbers.json", "table.md", "fig_cw_vs_froude.png",
                           "fig_route_a_convergence.png", "fig_kelvin_wake.png", "wave_field.npz")],
        "tests": [
            "tests/test_case_neumann_kelvin_michell.py — auditor rerun 2026-11-13 on 27ecfe7 copy: 5 passed in 5.68 s",
            "tests/test_nk_panel3d.py + tests/test_neumann_kelvin.py — auditor rerun: 64 passed in 18.42 s "
            "(regression surface of the changed default prefactor)",
        ],
        "integration_callers": ["thin_ship_reference"],
        "integration_callers_detail": ("nk_panel3d.wave_resistance_far <- thin_ship_reference "
                                       "(src/tensorbem/nk_panel3d.py:922) <- caller "
                                       "scripts/nk_panel3d_study.py:199; grep hit in ECO TensorBEM main"),
        "benchmark": bench,
        "repro_cmd": ("cd <TensorBEM @ 27ecfe7> && PYTHONPATH=src python3 "
                      "scripts/case_bem_neumann_kelvin_michell.py --stage cheap "
                      "--only geometry,kernel,route_a,route_b,wave_field"),
        "external_anchor": f"{SUBJ}/docs/michell_exit_arbitration.md",
        "external_anchor_note": ("Michell 1898 Phil. Mag. 45:106-123 (thin-ship theory). Auditor-side "
                                 "independent re-derivation (scripts/audit/m17305_independent_michell.py, "
                                 "runs/m17305/independent_michell.json) agrees with the delivered ref to "
                                 f"{100*AUD['max_ref_spread']:.4f}%; no experimental towing-tank anchor used, "
                                 "so L5 is NOT claimed"),
        "audit": {
            "mission": 17305, "subject_mission": 17246, "subject_commit": "27ecfe7",
            "repro_verification": ("repro_cmd re-run on a copy of 27ecfe7 reproduced every numeric leaf of "
                                   "numbers.json (only generated_at_utc/git_head/python/platform/seconds/"
                                   "blocks_run differ)"),
            "findings": [
                "F1 high(platform): tensor_benchmark_cases had 0 rows for the anchor -> the machine referee "
                "did not exist; enrolled by this audit from case pool id=167 (tol <5%). Same break as "
                "verdicts #493/#499/#509/#514; root cause (case-pool dispatch not writing the benchmark row) unfixed.",
                "F2 medium(provenance): numbers.json records git_head=0516776 (pre-fix parent, no kochin_prefactor) "
                "and blocks_run=['route_b'] -> artifact is a merge of partial invocations from a dirty tree; "
                "attribution established only by the auditor re-run at 27ecfe7.",
                "F3 medium(delivery): 27ecfe7 lives only on branch agent/m17246; origin/main has neither "
                "docs/case_bem_neumann_kelvin/ nor kochin_prefactor -> not reproducible from a clean main checkout "
                "until the platform pushes the branch (process fact, not an author defect).",
                "F4 medium(judging boundary): Route A (thin-ship sheet into the steady Kochin operator) and the "
                "ref share analytic structure, so the 0.197-2.309% agreement certifies steady dispersion + "
                "residue constant kappa/(2 pi g) + normalisation + centreplane panel convergence, NOT the hull "
                "single-layer NK BEM (Route B: +1399%..+4957%, diverges under refinement). Disclosed by the "
                "delivery in section 6; audit confirms the disclosure is accurate. L4 is claimed only in this "
                "reduced sense.",
                "F5 medium(existing debt, confirmed): benchmarks/verified/bem_wigley_wave_resistance/results.json "
                "carries refs from the defective legacy exit wigley.michell_wave_resistance: 2.5752/12.2926/"
                "26.3239 N at Fr=0.20/0.30/0.40 vs correct 26.5393/144.1357/327.1101 (10.31x/11.73x/12.43x low, "
                "Fr-dependent). Reissue mission required.",
                "F6 low(artifact consistency): richardson_rw_extrapolated/observed_order = NaN (2-rung ladder; "
                "report states order ~1.15 in prose only); the quoted radiation ratio 3.02 is not stored in "
                "numbers.json (only downstream 0.9858) - auditor re-ran verify_radiation_direction and got "
                "3.019989, so the number is right but unregistered; wave-pattern half-angle 74.60 deg vs theory "
                "19.47 deg is declared qualitative by the delivery.",
            ],
            "not_run": ["--stage full ladder", "cluster/loong64 independent replay",
                        "independent wave-angle estimator", "experimental (towing tank) external comparison"],
            "disposition": "approve with notes",
            "checks_run": ["diff review of 27ecfe7", "repro_cmd re-run (bit-identical numeric leaves)",
                           "pytest 5 + 64 passed", "independent Michell quadrature (no TensorBEM import)",
                           "legacy exit / verified-benchmark ref audit", "verify_radiation_direction both modes"],
            "artifacts": ["docs/board_audit/m17305/REPORT.md", "docs/board_audit/m17305/table_audit.md",
                          "docs/board_audit/m17305/fig_audit_cw_vs_fr.png",
                          "runs/m17305/audit_numbers.json", "runs/m17305/independent_michell.json",
                          "runs/m17305/repro_tree/docs/case_bem_neumann_kelvin/numbers.json"],
        },
    }
    (WT / "docs/board_audit/m17305/verdict.json").write_text(json.dumps(ev, indent=1, ensure_ascii=False))

    METRIC = (bench["auditor_error_summary"] + " vs the auditor independent Michell quadrature; max "
              + format(100 * bench["max_rel_error"], ".4f")
              + "% (<5% tol, <3% ecosystem); subject m17246 commit 27ecfe7, audited by m17305")

    # -- 1) enrol the benchmark referee row (idempotent) --------------------
    have = psql("SELECT count(*) FROM tensor_benchmark_cases "
                "WHERE anchor_slug='CASE-BEM-NEUMANN-KELVIN'")
    if have.strip() == "0":
        psql("INSERT INTO tensor_benchmark_cases "
             "(anchor_slug,case_name,ref_source,metric,tolerance,repo,status,passed_at) VALUES ("
             "'CASE-BEM-NEUMANN-KELVIN',"
             "'Wigley III parabolic hull L=10 B=1 T=0.625 m (Cb=4/9), deep water, Fr=0.20/0.25/0.30/0.35/0.40: "
             "steady Neumann-Kelvin wave resistance Cw(Fr)',"
             f"'{esc(bench['ref'])}',"
             f"'{esc(METRIC)}',"
             f"'{esc(bench['tolerance'])}',"
             "'TensorBEM','passed',now())")
        print("benchmark row inserted")
    else:
        print("benchmark row already present")

    # -- 2) verdict row (idempotent on repo+scope+verdict_by) --------------
    scope = "m17246 CASE-BEM-NEUMANN-KELVIN"
    psql("INSERT INTO tensor_maturity_verdicts (repo,scope,mission_id,level,evidence,verdict_by) VALUES ("
         f"'TensorBEM','{esc(scope)}',17305,4,'{esc(json.dumps(ev, ensure_ascii=False))}','board-audit') "
         "ON CONFLICT (repo,scope,verdict_by) DO UPDATE SET mission_id=EXCLUDED.mission_id,"
         "level=EXCLUDED.level,evidence=EXCLUDED.evidence,gate_state='pending',gate_note=NULL")
    vid = psql(f"SELECT id,level,gate_state FROM tensor_maturity_verdicts WHERE repo='TensorBEM' "
               f"AND scope='{esc(scope)}' AND verdict_by='board-audit'")
    print("verdict row:", vid)

    # -- 3) gate dry-run ---------------------------------------------------
    sys.path.insert(0, "/root/tp-agent")
    import maturity_gate  # noqa: E402
    gl, why = maturity_gate.judge(4, ev, "TensorBEM", 17305)
    print(f"gate dry-run: claimed=4 gate_level={gl} why={why}")
    print("original mission resolution:", maturity_gate.link_original(scope, ev, "TensorBEM"))
    return 0 if gl == 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())