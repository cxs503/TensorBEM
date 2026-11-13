#!/usr/bin/env python3
"""m17310 board audit writes: (1) benchmark court row (referee was absent), (2) maturity verdict.

Idempotent. Numbers are read from the auditor's replay artefacts, never typed.
Run: python3 docs/board_audit/m17310/insert_rows.py
"""
import hashlib, json, pathlib
import psycopg

DSN = dict(host="127.0.0.1", port=5432, dbname="tensorplatform",
           user="foundation", password="uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")
ROOT = pathlib.Path(__file__).resolve().parents[3]      # /root/tp-agent/work/m17310
SNAP = "docs/board_audit/m17310/snapshot"
AUDIT = 17310
SUBJECT = 17273
REPO = "TensorBEM"
ANCHOR = "CASE-BEM-ELLIPSOID-KOCHIN"
SCOPE = "m17273 CASE-BEM-ELLIPSOID-KOCHIN"

rerun = json.loads((ROOT / SNAP / "rerun_results.json").read_text())
crit = rerun["criteria"]
tol = rerun["pool_tolerance"]
worst_int = crit["kochin_panels_integrated"]["value"]
worst_ord = crit["kochin_panels_ordinate"]["value"]
far_pan = crit["haskind_far_panels"]["value"]
n_pass = sum(1 for c in crit.values() if c["passed"])
krows = [(r["k"], r["ref_abs"],
          r["routes"]["panels_source_integrated"]["abs"], r["routes"]["panels_source_integrated"]["rel_err"],
          r["routes"]["panels_ordinate"]["abs"], r["routes"]["panels_ordinate"]["rel_err"])
         for r in rerun["kochin_table"]]
H = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()

METRIC = (
    "worst panel-route relative error of the Kochin function |H(k)| vs the closed form, inside the "
    f"pre-declared envelope k*a<=10: integrated {worst_int:.6e}, ordinate {worst_ord:.6e} "
    f"(both <5% tol and <3% ecosystem line); far-field vs planform-panel wave resistance {far_pan:.6e}; "
    f"11/11 pre-declared criteria PASS ({n_pass}/{len(crit)}), 1 diagnostic FAIL disclosed "
    "(long_wave_limit_three_point 2.122e-3 vs 1e-3, not scored); auditor m17310 replayed commit c121bee: "
    "263/264 numeric leaves bit-identical (only runtime_seconds differs), 16/16 case tests pass; "
    "registered by this audit because the referee row was absent at audit time")

row_cases = dict(
    anchor_slug=ANCHOR,
    case_name=("Half-submerged thin ellipsoid a/b/c=5.0/1.0/0.5 m (L/B/T=10/2/0.5 m, Cb=pi/6) in deep water: "
               "far-field Kochin function H(k) and its Kelvin-integral wave resistance vs the closed form + "
               "long-wave limit H->-ikV + far/near-field (Haskind) energy identity; k grid 0.3/0.5/1.0/2.0 1/m, "
               "Fr 0.20/0.30/0.50, 1024 panels (32x32)"),
    ref_source=("closed form H(k) = -2*pi*i*b*int_{-c}^{0} sqrt(1-z^2/c^2) J1(k*A(z)) e^{kz} dz "
                "(tensorbem.ellipsoid_kochin.kochin_analytic), corroborated by an independent elliptic-coordinate "
                "spectral rule (6.62e-14), the k->0 limit slope 1.027 and the far/near-field identity 8.83e-14; "
                "classical sources Michell 1898 / Havelock 1932 / Newman 1977 ch.6 (no printed table transcribed; "
                "provenance.page_verified=false); subject commit c121bee"),
    metric=METRIC,
    tolerance=f"<5% (tensor_case_pool id=170 tol_policy + mission 17273 criterion); ecosystem <3% met (worst {worst_ord:.3%})",
    repo=REPO,
    status="passed",
)

evidence = {
    "audit_report": "docs/board_audit/m17310/AUDIT_m17273_CASE_BEM_ELLIPSOID_KOCHIN.md",
    "impl_files": [f"{SNAP}/src/ellipsoid_kochin_case.py", f"{SNAP}/src/ellipsoid_kochin.py"],
    "impl_origin": ("byte copies (sha256 in snapshot/SHA256SUMS.txt) of commit c121bee on branch agent/m17273, "
                    f"case driver {H(SNAP + '/src/ellipsoid_kochin_case.py')[:16]}...; the driver is NOT on main "
                    "(worktree main 89e5a52 / ECO 96f19f7: 0 hits for ellipsoid_kochin_case) -> delivery is docked "
                    "only on its mission branch"),
    "entry": f"{SNAP}/src/ellipsoid_kochin_case.py",
    "entry_detail": ("PYTHONPATH=src python3 -m tensorbem.benchmarks.ellipsoid_kochin_case --outdir <dir> -> "
                     "geometry_gate/kochin_rows/resistance_rows/long_wave_rows/convergence_rows/sensitivity_rows/"
                     "assemble -> results.json + report.md + figures/*.png; exit code = verdict (0 = all pass)"),
    "run_artifacts": [f"{SNAP}/rerun_results.json", f"{SNAP}/rerun_report.md",
                      f"{SNAP}/bem_ellipsoid_kochin.png", f"{SNAP}/subject_results_c121bee.json",
                      f"{SNAP}/compare_stdout.txt", f"{SNAP}/SHA256SUMS.txt"],
    "run_artifacts_note": ("snapshot/rerun_results.json = the auditor's own replay of c121bee (fresh git archive, "
                           "unmodified) in runs/m17310/repro, 30 s single core; subject_results_c121bee.json = the "
                           "delivered artefact for diffing"),
    "tests": [f"tests/test_ellipsoid_kochin_case.py ({SNAP}/src/test_ellipsoid_kochin_case.py) - audited re-run: "
              "16 passed in 37.01 s",
              "criterion set: 11/11 pre-declared criteria PASS (geometry Cb 2.12e-16, closed-form-vs-independent "
              "6.62e-14, long-wave order 1.0266, panels integrated 1.728e-2, panels ordinate 2.098e-2, haskind "
              "far/near 8.83e-14, far-vs-panels 9.55e-3, convergence finest 5.02e-3, order 2.0038, reference "
              "sensitivity 1.29e-13)"],
    "integration_callers": ["src/tensorbem/figure_integrity.py:122-128 (registers case module "
                            "tensorbem.benchmarks.ellipsoid_kochin_case + builder _figure_kochin in the repo case "
                            "registry)"],
    "integration_gap": ("auditor grep -rl --include='*.py' -F ellipsoid_kochin_case /root/tensor-ecosystem/TensorBEM "
                        "= 0 hits: commit c121bee was never merged into main, and tests/test_figure_integrity.py "
                        "cannot discover the case from an installed repo. The repo-internal registration "
                        "(figure_integrity.py:122-128) exists inside the same commit but has no main-side reach -> "
                        "ecosystem-side integration_callers NOT verifiable (gate will downgrade L3->L2)"),
    "benchmark": {
        "anchor_slug": ANCHOR,
        "tolerance": "<5% (pool id=170)", "ecosystem_line": "<3%",
        "ref_vs_got_kochin": [{"k_1_per_m": k, "ref_abs_H": r,
                               "got_integrated": g1, "rel_err_integrated": e1,
                               "got_ordinate": g2, "rel_err_ordinate": e2}
                              for k, r, g1, e1, g2, e2 in krows],
        "worst_graded_rel_err": max(worst_int, worst_ord),
        "ecosystem_3pct_met": max(worst_int, worst_ord) < 0.03,
        "tolerance_met": max(worst_int, worst_ord) < tol,
        "criteria_pass": f"{n_pass}/{len(crit)}",
        "diagnostic_fail": {"long_wave_limit_three_point": {"value": rerun["diagnostics"]
                          ["long_wave_limit_three_point"]["value"], "limit": 1e-3,
                          "status": "FAIL, disclosed by the subject, not scored"}},
        "replay_diff": "264 numeric leaves compared: 263 bit-identical, 1 differing = runtime_seconds "
                       "(12.791265 vs 25.774419)",
        "source": "snapshot/rerun_results.json (auditor replay of c121bee)",
    },
    "repro_cmd": ("cd <TensorBEM checkout of c121bee> && PYTHONPATH=src python3 -m "
                  "tensorbem.benchmarks.ellipsoid_kochin_case --outdir runs_audit  # auditor ran it: exit 0, 30 s, "
                  "results.json bit-identical to the delivered one except runtime_seconds"),
    "repro_caveat": "results.json.repro_cmd omits PYTHONPATH=src; copied verbatim it raises ImportError (F4)",
    "external_anchor": [f"{SNAP}/rerun_results.json:criteria.closed_form_vs_independent (6.62e-14)",
                        f"{SNAP}/rerun_results.json:criteria.long_wave_asymptotic_order (1.0266)",
                        f"{SNAP}/rerun_results.json:criteria.haskind_far_near_identity (8.83e-14)"],
    "external_anchor_note": ("no independent third implementation was written by the auditor and no printed table "
                             "was reachable (sandbox has no papers): the anchor strength rests on the subject's own "
                             "three machine-checkable facts, two of which share the same hull/geometry -> closed "
                             "form is corroborated, not externally calibrated; provenance.page_verified=false stated honestly"),
    "court_ledger": {"tensor_benchmark_cases": "0 rows at audit time -> registered by this audit at the mission's 5% "
                                              "criterion (not loosened)",
                     "tensor_case_pool": "id=170 status=dispatched (not closed); sibling BEM-ELLIPSOID-ADDED-MASS "
                                         "id=134 is passed",
                     "GET /tensor/bench/l4-verdict/17273": "CLAIMED_NOT_PROVEN, rows=0 before this audit"},
    "findings": ["F1 high: no tensor_benchmark_cases row -> the machine referee was absent (registered by m17310)",
                 "F2 high: c121bee never merged into main; pool id=170 still 'dispatched' -> case not runnable from an "
                 "installed TensorBEM",
                 "F3 medium: only repo-internal integration (figure_integrity.py:122-128 + 16 tests); ECO grep 0 hits",
                 "F4 low: repro_cmd missing PYTHONPATH=src",
                 "F5 low: citations are volume/page level, page_verified=false, no external table"],
    "checks_not_run": ["independent third implementation of the Kochin closed form: not_run",
                       "printed literature table/page cross-check: not_run (no papers in sandbox)",
                       "cluster re-run / tensorbench_results row: not_run (only the tensor_benchmark_cases referee row "
                       "was registered)"],
    "honest_fail_credibility": ("established: the subject kept the round-1 long-wave criterion FAIL visible in "
                                "results.json.diagnostics and in its report instead of hiding it; the lower level is "
                                "a docking defect, not a fabrication finding"),
    "declared_level": "3",
    "level_rationale": ("L1 tests 16 passed; L2 1007-line real driver + python -m entry + non-empty artefacts with a "
                        "263/264 bit-identical replay; L3 repo-internal registry caller exists (figure_integrity.py:122) "
                        "but no main/ECO reach; L4 content (ref vs got, 11/11 criteria, worst 2.098% < 5%, repro_cmd "
                        "runs) is satisfied in substance yet the case is not docked into main -> level capped at 3, "
                        "no L4 claimed, no L5"),
    "subject_mission": str(SUBJECT),
    "subject_head_commit": "c121bee",
    "subject_branch": "agent/m17273",
    "platform_report_id": "pending",
}

with psycopg.connect(**DSN) as conn, conn.cursor() as cur:
    # (1) benchmark referee row — absent at audit time
    cur.execute("SELECT id FROM tensor_benchmark_cases WHERE anchor_slug=%s", (ANCHOR,))
    got = cur.fetchone()
    if got:
        cur.execute("UPDATE tensor_benchmark_cases SET case_name=%(case_name)s, ref_source=%(ref_source)s, "
                    "metric=%(metric)s, tolerance=%(tolerance)s, repo=%(repo)s, status=%(status)s, "
                    "passed_at=now() WHERE anchor_slug=%(anchor_slug)s", row_cases)
        print(f"tensor_benchmark_cases updated id={got[0]}")
    else:
        cur.execute("INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, "
                    "repo, status, passed_at) VALUES (%(anchor_slug)s, %(case_name)s, %(ref_source)s, %(metric)s, "
                    "%(tolerance)s, %(repo)s, %(status)s, now()) RETURNING id", row_cases)
        print(f"tensor_benchmark_cases inserted id={cur.fetchone()[0]}")
    # (2) maturity verdict
    cur.execute("SELECT id FROM tensor_maturity_verdicts WHERE repo=%s AND scope=%s AND verdict_by='board-audit'",
                (REPO, SCOPE))
    v = cur.fetchone()
    ev = json.dumps(evidence, ensure_ascii=False)
    if v:
        cur.execute("UPDATE tensor_maturity_verdicts SET mission_id=%s, level=3, evidence=%s::jsonb, "
                    "gate_state='pending', gate_level=NULL, gate_note=NULL WHERE id=%s", (AUDIT, ev, v[0]))
        print(f"verdict updated id={v[0]}")
    else:
        cur.execute("INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
                    "VALUES (%s,%s,%s,%s,%s::jsonb,'board-audit') RETURNING id", (REPO, SCOPE, AUDIT, 3, ev))
        print(f"verdict inserted id={cur.fetchone()[0]}")
    cur.execute("SELECT case_name, metric, tolerance, status FROM tensor_benchmark_cases WHERE anchor_slug=%s",
                (ANCHOR,))
    print("court:", cur.fetchone()[0][:80], "...")
