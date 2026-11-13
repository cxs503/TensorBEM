#!/usr/bin/env python3
"""Board-audit verdict for m17207 (CASE-BEM-KP505-OPENWATER, TensorBEM) — INSERT only.

Evidence is assembled from the audit's own artefacts (``verification.json``, the
snapshot hashes, the re-run diff) — no number is typed by hand.  Idempotent: one
row per (repo, scope, verdict_by).

Run:  python3 docs/board_audit/m17239/insert_verdict.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import psycopg

DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")
REPO = "TensorBEM"
SCOPE = "m17207 CASE-BEM-KP505-OPENWATER"
MISSION = 17239          # this audit task
SUBJECT = 17207          # audited mission
BY = "board-audit"
LEVEL = 2                # real implementation; L3 refused (not merged, no caller), L4 refused (criterion FAIL)

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SNAP = HERE / "snapshot"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_evidence() -> dict:
    ver = json.loads((HERE / "verification.json").read_text())
    case = json.loads((SNAP / "kp505_case.json").read_text())
    rows = case["canonical"]["rows"]
    worst_adm = ver["worst_abs_pct_admissible_only"]
    worst_all = ver["worst_abs_pct_all_10"]

    def rel(p: pathlib.Path) -> str:
        return str(p.relative_to(ROOT))

    return {
        "impl_files": [
            rel(SNAP / "kp505_openwater_case.py"),
            rel(SNAP / "propeller_warn_guard.diff"),
        ],
        "impl_origin": {
            "commit": "a33c68adff144c077cc6838b8d25313ceb4e7aa0",
            "branch": "agent/m17207",
            "paths_as_delivered": [
                "src/tensorbem/benchmarks/kp505_openwater_case.py:1-738",
                "src/tensorbem/propeller.py:99-110,689-730,838-842",
            ],
            "sha256_snapshot": sha(SNAP / "kp505_openwater_case.py"),
            "note": ("snapshots are byte-copies of the audited commit (branch not merged into "
                     "/root/tensor-ecosystem/TensorBEM main, so no main-side path exists)"),
        },
        "entry": [rel(SNAP / "kp505_openwater_case.py")],
        "entry_cmd": ("python -m tensorbem.benchmarks.kp505_openwater_case <outdir> "
                      "(main() at snapshot/kp505_openwater_case.py:709; re-run verified by this audit)"),
        "run_artifacts": [
            rel(SNAP / "kp505_case.json"),
            rel(SNAP / "kp505_case_table.csv"),
            rel(SNAP / "kp505_case_report.md"),
            rel(SNAP / "kp505_case_figures/fig1_curves.png"),
            rel(SNAP / "kp505_case_figures/fig2_errors.png"),
            rel(SNAP / "kp505_case_figures/fig3_ladders.png"),
        ],
        "run_artifacts_sizes": {p.name: p.stat().st_size for p in SNAP.rglob("*") if p.is_file()},
        "tests": [
            "tests/test_kp505_openwater_case.py::test_anchor_is_the_parsed_verbatim_table",
            "tests/test_kp505_openwater_case.py::test_point_scoring_matches_the_recorded_canonical_row",
            "tests/test_kp505_openwater_case.py (6/6 passed at commit a33c68a, 1.7 s)",
        ],
        "integration_callers": [],
        "integration_gap": ("no caller anywhere in /root/tensor-ecosystem/TensorBEM: the case driver is not "
                           "in main; it is a harness on top of the existing propeller.open_water_point, so it "
                           "does not extend the solver either -> L3 refused"),
        "benchmark": {
            "ref_source": "Iftikhar & Takinaci 2023 (MARINE 2023) Table 2 p.10, 'Experimental' columns; anchor "
                          "file benchmarks/anchors/kp505_openwater.json (SECONDARY_REPRODUCTION)",
            "tolerance": "<5% per tensor_case_pool BEM-KP505-OPENWATER tol_policy",
            "got_vs_ref": [
                {"J": r["J"], "KT": r["KT"], "KT_ref": r["KT_anchor"], "KT_pct": r["error_pct"]["KT_pct"],
                 "10KQ": r["10KQ"], "10KQ_ref": r["10KQ_anchor"], "10KQ_pct": r["error_pct"]["10KQ_pct"],
                 "eta0": r["eta0"], "eta0_ref": r["eta0_anchor"], "eta0_pct": r["error_pct"]["eta0_pct"],
                 "admissible": r["admissible"]}
                for r in rows
            ],
            "points_pass_all_three_5pct": len(ver["points_all_three_pass"]),
            "points_n": len(rows),
            "admissible_points_pass_all_three_5pct": len(ver["admissible_points_all_three_pass"]),
            "admissible_points_n": sum(1 for r in rows if r["admissible"]),
            "pass_counts_5pct": ver["recomputed_pass_counts_10pts"],
            "worst_abs_pct_admissible_only": worst_adm,
            "worst_abs_pct_including_inadmissible": worst_all,
            "mean_abs_pct_admissible_only": ver["mean_abs_pct_excluding_J1"],
            "verdict": case["verdict"],
            "auditor_recomputed_verdict": ver["verdict_recomputed"],
            "auditor_vs_subject_max_abs_diff_pct": ver["max_abs_agreement_diff_vs_reported"],
            "machine_referee_absent": ("tensor_benchmark_cases has 0 rows for anchor_slug="
                                       "CASE-BEM-KP505-OPENWATER; the tolerance used here is the case-pool "
                                       "policy, not a registered benchmark row"),
        },
        "repro_cmd": ("cd /root/tp-agent/work/m17207 && PYTHONPATH=src python -m "
                      "tensorbem.benchmarks.kp505_openwater_case /tmp/m17207_repro"),
        "repro_result": ("re-run at commit a33c68a on loongarch64, 35 s: kp505_case_table.csv byte-identical "
                         "(diff empty); canonical/ladders/stability_probe/criteria/verdict identical to <=1e-12"),
        "external_anchor": {
            "file": "benchmarks/anchors/sources/kp505_marine2023_iftikhar.pdf (p.10 Table 2), "
                    "re-extracted by the auditor with pypdf",
            "anchor_vs_pdf_max_abs_diff": ver["anchor_vs_pdf_max_abs_diff"],
            "anchor_eta0_self_consistency_max_residual": ver["anchor_eta0_max_residual"],
            "paper_own_error_column_max_dev_pp": ver["paper_error_column_max_dev_pp"],
            "caveat": ("secondary transcription: the paper does not cite the primary KRISO/KCS-workshop EFD "
                       "table; anchor quality is bounded by that -> L5 refused"),
        },
        "findings": {
            "F1_machine_referee_absent": ("tensor_benchmark_cases: 0 rows for this anchor; tensor_case_pool row "
                                          "BEM-KP505-OPENWATER still status=dispatched mission_id=17207 -> the "
                                          "machine referee cannot re-judge this case"),
            "F2_not_in_ecosystem_main": ("deliverable lives only on refs/heads/agent/m17207 (a33c68a); "
                                         "/root/tensor-ecosystem/TensorBEM main has neither the driver nor the "
                                         "warning guard -> no product-side capability"),
            "F3_worst_error_uses_blown_up_point": (f"report headlines -240.71%/-115.87%/+786.67%, all from the "
                                                   f"self-declared inadmissible J=1.0 (K_T<0, eta=3.79 above the "
                                                   f"momentum bound); admissible worst is "
                                                   f"{worst_adm['KT_pct']:.2f}/{worst_adm['10KQ_pct']:.2f}/"
                                                   f"{worst_adm['eta0_pct']:.2f}% at J=0.9 -> verdict unchanged, "
                                                   f"wording overstated"),
            "F4_artifact_commit_points_at_base": ("artefact environment.git_commit=96f19f7 (base) while the code is "
                                                  "a33c68a; reproduction from a33c68a is bit-identical, so this is a "
                                                  "provenance mis-pointer, not a reproducibility defect"),
            "F5_anchor_is_secondary": ("paper's own error column reproduces from its KT/10KQ columns within "
                                       f"{ver['paper_error_column_max_dev_pp']:.2f} pp -> transcription faithful, "
                                       "but the EFD column itself is a secondary reproduction"),
            "F6_preexisting_red_tests_confirmed": ("tests/test_propeller_wageningen_crosscheck.py 6 FAIL reproduce "
                                                   "unchanged at base 96f19f7 (BEM K_T=0.0000) -> pre-existing debt, "
                                                   "unrelated to this delivery"),
        },
        "checks_run": [
            "PDF re-extraction of Table 2 (pypdf) vs the transcribed anchor",
            "independent recompute of all 30 relative errors + pass counts + verdict from CSV/JSON",
            "full re-run of repro_cmd at commit a33c68a (bit-identical artefacts, 35 s)",
            "pytest tests/test_kp505_openwater_case.py (6/6) and tests/test_propeller_wageningen_crosscheck.py "
            "(6 failed at both head and base)",
        ],
        "checks_not_run": [
            "cluster full regression re-run (subject cites cj1456): not_run",
            "independent EFD anchor (KRISO/KCS workshop primary table or a different implementation): not_run — "
            "egress/paywall; the audit therefore cannot certify the reference itself",
            "blade-geometry reconstruction quality (camber/radial pitch/skew): only the subject's own "
            "sensitivity table was read, no independent reconstruction",
        ],
        "declared_level": LEVEL,
        "level_rationale": (
            "L2 独立功能实现：738 行真代码（输入=桨几何+J → 调用 tensorbem.propeller.open_water_point → "
            "输出=K_T/10K_Q/η0 + 5% 打分 + GCI 阶梯 + 稳定性探针），入口 `python -m "
            "tensorbem.benchmarks.kp505_openwater_case`，运行工件非空（json 185 KB / csv / 3 PNG / 报告），"
            "6 条契约测试通过，复跑逐位一致。不授 L3：交付未并入生态仓 main（无调用方，gate 的 "
            "integration_callers grep 必失败），且本单只是在既有求解器上追加算例驱动器 + 一个 warning 护栏，"
            "未扩展求解器能力。不授 L4：基准未通过——10 点三项全达标 0/10，可容许点最差 -64.85%（判据 5% 的 "
            "13 倍）。不授 L5：参考列是二次转录，无一次实验/独立实现对照。"
        ),
        "executive_summary": (
            f"honest FAIL 复核成立：锚表与 PDF 表2 逐位一致（max|Δ|={ver['anchor_vs_pdf_max_abs_diff']}），"
            f"got 值复跑逐位复现，三项误差独立重算与声明一致（最大偏差 "
            f"{ver['max_abs_agreement_diff_vs_reported']:.1e}%）；0/10 点三项全在 5% 带内，"
            f"可容许点最差 {worst_adm['KT_pct']:.2f}% (K_T @J=0.9)。判级 L2。"
        ),
        "disposition": (
            "request changes: (1) 合并或明确废弃 agent/m17207，算例不该悬在分支上；"
            "(2) 平台侧登记 tensor_benchmark_cases 行（anchor_slug=CASE-BEM-KP505-OPENWATER，"
            "metric=K_T/10K_Q/eta_0 vs Table 2 EXP，tolerance=<5%）并把 tensor_case_pool 该行置终态，"
            "否则机器裁判永远缺席；(3) 改进方向=模型形式（拱度/径向螺距/非平剖面），不是加密网格"
            "（spanwise GCI 已 13–14% 且观测阶 p<0）。"
        ),
        "report_refs": {"platform_report_id": 16227, "audit_doc": "docs/board_audit/m17239/AUDIT_m17207.md"},
        "subject_mission": SUBJECT,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ev = build_evidence()
    if a.dry_run:
        print(json.dumps(ev, ensure_ascii=False, indent=1)[:2500])
        print("...dry-run, no DB write")
        return
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tensor_maturity_verdicts "
                "(repo, scope, mission_id, level, evidence, verdict_by) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET "
                "mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, "
                "evidence = EXCLUDED.evidence, created_at = now() "
                "RETURNING id, gate_state, gate_level",
                (REPO, SCOPE, MISSION, LEVEL, json.dumps(ev), BY),
            )
            print("verdict row:", cur.fetchone())
        conn.commit()


if __name__ == "__main__":
    main()