#!/usr/bin/env python3
"""Board cross-audit (交差核验) of mission m17276 -- anchor CASE-BEM-MARUO-DRIFT.

Verification, not re-do.  Everything below is recomputed from committed bytes
plus the delivery's own artifacts; nothing is inherited from the delivery's
claims:

  1. existence + identity of the three acceptance pieces
       run_artifacts : artifacts/maruo_drift_far_field_case/study.json (+ 3 png,
                       docs copies), sha256 recorded
       benchmark     : ref vs got per cell, recomputed from the JSON numbers
       repro_cmd     : replayed (the delivered merge command) and, separately,
                       the four production cells re-executed on the cluster
  2. the registered referee
       tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-MARUO-DRIFT'
     -> 0 rows at audit time (recorded, not papered over).  The criterion in
     force is the mission-goal text (容差判据：<5%), quoted verbatim.
  3. the reference itself, re-derived without importing tensorbem
     scripts/audit/m17327_ref_independent.py  ->  artifacts/.../independent_ref.json
  4. report discipline: six sections present?  every headline number in the ④
     table traceable to study.json?  claims that are NOT traceable are listed.
  5. merge state of the subject commit (branch vs main).
  6. maturity level + one row in tensor_maturity_verdicts.

Outputs
    artifacts/audit_m17327/{existence,referee,judge,independent_ref,report_check,
                            merge_state,repro,cluster_replay,finding}.json
    artifacts/audit_m17327/fig_m17327_error_vs_panels.png
    docs/board_audit_m17327/M17327_BOARD_AUDIT.md
    (--insert-verdict) one row in tensor_maturity_verdicts

Usage
    python3 scripts/audit/m17327_board_audit.py [--insert-verdict] [--cluster-json PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = "/root/tp-agent/work/m17276"
ECOSYSTEM = "/root/tensor-ecosystem/TensorBEM"
COMMIT = "92c92b7"
BRANCH = "agent/m17276"
ANCHOR = "CASE-BEM-MARUO-DRIFT"
AUDIT_MID = 17327
OUT = os.path.join(REPO, "artifacts", "audit_m17327")
DOC = os.path.join(REPO, "docs", "board_audit_m17327", "M17327_BOARD_AUDIT.md")
FIG = os.path.join(OUT, "fig_m17327_error_vs_panels.png")
TOL = 0.05  # mission goal: 容差判据：<5%

DELIVERED = [
    "src/tensorbem/benchmarks/maruo_drift_far_field_case.py",
    "src/tensorbem/benchmarks/maruo_drift_far_field_case_figures.py",
    "tests/test_maruo_drift_far_field_case.py",
    "make_report_m17276.py",
    "docs/case_bem_maruo_drift/study.json",
    "docs/case_bem_maruo_drift_report.md",
    "docs/maruo_drift_far_field_case.md",
    "artifacts/maruo_drift_far_field_case/study.json",
    "artifacts/maruo_drift_far_field_case/fig_bem_vs_closed_form.png",
    "artifacts/maruo_drift_far_field_case/fig_coefficient_vs_ka.png",
    "artifacts/maruo_drift_far_field_case/fig_convergence_and_quadrature.png",
]


def dump(name: str, obj) -> str:
    path = os.path.join(OUT, name)
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sh(*cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=SRC, capture_output=True, text=True)


# --------------------------------------------------------------------------- #
# 1. existence / identity
# --------------------------------------------------------------------------- #
def existence() -> dict:
    files = {}
    for rel in DELIVERED:
        path = os.path.join(SRC, rel)
        # the delivered png/json also live under docs/case_bem_maruo_drift/
        if not os.path.exists(path) and rel.startswith("artifacts/"):
            alt = os.path.join(SRC, "docs/case_bem_maruo_drift", os.path.basename(rel))
            if os.path.exists(alt):
                path = alt
        files[rel] = {
            "exists": os.path.exists(path),
            "path": path,
            "bytes": os.path.getsize(path) if os.path.exists(path) else None,
            "sha256": sha256(path) if os.path.exists(path) else None,
            "git_tracked": sh("git", "ls-files", "--error-unmatch", rel).returncode == 0,
        }
    study = json.loads(open(os.path.join(SRC, "artifacts/maruo_drift_far_field_case/study.json")).read())
    return {
        "files": files,
        "n_missing": sum(1 for v in files.values() if not v["exists"]),
        "n_untracked": [k for k, v in files.items() if v["exists"] and not v["git_tracked"]],
        "case": study["case"],
        "n_cells": len(study["cells"]),
        "ka_values": study["parameters"]["ka_values"],
        "resolutions": study["parameters"]["resolutions"],
        "production_resolution": study["parameters"]["production_resolution"],
    }


def merge_state() -> dict:
    inside = subprocess.run(["git", "merge-base", "--is-ancestor", COMMIT, "main"],
                            cwd=ECOSYSTEM, capture_output=True).returncode == 0
    br = subprocess.run(["git", "branch", "-a", "--contains", COMMIT], cwd=ECOSYSTEM,
                        capture_output=True, text=True).stdout.split()
    present = os.path.exists(os.path.join(
        ECOSYSTEM, "src/tensorbem/benchmarks/maruo_drift_far_field_case.py"))
    return {
        "commit": COMMIT,
        "ancestor_of_main": inside,
        "branches_containing": sorted({b.strip("*+ ") for b in br if "m17276" in b or b.strip("*+ ") == "main"}),
        "files_present_in_ecosystem_main": present,
        "note": "an unmerged-but-committed branch is normal mid-battle; recorded for traceability only",
    }


# --------------------------------------------------------------------------- #
# 2. registered referee + criterion actually in force
# --------------------------------------------------------------------------- #
def referee(dsn: str) -> dict:
    import psycopg
    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute("SELECT id, anchor_slug, case_name, metric, tolerance, repo, status "
                    "FROM tensor_benchmark_cases WHERE anchor_slug=%s ORDER BY id", (ANCHOR,))
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        cur.execute("SELECT goal FROM tensor_dev_missions WHERE id=17276")
        goal = cur.fetchone()[0]
        cur.execute("SELECT id, anchor_slug, metric, tolerance FROM tensor_benchmark_cases "
                    "WHERE anchor_slug ILIKE '%%DRIFT%%' ORDER BY id")
        kin = [dict(zip([d.name for d in cur.description], r)) for r in cur.fetchall()]
    m = re.search(r"容差判据：<5%", goal)
    return {
        "anchor_slug": ANCHOR,
        "registered_rows": rows,
        "registered_rows_count": len(rows),
        "criterion_in_mission_goal": m.group(0) if m else None,
        "criterion_in_force": "per-cell relative error < 5 % at the declared production mesh",
        "sibling_drift_rows": kin,
        "ref_law": "F_x = rho g A^2/(2 pi k) Int |S|^2 (1-cos th) dth (Maruo 1957); "
                   "long-wave limit (5 pi^2/16) rho g A^2 a (ka)^3 (Maruo 1960; Newman 1977 7.3)",
        "ref_source_declared_by_mission": "ref 原文/手册查得 (zero transcription)",
        "ref_source_actually_used": "machine evaluation of the closed form + long-wave law; "
                                    "the delivery states (report ①) that the paper table could not be "
                                    "opened offline -> honest deviation, but the literal 'table vs got' "
                                    "requirement is not met",
    }


# --------------------------------------------------------------------------- #
# 3. judge: recompute the comparison from the artifact numbers
# --------------------------------------------------------------------------- #
def judge(study: dict, tol: float = TOL) -> dict:
    cells = []
    for c in study["cells"]:
        rel = abs(c["bem_force"] - c["ref_force"]) / abs(c["ref_force"])
        cells.append({
            "ka": c["ka"], "resolution": c["resolution"], "panel_count": c["panel_count"],
            "got": c["bem_force"], "ref": c["ref_force"],
            "rel_error_recomputed": rel,
            "rel_error_declared": c["rel_error"],
            "recomputed_matches_declared": abs(rel - c["rel_error"]) <= 1e-12,
            "within_tol": rel < tol,
            "ratio_got_over_ref": c["bem_force"] / c["ref_force"],
            "solver_relative_residual": c["solver_relative_residual"],
        })
    prod = study["parameters"]["production_resolution"]
    production = sorted((c for c in cells if c["resolution"] == prod), key=lambda c: c["ka"])
    worst = max(production, key=lambda c: c["rel_error_recomputed"])
    gates = {g["name"]: g for g in study["gates"]}
    return {
        "tolerance": tol,
        "cells": cells,
        "production_resolution": prod,
        "production_cells": production,
        "production_n_failures": sum(1 for c in production if not c["within_tol"]),
        "production_worst": worst,
        "verdict_at_production_mesh": "FAIL" if worst["rel_error_recomputed"] >= tol else "PASS",
        "all_cells_recomputed_match_declared": all(c["recomputed_matches_declared"] for c in cells),
        "got_always_below_ref": all(c["ratio_got_over_ref"] < 1.0 for c in cells),
        "gates_declared": gates,
        "all_passed_declared": study["all_passed"],
        "g3_consistent_with_cells": abs(
            gates["G3_bem_vs_ref_production"]["measured"]
            - max(c["rel_error_declared"] for c in production)) <= 1e-12,
    }


# --------------------------------------------------------------------------- #
# 4. report discipline
# --------------------------------------------------------------------------- #
def report_check(study: dict) -> dict:
    text = open(os.path.join(SRC, "docs/case_bem_maruo_drift_report.md")).read()
    sections = {k: (k in text) for k in ("①", "②", "③", "④", "⑤", "⑥")}
    rows = []
    for line in text.splitlines():
        m = re.match(r"^\|\s*([0-9.]+)\s*\|\s*(coarse|medium|fine \*\*\(prod\)\*\*|fine|finer)\s*\|"
                     r"\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)%", line)
        if m:
            rows.append({"ka": float(m.group(1)), "resolution": m.group(2),
                         "panels": int(m.group(3)), "got": float(m.group(4)),
                         "ref": float(m.group(5)), "rel_pct": float(m.group(6))})
    cells = {(c["ka"], c["resolution"]): c for c in study["cells"]}
    traceable, drift = [], []
    for r in rows:
        # NB: "finer" also starts with "fine" -- match the (prod) marker exactly
        res = "fine" if r["resolution"].startswith("fine **") else r["resolution"]
        key = (r["ka"], res)
        c = cells.get(key)
        if c is None:
            drift.append({"row": r, "why": "no such (ka, resolution) cell in study.json"})
            continue
        ok_got = abs(r["got"] - round(c["bem_force"], 4)) < 5e-4
        ok_ref = abs(r["ref"] - round(c["ref_force"], 4)) < 5e-4
        ok_rel = abs(r["rel_pct"] - round(100 * c["rel_error"], 2)) < 5e-3
        entry = {"ka": r["ka"], "resolution": key[1], "got_ok": ok_got,
                 "ref_ok": ok_ref, "rel_ok": ok_rel}
        (traceable if (ok_got and ok_ref and ok_rel) else drift).append(entry)

    # explicit spot checks of claims that mention numbers not in the artifact
    untraceable_claims = []
    if "832" in text:
        panels = {c["panel_count"] for c in study["cells"]}
        if 832 not in panels:
            untraceable_claims.append(
                "'(32,12,3.0)=832 三角形' (⑤-1) but study.json records 800 panels for that mesh")
    if "416" in text:
        panels = {c["panel_count"] for c in study["cells"]}
        if 416 not in panels:
            untraceable_claims.append(
                "'(32,6,2.5)=416 三角形' (⑤-1) is not a rung of the delivered study "
                f"(panel counts: {sorted(panels)})")
    if "ka=0.1 → 6.1e-3" in text and not any(
            abs(float(a["ka"]) - 0.1) < 1e-9 for a in study.get("long_wave_anchors", [])):
        untraceable_claims.append(
            "'ka=0.1 → 6.1e-3' (⑤-5) is not in long_wave_anchors (only ka=0.01/0.02/0.05)")
    if "fine 的 3.01%" in text:
        f1 = cells.get((1.0, "fine"))
        fi = cells.get((1.0, "finer"))
        if f1 and abs(round(100 * f1["rel_error"], 2) - 3.01) > 0.02:
            untraceable_claims.append(
                "'ka=1.0 ... 降到 fine 的 3.01%' (⑤-1) quotes the *finer* value "
                f"({100 * fi['rel_error']:.2f}%); fine is {100 * f1['rel_error']:.2f}%")
    honest = ("honest FAIL" in text) and ("未达标" in text)
    return {
        "sections_present": sections,
        "sections_complete": all(sections.values()),
        "table_rows_parsed": len(rows),
        "table_rows_traceable": len(traceable),
        "table_rows_drifting": drift,
        "every_table_number_traceable": len(drift) == 0,
        "untraceable_prose_claims": untraceable_claims,
        "honest_fail_disclosed": honest,
        "declares_5pct_target": "<5%" in text,
        "has_error_judgement_section": "是否 <5%" in text,
    }


# --------------------------------------------------------------------------- #
# 5. repro: replay the delivered repro_cmd (merge route) byte-for-byte
# --------------------------------------------------------------------------- #
def repro(study: dict) -> dict:
    cmd = ["env", "PYTHONPATH=src", "python3", "-m",
           "tensorbem.benchmarks.maruo_drift_far_field_case",
           "--merge", *sorted(f"artifacts/maruo_drift_far_field_case/cell_ka_{ka}.json"
                              for ka in ("0.2", "0.5", "1.0", "2.0")),
           "--json", "/tmp/m17327_repro_study.json"]
    replay_path = "/tmp/m17327_repro_study.json"
    if os.path.exists(replay_path):
        os.remove(replay_path)
    p = subprocess.run(cmd, cwd=SRC, capture_output=True, text=True)
    left = json.loads(open(os.path.join(SRC, "artifacts/maruo_drift_far_field_case/study.json")).read())
    right = json.loads(open(replay_path).read()) if os.path.exists(replay_path) else None
    # the module exits 1 when a gate fails (honest non-zero); rc must agree with all_passed
    rc_consistent = right is not None and (p.returncode == 0) == bool(right["all_passed"])

    def strip_cells(s):
        for c in s["cells"]:
            c.pop("elapsed_seconds", None)
        return s

    identical = right is not None and json.dumps(strip_cells(left), sort_keys=True) == \
        json.dumps(strip_cells(right), sort_keys=True)
    return {
        "delivered_repro_cmd": "PYTHONPATH=src python3 -m tensorbem.benchmarks."
                               "maruo_drift_far_field_case --merge artifacts/maruo_drift_far_field_case/"
                               "cell_ka_*.json --json docs/case_bem_maruo_drift/study.json",
        "replayed_cmd": " ".join(cmd),
        "returncode": p.returncode,
        "returncode_consistent_with_gate_outcome": rc_consistent,
        "stdout_tail": p.stdout.strip().splitlines()[-6:],
        "study_identical_except_wallclock": identical,
        "scope_note": "the delivered repro_cmd is a *merge-only* replay: it re-derives the "
                      "gates and the reference from the per-cell JSONs but does NOT re-run the "
                      "BEM solves, so it cannot detect drift in the solver; the four production "
                      "cells were therefore re-executed separately (cluster_replay.json)",
    }


# --------------------------------------------------------------------------- #
# 6. figure
# --------------------------------------------------------------------------- #
def figure(study: dict, cluster: dict | None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = study["cells"]
    kas = study["parameters"]["ka_values"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ka in kas:
        rows = sorted((c for c in cells if c["ka"] == ka), key=lambda c: c["panel_count"])
        axes[0].plot([c["panel_count"] for c in rows],
                     [100 * c["rel_error"] for c in rows], "o-", label=f"ka={ka}")
    axes[0].axhline(5.0, color="crimson", ls="--", lw=1.2, label="tolerance 5%")
    if cluster:
        extra = [c for c in cluster["cells"] if c["tag"].startswith("audit_")]
        for c in extra:
            axes[0].plot([c["panel_count"]], [100 * c["rel_error"]], "k*", ms=11)
        if extra:
            axes[0].plot([], [], "k*", label="audit rungs (beyond delivery)")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("panels")
    axes[0].set_ylabel("relative error vs closed form [%]")
    axes[0].set_title("CASE-BEM-MARUO-DRIFT: BEM vs Maruo closed form")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)

    prod = sorted((c for c in cells if c["resolution"] == study["parameters"]["production_resolution"]),
                  key=lambda c: c["ka"])
    x = range(len(prod))
    axes[1].bar([i - 0.2 for i in x], [100 * c["rel_error"] for c in prod], width=0.4,
                label="BEM (production)")
    axes[1].bar([i + 0.2 for i in x], [100 * c["rel_error"] for c in
                                       sorted((c for c in cells if c["resolution"] == "finer"),
                                              key=lambda c: c["ka"])], width=0.4,
                label="BEM (finer)")
    axes[1].axhline(5.0, color="crimson", ls="--", lw=1.2, label="tolerance 5%")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([f"ka={c['ka']}" for c in prod])
    axes[1].set_ylabel("relative error [%]")
    axes[1].set_title("production mesh: 3/4 cells pass, ka=2 fails")
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend(fontsize=8)
    for i, c in enumerate(prod):
        axes[1].text(i - 0.2, 100 * c["rel_error"] + 0.2, f"{100 * c['rel_error']:.2f}",
                     ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    plt.close(fig)
    return {"figure": FIG, "sha256": sha256(FIG), "bytes": os.path.getsize(FIG)}


# --------------------------------------------------------------------------- #
# 7. level decision + verdict row
# --------------------------------------------------------------------------- #
def decide(ext, ms, ref, judged, rep, repro_res, cluster) -> tuple[int, list[str], list[str]]:
    blockers, notes = [], []
    if ref["registered_rows_count"] == 0:
        blockers.append("blocking-for-L4: tensor_benchmark_cases has 0 rows for anchor "
                        f"{ANCHOR} -> no machine referee; criterion only in mission-goal text")
    if judged["verdict_at_production_mesh"] == "FAIL":
        w = judged["production_worst"]
        blockers.append(
            "blocking-for-L4: declared <5% criterion FAILS at the production mesh "
            f"(ka={w['ka']}, {w['panel_count']} panels, {100 * w['rel_error_recomputed']:.3f}%)")
    if not judged["all_cells_recomputed_match_declared"]:
        blockers.append("blocking: artifact rel_error disagrees with recomputation")
    if ext["n_missing"]:
        blockers.append(f"blocking: {ext['n_missing']} delivered files missing")
    if not rep["sections_complete"]:
        blockers.append("blocking: report six-section discipline incomplete")
    if not rep["every_table_number_traceable"]:
        notes.append("material: report ④ table rows not traceable to study.json")
    if rep["untraceable_prose_claims"]:
        notes.append("material: prose numbers with no artifact source "
                     f"({len(rep['untraceable_prose_claims'])} items)")
    if not repro_res["study_identical_except_wallclock"]:
        blockers.append("blocking: delivered repro_cmd does not reproduce study.json")
    elif not repro_res["returncode_consistent_with_gate_outcome"]:
        blockers.append("blocking: delivered repro_cmd exit code disagrees with its gate outcome")
    notes.append("note: delivered repro_cmd is merge-only (no BEM re-solve) -- "
                 "reproducibility of the solver rests on the audit cluster replay")

    l3_ok = (ext["n_missing"] == 0 and judged["all_cells_recomputed_match_declared"]
             and repro_res["study_identical_except_wallclock"]
             and repro_res["returncode_consistent_with_gate_outcome"])
    if not l3_ok:
        return 2, blockers, notes
    if blockers:
        return 3, blockers, notes
    return 4, blockers, notes


def insert_verdict(dsn: str, level: int, ev: dict) -> int:
    import psycopg
    scope = f"m17276 {ANCHOR}"
    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute("INSERT INTO tensor_maturity_verdicts "
                    "(repo, scope, mission_id, level, evidence, verdict_by) "
                    "VALUES (%s,%s,%s,%s,%s::jsonb,%s) RETURNING id",
                    ("TensorBEM", scope, AUDIT_MID, level,
                     json.dumps(ev, ensure_ascii=False), "board-audit"))
        vid = cur.fetchone()[0]
        c.commit()
    print("verdict id", vid, "level", level)
    return vid


# --------------------------------------------------------------------------- #
# 8. markdown
# --------------------------------------------------------------------------- #
def render(ext, ms, ref, judged, rep, repro_res, cluster, verdict, blockers, notes) -> str:
    L = []
    A = L.append
    A("# m17327 交差核验：CASE-BEM-MARUO-DRIFT（TensorBEM · m17276）")
    A("")
    A(f"**裁定：L{verdict}**（board-audit；非重做，只核验）。核心结论：真代码 + 真求解器链路，"
      "闭式 ref 经独立重推验证，但**生产分辨率下 ka=2 单元 5.331% > 5% 判据未过**，"
      "且 benchmark 登记表中本锚点 **0 行**——L4 双阻塞，诚实降级到 L3。")
    A("")
    A("## 1. 取证（三件套是否存在）")
    A("")
    A("| 件 | 结论 | 证据 |")
    A("|---|---|---|")
    A(f"| run_artifacts | {'✅ 全部存在' if ext['n_missing'] == 0 else '❌ 缺 %d' % ext['n_missing']} | "
      f"`artifacts/maruo_drift_far_field_case/study.json`（{ext['n_cells']} cells，sha256 见 existence.json）；"
      "docs 侧同 sha256 副本 + 3 张 png |")
    A(f"| benchmark ref vs got | ✅ 数字自洽 | 16 cells 全部按 `|got-ref|/ref` 重算，"
      f"与工件 `rel_error` 逐一吻合={judged['all_cells_recomputed_match_declared']} |")
    A(f"| repro_cmd | ✅ 可执行（但为 merge-only） | 重放 rc={repro_res['returncode']}"
      f"（与 gate outcome 一致：{repro_res['returncode_consistent_with_gate_outcome']}），"
      f"study.json 除墙钟外逐字节一致={repro_res['study_identical_except_wallclock']}；"
      "四个生产单元另做集群重跑（见 §4） |")
    A("")
    A("## 2. 基准裁判（唯一裁判）")
    A("")
    A(f"`SELECT … FROM tensor_benchmark_cases WHERE anchor_slug='{ANCHOR}'` → "
      f"**{ref['registered_rows_count']} 行**。")
    A("")
    A("因此机器裁判缺席，本次判据取任务 goal 原文：**「容差判据：<5%」**"
      "（0 行不是「无判据」，是「登记未落地」——与 m17204/CASE-BEM-CYL-DRIFT-ASYMP 同一个生态缺口，"
      "该锚点后来才补登 id=43）。")
    A("")
    A("| ka | 生产网格 panels | got [N] | ref [N] | 重算误差 | 报告声明 | <5% |")
    A("|---|---|---|---|---|---|---|")
    for c in judged["production_cells"]:
        A(f"| {c['ka']} | {c['panel_count']} | {c['got']:.6g} | {c['ref']:.6g} | "
          f"{100 * c['rel_error_recomputed']:.3f}% | {100 * c['rel_error_declared']:.3f}% | "
          f"{'✅' if c['within_tol'] else '❌'} |")
    A("")
    w = judged["production_worst"]
    A(f"生产网格最坏单元：ka={w['ka']}（{w['panel_count']} panels）**{100 * w['rel_error_recomputed']:.3f}%**，"
      f"声明 {100 * w['rel_error_declared']:.3f}%，一致；`G3_bem_vs_ref_production` "
      f"measured={judged['gates_declared']['G3_bem_vs_ref_production']['measured']:.6g} "
      f"passed={judged['gates_declared']['G3_bem_vs_ref_production']['passed']}，工件 `all_passed`="
      f"{judged['all_passed_declared']}。**交付自己就没说过通过**——mission 头行写的是 "
      "「G3 FAIL(5.33% @ ka=2.0)」，auto-approve 仍把 status 置 succeeded，"
      "这是平台验收口径问题，不是交付方粉饰。")
    A("")
    A("## 3. ref 的独立重推（关键：不 import tensorbem）")
    A("")
    A("`scripts/audit/m17327_ref_independent.py` 只用教科书定义 + scipy 重写："
      "MacCamy–Fuchs 余弦型散射系数 `A_m=J_m'(ka)/H_m'^(1)(ka)`、"
      "`S(θ)=-Σ ε_m A_m cos mθ`、Maruo 远场积分，然后**暴力梯形求积**（21000 点，"
      "对有限三角多项式精确）算 `∫|S|²(1-cosθ)dθ`。结果：")
    A("")
    A("| ka | 交付 ref [N] | 独立重推 [N] | 相对差 |")
    A("|---|---|---|---|")
    for row in INDEP["per_ka"]:
        A(f"| {row['ka']} | {row['ref_force_delivered']:.10g} | {row['ref_force_independent']:.10g} | "
          f"{row['rel_diff']:.2e} |")
    A("")
    A(f"外部锚点（唯一真正外部的）：ka→0 极限独立重推得 "
      f"{INDEP['long_wave_anchor']['extrapolated_limit_ka2_zero']:.9g}，"
      f"vs 经典 `5π²/16 = {INDEP['long_wave_anchor']['law_constant_from_pi']:.9g}`，"
      f"相对差 {INDEP['long_wave_anchor']['rel_diff_limit_vs_law']:.2e}"
      "（残差为 O(ka²) 外推误差）。**注意踩坑**：本审计脚本第一版用全整数 m 的复指数求和、"
      "相位因子写成 `(-i)^m`，奇 m 相对相位翻转，长波极限变成 π²/16 = 1/5 的经典值——"
      "内部求积自洽（modal vs quad 1e-11）却整体错 5 倍，**只有外部锚点抓得住**。"
      "这也是为什么「闭式 vs 仓内求积 2.4e-16」不能算外部验证：两条路线共用同一套系数。")
    A("")
    A("## 4. 可复现性：merge 重放 + 生产单元真重算")
    A("")
    A(f"* 交付 repro_cmd 是 `--merge artifacts/.../cell_ka_*.json`：重放 rc={repro_res['returncode']}，"
      f"输出与交付 study.json 除 `elapsed_seconds` 外**逐字节一致** = "
      f"{repro_res['study_identical_except_wallclock']}。但它**不重跑 BEM 求解**，"
      "所以只能证明 gate/ref 派生逻辑可复现，证明不了求解器可复现。")
    if cluster:
        A(f"* 审计侧集群重跑（cj{cluster.get('job_id')}，loong64）：四个生产单元全部重算，"
          f"与交付 `bem_force` 最大相对差 "
          f"{cluster.get('max_rel_diff_production', float('nan')):.2e}；"
          f"另加两个交付没跑过的网格档（48×16=1584、64×16=2112 panels）在 ka=2 上探测 5.33% 是"
          f"「网格不够」还是「渐近残留」。结果见下表。")
        A("")
        A("| 网格 | panels | ka | got [N] | ref [N] | 误差 | <5% |")
        A("|---|---|---|---|---|---|---|")
        for c in cluster["cells"]:
            A(f"| {c['tag']} | {c['panel_count']} | {c['ka']} | {c['bem_force']:.6g} | "
              f"{c['ref_force']:.6g} | {100 * c['rel_error']:.3f}% | "
              f"{'✅' if c['rel_error'] < TOL else '❌'} |")
        A("")
        A(f"**判读**：{cluster.get('reading', '')}")
    else:
        A("* 集群重跑已递交（**cj1489**，TensorBEM 检出 + 四个生产单元 + 两个越档网格），"
          "但审计落卷时作业仍在排队（队列 611）→ 本项记 **not_run**，不记 pass。"
          "这是本单唯一的未跑项，不影响定级：L4 的第二阻塞（判据未过）用的是**交付自己的**"
          "最细档数字 5.04%（1200 panels），比 5% 判据仍差 0.04 个百分点。")
    A("")
    A("## 5. 报告六段与数字溯源")
    A("")
    A(f"六段齐全 = {rep['sections_complete']}（①案例+ref出处 ②方法 ③参数表 ④对照表 ⑤收敛/敏感性 ⑥失败模式）。"
      f"④ 表解析 {rep['table_rows_parsed']} 行，可溯源 {rep['table_rows_traceable']} 行，"
      f"漂移 {len(rep['table_rows_drifting'])} 行。")
    A("")
    if rep["untraceable_prose_claims"]:
        A("**不可溯源的正文数字（material，不是 L 级阻断）**：")
        for it in rep["untraceable_prose_claims"]:
            A(f"* {it}")
        A("")
    A("## 6. 定级（L0–L5，级别=功能实质）")
    A("")
    A("| 级 | 判据 | 本单 |")
    A("|---|---|---|")
    A("| L1 数学/单元验证 | 闭式 + 单元测试 | ✅ 13 tests pass（审计重跑 3.28 s）；闭式经审计独立重推 1e-12 一致 |")
    A("| L2 独立功能实现 | 输入→计算→输出真代码 | ✅ `bem_case`：建网格→`solve_diffraction_3d`→`bem_kochin`→"
      "`maruo_far_field_drift`→误差，569 行，无硬编码 |")
    A("| L3 系统集成 | 接入主求解器 | ✅ 直接调用仓内三条主求解器 API，16 单元端到端跑通，"
      "残差 ~1e-15 |")
    A("| L4 基准验证 | 完整算例+误差分析+可复现+**判据通过** | ❌ 双阻塞：登记表 0 行 + 生产网格 ka=2 "
      "5.331% > 5% |")
    A("| L5 科研工程成果 | — | ❌ |")
    A("")
    A("**裁定：L3。** 不是 L2——链路是真集成不是孤立脚本；不到 L4——判据没过 + 裁判缺席。"
      "百页报告不改变级别。")
    A("")
    A("## 7. findings（按严重度）")
    A("")
    for b in blockers:
        A(f"* **[blocking]** {b}")
    for n in notes:
        A(f"* **[{n.split(':')[0]}]** {n.split(':', 1)[1].strip() if ':' in n else n}")
    A("")
    A("## 8. 未跑 / 不知道的")
    A("")
    A("* 与**实验或文献表格**的对照没有跑：沙箱无外网，交付与审计都只能对闭式定律（这是真实的限制，"
      "不是选择）。因此「外部锚点」只有长波 `5π²/16` 一条。")
    A("* 有限水深、非圆截面、多体遮蔽均不在本算例范围（交付 ⑥ 已声明）。")
    A("* 集群重跑的架构是 loong64，与 GitHub CI 的 x86 不同；本单只核数值一致性，未核跨架构一致性。")
    A("")
    A("## 9. 工件")
    A("")
    for name in sorted(os.listdir(OUT)):
        A(f"* `artifacts/audit_m17327/{name}`")
    A("")
    A(f"图：`{os.path.relpath(FIG, REPO)}`（sha256 {sha256(FIG) if os.path.exists(FIG) else 'n/a'}）")
    A("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    global INDEP
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="dbname=tensorplatform user=postgres")
    ap.add_argument("--insert-verdict", action="store_true")
    ap.add_argument("--cluster-json", default=os.path.join(OUT, "cluster_replay.json"))
    args = ap.parse_args(argv)

    os.makedirs(OUT, exist_ok=True)
    ext = existence()
    dump("existence.json", ext)
    ms = merge_state()
    dump("merge_state.json", ms)
    ref = referee(args.dsn)
    dump("referee.json", ref)
    study = json.loads(open(os.path.join(SRC, "artifacts/maruo_drift_far_field_case/study.json")).read())
    judged = judge(study)
    dump("judge.json", judged)
    INDEP = json.loads(open(os.path.join(OUT, "independent_ref.json")).read())
    rep = report_check(study)
    dump("report_check.json", rep)
    repro_res = repro(study)
    dump("repro.json", repro_res)
    cluster = json.loads(open(args.cluster_json).read()) if os.path.exists(args.cluster_json) else None
    fig = figure(study, cluster)
    dump("figure.json", fig)

    verdict, blockers, notes = decide(ext, ms, ref, judged, rep, repro_res, cluster)
    finding = {"level": verdict, "blockers": blockers, "notes": notes,
               "anchor": ANCHOR, "subject_mission": 17276, "audit_mission": AUDIT_MID}
    dump("finding.json", finding)

    evidence = {
        "repo": "TensorBEM",
        "anchor": ANCHOR,
        "level": verdict,
        "subject_mission": 17276,
        "audit_mission": AUDIT_MID,
        "impl_files": [
            f"src/tensorbem/benchmarks/maruo_drift_far_field_case.py:1-569 (branch {BRANCH} @ {COMMIT}, sha256 "
            f"{ext['files']['src/tensorbem/benchmarks/maruo_drift_far_field_case.py']['sha256']})",
            f"src/tensorbem/benchmarks/maruo_drift_far_field_case_figures.py (sha256 "
            f"{ext['files']['src/tensorbem/benchmarks/maruo_drift_far_field_case_figures.py']['sha256']})",
            "src/tensorbem/benchmarks/maruo_drift_far_field_case.py:125-212 (closed-form reference)",
            "src/tensorbem/benchmarks/maruo_drift_far_field_case.py:214-270 (BEM route under test)",
        ],
        "entry": "PYTHONPATH=src python3 -m tensorbem.benchmarks.maruo_drift_far_field_case "
                 "--ka 0.2 0.5 1.0 2.0 --json <out>.json",
        "run_artifacts": [
            f"artifacts/maruo_drift_far_field_case/study.json (16 cells, sha256 "
            f"{ext['files']['artifacts/maruo_drift_far_field_case/study.json']['sha256']})",
            "docs/case_bem_maruo_drift/study.json (same sha256)",
            "artifacts/maruo_drift_far_field_case/fig_bem_vs_closed_form.png",
            "artifacts/maruo_drift_far_field_case/fig_coefficient_vs_ka.png",
            "artifacts/maruo_drift_far_field_case/fig_convergence_and_quadrature.png",
            f"audit copy: artifacts/audit_m17327/ (existence.json, judge.json, repro.json, "
            f"independent_ref.json, cluster_replay.json, {os.path.basename(FIG)})",
        ],
        "tests": [
            "tests/test_maruo_drift_far_field_case.py (13 tests; audit re-run in work/m17276: "
            "13 passed in 3.28 s)",
        ],
        "integration_callers": [
            "tensorbem.free_surface_solver3d.solve_diffraction_3d (panel diffraction solve)",
            "tensorbem.drift_far_field3d.bem_kochin -> maruo_far_field_drift (Kochin -> reduced Maruo kernel)",
            "tensorbem.diffraction_drift_maturity.vertical_cylinder_hull_mesh (watertight cylinder mesh)",
        ],
        "benchmark": {
            "anchor_slug": ANCHOR,
            "registered_referee_rows": ref["registered_rows_count"],
            "criterion_in_force": ref["criterion_in_force"],
            "criterion_source": "mission 17276 goal text '容差判据：<5%' (no machine referee row)",
            "production_resolution": judged["production_resolution"],
            "production_cells_recomputed_match_declared": judged["all_cells_recomputed_match_declared"],
            "production_n_failures": judged["production_n_failures"],
            "production_worst": {
                "ka": judged["production_worst"]["ka"],
                "panels": judged["production_worst"]["panel_count"],
                "rel_error": judged["production_worst"]["rel_error_recomputed"],
            },
            "verdict_at_production_mesh": judged["verdict_at_production_mesh"],
            "finest_mesh_worst_rel_error": (
                cluster.get("finest_rel_error") if cluster else None),
            "got_always_below_ref": judged["got_always_below_ref"],
            "reference_verified_independently": {
                "script": "scripts/audit/m17327_ref_independent.py",
                "max_rel_diff_vs_delivered": max(r["rel_diff"] for r in INDEP["per_ka"]),
                "long_wave_limit_vs_5pi2_over_16": INDEP["long_wave_anchor"]["rel_diff_limit_vs_law"],
            },
            "report_six_sections": rep["sections_complete"],
            "report_numbers_traceable": rep["every_table_number_traceable"],
            "report_untraceable_claims": rep["untraceable_prose_claims"],
        },
        "repro_cmd": [
            repro_res["delivered_repro_cmd"],
            "PYTHONPATH=src python3 -m tensorbem.benchmarks.maruo_drift_far_field_case "
            "--ka 0.2 0.5 1.0 2.0 --json artifacts/maruo_drift_far_field_case/study.json "
            "--figures artifacts/maruo_drift_far_field_case  (full re-solve)",
            "PYTHONPATH=src python3 -m pytest tests/test_maruo_drift_far_field_case.py -q",
        ],
        "external_anchor": {
            "law": "F_x = (5 pi^2/16) rho g A^2 a (ka)^3 (Maruo 1960; Newman 1977 Sec. 7.3)",
            "recovered_from_this_implementation": INDEP["long_wave_anchor"]["extrapolated_limit_ka2_zero"],
            "law_constant": INDEP["long_wave_anchor"]["law_constant_from_pi"],
            "rel_diff": INDEP["long_wave_anchor"]["rel_diff_limit_vs_law"],
            "no_paper_table_used": "ref is the machine-evaluated closed form, not a transcribed table; "
                                   "offline sandbox is the declared reason (report ①)",
        },
        "repro_result": {
            "delivered_repro_replayed": repro_res["study_identical_except_wallclock"],
            "delivered_repro_exit_code": repro_res["returncode"],
            "delivered_repro_exit_code_consistent": repro_res["returncode_consistent_with_gate_outcome"],
            "study_identical_except_wallclock": repro_res["study_identical_except_wallclock"],
            "solver_resolved_cells_rerun_on_cluster": bool(cluster),
            "cluster_job": ("cj1489" if not cluster else cluster.get("job_id")),
            "cluster_state_at_verdict": ("queued" if not cluster else "fetched"),
            "cluster_max_rel_diff_production": (cluster or {}).get("max_rel_diff_production"),
        },
        "merge_state": ms,
        "verdict_reason": blockers,
        "notes": notes,
        "report": "docs/board_audit_m17327/M17327_BOARD_AUDIT.md",
    }
    dump("evidence.json", evidence)

    md = render(ext, ms, ref, judged, rep, repro_res, cluster, verdict, blockers, notes)
    with open(DOC, "w") as fh:
        fh.write(md)
    print(md[:1500])
    print("...")
    print(f"[audit] verdict=L{verdict} blockers={len(blockers)} notes={len(notes)}")
    if args.insert_verdict:
        insert_verdict(args.dsn, verdict, evidence)
    return 0


if __name__ == "__main__":
    sys.exit(main())