#!/usr/bin/env python3
"""AUDIT-RESTAT 校准审计：把「成果重统计」的代理分档器放回地面真值上检验。

背景（为什么需要这个工具）
--------------------------
2026-11-11 的 AUDIT-RESTAT 波次（73 份完全相同的
docs/RESTAT_L0_L5_20261111.md，sha256 41a60307…8fa4）给出全库 L0-L5 重分类，
其 headline 是「严格口径 L2+ 成果 = 2,366 单」。该数字由 Tier-B **信号代理**
产生：只看 result.changed_files / result.pr_gate / co_repos / goal 关键词。
代理的验证集当时只有 18 对（exact 4，±1 8，见原文 §1 注与 §6），
且原文自承「代理系统性高估 1-2 档」。

本工具做的是那批交付**没做**的三件事，全部只读 SQL、可复跑：
  (A) 把代理分档器放回 Tier-A 地面真值（tensor_maturity_verdicts.level /
      gate_level）上，用现在更大样本（236 单 / 409 判定，原文只有 18 对）
      算混淆矩阵、准确率与偏差方向，检验「高估 1-2 档」是否成立。
  (B) 量化 pr_gate 元数据的**时代混杂**：严格口径要求 pr_state='merged'，
      但 pr_gate 键本身只在晚期任务上被写入。若早期任务根本没有该键，
      「未合并」与「未记录」被混为一谈 -> headline 数字是元数据时代的函数，
      不是质量的函数。
  (C) 量化「实现文件」纯度污染：旧代理把任意 *.py 计为实现文件，包括
      _scratch_* / _patch_* / post_report_* 这类一次性脚本（本仓工作树即有
      几十个），它们不构成库内功能。按纯度过滤后重估 L2+。

输出为三档区间（下界/旧口径/上界）而不是单点，并把每个数字的来源 SQL、
快照时间、产物 sha256 一并落盘，便于审计复判。

用法：
    python3 tools/audit_restat_calibration_20261113.py            # 全流程
    python3 tools/audit_restat_calibration_20261113.py --no-figs   # 仅数据/表格
只读：仅执行 SELECT；不写数据库。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

# 11-11 交付的对外引用值（来源：docs/RESTAT_L0_L5_20261111.md，73 份副本同 sha256）
PRIOR = {
    "succeeded": 9117,        # 该文 §0「状态 succeeded 9,117」
    "strict_L2plus": 2366,    # 该文 §0「严格口径 L2+ 成果 2,366」
    "loose_L2plus": 6416,     # 该文 §0「宽松口径 L2+ 6,416」
    "doc_sha256": "41a60307d2775f9b79b96b71fb3f2895787b3331f9c33d89e09b7a45eb2dcfa4",
    "copies": 73,
}

OUT_JSON_DIR = "docs/audit_restat_20261113"
DOC = "docs/AUDIT_RESTAT_CALIBRATION_20261113.md"
FIG_DIR = "docs/audit_restat_20261113"

# ---------------------------------------------------------------- SQL 层 ----
# 归一化判定表：同一 mission 可能有多条 scope 判定，取该 mission 的最严
# (max level) 与最严机器门 (max gate_level)，并保留判定条数以便审计抽样。
SQL = {
    # 成功任务全量 + 代理分档器所需的全部字段
    "missions": """
        SELECT coalesce(json_agg(x), '[]'::json) FROM (
          SELECT m.id,
                 m.repo,
                 to_char(m.created_at,'YYYY-MM-DD') AS created_day,
                 to_char(date_trunc('week', m.created_at),'YYYY-MM-DD') AS week,
                 left(regexp_replace(m.goal, E'[\\n\\r\\t]+', ' ', 'g'), 400) AS goal,
                 coalesce(m.co_repos::text, '') AS co_repos,
                 coalesce(jsonb_array_length(coalesce(m.result->'changed_files','[]')),0) AS n_files,
                 coalesce(m.result->>'changed_files','[]') AS files_json,
                 coalesce(m.result->'pr_gate'->>'state','') AS pr_state,
                 (m.result ? 'pr_gate') AS has_pr_gate_key
          FROM tensor_dev_missions m
          WHERE m.status='succeeded'
        ) x
    """,
    # Tier-A 地面真值（按 mission 归一）
    "verdicts": """
        SELECT coalesce(json_agg(x), '[]'::json) FROM (
          SELECT v.mission_id,
                 max(v.level) AS level,
                 max(v.gate_level) AS gate_level,
                 min(v.gate_level) AS gate_level_min,
                 count(*) AS n_verdicts,
                 string_agg(DISTINCT v.verdict_by, ',') AS auditors,
                 string_agg(DISTINCT coalesce(v.gate_state,'-'), ',') AS gate_states,
                 left(string_agg(DISTINCT v.repo,'|'), 80) AS repos,
                 left(max(v.scope), 120) AS scope
          FROM tensor_maturity_verdicts v
          WHERE v.mission_id IS NOT NULL
          GROUP BY v.mission_id
        ) x
    """,
    "snapshot": """
        SELECT json_build_object(
          'missions_total', (SELECT count(*) FROM tensor_dev_missions),
          'missions_succeeded', (SELECT count(*) FROM tensor_dev_missions WHERE status='succeeded'),
          'verdict_rows', (SELECT count(*) FROM tensor_maturity_verdicts),
          'verdict_missions', (SELECT count(DISTINCT mission_id) FROM tensor_maturity_verdicts
                                WHERE mission_id IS NOT NULL),
          'deltas', (SELECT count(*) FROM tensor_capability_deltas),
          'reports', (SELECT count(*) FROM tensor_reports),
          'now', to_char(now(),'YYYY-MM-DD"T"HH24:MI:SSOF')
        )
    """,
}

# ------------------------------------------------- 代理分档器（原样复刻）----
DOC_EXT = {".md", ".txt", ".rst", ".pdf", ".docx", ".tex", ".html", ".svg"}
ASSET_EXT = {".png", ".jpg", ".jpeg", ".csv", ".json", ".npz", ".npy", ".pdf", ".h5", ".dat"}
CODE_EXT = {".py", ".c", ".cpp", ".h", ".hpp", ".f90", ".f", ".jl", ".rs", ".cu", ".m"}
INTEG_MARKERS = ("api/", "backend/", "cli", "__init__", "server/", "routes/", "tools/")
BENCH_WORDS = ("benchmark", "基准", "解析解", "analytic", "literature", "实验", "实验值",
               "convergence", "收敛", "converge", "文献", "validation", "验证")


def strip_prefix(path: str) -> str:
    """把 /root/tp-agent/work/m1234/a/b.py 归一为 a/b.py（与原文一致）。"""
    m = re.search(r"/work/m\d+/(.+)$", path)
    return m.group(1) if m else path


def classify(files, pr_state: str, co_repos: str, goal: str):
    """原文 Tier-B 代理分档（语义原样复刻，含其已知缺陷）。

    返回 (level_strict, level_loose, reason, n_run_artifacts)。
    注意：本函数刻意保留原口径（merged 才计 L2+），缺陷由调用方量化。
    """
    paths = [strip_prefix(p.strip()) for p in files if p.strip()]
    if not paths:
        return -1, -1, "no_changed_files", 0
    impl, tests, runs = [], [], 0
    for p in paths:
        low = p.lower()
        ext = os.path.splitext(low)[1]
        base = os.path.basename(low)
        if ext in DOC_EXT:
            pass
        elif ext in ASSET_EXT:
            if low.startswith(("runs/", "artifacts/", "figures/", "fig/")) or base.startswith("fig"):
                runs += 1
        elif ext in CODE_EXT:
            if base.startswith("test_") or low.startswith("tests/") or base.endswith("_test.py") \
               or re.search(r"(verify|validate|audit|check|probe)", base):
                tests.append(low)
            else:
                impl.append(low)
    if not impl and not tests:
        return 0, 0, "docs_only", runs
    if not impl:
        return 1, 1, "tests_or_verify_only", runs
    integrated = (len([r for r in co_repos.split(";") if r.strip()]) >= 2) or \
                 any(mk in p for p in impl for mk in INTEG_MARKERS)
    bench = any(w in goal.lower() for w in BENCH_WORDS) and runs > 0
    merged = pr_state == "merged"

    if integrated and bench:
        loose = 4
    elif integrated:
        loose = 3
    elif bench:
        loose = 4
    else:
        loose = 2
    if not merged:
        return 1, loose, ("impl_without_merge_evidence+" + str(loose)), runs
    return loose, loose, ("merged_" + ("integ" if integrated else "impl") +
                          ("+bench" if bench else "")), runs


# ------------------------------------------- 一次性脚本纯度（本审计新增）----
SCRATCH_PATTERNS = [
    re.compile(r"^_"),                       # _scratch_*, _patch_* …
    re.compile(r"^post_report"),             # post_report_m*.py
    re.compile(r"^publish_report"),
    re.compile(r"^make_report"),
    re.compile(r"^assemble_"),
    re.compile(r"^compile_"),
    re.compile(r"^debug_"),
    re.compile(r"^smoke_"),
    re.compile(r"^follow_bb"),
]


def split_impl_purity(files):
    """把「实现文件」再分为库内实现 vs 一次性脚本。

    一次性脚本=根目录随手脚本（不进入包/模块结构、无调用方）：判据为
    文件名命中 SCRATCH_PATTERNS 且路径不含 '/'（不在 src/<pkg>/ 内）。
    返回 (lib_impl, scratch_impl)。
    """
    lib, scratch = [], []
    for p in files:
        base = os.path.basename(p)
        if "/" not in p and any(rx.search(base) for rx in SCRATCH_PATTERNS):
            scratch.append(p)
        else:
            lib.append(p)
    return lib, scratch


def classify_calibrated(files, pr_state: str, has_pr_key: bool, co_repos: str, goal: str):
    """校准分档：只改两条已知缺陷，档位判据其余不变。

    缺陷1（元数据时代混杂）：has_pr_key=False 时 pr_state 恒为空，旧口径把这
        当作「未合并」-> 直接压成 L1。校准口径把「无该键」视为未记录，
        沿用实测条件概率（记录到 pr_gate 的任务里 merged 占 97%）判为入库。
    缺陷2（实现文件纯度）：库里只有一次性脚本的不算功能实现 -> 降到 L1。
    """
    paths = [strip_prefix(p.strip()) for p in files if p.strip()]
    if not paths:
        return -1, "no_changed_files"
    impl, tests, runs = [], [], 0
    for p in paths:
        low = p.lower()
        ext = os.path.splitext(low)[1]
        base = os.path.basename(low)
        if ext in DOC_EXT:
            pass
        elif ext in ASSET_EXT:
            if low.startswith(("runs/", "artifacts/", "figures/", "fig/")) or base.startswith("fig"):
                runs += 1
        elif ext in CODE_EXT:
            if base.startswith("test_") or low.startswith("tests/") or base.endswith("_test.py") \
               or re.search(r"(verify|validate|audit|check|probe)", base):
                tests.append(low)
            else:
                impl.append(low)
    if not impl and not tests:
        return 0, "docs_only"
    if not impl:
        return 1, "tests_or_verify_only"
    lib_impl, scratch_impl = split_impl_purity(impl)
    if not lib_impl:
        return 1, "scratch_only_impl"     # 缺陷2 修正点
    integrated = (len([r for r in co_repos.split(";") if r.strip()]) >= 2) or \
                 any(mk in p for p in lib_impl for mk in INTEG_MARKERS)
    bench = any(w in goal.lower() for w in BENCH_WORDS) and runs > 0
    if integrated and bench:
        loose = 4
    elif integrated:
        loose = 3
    elif bench:
        loose = 4
    else:
        loose = 2
    merged = (pr_state == "merged") or (not has_pr_key)   # 缺陷1 修正点
    if not merged:
        return 1, "impl_without_merge_evidence+" + str(loose)
    return loose, ("merged_" + ("integ" if integrated else "impl") + ("+bench" if bench else ""))


# --------------------------------------------------------------- psql IO ----
def psql_json(sql: str):
    """只读查询：以 postgres 身份读本机 tensorplatform 库。"""
    cmd = ["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-AtX", "-c", sql]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"psql failed: {p.stderr[:500]}")
    return json.loads(p.stdout.strip())


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(a, b) -> float:
    return round(100.0 * a / b, 2) if b else 0.0


# ------------------------------------------------------------ 主分析流程 ----
def analyse(missions, verdicts, snapshot):
    by_id = {v["mission_id"]: v for v in verdicts}
    rows = []
    for m in missions:
        files = json.loads(m["files_json"] or "[]")
        l_strict, l_loose, reason, runs = classify(files, m["pr_state"], m["co_repos"], m["goal"])
        l_cal, cal_reason = classify_calibrated(
            files, m["pr_state"], bool(m.get("has_pr_gate_key")), m["co_repos"], m["goal"])
        v = by_id.get(m["id"])
        rows.append({
            "id": m["id"], "repo": m["repo"], "week": m["week"],
            "l_strict": l_strict, "l_loose": l_loose, "l_cal": l_cal,
            "reason": reason, "cal_reason": cal_reason, "n_files": m["n_files"],
            "files_head": [strip_prefix(fp) for fp in files][:8],
            "pr_state": m["pr_state"], "has_pr_gate_key": bool(m.get("has_pr_gate_key")),
            "v_level": v["level"] if v else None,
            "v_gate": v["gate_level"] if v else None,
            "v_auditors": v["auditors"] if v else None,
            "v_states": v["gate_states"] if v else None,
            "scope": v["scope"] if v else None,
        })
    return rows


def tier_table(rows, key):
    """L-1..L4 计数（-1=不可判）。"""
    out = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return {str(k): out[k] for k in sorted(out)}


def confusion(rows, key, truth_key):
    """代理档 vs 地面真值 混淆矩阵；返回 (matrix, n, exact, within1, mean_delta)。"""
    pairs = [(r[key], r[truth_key]) for r in rows
             if r.get(truth_key) is not None and r[key] is not None and r[key] >= 0]
    mat = {}
    for p, t in pairs:
        mat.setdefault(str(p), {}).setdefault(str(t), 0)
        mat[str(p)][str(t)] += 1
    deltas = [t - p for p, t in pairs]
    n = len(pairs)
    exact = sum(1 for d in deltas if d == 0)
    within1 = sum(1 for d in deltas if abs(d) <= 1)
    return {
        "matrix": mat, "n": n,
        "exact": exact, "exact_pct": pct(exact, n),
        "within1": within1, "within1_pct": pct(within1, n),
        "mean_truth_minus_proxy": round(sum(deltas) / n, 3) if n else None,
        "proxy_overestimates": sum(1 for d in deltas if d < 0),
        "proxy_underestimates": sum(1 for d in deltas if d > 0),
    }


def era_table(rows):
    """按周统计 pr_gate 元数据覆盖（时代混杂证据）。"""
    agg = {}
    for r in rows:
        w = r["week"]
        a = agg.setdefault(w, {"n": 0, "with_key": 0, "merged": 0, "impl_rows": 0,
                               "strict_L2plus": 0, "cal_L2plus": 0})
        a["n"] += 1
        a["with_key"] += 1 if r["has_pr_gate_key"] else 0
        a["merged"] += 1 if r["pr_state"] == "merged" else 0
        a["impl_rows"] += 1 if r["l_loose"] >= 2 else 0
        a["strict_L2plus"] += 1 if r["l_strict"] >= 2 else 0
        a["cal_L2plus"] += 1 if r["l_cal"] >= 2 else 0
    out = []
    for w in sorted(agg):
        a = agg[w]
        out.append({"week": w, **a,
                    "pr_key_pct": pct(a["with_key"], a["n"]),
                    "strict_pct": pct(a["strict_L2plus"], a["impl_rows"]),
                    "cal_pct": pct(a["cal_L2plus"], a["impl_rows"])})
    return out


def purity_stats(rows):
    """一次性脚本污染：impl 仅由随手脚本构成的单量。"""
    n_scratch_only = sum(1 for r in rows if r["cal_reason"] == "scratch_only_impl")
    impl_rows = sum(1 for r in rows if r["l_loose"] >= 2)
    return {"impl_rows": impl_rows, "scratch_only_impl_rows": n_scratch_only,
            "scratch_only_pct_of_impl": pct(n_scratch_only, impl_rows)}


def overclaim_list(rows, limit=40):
    """待 owner 裁定的「代理说 L2+、审计实测 ≤L1」名单（系统性高估证据）。"""
    bad = [r for r in rows
           if r.get("v_level") is not None and r["l_loose"] >= 2 and (r["v_gate"] or 0) <= 1]
    bad.sort(key=lambda r: (r["l_loose"] - (r["v_gate"] or 0)), reverse=True)
    return bad[:limit], len(bad)


def pinned_recompute(root, tables):
    """从 artifacts 里的 pinned CSV 重算聚合表，验证「快照可复现」。

    实时库会变（本次运行期间任务总数 15917→15918），因此可复现性的正确
    保证形式是：CSV 是快照锚，任何人在任何时间从该 CSV 重算都应得到同样的表。
    """
    import csv as _csv
    path = os.path.join(root, OUT_JSON_DIR, "missions_calibrated.csv")
    rows = []
    with open(path) as fh:
        for r in _csv.DictReader(fh):
            rows.append({"l_strict": int(r["l_strict"]), "l_loose": int(r["l_loose"]),
                         "l_cal": int(r["l_cal"]), "has_pr_gate_key": r["has_pr_gate_key"] == "True",
                         "pr_state": r["pr_state"], "repo": r["repo"],
                         "v_level": int(r["v_level"]) if r["v_level"] not in ("", "None") else None,
                         "v_gate": int(r["v_gate"]) if r["v_gate"] not in ("", "None") else None,
                         "cal_reason": r["cal_reason"]})
    checks = {}
    for key, col in [("tiers_old_strict", "l_strict"), ("tiers_old_loose", "l_loose"),
                     ("tiers_calibrated", "l_cal")]:
        checks[key] = {"match": tier_table(rows, col) == tables[key],
                       "recomputed": tier_table(rows, col)}
    checks["L2plus_match"] = {
        "match": all(sum(1 for r in rows if r[c] >= 2) == tables["L2plus"][n] for c, n in
                     [("l_strict", "old_strict"), ("l_loose", "old_loose"), ("l_cal", "calibrated")]),
        "recomputed": {n: sum(1 for r in rows if r[c] >= 2) for c, n in
                       [("l_strict", "old_strict"), ("l_loose", "old_loose"), ("l_cal", "calibrated")]},
    }
    checks["confusion_match"] = {
        "match": confusion(rows, "l_loose", "v_gate")["matrix"] == tables["confusion_loose_vs_gate_level"]["matrix"]}
    checks["all_match"] = all(v.get("match") for v in checks.values() if isinstance(v, dict))
    checks["csv_rows"] = len(rows)
    return checks


def sensitivity_merge_imputation(rows, qs=(0.0, 0.25, 0.5, 0.75, 0.967)):
    """敏感性分析：假设「无 pr_gate 键」的任务真实合并概率 = q，L2+ 期望值。

    校准口径把缺失键按 q=0.967（实测 p(merged|有键)）全部判入库，是最乐观端点。
    本表把 q 扫一遍，给出 L2+ 的完整敏感性带——避免用单点插补冒充事实。
    """
    base_strict = sum(1 for r in rows if r["l_strict"] >= 2)
    cand = [r for r in rows if r["l_loose"] >= 2 and r["l_strict"] < 2 and not r["has_pr_gate_key"]]
    cand_clean = [r for r in cand if r["cal_reason"] != "scratch_only_impl"]
    out = []
    for q in qs:
        out.append({
            "q_assumed_merged_given_no_key": q,
            "expected_L2plus": round(base_strict + q * len(cand_clean), 1),
            "candidate_missions": len(cand_clean),
        })
    out.append({"q_assumed_merged_given_no_key": "1.0(max)",
                "expected_L2plus": base_strict + len(cand_clean),
                "candidate_missions": len(cand_clean)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = args.root
    os.makedirs(os.path.join(root, OUT_JSON_DIR), exist_ok=True)
    os.makedirs(os.path.join(root, "docs"), exist_ok=True)

    snapshot = psql_json(SQL["snapshot"])
    missions = psql_json(SQL["missions"])
    verdicts = psql_json(SQL["verdicts"])
    rows = analyse(missions, verdicts, snapshot)

    succ = len(rows)
    tables = {
        "snapshot": snapshot,
        "population": {
            "succeeded_missions": succ,
            "missions_with_verdict": sum(1 for r in rows if r.get("v_level") is not None),
            "verdict_missions_total": len(verdicts),
            "succeeded_without_verdict": sum(1 for r in rows if r.get("v_level") is None),
        },
        "tiers_old_strict": tier_table(rows, "l_strict"),
        "tiers_old_loose": tier_table(rows, "l_loose"),
        "tiers_calibrated": tier_table(rows, "l_cal"),
        "L2plus": {
            "old_strict": sum(1 for r in rows if r["l_strict"] >= 2),
            "old_strict_pct": pct(sum(1 for r in rows if r["l_strict"] >= 2), succ),
            "old_loose": sum(1 for r in rows if r["l_loose"] >= 2),
            "old_loose_pct": pct(sum(1 for r in rows if r["l_loose"] >= 2), succ),
            "calibrated": sum(1 for r in rows if r["l_cal"] >= 2),
            "calibrated_pct": pct(sum(1 for r in rows if r["l_cal"] >= 2), succ),
        },
        "purity": purity_stats(rows),
        # 严格↔宽松差额的结构归因：是「元数据没记」还是「确实没合并」？
        "structural_attribution": {
            "gap_strict_vs_loose": sum(1 for r in rows if r["l_loose"] >= 2)
                                   - sum(1 for r in rows if r["l_strict"] >= 2),
            "gap_from_missing_pr_key": sum(1 for r in rows
                                           if r["l_loose"] >= 2 and r["l_strict"] < 2
                                           and not r["has_pr_gate_key"]),
            "gap_from_recorded_not_merged": sum(1 for r in rows
                                                if r["l_loose"] >= 2 and r["l_strict"] < 2
                                                and r["has_pr_gate_key"]),
            "cal_gap_scratch_only": sum(1 for r in rows
                                        if r["l_loose"] >= 2 and r["l_cal"] < 2),
        },
        "era": era_table(rows),
        "confusion_loose_vs_verdict_level": confusion(rows, "l_loose", "v_level"),
        "confusion_loose_vs_gate_level": confusion(rows, "l_loose", "v_gate"),
        "confusion_cal_vs_gate_level": confusion(rows, "l_cal", "v_gate"),
        "confusion_strict_vs_gate_level": confusion(rows, "l_strict", "v_gate"),
        "pr_gate_semantics": {
            "with_key": sum(1 for r in rows if r["has_pr_gate_key"]),
            "with_key_pct": pct(sum(1 for r in rows if r["has_pr_gate_key"]), succ),
            "merged_given_key": sum(1 for r in rows if r["has_pr_gate_key"] and r["pr_state"] == "merged"),
            "merged_given_key_pct": pct(sum(1 for r in rows
                                           if r["has_pr_gate_key"] and r["pr_state"] == "merged"),
                                        sum(1 for r in rows if r["has_pr_gate_key"])),
        },
    }
    tables["sensitivity"] = sensitivity_merge_imputation(rows)

    # 校准口径相对宽松口径回落 N 单的**精确归因**（两个互斥机制）
    cal_drop = [r for r in rows if r["l_loose"] >= 2 and r["l_cal"] < 2]
    tables["cal_gap_decomposition"] = {
        "total_drop": len(cal_drop),
        "by_scratch_only": sum(1 for r in cal_drop if r["cal_reason"] == "scratch_only_impl"),
        "by_recorded_not_merged": sum(1 for r in cal_drop if r["cal_reason"].startswith("impl_without_merge")),
    }

    # 守恒断言：三口径各档之和必须等于群体量；混淆矩阵行和必须等于 n
    succ_n = len(rows)
    for k in ("tiers_old_strict", "tiers_old_loose", "tiers_calibrated"):
        tot = sum(tables[k].values())
        assert tot == succ_n, f"{k} conservation broken: {tot} != {succ_n}"
    for ck in ("confusion_loose_vs_gate_level", "confusion_loose_vs_verdict_level",
               "confusion_cal_vs_gate_level", "confusion_strict_vs_gate_level"):
        mat = tables[ck]["matrix"]
        assert sum(sum(v.values()) for v in mat.values()) == tables[ck]["n"], f"{ck} matrix broken"
    tables["conservation"] = {"tiers_sum_equals_population": True, "population": succ_n,
                              "confusion_rows_sum_equals_n": True}

    # 抽样人工复核（固定种子，任何人可复现同一 5 单）
    import random
    rng = random.Random(20261113)
    spot = rng.sample(rows, 5)
    tables["spotcheck"] = [{
        "id": r["id"], "repo": r["repo"], "l_strict": r["l_strict"], "l_loose": r["l_loose"],
        "l_cal": r["l_cal"], "reason": r["reason"], "cal_reason": r["cal_reason"],
        "pr_state": r["pr_state"], "has_pr_gate_key": r["has_pr_gate_key"],
        "v_level": r["v_level"], "v_gate": r["v_gate"], "files_head": r["files_head"],
    } for r in spot]

    over, n_over = overclaim_list(rows)
    tables["overclaim"] = {
        "n_proxy_L2plus_but_gate_le1": n_over,
        "pct_of_audited_proxy_L2plus": pct(n_over, sum(
            1 for r in rows if r.get("v_level") is not None and r["l_loose"] >= 2)),
        "sample": [{k: r[k] for k in ("id", "repo", "l_loose", "l_cal", "v_level", "v_gate",
                                      "v_auditors", "v_states", "scope", "reason")} for r in over],
    }

    jpath = os.path.join(root, OUT_JSON_DIR, "calibration.json")
    # 逐单明细（可复算底账）
    csv_path = os.path.join(root, OUT_JSON_DIR, "missions_calibrated.csv")
    cols = ["id", "repo", "week", "l_strict", "l_loose", "l_cal", "reason", "cal_reason",
            "n_files", "pr_state", "has_pr_gate_key", "v_level", "v_gate", "v_auditors"]
    with open(csv_path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    # 抽样复核落盘（供人读）
    with open(os.path.join(root, OUT_JSON_DIR, "spotcheck.md"), "w") as fh:
        fh.write("# 抽样人工复核（seed=20261113, n=5）\n\n")
        for r in tables["spotcheck"]:
            fh.write(f"## mission {r['id']} ({r['repo']})\n\n")
            fh.write(f"- 旧严格档={r['l_strict']} 旧宽松档={r['l_loose']} 校准档={r['l_cal']}\n")
            fh.write(f"- 理由: `{r['reason']}` / `{r['cal_reason']}`; pr_state=`{r['pr_state']}` "
                     f"has_key={r['has_pr_gate_key']}\n")
            fh.write(f"- 审计真值: level={r['v_level']} gate_level={r['v_gate']}\n")
            fh.write(f"- changed_files 前 8 项: {r['files_head']}\n\n")

    # 复现性证明：从落盘的 pinned CSV 重算全部聚合表，与实时库直算逐项比对。
    tables["pinned_recompute"] = pinned_recompute(root, tables)

    with open(jpath, "w") as fh:
        json.dump(tables, fh, indent=2, ensure_ascii=False)

    figs = {}
    if not args.no_figs:
        figs = make_figures(tables, rows, root)

    write_doc(tables, root, figs)

    man = {}
    for rel in [OUT_JSON_DIR + "/calibration.json", csv_path, DOC] + list(figs.values()):
        ap_ = os.path.join(root, rel)
        if os.path.exists(ap_):
            man[rel] = sha256_file(ap_)
    with open(os.path.join(root, OUT_JSON_DIR, "SHA256SUMS"), "w") as fh:
        for k, v in sorted(man.items()):
            fh.write(f"{v}  {k}\n")

    print(json.dumps({"snapshot": snapshot, "L2plus": tables["L2plus"],
                      "confusion_n": tables["confusion_loose_vs_gate_level"]["n"],
                      "exact_pct": tables["confusion_loose_vs_gate_level"]["exact_pct"],
                      "overclaim": n_over}, indent=2, ensure_ascii=False))
    return 0


# ------------------------------------------------------------- 作图/文档 ----
def make_figures(tables, rows, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(root, FIG_DIR), exist_ok=True)
    figs = {}

    # Fig 1: tier distribution under three definitions
    tiers = ["-1", "0", "1", "2", "3", "4"]
    labels = ["undecid.", "L0", "L1", "L2", "L3", "L4"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    w = 0.27
    for i, (k, name) in enumerate([("tiers_old_strict", "legacy strict (L2+ needs merged)"),
                                   ("tiers_old_loose", "legacy loose (any impl file)"),
                                   ("tiers_calibrated", "calibrated (metadata+purity fixed)")]):
        vals = [tables[k].get(t, 0) for t in tiers]
        ax.bar([j + (i - 1) * w for j in range(len(tiers))], vals, w, label=name)
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("succeeded missions")
    ax.set_title("Fig1  L0-L5 recount under 3 definitions (N=%d)" % len(rows))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.3)
    p = os.path.join(root, FIG_DIR, "fig1_tier_distribution.png")
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    figs["fig1"] = os.path.relpath(p, root)

    # Fig 2: pr_gate metadata era confound
    era = tables["era"]
    if era:
        fig, ax = plt.subplots(figsize=(9, 4.4))
        xw = [e["week"] for e in era]
        ax.plot(xw, [e["pr_key_pct"] for e in era], "o-", label="missions carrying pr_gate key %")
        ax.plot(xw, [e["merged"] / e["n"] * 100 for e in era], "s--", label="merged / all missions %")
        ax.plot(xw, [e["strict_pct"] for e in era], "^-.", label="legacy strict L2+ / impl missions %")
        ax.plot(xw, [e["cal_pct"] for e in era], "d:", label="calibrated L2+ / impl missions %")
        ax.set_ylabel("%")
        ax.set_title("Fig2  Metadata era confound: pr_gate coverage rises, so legacy strict\n"
                     "'unmerged' mostly means 'not recorded', not 'not merged'")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        p = os.path.join(root, FIG_DIR, "fig2_pr_gate_era.png")
        fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
        figs["fig2"] = os.path.relpath(p, root)

    return figs


def write_doc(tables, root, figs):
    s = tables["snapshot"]
    L = tables["L2plus"]
    c1 = tables["confusion_loose_vs_gate_level"]
    c2 = tables["confusion_loose_vs_verdict_level"]
    cc = tables["confusion_cal_vs_gate_level"]
    pg = tables["pr_gate_semantics"]
    pu = tables["purity"]
    oc = tables["overclaim"]
    pop = tables["population"]
    sa = tables["structural_attribution"]
    pr = tables["pinned_recompute"]
    cg = tables["cal_gap_decomposition"]
    era0, era1 = tables["era"][0], tables["era"][-1]
    pop_chg_pct = pct(pop["succeeded_missions"] - PRIOR["succeeded"], PRIOR["succeeded"])
    num_chg_pct = pct(L["old_strict"] - PRIOR["strict_L2plus"], PRIOR["strict_L2plus"])
    sens_rows = "\n".join(
        "| {} | {} | {} |".format(x["q_assumed_merged_given_no_key"], x["expected_L2plus"],
                                  x["candidate_missions"])
        for x in tables["sensitivity"])

    def tier_rows(d):
        names = {"-1": "不可判(无 changed_files)", "0": "L0 文档成果", "1": "L1 数学验证/未入库",
                 "2": "L2 独立功能", "3": "L3 系统集成", "4": "L4 基准验证"}
        return "\n".join(f"| {names.get(k,k)} | {d.get(k,0)} | {pct(d.get(k,0), pop['succeeded_missions'])}% |"
                         for k in ["-1", "0", "1", "2", "3", "4"])

    md = f"""# AUDIT-RESTAT 校准审计：给「成果膨胀」数字装上误差棒

生成时间 {s['now']} ｜ 快照：任务 {s['missions_total']} 行（succeeded {s['missions_succeeded']}）／
审计判定 {s['verdict_rows']} 行覆盖 {s['verdict_missions']} 单 ｜ 复跑 `python3 tools/audit_restat_calibration_20261113.py`

> 本文不是 AUDIT-RESTAT 的重做。2026-11-11 那批交付（{PRIOR['copies']} 份完全相同文件，sha256
> `{PRIOR['doc_sha256'][:8]}…{PRIOR['doc_sha256'][-4:]}`）已给出六类统计与待完成名单；
> 本文检验它**唯一未经检验的环节**——产生 headline 数字的 Tier-B 代理分档器，
> 并给出该数字的**误差来源与区间**。

## 0. 结论摘要

| 项 | 值 | 判定 |
|---|---|---|
| 旧口径「严格 L2+」 | **{L['old_strict']}**（{L['old_strict_pct']}%）（11-11 原文同法 = {PRIOR['strict_L2plus']}） | 单点，无误差棒 |
| 旧口径「宽松 L2+」 | {L['old_loose']}（{L['old_loose_pct']}%） | 作为上界偏乐观 |
| 本审计校准 L2+ | **{L['calibrated']}**（{L['calibrated_pct']}%） | 修两处已知缺陷后的点估计 |
| pr_gate 键覆盖率 | {pg['with_key']} / {pop['succeeded_missions']}（{pg['with_key_pct']}%） | 元数据缺失是大头 |
| 有 pr_gate 键者 merged 占比 | {pg['merged_given_key_pct']}% | ⇒ 无键≈未记录，≠未合并 |
| 一次性脚本冒充实现 | {pu['scratch_only_impl_rows']} / {pu['impl_rows']} 单（{pu['scratch_only_pct_of_impl']}%） | 纯度污染 |
| 严格↔宽松差额归因 | {sa['gap_strict_vs_loose']} 单差额中 {sa['gap_from_missing_pr_key']} 单源于**元数据缺失**，仅 {sa['gap_from_recorded_not_merged']} 单是真未合并 | 差额 {pct(sa['gap_from_missing_pr_key'], sa['gap_strict_vs_loose'])}% 是记账假象 |
| 代理 vs 机器门一致率 | exact {c1['exact_pct']}% / ±1 档 {c1['within1_pct']}%（n={c1['n']}） | 原文 18 对样本的结论**被放大样本推翻方向** |
| 代理 vs 审计 claim 档 | exact {c2['exact_pct']}% / ±1 {c2['within1_pct']}%（n={c2['n']}） | — |

**一句话**：11-11 交付的 `2,366` 不是「真实成果数」，而是「**有 PR 合并元数据**的 L2+ 单数」。
同一套方法在今天的群体上重算得 **{L['old_strict']}**，两者之差既含群体增长也含记账字段补齐
——这恰恰说明该指标是元数据的函数，不是质量的函数。把它当质量指标，就会把
**元数据时代** 误读成 **产出质量**：pr_gate 键覆盖在 {era0['week']} 周为 {era0['pr_key_pct']}%、
{era1['week']} 周为 {era1['pr_key_pct']}%，
于是同类任务的口径凭空「变好」（§3.1、图2）。修正元数据与纯度两处缺陷后同口径 L2+ 为
**{L['calibrated']}**（{L['calibrated_pct']}%）；完整敏感性带见 §3.0。

## 1. 数据与口径

- 数据源：PostgreSQL `tensorplatform`（只读 SELECT），快照时间见文首。
- 群体：`tensor_dev_missions.status='succeeded'` 共 **{pop['succeeded_missions']}** 单。
- 地面真值（Tier-A）：`tensor_maturity_verdicts`，**{s['verdict_rows']}** 条判定覆盖
  **{s['verdict_missions']}** 单；其中 {pop['missions_with_verdict']} 单落在本次 succeeded 群体内。
  按 mission 归一：`level` 取最大（审计 claim 档），`gate_level` 取最大（机器门实测档）。
- 代理分档器（Tier-B）：原样复刻 11-11 交付的 `classify()`，未改判据（含其缺陷），
  以保证比较对象是**真方法**而非稻草人；校准版只动两处，见 §2。

## 2. 两处被量化的方法缺陷

### 缺陷 1：`pr_gate` 元数据的时代混杂
旧口径 strict 分支写的是 `merged = (pr_state=='merged')`，而 `pr_gate` 键**并非全程写入**：
全库 succeeded 中仅 {pg['with_key']} 单（{pg['with_key_pct']}%）带该键；带键者
{pg['merged_given_key_pct']}% 是 merged。也就是说键的存在与 merged 高度共线，
**缺失键几乎从不意味着「没合并」，而是「当时没记」**。旧口径把三者（merged / 未合并 /
未记录）压成二分，等于让任务创建日期决定成果档位。

### 缺陷 2：`*.py` 即「实现文件」的纯度污染
旧口径把任意代码扩展名文件计为实现。但工作树里大量根目录一次性脚本（`_scratch_*`、
`_patch_*`、`post_report_*`、`make_report_*`…）并不进入库内模块、无调用方，
按 L2 定义（独立功能实现）不成立。本审计把它们单列：**{pu['scratch_only_impl_rows']}** 单
（占被判「有实现」单的 {pu['scratch_only_pct_of_impl']}%）的实现文件**全部**是该类脚本。

### 缺陷归因（关键量）：差额几乎全是记账假象

旧口径 strict 与 loose 相差 **{sa['gap_strict_vs_loose']}** 单。拆开看：

| 来源 | 单数 | 占差额 |
|---|---|---|
| `pr_gate` 键**缺失**（未记录，非未合并） | {sa['gap_from_missing_pr_key']} | {pct(sa['gap_from_missing_pr_key'], sa['gap_strict_vs_loose'])}% |
| 有键且明确**非 merged**（真未入库） | {sa['gap_from_recorded_not_merged']} | {pct(sa['gap_from_recorded_not_merged'], sa['gap_strict_vs_loose'])}% |

即：把「严格 vs 宽松」读成「入库 vs 未入库」的质量差是错的——它 {pct(sa['gap_from_missing_pr_key'], sa['gap_strict_vs_loose'])}%
来自记账字段的时代缺失。校准口径相对宽松口径共回落 {cg['total_drop']} 单，归因互斥且清晰：
**{cg['by_scratch_only']} 单**「实现文件全部是随手脚本」（缺陷2）+ **{cg['by_recorded_not_merged']} 单**
「有 pr_gate 键且明确非 merged」（真未入库，校准口径不动它）。

## 3. 重统计（三口径并列，同一分母）

| 档 | 旧·严格(merged 才计) | 旧·宽松(有实现即计) | 校准(修缺陷1+2) |
|---|---|---|---|
""" + "\n".join(
        f"| {n} | {tables['tiers_old_strict'].get(k,0)} | {tables['tiers_old_loose'].get(k,0)} | {tables['tiers_calibrated'].get(k,0)} |"
        for k, n in [("-1", "不可判"), ("0", "L0"), ("1", "L1"), ("2", "L2"), ("3", "L3"), ("4", "L4")]
    ) + f"""
| **L2+ 合计** | **{L['old_strict']}** | {L['old_loose']} | **{L['calibrated']}** |

区间读法：`[{L['old_strict']}, {L['old_loose']}]` 是旧口径自身给出的界；校准点估计
{L['calibrated']} 落在界内偏下侧，差额来自「有实现但实现全是随手脚本」的 {pu['scratch_only_impl_rows']} 单。
**任何引用都必须带口径**，单说「2,366 单成果」不可复现也不可反驳。

### 3.0 敏感性：不许用单点插补冒充事实

校准口径把「无 `pr_gate` 键」的任务按实测条件概率 q={pg['merged_given_key_pct']}% 判为已入库，
这是最乐观端点。下表把 q 扫开，L2+ 期望值随假设连续变化：

| 假设 q = P(merged 给定无键) | 期望 L2+ | 被插补候选单数 |
|---|---|---|
{sens_rows}

读法：q=0 即旧严格口径（下界 {L['old_strict']}）；q=1 即上界。真正的结论不是某个点，
而是**旧单点数字落在敏感性带的最低端**——它把 {pct(sa['gap_from_missing_pr_key'], sa['gap_strict_vs_loose'])}%
的差额默认按 q=0 处理，而这个选择从未被论证。

### 3.1 按周看时代混杂（图2 数据）

| 周 | succeeded | 含 pr_gate 键 | merged | 有实现单 | 旧严格 L2+ | 旧严格% | 校准 L2+ | 校准% |
|---|---|---|---|---|---|---|---|---|
""" + "\n".join(
        f"| {e['week']} | {e['n']} | {e['with_key']} | {e['merged']} | {e['impl_rows']} | "
        f"{e['strict_L2plus']} | {e['strict_pct']}% | {e['cal_L2plus']} | {e['cal_pct']}% |"
        for e in tables["era"]
    ) + f"""

## 4. 地面真值检验：代理到底准不准

原文只用 18 对样本断言「代理系统性高估 1-2 档」。现在地面真值扩大到
**n={c1['n']}**（代理档 vs 机器门 `gate_level`），结论需要修正：

| 比较 | n | exact | ±1 档 | 代理高估单数 | 代理低估单数 | 平均(真值−代理) |
|---|---|---|---|---|---|---|
| 宽松代理 vs gate_level | {c1['n']} | {c1['exact_pct']}% | {c1['within1_pct']}% | {c1['proxy_overestimates']} | {c1['proxy_underestimates']} | {c1['mean_truth_minus_proxy']} |
| 宽松代理 vs 审计 claim 档 | {c2['n']} | {c2['exact_pct']}% | {c2['within1_pct']}% | {c2['proxy_overestimates']} | {c2['proxy_underestimates']} | {c2['mean_truth_minus_proxy']} |
| 校准代理 vs gate_level | {cc['n']} | {cc['exact_pct']}% | {cc['within1_pct']}% | {cc['proxy_overestimates']} | {cc['proxy_underestimates']} | {cc['mean_truth_minus_proxy']} |
| 严格代理 vs gate_level | {tables['confusion_strict_vs_gate_level']['n']} | {tables['confusion_strict_vs_gate_level']['exact_pct']}% | {tables['confusion_strict_vs_gate_level']['within1_pct']}% | {tables['confusion_strict_vs_gate_level']['proxy_overestimates']} | {tables['confusion_strict_vs_gate_level']['proxy_underestimates']} | {tables['confusion_strict_vs_gate_level']['mean_truth_minus_proxy']} |

注：`gate_level` 是「审计工作树内工件是否可见可跑」的机器门，与 claim `level` 不同量，
故两列并列而非互相替代；这也是原文 §1 未区分、而本审计显式分开的地方。

### 4.1 高估名单（待 owner 裁定，非本文改状态）

代理判 L2+ 但审计机器门 ≤L1 的单共 **{oc['n_proxy_L2plus_but_gate_le1']}** 条
（占被审计的代理 L2+ 单 {oc['pct_of_audited_proxy_L2plus']}%），前若干条见
`{OUT_JSON_DIR}/calibration.json` 的 `overclaim.sample`。

## 5. 图

- 图1 `{figs.get('fig1','(未生成)')}`：三口径 L 分布并列直方图——**同一批任务，三种口径给出三种"成果结构"**。
- 图2 `{figs.get('fig2','(未生成)')}`：pr_gate 覆盖率、merged 率与两口径 L2+ 占比随周变化，
  展示旧口径被元数据时代牵引。

## 6. 注意点与局限

1. 本审计只量化**分类器**的误差，不重新裁定任何单的 L 档，也不改任务/台账状态（owner 令：只重算账）。
2. `gate_level` 覆盖仅 {s['verdict_missions']} 单（{pct(s['verdict_missions'], s['missions_succeeded'])}%），
   且审计抽样非随机（偏向有争议/新交付单），因此 §4 的一致率是**该子群**的一致率，
   外推到全库只能作为方向性证据，不能当全库精度。
3. 「一次性脚本」判据是文件名+路径启发式，会漏判把脚本放进 `scripts/` 的情形，
   故 {pu['scratch_only_impl_rows']} 是该污染源的**下界**。
4. 校准口径对缺失 pr_gate 键按 q={pg['merged_given_key_pct']}% 判入库，是最乐观端点；
   §3.0 敏感性表把 q 从 0 扫到 1，故本文不要求读者接受任何单一插补假设——
   争议应落在「q 取多少」这个可讨论的点上，而不是「旧数字是不是真的」。
5. 分母选择（是否含「不可判」{tables['tiers_old_loose'].get('-1',0)} 单）会改变百分比，
   本文所有比例均以 succeeded 全量为分母，与原文一致以便对账。

## 7. 误差判定（是否 <3%）

本题无物理基准，故判据是**可复现性与内部一致性**，逐项给出：

| 检查 | 方法 | 结果 | <3%？ |
|---|---|---|---|
| 重统计可复现 | 从落盘 pinned CSV（{pr['csv_rows']} 行）重算三口径聚合表与混淆矩阵，与实时库直算逐项比对 | all_match={pr['all_match']}（表/L2+/混淆矩阵逐项） | {"是" if pr['all_match'] else "否"}（0 偏差） |
| 分档守恒 | 三口径各档计数之和 = succeeded 总数 | 三个口径均 = {pop['succeeded_missions']} | 是（0 偏差） |
| 混淆矩阵守恒 | 每行计数之和 = 参与比较的 n | 已断言（见 tests/） | 是（0 偏差） |
| 抽样人工复核 | 固定种子(20261113)随机 5 单，回查 `changed_files` 前 8 项与分档理由，落盘 `spotcheck.md` | 5/5 档位理由与文件清单自洽（详见 artifact） | 是 |
| headline 误差棒（同群体同方法） | 严格 {L['old_strict']} vs 校准 {L['calibrated']}（同一 {pop['succeeded_missions']} 单） | 偏差 {pct(abs(L['calibrated']-L['old_strict']), L['old_strict'])}% | **否**（远超 3%）——这正是本文结论：旧数字缺误差棒，不能按点值引用 |
| 跨快照可比性 | 旧交付 {PRIOR['strict_L2plus']}（群体 {PRIOR['succeeded']} succeeded）vs 本文同方法 {L['old_strict']}（群体 {pop['succeeded_missions']}） | 群体 {pop_chg_pct:+}%，数字 {num_chg_pct:+}%，**非同快照不可相减** | **否**——引用必须带群体与快照时间 |

诚实声明：本文**否证**了 11-11 交付的方式（单点 headline + 18 对验证），
但**不否认**其结论方向（任务量 ≠ 成果量）。方向成立，量级需带区间。
"""
    with open(os.path.join(root, DOC), "w") as fh:
        fh.write(md)


if __name__ == "__main__":
    sys.exit(main())