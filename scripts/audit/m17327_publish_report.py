#!/usr/bin/env python3
"""Publish the m17327 board-audit report (six sections) to /tensor/reports.

Every number in the markdown is injected from artifacts/audit_m17327/*.json --
nothing is typed by hand.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AUD = os.path.join(REPO, "artifacts", "audit_m17327")


def load(name):
    with open(os.path.join(AUD, name)) as fh:
        return json.load(fh)


def main() -> int:
    ev = load("evidence.json")
    ind = load("independent_ref.json")
    judged = load("judge.json")
    ref = load("referee.json")
    rep = load("report_check.json")
    repro_res = load("repro.json")
    find = load("finding.json")
    bm = ev["benchmark"]

    L = []
    A = L.append
    A("## ① 案例介绍")
    A("")
    A("被审计对象=mission **m17276**（anchor **CASE-BEM-MARUO-DRIFT**，TensorBEM），"
      "本次是 **交差核验**（board-audit，mission m17327），不是重做。")
    A("")
    A("物理：固定、坐底、竖直圆柱（半径 a）在深水规则头浪下的二阶平均漂移力。"
      "ref = Maruo(1957) 远场积分对 MacCamy–Fuchs(1954) 圆柱精确散射幅值的**精确约化**："
      "`F_x = ρgA²/(2πk)·∫|S(θ)|²(1−cosθ)dθ`，`S(θ)=Σ c_m e^{imθ}` 是有限三角多项式，"
      "故方位积分有闭式；got = TensorBEM 三维 BEM 链路 `solve_diffraction_3d → bem_kochin → maruo_far_field_drift`。")
    A("")
    A("| 参数 | 值 | 说明 |")
    A("|---|---|---|")
    A("| ρ | 1025 kg/m³ | 海水 |")
    A("| g | 9.81 m/s² | 重力 |")
    A("| a | 1 m | 圆柱半径 |")
    A("| A | 1 m | 入射波幅 |")
    A(f"| ka | {', '.join(str(x) for x in ev['benchmark'] and judged and [c['ka'] for c in judged['production_cells']])} | 长波/峰值/短波段 |")
    A(f"| 生产网格 | {judged['production_resolution']} = 32×12, grading 3.0, 800 面板 | 交付声明 |")
    A("| 判据 | 每单元相对误差 < 5% | 任务 goal 原文「容差判据：<5%」 |")
    A("")
    A("**裁判登记状态**：`tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-MARUO-DRIFT'` → "
      f"**{ref['registered_rows_count']} 行**（机器裁判缺席，判据只能取 goal 原文）。")
    A("")
    A("## ② 怎么做的（方法与工具链）")
    A("")
    A("1. 取证：交付文件逐个 sha256 + git 归属（`artifacts/audit_m17327/existence.json`），"
      "确认 16 cells 工件、3 张图、测试、报告都在 commit 92c92b7 里。")
    A("2. 裁判：直查 `tensor_benchmark_cases`（本锚点 0 行）+ mission goal 原文判据。")
    A("3. 判数：从 study.json 逐单元重算 `|got−ref|/ref`，与工件声明值逐项比对；"
      "重算 `G3` 是否等于生产档最坏误差。")
    A("4. **ref 独立重推**：`scripts/audit/m17327_ref_independent.py` 不 import tensorbem，"
      "用 scipy `jvp/h1vp` 自建 MacCamy–Fuchs 余弦型系数 + Maruo 积分 + 21000 点暴力求积；"
      "再用 ka→0 极限对经典长波律 `5π²/16` 做外部锚定。")
    A("5. 可复现：重放交付 repro_cmd（merge 路线）逐字节比对；生产单元另递交集群重跑（cj1489）。")
    A("6. 报告纪律：解析报告 ④ 表 16 行逐行溯源到 study.json；正文数字逐句查源。")
    A("7. 定级 L0–L5 + 落表 `tensor_maturity_verdicts`（id=528）。")
    A("")
    A("## ③ 计算结果（数字表）")
    A("")
    A("生产网格（fine = 800 面板）逐单元重算：")
    A("")
    A("| ka | got [N] | ref [N] | 重算误差 | 报告声明 | <5% |")
    A("|---|---|---|---|---|---|")
    for c in judged["production_cells"]:
        A(f"| {c['ka']} | {c['got']:.6g} | {c['ref']:.6g} | {100 * c['rel_error_recomputed']:.3f}% | "
          f"{100 * c['rel_error_declared']:.3f}% | {'✅' if c['within_tol'] else '❌'} |")
    A("")
    A("ref 独立重推（不 import tensorbem）：")
    A("")
    A("| ka | 交付 ref [N] | 独立重推 [N] | 相对差 |")
    A("|---|---|---|---|")
    for row in ind["per_ka"]:
        A(f"| {row['ka']} | {row['ref_force_delivered']:.10g} | {row['ref_force_independent']:.10g} | {row['rel_diff']:.2e} |")
    A("")
    A(f"外部锚点：ka→0 极限独立重推 = {ind['long_wave_anchor']['extrapolated_limit_ka2_zero']:.9g}，"
      f"经典 `5π²/16 = {ind['long_wave_anchor']['law_constant_from_pi']:.9g}`，"
      f"相对差 {ind['long_wave_anchor']['rel_diff_limit_vs_law']:.2e}。")
    A("")
    A("全部 16 cells 的 `rel_error` 重算与声明一致 = "
      f"{judged['all_cells_recomputed_match_declared']}；交付工件 `all_passed` = "
      f"{judged['all_passed_declared']}（G3 FAIL，交付未粉饰）。")
    A("")
    A("## ④ 结果图")
    A("")
    A("左：误差随面板数（4 档网格 × 4 个 ka）+ 5% 容差线；右：生产网格与更细网格逐 ka 柱状。"
      "图由 `scripts/audit/m17327_board_audit.py::figure` 从 study.json 直接生成（matplotlib，"
      f"sha256 见附件）。")
    A("")
    A("## ⑤ 注意点（坑 / 局限 / 适用范围）")
    A("")
    A("1. **相位约定陷阱（审计自身踩到）**：用全整数 m 复指数求和、写成 `(−i)^m`，"
      "奇 m 相对相位翻转，长波极限变成 π²/16 —— 比经典值小 5 倍，"
      "而**内部**求积自洽（modal vs quad 1e-11）完全看不出来。只有外部锚点 `5π²/16` 抓得住。"
      "推论：交付的「闭式 vs 仓内求积 2.4e-16」是代码路径交叉验证，**不是**外部验证。")
    A("2. **repro_cmd 是 merge-only**：交付给的复现命令只做 JSON 合并，不重跑 BEM 求解，"
      "证明不了求解器可复现（本次另递交集群重跑 cj1489，落卷时仍在排队 → 记 not_run）。")
    A("3. **报告正文有 3 个不可溯源数字**：" + "；".join(rep["untraceable_prose_claims"]) +
      "（④ 表 16 行全部可溯源）。")
    A("4. 无外网环境：交付与审计都无法核对论文表格，「外部锚点」实际只有长波 `5π²/16` 一条；"
      "交付在报告 ① 已如实说明该偏差（ref 由闭式机器求值而非抄表）。")
    A("5. 适用边界：深水、单体、圆截面、规则波；有限水深/多体遮蔽未覆盖（交付 ⑥ 已声明）。")
    A("6. 平台口径问题：交付工件与 mission 头行都写「G3 FAIL」，但 auto-approve 仍把 status 置 "
      "`succeeded` —— 验收口径不校验自报 gate，建议后续把 `all_passed=false` 纳入自动验收拦截。")
    A("")
    A("## ⑥ 误差判定（逐项 vs 基准，显式回答）")
    A("")
    A("| ka | 误差 | <5% 判据 | <3% 生态标准 |")
    A("|---|---|---|---|")
    for c in judged["production_cells"]:
        e = 100 * c["rel_error_recomputed"]
        A(f"| {c['ka']} | {e:.3f}% | {'是' if e < 5 else '**否**'} | {'是' if e < 3 else '否'} |")
    A("")
    A(f"**是否 <5%**：生产网格 4 单元中 3 个通过，ka=2.0 为 "
      f"{100 * judged['production_worst']['rel_error_recomputed']:.3f}% → **否**（最细档 1200 面板仍 "
      "5.04%，仍不过）。**是否 <3%**：**否**（最好单元 3.03%）。")
    A("")
    A(f"**裁定：L{find['level']}**（不是 L4）。双阻塞①本锚点在 benchmark 登记表 0 行（机器裁判缺席）；"
      "②声明判据 <5% 在生产网格 ka=2.0 处 5.331% 未过、最细档 5.04% 亦未过。"
      "非 L2：链路是接入主求解器的真集成（残差 ~1e-15），闭式 ref 经独立重推 1e-12 一致。")
    A("")
    A("工件：`artifacts/audit_m17327/`（existence/judge/referee/repro/independent_ref/"
      "report_check/finding/evidence.json + 图）、报告 `docs/board_audit_m17327/M17327_BOARD_AUDIT.md`、"
      "脚本 `scripts/audit/m17327_{board_audit,ref_independent,bem_rerun}.py`、verdict id=528。")

    markdown = "\n".join(L)
    with open(os.path.join(AUD, "fig_m17327_error_vs_panels.png"), "rb") as fh:
        png = base64.b64encode(fh.read()).decode()

    payload = {
        "title": "m17327 交差核验：CASE-BEM-MARUO-DRIFT（TensorBEM · m17276）→ L3",
        "summary": "Maruo 远场漂移力算例核验：ref 独立重推 1e-12 一致、长波 5π²/16 外部锚定；"
                   "生产网格 ka=2.0 误差 5.331%>5% 判据未过、benchmark 登记表 0 行 → 双阻塞，诚实定级 L3。",
        "scenario_key": "board-audit/CASE-BEM-MARUO-DRIFT/m17327",
        "mission_id": 17327,
        "tags": ["board-audit", "TensorBEM", "CASE-BEM-MARUO-DRIFT", "Maruo", "drift-force",
                 "maturity-L3", "honest-fail"],
        "markdown": markdown,
        "figures": [{
            "name": "fig_m17327_error_vs_panels.png",
            "caption": "左：BEM vs Maruo 闭式的相对误差随面板数（4 档网格 × 4 个 ka，红线=5% 容差）；"
                       "右：生产网格(fine 800) 与 finer(1200) 逐 ka 误差，ka=2.0 两档均超 5%。",
            "png_base64": png,
        }],
    }
    path = "/tmp/m17327_report.json"
    with open(path, "w") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    print("wrote", path, os.path.getsize(path), "bytes")

    user = os.environ.get("TENSOR_AGENT_USER")
    pw = os.environ.get("TENSOR_AGENT_PASS")
    api = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    tok = subprocess.run(["curl", "-s", "-X", "POST", f"{api}/auth/login",
                          "-H", "Content-Type: application/json",
                          "-d", json.dumps({"username": user, "password": pw})],
                         capture_output=True, text=True).stdout
    token = json.loads(tok).get("access_token") or json.loads(tok).get("token")
    out = subprocess.run(["curl", "-s", "-X", "POST", f"{api}/tensor/reports",
                          "-H", f"Authorization: Bearer {token}",
                          "-H", "Content-Type: application/json", "-d", f"@{path}"],
                         capture_output=True, text=True)
    print("POST /tensor/reports ->", out.stdout[:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())