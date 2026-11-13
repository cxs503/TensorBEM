#!/usr/bin/env python3
"""Publish the m17398 board-audit report to the platform (six required elements).

Numbers are injected from the audit artifacts; figures are the two PNGs this
audit produced.  Usage: python3 post_report.py
"""
from __future__ import annotations

import base64
import json
import pathlib
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
REFS = HERE / "refs"
ART = HERE / "artifacts"
ENV = dict(l.strip().split("=", 1) for l in open("/root/tp-agent/agent.env")
           if l.strip() and "=" in l and not l.startswith("#"))
BASE = ENV["TENSOR_API_BASE"]


def api(method: str, path: str, payload=None, token: str | None = None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data=data, timeout=120) as r:
        return r.status, json.loads(r.read() or b"{}")


def fig(name: str, caption: str) -> dict:
    return {"name": name, "caption": caption,
            "png_base64": base64.b64encode((ART / name).read_bytes()).decode()}


def main() -> int:
    ev = json.loads((ART / "verdict_evidence.json").read_text())
    study = json.loads((REFS / "docs/case_bem_maruo_drift/study.json").read_text())
    indep = json.loads((ART / "independent_ref_v2.json").read_text())
    b = ev["benchmark"]
    prod = study["parameters"]["production_resolution"]
    prod_cells = sorted((c for c in study["cells"] if c["resolution"] == prod), key=lambda c: c["ka"])
    ladder = "\n".join(
        f"| {c['ka']:g} | {c['panel_count']} | {c['bem_force']:.4f} | {c['ref_force']:.4f} | "
        f"{100 * c['rel_error']:.3f}% | {'PASS' if c['rel_error'] < 0.05 else 'FAIL'} | "
        f"{'PASS' if c['rel_error'] < 0.03 else 'FAIL'} |" for c in prod_cells)
    worst = max(prod_cells, key=lambda c: c["rel_error"])
    repro = "\n".join(f"| {r['cell']} | {r['delivered_force']:.4f} | {r['recomputed_force']:.4f} | "
                      f"{r['abs_diff']:.1e} |" for r in ev["auditor_reproduction"])

    md = f"""## ① 案例介绍（审计对象与物理背景）

**交差对象**：m17336（branch `agent/m17336` @ `{ev['subject_commit'][:7]}`，succeeded），锚点
`{ev['anchor']}`——固定坐底竖直圆柱（a=1 m，深水）在规则头浪下的二阶平均漂移力，用 Maruo (1957) 远场
公式 `F_x = ρgA²/(2πk)·∮|S(θ)|²(1-cosθ)dθ` 作为闭式参考，检验 TensorBEM 的三维 BEM 链路
`vertical_cylinder_hull_mesh → solve_diffraction_3d → bem_kochin → maruo_far_field_drift`。

参数：ρ=1025 kg/m³，g=9.81 m/s²，a=1 m，A=1 m，ka ∈ {{0.2, 0.5, 1.0, 2.0}}，H/a=16（深水），
m_max=14，n_azimuth=96，n_theta=2048；网格阶梯 312/544/800/1200 三角形，**生产档 = fine (800)**。

**任务性质**：交差核验单（非重做）。唯一裁判 = 基准台账 `tensor_benchmark_cases`。

## ② 怎么做的（取证流程）

1. 读代码：算例驱动 `src/tensorbem/benchmarks/maruo_drift_far_field_case.py`（闭式 ref :153-212，
   被测路线 bem_case :214-252），生产核 `src/tensorbem/drift_far_field3d.py`（`bem_kochin` 只吃 BEM 散射势
   与 `-∂φ_I/∂n`，无解析注入）。
2. **独立性检验**：审计方不 import TensorBEM，从第一性原理重建 MacCamy–Fuchs 远场幅值（入射展开→物面条件→
   出射 Hankel 辐射条件），用「模态恒等式」与「暴力梯形求积」两条互不相同的角向积分互校，再与交付 ref 比对，
   并在 ka→0 比对长波律 5π²/16。脚本 `docs/board_audit/m17398/independent_ref_check.py`。
3. **本地逐位复现**两格（不是重做全算例），在 m17336 检出内跑 module 入口。
4. **集群复现**：`run_cluster` 递交 cj1530（stage_dir=被审工作树，完整 4×4 网格）；报告时刻仍 queued → **NOT RUN**。
5. 测试：`PYTHONPATH=src pytest tests/test_maruo_drift_far_field_case.py -q` → **13 passed in 2.48 s**。
6. 判级 → `tensor_maturity_verdicts` 落表 → maturity_gate 自动裁定（审计已用 gate 的 `judge()` 预演 `(3, [])`）。

## ③ 计算结果（数字表，程序化注入）

**基准裁判查询**：`SELECT … FROM tensor_benchmark_cases WHERE anchor_slug='{ev['anchor']}'` →
**{b['registered_referee_rows']} 行**（无机器裁判；容差只存在于 mission 文本）。

生产网格（fine, 800 面板）ref vs got：

| ka | panels | got F_BEM [N] | ref F_closed [N] | rel err | <5% | <3% |
|---|---|---|---|---|---|---|
{ladder}

生产网格失败格数 **{b['production_n_failures']}/{len(prod_cells)}**，最差 ka={worst['ka']:.1f} {100 * worst['rel_error']:.3f}%；
最密网格（1200）最差 {100 * b['finest_mesh_worst_rel_error']:.3f}%（仍 >5%）→ 非网格偶然。

本地逐位复现（同一 commit）：

| cell | 交付 got [N] | 审计重算 [N] | |Δ| |
|---|---|---|---|
{repro}

ref 独立重建（第一性原理，零 TensorBEM import）：与交付 ref 最大相对差
**{indep['max_rel_diff_vs_delivered']:.1e}**（受交付值 4 位小数存储限制；同精度全量值比对 5e-16），
审计方自身「模态 vs 求积」{indep['max_modal_vs_quadrature_rel']:.1e}；长波律 5π²/16 恢复到
{indep['long_wave'][0]['rel_err']:.1e}（ka=0.01）/ {indep['long_wave'][-1]['rel_err']:.1e}（ka=0.05）。

**判级 = L3**（L4 拒绝，见 ⑥）。verdict 落表：`tensor_maturity_verdicts` id=545，
scope `m17336 CASE-BEM-MARUO-DRIFT`，evidence 键=impl_files/entry/run_artifacts/tests/integration_callers/
benchmark/repro_cmd/external_anchor（值均为真实指针，gate 复核返回 `(3, [])`）。

## ④ 结果图

- `fig_m17398_error_vs_panels.png`：4 个 ka × 4 档网格的误差阶梯，叠 5%（本算例判据）与 3%（生态目标）；
  ka=2 在 800 与 1200 面板均落 5% 线之上。
- `fig_m17398_ref_validation.png`：左=生产网格 got vs ref 直方对比；右=审计方独立重建 vs 交付 ref 的相对差
  （~1e-8，量化噪声级，证明参考值不是自证）。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **无裁判行 ≠ 算例无效**：台账 0 行只说明「平台无法机器裁定」，不否定交付的算例本身；但按「基准判据是唯一
   裁判」的纪律，本单不能给 L4。修法：登记 case_name/metric/tolerance/ref_source 一行。
2. **`artifacts/` 被 .gitignore 忽略**（`.gitignore:10`）：verdict 的 `run_artifacts` 若指向
   `artifacts/…` 会被 gate 判「工件缺失」（m17276 即如此被 reopen）。本单改指 tracked 的 `docs/case_bem_maruo_drift/`
   与审计工作树内的审计自产工件——这是可复用教训（F7）。
3. **入口健壮性小坑**：`--resolutions` 不含生产档位时 `ValueError: max() arg is an empty sequence`
   （`maruo_drift_far_field_case.py:396-397` 对空 production 列表取 max），一行守卫即可。
4. **报告散文仍有三处不可溯源数字**（`docs/case_bem_maruo_drift_report.md:151-153`：800 面板写成「832 三角形」、
   study 里不存在的「416 三角形」档、「fine 的 3.01%」实为 finer 的 3.006%）——正文表格全对，仅 ⑤ 散文有瑕疵。
5. **审计方法论教训（诚实记录）**：独立重建 v1 用负阶次折叠捷径，整体偏 20%（长波系数 π²/4 vs 5π²/16），
   靠自身两条路线不自洽自检出，改为全整数阶直接求和后才收敛到 1e-12 —— 不是交付缺陷，是审计方的方法学纠错。
6. **适用范围**：深水、固定体、无航速、一阶场、二阶平均力；有限水深与有航速不在本算例内。

## ⑥ 误差判定（逐项，显式回答）

| 关键量 | 基准 | 误差 | <3%? | <5%? |
|---|---|---|---|---|
| F_x ka=0.2 生产网格 | 闭式（独立复现） | {100 * [c for c in prod_cells if c['ka'] == 0.2][0]['rel_error']:.3f}% | 否 | 是 |
| F_x ka=0.5 生产网格 | 同上 | {100 * [c for c in prod_cells if c['ka'] == 0.5][0]['rel_error']:.3f}% | 否 | 是 |
| F_x ka=1.0 生产网格 | 同上 | {100 * [c for c in prod_cells if c['ka'] == 1.0][0]['rel_error']:.3f}% | 否 | 是 |
| F_x ka=2.0 生产网格 | 同上 | {100 * [c for c in prod_cells if c['ka'] == 2.0][0]['rel_error']:.3f}% | 否 | **否（honest FAIL）** |
| 长波系数 ka=0.02 | 5π²/16 | {indep['long_wave'][1]['rel_err']:.2e} | 是 | 是 |
| 闭式 ref vs 仓内独立求积 | 两条独立路线 | {study['gates'][0]['measured']:.2e} | 是 | 是 |

**「是否 <3%」= 否**（生产网格最好 {100 * min(c['rel_error'] for c in prod_cells):.3f}%，无格达标；
最密网格最好 {100 * min(c['rel_error'] for c in study['cells'] if c['panel_count'] == 1200):.3f}%）。
**「是否 <5%」= 部分成立**（ka≤1.0 成立，ka=2.0 不成立）→ 整体 FAIL，交付方自报 `all_passed=false` 未粉饰。

**判级：L3（L4 拒绝）**。两项 L4 阻断项自 m17276 起未变；m17336 只做 re-land + 能力登记，故不升L4。
**未运行**：集群完整 4×4 复现 cj1530 报告时仍 queued（NOT RUN，非 PASS）；Maruo 1957/1960 印刷原表因无外网未逐页核对。
"""
    payload = {
        "title": "m17398 交差审计：m17336 CASE-BEM-MARUO-DRIFT (TensorBEM) → L3，L4 拒绝",
        "summary": ("核验三件套+基准裁判：台账 0 行（无机器裁判）、生产网格 ka=2.0 超差 5.331%（最密网格仍 5.037%）→ "
                    "L4 拒绝；真实 BEM 链路 + 13 测试 + 两格逐位复现 + 参考值第一性原理独立重建到 1e-8 → L3 成立，"
                    "verdict id=545，gate 预演 (3, [])。"),
        "scenario_key": "board-audit/maruo-drift",
        "mission_id": 17398,
        "tags": ["board-audit", "TensorBEM", "CASE-BEM-MARUO-DRIFT", "maturity-verdict", "L3", "honest-FAIL"],
        "markdown": md,
        "figures": [
            fig("fig_m17398_error_vs_panels.png",
                "误差-网格阶梯（4 ka × 4 档），5% 案例判据与 3% 生态目标；ka=2 在 800/1200 面板均落线之上"),
            fig("fig_m17398_ref_validation.png",
                "左：生产网格 got vs ref；右：审计方第一性原理独立重建 vs 交付 ref（~1e-8）"),
        ],
    }
    path = pathlib.Path("/tmp/report_m17398.json")
    path.write_text(json.dumps(payload, ensure_ascii=False))
    st, d = api("POST", "/auth/login", {"username": ENV["TENSOR_AGENT_USER"],
                                        "password": ENV["TENSOR_AGENT_PASS"]})
    token = d.get("access_token")
    st2, d2 = api("POST", "/tensor/reports", payload, token=token)
    print(st2, json.dumps(d2)[:400])
    return 0 if st2 < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())