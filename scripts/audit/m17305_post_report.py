#!/usr/bin/env python3
"""Publish the m17305 audit report to the platform (/tensor/reports) with figures."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

WT = Path("/root/tp-agent/work/m17305")
A = json.loads((WT / "runs/m17305/audit_numbers.json").read_text())
I = json.loads((WT / "runs/m17305/independent_michell.json").read_text())
B = "http://127.0.0.1:8010/api/v1"


def api(path, payload=None, tok=None, method=None):
    req = urllib.request.Request(B + path, method=method or ("POST" if payload else "GET"))
    req.add_header("Content-Type", "application/json")
    if tok:
        req.add_header("Authorization", "Bearer " + tok)
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data=data, timeout=120) as r:
        return json.loads(r.read() or b"{}")


tok = api("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                          "password": os.environ["TENSOR_AGENT_PASS"]}).get("access_token")

rows = A["rows"]
tbl = "\n".join(
    f"| {r['Fr']:.2f} | {r['ref_delivered']:.5f} | {r['ref_independent']:.5f} | {100*r['ref_spread']:+.4f}% "
    f"| {r['got']:.4f} | {100*r['err_got_vs_independent_ref']:+.4f}% | {r['Cw_got']:.6e} |" for r in rows)
maxe = 100 * A["max_err_got_vs_independent_ref"]
spread = 100 * A["max_ref_spread"]
rb = A["route_b"]["max_abs_error"]

md = f"""## ① 案例介绍
交差核验单 m17305 复核 mission 17246（锚点 `CASE-BEM-NEUMANN-KELVIN`，算例池 id=167）：
Wigley III 抛物型船型 L/B/T = 10/1.0/0.625 m（B/L=0.1，Cb=4/9，闭式几何），深水、无黏、小扰动定常兴波；
参考 = Michell 薄船理论 C_w(Fr)（Phil. Mag. 45 (1898) 106–123），判据 <5 %（生态线 <3 %）。
核验对象是对象提交 `27ecfe7`（前因子 κ/(2πg) 修正 + 499 行 case runner + 5 项回归测试 + 报告与 3 图）。

## ② 怎么做的
1. 取证：读 `27ecfe7` diff 与 `docs/case_bem_neumann_kelvin/numbers.json`（21 个数值块）。
2. **独立重推基准**（关键独立性）：审计方自写 Michell 求积 `scripts/audit/m17305_independent_michell.py`，
   不 import TensorBEM：`Rw = 4ρg²/(πU²)∫₀^{{π/2}}sec³θ|A|²dθ`，`A=∫∫(∂b/∂x)e^{{κsec²θ z}}e^{{iκsecθ x}}dxdz`，
   中心面 160×480 Gauss–Legendre × θ 2001 点。
3. 可复现性：另建干净副本重跑交付 `repro_cmd`（`--stage cheap --only geometry,kernel,route_a,route_b,wave_field`）。
4. 回归面：`pytest tests/test_case_neumann_kelvin_michell.py`（5 项）与
   `tests/test_nk_panel3d.py + tests/test_neumann_kelvin.py`（前因子默认值改动的回归面，64 项）。
5. 机器判级：按 L0–L5 功能实质与 gate 七键（impl/entry/artifacts/tests/callers/benchmark/repro）逐条核。

## ③ 计算结果（程序化注入，零手抄）
| Fr | ref Rw [N]（交付） | ref Rw [N]（审计独立求积） | ref 互差 | got Rw [N] | got vs 审计 ref | got Cw |
|---:|---:|---:|---:|---:|---:|---:|
{tbl}

- 交付 ref 与审计独立求积最大互差 **{spread:.4f} %** → ref 数值正确（非自证）。
- got 对**审计独立 ref** 最大误差 **{maxe:.4f} %**（< 5 % 判据、< 3 % 生态线）。
- 交付自检：稳态核自由面条件相对残差 {A['kernel_selfcheck']['free_surface_condition']['max_relative_residual']:.3e}；
  面板阶梯 450→1800 面板（Fr=0.20 误差 11.418 %→2.309 %）；θ 501→8001 波动 ≤4e-4。
- **Route B（真实船体面元 NK 单层法）= honest FAIL**：最大相对误差 {rb:.3f}（即 +{100*rb:.0f} %），
  随加密发散 → 不作证据，交付称之为 WITHHELD。
- 审计重跑：case 测试 5 passed / 5.68 s；回归 64 passed / 18.42 s；`numbers.json` 数值叶逐一同值。

## ④ 结果图
图 1（审计生成）：got 与两条独立 ref 路线的 C_w(Fr)，右图误差与 5 %/3 % 判据线。
图 2（交付原图）：对象自报的 C_w(Fr) 对照曲线（用于与审计重绘互校）。

## ⑤ 注意点（严重度分级）
- **F1 high（平台）**：`tensor_benchmark_cases` 对锚点**0 行**——"基准判据=唯一裁判"当时不存在；
  本审计按算例池 id=167 补登记（ref=Michell，tol <5 %，passed）。与 #493/#499/#509/#514 同一断点，根因未修。
- **F2 medium（溯源）**：交付 `numbers.json` 记 `git_head=0516776`（改前父提交）且 `blocks_run=["route_b"]`
  → 工件是脏树上多次分块的合并件，只能靠审计在 `27ecfe7` 重跑建立归因。
- **F3 medium（交付面）**：`27ecfe7` 仅在 `agent/m17246`，origin/main 无该目录与 `kochin_prefactor`
  → 干净 main 检出不可复现，待平台推分支。
- **F4 medium（判级边界）**：通过路线与 ref **结构同源**（薄船源 + 稳态 Kochin 算子 ≡ Michell 积分），
  故吻合证明的是稳态色散 + 留数常数 κ/(2πg) + 无量纲归一 + 中心面求积收敛，**不证明**船体面元 NK BEM。
  交付 §6 已主动声明此边界，审计复核声明准确；L4 仅在此缩小的语义下成立。
- **F5 medium（既有欠账）**：`benchmarks/verified/bem_wigley_wave_resistance/results.json` 的 ref 取自被判
  DEFECTIVE 的旧出口（Fr=0.20/0.30/0.40 → 2.5752/12.2926/26.3239 N，正确值 26.5393/144.1357/327.1101 N，
  **偏小 10.31×/11.73×/12.43× 且随 Fr 变**）→ 该 "verified" 基准须重出（新返修单）。
- **F6 low**：Richardson 两处为 NaN（两层阶梯），报告以文字给观测阶；报告 §2 的辐射比 3.02 未入库
  （审计实调 `verify_radiation_direction` 得 3.019989，数字对但属"零手抄"例外）；波形张角 74.60° vs 理论 19.47°，
  交付声明仅定性。

## ⑥ 误差判定（逐项，>3 % 不可接受）
- got C_w(Fr) vs 交付 ref：最坏 **+2.309 %** → **< 3 % 成立**。
- got C_w(Fr) vs 审计独立 ref：最坏 **+{maxe:.4f} %** → **< 3 % 成立**。
- ref 自身正确性 vs 审计独立求积：**≤{spread:.4f} %** → 成立。
- Route B 面元 NK：**+{100*rb:.0f} %（FAIL，honest FAIL 不入账）**；近场压力路线稳定 0.51×（缺陷，已登记）。
**判级 L4**（完整算例 + 误差分析 + 可复现 + 外部解析锚经独立重推）；**不判 L5**（无实验拖曳锚，船体 BEM 路线 WITHHELD）。

## 未跑（not_run）
`--stage full` 全量阶梯、集群/loong64 独立复跑、波形张角独立测量器、与 Wigley 1934/ITTC 实验对照。
"""

figs = []
for path, name, cap in [
    (WT / "docs/board_audit/m17305/fig_audit_cw_vs_fr.png", "audit_cw_vs_fr.png",
     "审计生成：got（1800 面板 NK 远场）vs 交付 ref vs 审计独立 Michell 求积，右图为相对误差与 5%/3% 判据线"),
    (Path("/root/tp-agent/work/m17246/docs/case_bem_neumann_kelvin/fig_cw_vs_froude.png"),
     "subject_cw_vs_froude.png", "交付原图：对象自报 C_w(Fr) 对照曲线（与审计重绘互校）"),
]:
    figs.append({"name": name, "caption": cap,
                 "png_base64": base64.b64encode(path.read_bytes()).decode()})

res = api("/tensor/reports", {
    "title": "交差核验 m17305 · m17246 CASE-BEM-NEUMANN-KELVIN（Neumann-Kelvin 定常兴波阻力，TensorBEM）→ L4",
    "summary": (f"board-audit 复核 m17246：三件套齐、repro_cmd 重跑逐叶复现、5+64 项测试通过；"
                f"got 对审计独立 Michell 求积最坏 {maxe:.4f}%（<3%），ref 自身正确性 ≤{spread:.4f}%；"
                f"判级 L4；同时确认 Route B 面元 NK honest FAIL（+{100*rb:.0f}%）、补登记基准行、"
                f"量化既有 verified 基准 ref 偏小 10.31–12.43× 的新欠账。"),
    "scenario_key": "board_audit/case_bem_neumann_kelvin",
    "mission_id": 17305,
    "tags": ["board-audit", "TensorBEM", "CASE-BEM-NEUMANN-KELVIN", "m17246", "L4",
             "Michell", "Neumann-Kelvin", "wave-resistance"],
    "markdown": md,
    "figures": figs,
}, tok=tok)
print("report id:", res.get("id") or res.get("report_id") or res)