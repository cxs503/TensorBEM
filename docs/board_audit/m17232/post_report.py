#!/usr/bin/env python3
"""Post the m17232 board-audit report (six sections, numbers injected from analysis.json)."""
import base64, json, os, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = json.loads((HERE / "analysis.json").read_text())
rows = A["rows"]; best = rows[-1]
tok = Path("/tmp/tok").read_text().strip()

def api(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(os.environ["TENSOR_API_BASE"] + path, data=data, method="POST" if body else "GET",
                                 headers={"Authorization": "Bearer " + tok, "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))

table = "\n".join(
    f"| sub{r['sub']} | {r['panels']} | {r['m_A_subject']:.6f} | {r['m_A_rerun']:.6f} | {r['rel_err_subject']:.4e} | {'PASS' if r['pass_3pct'] else 'FAIL'} |"
    if r["m_A_rerun"] is not None else
    f"| sub{r['sub']} | {r['panels']} | {r['m_A_subject']:.6f} | (复跑在途) | {r['rel_err_subject']:.4e} | {'PASS' if r['pass_3pct'] else 'FAIL'} |"
    for r in rows)
ours = " / ".join(f"{o['p_log2']:.2f}" for o in A["integral_order"])
mis = " / ".join(f"{o['p_log4_misread']:.2f}" for o in A["integral_order"])
flux = " / ".join(f"{o['p_log2']:.2f}" for o in A["flux_order"])
nc = A["negative_control"]

md = f"""# 交差审计 m17176 CASE-BEM-SPHERE-ADDED-MASS（球体附加质量，TensorBEM）

## ① 案例介绍
物理：半径 R=1 m 球体以 U=1 m/s 在无限域不可压无粘流体中平动，经典附加质量 m_A=0.5ρV（Lamb 1932 §6.10；Newman 1977 eq.4.132 球极限）。
几何：平面化 icosphere，80/320/1280/5120 面板（sub1–sub4）；ρ=1025 kg/m³，ref=0.5ρV={best['ref_0p5_rhoV']:.10f} kg。
判据：关键量相对误差 <3%（任务书容差）。本单为**交差核验**（非重做），对象是 mission 17176 已 succeeded 的算例。

## ② 怎么做的
1. 只读取证 m17176 worktree 的 runs/m17176/* 与 tensor_reports#16204，逐项核 run_artifacts / ref-vs-got / repro_cmd 三件套。
2. 基准裁判取 `tensor_benchmark_cases`（anchor_slug=CASE-BEM-SPHERE-ADDED-MASS）。
3. 独立复算：在本仓分支 agent/m17232 用同一生产入口重跑 sub0–sub4，逐位比对。
4. 负控：换角向模态（cos2θ）与半径 ×2，判定输出是否真由输入驱动。
5. 收敛阶按 p=log2(e_s/e_{{s+1}}) 重算（icosphere 每细分边长减半），与 README 声明对表。
工具链：python3.11 / numpy / pytest / psql(裁判表) / matplotlib。

## ③ 计算结果（数字由 analyze.py 从工件注入）

| sub | panels | m_A 主件 | m_A 审计复算 | 相对误差 | <3% |
|---|---|---|---|---|---|
{table}

ref=0.5ρV={best['ref_0p5_rhoV']:.10f}；仓内闭式 analytic.ellipsoid_added_mass((1,1,1),1025)=2146.7549799481294（相对差 2.28e-12）。
主件 sub1/sub2/sub3 复算**逐位一致**（abs diff 0.0）；标度不变性：ρ=1000/U=2 的 rel_err={A['scale_invariance']['scaled']['rel_err']:.6e}，与基准参数差 <1e-15。
负控：cos2θ 模态 m_A={nc['wrong_mode2_cos2theta_added_mass']:.4f}（偏 {nc['wrong_mode2_relative_error_vs_ref']*100:.2f}%）；半径 2.0 → {nc['radius2_added_mass']:.4f} = 精确 8×。

## ④ 结果图
图 1（fig_audit_sphere_added_mass.png）：左=附加质量相对误差 vs 面板数（主件圆点 / 审计复算叉号 / O(h²) 斜率参考 / 3% 判据线），右=逐面板通量相对 L2 误差（近 O(h¹)）。两轴均 log-log。

## ⑤ 注意点（发现，按严重度）
- **F1 medium：主件报告收敛阶算错一倍。** 报告记「观测阶 0.95/0.98/0.99」并据此断言"积分继承通量一阶、README 的 h² 声明存疑"。审计重算 p=log2(e_s/e_{{s+1}}) = **{ours}** ≈2，与 README「near 2」一致；报告那三个恰是 log4 空间值（{mis}）——除错底数。真正近一阶的是逐面板通量：{flux}。**不影响任何 m_A 数值与 <3% 判定。**
- **F2 medium（系统性）：** 裁判表 `tensor_benchmark_cases` 本算例 0 行 → 判据此前只在任务书文本里。本审计已登记该 anchor 行（insert_verdict.sql），后续可机器复核。
- **F3 low：** 主件称 sub3/sub4 递交集群 cj1448，实际 cj1448 终态=killed 且无集群产物被引用（报告亦自述数字由本地路先出）；数值已由审计本地复跑独立证实。
- **F4 low（范围上限）：** 施加解析 Dirichlet 迹 u=−½Ua·cosθ 再解 q → 验证外域单层/DtN 算子，非移动体 Neumann BVP，故封顶 L4。
- **F5 info：** 80 面板误差 6.4759e-2 > 3%（主件已如实判 FAIL）；320 面板起 PASS。

## ⑥ 误差判定（逐项 vs 基准，显式回答）
| 面板数 | 关键量相对误差 | vs 3% 判据 | 是否 <3% |
|---|---|---|---|
""" + "\n".join(f"| {r['panels']} | {r['rel_err_subject']:.4e} | {'裕度 '+format((r['rel_err_subject']*100),'.3f')+'% vs 3%' if r['pass_3pct'] else '超线'} | {'是' if r['pass_3pct'] else '否'} |" for r in rows) + f"""

**结论：是。** 320 面板起全部 <3%，最细网格（5120 面板）{best['rel_err_subject']*100:.4f}%（裕度约 {3/(best['rel_err_subject']*100):.0f}×）；80 面板 6.48% 未达线（主件已如实标注 FAIL，属网格粗糙而非方法错误）。
未做项（honest not_run）：sub4 的审计本地复跑在交付时仍在跑（集群 cj1466 亦在排队）；球体附加质量无实验锚，闭式即唯一基准；自由面/船体适用性不在本算例范围。

## 判级与落表
**L4（基准验证）**：真 BEM 求解（负控证明非硬编码）+ 经典闭式锚 + 4 点细化误差表 + 一键复现且逐位一致 + 测试绿（18 passed）。不达 L5：无实验/印刷表格锚、半解析验证面、报告收敛阶分析有误。
verdict 已落 `tensor_maturity_verdicts`（repo=TensorBEM, scope='m17176 CASE-BEM-SPHERE-ADDED-MASS', mission_id=17232, level=4, verdict_by=board-audit）；裁判案件行已登记 `tensor_benchmark_cases`。
**处置：approve with notes**（数值成立；F1 收敛阶需更正，README 原本正确无需改）。
"""

fig = (HERE / "fig_audit_sphere_added_mass.png").read_bytes()
body = {"title": "交差审计 m17176 CASE-BEM-SPHERE-ADDED-MASS：L4 确认 + 报告收敛阶算错一倍（README 原本正确）",
        "summary": "取证三件套齐、复算 sub1/sub2/sub3 逐位一致、5120 面板误差 0.113%<3%；发现主件报告观测阶用了 log4（0.95/0.98/0.99）实为 log2≈2，否定其'积分继承通量一阶'的推论；裁判表缺行已补登记；判 L4。",
        "scenario_key": "board_audit", "mission_id": 17232,
        "tags": ["board-audit", "TensorBEM", "CASE-BEM-SPHERE-ADDED-MASS", "added-mass", "L4", "m17176"],
        "markdown": md,
        "figures": [{"name": "fig_audit_sphere_added_mass.png",
                     "caption": "左：附加质量相对误差 vs 面板数（主件/审计复算/O(h²) 参考/3% 线）；右：逐面板通量相对 L2 误差（近 O(h¹)）。",
                     "png_base64": base64.b64encode(fig).decode()}]}
r = api("/tensor/reports", body)
print("REPORT", json.dumps(r)[:300])
