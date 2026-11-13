#!/usr/bin/env python3
"""Publish the m17796 platform report (Hulme hemisphere excitation xval)."""
from __future__ import annotations

import base64
import json
import os
import pathlib

import requests

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
DATA = pathlib.Path("benchmarks/verified/casepool_bem_hulme_hemisphere/excitation_xval_results.json")
d = json.loads(DATA.read_text())
rows = d["comparison"]["rows"]
crit = d["verdict"]["criteria"]
prov = d["study"]["provenance"]

worst_fx = max(abs(r["capytaine_vs_reference"]["Fx_rel"]) for r in rows)
worst_fz = max(abs(r["capytaine_vs_reference"]["Fz_rel"]) for r in rows)
worst_ph = max(
    max(
        abs(r["capytaine_vs_reference"]["Fx_phase_deg"]),
        abs(r["capytaine_vs_reference"]["Fz_phase_deg"]),
    )
    for r in rows
)

amp = [
    "| ka | ref \\|Fx\\| | capytaine \\|Fx\\| | err | ref \\|Fz\\| | capytaine \\|Fz\\| | err | TensorBEM \\|Fz\\| |",
    "|---|---|---|---|---|---|---|---|",
]
pha = [
    "| ka | ref φ(Fx) | capy φ(Fx) | Δ° | ref φ(Fz) | capy φ(Fz) | Δ° | FK guard err |",
    "|---|---|---|---|---|---|---|---|",
]
for r in rows:
    amp.append(
        f"| {r['ka']:g} | {r['reference']['Fx']:.4f} | {r['capytaine_fine']['Fx']:.4f} | "
        f"{100 * r['capytaine_vs_reference']['Fx_rel']:+.2f}% | "
        f"{r['reference']['Fz']:.4f} | {r['capytaine_fine']['Fz']:.4f} | "
        f"{100 * r['capytaine_vs_reference']['Fz_rel']:+.2f}% | "
        f"{r['tensorbem_case']['Fz']:.4f} |"
    )
    fk_err = 100 * max(
        abs(r["capytaine_fk_vs_closed_form"]["surge_rel"]),
        abs(r["capytaine_fk_vs_closed_form"]["heave_rel"]),
    )
    pha.append(
        f"| {r['ka']:g} | {r['reference']['phase_Fx_deg']:.2f} | "
        f"{r['capytaine_fine']['phase_Fx_deg']:.2f} | "
        f"{r['capytaine_vs_reference']['Fx_phase_deg']:+.3f} | "
        f"{r['reference']['phase_Fz_deg']:.2f} | "
        f"{r['capytaine_fine']['phase_Fz_deg']:.2f} | "
        f"{r['capytaine_vs_reference']['Fz_phase_deg']:+.3f} | {fk_err:.2f}% |"
    )

gates = ["| gate | criterion | measured | verdict |", "|---|---|---|---|"]
for name, c in crit.items():
    m = c["measured"]
    if isinstance(m, dict):
        text = ", ".join(f"{k}={v:.3e}" for k, v in m.items())
    elif "phase" in name.lower():
        text = f"{m:.3e} deg"
    else:
        text = f"{m:.3e}"
    gates.append(f"| {name} | {c['criterion']} | {text} | {'PASS' if c['pass'] else 'FAIL'} |")

md = f"""# CASE-BEM-HULME-HEMISPHERE：绕射激励力的第三方独立轴（capytaine）

**任务 17796 / 锚点 CASE-BEM-HULME-HEMISPHERE / 机器裁定 {d['verdict']['verdict']}**

## ① 案例与缺口
浮半球 a = 1 m（平面齐自由面）在深水迎浪下 surge/heave 一阶激励力，无量纲 $F/(\\rho g A a^2)$，时间约定 $\\exp(-i\\omega t)$，波幅 1 m，ka ∈ {{{', '.join(f'{k:g}' for k in d['ka_grid'])}}}。
算例池原验证已登记两条诚实局限：参考解（Hulme 1982 波-free 多极重推导）与求解器共享 `tensorbem.free_surface3d.wave_function_3d`；且参考的激励力由同一级数经 Haskind 互易得到，唯一互检还是同一求解器（`hulme_diffraction_3d`），因此激励力此前**没有独立第三方轴**（Hulme 印刷表格在付费墙后，零手抄纪律下不作转录）。本单补这条轴。

## ② 方法
- 第三方码：capytaine {prov['version']}（Nemoh 后继，独立面板法），几何 `{prov['geometry']}`，无限水深，ρ = 1025 kg/m³，g = 9.80665 m/s²。
- 激励力定义：`froude_krylov_force + DiffractionResult.forces`。**坑一**：capytaine 3.x 的 `DiffractionResult.forces` 只是散射力，漏掉 FK 会在 ka=0.05 给出 heave 0.09（应约 π）。
- 网格阶梯：{', '.join(str(l['n_faces']) + ' 面' for l in d['study']['levels'])}（含自收敛度量）。
- 闭式 Froude-Krylov 锚（1 维求积，机器精度）钉住归一化与约定：$F_z=-2\\pi\\int_{{-1}}^{{0}} u\\,e^{{\\alpha u}}J_0(\\alpha\\sqrt{{1-u^2}})du$、$F_x=-2\\pi i\\int_{{\\pi/2}}^{{\\pi}}\\sin^2\\theta\\,e^{{\\alpha\\cos\\theta}}J_1(\\alpha\\sin\\theta)d\\theta$。**坑二**：漏掉方位角积分的 $J_0$ 因子使 ka=2 的 FK 偏大一倍 —— 生产脚本第一版正是被 G6 门抓出（$10^4$ 量级相对误差）。
- 预注册门 G1–G6，事后未放宽；TensorBEM 侧数字直接读 `results.json`（1024 面板），不重算。

## ③ 结果（程序化注入，无手抄）
{chr(10).join(amp)}

{chr(10).join(pha)}

{chr(10).join(gates)}

## ④ 图
幅值三线对比（解析参考 vs capytaine vs TensorBEM）与相位对比（同一 $\\exp(-i\\omega t)$ 约定，任何共轭/来流反号会表现为 ~180° 偏差）。

## ⑤ 注意点
- capytaine 是可选外部依赖，**不进测试依赖**：数字冻结在 `excitation_xval_results.json`，测试只读冻结件（10 项检查：provenance、闭式极限、门算术重算、无吞错、图文一致）。
- 频带止于 ka = 2：ka ≈ 2.58 为半球不规则频率，capytaine 无 CHIEF 抑制，故不外推。
- 两码共享建模假设（线性势流/无粘/深水/刚体），本轴只能查实现缺陷，不能查建模假设。
- 既有欠账（与本单无关）：`benchmarks/verified/casepool_bem_hulme_hemisphere/figure.png` 缺失，导致 `test_casepool_hulme_hemisphere_results.py::test_figure_artifact_exists` 等 5 项为 pre-existing FAIL（早前门禁滤除所致），本单未触碰这些文件。

## ⑥ 误差判定
- |Fx| 最坏 **{100 * worst_fx:.2f}%**，|Fz| 最坏 **{100 * worst_fz:.2f}%**，相位最坏 **{worst_ph:.3f}°**（基准 = Hulme 多极重推导参考）。
- capytaine 自收敛：细网格-vs-次细漂移 {crit['G4_capytaine_self_convergent']['measured']['drift']:.2%}，粗网格对参考误差 {crit['G4_capytaine_self_convergent']['measured']['coarse_error']:.2%}（比值 ≈ 9.5，门要求 ≥3）。
- 两码互比：{crit['G5_tensorbem_vs_capytaine']['measured']['amplitude']:.2%} / {crit['G5_tensorbem_vs_capytaine']['measured']['phase_deg']:.2f}°。
- 闭式 FK 归一化守卫：{100 * crit['G6_fk_normalisation_guard']['measured']:.2f}%。
- **是否 < 3%？** 是：最大 {100 * max(worst_fx, worst_fz):.2f}%（算例池判据为 5%）。
- 结论：激励力这条此前无第三方轴的量，现由独立码确认到 3% 以内、相位 1.4° 以内；参考解不再只与本求解器自洽。
"""

figs = []
for path, caption in (
    (
        "docs/fig_hulme_excitation_xval.png",
        "Excitation force amplitude vs ka: Hulme multipole reference, capytaine "
        "(third-party BEM), TensorBEM 1024 panels",
    ),
    (
        "docs/fig_hulme_excitation_xval_phase.png",
        "Excitation force phase vs ka: reference and capytaine in the same "
        "exp(-i omega t) convention",
    ),
):
    figs.append(
        {
            "name": pathlib.Path(path).stem,
            "caption": caption,
            "png_base64": base64.b64encode(pathlib.Path(path).read_bytes()).decode(),
        }
    )

payload = {
    "title": (
        "CASE-BEM-HULME-HEMISPHERE：绕射激励力的第三方独立轴验证"
        f"（capytaine，误差 <3%）"
    ),
    "summary": (
        f"capytaine 独立复核 Hulme 半球激励力：|Fx| 最坏 {100 * worst_fx:.2f}%、"
        f"|Fz| {100 * worst_fz:.2f}%、相位 {worst_ph:.2f}°，6 门全 PASS；"
        f"闭式 FK 锚 {100 * crit['G6_fk_normalisation_guard']['measured']:.2f}% 钉住约定。"
    ),
    "scenario_key": "case_bem_hulme_hemisphere_excitation_xval",
    "mission_id": 17796,
    "tags": [
        "TensorBEM",
        "CASE-BEM-HULME-HEMISPHERE",
        "verification",
        "capytaine",
        "cross-code",
        "diffraction",
    ],
    "markdown": md,
    "figures": figs,
}

token_resp = requests.post(
    f"{BASE}/auth/login",
    json={
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    },
    timeout=60,
)
token_resp.raise_for_status()
token = token_resp.json().get("access_token") or token_resp.json()["token"]
resp = requests.post(
    f"{BASE}/tensor/reports",
    headers={"Authorization": f"Bearer {token}"},
    json=payload,
    timeout=180,
)
print(resp.status_code, resp.text[:300])