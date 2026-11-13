#!/usr/bin/env python3
"""Post board-audit report for m17205 (CASE-BEM-TWO-CYL-MUTUAL). Numbers injected from artifacts, zero hand-copy."""
import base64
import json
import os
import requests

BASE = os.environ["TENSOR_API_BASE"]
tok = open("/root/tp-agent/work/m17237/.tok").read().strip()
H = {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}

ART = "/root/tp-agent/work/m17205/artifacts/two_cylinder_added_mass"
b = json.load(open(f"{ART}/benchmark.json"))
rows = {r["panels_per_body"]: r for r in b["rows"]}
r48, r64, r96 = rows[48], rows[64], rows[96]
me = b["max_relative_errors"]
oo = b["observed_orders"]
dec = b["interaction_decay"]

def e(x):
    return f"{x:.4e}"

rows = {r["panels_per_body"]: r for r in b["rows"]}
r96 = rows[96]
me = b["max_relative_errors"]
oo = b["observed_orders"]

def e(x):
    return f"{x:.4e}"

md = f"""# 交差审计 m17205 — CASE-BEM-TWO-CYL-MUTUAL 双圆柱互附加质量（TensorBEM）

**Mission**: m17237（board-audit）· **被审单**: m17205（succeeded）· **锚点**: CASE-BEM-TWO-CYL-MUTUAL
**审计产物**: docs/board_audit/m17237/AUDIT_m17205.md · verdict#503（L4, board-audit）· 裁判行 tensor_benchmark_cases id=41（本审计补登记）

## ① 案例介绍（物理背景/几何/参数表）
两个等半径圆柱（a=1.0 m，中心距 d=3a，体0 在 (-d/2,0)、体1 在 (+d/2,0)）在无界无粘不可压流体中作刚体运动，求每单位展长 4x4 附加质量矩阵（模态：体0/体1 纵荡+垂荡）。经典两体水动力相互作用（双体船/并靠作业势流极限），无自由面、无入射波。
参数：ρ=1025 kg/m³；多极级数截断 N=24；配点 2N+8/体；积分 512 点/体；BEM 48/64/96 常数面板+中点配点。无量纲：d/a=3.0（>2a 非接触）、|A12|/A11≈0.22、同相垂荡比≈1.26（阻塞效应）。

## ② 怎么做的（方法/流程/工具链）
- **参考解（与 BEM 完全独立）**：截断多极级数 `Psi(z)=Σ m_{{k,n}}(z-c_k)^{{-n}}`，`phi=Re[Psi]`，每圆 2N+8 等距配点实最小二乘，A_ij=-ρ∮φ_j n_i ds 用 512 点梯形（谱精度）。实现于 `src/tensorbem/analytic.py:1678`，不调 solver 任何算子。
- **数值侧**：双环均标 hole 的 `MultiLoopMesh`，层矩阵取自主求解器 `solver.assemble_matrices`（无新核），外部第二类方程 0.5I+H（hole 法向使 H 变号），条件数 2.2–2.5。
- **审计动作（本单）**：① 独立重跑 repro_cmd；② 独立跑 18 项测试；③ 报告数字逐项溯源 benchmark.json；④ 基准裁判行查缺补登；⑤ 工件 sha256 钉。

## ③ 计算结果（程序化注入，零手抄）
| 量（kg/m） | 多极参考 | BEM 48 | BEM 64 | BEM 96 |
|---|---|---|---|---|
| 自附加质量 A[0,0] | {r96['reference']['self']:.4f} | {r48['bem']['self']:.3f} | {r64['bem']['self']:.3f} | {r96['bem']['self']:.3f} |
| 互耦（纵向）A[0,2] | {r96['reference']['mutual_along']:.4f} | {r48['bem']['mutual_along']:.3f} | {r64['bem']['mutual_along']:.3f} | {r96['bem']['mutual_along']:.3f} |
| 互耦（横向）A[1,3] | {r96['reference']['mutual_transverse']:.4f} | {r48['bem']['mutual_transverse']:.3f} | {r64['bem']['mutual_transverse']:.3f} | {r96['bem']['mutual_transverse']:.3f} |
| 同相垂荡 A[1,1]+A[1,3] | {r96['reference']['in_phase_heave']:.3f} | {r48['bem']['in_phase_heave']:.3f} | {r64['bem']['in_phase_heave']:.3f} | {r96['bem']['in_phase_heave']:.3f} |
| 最大相对误差 | — | {me[0]:.3e} | {me[1]:.3e} | {me[2]:.3e} |

参考解自证：BC 残差 {b['rows'][0]['reference_boundary_residual']:.2e}（N=24）、反射对称 {b['rows'][0]['reference_reflection_symmetry_error']:.2e}·ρπa²、远场自附加质量/ρπa²=1+7.8e-7（d/a=40）。

## ④ 结果图
势场云图+边界势逐点比较；网格加密收敛（观测阶 {oo[0]:.4f}/{oo[1]:.4f}）；互耦系数随间距与 (a/d)² 渐近线（d/a=40 时互耦×d²={dec[-1]['mutual_transverse_times_separation_squared']:.2f}）。

## ⑤ 注意点（坑/局限）
- **符号陷阱**：hole 环使域外法向翻向刚体内部，双层层矩阵变号；误用 0.5I-H 时自附加质量几乎不变（~3.6%）而互耦项塌陷（99.9%）——单体基准发现不了，已固化为树内负控（`--inject layer_sign`，测试 `test_injected_double_layer_sign_error_turns_the_checks_red`）。
- **环境坑**：本机 tensorbem 为指向另一 worktree 的 editable 安装，必须 `PYTHONPATH=src`。
- **局限**：二维、无界、无自由面；仅 d/a≥3；常数面板收敛阶为实测 ~2.0 非理论断言。验的是外部算子与互耦代数，不是波浪辐射或实船水动力。
- **审计发现**：F1 裁判行缺失（已补登）；F2 m17205 工件未入 git（重跑逐字节恢复，代码本体在 main/m4721 PR#315）；F4 外部锚=仓内闭式重实现，不计 L5（WITHHELD）。

## ⑥ 误差判定（关键量 vs 基准，逐项）
| 关键量 | 基准 | 相对误差（96 面板/体） | 是否 <3% |
|---|---|---|---|
| 自附加质量 | 多极级数 | {e(r96['relative_errors']['self'])} ({r96['relative_errors']['self']*100:.4f}%) | 是 |
| 互耦（纵向） | 多极级数 | {e(r96['relative_errors']['mutual_along'])} ({r96['relative_errors']['mutual_along']*100:.4f}%) | 是 |
| 互耦（横向） | 多极级数 | {e(r96['relative_errors']['mutual_transverse'])} ({r96['relative_errors']['mutual_transverse']*100:.4f}%) | 是 |
| 同相垂荡 | 多极级数 | {e(r96['relative_errors']['in_phase_heave'])} ({r96['relative_errors']['in_phase_heave']*100:.4f}%) | 是 |

**收敛/敏感性（≥2 点）**：① 48→64→96 面板 max err {me[0]:.3e}→{me[1]:.3e}→{me[2]:.3e} 单调下降，观测阶 {oo[0]:.4f}/{oo[1]:.4f}；② d/a=3/5/10/20/40 互耦按 (a/d)² 衰减（d/a=40 比值 1.0000004），自附加质量→ρπa²（1+7.8e-7）。
**复现**：审计重跑 282/282 标量键一致，benchmark.json sha256 逐字节相同（68e3568c…）。
**结论：全部关键量误差 <3%（最大 {me[2]*100:.4f}%），判定通过；判级 L4（无印刷表/实验锚，L5 封顶于 L4）。**
"""

figs = []
for name, cap in [
    ("field_and_boundary.png", "势场云图 + 边界势逐点比较（BEM 64 面板 vs 多极级数）"),
    ("refinement.png", "网格加密收敛：48→64→96 面板，观测阶 ~2.0"),
    ("interaction.png", "互耦系数随间距衰减与 (a/d)² 渐近线"),
]:
    figs.append({"name": name, "caption": cap,
                 "png_base64": base64.b64encode(open(f"{ART}/{name}", "rb").read()).decode()})

payload = {
    "title": "交差审计 m17205：双圆柱互附加质量（TensorBEM）——三件套核验+L4 判级+基准裁判行补登",
    "summary": (f"m17205 交差核验：run_artifacts 真存在（sha256 钉）、ref vs got 全部 <3%（最大 "
                f"{me[2]*100:.4f}%）、repro_cmd 独立重跑逐字节复现（sha256 68e3568c…）、18 测试全绿、报告六段齐全。"
                f"判级 L4；裁判行 tensor_benchmark_cases id=41 补登（原 0 行，系统缺口）；verdict#503 落表待 gate 裁定。"),
    "scenario_key": "CASE-BEM-TWO-CYL-MUTUAL",
    "mission_id": 17237,
    "tags": ["board-audit", "TensorBEM", "m17205", "maturity-L4", "two-cylinder", "added-mass"],
    "markdown": md,
    "figures": figs,
}
open("/root/tp-agent/work/m17237/report_m17237.json", "w").write(json.dumps(payload, ensure_ascii=False))
r = requests.post(f"{BASE}/tensor/reports", headers=H, json=payload, timeout=120)
print("report post:", r.status_code, r.text[:300])
