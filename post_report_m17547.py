#!/usr/bin/env python3
"""Publish the m17547 progress report to the platform (:8010 tensor/reports)."""
from __future__ import annotations

import base64
import json
import os
import pathlib
import subprocess
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
FIG = ROOT / "benchmarks" / "verified" / "bem_rigid_lid_hemisphere_axisym" / "figure.png"
PAYLOAD = json.loads(
    (ROOT / "benchmarks" / "verified" / "bem_rigid_lid_hemisphere_axisym" / "results.json").read_text()
)

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def _post(url: str, body: dict, token: str | None = None) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


token = _post(f"{BASE}/auth/login", {"username": USER, "password": PASS})
token = token.get("access_token") or token.get("token") or token.get("data", {}).get("access_token")
assert token, token

pin = PAYLOAD["pin"]
best = PAYLOAD["mfs_best"]
audit = {r["key"]: r for r in PAYLOAD["reference_audit"]["rows"]}
gauss = audit["gauss_quadrature_series_m16802_m17243"]
head = PAYLOAD["anchor_headline"]

markdown = f"""# CASE-BEM-RIGID-LID-HEMISPHERE：参考值钉扎到 1e-11 + 独立解法复核（m17547）

## ① 案例介绍
浮半球（a=1 m）升沉辐射的 **ka→0（刚性盖）端点**。镜像陷阱：升沉速度垂直于对称面，镜像体速度反号，
双体为**偶宇称滑动双体**（`∂φ/∂r = -U|cosθ|`），不是整球平动，故端点值不是 ρ∇ 而是 ≈0.831ρ∇
（旧锚 1.0 已被 m16850 裁决为参考错误）。ρ=1025 kg/m³，∇=(2/3)πa³=2.094395 m³，A33 以 (ρ∇) 无量纲化。

## ② 怎么做的
1. **参考值钉扎（无求积）**：改用闭式 Legendre 系数
   `a_n=(2n+1)∫₀¹μP_n dμ = (n+1)J_(n+1)+nJ_(n-1)`，`J_m=[P_(m-1)(0)-P_(m+1)(0)]/(2m+1)`，
   `P_2j(0)=(-1)^j C(2j,j)/4^j`，级数 `A33/(ρ∇)=3Σ a_n²/((n+1)(2n+1))`；`fractions.Fraction`
   精确有理部分和 + 实测尾项指数 p=3.965 → Richardson。**不依赖任何 Gauss 求积**。
2. **独立解法（新方法族）**：轴对称 MFS 环源 `φ=Σq_k[G_ring(r,z;r_k,z_k)+G_ring(r,z;r_k,-z_k)]`，
   `G_ring=K(κ)/(π√(a+b))`（精确椭圆积分），镜像环精确满足 ∂φ/∂z=0，源在体内 → 无奇异积分；
   Neumann 最小二乘拟合；A33=−ρ∫φn_z dS 用与配置网格无关的 4001 点均匀求积。
   路线自检＝无限流体平动整球（精确值 0.5）。
3. 重算 m17243 已交付算例对**修正后参考**的误差，确认判据不变。

## ③ 计算结果
| 量 | 值 |
|---|---|
| **钉扎参考 A33(0)/(ρ∇)** | **{pin['value']:.12f}**（尾项 {pin['truncation_tail_estimate']:.2e}，p={pin['measured_tail_exponent']:.3f}） |
| 旧 Gauss 求积参考（已提交工件） | {gauss['value']:.10f} → 偏置 **{gauss['deviation_ppm']:+.3f} ppm** |
| m16850 ERRATUM 散文值 | {audit['erratum_prose_m16850']['value']:.7f} → {audit['erratum_prose_m16850']['deviation_ppm']:+.3f} ppm |
| 旧 docstring 值 | {audit['stale_module_docstring']['value']:.6f} → {audit['stale_module_docstring']['deviation_ppm']:+.1f} ppm |
| **MFS 独立路线最佳** | {best['added_mass_dimless']:.7f}（相对钉扎值 **{100*best['rel_error_vs_pin']:+.4f} %**，{best['n_sources']} 源/cond {best['condition_number']:.1e}） |
| MFS 路线自检（平动整球 vs 精确 0.5） | {PAYLOAD['sphere_validation']['added_mass_dimless']:.8f}（{PAYLOAD['sphere_validation']['rel_error']:+.1e}） |
| 3-D Kelvin BEM 1024 面板（m17243 headline） | {head['added_mass_dimless']:.6f} → 对钉扎值 **{100*head['rel_error_vs_pin']:+.4f} %** |

MFS 9 组设置（收缩比 0.6/0.8 × 16~128 源）取值区间 [0.83059, 0.83162]，即 ±0.05 % 内；
与面元法（+0.2831 %）从上下两侧夹住同一参考。

## ④ 图
三联图：A MFS 阶梯 vs 钉扎值/±0.1 % 带 + 256 面板面元值；B 精确有理部分和对钉扎值的偏差 vs 截断阶数
（log-log，叠 O(n⁻⁴) 斜率与 +3.4 ppm 的 Gauss 偏置水平线）；C 流通中各数值偏差（ppm）条形图。

## ⑤ 注意点
- "尾项收敛"≠"值正确"：旧值的系数由**折点函数 |μ| 的 Gauss 求积**得到，带 +3.4 ppm 系统偏置；
  既有测试只比较不同截断阶之间的差，结构上发现不了该偏置（本单新增测试显式断言该间隙必须被复现）。
- MFS 精度上限是条件数（~1e11）而非离散化，故不同设置 ±0.05 % 非单调抖动，按"最佳值+区间"报告，不宣称阶数。
- 接触线角点奇性是所有方法共同上限（m17243 的 256→1024 表观阶 0.13 即源于此）。
- 未覆盖 m16802/m17243 已提交工件（honest-FAIL/历史记录纪律）；偏差记入 ERRATUM addendum。
- 既有红灯（先于本单存在）：`test_rigid_lid_hemisphere_attribution::test_all_criteria_pass_and_summary_is_rendered`
  断言 `bem_rigid_lid_hemisphere/figure.png` 存在，但该图不在仓内；本单未伪补。

## ⑥ 误差判定（显式回答「是否<3%」）
- 3-D BEM headline vs 钉扎参考：**+0.2831 % → 是（<3 %）**
- 独立 MFS 最佳 vs 钉扎参考：**−0.0036 % → 是**
- 钉扎参考自身：尾项 1.9e-11（相对 2.3e-11）→ 是
- 旧参考 vs 钉扎参考：+3.4 ppm → 数值上 <3 %，但属**系统性偏置**（已量化入账，非随机误差）
- 锚点判据（<5 %）：**PASS 不变**；八条判据 P1–P8 全 PASS，复现 1.3 s。

工件：`benchmarks/verified/bem_rigid_lid_hemisphere_axisym/{{results.json,summary.md,figure.png,run_study.py,make_figure.py}}`；
代码 `src/tensorbem/benchmarks/rigid_lid_reference_pin.py`；测试 `tests/test_rigid_lid_reference_pin.py`（12 passed）；
报告 `docs/dev_report_m17547_rigid_lid_reference_pin.md`。
"""

png = base64.b64encode(FIG.read_bytes()).decode()
body = {
    "title": "CASE-BEM-RIGID-LID-HEMISPHERE：参考值钉扎 1e-11 + 独立 MFS 路线复核（m17547）",
    "summary": (
        f"刚性盖半球 ka→0 附加质量：参考值以闭式 Legendre 系数+Richardson 钉扎到 "
        f"{pin['value']:.9f}（旧值偏置 {gauss['deviation_ppm']:+.2f} ppm）；新增轴对称 MFS 环源独立路线"
        f"复现至 {100*abs(best['rel_error_vs_pin']):.4f} %；m17243 headline 误差由 +0.2828 % 更新为 "
        f"+{100*head['rel_error_vs_pin']:.4f} %，判据 PASS 不变；八条判据全 PASS。"
    ),
    "scenario_key": "bem-rigid-lid-hemisphere",
    "mission_id": 17547,
    "tags": ["TensorBEM", "rigid-lid", "added-mass", "verification", "MFS", "case-pool"],
    "markdown": markdown,
    "figures": [
        {
            "name": "fig_m17547_rigid_lid_pin.png",
            "caption": (
                "A: MFS 环源阶梯 vs 钉扎参考（±0.1 % 带）与 256 面板面元值；"
                "B: 精确有理部分和偏差 vs 截断阶数（O(n^-4)）与 +3.4 ppm Gauss 偏置线；"
                "C: 流通中各数值对钉扎值的偏差（ppm）。"
            ),
            "png_base64": png,
        }
    ],
}
out = _post(f"{BASE}/tensor/reports", body, token)
print(json.dumps(out, ensure_ascii=False)[:600])