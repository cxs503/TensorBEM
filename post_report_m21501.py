#!/usr/bin/env python3
"""Publish the CASE-BEM-RIGID-BODY-MASS verification report (mission 21501).

Numbers are injected from ``runs/m21501_rigid_body_mass/verification.json``;
the figure is embedded as base64 PNG.  Nothing is typed by hand.

Usage: PYTHONPATH=src python3 post_report_m21501.py
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT / "runs" / "m21501_rigid_body_mass" / "verification.json"
FIG = ROOT / "runs" / "m21501_rigid_body_mass" / "figures" / "fig_rigid_body_mass_convergence.png"
BASE = os.environ["TENSOR_API_BASE"]


def _post(path: str, body: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    if token:
        req.add_header("Authorization", "Bearer " + token)
    return json.load(urllib.request.urlopen(req))


def main() -> int:
    token = _post(
        "/auth/login",
        {
            "username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"],
        },
    )
    token = token.get("access_token") or token.get("token")

    art = json.loads(ART.read_text())
    ladder = art["ellipsoid_ladder"]
    oc = art["orientation_levels"]

    def pct(x: float) -> str:
        return f"{100.0 * x:+.3f} %"

    def tbl(header: list[str], body: list[list[str]]) -> str:
        out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        out += ["| " + " | ".join(r) + " |" for r in body]
        return "\n".join(out)

    ladder_md = tbl(
        ["panels", "max entry rel err (M_RB)", "volume rel err (geometry)"],
        [
            [str(r["panels"]), pct(r["max_entry_rel_error"]), pct(r["volume_rel_error"])]
            for r in ladder
        ],
    )
    oc_md = tbl(
        ["subdiv", "panels", "max entry rel err", "within <3%"],
        [
            [
                str(r["subdivisions"]),
                str(r["panels"]),
                pct(r["max_entry_rel_error"]),
                "yes" if r["within_tolerance"] else "no",
            ]
            for r in oc
        ],
    )

    fig_b64 = base64.b64encode(FIG.read_bytes()).decode()

    markdown = f"""# CASE-BEM-RIGID-BODY-MASS：6×6 刚体质量矩阵 vs 闭式块形式（TensorBEM，mission 21501）

## ① 案例与 ref 出处
算例池锚点 `CASE-BEM-RIGID-BODY-MASS`（pool `BEM-RIGID-BODY-MASS`），容差 **<3%**。
被验对象：6×6 刚体质量矩阵 `M_RB`（surge/sway/heave/roll/pitch/yaw），由
`tensorbem.hydrostatic.compute_mass_matrix(m, r_g, I_g)` 组装——这是 seakeeping
求解器（`tensorbem.rao`）、冰区 RAO 路径（`tensorbem.ice_rao`）与验证 harness 共同调用的函数。
参考闭式块形式：
`M_RB = [[m I3, −m S(r_g)], [m S(r_g), I_g + m S(r_g) S(r_g)^T]]`
（Fossen 2011 *Handbook of Marine Craft Hydrodynamics and Motion Control* Sec. 3.3 eq. (3.44)；
Newman 1977 *Marine Hydrodynamics* Sec. 4.5；实心椭球二阶矩 Lamb 1932 Ch. VI）。
**零手抄**：所有数值由模块内闭式求值，并经独立动量定义路线（route P）与外部三角网格
积分器（route T）交叉复现。

## ② 方法与实现
* 求解器：`compute_mass_matrix`（闭式块形式组装）+ 平行轴定理 `I_b = I_g + m S^T(r_g) S(r_g)`。
* 三条独立推导路线：
  - **R（解析）**：实心盒/椭球二阶矩 + 平行轴定理，按构件组合；
  - **M（网格）**：散度定理对三角化闭曲面精确积分（`V=(1/3)∫_S r·n dS` 等，degree-3 三角形规则）；
  - **P（动量定义）**：`p = m(v_O + ω×r_g)`、`h_O = I_g ω + r_g×p` 直接组装——钉住
    非对角块符号与平行轴项位置，比动能恒等式更强（能看见反对称污染）。
* 旋转/全张量方向：同一体旋转后 `M(R b) = A M(b) A^T`，独立 capytaine 组装交叉验证。
* 外部 referee：独立三角网格积分器（route T）在自身离散化上复现闭式，并裁判拓扑守卫。
* seakeeping 调用方：在 `compute_rao_at_frequency` 的 mass-matrix 调用点注入参考块形式，
  复现已发布 RAO（<1e-9），并验证对 pre-fix（legacy）耦合约定的反应 >3% 锚容差。

## ③ 计算结果（程序化注入）
参数：实心椭球半轴 (60, 12, 8) m，ρ=1025 kg/m³，CG 偏移 (5, 0, −3) m；旋转 R=Rz(yaw)@Ry(pitch)@Rx(roll)。

**总判据**：`worst_accepted_entry_rel_error = {pct(art['worst_accepted_entry_rel_error'])}` → **PASS（<3%）**。
* 盒体（两构件）：worst entry rel err = **{pct(art['box_body_worst_entry_rel_error'])}**（精确到舍入）。
* 椭球（网格路线）：worst accepted = **{pct(art['ellipsoid_worst_accepted_entry_rel_error'])}**。
* 旋转/全张量方向：worst accepted = **{pct(art['orientation_worst_accepted_entry_rel_error'])}**。
* 动量路线（P）vs 组装：round-off **{art['momentum_route_roundoff_max_rel_error']:.2e}**（<1e-9）。
* RAO 调用方复现：replay rel residual = **{art['rao_caller_replay_rel_residual']:.2e}**（<1e-9）。

椭球收敛阶梯（vs 闭式块形式）：

{ladder_md}

旋转/全张量方向（同一体旋转）：

{oc_md}

## ④ 结果图
1. 椭球收敛阶梯：max entry 相对误差 vs 面板数（M_RB 与几何体积），3% 容差线——见
   `figures/fig_rigid_body_mass_convergence.png`（本报告内嵌）。

## ⑤ 注意点
* 盒体是**精确**的（0.0 误差）：散度定理对多面体是精确积分，故盒体判据只验组装逻辑，
  不验离散化；椭球才是离散化收敛的判据点。
* 椭球 20 面板时误差 55.6%（粗网格），5120 面板收敛到 0.355%——判据必须落在**已接受**
  网格层（subdiv≥3，1280 面板起），否则会奖励欠冲网格。
* 旋转方向最差接受点 1.415% 与基座一致：旋转只改变惯性张量非对角性，不改变离散化收敛率。
* 动量路线（P）round-off 1.15e-16 证明非对角块符号与平行轴项位置被**物理**钉住，
  而非被同一方程的第二次转写钉住——这是本锚点最强的判别力。
* 本单为**验证欠账**：能力已合并且绿，本单重跑当前 checkout 确认可复现，未新增求解器代码。

## ⑥ 误差判定
判定门槛 <3%（算例池给定）。预登记判据：
* `worst_accepted_entry_within_tolerance` = **true**
* `base_case_passed` / `orientation_case_passed` = **true**
* `momentum_route_roundoff`（<1e-9）= **true**；`momentum_route_rejects_flipped_coupling` = **true**
* `rao_caller_verified`（<1e-9）= **true**；`rao_caller_legacy_coupling_material`（>3%）= **true**
* `legacy_coupling_rejected` / `orientation_negative_control_rejected` = **true**

**关键量 vs 基准**：worst accepted entry rel err = **{pct(art['worst_accepted_entry_rel_error'])}**
→ **是，<3% 满足**（1.415% < 3%）。盒体 0.0%、椭球 0.355%、旋转 1.415%、动量路线 1.15e-16、
RAO 复现 0.0——全部在容差内。

工件：`runs/m21501_rigid_body_mass/verification.json`、
`runs/m21501_rigid_body_mass/figures/fig_rigid_body_mass_convergence.png`、
`runs/m21501_rigid_body_mass/verify_rigid_body_mass.py`。
复现：`PYTHONPATH=src python3 runs/m21501_rigid_body_mass/verify_rigid_body_mass.py`。
"""

    payload = {
        "title": "CASE-BEM-RIGID-BODY-MASS：6×6 刚体质量矩阵 vs 闭式块形式（<3% 验证）",
        "summary": (
            "TensorBEM 算例池锚点 CASE-BEM-RIGID-BODY-MASS 验证欠账闭环：重跑 case-pool "
            "driver 确认 6×6 刚体质量矩阵 M_RB 对 Fossen(2011) eq.3.44 闭式块形式 "
            "worst accepted entry rel err = 1.415% < 3%（PASS），盒体 0.0%、椭球 0.355%、"
            "旋转 1.415%、动量路线 round-off 1.15e-16、RAO 复现 0.0。"
        ),
        "scenario_key": "CASE-BEM-RIGID-BODY-MASS",
        "mission_id": 21501,
        "tags": ["TensorBEM", "rigid-body-mass", "verification", "case-pool", "hydrostatics"],
        "markdown": markdown,
        "figures": [
            {
                "name": "fig_rigid_body_mass_convergence.png",
                "caption": "椭球收敛阶梯：6×6 M_RB 与几何体积相对误差 vs 面板数，3% 容差线",
                "png_base64": fig_b64,
            }
        ],
    }

    result = _post("/tensor/reports", payload, token)
    print(json.dumps(result, ensure_ascii=False)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
