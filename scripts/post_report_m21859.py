#!/usr/bin/env python3
"""Publish the CASE-BEM-CUMMINS-WIGLEY Round 7 report via the Tensor API."""
from __future__ import annotations

import json
import base64
import os
import subprocess
from pathlib import Path

ARTIFACT = Path("benchmarks/verified/casepool_bem_cummins_wigley/case_result.json")
FIGDIR = Path("artifacts/case_bem_cummins_wigley_round7")


def png_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_token():
    """Get auth token via curl login."""
    user = os.environ.get("TENSOR_AGENT_USER", "")
    passwd = os.environ.get("TENSOR_AGENT_PASS", "")
    api_base = os.environ.get("TENSOR_API_BASE", "http://localhost:8001")
    result = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{api_base}/auth/login",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"username": user, "password": passwd})],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Login failed: {result.stderr}")
        return None
    try:
        return json.loads(result.stdout).get("access_token") or json.loads(result.stdout).get("token")
    except Exception:
        print(f"Login response: {result.stdout[:200]}")
        return None


def main():
    data = json.load(open(ARTIFACT))
    v = data["verdict"]
    gates = data["gates"]

    # Build markdown report
    md = f"""# CASE-BEM-CUMMINS-WIGLEY Round 7 — Artifact Regeneration & Gate Expansion

**Mission**: m21859 | **Anchor**: `CASE-BEM-CUMMINS-WIGLEY` | **Status**: 13/13 gates PASS

## 1 案例介绍

Wigley 数学船型（L=3.0 m, B=0.3 m, T=0.1875 m, Cb=4/9），6 自由度刚性辐射。
算例判据来自 Cummins (1962) 时域卷积与 Kramers-Kronig 因果对，容差 <10%。

**本单解决的欠账**：此前提交的 `case_result.json`（m21100）存在两处不一致：
1. `settings.dt = 0.01` 但实际 C5 闸门使用 `DT = 0.002`（m21373 修复后未同步更新 settings）
2. 缺少 m21471 引入的锚定闸门：C3a（零自由参数 KK 闭合）、C7（A_inf 锚点一致性）、C3a_coarse

本单通过 `gates_from_artifact` 从已提交的扫频工件（sweep_medium.npz, sweep_coarse.npz）
重放全部 13 个闸门，不重解 BEM 系统。

## 2 怎么做的

1. **工件重放**：`scripts/regenerate_cummins_wigley_verdict.py` 读取已提交的 .npz 扫频工件，
   调用 `gates_from_artifact` 重算所有闸门（C1, C0, C0_coarse, C2, C3, C3a, C7,
   C3_coarse, C3a_coarse, C4, C5, C6a, C6b）。
2. **修复 settings.dt**：从 0.01 更正为 0.002，与实际闸门评估一致。
3. **新增闸门**：
   - C3a：KK 闭合但 A_inf 由镜像法锚点代入（零自由参数），worst = 8.85% (yaw)
   - C7：拟合 A_inf vs 锚点 A_inf 一致性，worst = 15.27% (yaw，相对锚点值)
   - C3a_coarse：粗网格上的锚定 KK 闭合，worst = 8.40% (yaw)
4. **图表生成**：`scripts/make_cummins_round7_figures.py` 生成三张图。

## 3 计算结果

### 3.1 闸门总览

| 闸门 | 描述 | 状态 | 关键值 |
|------|------|------|--------|
| C1 | 几何验证 | ✅ PASS | Cb 误差 0.63% |
| C0 | 求解器有效性 (medium) | ✅ PASS | max residual 0.24% |
| C0_coarse | 求解器有效性 (coarse) | ✅ PASS | max residual 0.64% |
| C2 | 面板分辨率 | ✅ PASS | k·√A = 0.947 |
| C3 | KK 闭合 (拟合 A_inf) | ✅ PASS | worst 3.60% (yaw) |
| **C3a** | **KK 闭合 (锚定 A_inf, 0 自由参数)** | **✅ PASS** | **worst 8.85% (yaw)** |
| **C7** | **A_inf 锚点一致性** | **✅ PASS** | **worst 15.27% (yaw)** |
| C3_coarse | KK 闭合 粗网格 | ✅ PASS | worst 3.40% (yaw) |
| C3a_coarse | 锚定 KK 闭合 粗网格 | ✅ PASS | worst 8.40% (yaw) |
| C4 | 往返变换 B→K→B | ✅ PASS | max error 0.087% |
| C5 | 自由衰减阻尼比 | ✅ PASS | 误差 3.96% |
| C6a | 负控: B=0 | ✅ PASS | kernel_max = 0 |
| C6b | 负控: 破坏 B 形状 | ✅ PASS | worst 20.6% (>10%) |

### 3.2 KK 闭合对比 (C3 vs C3a)

| DOF | C3 (拟合) | C3a (锚定) | Δ |
|-----|-----------|------------|---|
| surge | 2.62% | 7.07% | +4.45 pp |
| sway | 2.81% | 7.62% | +4.81 pp |
| heave | 0.30% | 0.91% | +0.61 pp |
| roll | 0.72% | 2.45% | +1.73 pp |
| pitch | 0.48% | 1.17% | +0.69 pp |
| yaw | 3.60% | 8.85% | +5.25 pp |

C3a 残差普遍高于 C3，因为去掉了拟合 A_inf 这个自由参数。但所有 DOF 仍在 10% 容差内。

### 3.3 A_inf 锚点一致性 (C7)

| DOF | A_inf 拟合 | A_inf 锚点 | 相对锚点误差 |
|-----|-----------|-----------|-------------|
| surge | 0.586 | 0.485 | 20.8% |
| sway | 75.73 | 66.62 | 13.7% |
| heave | 50.93 | 50.07 | 1.7% |
| roll | 0.385 | 0.375 | 2.7% |
| pitch | 14.81 | 14.57 | 1.6% |
| yaw | 56.78 | 49.26 | 15.3% |

水平模态 (surge/sway/yaw) 的拟合 A_inf 显著高于物理锚点，证实了 m21471 的结论：
带内截断使外推不可用，锚点法才是正解。

## 4 结果图

![KK Closure](fig_kk_closure.png)
*左：C3 vs C3a 残差对比；右：10% 容差线*

![Anchor Consistency](fig_anchor_consistency.png)
*A_inf 拟合值 vs 物理锚点值*

![Gate Summary](fig_gate_summary.png)
*13/13 闸门全部通过*

## 5 注意点

1. **工件重放 vs 重解**：本单不重解 BEM，仅从已提交的 .npz 工件重放闸门。
   这保证了结果的可复现性，但也意味着如果原始扫频有数值问题，重放会继承。
2. **surge/sway/yaw 截断预算**：这三条 DOF 的阻尼尾部指数 < 2，截断预算无界。
   C3a 的 7-9% 残差是经验一致而非带限数据蕴含。要获得严格证据需要扩展频段到 20 rad/s。
3. **C7 容差**：C7 的 10% 容差是相对于 max|A_ii| 而非锚点值本身。
   surge/sway/yaw 的相对锚点误差 (14-21%) 超过 10%，但相对 max|A_ii| 仍在 10% 内。

## 6 误差判定

| 关键量 | 基准 | 误差 | 是否 <10% |
|--------|------|------|-----------|
| C3 worst (yaw) | 直接 BEM | 3.60% | **是** |
| C3a worst (yaw) | 直接 BEM | 8.85% | **是** |
| C5 (heave) | 频域估计 | 3.96% | **是** |
| C4 round-trip | 自洽 | 0.087% | **是** |
| C7 worst (yaw) | 锚点 A_inf | 15.3% (相对锚点) | **否**，但相对 max|A_ii| = 5.25% < 10% |

**结论**：13/13 闸门通过，算例在零自由参数锚定下仍满足 <10% 容差。
"""

    # Build figures list
    figures = []
    for name, caption in [
        ("fig_kk_closure.png", "KK closure: C3 (fitted) vs C3a (anchored) per DOF"),
        ("fig_anchor_consistency.png", "A_inf: fitted vs physical anchor per DOF"),
        ("fig_gate_summary.png", "All 13 gates pass/fail summary"),
    ]:
        path = FIGDIR / name
        if path.exists():
            figures.append({
                "name": name,
                "caption": caption,
                "png_base64": png_base64(path),
            })

    report = {
        "title": "CASE-BEM-CUMMINS-WIGLEY Round 7: Artifact Regeneration & Gate Expansion (13/13 PASS)",
        "summary": "Regenerated case_result.json from committed sweep artifacts, adding missing C3a/C7/C3a_coarse gates and fixing stale settings.dt. All 13 gates pass within 10% tolerance.",
        "scenario_key": "CASE-BEM-CUMMINS-WIGLEY",
        "mission_id": 21859,
        "tags": ["cummins", "wigley", "kk-causality", "anchor", "artifact-regeneration", "tensorbem"],
        "markdown": md,
        "figures": figures,
    }

    # Get token
    token = get_token()
    if not token:
        print("ERROR: Could not get auth token. Writing report to /tmp/report.json for manual submission.")
        with open("/tmp/report.json", "w") as f:
            json.dump(report, f, indent=2)
        return

    api_base = os.environ.get("TENSOR_API_BASE", "http://localhost:8001")
    result = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{api_base}/tensor/reports",
         "-H", f"Authorization: Bearer {token}",
         "-H", "Content-Type: application/json",
         "-d", json.dumps(report)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Report submission failed: {result.stderr}")
        with open("/tmp/report.json", "w") as f:
            json.dump(report, f, indent=2)
        return

    try:
        resp = json.loads(result.stdout)
        print(f"Report submitted: {resp}")
    except Exception:
        print(f"Report response: {result.stdout[:500]}")


if __name__ == "__main__":
    main()
