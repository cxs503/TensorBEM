"""Publish the m17302 audit report to the platform (/tensor/reports) with figures."""
import base64
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
W = "/root/tp-agent/work/m17302"


def api(method, path, payload=None, token=None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data=data, timeout=120) as r:
        return json.loads(r.read() or b"{}")


tok = api("POST", "/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                  "password": os.environ["TENSOR_AGENT_PASS"]})["access_token"]


def fig(path, name, caption):
    return {"name": name, "caption": caption,
            "png_base64": base64.b64encode(open(path, "rb").read()).decode()}


md = open(f"{W}/docs/board_audit/m17302/AUDIT_m17265.md").read()
md += """

---

## 审计裁定（落表结果）

* `tensor_maturity_verdicts.id=515` — repo=TensorBEM, scope=`m17265 CASE-BEM-CATHODIC-SPHERE`, **level=2**,
  verdict_by=board-audit；本地按 maturity_gate 的判据方案预跑复核 → gate_level=2, why=[]。
* `tensor_benchmark_cases.id=47` — 本审计登记该锚点的机器裁判行（此前 0 行），status=passed。

## 误差判定（逐项）

| 关键量 | 基准 | 实测 | 误差 | <3%? |
|---|---|---|---|---|
| 总电流 I（1280 面板） | 4πσaφ_a = −43.982297 A | −43.836366 A | 0.3318 % | **是** |
| 总电流 I（320 面板，CI 级） | 同上 | −43.410210 A | 1.3007 % | **是** |
| 总电流 I（80 面板） | 同上 | −41.820245 A | 4.9157 % | 否（该级不计入判据，已披露） |
| 远场 Gauss 定律路电流（外推） | 同上 | −43.836367 A | 0.3318 % | **是** |
| 外场电位 φ(r)（r=1.25–12 m） | φ_a a/r | 见 results.json | 1.30 %（sub2）/4.92 %（sub1），沿 r 平坦 | **是** |
| 面积分 vs 远场 两路一致 | 0 | 3.51e-09 | — | **是** |
| 无量纲不变性 I/(4πσaφ) | 1 | 0.98699278050491（4 个参数点） | 离散度 2.2e-16 | **是** |

**显式回答：关键量误差 <3% 成立**（最细级 0.3318 %，CI 级 1.3007 %）。审计另加一项非原报告项的检查：
误差/面板面积亏损 = 0.686/0.691/0.696（过原点斜率 0.6864，最大残差 1.8e-03）→ 残差是内接多面球几何
（弦代弧）而非边界求积精度，原报告 §6.3 的表述过宽（F3 已记）。
"""

payload = {
    "title": "交差审计 L2：CASE-BEM-CATHODIC-SPHERE（m17265 / TensorBEM）——基准数字为真，集成面缺席，精度归因纠正",
    "summary": ("独立核验 m17265 三件套：run_artifacts 字节可复、闭式解 4πσaφ_a 审计重算逐位一致、"
                "1280 面板误差 0.3318 %（<3 % 成立）、10 条测试 auditor 复跑通过；判决 L2 —— 分支未并入 main、"
                "main 侧 0 引用故拒 L3/L4。新增证据：误差正比于内接球面积亏损（斜率 0.6864），说明该锚点证明的是 "
                "σ/4π 记账与外域表示式，不是边界求积精度。已登记缺失的 tensor_benchmark_cases 行（id=47）与 verdict（id=515）。"),
    "scenario_key": "bem_cathodic_sphere_dnv_b401_audit",
    "mission_id": 17302,
    "tags": ["TensorBEM", "board-audit", "cathodic-protection", "DNV-RP-B401",
             "maturity-L2", "CASE-BEM-CATHODIC-SPHERE", "m17265", "verification"],
    "markdown": md,
    "figures": [
        fig(f"{W}/docs/board_audit/m17302/fig_audit_cathodic_sphere.png",
            "audit_error_vs_geometry.png",
            "左：实测电流误差（圆点）与内接 icosphere 面积亏损（方块）随 h/a 平行下降 —— 误差机理=几何弦代弧，非求积误差；右：外场电位相对误差沿半径 r 平坦（1/r 定律复现到机器精度，残差为整体标度误差）。"),
        fig(f"{W}/docs/board_audit/m17302/snapshot/figures/convergence_current.png",
            "subject_convergence_current.png",
            "被审方原图（subject artifact，sha256 9cc016c9…）：电流误差 vs 面板数、二阶参考线与 5 %/3 % 判据带 —— 审计核对与 results.json 数值一致。"),
    ],
}
out = api("POST", "/tensor/reports", payload, tok)
print(json.dumps(out)[:400])