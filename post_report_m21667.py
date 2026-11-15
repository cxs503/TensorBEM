#!/usr/bin/env python3
"""Publish the CASE-BEM-CUMMINS-WIGLEY verification-debt report (mission 21667)."""
from __future__ import annotations
import base64, json, os, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = os.environ["TENSOR_API_BASE"]

def _post(path, body, token=None):
    req = urllib.request.Request(BASE+path, data=json.dumps(body).encode(),
                                 headers={"Content-Type":"application/json"})
    if token: req.add_header("Authorization","Bearer "+token)
    return json.load(urllib.request.urlopen(req))

def main():
    token = _post("/auth/login", {"username":os.environ["TENSOR_AGENT_USER"],
                                  "password":os.environ["TENSOR_AGENT_PASS"]})["access_token"]
    fig = (ROOT/"benchmarks"/"verified"/"casepool_bem_cummins_wigley"/"fig_kk_closure.png").read_bytes()
    md = """# CASE-BEM-CUMMINS-WIGLEY：Wigley 船 6DOF Cummins 记忆核——算例池机器注册收口（10/10）

## ① 案例介绍（物理背景/几何/参数表）

浮体在频域内的辐射问题给出每个遭遇频率下的附加质量 A(ω) 与辐射阻尼 B(ω)。Cummins (1962)
把这对频域系数写成时域卷积的记忆核 K(t)=(2/π)∫₀^∞ B(ω)cos(ωt)dω，而因果性要求
A(ω)-A_inf 是 B(ω) 的 Kramers-Kronig (Hilbert) 变换（Kotik & Mangulis 1962; Newman 1977）。

**几何**：数学 Wigley 船型 y=(B/2)(1-(2x/L)²)(1-(z/T)²)，L=3.0 m、B=0.3 m、T=0.1875 m
（L/B=10、B/T=1.6），Cb=4/9≈0.4444，排水体积 0.075 m³。频带 0.2–12 rad/s、41 节点（log 间隔）、
medium 20x6 生产网格（480 面板）。

## ② 怎么做的（方法/流程/关键设置/工具链）

1. **CHIEF 不规则频率抑制**：表面穿透船体外场 BIE 在离散不规则频率处奇异，
   `solve_radiation_3d(chief_points="auto")` 用 CHIEF 内点行最小二乘抑制（John 1950; Hulme 1982）。
2. **带限 KK 闭合**：`kk_added_mass_closure` 把 s>s_max 的阻尼用幂律尾 B~C·s^-p 拟合并以闭式
   主值积分补上；A_inf 由最小二乘闭合定（一个自由参数）。
3. **有效性闸门 C0**：`max solver residual <= 1e-2` 且 `min B >= 0` 才允许 C2–C6 发言
   （m20419 复算有效性欠账的收口——旧提交件曾被未收敛解污染）。
4. **工具链**：`tools/rerender_casepool_bem_cummins_wigley.py` 从已提交 `sweep_*.npz` 重画图并
   重渲染 CASE.md，不重解 BEM。

## ③ 计算结果（数字表）

| 闸门 | 判据 | 结果 |
| --- | --- | --- |
| C1 几何 | Cb 1%、体积 2%、L/B=10、B/T=1.6 | PASS |
| C0 求解有效性 | residual≤1e-2、min B≥0 | PASS（residual 2.45e-3）|
| C2 面板分辨率 | k√A_panel≤1 | PASS |
| C3 KK 闭合（6DOF）| \|A_BEM-A_KK\|/max\|A\|≤10% | PASS（逐 DOF 2.62/2.81/0.30/0.72/0.48/3.60%）|
| C4 往返 B→K→B | ≤2% | PASS（0.087%）|
| C5 自由衰减 vs 频域阻尼比 | ≤5% | **PASS（3.96%，dt=0.002）** |
| C6a 负控（零阻尼→零衰减）| \|K\|≤1e-12 | PASS |
| C6b 负控（破坏 B 形状→C3 FAIL）| >10% | PASS（worst 20.6%）|

## ④ 结果图

![fig_kk_closure](fig_kk_closure.png)
左：BEM 直接解 vs KK 重构的 A_ii(ω)/max|A_ii|（6 DOF，log 频轴）；右：逐 DOF KK 闭合残差
（% of max|A|），红色虚线为 10% 判据线。全部 DOF 残差落在判据线以下。

## ⑤ 注意点（踩过的坑/局限）

* **C5 的 dt 敏感性**：旧提交件（dt=0.01）自由衰减阻尼比相对误差 7.09%（>5%→FAIL）；当前提交件
  用 dt=0.002 重估得 3.96%（≤5%→PASS）。这是同一物理量在更细时间步上的收敛，不是换判据。
* **参考值是另一条路线**（直接 BEM 解）而非实验/解析表：Wigley 辐射系数无闭式解，L5 外部锚点
  记为 WITHHELD；本算例给出的是路线一致性证据（direct BEM vs KK 重构 vs 时域）。
* **由 A 反演 B 的那一支**（kkr_damping）对截断极敏感，只作诊断量报告，不作判据。

## ⑥ 误差判定（vs 基准）

* **C3（算例判据）**：逐 DOF KK 闭合残差全部 **<10%**（判据容差），显式回答：**是**。
* **C4**：B→K→B 往返阻尼相对误差 0.087% **<2%**（判据），**是**。
* **C5**：自由衰减 vs 频域阻尼比相对误差 3.96% **<5%**（判据），**是**。
* 无实验/解析基准（Wigley 辐射系数无闭式解），故不判 <3%；判据为算例池容差 <10%，已达标。
"""
    report = {
        "title": "CASE-BEM-CUMMINS-WIGLEY：Wigley 船 6DOF Cummins 记忆核——算例池机器注册收口（10/10）",
        "summary": "把已提交的 CASE-BEM-CUMMINS-WIGLEY（Wigley 6DOF Cummins 记忆核，KK 闭合<10%）"
                   "注册进 case_pool_registry 机器账本（此前缺失），并用 rerender 工具把 stale 的 "
                   "CASE.md/CASE_latest.md（旧 9/10、C5 FAIL）重渲染为当前 10/10 全过状态（C5 用 "
                   "dt=0.002 重估得 3.96%<5%）。当前提交件（m21100，band 0.2-12/41 节点/medium 480 "
                   "面板）通过 C0 有效性闸门（residual 2.45e-3、无负阻尼），逐 DOF KK 残差全部<10%。",
        "scenario_key": "CASE-BEM-CUMMINS-WIGLEY",
        "mission_id": 21667,
        "tags": ["BEM","Cummins","Wigley","KK-causality","6DOF","case-pool","verification-debt"],
        "markdown": md,
        "figures": [{
            "name": "fig_kk_closure.png",
            "caption": "Wigley 6DOF：BEM 直接解 vs KK 重构的 A_ii(ω)/max|A_ii|（左）与逐 DOF "
                       "KK 闭合残差（右，红色虚线=10% 判据线）。全部 DOF 残差落在判据线以下。",
            "png_base64": base64.b64encode(fig).decode(),
        }],
    }
    r = _post("/tensor/reports", report, token)
    print(json.dumps(r, indent=1)[:500])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
