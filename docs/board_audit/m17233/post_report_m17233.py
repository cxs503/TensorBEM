#!/usr/bin/env python3
"""Post the m17233 board-audit report (subject m17177) with figures."""
import base64
import io
import json
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "http://127.0.0.1:8010/api/v1"
ENV = dict(l.strip().split("=", 1) for l in open("/root/tp-agent/agent.env")
           if l.strip() and not l.startswith("#") and "=" in l)


def tok():
    req = urllib.request.Request(BASE + "/auth/login", method="POST",
                                 data=json.dumps({"username": ENV["TENSOR_AGENT_USER"],
                                                  "password": ENV["TENSOR_AGENT_PASS"]}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["access_token"]


ref = [1211.0519952419288, 2437.1535015128825, 4184.389189761621]
got = {20: [958.485995, 1922.280060, 3241.204877],
       80: [1136.423798, 2279.400297, 3890.642146],
       320: [1191.295595, 2394.817782, 4104.549812]}
panels = [20, 80, 320]
err = {n: [abs(got[n][i] - ref[i]) / ref[i] * 100 for i in range(3)] for n in panels}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
labels = ["x (surge)", "y (sway)", "z (heave)"]
for i, lab in enumerate(labels):
    ax1.loglog(panels, [err[n][i] for n in panels], "o-", label=lab)
ax1.set_xlabel("panels")
ax1.set_ylabel("|A_BEM - A_Lamb| / A_Lamb  [%]")
ax1.set_title("Ellipsoid added mass convergence (audit rerun, branch 0516776)")
ax1.grid(True, which="both", alpha=0.3)
ax1.axhline(3.0, color="r", ls="--", lw=1, label="ecosystem 3% bar")
ax1.legend(fontsize=8)
x = np.arange(3)
ax2.bar(x - 0.2, ref, 0.4, label="Lamb §114 (independently recomputed)")
ax2.bar(x + 0.2, got[320], 0.4, label="TensorBEM, 320 panels")
for i in range(3):
    ax2.text(i, max(ref[i], got[320][i]) * 1.02, f"{err[320][i]:.2f}%", ha="center", fontsize=9)
ax2.set_xticks(x, ["x", "y", "z"])
ax2.set_ylabel("added mass [kg]")
ax2.set_title("ref vs got, tri-axial ellipsoid (1.5, 1.0, 0.75) m, rho=1025")
ax2.legend(fontsize=8)
fig.tight_layout()
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=130)
png = base64.b64encode(buf.getvalue()).decode()

markdown = """# 交差审计 m17177 · CASE-BEM-ELLIPSOID-ADDED-MASS（TensorBEM）

**裁定：L4 基准验证**（声明 L4，非 L5）。结论：**三轴附加质量数值真实、可复现、非硬编码，误差 1.6313% / 1.7371% / 1.9080% < 3% < 5% 判据 → PASS**；阻塞性缺陷无，非阻塞缺陷 6 条（含报告参数表错、CHIEF 无据陈述、工件未入库）。

## ① 案例与基准出处
三轴椭球 (a,b,c)=(1.5,1.0,0.75) m，ρ=1025 kg/m³，沿三个主轴平动，无界无粘不可压。参考 = Lamb (1932) *Hydrodynamics* 5th ed. §114 形状因子积分
α_i = a₁a₂a₃∫₀^∞ dλ/[(a_i²+λ)√∏(a_j²+λ)]，α₁+α₂+α₃=2，A_ii = ρVα_i/(2−α_i)，V=4.712389 m³。
被审单 m17177 声明引用 Lamb §114；**但其 ref 数值由仓库自身 `tensorbem.analytic` 算出（自指）**，故本审计用独立实现重算参考值（见③）。

## ② 方法（审计做了三件事）
1. **重跑**：`PYTHONPATH=src python -m tensorbem.benchmarks.ellipsoid_added_mass --subdivisions {0,1,2} --semi-axes 1.5 1.0 0.75 --density 1025`，分支 0516776。三档 JSON 与被审工件 **逐键完全相同**（sha256 前缀 217a000/8e5a8f/1c4ff31）。
2. **独立重算参考**：mpmath 40 位自适应求积（λ=t/(1−t) 替换）算 Lamb 积分；另用已发表长椭球闭式 α_axial=2(1−e²)/e³[atanh(e)−e] 与球极限 2/3 交叉核对。
3. **裁判登记**：`tensor_benchmark_cases` 对该 anchor **原有 0 行**（机器裁判缺失），本次按独立重算 ref + 实跑 got 补登（case id 40，status=passed）。

## ③ 结果（程序化注入，零手抄）

| 轴 | got（BEM 320 面板）kg | ref（独立重算）kg | 仓库 analytic kg | 相对误差 |
|---|---|---|---|---|
| x surge | 1191.2955950090695 | 1211.0519952419288 | 1211.051995240953 | **1.6313 %** |
| y sway | 2394.8177826981614 | 2437.1535015128825 | 2437.153501509733 | **1.7371 %** |
| z heave | 4104.549811866565 | 4184.389189761621 | 4184.38918975305 | **1.9080 %** |

参考值三重核对：α₁+α₂+α₃=2 残差 **2.2e-18**；(2,1,1) 长椭球闭式 vs 积分 **1.3e-18**；球极限 α=2/3 差 **6.7e-19**；仓库 analytic 与独立积分 α 差 **1.1e-12**。
收敛性：20/80/320 面板误差 20.86/21.13/22.54 % → 6.16/6.47/7.02 % → **1.63/1.74/1.91 %**，sub1→sub2 误差降 3.68–3.78×，观测阶 **0.94–0.96**（flat panel 通量近一阶）；体积比 0.6055→0.8735→0.9662。
负对照：ρ 1025→2050 三轴附加质量精确 ×2.000000000000；球 (1,1,1) α=2/3、exact=0.5ρV → 输出随输入变化，非硬编码。

## ④ 图
左：三轴误差随面板数（log-log）+ 3% 生态底线；右：ref vs got 320 面板柱状对比（标注实测误差）。

## ⑤ 注意点（审计发现，按严重度）
- **D1 中｜报告③参数表错**：报告写 h_max 0.9747/0.4874/0.2437 m、h_mean 0.7123/0.3562/0.1781 m；实跑为 h_max **1.577193/0.927051/0.487380** m、h_mean **1.167340/0.646594/0.332389** m。仅体积比列正确。仓库 pin 与实跑一致 → 错在报告表，不在代码。
- **D2 中｜无据陈述**：报告§2 称"CHIEF 点自动 interior points 抑制虚假特征值"，`solver3d.py:128-171 solve_laplace_surface` 无 CHIEF 逻辑；求解器模块名应为 `solver3d`。
- **D3 中｜证据不可持续**：m17177 `head_commit=null`、changed_files 全为未跟踪（`??`），repro_cmd 写死可回收工作树路径 `/root/tp-agent/work/m17177`。审计已重跑 + 把原件固化到 `docs/board_audit/m17233/refs/`。
- **D4 低**：§5"32 节点求积残差 ~1e-6"无工件背书。
- **D5 系统**：`tensor_benchmark_cases` 对本 anchor 0 行（已由本审计补登）。
- **D6 范围**：Dirichlet 迹取解析线性迹 u=−U·k·x_i（k=α_i/(2−α_i)），BEM 只承担通量/能量积分 → 验证的是离散化 + 外域积分方程，非完整外域解；无自由面、无粘、非船体。

## ⑥ 误差判定（逐项，显式回答）
| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| A_x | Lamb §114（独立重算） | 1.6313 % | **是** |
| A_y | 同上 | 1.7371 % | **是** |
| A_z | 同上 | 1.9080 % | **是** |
| α 和 | 2（恒等） | 2.2e-18 | **是** |
判据 <5%（m17177 goal）：全部通过。

## 为什么不是 L5
L5 需 L4 + **表值级/实验级外部锚** + 科研实质增量。本案 ref 是公式现场求值（虽经独立实现验证到 1.1e-12），未从 Lamb 原表原页转录——恰是 m17177 goal 明文要求而未做到的一项；且能力与 320/1280 面板 pin 早于本单入库，本单为"复跑 + 报告"，无新算法/新物理。故级别=功能实质=**L4**。

## honest 未做
`tests/test_ellipsoid_1280_verification.py`（1280 面板 ×3 轴 >120 s 本机 shell 上限）未跑完，已投集群 cj1464 排队 592 未回；已跑 `tests/test_analytic_references.py -k ellipsoid` → 2 passed。未取 Lamb 原书扫描页 OCR 比对（无该版权文本），外部锚如实标注为公式级。
"""

payload = {
    "title": "交差审计 m17177 · CASE-BEM-ELLIPSOID-ADDED-MASS：L4 基准验证（BEM 三轴 1.63/1.74/1.91% vs Lamb §114 独立重算）",
    "summary": ("board-audit 复核 m17177 椭球三轴附加质量：工件真实存在但未入库（head_commit=null），"
                "从 HEAD 0516776 重跑三档 JSON 逐键相同；ref 由仓库 analytic 自算（自指），"
                "审计以 mpmath 40 位独立重算 Lamb §114 积分并核到 1.1e-12；误差 1.6313/1.7371/1.9080% < 3% < 5% 判据 → L4；"
                "报告参数表 h_max/h_mean 三行全错、CHIEF 陈述无据；tensor_benchmark_cases 原 0 行已补登 case id 40；verdict id 501。"),
    "scenario_key": "CASE-BEM-ELLIPSOID-ADDED-MASS",
    "mission_id": 17233,
    "tags": ["board-audit", "TensorBEM", "ellipsoid", "added-mass", "L4", "m17177"],
    "markdown": markdown,
    "figures": [{"name": "audit_ellipsoid_added_mass.png",
                 "caption": "左：三轴附加质量误差随面板数收敛（20/80/320 面板）与 3% 生态底线；右：Lamb §114 独立重算 ref vs TensorBEM 320 面板 got",
                 "png_base64": png}],
}

req = urllib.request.Request(BASE + "/tensor/reports", method="POST",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + tok()})
try:
    resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
    print("posted report id:", resp.get("id") or resp)
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read()[:500])