"""Figure + report publication for m21326 (added mass at the two spectral limits)."""

from __future__ import annotations

import base64
import json
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

OUT = pathlib.Path("docs/cummins_wigley_added_mass_limit")
DATA = json.loads((OUT / "added_mass_limit.json").read_text())
SWEEP = np.load("docs/m18249_case_bem_cummins_wigley/sweep_coarse.npz")
DOFS = ["surge", "sway", "heave", "roll", "pitch", "yaw"]

# ----------------------------------------------------------------- figure --
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

rows = DATA["analytic_check"]["rows"]
panels = [r["panels"] for r in rows]
err = [100.0 * r["rel_err"] for r in rows]
axes[0].loglog(panels, err, "o-", color="C3", label="image-BEM $A_\\infty$ (heave)")
ref = err[-1] * (np.asarray(panels, float) / panels[-1]) ** -0.6
axes[0].loglog(panels, ref, "k--", lw=1, label="slope $-0.6$ guide")
axes[0].axhline(3.0, color="0.4", ls=":", label="3 % line")
axes[0].set_xlabel("panels on the half sphere")
axes[0].set_ylabel("rel. error vs $m_{full}/4$  [%]")
axes[0].set_title("exact check: half-immersed sphere, $A_\\infty$(heave)")
axes[0].legend(fontsize=8)
axes[0].grid(True, which="both", alpha=0.3)

omega = np.asarray(SWEEP["omega"], float)
added = np.asarray(SWEEP["added_mass"], float)
inf = DATA["levels"]["coarse"]["infinite_frequency"]["diag"]
zero = DATA["levels"]["coarse"]["zero_frequency"]["diag"]
for dof, colour in (("heave", "C0"), ("sway", "C1")):
    j = DOFS.index(dof)
    axes[1].plot(omega, added[:, j, j], "-", color=colour,
                 label=f"{dof} $A(\\omega)$ (coarse sweep)")
    axes[1].axhline(inf[dof], color=colour, ls="--", lw=1,
                    label=f"{dof} $A_\\infty$ = {inf[dof]:.2f}")
axes[1].axhline(zero["heave"], color="C0", ls=":", lw=1,
                label=f"heave $A(0)$ = {zero['heave']:.1f}")
axes[1].set_xlabel("$\\omega$ [rad/s]")
axes[1].set_ylabel("added mass [kg]")
axes[1].set_title("Wigley III: band sweep vs the two limits")
axes[1].legend(fontsize=7)
axes[1].grid(alpha=0.3)

per = DATA["closure"]["per_dof"]
x = np.arange(6)
width = 0.27
tail = [per[d]["tail_fit_inf"] for d in DOFS]
prof = [per[d]["profile_optimal_inf"] for d in DOFS]
phys = [per[d]["physical_inf"] for d in DOFS]
axes[2].bar(x - width, tail, width, label="$A_\\infty$ two-term tail fit")
axes[2].bar(x, prof, width, label="$A_\\infty$ KK-optimal (m20594)")
axes[2].bar(x + width, phys, width, label="$A_\\infty$ image BEM (this work)")
axes[2].axhline(0.0, color="k", lw=0.8)
axes[2].set_xticks(x)
axes[2].set_xticklabels(DOFS, rotation=30)
axes[2].set_ylabel("$A_\\infty$ (diagonal)")
axes[2].set_title("three estimates of $A_\\infty$; the tail fit goes negative")
axes[2].legend(fontsize=8)
axes[2].grid(axis="y", alpha=0.3)
fig.tight_layout()
png = OUT / "fig_m21326_added_mass_limit.png"
fig.savefig(png, dpi=140)

# ----------------------------------------------------------------- report --
G3 = {d: {k: per[d]["residuals"][k]["G3"] for k in
          ("tail_fit", "profile_optimal", "physical_inf")} for d in DOFS}
analytic = DATA["analytic_check"]["rows"]
lines = []
lines.append("## 3 计算结果（全部由 `docs/cummins_wigley_added_mass_limit/added_mass_limit.json` 程序化注入）\n")
lines.append("### 3.1 解析校验（半浸球，真值闭式）\n")
lines.append("| panels | $A_\\infty$(heave) [kg] | 真值 $m_{full}/4$ [kg] | 相对误差 |")
lines.append("|---|---|---|---|")
for r in analytic:
    lines.append(f"| {r['panels']} | {r['A_inf_heave']:.2f} | {r['exact_heave']:.2f} | "
                 f"{100*r['rel_err']:.2f} % |")
lines.append("")
lines.append("### 3.2 Wigley III 六自由度两极限（coarse 网格，192 面元）\n")
lines.append("| DOF | $A_\\infty$（Dirichlet） | $A(0)$（刚壁） | 比值 | 扫描最低频 $A(0.2)$ |")
lines.append("|---|---|---|---|---|")
for d in DOFS:
    lines.append(f"| {d} | {inf[d]:.4g} | {zero[d]:.4g} | {zero[d]/inf[d]:.3f} | "
                 f"{per[d]['sweep_A_at_lowest_omega']:.4g} |")
lines.append("")
lines.append("### 3.3 三个 $A_\\infty$ 估计与 G3 残差\n")
lines.append("| DOF | 尾拟合 | KK 最优(m20594) | 镜像 BEM | G3(尾拟合) | G3(KK 最优) | G3(物理) |")
lines.append("|---|---|---|---|---|---|---|")
for d in DOFS:
    lines.append(f"| {d} | {per[d]['tail_fit_inf']:.3f} | {per[d]['profile_optimal_inf']:.3f} | "
                 f"{per[d]['physical_inf']:.3f} | {100*G3[d]['tail_fit']:.1f} % | "
                 f"{100*G3[d]['profile_optimal']:.1f} % | {100*G3[d]['physical_inf']:.1f} % |")
lines.append("")
tables = "\n".join(lines)

markdown = f"""# m21326 — Wigley Cummins 锚点的 $A_\\infty$ 欠账：用镜像体 BEM 直接算两个频谱极限

锚点 `CASE-BEM-CUMMINS-WIGLEY`（算例池·Wigley 船 Cummins 记忆核（6DOF），ref = Cummins 方程 + Kramers–Kronig 因果，容差 <10 %），任务 m21326，分支 `agent/m21326`。

## 1 案例介绍

Wigley III：L=3.0 m，B=0.3 m，T=0.1875 m，Cb=0.543，ρ=1025 kg/m³，g=9.80665 m/s²，无限水深，旋转中心 (0,0,−T/2)；几何门先过（Cb 体积误差 0，L/B=10，B/T=1.6）。数据沿用仓内已入库的 41 点粗网格辐射扫频 `docs/m18249_case_bem_cummins_wigley/sweep_coarse.npz`（ω∈[0.2,12] rad/s，192 面元），本轮不重跑频率域扫频。

**要解决的问题**：G3/G5 的失败在 m20594/m21140 里被归因到 `A_∞` 的尾拟合病态（两参数尾拟合给出负的 sway/yaw 加质量），但那条归因只用"KK 一致的最优 $A_\\infty$"做过自校验——$A_\\infty$ 到底是**多少**，仓内一直没有第二把尺子。本轮把 $A_\\infty$ 从"估计"变成"计算"。

## 2 怎么做的

线性化自由面条件 $\\partial\\phi/\\partial z=(\\omega^2/g)\\phi$ 在频率两端退化：$\\omega\\to0$ 得 $\\partial\\phi/\\partial z=0$（刚壁），$\\omega\\to\\infty$ 得 $\\phi=0$（压力释放）。两者都是**下半空间的 Laplace 边值问题**，因此都不必在频率域里求解：

1. 对水线截断的船体面元（`WettedHullMesh`）做 $z=0$ 镜像，焊接水线顶点，得到**闭合的双倍体**（体积恰为 2×排水体积，实测比值 2.0000000000000004）；
2. 用 `solver3d` 的解析单/双层算子解该闭合体的**无限流体**外问题，边值按势的奇偶性延拓 $q'(Mx)=s\\,q(x)$（$s=-1$ 对应 Dirichlet 极限，$s=+1$ 对应刚壁极限），$q$ 为广义法向速度 $(n,\\ (r-c)\\times n)$；
3. 只在**物理（下半）面元**上做 Green 积分：$A_{{ij}}=-\\rho\\int_{{hull}}\\phi_j n_i\\,dS$。

镜像法使自由面条件**在整个平面 $z=0$ 上精确成立**，没有截断半径、没有自由面面元——唯一离散就是船体本身。新模块 `src/tensorbem/added_mass_limit.py`，驱动 `scripts/cummins_wigley_added_mass_limit.py`，测试 `tests/test_added_mass_limit.py`（9 passed）。

**为什么有真值**：半浸球的下半水面与其镜像在几何上拼成整球；在 heave 上 $s=-1$ 的延拓恰是整球的刚性平移，故 $A_\\infty(\\text{{heave}})=m_{{full}}/4$ 是闭式。

{tables}

## 4 结果图

![fig](fig_m21326_added_mass_limit.png)

左：半浸球解析校验（7.8 % → 2.0 % → 0.91 %，随面元单调收敛）；中：粗网格扫频 $A(\\omega)$ 与两条极限线——高频端 $A(12)=35.6$ kg 位于 $A_\\infty=65.9$ kg 之下，说明该段扫频被高频数值污染（round-5 记录的 ω=20 负阻尼同源），不能用来外推 $A_\\infty$；右：三个 $A_\\infty$ 估计，尾拟合在 surge/sway/yaw 上为负（物理上不可能），KK 最优与镜像 BEM 在 sway/yaw/pitch 上相差 40 %–70 %。

## 5 注意点（局限与适用范围）

* **解析校验的门限**：半浸球 828 面元时 $A_\\infty$ 偏低 0.91 %，且恒**偏低**（球面内接多面体面积小于真球），故镜像法给出的是**下界偏置**；真实误差量级即 1 % 以内。
* **$A(0)$ 与扫频低端不可直接对比**：Wigley 的水线处船体与水面相切（$y\\propto(1-\\zeta^2)$），刚壁极限下船体与镜像之间是零夹角楔形挤压流，$A(0)$ 对网格敏感；粗网格 heave 给 175.6 kg，而频率域 ω=0.2 rad/s 只给 130.8 kg 且随 ω **上升**（130.8→140.6@1.26）。这条差**不是**本模块的验证指标，而是"低端需要专门研究"的线索（medium 网格阶梯已递交集群 cj3347，本轮预算内未回收）。
* **旋转中心**：延拓规则由势的奇偶性推出，与中心是否落在 $z=0$ 无关，故 (0,0,−T/2) 可直接用；矩阵对称性误差 $<10^{{-8}}$，双层行和误差 $<10^{{-10}}$。
* 本模块只给**两个端点**，不给频带内的 $A(\\omega)$；频带内仍须频率域求解器。

## 6 误差判定（逐项，显式回答"是否 <3 %"）

| 关键量 | 基准 | 误差 | 是否 <3 % |
|---|---|---|---|
| 半浸球 $A_\\infty$(heave)，828 面元 | 闭式 $m_{{full}}/4=1073.38$ kg | **0.91 %** | **是** |
| 双倍体体积 / 船体体积 | 2 | $4\\times10^{{-16}}$ | 是 |
| 矩阵对称性、双层行和 | 0 | $<10^{{-8}}$ / $<10^{{-10}}$ | 是 |
| Wigley $A_\\infty$(heave) 网格收敛性 | 无第二参考 | **未能判定**（medium 阶梯 cj3347 排队中） | 未判 |
| Wigley $A(0)$ vs 频率域 ω=0.2 | 同一物理量 | heave 34 %、pitch 80 %、yaw −12 % | **否**（见 §5，刚壁极限的接触线敏感性，不作合格性判据） |

## 7 结论与 honest 状态

1. **能力增量（已在真值上校验）**：`added_mass_limit` 一次给出 6×6 的 $A_\\infty$ 与 $A(0)$，与频率域求解器零共享（只用解析单双层算子），半浸球上 0.91 %。
2. **锚点的归因被修正（负面结果，如实记录）**：m20594 的结论"G3 残差 = $A_\\infty$ 估计偏置，校准后六 DOF <10 %"只在"$A_\\infty$ 自由取"的意义上成立。用**物理**的 $A_\\infty$ 代入同一判据，G3 = {min(G3[d]['physical_inf'] for d in DOFS)*100:.1f} %–{max(G3[d]['physical_inf'] for d in DOFS)*100:.1f} %，**不闭合**；同时尾拟合的负值确认是病态（本模块给出正的下界量级）。也就是说：粗网格 $(A,B)$ 对的非因果残差**不能全部归给 $A_\\infty$ 估计器**，扫频本身的偏差（尤其 sway/yaw 高端）至少同等重要。
3. **锚点仍是 FAIL，未粉饰**：G3/G5 未通过，G4 仍未决；本轮没有把任何判据改宽，只增加了独立的 $A_\\infty$ 基准与一条被证伪的归因。
4. 复现：`PYTHONPATH=src python3 scripts/cummins_wigley_added_mass_limit.py --levels coarse`（约 20 s）；测试 `PYTHONPATH=src python3 -m pytest tests/test_added_mass_limit.py -q`。
"""

(OUT / "dev_report_m21326_added_mass_limit.md").write_text(markdown)
pathlib.Path("docs/dev_report_m21326_added_mass_limit.md").write_text(markdown)

base = os.environ.get("TENSOR_API_BASE", "http://localhost:8001/api/v1")
user = os.environ.get("TENSOR_AGENT_USER")
password = os.environ.get("TENSOR_AGENT_PASS")
token = None
if user and password:
    response = requests.post(f"{base}/auth/login",
                            json={"username": user, "password": password}, timeout=30)
    response.raise_for_status()
    token = response.json().get("access_token")
headers = {"Authorization": f"Bearer {token}"} if token else {}

payload = {
    "title": "Wigley Cummins 锚点：镜像体 BEM 直接计算 A_∞/A(0) 两频谱极限（半浸球校验 0.91 %）",
    "summary": (
        "CASE-BEM-CUMMINS-WIGLEY 的 A_inf 欠账：用镜像/双倍体 Rankine BEM（自由面条件精确满足，"
        "无截断面元）算出 Wigley III 六自由度的高频(Dirichlet)与零频(刚壁)加质量；半浸球闭式 "
        "A_inf=m_full/4 校验 0.91 %。物理 A_inf 代入 G3 后残差仍 14.6 %–65.3 %，说明 G3 失败"
        "不能全部归因于 A_inf 尾拟合，粗网格扫频本身有偏差（负面结果入账）。"
    ),
    "scenario_key": "CASE-BEM-CUMMINS-WIGLEY",
    "mission_id": 21326,
    "tags": ["TensorBEM", "BEM", "added-mass", "Wigley", "Cummins", "Kramers-Kronig",
             "image-method", "verification"],
    "markdown": markdown,
    "figures": [{
        "name": "fig_m21326_added_mass_limit.png",
        "caption": ("左：半浸球 A_inf(heave) 对闭式 m_full/4 的网格收敛（7.8 %→2.0 %→0.91 %）；"
                    "中：粗网格扫频 A(ω) 与镜像 BEM 的两条极限线；右：尾拟合 / KK 最优 / 镜像 BEM "
                    "三个 A_inf 估计，尾拟合在 surge/sway/yaw 上为负。"),
        "png_base64": base64.b64encode(png.read_bytes()).decode(),
    }],
}
response = requests.post(f"{base}/tensor/reports", json=payload, headers=headers, timeout=120)
print(response.status_code)
print(response.text[:600])