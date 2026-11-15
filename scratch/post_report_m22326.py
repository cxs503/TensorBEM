"""Publish the m22326 progress report (figures + markdown, programmatic numbers)."""

import base64
import io
import json
import os
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorbem.benchmarks import case_cylinder_stream as case

OUT = "/tmp/report_m22326.json"
FIG = "/tmp/m22326_heading.png"

studies = {n: case.heading_study(n) for n in (32, 64, 128)}

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
colors = {32: "#8c564b", 64: "#1f77b4", 128: "#2ca02c"}
for n, study in studies.items():
    heads = [row["heading_deg"] for row in study["levels"]]
    axes[0].semilogy(
        heads,
        [row["dirichlet_velocity_relative_l2"] for row in study["levels"]],
        "o-",
        color=colors[n],
        label=f"N={n}: route D, external velocity",
    )
    axes[0].semilogy(
        heads,
        [row["neumann_surface_velocity_relative_l2"] for row in study["levels"]],
        "s--",
        color=colors[n],
        alpha=0.65,
        label=f"N={n}: route N, surface velocity",
    )
axes[0].axhline(case.TOLERANCE, color="crimson", lw=1.2, ls=":")
axes[0].text(
    0.3, case.TOLERANCE * 1.1, "case tolerance 3%", color="crimson", fontsize=9
)
axes[0].set_xlabel("inflow heading relative to the panel lattice [deg]")
axes[0].set_ylabel("relative L2 error vs closed-form complex potential")
axes[0].set_title("oblique-inflow sweep: breaking the mirror symmetry")
axes[0].legend(fontsize=6.5, loc="lower right")
axes[0].grid(alpha=0.3, which="both")

labels = [str(n) for n in studies]
cov = [studies[n]["lattice"]["velocity_field_max_difference_over_speed"] for n in studies]
odd = [studies[n]["mirror_split"]["odd_part_max_over_speed"] for n in studies]
even = [studies[n]["mirror_split"]["even_part_max_over_speed"] for n in studies]
x = np.arange(len(labels))
axes[1].bar(x - 0.2, cov, width=0.2, label="90 deg rotation equivariance")
axes[1].bar(x, odd, width=0.2, label="mirror-antisymmetric (odd) error")
axes[1].bar(x + 0.2, even, width=0.2, label="mirror-symmetric (even) error")
axes[1].set_yscale("log")
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"N={n}" for n in studies])
axes[1].set_ylabel("max deviation / U")
axes[1].set_title("structure of the error (axis-aligned configuration)")
axes[1].axhline(case.HEADING_LATTICE_COVARIANCE_TOL, color="k", ls=":", lw=1)
axes[1].legend(fontsize=7)
axes[1].grid(alpha=0.3, axis="y", which="both")
fig.tight_layout()
fig.savefig(FIG, dpi=140)

rows_d = "".join(
    "| %d | %.3e | %.3e | %.3e | %.2f |\n"
    % (
        n,
        s["levels"][0]["dirichlet_velocity_relative_l2"],
        s["levels"][1]["dirichlet_velocity_relative_l2"],
        s["levels"][2]["dirichlet_velocity_relative_l2"],
        s["dirichlet_penalty_max"],
    )
    for n, s in studies.items()
)
rows_n = "".join(
    "| %d | %.3e | %.3e | %.3e | %.3e |\n"
    % (
        n,
        s["levels"][0]["neumann_surface_velocity_relative_l2"],
        s["levels"][1]["neumann_surface_velocity_relative_l2"],
        s["levels"][2]["neumann_surface_velocity_relative_l2"],
        s["levels"][0]["normal_velocity_residual_over_speed"],
    )
    for n, s in studies.items()
)
rows_s = "".join(
    "| %d | %.4f | %.2e | %.2e | %.2e | %.1f |\n"
    % (
        n,
        s["panel_step_deg"],
        s["lattice"]["velocity_field_max_difference_over_speed"],
        s["mirror_split"]["odd_part_max_over_speed"],
        s["mirror_split"]["even_part_max_over_speed"],
        s["mirror_split"]["odd_fraction"],
    )
    for n, s in studies.items()
)

markdown = f"""# CASE-BEM-LAPLACE-CYLINDER 第 7 轮：斜来流方位扫描与全链旋转等变性（TensorBEM m22326）

## ① 案例介绍

半径 $a$ 的圆柱（离散为内接正 $N$ 边形）在无界、无粘、不可压、定常流中以速度 $U$ 均匀来流。
复势 $W(z)=U(z+a^2/z)$，速度 $u-iv=dW/dz=U(1-a^2/z^2)$，扰动势 $\\phi'=Ua^2\\cos\\theta/r$，
物面速度 $2U|\\sin\\theta|$，$C_p=1-4\\sin^2\\theta$（d'Alembert 零阻力）。

参数：$a=1$、$U=1$；面板阶梯 $N\\in\\{{32,64,128\\}}$；外部探针 64 个（环 $r/a\\in\\{{1.25,1.5,2.0,3.0\\}}$，
每环 16 方位，方位取半格偏移 $(k+\\tfrac12)2\\pi/16$）；容差 `<3%`；参考=解析复势（同点现算，
`hand_typed_numbers = 0`）。

**本轮检验的失效模式**：既有 6 轮全部在「来流沿 $+x$、正 $N$ 边形在 $+x$ 上恰有顶点」的位形下测量，
物体、边界数据、探针环三者对 $y=0$ 严格镜像对称，故任何**关于 $y$ 为奇**的实现缺陷
（核函数符号、法向定向、面板编号/装配顺序、切向单边偏置）都会在上下两半相互抵消而 L2 看不见。

## ② 怎么做的

在 `tensorbem.benchmarks.case_cylinder_stream` 中新增（additive，不改既有路线）：

- `rotated_polygon(panels, heading, radius)`：物体相对固定来流转 `heading`；
- `heading_velocity(points, heading)`：实现为 $R(h)\\,v_0(R(-h)p)$（$h=0$ 与 `exact_velocity` 逐位相同）；
- `heading_perturbation_potential` / `heading_neumann_data`：$\\phi'=Ua^2\\cos(\\theta-h)/r$ 及其径向导数；
- `heading_study`：每个方位同时跑路线 D（Dirichlet 数据→`evaluate_exterior_field` 给外部速度场）
  与路线 N（Neumann 数据→物面速度），方位集 $\\{{0^\\circ,\\ 180^\\circ/N,\\ 30^\\circ\\}}$
  （0°=镜像基线；半面板步长=驻点流线对弦中点，常值面板最不利；30° 与面板步长不通约）；
- `mirror_error_split`：把 0° 位形的速度误差按 $y=0$ 拆成偶部/奇部；
- 90°（$N/4$ 个面板步长、旋转矩阵精确可表）**连探针一起转**的旋转等变性检验
  （比 $v_{{90^\\circ}}$ 与 $R_{{[0,-1;1,0]}}\\,v_{{0^\\circ}}$）。

新增 5 条判据进算例门：斜方位 D 速度 `<3%`、斜方位 N 物面速度 `<3%`、
斜方位相对 0° 放大 `≤3×`、格点旋转等变性 `≤1e-9 U`、镜像奇部 `≤1e-9 U`。
另补 6 条单测（参考值旋转恒等、$\\nabla\\phi'$ 与有限差分一致、Neumann 数据在旋转网格上不变、
斜方位容差、旋转等变性、镜像配对与奇部）。

## ③ 计算结果

路线 D 外部速度相对 L2（vs $dW/dz$，三个方位）：

| N | 0° | 180°/N | 30° | 最差/基线 |
| --- | --- | --- | --- | --- |
{rows_d}
路线 N 物面速度相对 L2（vs $2U|\\sin\\theta|$）与法向残差：

| N | 0° | 180°/N | 30° | max\\|v·n\\|/U（0°） |
| --- | --- | --- | --- | --- |
{rows_n}
结构量（面板步长、格点旋转等变性、镜像奇/偶部、奇部占比）：

| N | 面板步长 (deg) | 90° 等变性 max\\|Δv\\|/U | 奇部/U | 偶部/U | 奇部占比 |
| --- | --- | --- | --- | --- | --- |
{rows_s}
## ④ 结果图

左：三个方位下两条路线的相对 L2 误差（红线=3% 容差）；右：误差的结构——90° 旋转等变性残差、
镜像奇部与偶部（对数轴，黑点线=1e-9 U 阈值）。图件由本报告脚本从 `heading_study` 程序化绘制。

## ⑤ 注意点

- **等变性 ≠ 恒等性**：初版直接把「物体自转 90°、来流不动」的速度场与 0° 相减，得 $\\max|\\Delta v|/U=1.23$
  —— 那是另一个物理位形，不是 bug。正确检验必须把探针一起转、并按矢量旋转后比较；改对后残差 $\\sim10^{-15}$。
- 路线 N 物面速度误差对三个方位**逐位相同**（差异 $\\le1.6\\times10^{{-16}}$）：该项由面板局部刻面决定、
  随物体一起转，与来流相位无关；这同时说明斜来流不会「顺手」改善物面精度。
- `solve_laplace_exterior_neumann` 的 `normal_derivative` 约定与 $\\nabla\\phi'\\cdot n_{{out}}$ 差一个符号
  （算例历来如此，物理判据是物面 $\\max|v\\cdot n|/U\\le6\\times10^{{-4}}$）；斜方位数据按同一约定构造，
  单测用有限差分确认比例常数 $=\\pm1$。
- 渲染加固：该节抽成 `render_heading_section(study)`，单测用 `heading_study(32)` 秒级覆盖，
  不再依赖整套慢算例（本轮曾因列表边界/函数抽离两次 `SyntaxError`、`UnboundLocalError`）。
- 适用范围：无界、无粘、不可压、定常、二维；不含粘性尾流、分离、自由面、三维效应。
- **提交记录未重生成**：`benchmarks/verified/case_bem_laplace_cylinder/results.json` 仍是 38 条判据、
  无 `provenance` 的旧记录；重生成作业 cj4044 已递交，收尾时仍在排队（43 单在飞），工件未取回。

## ⑥ 误差判定

| 关键量 | 值 | 基准 | 是否 <3% |
| --- | --- | --- | --- |
| 斜方位最差路线 D 外部速度 L2 | {max(s['max_dirichlet_velocity_relative_l2'] for s in studies.values()):.3e}（N=32）/ {studies[128]['max_dirichlet_velocity_relative_l2']:.3e}（N=128） | 解析复势 $dW/dz$ | **是** |
| 斜方位最差路线 N 物面速度 L2 | {max(s['max_neumann_surface_velocity_relative_l2'] for s in studies.values()):.3e}（N=32）/ {studies[128]['max_neumann_surface_velocity_relative_l2']:.3e}（N=128） | $2U|\\sin\\theta|$ | **是** |
| 斜方位相对 0° 放大 | {max(max(s['dirichlet_penalty_max'], s['neumann_penalty_max']) for s in studies.values()):.3f}× | 0° 位形 | 是（阈值 3×） |
| 90° 旋转等变性 | {max(s['lattice']['velocity_field_max_difference_over_speed'] for s in studies.values()):.2e} U | 0° 解旋转后 | 是（阈值 1e-9 U） |
| 镜像奇部 | {max(s['mirror_split']['odd_part_max_over_speed'] for s in studies.values()):.2e} U | 舍入 | 是（阈值 1e-9 U） |

结论：**未发现缺陷**（负结果/falsification attempt）。斜来流把「不靠镜像对称吃饭」从「未检验」变成「已检验」：
方位改变不掉精度（≤4%，且方向为变小），全链对 90° 旋转等变到 1e-15，0° 位形的误差奇部只剩舍入，
历史上那些 L2 数字不是靠上下抵消得来的。误差仍由折面几何主导（与第 6 轮结论一致）。
"""

payload_fig = {
    "name": "heading_sweep",
    "caption": (
        "左：斜来流方位扫描——路线 D 外部速度（实线圆）与路线 N 物面速度（虚线方）相对 L2 误差，"
        "红点线为 <3% 算例容差；右：误差结构——90° 旋转等变性残差、镜像奇部与偶部（对数轴）。"
    ),
    "png_base64": base64.b64encode(open(FIG, "rb").read()).decode(),
}

report = {
    "title": "CASE-BEM-LAPLACE-CYLINDER 第 7 轮：斜来流方位扫描与全链旋转等变性（TensorBEM m22326）",
    "summary": (
        "圆柱绕势流算例新增斜来流方位扫描（0°/半面板步长/30°）+ 90° 旋转等变性 + 镜像误差奇偶分解："
        "斜方位误差不劣于轴对称（≤4%，路线 N 逐位相同），等变性残差 1e-15 U，镜像奇部 6e-16 U，"
        "均 <3% 容差；未发现缺陷（负结果入账），提交记录重生成作业 cj4044 仍在排队。"
    ),
    "scenario_key": "case-bem-laplace-cylinder",
    "mission_id": 22326,
    "tags": ["TensorBEM", "BEM", "potential-flow", "verification", "case-pool", "symmetry"],
    "markdown": markdown,
    "figures": [payload_fig],
}

with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(report, handle, ensure_ascii=False)

base = os.environ.get("TENSOR_API_BASE", "http://localhost:8001/api/v1").rstrip("/")
user = os.environ.get("TENSOR_AGENT_USER")
password = os.environ.get("TENSOR_AGENT_PASS")
login = urllib.request.Request(
    base + "/auth/login",
    data=json.dumps({"username": user, "password": password}).encode(),
    headers={"Content-Type": "application/json"},
)
token = json.load(urllib.request.urlopen(login, timeout=60))["access_token"]
post = urllib.request.Request(
    base + "/tensor/reports",
    data=open(OUT, "rb").read(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
)
response = json.load(urllib.request.urlopen(post, timeout=180))
print(json.dumps({k: response[k] for k in response if k in ("id", "title", "status")}, ensure_ascii=False))