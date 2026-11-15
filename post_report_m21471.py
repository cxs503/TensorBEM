"""Publish the m21471 report: docs markdown + platform report (figures inlined)."""
from __future__ import annotations

import base64
import json
import os
import pathlib
import subprocess

DATA = json.loads(
    pathlib.Path("benchmarks/verified/casepool_bem_cummins_wigley/anchor_audit.json").read_text()
)
DOCS = pathlib.Path("docs/m21471_cummins_anchor")
FIG = DOCS / "fig_m21471_cummins_anchor.png"
TOL = DATA["tolerances"]
SC = DATA["anchor_self_check"]
rows = DATA["per_dof_medium"]
verdicts = DATA["verdicts"]
conv = DATA["anchor_convergence"]
ladder = DATA["anchor_ladder"]


def pc(x, d=2):
    return f"{100.0 * x:.{d}f}%"


lines = []
A = lines.append
A("# CASE-BEM-CUMMINS-WIGLEY：$A_\\infty$ 的独立锚点与**零自由参数**的 Kramers-Kronig 闭合")
A("")
A("m21471 · platform 席自驱工时 · 仓 TensorBEM · 锚点 `CASE-BEM-CUMMINS-WIGLEY`（算例池 id=145，容差 <10%）")
A("")
A("## 1 案例介绍")
A("")
A("Wigley 数学船型（$L=3.0$ m, $B=0.3$ m, $T=0.1875$ m, $C_b=4/9$，排水体积 $0.075$ m³，"
  "$\\rho=1025$ kg/m³, $g=9.80665$ m/s²），6 自由度刚性辐射。算例判据来自 Cummins (1962) 时域卷积与 "
  "Kramers-Kronig 因果对：由 $B(\\omega)$ 反演 $A(\\omega)-A_\\infty$，容差 10%。")
A("")
A("**本单解决的验证欠账**：闸门 `C3` 里 $A_\\infty$ 是**在同一段被判的频带上最小二乘拟合出来的自由参数**。"
  "一个自由参数足以吸收常数偏置，于是「KK 闭合 <10%」这条验收判据可以被一个非物理的 $A_\\infty$ 满足。"
  "本单用**另一条边界值问题**（自由面高频极限 $\\phi=0$，反对称镜像精确闭合）给出 $A_\\infty$，"
  "把判据改成**零自由参数**版本（新闸门 `C3a`），并新增 `C7` 反过来量化被拟合的那个参数偏离物理值多远。")
A("")
A("## 2 怎么做的")
A("")
A("1. `tensorbem.added_mass_limit`（镜像 BEM，m21326）给出 6×6 的 $A_\\infty$；本单把它接进算例："
  "`added_mass_inf_anchor(n_sections, n_levels)` 用**与扫频相同的旋转中心** `CENTER=(0,0,0)` 求解"
  "（旋转中心不一致会让两列 $A_\\infty$ 相差一个刚体变换，比较作废——这是本轮踩到的第一个坑）。")
A("2. 新增 `C7`：$|A_\\infty^{\\text{fit}}-A_\\infty^{\\text{anchor}}|/\\max|A_{ii}| \\le 10\\%$，"
  "并以半浸球解析解 $A_\\infty(\\text{heave})=m_{\\text{full}}/4$ 对锚点自身做前置校验（误差须 <3%）。")
A("3. 新增 `C3a`：同一 KK 闭合机器（`kk_gate`，含截断预算判可判性窗口）但 $A_\\infty$ 由锚点**代入**，"
  "自由参数个数 0；原 `C3` 一字未动（判据不放宽、不替换）。")
A("4. 全部结果由**已提交的扫频工件**重放得到（`gates_from_artifact`），不重解 BEM：`scripts/case_cummins_wigley_anchor_audit.py`。")
A("")
A("## 3 计算结果")
A("")
A("### 3.1 锚点自校验（闭式真值）")
A("")
A("| 校验量 | 值 | 基准 | 误差 | 门限 |")
A("| --- | --- | --- | --- | --- |")
A(f"| 半浸球 $A_\\infty$(heave)，{SC['panels']} 面元 | {SC['added_mass_inf_heave']:.2f} kg | "
  f"闭式 {SC['added_mass_inf_exact']:.2f} kg | **{pc(SC['relative_error'])}** | {pc(SC['tolerance'],0)} |")
A("")
A("### 3.2 锚点网格阶梯（$A_\\infty$ 对角项）")
A("")
A("| DOF | " + " | ".join(f"{n}×{m} ({r['panel_count']} 面元)" for (n, m), r in zip([(12, 4), (20, 6), (28, 8)], ladder))
  + " | coarse→medium | medium→fine |")
A("| --- | --- | --- | --- | --- | --- |")
for name in ("surge", "sway", "heave", "roll", "pitch", "yaw"):
    vals = conv[name]["values"]
    A(f"| {name} | " + " | ".join(f"{v:.6g}" for v in vals) + f" | {pc(conv[name]['coarse_to_medium_relative_change'])} "
      f"| {pc(conv[name]['medium_to_fine_relative_change'])} |")
A("")
A("### 3.3 拟合 $A_\\infty$ vs 物理锚点，以及两种闭合残差（medium 网格）")
A("")
A("| DOF | $A_\\infty$ 拟合 | $A_\\infty$ 锚点 | $\\Delta/\\max|A_{ii}|$ | $\\Delta/|A_\\infty^{anchor}|$ | "
  "KK 残差（拟合，C3） | KK 残差（锚定，C3a） | 可判样本 |")
A("| --- | --- | --- | --- | --- | --- | --- | --- |")
for r in rows:
    A(f"| {r['dof']} | {r['added_mass_inf_fitted']:.6g} | {r['added_mass_inf_anchor']:.6g} | "
      f"{pc(r['relative_to_added_mass_scale'])} | {pc(r['relative_to_anchor'])} | "
      f"{pc(r['kk_residual_fitted'])} | **{pc(r['kk_residual_anchored'])}** | "
      f"{r['n_decidable']}/{r['n_samples']} |")
A("")
m = verdicts["medium"]
c = verdicts["coarse"]
A("### 3.4 判据结果")
A("")
A("| 闸门 | 网格 | 自由参数 | 结论 | worst DOF | 值 | 门限 |")
A("| --- | --- | --- | --- | --- | --- | --- |")
for label, v in (("medium", m), ("coarse", c)):
    A(f"| C3（原判据） | {label} | 1 | {'PASS' if v['C3']['passed'] else 'FAIL'} | {v['C3']['worst_dof']} | "
      f"{pc(v['C3']['worst_relative_residual'])} | 10% |")
    A(f"| **C3a（零自由参数，新）** | {label} | **0** | {'PASS' if v['C3a']['passed'] else 'FAIL'} | "
      f"{v['C3a']['worst_dof']} | **{pc(v['C3a']['worst_relative_residual'])}** | 10% |")
    A(f"| C7（$A_\\infty$ 锚点一致性，新） | {label} | — | {'PASS' if v['C7']['passed'] else 'FAIL'} | "
      f"{v['C7']['worst_dof']} | {pc(v['C7']['worst_relative_to_added_mass_scale'])} | 10% |")
A("")
A(f"medium 网格整案裁决：**{m['verdict']['gates_passed']}/{m['verdict']['gates_total']} 通过**，"
  f"failed = {m['verdict']['failed_list'] or '[]'}。")
A("")
A(f"![fig](fig_m21471_cummins_anchor.png)")
A("")
A("*左：拟合 $A_\\infty$ 与镜像锚点（symlog）；中：拟合 vs 锚定的 KK 残差与 10% 判据线；"
  "右：锚定闭合残差随频率的分布（medium 网格）。*")
A("")
A("## 4 注意点")
A("")
A("1. **旋转中心必须一致**：扫频以原点（水线面内）为旋转中心，首轮探针用 $(0,0,-T/2)$ 得到 roll 的 "
  "$A_\\infty$ 相差 105%——那不是物理差异而是刚体变换。修正后 roll 只差 2.7%。")
A("2. **截断预算仍是空窗**：surge/sway/yaw 的实测阻尼尾部指数 0.28–1.09（未进入代数衰减），"
  "上端未解频段的贡献无法界定，故这三条 DOF 的 `n_decidable=0`；它们的锚定残差 7.1%/7.6%/8.9% "
  "是**经验一致**而非带限数据蕴含。heave/roll/pitch 三条 12/12 可判。")
A("3. **余量薄**：锚定后 worst 由 3.6%（yaw）涨到 8.9%，离 10% 只剩 1.1 pp；最坏样本固定在 $\\omega=9.29$ rad/s"
  "（带顶附近），正是被截断尾部支配的样本。要把余量做厚必须解更高频段，而 $k\\sqrt{A}\\le1$ "
  "在 medium 网格上到 $\\omega\\approx15.5$ rad/s 就失守，$\\omega=20$ 需 $\\sqrt{A}\\le0.0245$ m "
  "（约 2000+ 面元）。")
A("4. **拟合 $A_\\infty$ 是坏估计量**：surge/sway/yaw 上比物理值高 21%/14%/15%（占带内幅值 4.4%/4.8%/5.3%）；"
  "而两参数尾拟合（$A_\\infty+c/\\omega^2$）在 surge/sway/yaw 上仍给出**负** $A_\\infty$（非物理），"
  "与 m21326 结论一致——带内截断使外推不可用，锚点法才是正解。")
A("5. 适用范围：本单只判**因果性/闭合**，不判 $A(\\omega)$ 的绝对精度；该算例的 L5 外部锚点"
  "（实验/解析辐射系数表）仍为 WITHHELD。")
A("")
A("## 5 误差判定（逐项，显式回答是否 <3%）")
A("")
A("| 关键量 | 基准 | 误差 | 是否 <3% |")
A("| --- | --- | --- | --- |")
A(f"| 半浸球 $A_\\infty$(heave)，锚点机器自校验 | 闭式 $m_\\text{{full}}/4$ | {pc(SC['relative_error'])} | "
  f"**是**（门限 3%） |")
A(f"| KK 闭合（零自由参数，C3a），worst DOF = {m['C3a']['worst_dof']} | 直接 BEM $A(\\omega)$（同路线） | "
  f"{pc(m['C3a']['worst_relative_residual'])} | **否**（>3%，但 <10% 算例判据） |")
A(f"| 同上，heave | 直接 BEM | {pc(rows[2]['kk_residual_anchored'])} | **是** |")
A(f"| 同上，roll | 直接 BEM | {pc(rows[3]['kk_residual_anchored'])} | **是** |")
A(f"| 同上，pitch | 直接 BEM | {pc(rows[4]['kk_residual_anchored'])} | **是** |")
A(f"| 拟合 $A_\\infty$ vs 锚点，worst = {m['C7']['worst_dof']} | 镜像 BEM 锚点 | "
  f"{pc(m['C7']['worst_relative_to_anchor'])}（相对锚点值） | **否**（估计量缺陷，已如实记入 C7） |")
A("")
A("结论：算例判据（<10%）在**零自由参数**下成立（worst 8.9%），但只有 heave/roll/pitch 三条 DOF "
  "同时满足 <3% 与截断可判；surge/sway/yaw 的 <10% 属经验一致。")
A("")
A("## 6 复现")
A("")
A("```bash")
A("# 全部由已提交的扫频工件重放（不重解 BEM），约 1 min")
A("PYTHONPATH=src python3 scripts/case_cummins_wigley_anchor_audit.py")
A("# 单元测试（锚点自校验、零自由参数不变量、C7 咬合、工件重放）")
A("PYTHONPATH=src python3 -m pytest tests/test_case_bem_cummins_wigley.py tests/test_added_mass_limit.py -q")
A("```")
A("")
A("改动文件：`src/tensorbem/benchmarks/case_bem_cummins_wigley.py`（锚点 + C3a/C7 + 旋转中心常量化 + 文档渲染）、"
  "`src/tensorbem/added_mass_limit.py`（半浸球解析体，供锚点自校验复用）、"
  "`tests/test_case_bem_cummins_wigley.py`（+5 测试）、"
  "`scripts/case_cummins_wigley_anchor_audit.py`、`benchmarks/verified/casepool_bem_cummins_wigley/anchor_audit.json`。")
markdown = "\n".join(lines)
DOCS.mkdir(parents=True, exist_ok=True)
(DOCS / "dev_report_m21471_cummins_anchor.md").write_text(markdown, encoding="utf-8")
pathlib.Path("dev_report_m21471_cummins_anchor.md").write_text(markdown, encoding="utf-8")

png = base64.b64encode(FIG.read_bytes()).decode()
report = {
    "title": "CASE-BEM-CUMMINS-WIGLEY：A∞ 独立锚点与零自由参数的 Kramers-Kronig 闭合（TensorBEM/m21471）",
    "summary": (
        "把 KK 判据里的自由参数 A∞ 换成镜像 BEM 的物理值，零自由参数下 6DOF 闭合 worst 8.9%（判据 10%）；"
        "同时量化出拟合 A∞ 在 surge/sway/yaw 上偏高 14–21%。"
    ),
    "scenario_key": "case_bem_cummins_wigley_anchor",
    "mission_id": 21471,
    "tags": ["TensorBEM", "Cummins", "Kramers-Kronig", "added-mass-limit", "Wigley", "verification"],
    "markdown": markdown,
    "figures": [
        {
            "name": "fig_m21471_cummins_anchor.png",
            "caption": "左：拟合 A∞ vs 镜像锚点；中：拟合/锚定 KK 残差与 10% 判据；右：锚定残差随频率。",
            "png_base64": png,
        }
    ],
}
pathlib.Path("/tmp/report.json").write_text(json.dumps(report), encoding="utf-8")

base = os.environ.get("TENSOR_API_BASE", "").rstrip("/")
user = os.environ.get("TENSOR_AGENT_USER", "")
pw = os.environ.get("TENSOR_AGENT_PASS", "")
token = subprocess.run(
    ["curl", "-s", "-X", "POST", f"{base}/auth/login",
     "-H", "Content-Type: application/json",
     "-d", json.dumps({"username": user, "password": pw})],
    capture_output=True, text=True,
).stdout
parsed = json.loads(token)
tok = parsed.get("access_token") or parsed.get("token")
print("login ok" if tok else f"login failed: {token[:200]}")
resp = subprocess.run(
    ["curl", "-s", "-X", "POST", f"{base}/tensor/reports",
     "-H", f"Authorization: Bearer {tok}",
     "-H", "Content-Type: application/json",
     "-d", "@/tmp/report.json"],
    capture_output=True, text=True,
).stdout
print("report:", resp[:600])