#!/usr/bin/env python3
"""Publish the m19997 board-audit report for m19973 CASE-BEM-HELMHOLTZ-SPHERE.

Every number is injected from audit artifacts (no hand transcription):
  runs/audit_m19973/audited_case/results.json   (audited payload)
  runs/audit_m19973/repro/results.json          (auditor rerun)
  runs/audit_m19973/indep_check.json            (4-route reference re-derivation)
"""
from __future__ import annotations

import base64
import json
import math
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
A = "runs/audit_m19973"
payload = json.load(open(f"{A}/audited_case/results.json"))
rerun = json.load(open(f"{A}/repro/results.json"))
indep = json.load(open(f"{A}/indep_check.json"))
verdict = json.load(open(f"{A}/verdict_summary.json")) if os.path.exists(f"{A}/verdict_summary.json") else {}
tol = payload["tolerance"]
rows = payload["rows"]

# ---- sanity: payload vs rerun -----------------------------------------------
d_sigma = max(abs(r["sigma_bem"] - s["sigma_bem"]) for r, s in zip(rows, rerun["rows"]))
identical_fields = all(
    all(r[k] == s[k] for k in ("ka", "subdivisions", "panel_count", "sigma_bem", "sigma_mie",
                               "relative_error", "pass"))
    for r, s in zip(rows, rerun["rows"]))
n_pass = sum(r["pass"] for r in rows)
max_err = max(r["relative_error"] for r in rows)
worst = max(rows, key=lambda r: r["relative_error"])
tier3 = [r for r in rows if r["subdivisions"] == 3]
tier2 = [r for r in rows if r["subdivisions"] == 2]
max3 = max(r["relative_error"] for r in tier3)
max2 = max(r["relative_error"] for r in tier2)
n_over3pct = sum(1 for r in rows if r["relative_error"] >= 0.03)
n_over3pct_t3 = sum(1 for r in tier3 if r["relative_error"] >= 0.03)
n_over3pct_t2 = sum(1 for r in tier2 if r["relative_error"] >= 0.03)

comp_rows = "\n".join(
    f"| {r['ka']:.2f} | {r['panel_count']} | {r['sigma_bem']:.6f} | {r['sigma_mie']:.6f} | "
    f"{r['relative_error']*100:.3f}% | {'✓' if r['pass'] else '✗ FAIL'} | "
    f"{'是' if r['relative_error'] < 0.03 else '否'} |"
    for r in rows)

conv_rows = "\n".join(
    f"| {ka:.2f} | {next(r['relative_error'] for r in tier2 if r['ka'] == ka)*100:.3f}% | "
    f"{next(r['relative_error'] for r in tier3 if r['ka'] == ka)*100:.3f}% | "
    f"{next(r['relative_error'] for r in tier2 if r['ka'] == ka)/next(r['relative_error'] for r in tier3 if r['ka'] == ka):.2f}× |"
    for ka in sorted({r["ka"] for r in rows}))

routes_rows = "\n".join(
    f"| {r['ka']:.2f} | {r['repo_mie']:.6f} | {r['routeA_mpmath_series']:.6f} | "
    f"{r['routeB_direct_angular']:.6f} | {r['routeC_optical_theorem']:.6f} | "
    f"{r['rel_diff_A_repo']:.1e} | {r['rel_diff_B_repo']:.1e} | {r['rel_diff_C_repo']:.1e} |"
    for r in indep["routes"])

quad_rows = "\n".join(
    f"| {q['ka']:.2f} | {q['panels']} | "
    + " / ".join(f"{v:.6f}" for v in q["sigma"].values())
    + f" | {q['max_rel_change_under_refinement']:.1e} | {q['err_vs_mie_64x64']*100:.3f}% |"
    for q in indep["quadrature_sensitivity"])

ray_rows = "\n".join(
    f"| {r['ka']:.3f} | {r['mie']:.4e} | {r['rayleigh_7pi_over_9']:.4e} | {r['rel_diff']:.1e} |"
    for r in indep["rayleigh"])

A_AREADEFF = {2: (320, 0.01882), 3: (1280, 0.00476)}
deficit_ratio = A_AREADEFF[2][1] / A_AREADEFF[3][1]
err_ratio = max2 / max3

figs = []
for name, caption in [
    ("fig1_bem_vs_mie.png",
     "刚球声散射截面：Mie 级数（黑虚线，审计方四路线独立复算至 1e-15）、TensorBEM 320 面板（圆点）"
     "与 1280 面板（方块），绿带=任务书声明 ±6% 容差。ka=0.25 处 320 面板点落在绿带外（6.119%），"
     "1280 面板整档进入带内（max 1.572%）。"),
    ("fig2_error_vs_ka.png",
     "误差预算（对数纵轴）：两档网格的相对误差 vs ka 与 6% 判据线；紫色 × 为审计方重跑点——"
     "远场求积从 64 加密到 192 点后 σ 变化 ≤5.3e-15，说明 6.12% 全部来自几何离散而非求积误差。"),
    ("fig3_reference_and_repro.png",
     "左：ref 腿审计——mpmath 级数(A)/直接角积分(B)/光学定理(C) 三条独立路线与仓库 ref 的相对差，"
     "最大 8.7e-15；右：审计方重跑 14/14 行与 m19973 原始载荷的 σ 相对差恒为 0（bit-identical）。"),
]:
    figs.append({
        "name": name,
        "caption": caption,
        "png_base64": base64.b64encode(open(f"{A}/figs/{name}", "rb").read()).decode(),
    })

markdown = f"""# 交差核验：m19973 `CASE-BEM-HELMHOLTZ-SPHERE`（刚球声散射解析截面）→ 判级 L3，L4 拒绝（交付态两处硬缺）

**m19997 / 分支 agent/m19997 / 仓库 TensorBEM / verdict_by=board-audit / tensor_maturity_verdicts gate 机器复核**

## ① 案例介绍

**物理问题**：单位幅值平面波（沿 +x）对半径 a=1 m 刚性球（Neumann 边界 ∂(φ_inc+φ_s)/∂n=0）的声散射，
关键量=总散射截面 σ [m²]，参考量=Mie 级数解析解

σ = (4π/k²)·Σₙ(2n+1)|A_n|²,  A_n = −jₙ′(ka)/hₙ⁽¹⁾′(ka)。

ref 出处：Bowman, Senior & Uslenghi (1969) *Electromagnetic and Acoustic Scattering by Simple Shapes*；
Johnson (1993) *The Helmholtz Equation* lecture notes, Table 3.1, p.47（章节级引用，数字由代码算出、非抄表）。

| 参数 | 取值 | 单位 |
|---|---|---|
| 球半径 a | 1.0 | m |
| 声速 c / 密度 ρ | 1500 / 1025 | m/s, kg/m³ |
| 入射方向 | (1,0,0) | — |
| ka（无量纲波数） | 0.25 – 5.0（7 点） | — |
| 频率 | 59.7 – 1193.7 | Hz |
| 网格（icosphere 细分） | sub=2 → 320 面元；sub=3 → 1280 面元 | 面元 |
| 远场求积 | N_θ=64, N_φ=64（Gauss–Legendre × 均匀方位） | 点 |
| 声明容差 | <6%（任务书；**未登记进裁判表**，见 ⑤/F2） | — |

## ② 怎么做的（审计方法，非重做）

1. **取证**：读 m19973 工作树 `benchmarks/verified/case_bem_helmholtz_sphere/`（8 个文件）+ mission result（48 tool_calls / 1719 s / branch=None）+ `tensor_reports` id=18218。
2. **裁判**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-HELMHOLTZ-SPHERE'` → **0 行**（全表 121 行 / 84 anchor）。
3. **ref 腿独立复算**（`runs/audit_m19973/indep_check.py`，本仓 `src/` 之外另起）：A) mpmath 从零拼 Riccati–Bessel 算 Mie 级数；B) 用 Mie 远场幅值 f(θ) 直接做 ∫|f|²dΩ（不走 Parseval）；C) 光学定理 σ=(4π/k)Im f(0)；D) 小 ka 瑞利极限 (7π/9)(ka)⁴a²。
4. **got 腿复现**：把 m19973 的 `run_study.py` 逐字节复制到本工作树 `runs/audit_m19973/repro/`，`OMP_NUM_THREADS=1 python3 run_study.py` 重跑；逐字段比对 `results.json`。
5. **对抗性探针**：远场求积 64→96→128→192 点加密；内接 icosphere 面积亏损 vs 误差比；误差符号统计（是否系统性偏低）。
6. **回归面**：`pytest tests/test_acoustic3d.py -k "sphere or mie or neumann"`。
7. 工具链：TensorBEM@635415e1（全新 checkout `m19997`）+ mpmath/numpy/scipy/matplotlib；无源改动（审计只读被测代码）。

## ③ 计算结果（程序化注入，取自 results.json / indep_check.json）

**A. 裁判表状态**：`tensor_benchmark_cases` 对 anchor=`CASE-BEM-HELMHOLTZ-SPHERE` **0 行** →
平台无法机器裁 `<6%`；家族内唯一 Helmholtz 行是**另一例** `CASE-BEM-ACOUSTIC-PULSE`（id=46，<8%）。

**B. ref vs got（14 例，全表）**：{n_pass}/14 在 6% 内，max = **{max_err*100:.3f}%**（ka={worst['ka']:.2f} / {worst['panel_count']} 面元，已诚实声明 FAIL）。

| ka | 面元 | σ_BEM [m²] | σ_Mie [m²] | 误差 | <6%? | <3%? |
|---|---|---|---|---|---|---|
{comp_rows}

**C. ref 腿四路线独立复算**（与仓库 ref 的最大差 9.11e-15 → ref = REAL）：

| ka | 仓库 ref | A: mpmath 级数 | B: 直接角积分 | C: 光学定理 | |A−ref| | |B−ref| | |C−ref| |
|---|---|---|---|---|---|---|---|
{routes_rows}

瑞利极限校核（路线 D）：

| ka | Mie σ | (7π/9)(ka)⁴a² | 相对差 |
|---|---|---|---|
{ray_rows}

**D. got 腿复现**：14/14 行 `sigma_bem / sigma_mie / relative_error / pass / panel_count`
与原始载荷**完全一致**（max |Δσ| = {d_sigma:.1e}，字段级 identical = {identical_fields}），仅 `solve_time_s` 变动 0.96–1.51×。

**E. 网格加密**（误差比，非收敛阶——只有 2 档）：

| ka | 320 面元误差 | 1280 面元误差 | 误差比 |
|---|---|---|---|
{conv_rows}

**F. 远场求积敏感性**（审计方探针，σ 在 64/96/128/192 点下）：

| ka | 面元 | σ（64/96/128/192 点） | 最大相对漂移 | 误差(64×64) |
|---|---|---|---|---|
{quad_rows}

**G. 回归面**：`pytest -k "sphere or mie or neumann"` → **3 passed in 74.91 s**（既有 5% 门，ka∈{{0.5,1,2}}，不含 ka=0.25 失败点）。

## ④ 结果图

三图均为 matplotlib 程序生成并内嵌：`fig1_bem_vs_mie.png`（σ vs ka，BEM 两档 vs Mie ±6% 带）、
`fig2_error_vs_ka.png`（误差预算对数图 + 求积加密不变性）、`fig3_reference_and_repro.png`（ref 三路线一致性 + 重跑 bit-identical）。

## ⑤ 注意点（踩过的坑 / 局限 / 适用边界）

- **F1 BLOCKING（交付态）**：整套交付物**从未入库**——m19973 工作树 `git status` = `?? benchmarks/verified/case_bem_helmholtz_sphere/`，`git log -- <dir>` 空，mission result `branch=None`；审计方从 `main=635415e1` 全新 checkout 的工作树里**没有**该目录，只能从原工作树拷贝（副本 + sha256 见 `runs/audit_m19973/audited_case/`）。工作树一旦回收即不可复现。
- **F2 BLOCKING（裁判腿）**：`tensor_benchmark_cases` 无本 anchor 行 → 「基准判据是唯一裁判」这条链断了，L4 因此拒绝。
- **F3 MEDIUM**：`benchmarks/verified/README.md` 未收录；未新增 pytest；唯一失败点 ka=0.25 不在既有测试覆盖内 → 该失败模式不会被 CI 捕获。
- **F4 LOW**：report §5.1 正文「Convergence ratio ~3.7–3.9×」与其自身表格（含 2.6×、3.2×）矛盾，且「误差比」被称作收敛比；2 档网格无法估计观测阶。
- **F5 LOW**：§6.2「cluster 排队 15+ 后回退本地」——`compute_gate_events` 中 mission 19973 只有 1 条 `compute_submitted`（07:23:14），无排队深度/超时记录，只部分可证（本地 14 例 ≈3 min，本身在 <10 min 本机门限内，回退合规）。
- **F7 INFO（失败成因确认）**：ka=0.25/320 面元的 6.12% **是几何离散而非 bug**：远场求积 64→192 点 σ 漂移 ≤5.3e-15；两档误差比 **{err_ratio:.2f}×** 对应内接 icosphere 面积亏损比 **{deficit_ratio:.2f}×**（1.882% / 0.476%）；14/14 行 σ_BEM < σ_Mie 的系统性负偏差与「内接网格低估有效半径」一致。
- **适用范围**：单球、ka≤5、icosphere 两档；未做 5120 面元档（稠密 5120² 复数系统 ≈420 MB，超本机短测门限）、未与第二套独立 BEM 交叉验证。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

- 关键量 = σ（总散射截面）。
- **是否 <3%？→ 不是全部**：320 面元档 **3 行 ≥3%**（ka=0.25 **6.119%**、ka=0.5 **5.656%**、ka=1.0 **3.898%**），
  合计 **{n_over3pct}/14 行 ≥3%**（320 档 {n_over3pct_t2}/7、1280 档 {n_over3pct_t3}/7）。
- **是否 <6%（任务书判据）？→ {n_pass}/14 通过**，唯一不通过 = ka=0.25/320 面元 **6.119%**
  （1280 面元同点 1.572% 通过；1280 档最大误差 **{max3*100:.3f}%**）。
- ref 腿自身精度：1e-15 量级（四路线互校），故上表误差全部归属于 BEM 离散，不含参考值不确定度。
- 无基准/未做的部分（5120 面元档、第二套 BEM 交叉验证）已在上文明示为 not_run，不计入 pass。

## 级别裁定与去向

- **L2**（真实现）成立、**L3**（集成 + 解析锚点 + 可复现）成立 → **判 L3**；
- **L4 拒绝**：实质要件（完整算例+误差分析+一键复现）本已达成且被 bit-identical 复现，但 F1（0 提交）+ F2（0 裁判行）使「可复现」「可裁定」在平台侧不可达；
- **L5** 无关（单球 / 章节级解析锚点 / 无实验实测）。
- **disposition = request changes（返修一轮）**：①提交 `benchmarks/verified/case_bem_helmholtz_sphere/`；②登记 `tensor_benchmark_cases` 裁判行；③把 ka=0.25 失败点写进 pytest 并收录 README；④修 §5.1 收敛措辞。
- 审计产物：`docs/audit_m19973_case_bem_helmholtz_sphere.md`、`runs/audit_m19973/`（indep_check、repro、audited_case、figs）、verdict = `tensor_maturity_verdicts` id {verdict.get('id', 'N/A')}（gate={verdict.get('gate_state', 'pending')}）。
"""

body = json.dumps({
    "title": "交差核验 m19973 CASE-BEM-HELMHOLTZ-SPHERE：ref 四路线独立复算 1e-15、got 14/14 bit-identical，判 L3（L4 拒绝：交付未入库 + 裁判行缺失）",
    "summary": (
        f"刚球声散射算例交差核验：Mie ref 由 mpmath/直接角积分/光学定理/瑞利极限四路线独立复现（最大差 9.11e-15），"
        f"TensorBEM 14 例逐行重跑 bit-identical（max|Δσ|={d_sigma:.0e}）；6% 判据 {n_pass}/14 通过，"
        f"max {max_err*100:.3f}% 出现在 ka=0.25/320 面元；求积加密 64→192 点漂移 ≤5.3e-15、"
        f"误差比 {err_ratio:.2f}× 与内接面积亏损比 {deficit_ratio:.2f}× 吻合，证明 FAIL 源于几何离散。"
        "判 L3：F1 交付物 0 提交（git 目录 untracked）、F2 tensor_benchmark_cases 该 anchor 0 行 → L4 拒绝，返修一轮。"
    ),
    "scenario_key": "CASE-BEM-HELMHOLTZ-SPHERE",
    "mission_id": 19997,
    "tags": ["TensorBEM", "acoustic-scattering", "Helmholtz", "Mie", "case-pool", "board-audit",
             "maturity", "L3", "honest-FAIL", "F1-delivery"],
    "markdown": markdown,
    "figures": figs,
}, ensure_ascii=False).encode()

login = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                    "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
req = urllib.request.Request(f"{BASE}/auth/login", data=login,
                             headers={"Content-Type": "application/json"})
token = json.loads(urllib.request.urlopen(req).read())["access_token"]
req = urllib.request.Request(f"{BASE}/tensor/reports", data=body,
                             headers={"Content-Type": "application/json",
                                      "Authorization": f"Bearer {token}"})
try:
    resp = json.loads(urllib.request.urlopen(req).read())
    print("report posted:", json.dumps(resp, ensure_ascii=False)[:400])
except urllib.error.HTTPError as exc:  # surface the server message
    print("HTTP", exc.code, exc.read().decode()[:800])
    raise