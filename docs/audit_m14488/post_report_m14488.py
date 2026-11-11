"""Publish the m14488 TensorBEM quality-audit report (numbers injected from artifacts)."""

from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
HERE = pathlib.Path(__file__).resolve().parent


def token() -> str:
    login = json.dumps({
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }).encode()
    req = urllib.request.Request(f"{BASE}/auth/login", data=login,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["access_token"]


inv = json.loads((HERE / "inventory.json").read_text())
p1 = json.loads((HERE / "probe_results_p1.json").read_text())["p1"]["cases"]
p2 = json.loads((HERE / "probe_results.json").read_text())["p2"]["rows"]
fdd = json.loads((pathlib.Path(HERE.parents[1], "docs/finite_depth_solver3d/results.json")).read_text())

# ---- inventory numbers (computed, never typed) -----------------------------
cats: dict[str, int] = {}
for merge in inv["merges"]:
    for f in merge["files"]:
        cats[f["category"]] = cats.get(f["category"], 0) + f["added"]
total = sum(cats.values())
wiring = inv["wiring"]
new_core = ["nk_panel3d", "finite_depth_solver3d", "flexural_gravity", "near_field_quadrature",
            "irregular_frequency", "drift_far_field3d", "memory_kernel_matrix", "state_space"]
core_wired = sum(1 for m in new_core if wiring[m]["core"])

cat_rows = "\n".join(
    f"| {k} | {v:,} | {100*v/total:.1f}% |" for k, v in sorted(cats.items(), key=lambda kv: -kv[1])
)
wire_rows = "\n".join(
    f"| `{m}` | {wiring[m]['wiring']} | {', '.join(wiring[m]['core'] or wiring[m]['bench'] or wiring[m]['tool']) or '—'} |"
    for m in new_core
)

# ---- probes ----------------------------------------------------------------
p1_rows = "\n".join(
    f"| {c['case']} | {c['reference_self_error_single']:.1e} | {c['adaptive_rel_err_single']:.1e} | "
    f"{c['polar_rel_err_single']:.1e} | {c['arbiter_rel_err_single']:.1e} |"
    for c in p1
)
p2_rows = "\n".join(
    f"| {r['k']:.3f} | {r['omega_flexural_gravity']:.4f} | {r['omega_hydroelastic']:.4f} | "
    f"{100*r['hydroelastic_rel_diff']:.2f}% | {r['cg_rel_err']:.1e} |"
    for r in p2
)
a1 = fdd["A1"]["ladder"][1]

md = f"""# TensorBEM 质量审计（AUDIT-ASSET-12REPO · m14488）

**对象**：`cxs503/TensorBEM`，审计 HEAD `481be53`（Merge #625），窗口 = `b17b169` 之后 **26 个已并 agent PR（#600–#625）**。
**唯一标准**：可验证的功能交付。**判级**：L0=文档、L1=数学/单元、L2=独立功能、L3=系统集成、L4=基准验证、L5=科研工程成果。

## ① 案例介绍（审计对象与口径）

不是物理算例，而是"能力账"审计：仓内 205 个测试文件 / 2,409 个测试函数 / 56,411 行新代码；
本审计回答四件事——资产怎么分类、每个核心功能是否真算/真接线/真有基准、成熟度落几级、哪些声明是膨胀。
审计环境：Python 3.11.6 / numpy 2.4.6 / scipy 1.16.0。**关键环境事实**：本机 `__editable__.tensorbem-0.5.0.pth`
指向 `/root/tp-agent/work/m14010/src`（另一个 worktree）——不带 `PYTHONPATH` 的 `import tensorbem` 读的是别的树，
故本审计全部证据均在 `PYTHONPATH=<本 worktree>/src` 下产生并打印模块路径断言。

## ② 怎么做的（方法链）

1. `inventory_scan.py`：git first-parent 解析 26 个 `Merge #N` → 逐文件分类（core/bench/test/doc/tool/artifact），
   AST 扫描 import 生成**接线层**（core/bench/tool/tests_only/orphan），AST 扫描占位符（`pass`/`...`/`NotImplementedError`）。
2. `probes.py`：三个**自建参考实现**的独立数值复核（不复用被审代码）：
   P1 近场求积三分支 vs 我自己的均匀细分+6 点规则；P2 三份 flexural 色散实现的数值分歧；
   P3 用仓内 `verify_*` 复现 #614 报告的门指标（含一次自我纠错）。
3. 定向实跑测试（每模块单跑，避免共享主机竞争导致的假结论），记录 passed/failed/skipped。
4. 逐 PR 读报告声明 → 与代码/工件/测试三方对表 → 判级 → 落 `tensor_maturity_verdicts`（16 行）。

## ③ 计算结果

**资产分类（{total:,} 新增行 / {len(inv['merges'])} 个 PR / {sum(len(m['files']) for m in inv['merges'])} 个文件）**

| 类别 | 新增行 | 占比 |
|---|---|---|
{cat_rows}

**接线审计：窗口内 8 个新核心模块中进入库主面（core 层）的数量 = {core_wired}**

| 新模块 | 接线层 | 消费者 |
|---|---|---|
{wire_rows}

**独立复核 P1：近场求积（单层核，第三方参考）**

| 案例 | 参考自身误差 | adaptive | polar | semi-analytic |
|---|---|---|---|---|
{p1_rows}

**独立复核 P2：两份冰板色散实现的分歧（同参数 h=1m, E=5GPa, ν=0.334, H=200m）**

| k (1/m) | flexural_gravity ω | hydroelastic ω | 相对差 | 群速度自洽误差 |
|---|---|---|---|---|
{p2_rows}

**#621 三维有限水深求解器 vs 仓内多极解析解（最细已发布网格 80 面元，h/a=20）**

| 量 | BEM | 解析 | 相对误差 |
|---|---|---|---|
| A33 | {a1['A33']:.2f} | {fdd['A1']['analytic']['A33']:.2f} | **{100*a1['rel_err_A33']:.2f}%** |
| B33 | {a1['B33']:.2f} | {fdd['A1']['analytic']['B33']:.2f} | **{100*a1['rel_err_B33']:.2f}%** |

脚本 `scripts/verify_finite_depth_solver3d.py:17` 承诺 A4=20/80/**320** 面元阶梯，`results.json` 中 **A4 不存在**；
报告 §6 只引 20 面元行（17.21%/53.78%），未披露 80 面元行。

**判级分布（16 行 verdict；L0=1, L1=9, L2=4, L3=1, L4=0, L5=0）**

| 级别 | 对象 |
|---|---|
| L3 | `finite_depth3d_offaxis_fix#624`（core 路径上的缺陷修复，7 passed） |
| L2 | `nk_panel3d#614`（38 passed；探针复现残差 4.53e-7 / 方向比 3.02）、`finite_depth_solver3d#621`（16 passed）、`near_field_quadrature#615`（14 passed 2 skipped；第三方参考 1.4e-13…1.7e-4）、`memory_kernel_matrix#601-603`（13 passed） |
| L1 | #618 #625 #622 #606 #608 #617 #602 #609 #623（公式/自洽极限/未跑完/自带红测试） |
| L0 | #610-611（文献锚转写，不产生新解） |

## ④ 结果图

图 1（左）：三条已交付求积路线与审计自建暴力参考的相对误差随目标高度的变化——三条曲线全部落在我参考自身的
误差地板上（h=1e-1 时 1.4e-13），说明该求积是真能力。图 1（右）：两份冰板色散实现随 k 分歧升至 14.18%，
而各自群速度自洽误差在 1e-10（即"错得一致"无法被自洽检查发现）。

## ⑤ 注意点（踩过的坑）

1. **证据污染**：本机 editable `.pth` 指向 m14010，不带 `PYTHONPATH` 的"绿"属于另一个分支。
2. **自我纠错**：P1 首轮得到"三条路线齐差 17.1%"的假警报，根因是我抄错 Dunavant 权重（权重和 0.8536）；归一后归零。
   P3 首轮按默认 `radiation='downstream'` 得方向比 0.986，看似与报告 3.0 冲突——报告用的是 `halfplane`，改后复现 3.02。
   ⇒ **复现不出不等于造假，先自查参数与参考实现**。
3. **仓级门是红的**：HEAD 上 `test_state_space.py` 5 failed（拟合最大误差 61.5% vs 15% 判据）、
   `test_ellipsoid_kochin_three_route.py::test_panels_source_bounded` failed（k=4.0 误差 65.2% vs 自设 20% 界）、
   `test_capability_evidence.py` 2 failed（能力台账与树漂移，25+ benchmark 模块无测试无理由）。
   PR 报告里的"X passed"是局部口径，不等于仓绿。
4. **发布工件不入仓**：`.gitignore:10 artifacts/` ⇒ #615 报告数字唯一来源不在仓内，fresh checkout 下 2/15 测试 SKIP。
5. **`docs/audit_baseline.md` 的 REAL 是结构性判据**（模块+触及测试+文档锚点可解析），不是数值判据；不可当精度背书。

## ⑥ 误差判定（逐项显式回答"是否 <3%"）

| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| #615 单层求积 h=1e-1 | 审计自建暴力求积 | 1.4e-13 | **是** |
| #615 单层求积 h=1e-3 | 同上 | 1.7e-4（参考自身误差 6.7e-4） | 是（在参考不确定度内） |
| #614 自由面条件残差 | 解析 G_xx+κG_z=0 | 4.53e-7 | **是** |
| #614 兴波阻力 | Michell 薄船 | −60×…+47×，随网格变号 | **否（honest FAIL，与作者自评一致）** |
| #621 A33（80 面元） | 仓内多极解析解 | {100*a1['rel_err_A33']:.2f}% | 否（略超 3%） |
| #621 B33（80 面元） | 仓内多极解析解 | {100*a1['rel_err_B33']:.2f}% | **否（>15% 不可接受线）** |
| #618 冰板色散 | 仓内另一实现 | 0.17%…14.18%（k=0.005…0.4） | 低波数 是 / 高波数 **否**（且无仲裁） |
| #625 V1–V10 | 同一闭式代数极限 | 1e-11…1e-16 | "是"但**无独立基准**，不计功 |

**honest-未完成**：全量测试套件与 `test_drift_far_field3d / test_irregular_frequency_sanitisation /
test_seakeeping_verification` 未跑完（共享主机竞争；集群作业 `cj834` 排队 40 min 未派发），相应模块判级压到 L1，
不宣称全绿。
"""

fig_b64 = base64.b64encode((HERE / "fig_audit.png").read_bytes()).decode()
body = {
    "title": "TensorBEM 质量审计（m14488）：26 个已并 PR 的资产盘点、六问审查与 L0–L5 判级",
    "summary": "审计 TensorBEM #600–#625 共 26 个 PR / 56,411 行新增：8 个新核心模块 0 个接入库主面；"
               "独立复核确认近场求积真实（1.4e-13）、#621 阻尼误差 19.6% 且承诺的 320 面元阶梯缺失；"
               "HEAD 上仓级门为红（state_space 5 失败、#622 自带红测试、能力台账漂移）；16 条判级落表，L4/L5=0。",
    "scenario_key": "audit-asset-12repo/tensorbem",
    "mission_id": 14488,
    "tags": ["audit", "TensorBEM", "quality", "maturity", "honest-fail"],
    "markdown": md,
    "figures": [{
        "name": "fig_audit.png",
        "caption": "左：近场求积三分支 vs 审计自建参考（全部落在参考误差地板）；右：两份冰板色散实现分歧随 k 升至 14.18%",
        "png_base64": fig_b64,
    }],
}

req = urllib.request.Request(
    f"{BASE}/tensor/reports",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {token()}"},
)
with urllib.request.urlopen(req, timeout=120) as resp:
    print(resp.status, resp.read().decode()[:400])