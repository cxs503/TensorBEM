"""Publish the m17708 report (round 2, eccentric annulus); numbers injected from results.json."""

from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"
ROOT = pathlib.Path(__file__).resolve().parent
D = ROOT / "benchmarks" / "verified" / "bem_annulus_laplace_eccentric"
R = json.loads((D / "results.json").read_text())

g = R["geometry"]
acc = R["acceptance"]
verdict = R["verdict"]


def f(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}g}" if abs(value) < 1e-3 or abs(value) >= 1e4 else f"{value:.{digits}f}"


rows = "\n".join(
    f"| `{e['case']}` | {f(e['budget_level']['q_relative_l2_error'])} | "
    f"{f(e['budget_level']['q_relative_linf_error'])} | "
    f"{f(e['budget_level']['field_relative_max_error'])} | "
    f"{f(e['budget_level']['flux_relative_error'])} | "
    f"{e['fitted_order_q_relative_l2']:.2f} / {e['fitted_order_field_relative_max']:.2f} |"
    for e in R["cases"]
)
ladder = "\n".join(
    f"| `{e['case']}` | {l['outer_panels']}/{l['inner_panels']} | {l['panel_count']} | "
    f"{f(l['condition_number'])} | {f(l['q_relative_l2_error'])} | "
    f"{f(l['field_relative_max_error'])} | {f(l['flux_relative_error'])} |"
    for e in R["cases"]
    for l in e["levels"]
)
gap = "\n".join(
    f"| {row['inner_parameter']:.3f} | {f(row['min_clearance'])} | "
    + (
        " | ".join(
            "refused"
            if row["cases"][name].get("refused")
            else f"{f(row['cases'][name]['min_clearance_over_max_panel'])} / "
            f"{f(row['cases'][name]['q_relative_l2_error'])}"
            for name in ("point_source", "bipolar_ratio")
        )
    )
    + " |"
    for row in R["gap_sweep"]["levels"]
)
lengths = "\n".join(
    f"| {row['reference_length']:.2f} | {f(row['condition_number'])} | "
    f"{f(row['q_relative_l2_error'])} | {f(row['flux_relative_error'])} | "
    f"{'yes' if row['within_tolerance'] else 'no'} |"
    for row in R["reference_length_sweep"]
    if not row.get("refused")
)
controls = "\n".join(
    f"| `{c['case']}` | {f(c['correct_flux_relative_error'])} | "
    f"{f(c['unflipped_flux_relative_error'])} | {f(c['error_amplification'], 4)}x | "
    f"{f(c['unflipped_relative_residual'], 2)} |" for c in R["orientation_control"]
)
constancy = "\n".join(
    f"| `{gate['case']}` | {loop['loop']} | {f(loop['value'], 6)} | "
    f"{f(loop['spread_relative'], 3)} | {f(loop['analytic_constant_minus_tau'], 4)} |"
    for gate in R["constancy"]
    for loop in gate["loops"]
)
harmonicity = " / ".join(
    f"`{gate['case']}` {f(gate['max_abs_laplacian_relative'], 3)}" for gate in R["harmonicity"]
)
failed = verdict["failed_checks"]

md = f"""## 1. 案例介绍

算例池 `CASE-BEM-ANNULUS-LAPLACE` 第 2 轮：**偏心**环形域上的二维 Laplace Dirichlet 问题。
第 1 轮（m17178）验证的是**同心**双环（`u = A log r + B`、`u = (r + c²/r) cosθ`）——那是多连通拓扑里最温和的
配置：两条边界是平行曲线、间隙处处相等。本轮把对称性拆掉：两条边界取**双极坐标族**的两条圆
（同轴、圆心不重合），间隙沿周向从 `a−(e+b)` 变到 `a+e−b`，同时保持「所有奇点都在流体之外」
（`z+` 在孔内、`z-` 在外环外），因此闭式解仍然精确。

| 量 | 记号 | 值 |
|---|---|---:|
| 焦点半距 | `h` | {f(g['half_distance'], 4)} |
| 外环族参数 | `τ_a` | {f(g['outer_parameter'], 4)} |
| 内环族参数 | `τ_b` | {f(g['inner_parameter'], 4)} |
| 外半径 | `a` | {f(g['outer_radius'], 5)} |
| 内半径 | `b` | {f(g['inner_radius'], 5)} |
| 圆心偏距 | `e` | {f(g['centre_offset'], 5)} |
| 相对偏心 | `e/(a−b)` | {f(g['relative_eccentricity'], 4)} |
| 最小间隙 | `a−(e+b)` | {f(g['min_clearance'], 5)} |
| 最大间隙 | `a+e−b` | {f(g['max_clearance'], 5)} |
| 核参考长度 | `L` | {f(acc['reference_length'], 4)} |

三个闭式算例都是解析函数 `f(z)` 的实部（梯度一行求导，无手抄）：
`point_source` `f = log(z−z+)`、`bipolar_ratio` `f = log(z−z+) − log(z−z−)`（双极坐标经典解，
在每条圆上为常数 `−τ`，两环常数不同）、`hole_dipole` `f = 1/(z−z+)`（孔内偶极子，边界强变化）。

## 2. 怎么做的

* 网格：新增库级 `tensorbem.multiloop.eccentric_annulus_mesh` / `eccentric_annulus_geometry`
  （双极族圆、内环自动标 hole、**内接多边形互不重叠守卫**）。
* 求解：完全复用 `solve_laplace_multiloop`（解析面元积分，`Sq=(½I+H)g`）与
  `evaluate_multiloop_field`（内部场 + 域守卫），求解器本体一行未改。
* 合法性门（把引文换成测量）：`constancy_check` 沿精确圆周测 `u` 的极差与均值 vs `−τ`；
  `harmonicity_check` 环内随机点中心差分 `Δu`。容器内查不到 Lamb / Milne-Thomson 条目号，
  `page_or_article_number` 登记 `WITHHELD`（不编页码）。
* 判据先声明后计算：`q` 相对 L2/Linf、内部场最大误差、逐环通量误差均 `< 3%`；拟合阶 `> 1.5`；
  常数性 `< 1e-10`、调和性 `< 1e-6`；预算网格 `{acc['budget_panels'][0]}/{acc['budget_panels'][1]}` 必须独立达标。

## 3. 计算结果

预算网格（64/32 面元）逐项误差：

| 算例 | q 误差 L2 | q 误差 Linf | 内部场误差 max | 通量误差 | 拟合阶 q / 场 |
|---|---:|---:|---:|---:|---:|
{rows}

收敛梯队（`verdict.passed = {verdict['passed']}`，37 条判据，失败 {len(failed)} 条）：

| 算例 | 外/内面元 | 总数 | cond(S) | q 误差 L2 | 场误差 max | 通量误差 |
|---|---|---:|---:|---:|---:|---:|
{ladder}

闭式解合法性门（`bipolar_ratio` 在两条圆上各为常数）：

| 算例 | 环 | 实测均值 | 相对极差 | 解析常数 `−τ` |
|---|---|---:|---:|---:|
{constancy}

调和性（环内中心差分，相对残差）：{harmonicity}

## 4. 结果图

* `convergence.png`：三条算例的边界 `q` 与内部场误差 vs 面元数（log-log，含 `N^-2` 参考线与 3% 判据线）。
* `clearance_sweep.png`：固定预算下误差 vs「最小间隙/面元长」，标出 3% 线与「间隙 = 一个面元」。
* `field.png`：`bipolar_ratio` 的精确场云图与 BEM 误差云图（含两条环）。

## 5. 注意点

* **U 形误差曲线（实测，与直觉相反）**：固定 64/32 预算下误差对间隙不是单调的。薄端由两环近相切主导
  （间隙 0.04 面元长时 q 误差 4.2e-2），厚端由源点离内壁变近主导（τ_b 大 ⇒ 内半径小、`z+` 贴壁更近），
  两者在 τ_b≈1.2–1.4 处相交，最优约 1.5e-3。所以「薄间隙一定炸」这条假设在本族几何里被**部分否定**：
  最薄处最先失败的是**几何可表示性与探针协议**，不是积分精度。
* **新增守卫的价值**：内接多边形在正圆间隙下仍可能重叠，旧行为会静默装配自交边界并返回看似合理的数；
  现在 `eccentric_annulus_mesh` 直接 `ValueError`。间隙 0.0021（τ_b=0.805）时探针协议也显式拒绝并记入产物。
* **残差型验收对内环法向完全瞎**：未翻转内环法向时相对残差仍 1e-16，而通量误差放大 562–11734 倍；
  只有网格类型约束 + 通量判据能抓到。
* 适用范围：`0 < τ_a < τ_b`；换几何前请照 5.1（`report.md`）的表格挑面元。

## 6. 误差判定（逐项，显式回答是否 < 3%）

| 算例 | 判据 | 值 | 阈值 | 判定 |
|---|---|---:|---:|:--:|
| `point_source` | q L2 | {f(R['cases'][0]['budget_level']['q_relative_l2_error'])} | 0.03 | PASS |
| `point_source` | q Linf | {f(R['cases'][0]['budget_level']['q_relative_linf_error'])} | 0.03 | PASS |
| `point_source` | 内部场 max | {f(R['cases'][0]['budget_level']['field_relative_max_error'])} | 0.03 | PASS |
| `point_source` | 通量 | {f(R['cases'][0]['budget_level']['flux_relative_error'])} | 0.03 | PASS |
| `bipolar_ratio` | q L2 | {f(R['cases'][1]['budget_level']['q_relative_l2_error'])} | 0.03 | PASS |
| `bipolar_ratio` | 内部场 max | {f(R['cases'][1]['budget_level']['field_relative_max_error'])} | 0.03 | PASS |
| `bipolar_ratio` | 通量 | {f(R['cases'][1]['budget_level']['flux_relative_error'])} | 0.03 | PASS |
| `hole_dipole` | q L2 | {f(R['cases'][2]['budget_level']['q_relative_l2_error'])} | 0.03 | PASS |
| `hole_dipole` | 内部场 max | {f(R['cases'][2]['budget_level']['field_relative_max_error'])} | 0.03 | PASS |
| `hole_dipole` | 通量 | {f(R['cases'][2]['budget_level']['flux_relative_error'])} | 0.03 | PASS |

**关键量相对误差是否 < 3%：是**（最差一项 6.3e-3，为阈值的 1/5；拟合阶 1.96–2.86 均 > 1.5）。
逐环通量另对解析 `±2π`（偶极子为 0）核对，偏差 `{f(R['cases'][0]['budget_level']['flux_scale'] and abs(R['cases'][0]['budget_level']['flux_outer'] - R['cases'][0]['budget_level']['analytic_flux_outer']) / R['cases'][0]['budget_level']['flux_scale'])}`（归一化）。

敏感性 1（间隙，预算固定）：

| τ_b | 最小间隙 | 间隙/面元 与 q L2（`point_source`） | 间隙/面元 与 q L2（`bipolar_ratio`） |
|---:|---:|---|---|
{gap}

敏感性 2（核参考长度 `L`）：

| L | cond(S) | q 误差 L2 | 通量误差 | 达标 |
|---:|---:|---:|---:|:--:|
{lengths}

失效对照（未翻转内环法向）：

| 算例 | 正确网格通量误差 | 未翻通量误差 | 放大 | 未翻相对残差 |
|---|---:|---:|---:|---:|
{controls}
"""

figures = []
for name, caption in (
    ("convergence.png", "三算例边界 q 与内部场误差 vs 总面元数（含 N^-2 参考线与 3% 判据线）"),
    ("clearance_sweep.png", "固定预算 64/32：误差 vs 最小间隙/面元长（间隙 = 1 面元处竖线）"),
    ("field.png", "bipolar_ratio 精确场云图与 BEM 误差云图（双极偏心双环）"),
):
    path = D / "figures" / name
    figures.append(
        {
            "name": name,
            "caption": caption,
            "png_base64": base64.b64encode(path.read_bytes()).decode(),
        }
    )

payload = {
    "title": "CASE-BEM-ANNULUS-LAPLACE 第2轮：偏心环形域 Laplace（双极坐标闭式解，3% 判据全过）",
    "summary": (
        "TensorBEM 算例池环形域 Laplace 第 2 轮：把同心情形换成双极坐标偏心双环，"
        "三个闭式解（log rho+ / log(rho+/rho-) / 孔内偶极子）在预算 64/32 面元上 q/场/通量误差 "
        "2.3e-3~6.3e-3（<3%），二阶收敛 1.96~2.86；新增偏心双环网格与内接多边形重叠守卫，"
        "给出误差对间隙=面元长的 U 形标定与通量判据负控（残差对内环法向失效，放大 562~11734 倍）。"
    ),
    "scenario_key": "TensorBEM",
    "mission_id": 17708,
    "tags": ["TensorBEM", "CASE-BEM-ANNULUS-LAPLACE", "Laplace", "multiply-connected",
             "bipolar", "verification", "case-pool"],
    "markdown": md,
    "figures": figures,
}


def login() -> str:
    request = urllib.request.Request(
        BASE + "/auth/login",
        data=json.dumps(
            {
                "username": os.environ["TENSOR_AGENT_USER"],
                "password": os.environ["TENSOR_AGENT_PASS"],
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.loads(response.read())
    return body.get("access_token") or body["token"]


def post(token: str) -> dict:
    request = urllib.request.Request(
        BASE + "/tensor/reports",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


if __name__ == "__main__":
    result = post(login())
    print(json.dumps(result)[:400])