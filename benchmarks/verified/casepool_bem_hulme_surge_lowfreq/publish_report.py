#!/usr/bin/env python3
"""Publish the platform progress report for CASE-BEM-HULME-SURGE-LOWFREQ.

Every number is injected from ``results.json`` (no hand-copied digits) and the
figure is embedded from ``figure.png``.

    PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_surge_lowfreq/publish_report.py
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MISSION_ID = int(os.environ.get("MISSION_ID", "19925"))


def _login(base: str) -> str:
    body = json.dumps({
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }).encode()
    req = urllib.request.Request(f"{base}/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())["access_token"]


def _fmt(value, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def build_markdown(payload: dict) -> str:
    pc = payload["physical_case"]
    ex = payload["exact_anchor"]
    extrap = payload["extrapolation"]
    harness = payload["doubled_body_self_test"]
    gates = payload["gates"]
    ref_rows = sorted(payload["reference_ladder"], key=lambda r: r["ka"])
    mesh_rows = payload["solver_mesh_ladder"]
    ka_rows = sorted(payload["solver_ka_ladder"], key=lambda r: r["ka"])
    near_band = min(ref_rows, key=lambda r: abs(r["ka"] - 0.2))
    band_excess = 100.0 * (near_band["a11_dimless"] / 0.5 - 1.0)

    ref_table = "\n".join(
        f"| {r['ka']:.3f} | {_fmt(r['a11_dimless'])} | {_fmt(r['excess_over_exact'])} | "
        f"{_fmt(r['excess_per_ka'])} | {_fmt(r['b11_dimless'])} | {_fmt(r['a33_dimless'])} |"
        for r in ref_rows
    )
    mesh_table = "\n".join(
        f"| {r['subdivisions']} | {r['panels']} | {_fmt(r['added_mass_dimless'])} | "
        f"{100 * r['rel_error_vs_exact']:+.3f}% | {_fmt(r['condition_number'], 3)} | "
        f"{r['relative_residual']:.2e} |"
        for r in mesh_rows
    )
    ka_table = "\n".join(
        f"| {r['ka']:.4f} | {r['panels']} | {_fmt(r['added_mass_dimless'])} | "
        f"{100 * r['rel_error_vs_exact']:+.3f}% | {_fmt(r['damping_dimless'])} |"
        for r in ka_rows
    )
    gate_table = "\n".join(
        f"| {name} | {gate.get('criterion', '')} | {'PASS' if gate['pass'] else 'FAIL'} |"
        for name, gate in gates.items()
    )
    err_rows = [
        ("A11(0) 级数线性外推", extrap["a0_dimless"], extrap["rel_error_vs_exact"]),
        ("A11(ka=2e-3) 级数直接值", extrap["value_at_smallest_ka"],
         extrap["rel_error_at_smallest_ka"]),
        ("A11(ka=1e-3) 生产求解器 1024 面元", mesh_rows[-1]["added_mass_dimless"],
         mesh_rows[-1]["rel_error_vs_exact"]),
        ("R2 谱展开下半球（基准 1/2）",
         harness["spectral_lower_half_dimensionless_vs_rho_nabla"],
         harness["spectral_lower_half_dimensionless_vs_rho_nabla"] / 0.5 - 1.0),
    ]
    err_table = "\n".join(
        f"| {label} | {_fmt(value, 9)} | {100 * rel:+.4f}% | "
        f"{'是' if abs(rel) < 0.03 else '否'} |"
        for label, value, rel in err_rows
    )

    return f"""# TensorBEM 浮半球水平（surge）长波极限 `A11(ka→0)`：给 CASE-BEM-HULME-HEMISPHERE 补一条带外精确锚

**m{MISSION_ID} / 分支 agent/m{MISSION_ID} / 仓库 TensorBEM / 算例 `{payload['case_code']}`（锚点 `{payload['anchor']}`）**

## ① 案例介绍

半浸浮半球（平底面与平均自由液面平齐，a = {_fmt(pc['radius_m'], 3)} m，无限水深）的水平（surge）
辐射问题。锚点算例 `CASE-BEM-HULME-HEMISPHERE`（算例池 id 138，容差 <5%）在 `ka ∈ [0.2, 2.0]` 带内已闭环：
辐射/绕射、面元与参考项阶梯、capytaine 第三方路线、Haskind 互易、长波激励力极限均已交付。

| 参数 | 取值 |
|---|---|
| 半径 a | {_fmt(pc['radius_m'], 3)} m |
| 排水体积 ∇ | {_fmt(pc['displaced_volume_m3'])} m³ |
| 水线面 πa² | {_fmt(pc['waterplane_area_m2'])} m² |
| ρ / g | {_fmt(pc['density_kg_m3'], 1)} kg/m³ / {_fmt(pc['gravity_m_s2'], 5)} m/s² |
| ρ∇（量纲单位） | {_fmt(pc['rho_nabla_kg'], 3)} kg |
| 量纲约定 | {pc['normalisation']}，{pc['mode']} |

**缺口**：heave 列已有精确低频锚 `A33(ka→0) = 0.8309572 ρ∇`（`bem_rigid_lid_hemisphere_ka0`），
而 **surge 列此前没有任何带外锚**——其全部非自证对照（capytaine、Haskind）都在 `ka ≥ 0.2` 内，
参考级数若在长波段整体偏移仍可全绿。本单把该端用精确闭式钉死。

## ② 怎么做的（四条互相独立的路线）

ω→0 时自由液面条件 `∂φ/∂z − νφ = 0`（ν = ω²/g）退化为刚壁 `∂φ/∂z = 0`。对 z 做偶延拓：

1. 水平速度在 z→−z 下**不变**，而镜像外法向的 z 分量变号，故 Neumann 数据 `u·n = U n_x` **保持**；
2. 镜像问题 = 无界流体中**整球平移**，其附连质量为 Kirchhoff 闭式 `A_sph = 0.5 ρ Vol_sph`（Vol_sph = 2∇）；
3. 同等速度下延拓场的动能为半空间场的 2 倍 ⇒ `A11(0) = A_sph/2 = ρ∇/2`。

即 **A11(ka→0)/(ρ∇) = 1/2（精确）**。这正是 heave 的反例：heave 的 `u·n = U n_z` 在反射下变号，
镜像成"对滑"两半（→ 0.831），混用会得到 0.5 vs 0.831 的错误锚（m16802 erratum 的坑）。

四条验证路线：

| 路线 | 实现 | 独立性 |
|---|---|---|
| R1 闭式 | Kirchhoff 球 + 能量减半 | 手推解析，不依赖代码 |
| R2 谱展开 | `{harness['method']}` | 数值复核镜像球常数 |
| R3 参考级数 | `tensorbem.analytic.hulme_hemisphere_radiation`，ka=2e-3…0.5，项数阶梯 48/96/144，quad_order=512 | 半解析级数，与面元无关 |
| R4 生产求解器 | `tensorbem.free_surface_solver3d.solve_radiation_3d`（3-D Kelvin 格林函数 BEM），ka=1e-3，面元阶梯 64/256/1024 | 与 R3 方法完全不同 |

工具链：`benchmarks/verified/casepool_bem_hulme_surge_lowfreq/limits.py`（单一事实源）→ `results.json`
→ `report.md` + `figure.png`（程序化渲染）→ `tests/test_casepool_hulme_surge_lowfreq.py`。
**零源码改动**（`src/tensorbem/*` 一行未动），只有新增算例包与测试。

## ③ 计算结果

R3 参考级数低频阶梯（144 项；截断漂移 48 vs 144 项最大 {gates['G6_reference_term_truncation']['measured']:.2e}）：

| ka | A11/(ρ∇) | A11 − 1/2 | (A11 − 1/2)/ka | B11/(ωρ∇) | A33/(ρ∇) |
|---|---|---|---|---|---|
{ref_table}

外推（最小三点线性拟合 A11 = a0 + a1·ka）：A11(0) = {_fmt(extrap['a0_dimless'], 6)}，
斜率 {_fmt(extrap['a1_dimless_per_ka'], 4)}/ka，最大残差 {extrap['max_residual']:.2e}；
与精确 1/2 的相对差 **{100 * extrap['rel_error_vs_exact']:+.4f}%**。
R2 谱展开：整球平移闭式 {_fmt(harness['full_sphere_dimensionless_vs_rho_nabla'], 9)}（ρ∇ 单位）、
下半球积分（即半球附连质量）{_fmt(harness['spectral_lower_half_dimensionless_vs_rho_nabla'], 9)}（应为 1/2）。

R4 生产求解器，面元阶梯 @ ka = 1e-3：

| sub | 面元 | A11/(ρ∇) | vs 精确 1/2 | 条件数 | 相对残差 |
|---|---|---|---|---|---|
{mesh_table}

R4 ka 阶梯（256 面元）：

| ka | 面元 | A11/(ρ∇) | vs 精确 1/2 | B11/(ωρ∇) |
|---|---|---|---|---|
{ka_table}

## ④ 结果图

`figure.png`（matplotlib，从 `results.json` 生成）：左＝参考级数 A11/(ρ∇) vs ka（对数横轴）＋精确渐近线 1/2
＋生产求解器阶梯点；中＝(A11 − 1/2)/ka，显示超标量在窗口内近似常数（极限是有限值而非数值平台）；
右＝B11/(ωρ∇) 的低频消失与求解器面元收敛（64→256→1024 面元，绝对相对误差）。

## ⑤ 注意点

- **滑移 vs 平移**：heave 与 surge 的刚壁镜像语义完全不同（0.831 vs 0.5），本包的 R1 显式写出该不对称性。
- **极限只在 ka→0 成立**：带内 ka = 0.2 的参考值已比极限高 {band_excess:.1f}%，
  带内 gate 仍必须以参考级数逐点为准，不能拿 1/2 去判带内数值。
- **低频求解器不确定性主项是面元离散**（见 ③ 阶梯），不是格林函数消波项；ka = 1e-3 下条件数 O(1)。
- **参考解仍非 Hulme 原文印刷表格**，而是本仓重推级数；本包补的正是与级数无关的低频端精确锚。
- **B11 只有定性判据**（随 ka→0 单调趋零），无解析基准，属未验证项，如实记录。

## ⑥ 误差判定（基准 = 精确闭式 A11(0)/(ρ∇) = 1/2）

| 关键量 | 计算值 | 相对基准误差 | < 3%？ |
|---|---|---|---|
{err_table}

**总判定：{payload['verdict']}**{'（未过：' + ', '.join(payload['failing']) + '）' if payload['failing'] else '（全部 gate 通过）'}

| gate | 判据 | 结论 |
|---|---|---|
{gate_table}
"""


def main() -> int:
    payload = json.loads((HERE / "results.json").read_text())
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    token = _login(base)
    figure_png = base64.b64encode((HERE / "figure.png").read_bytes()).decode()

    body = {
        "title": "TensorBEM 浮半球 surge 长波极限 A11(ka→0)=ρ∇/2：Hulme 锚点带外精确验证",
        "summary": (
            f"CASE-BEM-HULME-HEMISPHERE 的 surge 列补上带外精确锚：四条独立路线"
            f"（闭式 1/2 / 谱展开 / 参考级数外推 {100*payload['extrapolation']['rel_error_vs_exact']:+.4f}% "
            f"/ 生产 3-D 求解器 {100*payload['solver_mesh_ladder'][-1]['rel_error_vs_exact']:+.3f}%）"
            f"一致，判定 {payload['verdict']}。"
        ),
        "scenario_key": "tensorbem.casepool.hulme.surge_longwave_anchor",
        "mission_id": MISSION_ID,
        "tags": ["TensorBEM", "CASE-BEM-HULME-HEMISPHERE", "长波极限", "A11", "验证欠账",
                 "rigid-lid", "算例池"],
        "markdown": build_markdown(payload),
        "figures": [{
            "name": "fig_surge_longwave_anchor.png",
            "caption": (
                "左：Hulme 参考级数 A11/(ρ∇) vs ka（对数横轴，144 项）＋精确渐近线 1/2 ＋"
                "TensorBEM 3-D 求解器（256 面元，ka=1e-3…0.1）；中：(A11−1/2)/ka，显示超标量线性于 ka，"
                "极限为有限值；右：B11/(ωρ∇) 低频消失（级数与求解器）＋求解器面元收敛误差 64/256/1024。"
            ),
            "png_base64": figure_png,
        }],
    }
    req = urllib.request.Request(
        f"{base}/tensor/reports", data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    result = json.loads(urllib.request.urlopen(req).read())
    print(json.dumps({"report_id": result.get("id"), "created": result.get("created_at")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())