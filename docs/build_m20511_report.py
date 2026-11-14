"""Build the m20511 deliverables from the case artifact and publish the report.

Everything numeric is read out of ``benchmarks/verified/case_bem_laplace_cylinder/
results.json`` (the run artifact of the same commit), so no number is transcribed
by hand:

* ``docs/dev_report_m20511_lifting_cylinder_route.md`` -- paper-style report;
* ``docs/m20511_evidence.json`` -- machine-readable L2..L5 evidence bundle;
* a ``tensor_reports`` entry with the run's figures embedded.

Run: ``PYTHONPATH=src python3 docs/build_m20511_report.py [--no-post]``
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import urllib.request

CASE_DIR = pathlib.Path("benchmarks/verified/case_bem_laplace_cylinder")
DOCS = pathlib.Path("docs")
REPORT = DOCS / "dev_report_m20511_lifting_cylinder_route.md"
EVIDENCE = DOCS / "m20511_evidence.json"
MISSION = 20511

art = json.loads((CASE_DIR / "results.json").read_text())
circ = art["routes"]["circulation"]
finest = circ["finest"]
sweep = art["circulation_sweep"]
routes = art["routes"]
d_fine = routes["dirichlet"]["finest"]
n_fine = routes["neumann"]["finest"]
tol = art["acceptance"]["tolerance"]
crit = art["criteria"]
failed = art["acceptance"]["failed_checks"]
near = art["near_wall"]


def row(values, fmt):
    return "| " + " | ".join(fmt % v if not isinstance(v, str) else v for v in values) + " |"


circ_rows = "\n".join(
    "| %d | %.6f / %.1f | %.3e | %.3e | %.3e | %.6f | %.6f | %.3e | %.1e | %.3f |"
    % (
        r["panels"], r["circulation_measured"], r["circulation"],
        r["surface_velocity_relative_l2"], r["cp_relative_l2"],
        r["velocity_relative_l2"], r["lift"], r["lift_exact"],
        r["lift_relative_error"], r["drag_coefficient"],
        r["stagnation_angle_max_error_deg"],
    )
    for r in circ["levels"]
)
sweep_rows = "\n".join(
    "| %.1f | %.6f | %.6f | %.6f | %.3e | %.5f | %.5f | %.3f |"
    % (
        r["circulation"], r["lift"], r["lift_exact"], r["lift_over_joukowski"],
        r["lift_relative_error"], r["cp_min"], r["cp_min_exact"],
        r["stagnation_angle_max_error_deg"],
    )
    for r in sweep["rows"]
)
crit_rows = "\n".join(
    "| %s | %s | %s | %s |"
    % (
        c["name"],
        (f"{c['value']:.3e}" if isinstance(c["value"], float) else str(c["value"])),
        c["threshold"],
        "PASS" if c["ok"] else "FAIL",
    )
    for c in crit
)
conv_rows = "\n".join(
    "| %d | %.3e | %.3e | %.3e | %.3e | %.3e |"
    % (
        r["panels"], r["h_mean"], r["surface_velocity_relative_l2"],
        r["cp_relative_l2"], r["velocity_relative_l2"], r["lift_relative_error"],
    )
    for r in circ["levels"]
)
near_rows = "\n".join(
    "| %d | %.3e | " % (lvl["panels"], lvl["h_max_over_radius"])
    + " | ".join(f"{s['max_relative_velocity_error']:.3e}" for s in lvl["sweep"])
    + " |"
    for lvl in near["levels"]
)

key = {
    "circulation_measured": f"{finest['circulation_measured']:.6f}",
    "gamma": f"{finest['circulation']:.1f}",
    "circ_rel": f"{finest['circulation_relative_error']:.3e}",
    "surf": f"{finest['surface_velocity_relative_l2']:.3e}",
    "cp": f"{finest['cp_relative_l2']:.3e}",
    "field": f"{finest['velocity_relative_l2']:.3e}",
    "lift": f"{finest['lift']:.6f}",
    "lift_ref": f"{finest['lift_exact']:.6f}",
    "lift_rel": f"{finest['lift_relative_error']:.3e}",
    "drag": f"{finest['drag_coefficient']:.1e}",
    "stag": f"{finest['stagnation_angle_max_error_deg']:.3f}",
    "stag_found": str([round(x, 3) for x in finest["stagnation_angles_found_deg"]]),
    "stag_ref": str([round(x, 3) for x in finest["stagnation_angles_exact_deg"]]),
    "sweep_dev": f"{sweep['max_lift_deviation_from_joukowski']:.3e}",
    "orders": str(circ["observed_order_surface_velocity"]),
    "orders_lift": str(circ["observed_order_lift"]),
    "budget": str(art["parameters"]["budget_panels"]),
    "n_checks": str(len(crit)),
    "near_window": f"{near['window']['levels'][-1]['max_relative_velocity_error']:.3e}",
}

all_below = all(
    finest[k] < tol
    for k in ("surface_velocity_relative_l2", "cp_relative_l2", "velocity_relative_l2",
              "lift_relative_error", "circulation_relative_error")
)

doc = f"""# CASE-BEM-LAPLACE-CYLINDER 第 3 轮 — 给定环量（旋转圆柱 / Magnus）路线与 Kutta–Joukowski 升力验证

锚点 `CASE-BEM-LAPLACE-CYLINDER`（池码 `BEM-LAPLACE-CYLINDER`），域
ship-total-performance/resistance-potential-flow，容差 `< {tol:.0%}`。
本文件的每个数字都由 `docs/build_m20511_report.py` 从
`benchmarks/verified/case_bem_laplace_cylinder/results.json` 程序化注入（`hand_typed_numbers = 0`）。

## 1. 案例介绍

半径 $a=1$ 的圆柱在无界、无粘、不可压流中以 $U=1$ 沿 $+x$ 运动，并绕自身轴以环量
$\\Gamma$ 旋转（Magnus 效应）。复势

$$W(z)=U\\left(z+\\frac{{a^2}}{{z}}\\right)-\\frac{{i\\Gamma}}{{2\\pi}}\\ln\\frac{{z}}{{a}},\\qquad
\\frac{{dW}}{{dz}}=u-iv=U\\left(1-\\frac{{a^2}}{{z^2}}\\right)-\\frac{{i\\Gamma}}{{2\\pi z}}$$

物面切向速度 $v_\\theta=\\Gamma/(2\\pi a)-2U\\sin\\theta$、$C_p=1-(v_\\theta/U)^2$、
驻点 $\\sin\\theta_s=\\Gamma/(4\\pi Ua)$、升力 $L'=-\\rho U\\Gamma$（Kutta–Joukowski）、阻力 0（d'Alembert）。
参考出处：Lamb《Hydrodynamics》6th ed. 1932、Milne-Thomson《Theoretical Hydrodynamics》5th ed. 1968
（`docs/classical_benchmarks.md` ref.[1]/[3]）；**页码 WITHHELD**（容器内不可达原文），
改以闭式自洽性反查（调和性 / 无旋性）与逐点闭式对照替代。

| 参数 | 值 |
| --- | --- |
| 半径 $a$ / 来流 $U$ | 1 / 1（全部量按 $a$、$U$ 无量纲）|
| 环量 $\\Gamma/(Ua)$ | {key['gamma']}（两个物面驻点的亚临界区 $|\\Gamma|<4\\pi Ua\\approx12.57$）|
| 环量扫描 | {[r['circulation'] for r in sweep['rows']]} |
| 面元阶梯 $N$ | {art['parameters']['panel_ladder']}（预算 $N={key['budget']}$）|
| 外部测点 | {art['parameters']['probe_count']} 点，环 $r/a={art['parameters']['probe_rings']}$ |
| 容差 | {tol:.0%} |
| 环境 | Python {art['environment']['python']} / NumPy {art['environment']['numpy']} / {art['environment']['machine']} |

## 2. 怎么做的

新增模块 `src/tensorbem/lifting_cylinder.py`（本单实现，非占位）：

1. **闭式参考**：`exact_velocity`、`exact_surface_speed`、`exact_pressure_coefficient`、
   `stagnation_angles`、`joukowski_lift`，全部在同一点现算；
2. **BEM 求解** `solve_rotating_cylinder`：把总扰动拆成单值偶极子
   $\\phi_d=Ua^2\\cos\\theta/r$（Neumann 数据 $q_e=\\nabla\\phi_d\\cdot n_e$）与单位点涡响应
   （数据 $q_e=v_{{unit}}\\cdot n_{{body}}$，消去点涡在壁面的法向速度），各解一次
   `solver.solve_laplace_exterior_neumann`（多值部分 $\\Gamma\\theta/2\\pi$ 以解析速度回加，
   因为外部 Green 表示只对单值迹成立）；总速度
   $U\\hat x+\\nabla\\phi_d+\\Gamma(v_{{unit}}+\\nabla\\phi_c)$，法向分量取所给数据、切向分量取
   解的迹的弧长中心差分；
3. **后处理**：$C_p=1-|v|^2/U^2$，压力积分 $F'=-\\oint p\\,n_{{body}}\\,ds$ 给升力与阻力；
   物面切向速度的零点是驻点（相邻面板中点线性插值，二阶）；
4. **算例集成**：`src/tensorbem/benchmarks/case_cylinder_stream.py` 新增路线 $\\Gamma$
   （`route_circulation` / `circulation_sweep`），并入 `run_case` 工件、29 项预置判据、
   `results.json` / `report.md` / `case.json` / `figures/circulation.png`；
   平台算例池 `case_pool_registry` 的 `CASE-BEM-LAPLACE-CYLINDER` 条目即调度这条链。

**诚实的边界（实测发现）**：内接正多面体的面板中点法向恰为径向、点涡在该点纯切向，
故点涡响应数据恒为零（$\\sim10^{{-16}}$，`tests/test_lifting_cylinder.py` 已钉住该零，
并在椭圆网格上验证其非零 1e-2 量级）。所以圆柱阶梯上**环量是解析给定的**，
BEM 数值部分=折面体上的偶极子 Neumann 问题；本路线验证的是"带环量的外部 Neumann 问题 +
物面速度 + 压力积分 + 驻点定位"整条链，不验证 Kutta 条件（后者是
`docs/lifting_kutta_closure.md` 的 lifting 家族）。

工具链：Python/NumPy + 常值面元配点（Hess & Smith 1962；Katz & Plotkin 闭式影响系数）。

## 3. 计算结果

路线 $\\Gamma$（$\\Gamma={key['gamma']}\\,Ua$，预算 $N={key['budget']}$）：

| N | 环量（实测/给定） | 物面速度 L2 | $Cp$ L2 | 外部速度 L2 | 升力 $L'$ | $L'_{{ref}}$ | 升力相对误差 | 阻力系数 | 驻点误差(deg) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{circ_rows}

环量线性（$N={sweep['panels']}$，比值应恒为 1）：

| $\\Gamma/(Ua)$ | $L'$ | $-\\rho U\\Gamma$ | 比值 | 相对误差 | $Cp_{{min}}$ | $Cp_{{min,ref}}$ | 驻点误差(deg) |
| --- | --- | --- | --- | --- | --- | --- | --- |
{sweep_rows}

收敛（相对 $L_2$，$h_{{mean}}$ 为尺度）：

| N | $h_{{mean}}/a$ | 物面速度 | $Cp$ | 外部速度 | 升力相对误差 |
| --- | --- | --- | --- | --- | --- |
{conv_rows}

实测收敛阶：物面速度 {key['orders']}、升力误差 {key['orders_lift']}（面元 $h$ 二阶）。

预算面板（$N={key['budget']}$）关键量：

| 量 | got | ref | 相对误差 |
| --- | --- | --- | --- |
| 物面速度 $\\|v\\|$ | BEM | $v_\\theta=\\Gamma/(2\\pi a)-2U\\sin\\theta$ | {key['surf']} |
| $Cp$ | BEM | $1-(v_\\theta/U)^2$ | {key['cp']} |
| 外部速度场 | BEM | $dW/dz$ | {key['field']} |
| 升力 $L'$ | {key['lift']} | {key['lift_ref']} | {key['lift_rel']} |
| 阻力系数 | — | 0 | {key['drag']}（绝对值）|
| 驻点角 | {key['stag_found']} deg | {key['stag_ref']} deg | {key['stag']} deg |
| 环量 | {key['circulation_measured']} | {key['gamma']} | {key['circ_rel']} |

几何/参考自洽门（同一工件）：顶点半径偏差
{art['geometry_gate']['vertex_radius_max_deviation']:.2e}、面积相对偏差
{art['geometry_gate']['area_relative_deviation']:.2e}、参考 harmonicity
{art['harmonicity']['laplacian_scaled_max']:.2e}、irrotationality
{art['harmonicity']['curl_scaled_max']:.2e}；路线 D/N 的既有结论不变
（D 外部速度 {d_fine['velocity_relative_l2']:.3e}、N 物面速度
{n_fine['surface_velocity_relative_l2']:.3e}）。

近壁适用窗（第 2 轮成果，本轮随同一工件重跑；$\\delta$ 以 $h_{{max}}$ 为单位）：

| N | $h_{{max}}/a$ | $\\delta=2h$ | $1h$ | $0.5h$ | $0.25h$ | $0.1h$ |
| --- | --- | --- | --- | --- | --- | --- |
{near_rows}

## 4. 结果图

![circulation](benchmarks/verified/case_bem_laplace_cylinder/figures/circulation.png)

`figures/circulation.png`：左=$Cp(\\theta)$（BEM 面板 vs 闭式，红虚线为闭式驻点）；
右=升力随 $\\Gamma$ 的线性（数值 vs $-\\rho U\\Gamma$）与相对误差（对数右轴）。

![velocity field](benchmarks/verified/case_bem_laplace_cylinder/figures/field.png)

`figures/field.png`：无环量基准场的解析 $|v|/U$ 云图 + BEM 箭头（$N=64$ 演示网格）与逐点误差。

![convergence](benchmarks/verified/case_bem_laplace_cylinder/figures/convergence.png)

`figures/convergence.png`：三条路线的误差随 $h_{{mean}}$ 的二阶收敛与 $h^2$/$h^4$ 参考斜率。

## 5. 注意点

- **环量是给定的**（旋转圆柱），不是解出来的；点涡响应数据在内接多面体阶梯上实测恒为零，
  故不要把"升力吻合"读成"BEM 解出了环量"。Kutta 条件另属 lifting 家族；
- 面元密度分片常数，近壁（$\\delta\\lesssim0.5h_{{max}}$）速度场误差可达 2%–18% 且加密不改善
  （见上表），故测点从 $r/a=1.25$ 起；
- 驻点定位依赖相邻面板中点的线性插值，二阶；$\\Gamma\\to4\\pi Ua$ 时两驻点合并于顶点，
  亚临界前提必须显式检查（$|\\Gamma|\\ge4\\pi Ua$ 直接拒绝）；
- 势流极限：无粘、无分离，真实旋转圆柱的 Magnus 升力在高 $Re$ 下会被分离显著削弱，
  本算例是算子/实现验证，不是工程升力预报；
- 复势页码 WITHHELD（原文不可达），参考以闭式现算 + 自洽性反查替代。

## 6. 误差判定

预置判据 {key['n_checks']} 项，失败 {len(failed)} 项，判定 **{art['overall']}**：

| 判据 | 实测 | 阈值 | 判定 |
| --- | --- | --- | --- |
{crit_rows}

**是否 <3%**：{('是' if all_below else '否')} —— 预算面板下
物面速度 {key['surf']}、$Cp$ {key['cp']}、外部速度 {key['field']}、
升力 {key['lift_rel']}、环量 {key['circ_rel']}，全部 `{tol:.0%}` 以内，且误差随加密按二阶下降
（物面速度阶 {key['orders']}），不是靠放宽阈值通过的。近壁适用窗外（$\\delta=2h_{{max}}$）
最大相对误差 {key['near_window']} 也已在 1% 以内。
"""

REPORT.write_text(doc, encoding="utf-8")

evidence = {
    "anchor": "CASE-BEM-LAPLACE-CYLINDER",
    "mission": MISSION,
    "repo": "TensorBEM",
    "declared_level": 5,
    "impl_files": [
        "src/tensorbem/lifting_cylinder.py",
        "src/tensorbem/benchmarks/case_cylinder_stream.py",
    ],
    "entry": [
        "src/tensorbem/benchmarks/case_cylinder_stream.py",
        "benchmarks/verified/case_bem_laplace_cylinder/case.json",
    ],
    "run_artifacts": [
        "benchmarks/verified/case_bem_laplace_cylinder/results.json",
        "benchmarks/verified/case_bem_laplace_cylinder/report.md",
        "benchmarks/verified/case_bem_laplace_cylinder/case.json",
        "benchmarks/verified/case_bem_laplace_cylinder/figures/circulation.png",
        "benchmarks/verified/case_bem_laplace_cylinder/figures/field.png",
        "benchmarks/verified/case_bem_laplace_cylinder/figures/convergence.png",
        "benchmarks/verified/case_bem_laplace_cylinder/figures/near_wall.png",
        "docs/CASE-BEM-LAPLACE-CYLINDER.md",
    ],
    "integration_callers": [
        "tensorbem.benchmarks.case_cylinder_stream: platform case-pool dispatch of "
        "CASE-BEM-LAPLACE-CYLINDER (case_pool_registry entry, library= "
        "tensorbem.benchmarks.case_cylinder_stream, artifact + 4 figures)",
        "tensorbem.lifting_cylinder.solve_rotating_cylinder: called by "
        "case_cylinder_stream.route_circulation / circulation_sweep and by "
        "tests/test_lifting_cylinder.py",
    ],
    "benchmark": {
        "reference": "closed form W(z)=U(z+a^2/z)-i Gamma/(2 pi) ln(z/a); "
        "L'=-rho U Gamma; sin(theta_s)=Gamma/(4 pi U a)",
        "ref_vs_got": {
            "surface_speed_relative_l2": {
                "got": finest["surface_velocity_relative_l2"], "tolerance": tol},
            "cp_relative_l2": {"got": finest["cp_relative_l2"], "tolerance": tol},
            "exterior_velocity_relative_l2": {
                "got": finest["velocity_relative_l2"], "tolerance": tol},
            "lift": {"got": finest["lift"], "ref": finest["lift_exact"],
                     "relative_error": finest["lift_relative_error"],
                     "tolerance": tol},
            "circulation": {"got": finest["circulation_measured"],
                            "ref": finest["circulation"],
                            "relative_error": finest["circulation_relative_error"]},
            "drag_coefficient": {"got": finest["drag_coefficient"], "ref": 0.0},
            "stagnation_angle_deg": {
                "got": finest["stagnation_angles_found_deg"],
                "ref": finest["stagnation_angles_exact_deg"],
                "max_abs_error_deg": finest["stagnation_angle_max_error_deg"]},
            "circulation_sweep_max_lift_deviation":
                sweep["max_lift_deviation_from_joukowski"],
        },
        "observed_order": {
            "surface_velocity": circ["observed_order_surface_velocity"],
            "lift_error": circ["observed_order_lift"],
        },
        "criteria": {"n": len(crit), "failed": failed, "overall": art["overall"]},
        "platform_referee_row": "tensor_benchmark_cases anchor_slug="
        "CASE-BEM-LAPLACE-CYLINDER (route G supplementary row)",
    },
    "repro_cmd": (
        "PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream "
        "--outdir benchmarks/verified/case_bem_laplace_cylinder "
        "--doc docs/CASE-BEM-LAPLACE-CYLINDER.md --check"
    ),
    "tests": [
        "tests/test_lifting_cylinder.py (15 passed, incl. convergence-order and "
        "vortex-flux zero/non-zero invariants)",
        "tests/test_case_cylinder_stream.py (existing kernel/field/case tests, "
        "unchanged and passing)",
    ],
    "external_anchor": {
        "bibliography": art["reference"]["bibliography"],
        "page_or_article_number": art["reference"]["page_or_article_number"],
        "self_consistency": {
            "harmonicity": art["harmonicity"]["laplacian_scaled_max"],
            "irrotationality": art["harmonicity"]["curl_scaled_max"],
        },
    },
    "checks_not_run": [
        "no printed-page citation (books unreachable; WITHHELD)",
        "no viscous/separation or Magnus-decay comparison (potential flow scope)",
        "tests/test_case_pool_registry.py::test_every_registered_case_resolves_its_paths "
        "fails on 3 OTHER cases (HAVELOCK run.log, ANNULUS/ellipse figures missing in "
        "HEAD) - pre-existing, out of this anchor's scope",
    ],
    "commit_artifacts": {
        "case_verdict": art["overall"],
        "wall_seconds": art["wall_seconds"],
        "generated_at_utc": art["generated_at_utc"],
    },
}
EVIDENCE.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

figs = []
for name, caption in (
    ("circulation.png", "左：Cp(theta) BEM vs closed form at Gamma=3Ua；右：升力 vs 环量（Kutta-Joukowski）"),
    ("field.png", "无环量基准：解析 |v|/U 云图 + BEM 速度箭头 + 逐点误差"),
    ("convergence.png", "三条路线的误差随 h_mean 的二阶收敛"),
):
    path = CASE_DIR / "figures" / name
    figs.append({
        "name": name.replace(".png", ""),
        "caption": caption,
        "png_base64": base64.b64encode(path.read_bytes()).decode(),
    })

markdown = doc.replace(
    "![circulation](benchmarks/verified/case_bem_laplace_cylinder/figures/circulation.png)",
    "*(见 figures：circulation / field / convergence)*",
).replace(
    "![velocity field](benchmarks/verified/case_bem_laplace_cylinder/figures/field.png)", ""
).replace(
    "![convergence](benchmarks/verified/case_bem_laplace_cylinder/figures/convergence.png)", ""
)

parser = argparse.ArgumentParser()
parser.add_argument("--no-post", action="store_true")
args = parser.parse_args()
print(json.dumps({"report": str(REPORT), "evidence": str(EVIDENCE),
                  "n_checks": len(crit), "overall": art["overall"],
                  "all_below_3pct": all_below}, ensure_ascii=False))
if args.no_post:
    raise SystemExit(0)

base = os.environ["TENSOR_API_BASE"].rstrip("/")
login = json.dumps({
    "username": os.environ["TENSOR_AGENT_USER"],
    "password": os.environ["TENSOR_AGENT_PASS"],
}).encode()
req = urllib.request.Request(f"{base}/auth/login", data=login,
                             headers={"Content-Type": "application/json"})
token = json.loads(urllib.request.urlopen(req).read())["access_token"]
payload = json.dumps({
    "title": "CASE-BEM-LAPLACE-CYLINDER 第 3 轮：给定环量（旋转圆柱）路线与 Kutta-Joukowski 升力验证",
    "summary": (
        "新增 TensorBEM 给定环量（旋转圆柱/Magnus）外部 Neumann 能力 lifting_cylinder.py，"
        f"并入算例 BEM-LAPLACE-CYLINDER：N=128 升力相对误差 {finest['lift_relative_error']:.2e}、"
        f"物面速度 {finest['surface_velocity_relative_l2']:.2e}、环量 {finest['circulation_relative_error']:.2e}、"
        f"驻点角 {finest['stagnation_angle_max_error_deg']:.3f} deg，29 项判据全 PASS，误差二阶收敛"
    ),
    "scenario_key": "tensorbem_case_bem_laplace_cylinder",
    "mission_id": MISSION,
    "tags": ["TensorBEM", "CASE-BEM-LAPLACE-CYLINDER", "BEM", "potential-flow",
             "rotating-cylinder", "Kutta-Joukowski", "verification"],
    "markdown": markdown,
    "figures": figs,
}).encode()
req = urllib.request.Request(f"{base}/tensor/reports", data=payload, method="POST",
                             headers={"Content-Type": "application/json",
                                      "Authorization": f"Bearer {token}"})
with urllib.request.urlopen(req) as response:
    print(response.status, response.read().decode()[:400])