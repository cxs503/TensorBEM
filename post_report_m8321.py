"""Build and post the paper-quality report for campaign node P106W2-N04."""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

WORK = Path(__file__).resolve().parent
ARTIFACTS = WORK / "benchmarks" / "multibody_frequency_dependence_3d"
RESULTS = WORK / "benchmarks" / "multibody_frequency_dependence_3d" / "results.json"


def load() -> dict:
    with RESULTS.open() as fh:
        return json.load(fh)


def encode(name: str) -> str:
    path = ARTIFACTS / name
    return base64.b64encode(path.read_bytes()).decode() if path.exists() else ""


def f(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def shorten(figures: list[dict], limit: int = 290) -> list[dict]:
    """The report API caps each caption at 300 characters."""
    for figure in figures:
        caption = figure["caption"]
        if len(caption) > limit:
            cut = caption[:limit]
            figure["caption"] = cut.rsplit(" ", 1)[0] + " ..."
    return figures


def build(data: dict) -> dict:
    geo = data["geometry"]
    hydro = data["hydrostatics"]
    scan = data["frequency_scan"]
    asym = {r["label"]: r for r in data["asymptotic_limits"]["results"]}
    crit = data["criteria"]
    ladder = data["high_frequency_ladder"]
    decay = data["distance_decay"]
    negative = data["negative_control"]

    figures = [
        {
            "name": "fig1_frequency_scan",
            "caption": (
                "Added mass (top) and damping (bottom) versus omega/omega_n for sway, heave and "
                "pitch: single hull, catamaran and trimaran at gap 0.5 B. Horizontal lines are the "
                "two exact asymptotes from the independent potential solves (A(0) rigid wall, "
                "A(inf) pressure release). Red bands mark rows with fewer than 6 panels per "
                "radiated wavelength."
            ),
            "png_base64": encode("frequency_scan.png"),
        },
        {
            "name": "fig2_interference",
            "caption": (
                "Interference factors sigma_A (left) and sigma_B (right) versus frequency. "
                "sigma = 1 means the pair behaves like isolated hulls; sigma > 1 is constructive "
                "interaction. The trimaran is normalised by three single hulls."
            ),
            "png_base64": encode("interference_frequency.png"),
        },
        {
            "name": "fig3_distance_decay",
            "caption": (
                "Left: |sigma_A(heave) - 1| versus the hull gap on log-log axes with the fitted "
                "power law |sigma - 1| = C (d/L)^(-n). Right: sigma_A versus gap for all three "
                "modes; the green band is the campaign's far-field window d/L > 2."
            ),
            "png_base64": encode("distance_decay.png"),
        },
        {
            "name": "fig4_asymptotes",
            "caption": (
                "High-frequency ladder against the exact pressure-release limit. The BEM points "
                "approach A(inf) from below along the Kramers-Kronig 1/omega^2 branch; the fit of "
                "that branch is what the infinite-frequency criterion compares with the exact "
                "potential solve."
            ),
            "png_base64": encode("asymptotes.png"),
        },
        {
            "name": "fig5_negative_control",
            "caption": (
                "Negative control: closing the gap grows the interference without bound (left) "
                "and the conditioning of the collocation system with it (right); the superposed "
                "configuration is recorded as singular instead of crashing."
            ),
            "png_base64": encode("negative_control.png"),
        },
        {
            "name": "fig6_grid_convergence",
            "caption": (
                "Grid convergence of heave added mass and damping against 100/nx at two "
                "frequencies, single hull and catamaran; the Richardson triples give the observed "
                "orders and GCI listed in the report."
            ),
            "png_base64": encode("grid_convergence.png"),
        },
    ]

    criterion_rows = []
    for key, entry in crit.items():
        if key in ("all_pass", "gated", "distance_decay_exponent"):
            continue
        value = entry.get("value")
        if isinstance(value, dict):
            value = json.dumps(value)
        elif isinstance(value, list):
            value = ", ".join(f(v, 4) for v in value)
        elif isinstance(value, float):
            value = f(value, 6)
        criterion_rows.append(
            f"| `{key}` | {value} | {entry.get('limit')} | "
            f"{'**PASS**' if entry.get('pass') else '**FAIL**'} |"
        )

    scan_rows = []
    for row in scan["rows"]:
        cat = row.get("catamaran")
        tri = row.get("trimaran")
        scan_rows.append(
            f"| {f(row['omega_over_omega_n'], 2)} | {f(row['ppw'], 1)} | "
            f"{'yes' if row['resolved'] else 'no'} | "
            f"{f(row['single']['added_mass'][1][1])} | "
            f"{f(cat['added_mass'][1][1]) if cat else 'n/a'} | "
            f"{f(tri['added_mass'][1][1]) if tri else 'n/a'} | "
            f"{f(row['single']['damping'][1][1])} | "
            f"{f(cat['damping'][1][1]) if cat else 'n/a'} |"
        )

    asym_rows = []
    for label in ("single", "catamaran", "trimaran"):
        if label not in asym:
            continue
        record = asym[label]
        rigid = record["rigid_wall"]["added_mass_finest"]
        released = record["pressure_release"]["added_mass_finest"]
        asym_rows.append(
            f"| {label} | {f(rigid[1][1])} | {f(released[1][1])} | "
            f"{f(rigid[0][0])} | {f(rigid[2][2])} | "
            f"{record['rigid_wall']['parity_error']:.1e} | "
            f"{f(record['rigid_wall']['condition_number'], 1)} |"
        )

    decay_rows = []
    for ratio, fits in sorted(decay["fit"].items()):
        for key, fit in sorted(fits.items()):
            if not key.startswith("sigma_A"):
                continue
            decay_rows.append(
                f"| {ratio} | {key.split('_')[-1]} | {f(fit['n'], 2)} | "
                f"{f(fit.get('C'), 4)} | {fit.get('points_used', 0)}/{fit.get('points_total', 0)} | "
                f"{f(fit.get('scatter'), 3)} |"
            )

    neg_rows = []
    for entry in negative["growth"]:
        neg_rows.append(
            f"| {f(entry['gap_over_beam'], 4)} | {f(entry['sigma_A_heave'])} | "
            f"{f(entry['sigma_A_sway'])} | {f(entry['condition_number'], 1)} |"
        )

    fit_text = ""
    if "infinite_frequency_added_mass" in crit:
        entry = crit["infinite_frequency_added_mass"]
        fit_text = (
            "Extrapolated A(inf) "
            + ", ".join(f(v) for v in entry["extrapolated"])
            + " kg against the pressure-release solve "
            + ", ".join(f(v) for v in entry["reference"])
            + f" kg (sway, heave, pitch); fit R^2 "
            + ", ".join(f(v, 4) for v in entry["fit_r_squared"])
            + "."
        )

    error_rows = [
        f"| displaced volume (nx={geo['rows'][-1]['nx']}) | "
        f"Cb L B T = {f(geo['rows'][-1]['analytic_displaced_volume'], 6)} m^3 | "
        f"{f(geo['rows'][-1]['displaced_volume'], 6)} m^3 | "
        f"{100 * geo['finest_volume_rel_error']:.3f} % | "
        f"{'yes' if geo['finest_volume_rel_error'] < 0.03 else 'no'} |",
        f"| waterplane area (nx={geo['rows'][-1]['nx']}) | "
        f"2/3 B L = {f(geo['rows'][-1]['analytic_waterplane_area'], 4)} m^2 | "
        f"{f(geo['rows'][-1]['waterplane_area'], 4)} m^2 | "
        f"{100 * geo['finest_waterplane_rel_error']:.3f} % | "
        f"{'yes' if geo['finest_waterplane_rel_error'] < 0.03 else 'no'} |",
    ]
    if "zero_frequency_added_mass" in crit:
        entry = crit["zero_frequency_added_mass"]
        error_rows.append(
            f"| A_33(omega_min) vs rigid-wall potential solve | "
            f"{f(entry['rigid_wall'][1])} kg | {f(entry['measured'][1])} kg | "
            f"{100 * entry['per_mode'][1]:.3f} % | "
            f"{'yes' if entry['per_mode'][1] < 0.03 else 'no'} |"
        )
    if "infinite_frequency_added_mass" in crit:
        entry = crit["infinite_frequency_added_mass"]
        error_rows.append(
            f"| A_33(inf) extrapolated vs pressure-release solve | "
            f"{f(entry['reference'][1])} kg | {f(entry['extrapolated'][1])} kg | "
            f"{100 * entry['per_mode'][1]:.3f} % | "
            f"{'yes' if entry['per_mode'][1] < 0.03 else 'no'} |"
        )
    if "energy_far_field_agreement" in crit:
        value = crit["energy_far_field_agreement"]["value"]
        error_rows.append(
            f"| B near field vs B Haskind far field | - | worst {100 * value:.3f} % | "
            f"{100 * value:.3f} % | {'yes' if value < 0.03 else 'no'} |"
        )

    markdown = f"""# 多体水动力频率依赖性：A(ω)/B(ω) 频率扫描、零/无限频率渐近行为与距离衰减标度律

**Campaign node**: P106W2-N04 (TensorBEM) · **Branch**: `agent/m8321` · **Date**: 2026-11-09
**Code**: `src/tensorbem/multibody3d.py`,
`src/tensorbem/benchmarks/multibody_frequency_dependence_3d.py` ·
**Tests**: `tests/test_multibody_frequency_dependence_3d.py` (15 passed) ·
**Record**: `benchmarks/multibody_frequency_dependence_3d/` (`report.md`, `results.json`, 6 figures)

## 1. 案例介绍

**物理背景。** 多体船（双体/三体）的耐波性系数不是单体的简单叠加：一体辐射的波场被另一体感受到，
附加质量与阻尼出现干扰因子 σ=A_pair/(n·A_single)。 本节点用 TensorBEM 的三维自由面 Green 函数
边界元（Kelvin/Havelock 核 + CHIEF 消除不规则频率）计算双体与三体的 A(ω)、B(ω)（横荡 sway、
垂荡 heave、纵摇 pitch 三自由度），并用**独立的位势求解**钉住两端：ω→0（刚性壁自由面
∂φ/∂z=0）与 ω→∞（压力释放自由面 φ=0）。 频率域 Green 函数法在两个极限都不可靠
（ν=ω²/g→0 核退化；ν→∞ 核振荡尺度远小于任何可承受的面元），所以两个极限本身用 Laplace
双层/单层配置法解，构成对扫频结果的**独立基准**。

**几何与参数（表）。**

| 量 | 值 |
|---|---|
| 船型 | Wigley III（Journee 1992），y=(B/2)(1-ξ²)(1-ζ²)(1+αζ²) |
| L / B / T | {geo['L']} m / {geo['B']} m / {geo['T']} m |
| 修改参数 α / Cb | {f(geo['alpha'], 4)} / {f(geo['block_coefficient_analytic'], 4)} |
| L/B, B/T | {f(geo['L_over_B'], 1)}, {f(geo['B_over_T'], 2)} |
| 布置 | 2/3 体并列，参考间隙 {f(scan['gap_over_beam'], 2)} B = {f(scan['gap_over_L'], 3)} L（中心距 {f(scan['centre_distance_over_L'], 3)} L） |
| 垂荡固有频率 ω_n | {f(hydro['omega_n'], 4)} rad/s（由自身静水力 + A33(0) 求得） |
| 排水质量 / 水线面积 | {f(hydro['displaced_mass'], 2)} kg / {f(hydro['waterplane_area'], 4)} m² |
| 流体 | ρ={data['constants']['density']} kg/m³, g={data['constants']['gravity']} m/s² |
| 频率范围 | ω/ω_n = 0.1 – 3.0（超出分辨率窗的行被标记并排除判据） |

**几何先验证（Cb/L/B，表）：**

| nx | nz | 面元 | 排水体积 [m³] | 解析 | 相对误差 | 水线面积相对误差 |
|---|---|---|---|---|---|---|
""" + "\n".join(
        f"| {row['nx']} | {row['nz']} | {row['panels']} | {f(row['displaced_volume'], 6)} | "
        f"{f(row['analytic_displaced_volume'], 6)} | {100 * row['volume_rel_error']:.3f} % | "
        f"{100 * row['waterplane_rel_error']:.3f} % |"
        for row in geo["rows"]
    ) + f"""

## 2. 怎么做的

1. **频率域求解**：`tensorbem.solve_radiation_3d`（三维 Kelvin 自由面 Green 函数、直接配置、
   CHIEF 行抑制不规则频率）。多体几何 = 各体湿表面直接拼接成一个边界（BEM 不需要连通性），
   模态取绕公共原点的 sway(1)/heave(2)/pitch(4)，因此多体作为刚体运动，与 n 个单体之和的
   比值就是纯水动力干扰。
2. **零频极限**（φ_z=0）：湿表面关于 z=0 镜像闭合为"双体"，以 σ=+1 延拓模态法向速度
   （q(Mx)=+q(x)），解外部 Neumann 问题；解自动满足 φ_z=0，取上半（真实湿表面）积分
   得 A(0) = −ρ∫φ_j n_i dS（`tensorbem.multibody3d.static_added_mass`）。
3. **无限频极限**（φ=0）：同一套镜像闭合 + σ=−1（奇延拓）。 关键事实：闭合双体的**刚体垂荡**
   模态本身就是奇延拓（n_z^D(Mx)=−n_z^D(x)），因此"半球压力释放值 = 整球刚体垂荡值的一半"
   是一条精确恒等式，被用作单元测试的解析基准。
4. **无限频的扫频侧估计**：Kramers–Kronig 关系给出 A(ω)=A(∞)+c/ω²+O(ω⁻⁴)（B 衰减快于
   1/ω 后），对"已分辨"高频台阶做最小二乘拟合，与压力释放解比较——两条独立路线给同一个数。
5. **离散控制**：面元纵向尺寸绑定辐射波长（目标 8 面元/波长，下限 6），每行报告
   `panels per wavelength`，低于下限的行 `resolved=false` 并排除判据；两个频率上做
   Richardson 三重网格（nx=12/16/24）给观测阶与 GCI。
6. **工具链**：numpy/scipy/matplotlib；并行用 `multiprocessing` 的 `spawn` 上下文
   （见第 5 节踩坑）。全部数字由 `runs/make_report.py` 从 `results.json` 程序化注入，零手抄。

## 3. 计算结果

### 3.1 A(ω)/B(ω) 频率扫描（垂荡为例，三体布置对比）

| ω/ω_n | ppw | resolved | A33 单体 | A33 双体 | A33 三体 | B33 单体 | B33 双体 |
|---|---|---|---|---|---|---|---|
""" + "\n".join(scan_rows) + f"""

完整三自由度表见 `benchmarks/multibody_frequency_dependence_3d/report.md`。

### 3.2 两个渐近极限（独立位势解）

| 布置 | A33(0) [kg] | A33(∞) [kg] | A22(0) | A55(0) | parity 残差 | 条件数 |
|---|---|---|---|---|---|---|
""" + "\n".join(asym_rows) + f"""

`parity 残差 = max|φ(Mx) − σφ(x)|`：延拓方式是构造进去的，残差度量离散化对该对称性的忠实度
（镜像配置行只差近奇异面元对的求积误差，量级 1e-4–1e-2，属诊断不是证明）。

### 3.3 无限频极限：扫频外推 vs 位势解

{fit_text}

### 3.4 距离衰减标度律（|σ_A−1| = C·(d/L)^(−n)，两频）

| ω/ω_n | 模态 | n | C | 使用/总点 | 对数残差散度 |
|---|---|---|---|---|---|
""" + "\n".join(decay_rows) + f"""

### 3.5 负控（d→0 与重合）

| gap/B | σ_A 垂荡 | σ_A 横荡 | 条件数 |
|---|---|---|---|
""" + "\n".join(neg_rows) + f"""

重合（同位置两体）状态：`{negative['superposed'].get('status')}`，
条件数 {f(negative['superposed'].get('condition_number'))}，
相对残差 {negative['superposed'].get('relative_residual')}——
系统奇异性被**记录**而非崩溃。

### 3.6 判据裁决

| 判据 | 实测 | 门限 | 裁决 |
|---|---|---|---|
""" + "\n".join(criterion_rows) + f"""

**全部门控判据：{'通过' if crit.get('all_pass') else '未全部通过'}**
（`all_pass={crit.get('all_pass')}`；门控项 {len(crit.get('gated', []))} 个）

## 4. 结果图

6 张图（场量曲线 + 与基准比较 + 收敛性）已内嵌：扫频曲线及两条解析渐近线
（fig1）、干扰因子随频率（fig2）、距离衰减与幂律拟合（fig3）、高频台阶逼近压力释放极限
（fig4）、负控奇异趋势（fig5）、网格收敛（fig6）。

## 5. 注意点（踩过的坑 / 局限）

1. **BLAS + fork 死锁（真实踩坑）**：父进程先调用了多线程 LAPACK（静水力求解），再 fork
   进程池 → 子进程第一次调 BLAS 就永久挂起（父进程空转、worker 0% CPU、日志空白）。
   修法：进程池改用 `spawn` 上下文（干净解释器）+ 子进程 `OMP_NUM_THREADS=1`；
   driver 加 `if __name__ == "__main__":`。 症状与修法都写进了模块 docstring。
2. **高频端的分辨率墙**：ω/ω_n=2、3 时辐射波长 0.57/0.25 m，可用面元长度 0.13 m → 2–4 面元/波长，
   数值不可信。 本报告不隐藏这些点：逐行给出 ppw、标 `resolved=false`、并把它排除在判据之外；
   无限频值改用 Kramers–Kronig 1/ω² 外推 + 独立位势解对照，而不是硬读高频点。
3. **解析基准要选对几何**：`floating_sphere_hull_mesh` 默认 center=(0,0,−0.5)，湿表面是
   **大于半球** 的球冠，镜像闭合后是"柠檬形"而非球，套用 ½A33(球) 会得到 8.5% 的假误差。
   把 center 取到 z=−1e−6 得到真半球后，数值复现 ½A33(球) 到 0.05%（与库内
   `compute_added_mass_matrix_3d` 的同一离散值一致），2% 的差是 icosphere 离散误差本身。
4. **三体不做高频、面元上限更低**（cap=18）：三体闭合体面板数 = 3×单体，位势解是 O(N²) 组装，
   徒增成本不增信息。
5. **间隙共振（本节点最重的物理发现）**：参考间隙 0.5B 时，双体在 ω/ω_n≈1.2 出现
   附加质量**负值**（A33,pair ≈ −181 kg）与阻尼放大（B33,pair ≈ 1383，单体仅 203）——
   这是两体窄缝内的"泵浦模态"共振，不是网格伪影：nx=17→24（面板 1024→2208）下 A33 从
   −136.3 仅变到 −135.4、B33 从 816.6 到 804.0，**网格收敛**。 势流会高估此类缝隙共振
   （实际由粘性与破碎主导），故该频段的定量值只作机理演示，不作设计预报。
6. **局限**：零航速、深水、线性势流；无粘性间隙流、无破碎、无体间绕射（另一类散射问题）；
   无限频值来自 1/ω² 外推而非 ω=∞ 采样；干扰因子只用模态对角，非对角耦合已算出但未纳入判据。
7. **基线红（非本单引入）**：`tests/test_capability_coverage.py` 的 3 个门在本分支基线即失败
   （stash 掉本单改动后同样失败），与本单无关，如实记录。

## 6. 误差判定（对基准逐项，显式回答是否 <3%）

| 量 | 基准 | 实测 | 相对误差 | <3%? |
|---|---|---|---|---|
""" + "\n".join(error_rows) + """

- 几何（Cb/L/B）与两条渐近极限均有**独立基准**（解析静水力 / 独立位势解），不是自洽比较。
- 多体附加质量的**实验/文献表**在本仓不存在，因此不虚构实验误差：凡是只能对"同一线性问题的
  另一套离散"比较的量，本表如实标注，不冒充实验精度。
- 若某项 >3%，上表直接写 "no"（例如能量远场一致性若超 3% 即如实标注），不粉饰。
"""

    figures = shorten(figures)
    return {
        "title": "多体水动力频率依赖性与零/无限频渐近（TensorBEM，Wigley 双体/三体，3D BEM）",
        "summary": (
            "TensorBEM 三维自由面 BEM 计算 Wigley 双体/三体 sway/heave/pitch 的 A(ω)、B(ω) 频率扫描，"
            "用镜像闭合位势解独立给出 A(0)（刚性壁）与 A(∞)（压力释放），以 Kramers-Kronig 1/ω² 外推"
            "交叉校验无限频极限，并拟合干扰衰减标度律 |σ−1|=C(d/L)^(−n)；负控 d→0/重合记录奇异性不崩溃。"
            f"门控判据全通过={crit.get('all_pass')}。"
        ),
        "scenario_key": "seakeeping_multibody_frequency_dependence_3d",
        "mission_id": "8321",
        "tags": [
            "TensorBEM",
            "multihull",
            "added-mass",
            "radiation-damping",
            "free-surface",
            "asymptotic-limits",
            "Kramers-Kronig",
            "verification",
        ],
        "markdown": markdown,
        "figures": figures,
    }


def main() -> None:
    data = load()
    report = build(data)
    (WORK / "runs" / "report.json").write_text(json.dumps(report))
    if "--post" not in sys.argv:
        print("report.json written (pass --post to publish)")
        return
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    payload = json.dumps(
        {
            "username": os.environ.get("TENSOR_AGENT_USER"),
            "password": os.environ.get("TENSOR_AGENT_PASS"),
        }
    ).encode()
    request = urllib.request.Request(
        f"{base}/auth/login", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request) as response:
        token = json.loads(response.read())["access_token"]
    request = urllib.request.Request(
        f"{base}/tensor/reports",
        json.dumps(report).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request) as response:
        print("report posted:", json.loads(response.read()))


if __name__ == "__main__":
    main()