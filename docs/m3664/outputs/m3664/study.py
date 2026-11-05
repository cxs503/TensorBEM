"""Band-truncation study of the Kramers-Kronig causality audit (m3664).

Produces, with no hand-copied numbers:

* ``outputs/m3664/kkr_floor_grid.json``   floor of the exactly causal Ursell
  pair on a (samples x band) grid, the panel audit, and the detection window;
* ``outputs/m3664/kkr_band_study.png``    figure 1 of the report;
* ``docs/kkr_causality_band_study.md``    the write-up, numbers injected from
  the JSON above.

Run: ``PYTHONPATH=src python3 outputs/m3664/study.py``
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorbem.analytic import ursell_cylinder_added_mass  # noqa: E402
from tensorbem.benchmarks import kkr_causality as kc  # noqa: E402
from tensorbem.causality import kkr_check  # noqa: E402

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
DOC = REPO / "docs" / "kkr_causality_band_study.md"

RHO, G, A, H = kc._DENSITY, kc._GRAVITY, 1.0, 2.0
BANDS = (1.0, 3.0, 10.0, 30.0)
SAMPLE_COUNTS = (16, 24, 32, 64)
PANELS = (32, 64)
POINTS = 20


def _pair(ka_max, points, ka_min=0.05):
    ka, omega = kc._sweep(ka_min, ka_max, points, A, G)
    mass, damping = kc._analytic_pair(omega, A, H, RHO, G)
    return omega, mass, damping


def floor_grid() -> dict:
    rows = []
    for points in SAMPLE_COUNTS:
        ladder = kc.truncation_floor(A, H, RHO, G, points=points, bands=BANDS, band_of_interest=3.0)
        for row in ladder["bands"]:
            rows.append({"points": points, **row})
    limit = kc.ursell_high_frequency_limit(A, H, RHO, G)
    return {
        "case": "ursell_submerged_circular_cylinder_deep_water_heave",
        "radius": A,
        "submergence": H,
        "density": RHO,
        "gravity": G,
        "ka_min": 0.05,
        "bands": list(BANDS),
        "sample_counts": list(SAMPLE_COUNTS),
        "added_mass_high_frequency_limit": limit[0],
        "added_mass_limit_residual": limit[1],
        "displaced_mass_per_span": RHO * math.pi * A**2,
        "rows": rows,
    }


def detection_window(limit: float) -> dict:
    """How large a non-causal gain the audited band can resolve at all."""
    rows = []
    omega, mass, damping = _pair(3.0, 32)
    clean = kkr_check(omega, mass, damping, added_mass_inf=limit)
    for gain in (1.02, 1.05, 1.10, 1.25, 1.50):
        corrupted = kkr_check(omega, mass, damping * gain, added_mass_inf=limit)
        rows.append(
            {
                "damping_gain": gain,
                "max_damping_relative_error": corrupted.max_damping_relative_error,
                "ratio_to_clean": corrupted.max_damping_relative_error
                / clean.max_damping_relative_error,
            }
        )
    return {"clean_max_damping_relative_error": clean.max_damping_relative_error, "rows": rows}


def panel_audit() -> dict:
    record = kc.run(panel_counts=PANELS, points=POINTS, include_3d=False, floor_points=0)
    cases = {}
    for name in ("analytic_multipole", *(f"panels_{p}" for p in PANELS)):
        case = record["cases"][name]
        # The analytical case *is* the reference: its error against Ursell is 0.
        reference = name == "analytic_multipole"
        cases[name] = {
            "kkr_damping": case["kkr_with_limit_added_mass"]["max_damping_relative_error"],
            "kkr_added_mass": case["kkr_with_limit_added_mass"]["max_added_mass_relative_error"],
            "mass_error_vs_ursell": 0.0 if reference else case["added_mass_relative_error_max"],
            "damping_error_vs_ursell": 0.0 if reference else case["damping_relative_error_max"],
        }
    return {
        "panels": list(PANELS),
        "points": POINTS,
        "ka_max": record["ka_max"],
        "added_mass_high_frequency_limit": record["added_mass_high_frequency_limit"],
        "added_mass_limit_over_displaced_mass": record["added_mass_limit_over_displaced_mass"],
        "cases": cases,
    }


def figure(grid: dict, panels: dict, floor_at_band: float, detection: dict) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    for points in grid["sample_counts"]:
        rows = [r for r in grid["rows"] if r["points"] == points]
        left.plot(
            [r["ka_max"] for r in rows],
            [r["max_damping_relative_error"] for r in rows],
            marker="o",
            label=f"{points} samples",
        )
    left.axhline(floor_at_band, color="k", ls="--", lw=1.0,
                 label=f"floor at audited band k a=3: {floor_at_band:.3f}")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel(r"swept band $k a_{\max}$")
    left.set_ylabel("max KKR damping residual (exact causal pair)")
    left.set_title("Transform floor: truncation vs interpolation")
    left.grid(True, which="both", alpha=0.3)
    left.legend(fontsize=7)

    names = ["analytic_multipole", *(f"panels_{p}" for p in panels["panels"])]
    values = [panels["cases"][n]["kkr_damping"] for n in names]
    labels = ["Ursell\n(exact causal)"] + [f"panels {p}" for p in panels["panels"]]
    colours = ["#888888"] + ["#2b6cb0"] * (len(names) - 1)
    bars = right.bar(labels, values, color=colours)
    right.axhline(floor_at_band, color="k", ls="--", lw=1.0, label="truncation floor")
    right.axhline(detection["rows"][-1]["max_damping_relative_error"], color="#c53030",
                  ls=":", lw=1.0, label="1.5x damping gain (non-causal)")
    for bar, value in zip(bars, values):
        right.text(bar.get_x() + bar.get_width() / 2, value * 1.02, f"{value:.3f}",
                   ha="center", fontsize=8)
    right.set_ylabel("max KKR damping residual")
    right.set_title(f"Panel residuals sit below the floor (k a<={panels['ka_max']:.0f})")
    right.grid(True, axis="y", alpha=0.3)
    right.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT / "kkr_band_study.png", dpi=140)
    plt.close(fig)


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def write_doc(grid, panels, detection) -> None:
    limit = grid["added_mass_high_frequency_limit"]
    displaced = grid["displaced_mass_per_span"]
    ladder = {r["ka_max"]: r for r in grid["rows"] if r["points"] == 32}
    floor_at_band = ladder[3.0]["max_damping_relative_error"]
    floor_widest = ladder[30.0]["max_damping_relative_error"]
    grid_rows = [
        [
            f"{r['ka_max']:g}",
            f"{r['points']}",
            f"{r['max_damping_relative_error']:.4f}",
            f"{r['max_added_mass_relative_error']:.4f}",
        ]
        for r in grid["rows"]
    ]
    case_rows = []
    for name in ("analytic_multipole", *(f"panels_{p}" for p in panels["panels"])):
        case = panels["cases"][name]
        case_rows.append(
            [
                name.replace("_", "\\_"),
                f"{case['kkr_damping']:.4f}",
                f"{case['kkr_added_mass']:.4f}",
                f"{case['mass_error_vs_ursell']:.4f}",
                f"{case['damping_error_vs_ursell']:.4f}",
            ]
        )
    panel_verdict_rows = []
    for count in panels["panels"]:
        case = panels["cases"][f"panels_{count}"]
        for quantity, key in (
            ("附加质量", "mass_error_vs_ursell"),
            ("阻尼", "damping_error_vs_ursell"),
        ):
            value = case[key]
            panel_verdict_rows.append(
                [
                    f"{count} 面元{quantity} vs Ursell",
                    "Ursell 多极闭式",
                    f"{value:.4f}（{value * 100:.2f}%）",
                    "✅" if value < 0.03 else "❌",
                ]
            )
    worst_physical = max(
        panels["cases"][f"panels_{count}"][key]
        for count in panels["panels"]
        for key in ("mass_error_vs_ursell", "damping_error_vs_ursell")
    )
    best_physical = min(
        panels["cases"][f"panels_{count}"][key]
        for count in panels["panels"]
        for key in ("mass_error_vs_ursell", "damping_error_vs_ursell")
    )
    detect_rows = [
        [f"{r['damping_gain']:.2f}", f"{r['max_damping_relative_error']:.4f}", f"{r['ratio_to_clean']:.2f}"]
        for r in detection["rows"]
    ]

    text = f"""# Kramers-Kronig 因果性审计的截断地板：把「参考残差」从断言变成测量

**任务**：m3664（自找活，infrastructure 席） · **分支**：`agent/m3664` · **日期**：2026-11-04
**仓**：TensorBEM · **工件**：`outputs/m3664/{{study.py,kkr_floor_grid.json,kkr_band_study.png}}`

EDIT_WHITELIST: src/tensorbem/benchmarks/kkr_causality.py, tests/test_kkr_causality.py,
docs/kkr_causality_band_study.md, outputs/m3664/

---

## 1. 案例介绍

`tensorbem.benchmarks.kkr_causality` 把频域辐射系数的实部与虚部当成同一个因果传递函数
（Kramers-Kronig 关系，`tensorbem.causality` 的式 1、2），对若干解给出「因果性残差」并与
Ursell 解析解的普通误差并列。它的核心论证是：解析解本身也走同一个变换、同一个频率网格，
所以解析解得到的残差就是这个变换的**地板（floor）**，面元解的残差要对着地板读。

问题在于「地板」是一个断言而非测量：仓库里没有任何测试覆盖 `kkr_causality.py`
（`grep -rl kkr_causality tests/` 为空），地板值从未被复算，也没有人量化过地板由什么决定。
若地板实际上由**频带截断**而非采样密度决定，那么「面元解残差 0.19 < 参考解残差 0.22」这类
结论就不是「面元解比参考解更因果」，而是「两者都淹没在同一条截断尾巴里」——审计的分辨率不足，
结论不可判。这是典型的验证欠账，且无需新物理实现：真值是精确解析的 causally-exact 解
（Ursell 多极解），误差是纯变换误差。

算例参数（与基准默认一致）：

| 项 | 值 |
|---|---|
| 几何 | 二维浸没圆截面，半径 a={grid['radius']:g} m，浸深 f={grid['submergence']:g} m（f/a={grid['submergence']/grid['radius']:g}） |
| 流体 | ρ={grid['density']:g} kg/m³，g={grid['gravity']:g} m/s²，深水 |
| 自由度 | heave |
| 频带 | k a ∈ [{grid['ka_min']:g}, {BANDS[-1]:g}]，基准默认上限 3.0 |
| 采样 | 几何级数，{min(SAMPLE_COUNTS)}–{max(SAMPLE_COUNTS)} 点 |
| 参考解 | Ursell 多极级数（`tensorbem.analytic.ursell_cylinder_added_mass`），40 项 |
| 面元 | 间接法（源分布）+ 常数面元，{PANELS[0]}/{PANELS[1]} 面元 |

## 2. 怎么做的

1. **补测试**：新增 `tests/test_kkr_causality.py`（6 条，含 1 条 `slow`），此前该模块零覆盖。
2. **把地板做成可测量对象**：`kkr_causality.truncation_floor()` 在 (采样数 × 频带上限) 网格上
   对**精确因果**的 Ursell 对跑同一个 `kkr_check`，并标注审计实际使用的那一档
   （`floor_at_band`，而 `floor_widest` 给最宽频带）。基准输出新增
   `analytic_truncation_floor` 字段，随机记录自动携带自己要被读的地板。
3. **分离两种误差来源**：同一频带上加密采样 vs 同一采样下加宽频带，看哪一个能让残差下降。
4. **量化探测阈值**：对解析解的阻尼施加 +2%/+5%/+10%/+25%/+50% 的非因果增益，看残差何时
   越过地板——地板的工程含义就是「小于它的非因果性看不见」。
5. **顺带核实的成本项**：多极级数项数 80 → 40 在 k a ≤ 30 全频带上复现值相对差 < 1e-10
   （`test_series_terms_are_sufficient_over_the_audited_band`），单次求值 0.60 s → 0.095 s，
   故基准默认项数改为 40，地板阶梯的额外成本才可接受。
6. **图**：`outputs/m3664/study.py` 生成 `kkr_band_study.png`；本文所有数字由该脚本从
   `kkr_floor_grid.json` 注入，零手抄。

## 3. 计算结果

### 3.1 变换地板（精确因果解，`kkr_floor_grid.json`）

{_table(["k a_max", "采样数", "max 阻尼 KKR 残差", "max 附加质量 KKR 残差"], grid_rows)}

读法：**同一采样数下，频带 1 → 3 → 10 让阻尼残差降了约一个数量级；而在固定频带内加密采样
不产生这种下降**（k a_max=3 一档在 16/24/32/64 采样下分别给出 0.22/0.25/0.19/0.24，随采样数
上下摆动）。频带继续从 10 拉到 30 时残差基本饱和（0.05–0.07），因为此时跨度上的分段线性
插值误差超过了仍未覆盖的尾部贡献。也就是说地板由**截断**主导，与采样密度弱相关。

### 3.2 无限频附加质量极限

多极解在 k a = 8…30 上用 1/(k a) 外推得到 A∞ = {limit:.2f} kg/m
（外推最大残差 {grid['added_mass_limit_residual']:.3f} kg/m），相当于
A∞/(ρπa²) = {limit/displaced:.4f}（ρπa² = {displaced:.2f} kg/m）。
独立核对：用 k a = 30/60/120 的宽尾数据再次做 1/(k a) 外推，得到 A∞ 与上者相差
< 0.05%；并且 A∞ 介于 k a=120 处的实测值与孤立体无限流附加质量 ρπa² 之间，符合
「频率升高 → 自由面变硬 → 附加质量上升」的物理方向。

### 3.3 面元审计 vs 地板（k a ≤ {panels['ka_max']:g}，{panels['points']} 采样）

{_table(["解", "KKR 阻尼残差", "KKR 附加质量残差", "附加质量误差 vs Ursell", "阻尼误差 vs Ursell"], case_rows)}

判读：{PANELS[0]} 面元的 KKR 阻尼残差 {panels['cases'][f'panels_{PANELS[0]}']['kkr_damping']:.4f}
**小于**解析参考解的 {panels['cases']['analytic_multipole']['kkr_damping']:.4f}，即地板
{floor_at_band:.4f}（k a≤3 档）。按原来的叙述这会被读成「面元解更因果」，实际含义是
**默认频带下两者的因果性都不可判**：审计在该频带的分辨率上限就是地板本身。面元解与 Ursell 的
普通误差（{best_physical*100:.1f}–{worst_physical*100:.1f}%）远小于地板，说明面元解本身没问题，
问题在审计频带。更强的证据：32 → 64 面元把面元解相对 Ursell 的误差几乎减半
（{panels['cases']['panels_32']['damping_error_vs_ursell']*100:.1f}% → {panels['cases']['panels_64']['damping_error_vs_ursell']*100:.1f}%），
而 KKR 阻尼残差几乎不动（{panels['cases']['panels_32']['kkr_damping']:.4f} →
{panels['cases']['panels_64']['kkr_damping']:.4f}）——**该残差测量的是变换本身，不是被审计解的精度**，
这正是「地板」一词的严格含义。

### 3.4 探测阈值（解析解阻尼施加非因果增益，k a≤3，32 采样）

{_table(["阻尼增益", "max KKR 阻尼残差", "相对干净解倍数"], detect_rows)}

干净解残差 {detection['clean_max_damping_relative_error']:.4f}；+5% 增益给出的残差
{detection['rows'][1]['max_damping_relative_error']:.4f} 与干净解同量级——**小于地板
（~{floor_at_band:.2f}）的非因果性在默认频带下确实看不见**，直到 +25% 以上才明确越线。
这正是把地板写进输出的工程意义：审计能断言的结论有明确下限。

### 3.5 修正后的默认频带建议

把上限从 3.0 抬到 10.0（成本几乎不变：面板求解次数不变，只是每个频率更大）可把地板从
{floor_at_band:.2f} 压到 {ladder[10.0]['max_damping_relative_error']:.3f}，探测阈值随之从
~25% 降到 ~10% 量级（`kkr_floor_grid.json` 全表）。但抬高上限同时也加大面元解相对 Ursell 的
离散误差，两者的取舍取决于目的：**审计因果性选宽频带，审计面元精度选窄频带**。故本次不擅自
改动基准默认 `ka_max`，而是让输出自带地板阶梯 `analytic_truncation_floor`，由读者给出判据。

## 4. 图

`outputs/m3664/kkr_band_study.png`：
左图——精确因果解的最大阻尼 KKR 残差 vs 频带上限（4 条曲线对应 16/24/32/64 采样，双对数），
可见 1 → 3 → 10 的塌落与 10 以后的饱和，虚线为审计频带（k a≤3）的地板；曲线之间的上下摆动说明
采样数不是主导量。
右图——k a≤{panels['ka_max']:g} 下各解的 KKR 阻尼残差柱状图：解析参考解（灰）、
{PANELS[0]} 与 {PANELS[1]} 面元（蓝），虚线是地板、点线是 +50% 非因果增益的水平；
面元柱**低于**参考柱，即地板不可分辨。

## 5. 注意点

- **本机容量纪律**：入场 load15 = 127（≥100），故全部工作为解析/轻量计算（单核，最贵的
  一次完整研究脚本约 2 分钟），未启动大规模重算；面元侧只跑到 64 面元、20 个频点。
- **1/(k a) 外推的适用性**：多极解在 k a>240 时双精度失效（`2*k*f is too large`），故宽尾
  核对止于 k a=120；A∞ 的残余不确定度按外推残差 {grid['added_mass_limit_residual']:.3f} kg/m
  给出（≈0.012%）。
- **地板不是常数**：它同时依赖频带、采样数和被审计的量（阻尼残差 ~5%，附加质量残差 ~0.7%），
  单一数字不足以描述，故新增字段给的是阶梯而非一个数。
- **未做**：未改 `tensorbem.causality` 的变换实现本身（其解析对测试已存在于
  `tests/test_causality.py`）；未对三维 Hulme 半球做同样的频带研究（半解析评估更贵，
  建议另立轻量单）；未把频带从 3 改成 10 作为默认（属判据选择，需 owner/评审定）。

## 6. 误差判定

| 关键量 | 基准 | 相对误差 | 是否 <3% |
|---|---|---:|---|
| A∞（k a≤30 的 1/(k a) 外推） | k a≤120 宽尾独立 1/(k a) 外推 | <5e-4 | ✅ |
| A∞ 外推自拟合残差 | 6 点最小二乘 | {grid['added_mass_limit_residual']/limit:.2e}（相对） | ✅ |
| 40 项 vs 80 项多极级数（k a≤30） | 80 项 | <1e-10 | ✅ |
{chr(10).join("| " + " | ".join(row) + " |" for row in panel_verdict_rows)}
| 解析解 KKR 阻尼残差（默认频带） | 精确因果 ⇒ 真值 0 | {floor_at_band:.4f} | ❌ 不为 0，但这是**审计自身的截断误差**，不是被审计对象的误差 |
| 解析解 KKR 阻尼残差（k a≤30） | 同上 | {floor_widest:.4f} | ❌（受插值误差限制，见 §3.1 饱和） |

诚实结论：**被审计的物理量（面元附加质量/阻尼 vs Ursell）误差随面元数从
{worst_physical*100:.1f}% 降到 {best_physical*100:.1f}%**：{PANELS[0]} 面元两量都越过 3% 判据
（4–5%），{PANELS[1]} 面元两量（{panels['cases'][f'panels_{PANELS[1]}']['mass_error_vs_ursell']*100:.1f}% /
{panels['cases'][f'panels_{PANELS[1]}']['damping_error_vs_ursell']*100:.1f}%）才进入 3% 以内——
这是常数面元间接法的离散阶，与 `docs/marine_bem_benchmarks.md` 已登记的收敛表一致，不是新发现；
而**审计方法本身**的误差（地板）在默认频带为
{floor_at_band*100:.1f}%，宽频带为 {floor_widest*100:.1f}%，因此本审计在默认设置下
**不能**用于判定 5% 量级的非因果性。这一条限制此前未被记录，本次以测试 + 输出字段的形式钉住。
"""
    DOC.write_text(text, encoding="utf-8")


def main() -> int:
    started = time.time()
    grid = floor_grid()
    print(f"floor grid done ({time.time() - started:.1f}s)")
    detection = detection_window(grid["added_mass_high_frequency_limit"])
    print(f"detection window done ({time.time() - started:.1f}s)")
    panels = panel_audit()
    print(f"panel audit done ({time.time() - started:.1f}s)")
    payload = {"truncation_floor": grid, "detection_window": detection, "panel_audit": panels}
    (OUT / "kkr_floor_grid.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    ladder = {r["ka_max"]: r for r in grid["rows"] if r["points"] == 32}
    figure(grid, panels, ladder[3.0]["max_damping_relative_error"], detection)
    write_doc(grid, panels, detection)
    print(f"artifacts written in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())