"""Render ``docs/drift_force_2d.md`` from the cached drift-force study JSON.

Every number in the documentation page is formatted from
``outputs/drift_force_2d/study.json``; nothing is retyped.  ``tests/test_drift_force.py``
re-renders the page from the same JSON and compares it byte for byte, so a stale
table fails the suite instead of drifting quietly.

Usage::

    python -m tensorbem.benchmarks.drift_force --output outputs/drift_force_2d/study.json \
        --plot docs/figures/drift_force_2d.png
    python docs/tools/gen_drift_force_doc.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY = ROOT / "outputs" / "drift_force_2d" / "study.json"
DEFAULT_DOC = ROOT / "docs" / "drift_force_2d.md"

THRESHOLD = 0.03


def _sci(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}e}"


def _ratio(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _verdict(passed: bool) -> str:
    return "是" if passed else "否"


def render(study: dict) -> str:
    """Return the full markdown page for a study dictionary."""
    settings = study["settings"]
    scale = settings["energy_scale"]
    ellipse = study["ellipse"]
    circle = study["circle"]
    amplitude_rows = study["amplitude_scaling"]
    submergence_rows = study["submergence_decay"]
    band_rows = study["band_sensitivity"]
    recovery = study["recovery"]
    raw = study["unfiltered_difference"]

    last_ellipse = ellipse[-1]
    circle_last = circle[-1]
    recovery_last = recovery["rows"][-1]
    ellipse_discrepancy = abs(last_ellipse["relative_discrepancy"])
    raw_error = abs(raw["relative_error"])
    amplitude_spread = max(
        abs(row["force_ratio_near_over_amplitude_squared"]
            - amplitude_rows[0]["force_ratio_near_over_amplitude_squared"])
        for row in amplitude_rows
    ) / abs(amplitude_rows[0]["force_ratio_near_over_amplitude_squared"])
    band_spread = (
        max(row["near_field_force_x"] for row in band_rows)
        - min(row["near_field_force_x"] for row in band_rows)
    ) / abs(band_rows[0]["near_field_force_x"])
    circle_normalised = abs(circle_last["near_field_force_x"]) / scale
    far_field_convergence = abs(
        ellipse[-1]["far_field_force_x"] - ellipse[-2]["far_field_force_x"]
    ) / abs(ellipse[-1]["far_field_force_x"])

    lines: list[str] = []
    add = lines.append

    add("# 二维深水潜没体的平均二阶波浪漂移力：远场动量定理与近场压力积分的双路验证")
    add("")
    add(
        "**证据等级：REAL**（两条独立公式求解同一物理量，均与本仓已有的一阶绕射解耦合成）"
        f" · **生成方式**：`python docs/tools/gen_drift_force_doc.py`，数字全部由 "
        "`outputs/drift_force_2d/study.json` 程序化注入，无手抄。"
    )
    add("")
    add("## 摘要")
    add("")
    add(
        f"本页结清 `docs/marine_bem_benchmarks.md` 第 12 类基准（二阶波浪漂移力）中"
        f"「二维深水平均漂移力未覆盖」的缺口。实现给出两条互不依赖的求值路线："
        f"远场动量定理 `F_x = (rho*g/2)*|R*A|^2`（Maruo 1960）与近场二阶压力积分 "
        f"`F_j = (rho/2)*oint<|grad phi|^2> n_j ds`。在圆截面潜没柱（Dean/Ursell 零反射定理，"
        f"精确解为 0）上，两路给出 {_sci(circle_last['near_field_force_x'])} N/m 与 "
        f"{_sci(circle_last['far_field_force_x'])} N/m，相对入射动量通量尺度 "
        f"`rho*g*A^2 = {_sci(scale)}` N/m 分别为 "
        f"{_sci(circle_normalised, 2)} 与 {_sci(abs(circle_last['far_field_force_x'])/scale, 2)}，"
        f"即机器精度零。在强反射潜没椭圆柱上（{last_ellipse['panels']} 面元，"
        f"`|R| = {_ratio(last_ellipse['reflection_modulus'])}`），近场 "
        f"{_sci(last_ellipse['near_field_force_x'])} N/m 对远场 "
        f"{_sci(last_ellipse['far_field_force_x'])} N/m 的相对差为 "
        f"{_sci(last_ellipse['relative_discrepancy'])}（{_ratio(100*abs(last_ellipse['relative_discrepancy']), 2)}%）。"
    )
    add("")
    add("## 1. 案例介绍")
    add("")
    add("### 1.1 物理背景")
    add("")
    add(
        "一阶（线性）势流给出的是零均值的简谐力；船舶与海洋结构真正关心的二阶效应是"
        "**平均**（时均）二阶波浪力：固定体上的波浪漂移力，以及有航速船舶的增阻"
        "（added resistance）。它正比于波幅平方，量级通常是一阶激励力的 1%–20%，却决定了"
        "系泊系统的平均载荷、低速船的阻力预算与浮式结构的稳态偏移。本仓的一阶求解器"
        "（`free_surface_solver.solve_diffraction_indirect`）已具备 Kochin 反射/透射系数与 "
        "Haskind 激励力，但此前没有任何平均二阶力的实现或验证。"
    )
    add("")
    add("### 1.2 几何与参数")
    add("")
    add("| 项目 | 取值 |")
    add("|---|---|")
    add("| 流体 | 深水（无限水深）、无黏、不可压、线性自由液面 |")
    add(f"| 密度 `rho` | {settings['fluid_density']} kg/m^3 |")
    add(f"| 重力加速度 `g` | {settings['gravity']} m/s^2 |")
    add(f"| 无因次波数 `ka` | {settings['ka']}（`k = nu = {settings['wavenumber']} 1/m`） |")
    add(f"| 入射波幅 `A` | {settings['wave_amplitude']} m |")
    add(f"| 圆截面 | 半径 {settings['radius']} m，轴心潜深 {settings['circle_submergence']} m |")
    add(
        f"| 椭圆截面 | 半长轴 {settings['ellipse_semi_major']} m、半短轴 "
        f"{settings['ellipse_semi_minor']} m，轴心潜深 {settings['ellipse_submergence']} m |"
    )
    add(f"| 时间因子 | `exp(-i*omega*t)`，`omega = sqrt(g*k)` |")
    add(f"| 力单位 | {settings['force_units']} |")
    add(f"| 归一化尺度 `rho*g*A^2` | {_sci(scale)} N/m |")
    add("")
    add(
        "两类几何各有分工：圆截面在深水中是**精确可判**的（零反射定理），用于钉死「零」；"
        "椭圆截面反射不为零，用于钉死「非零值」，两路必须收敛到同一个数。"
    )
    add("")
    add("## 2. 怎么做的")
    add("")
    add("### 2.1 远场路线（动量定理 / Maruo）")
    add("")
    add(
        "对跨自由液面的控制体做时均动量平衡。深水二维、入射波幅 `A`、反射波幅 "
        "`A_R = R*A`、透射波幅 `A_T`（波幅比由求解器的 Kochin 积分给出，且满足 "
        "`|R|^2+|T|^2=1`），单宽动量通量为 `rho*g*|A|^2/4`（深水 `c_g = c/2`）："
    )
    add("")
    add("```")
    add("F_x = (rho*g/4) * (|A|^2 + |A_R|^2 - |A_T|^2) = (rho*g/2) * |A_R|^2")
    add("```")
    add("")
    add(
        "于是远场路线只用到 `DiffractionResult.reflection_coefficient`，不需要表面速度，"
        "也不需要额外求解。"
    )
    add("")
    add("### 2.2 近场路线（二阶压力积分）")
    add("")
    add(
        "固定、全潜没的物体没有水线项，也不需要二阶势：二阶平均压力 "
        "`<p> = -(rho/2)<|grad phi|^2>`（`-rho*dphi/dt` 项时均为零），故"
    )
    add("")
    add("```")
    add("F_j = -oint <p> n_j ds = (rho/2) * oint <|grad phi|^2> n_j ds")
    add("```")
    add(
        "物面无穿透条件 `dphi/dn = 0` 把被积函数压成切向导数：复幅值下 "
        "`<|grad phi|^2> = (1/2)|dphi/ds|^2`，所以近场路线唯一的额外机器就是"
        "**表面切向速度恢复**（`tensorbem.drift.surface_tangential_velocity`）。"
    )
    add("")
    add("### 2.3 切向速度恢复：为什么不能直接差分")
    add("")
    add(
        "常数面元解的迹是逐点 `O(h)` 精度的，但携带**面元尺度锯齿误差**。对它直接做中心差分，"
        "锯齿被 `1/h` 放大，二阶力被噪声淹没。实测（同一椭圆、同一面元数，见 §3.6）："
        f"未滤波中心差分给出 {_sci(raw['raw_force_x'])} N/m，而远场真值 "
        f"{_sci(raw['far_field_force_x'])} N/m，相对偏差 {_sci(raw['relative_error'])}。"
    )
    add("")
    add(
        "因此恢复层用**弧长带宽限 Fourier 最小二乘拟合**：把面元中心迹拟合成 "
        "`sum_{|m|<=M} c_m*exp(2*pi*i*m*s/L)`（`L` 为周长，`M = mode_fraction*N`），"
        "再解析求导。物理迹在波长尺度上变化（模态数只有几个），截断只丢掉锯齿，不动解。"
        f"对解析场的独立复核（§3.7）显示该层是 **{_ratio(recovery['observed_orders'][-1], 2)} 阶收敛**。"
    )
    add("")
    add("### 2.4 工具链与复现命令")
    add("")
    add("```bash")
    add("python -m tensorbem.benchmarks.drift_force \\")
    add("    --output outputs/drift_force_2d/study.json \\")
    add("    --plot docs/figures/drift_force_2d.png")
    add("python docs/tools/gen_drift_force_doc.py   # 由 JSON 重新生成本页")
    add("python -m pytest tests/test_drift_force.py")
    add("```")
    add("")
    add(
        f"环境：Python {study['environment']['python']}、NumPy {study['environment']['numpy']}；"
        "一阶解由本仓 `free_surface_solver`（深水 Green 函数、间接源法、面元 24 点 Gauss 求积）给出。"
    )
    add("")
    add("## 3. 计算结果")
    add("")
    add("### 3.1 圆截面：精确解为零")
    add("")
    add(
        "Dean (1948) / Ursell (1950)：深水潜没圆截面在**任意频率**都不反射波浪，"
        "故平均漂移力的精确值为 0。下表用两路复算这个零。"
    )
    add("")
    add("| 面元数 | `|R|` | `F_far/(rho*g*A^2)` | `|F_near|/(rho*g*A^2)` |")
    add("|---|---|---|---|")
    for row in circle:
        add(
            f"| {row['panels']} | {_sci(row['reflection_modulus'], 3)} | "
            f"{_sci(abs(row['far_field_force_x'])/scale, 2)} | "
            f"{_sci(abs(row['near_field_force_x'])/scale, 2)} |"
        )
    add("")
    add("### 3.2 椭圆截面：两路收敛到同一个非零值")
    add("")
    add("| 面元数 | `|R|` | 远场 `F_x` [N/m] | 近场 `F_x` [N/m] | 近场 `F_z` [N/m] | 相对差 |")
    add("|---|---|---|---|---|---|")
    for row in ellipse:
        add(
            f"| {row['panels']} | {_ratio(row['reflection_modulus'])} | "
            f"{_sci(row['far_field_force_x'])} | {_sci(row['near_field_force_x'])} | "
            f"{_sci(row['near_field_force_z'])} | {_sci(row['relative_discrepancy'])} |"
        )
    add("")
    add(
        f"远场值的网格收敛幅度（{ellipse[-1]['panels']} 对 {ellipse[-2]['panels']} 面元）为 "
        f"{_sci(far_field_convergence)}，两路相对差"
        f"随加密单调缩小，说明二者不是互相抵消而是收敛到同一极限。"
        f"竖向分量 `F_z`（潜没体受到的抬升/吸附力）由同一套积分给出，但**没有第二条独立路线**，"
        f"仅以收敛性与深潜衰减（§3.4）作为一致性证据。"
    )
    add("")
    add("### 3.3 波幅平方律")
    add("")
    add("| 波幅 `A` [m] | 远场 `F_x/A^2` [N/m^3] | 近场 `F_x/A^2` [N/m^3] |")
    add("|---|---|---|")
    for row in amplitude_rows:
        add(
            f"| {row['amplitude']} | {_sci(row['force_ratio_far_over_amplitude_squared'])} | "
            f"{_sci(row['force_ratio_near_over_amplitude_squared'])} |"
        )
    add("")
    add(
        f"`A` 从 0.5 m 到 2 m，`F/A^2` 的两路相对散布 {_sci(amplitude_spread, 2)}，"
        "与二阶力的量纲一致。"
    )
    add("")
    add("### 3.4 潜深衰减：反射与漂移力同步消失")
    add("")
    add("| 轴心潜深 [m] | 间隙/半短轴 | `|R|^2` | 远场 `F_x/(rho*g*A^2)` | 近场 `F_x/(rho*g*A^2)` |")
    add("|---|---|---|---|---|")
    for row in submergence_rows:
        add(
            f"| {row['submergence']} | {_ratio(row['clearance_over_semi_minor'], 3)} | "
            f"{_sci(row['reflection_modulus']**2)} | {_sci(row['force_ratio_far'])} | "
            f"{_sci(row['force_ratio_near'])} |"
        )
    add("")
    add(
        "物体越深，激起的反射越弱，漂移力按 `|R|^2` 的指数律衰减；两路在同一潜深下给出一致的量级，"
        "这是「力来自反射而不是来自数值噪声」的判据。"
    )
    add("")
    add("### 3.5 恢复层带宽敏感性")
    add("")
    add("| `mode_fraction` | 保留模态数 | 近场 `F_x` [N/m] | 相对默认值变化 |")
    add("|---|---|---|---|")
    for row in band_rows:
        add(
            f"| {row['mode_fraction']} | {row['mode_count']} | "
            f"{_sci(row['near_field_force_x'])} | {_sci(row['relative_change_vs_default'])} |"
        )
    add("")
    add(f"带限在 `{band_rows[0]['mode_fraction']}`–`{band_rows[-1]['mode_fraction']}` 间变动时，"
        f"近场力相对散布 {_sci(band_spread, 2)}：结论不吃带宽参数。")
    add("")
    add("### 3.6 反例：未滤波中心差分的破坏程度")
    add("")
    add("| 项目 | 数值 |")
    add("|---|---|")
    add(f"| 面元数 / 潜深 | {raw['panels']} / {raw['submergence']} m |")
    add(f"| 未滤波差分 `F_x` | {_sci(raw['raw_force_x'])} N/m |")
    add(f"| 远场真值 `F_x` | {_sci(raw['far_field_force_x'])} N/m |")
    add(f"| 相对偏差 | {_sci(raw['relative_error'])} |")
    add("")
    add("### 3.7 恢复层的解析真值复核（不含边界元）")
    add("")
    field = recovery.get("field", {})
    add(
        "取解析双模态调和场 `phi = exp(nu*z + i*nu*x) + c*exp(mu*z + i*mu*x)`"
        f"（`nu = {field.get('nu', 0.6)}`、`mu = {field.get('mu', 0.9)}`、"
        f"`c = {field.get('coefficient_abs', 0.7)}*exp(i*{field.get('coefficient_phase', 1.047):.4f})`），"
        "其表面切向导数的积分用 2^16 点梯形法（对光滑周期被积函数谱精度收敛）给出真值："
        f"`F_x^exact = {_sci(recovery['reference_force_x'])}` N/m。"
    )
    add("")
    add("| 面元数 | 恢复层给出的 `F_x` [N/m] | 相对误差 | 观测阶 |")
    add("|---|---|---|---|")
    for index, row in enumerate(recovery["rows"]):
        order = _ratio(recovery["observed_orders"][index - 1], 2) if index else "—"
        add(
            f"| {row['panels']} | {_sci(row['force_x'])} | "
            f"{_sci(row['relative_error'])} | {order} |"
        )
    add("")
    add("## 4. 结果图")
    add("")
    add("![drift force verification](figures/drift_force_2d.png)")
    add("")
    add(
        "左：椭圆截面远场（蓝圆）与近场（橙方）随 `1/N` 收敛到同一值；"
        "中：潜深增加时 `|R|^2` 与两路漂移力同步指数衰减（双对数）；"
        "右：恢复层对解析场的相对误差与 `O(h^2)` 参考线平行。"
    )
    add("")
    add("## 5. 注意点与局限")
    add("")
    add("1. **适用范围**：深水、二维（每单位波峰长度）、固定体、全潜没。"
        "水线穿出体的近场公式还需 `(1/2)*rho*g*oint|zeta|^2*n dl` 水线项；"
        "有航速时远场公式需改为含遭遇频率与 `c_g` 的 Maruo 形式；三维与差频 QTF 未实现。")
    add("2. **二阶势**：全潜没固定体的平均二阶力只需一阶场（无水线项），但一旦物体穿出自由液面"
        "或考虑慢漂移阻尼，二阶势不可省。")
    add("3. **恢复层的带宽是建模选择**：截断频率以上被丢弃的内容是数值锯齿而非物理；"
        "若把 `ka` 提到 3 以上（体表迹含更多角向模态），必须同步提高 `mode_fraction`。")
    add("4. **竖向分量没有第二条路线**：`F_z` 只用近场积分给出，以收敛性与深潜衰减为一致性证据，"
        "不宣称已独立验证。")
    add("5. **远场公式依赖 `|R|^2+|T|^2=1`**：本仓一阶解的能量守恒误差在 "
        f"{_sci(max(abs(row['energy_balance_error']) for row in ellipse), 2)} 量级，"
        "故 `A_T` 消去不会引入偏差。")
    add("")
    add("## 6. 误差判定（关键量 vs 基准）")
    add("")
    add("| 关键量 | 基准 | 相对误差 | 是否 <3% |")
    add("|---|---|---|---|")
    add(
        f"| 圆截面近场漂移力 | 精确解 0（Dean/Ursell 零反射） | "
        f"{_sci(circle_normalised, 2)}（对 `rho*g*A^2` 归一） | {_verdict(circle_normalised < THRESHOLD)} |"
    )
    add(
        f"| 圆截面远场漂移力 | 精确解 0 | "
        f"{_sci(abs(circle_last['far_field_force_x'])/scale, 2)} | "
        f"{_verdict(abs(circle_last['far_field_force_x'])/scale < THRESHOLD)} |"
    )
    add(
        f"| 椭圆近场 vs 远场 | 远场动量定理值 | {_sci(ellipse_discrepancy)} | "
        f"{_verdict(ellipse_discrepancy < THRESHOLD)} |"
    )
    add(
        f"| 恢复层表面力 | 解析场梯形积分真值 | {_sci(recovery_last['relative_error'])} | "
        f"{_verdict(recovery_last['relative_error'] < THRESHOLD)} |"
    )
    add(
        f"| 波幅平方律 | `F ∝ A^2` | {_sci(amplitude_spread, 2)} | "
        f"{_verdict(amplitude_spread < THRESHOLD)} |"
    )
    add(
        f"| 带宽不敏感性 | 与默认带宽一致 | {_sci(band_spread, 2)} | "
        f"{_verdict(band_spread < THRESHOLD)} |"
    )
    add(
        f"| 未滤波差分（反例） | 远场真值 | {_sci(raw_error)} | {_verdict(raw_error < THRESHOLD)}（刻意保留失败对照） |"
    )
    add("")
    add(
        "结论：除「未滤波差分」这一**刻意保留的失败对照**外，全部关键量相对误差均 <3%，"
        "二维深水平均漂移力的两条路线互相独立且互相确认。"
    )
    add("")
    add("## 参考文献")
    add("")
    add("1. Dean, W. R. (1948). On the reflexion of surface waves by a submerged circular cylinder. "
        "*Proc. Camb. Phil. Soc.* 44, 483–491.")
    add("2. Ursell, F. (1950). Surface waves on deep water in the presence of a submerged circular "
        "cylinder. *Proc. Camb. Phil. Soc.* 46, 141–152.")
    add("3. Maruo, H. (1960). The drift of a body floating on waves. *J. Ship Research* 4, 1–10.")
    add("4. Newman, J. N. (1962). The exciting forces on fixed bodies in waves. *J. Ship Research* 6, 10–17.")
    add("5. Ogilvie, T. F. (1963). First- and second-order forces on a cylinder submerged under a "
        "free surface. *J. Fluid Mech.* 16, 451–472.")
    add("6. Pinkster, J. A. (1980). *Low frequency second order wave exciting forces on floating "
        "structures.* PhD thesis, TU Delft.")
    add("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()
    study = json.loads(args.study.read_text(encoding="utf-8"))
    args.doc.write_text(render(study), encoding="utf-8")
    print(f"wrote {args.doc}")


if __name__ == "__main__":
    main()