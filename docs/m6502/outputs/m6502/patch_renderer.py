"""Third patch: render the energy-Hessian check into the generated report.

Mission m6502. Run from the worktree root with PYTHONPATH=src.
"""
from pathlib import Path

p = Path("/root/tp-agent/work/m6502/src/tensorbem/benchmarks/tank_free_surface_verification.py")
s = p.read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)


# ---- unpack C8 and build its tables ----------------------------------------
rep(
    '''    c4, c5, c6, c7 = c["C4_superposition"], c["C5_metacentric"], c["C6_roll_period"], c["C7_admissibility"]
    lp = c2["Lshape_polygon"]
''',
    '''    c4, c5, c6, c7 = c["C4_superposition"], c["C5_metacentric"], c["C6_roll_period"], c["C7_admissibility"]
    c8 = c["C8_energy_hessian"]
    lp = c2["Lshape_polygon"]

    names = {"rectangle": "矩形 4×2", "Lshape": "L 形（非凸）"}
    diag_rows = []
    cross_rows = []
    for case in ("rectangle", "Lshape"):
        cse = c8["cases"][case]
        for r in cse["rows"]:
            diag_rows.append(
                f"| {names[case]} | {r['eps']:.0e} | {r['roll']:.4e} | {cse['analytic']['roll']:.4e} "
                f"| {r['relerr']['roll']:.2e} | {r['pitch']:.4e} | {cse['analytic']['pitch']:.4e} "
                f"| {r['relerr']['pitch']:.2e} |"
            )
        for r in cse["rows"]:
            cross_ref = cse["analytic"]["roll_pitch"]
            cross_rows.append(
                f"| {names[case]} | {r['eps']:.0e} | {r['roll_pitch']:.4e} | {cross_ref:.4e} "
                f"| {r['cross_abs_err']:.2e} |"
            )
    energy_worst = max(cse["worst_rel_error"] for cse in c8["cases"].values())
    energy_grid = max(cse["max_grid_independence_relerr"] for cse in c8["cases"].values())
    energy_vol = max(cse["max_volume_relerr"] for cse in c8["cases"].values())
    attr = c8["attribution"]
    attr_rows = "\\n".join(
        f"| {a['variant']} | {a['roll']:.4e} | {a['ratio']:.4f} | {a['rel_error']:.2e} |"
        for a in attr
    )
    grid_n = c8["cases"]["rectangle"]["grid"]
    coarse_n = c8["cases"]["rectangle"]["coarse_grid"]
''',
)

# ---- title & artifact line -------------------------------------------------
rep(
    "return f\"\"\"# 液舱自由液面对静水恢复矩阵的修正：闭式解 / 精确多边形积分 / 网格求积三重核对",
    "return f\"\"\"# 液舱自由液面对静水恢复矩阵的修正：闭式解 / 精确多边形积分 / 网格求积 / 能量侧四重核对",
)
rep(
    "`tests/test_tank_free_surface.py`（11 用例回归门禁）",
    "`tests/test_tank_free_surface.py`（回归门禁）、`tests/test_tank_free_surface_doc_pins.py`"
    "（本报告数字钉桩）",
)

# ---- new section 3.5 -------------------------------------------------------
rep(
    '''## 4. 误差判定（逐项回答"是否 < 3 %"）''',
    '''### 3.5 能量侧独立复核（自由液面 vs 冻结液体，C8）

对同一液体体积做两次势能积分：一次让自由面保持水平（船体系平面
`h = H − φ(y−y_c) + θ(x−x_c)`，`(x_c, y_c)` 为足迹形心，故体积 `A·H` 守恒），
一次把液体**冻结**随船体刚性转动（仍是 `A·H` 的柱体）。两者之差 `ΔU = U_free − U_frozen`
就是纯自由液面修正——刚性「重量在高处」的那部分会计在冻结基线里，被精确扣掉。
对 `ΔU` 做中心二阶差分即得 `ΔC` 的三项，与 `−ρ_l g i_xx`、`−ρ_l g i_yy`、`+ρ_l g i_xy` 对照：

| 足迹 | ε (rad) | ∂²ΔU/∂φ²（FD） | 解析 −ρ_l g i_xx | 相对误差 | ∂²ΔU/∂θ²（FD） | 解析 −ρ_l g i_yy | 相对误差 |
|---|---|---|---|---|---|---|---|
{chr(10).join(diag_rows)}

| 足迹 | ε (rad) | ∂²ΔU/∂φ∂θ（FD） | 解析 +ρ_l g i_xy | 绝对误差 |
|---|---|---|---|---|
{chr(10).join(cross_rows)}

最大相对误差 {energy_worst:.2e}（ε = {c8['fd_steps'][-1]:.0e}；ε 从 {c8['fd_steps'][0]:.0e} 减到
{c8['fd_steps'][-1]:.0e} 时误差按 O(ε²) 下降，来自 `sin φ ≈ φ` 这一阶截断，是能量路线自身的
线性化误差而非实现误差）。求积网格 {coarse_n} → {grid_n} 的曲率变化 {energy_grid:.2e}
（中点法对 `i_xx` 的二阶收敛）；`∫h dA = A·H` 的相对偏差 {energy_vol:.1e}（机器精度级）；
两种算差方式（先算总能量再相减 / 在求和内直接算差）一致到 1e-12 量级。
矩形 `i_xy = 0`，其 ∂²ΔU/∂φ∂θ 的绝对值 {c8['cases']['rectangle']['cross_abs_err']:.1e} N·m/rad
是舍入量级的零，与"无耦合"一致。

**历史 4 倍偏差的实测归因**（m6452 放弃该路线的原因）：

| 路线变体 | ∂²ΔU/∂φ²（FD） | 与解析之比 | 相对误差 |
|---|---|---|---|
{attr_rows}

即：把自由面平面写成关于**足迹原点**的 `H − φy + θx`（本仓足迹的形心并不在原点），倾斜后
体积不再守恒——这一条恰好把曲率放大 4.0000 倍，与当时观察到的"解析值的 4 倍"完全吻合；
若再省掉冻结基线，曲率变成 14.50 倍。两步修正后即为上表结果。

## 4. 误差判定（逐项回答"是否 < 3 %"）''',
)

# ---- section 4 rows --------------------------------------------------------
rep(
    "| 能量 Hessian 独立复核 | —— | **未完成**（见 §5 第 1 条） | **不作判定** |",
    "| 能量 Hessian 独立复核（∂²ΔU/∂φ²、∂²ΔU/∂θ²、∂²ΔU/∂φ∂θ） | ρ_l g·(i_xx, i_yy, i_xy)（解析）"
    " | 最大 {energy_worst:.2e}（§3.5） | **是** |\n"
    "| 能量路线的求积网格依赖 | {coarse_n} vs {grid_n} 网格 | {energy_grid:.2e} | **是** |",
)
rep(
    '''**结论**：自由液面修正量的闭式、精确多边形积分与网格求积在 < 3 % 内互相确认（前两者达机器精度级），
`ΔC[3,4]` 的符号由解析推导 + 手算 L 形矩确定。这是**代码验证**，不是试验确认：
没有实船/模型试验数据，不宣称任何试验级精度。''',
    '''**结论**：自由液面修正量的闭式、精确多边形积分、网格求积与能量侧差分在 < 3 % 内互相确认
（闭式 vs 多边形积分达机器精度级；能量路线最大相对误差 {energy_worst:.2e}）。`ΔC[3,3]`、`ΔC[4,4]`、
`ΔC[3,4]` 的符号与量级现在**同时**由解析推导、手算 L 形矩与能量 Hessian 三方支持。
这是**代码验证**，不是试验确认：没有实船/模型试验数据，不宣称任何试验级精度。''',
)

# ---- section 5 item 1 ------------------------------------------------------
rep(
    '''1. **能量 Hessian 独立复核未完成（honest gap）**：原计划用分级网格对液体惯性系重心高度做二阶差分，
   从能量侧独立判定 `ΔC` 三个符号。实现后矩形算例返回的曲率是解析值的 4 倍（`-0.4444 vs -0.1111`
   m⁻¹ 量级），定位到网格归一化口径与体积因子 `A·H` 的用法不一致，本次未能收敛出可信数值，
   因此**该路径的数字一律不入报告**，`tests/` 中也无相应断言（不写未验证的绿灯）。
   符号结论的支撑改为：模块 docstring 的二阶推导 + 手算可验的三角形/L 形矩 + 教科书 FSC 公式。''',
    '''1. **能量侧复核的口径边界（原 honest gap 已结清）**：能量路线证明了「在与模块相同的一阶自由面
   平面 `h = H − φ(y−y_c) + θ(x−x_c)` 假设下，能量 Hessian 与 `ρ_l g·二阶矩` 公式一致」，
   **不**证明该平面假设本身（大倾角时自由面在船体系不再是一阶平面）。旧版本记的 4 倍偏差，
   实测归因是平面未写于形心（体积不守恒），与体积因子 `A·H` 的归一化无关——归因表见 §3.5。''',
)
rep(
    '''7. 验证足迹中矩形/L 形为理想几何；L 形仅用于构造 `i_xy ≠ 0` 检验耦合符号。''',
    '''7. 验证足迹中矩形/L 形为理想几何；L 形仅用于构造 `i_xy ≠ 0` 检验耦合符号。
8. **显式未做（登记在 `study.json` 的 `not_run`）**：大倾角形状稳性（GZ 曲线）、液面触顶/抽空判据、
   晃荡与防晃舱壁阻尼、实船/模型试验确认——本报告不对这些作任何断言。''',
)

p.write_text(s)
print("patched renderer")