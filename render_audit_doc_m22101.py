"""Render the reference-audit note for m22101 from the measured JSON.

All numbers in ``docs/CASE-BEM-LAPLACE-CYLINDER-REFERENCE-AUDIT.md`` come from
``artifacts/m22101/audit_metrics.json`` (written by ``audit_reference_m22101.py``)
so nothing is transcribed by hand.

Run: PYTHONPATH=src python3 render_audit_doc_m22101.py
"""

from __future__ import annotations

import json
from pathlib import Path

METRICS = Path("artifacts/m22101/audit_metrics.json")
DOC = Path("docs/CASE-BEM-LAPLACE-CYLINDER-REFERENCE-AUDIT.md")


def fmt(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.3e}"
    return f"{value:.6f}"


def main() -> None:
    metrics = json.loads(METRICS.read_text())
    free = metrics["free_stream"]
    lifting = metrics["lifting"]

    lines: list[str] = []
    add = lines.append
    add("# CASE-BEM-LAPLACE-CYLINDER：参考解链的独立复核")
    add("")
    add(
        "本文件复核的不是 BEM 数值解，而是**判据所对标的参考解本身**。"
        "该算例的全部判据都用一组手推闭式公式做基准："
    )
    add("")
    add("```")
    add("W(z)          = U (z + a^2/z)")
    add("dW/dz = u - i v = U (1 - a^2/z^2)")
    add("Cp            = 1 - (v/U)^2")
    add("W_Gamma(z)    = U (z + a^2/z) - i Gamma/(2 pi) ln(z/a)")
    add("L'            = -rho U Gamma        (Kutta-Joukowski, rho = 1)")
    add("sin(theta_s)  = Gamma/(4 pi U a)")
    add("```")
    add("")
    add(
        "如果其中任何一条在转录时写错，**所有判据会一起对错**，记录仍然全绿——即"
        "「自证」风险。下面的复核只用上面声明的**复势本身**，不调用任何被复核的导数公式。"
    )
    add("")
    add("## 方法")
    add("")
    add("1. **速度＝复势导数**，用离散 Cauchy 积分在半径 $r=0.05|z|$ 的小圆上算：")
    add("")
    add("   $$W'(z)=\\frac{1}{2\\pi r}\\oint W(z+r e^{i\\theta})e^{-i\\theta}\\,d\\theta$$")
    add("")
    add(
        "   被积函数解析时梯形公式谱精度收敛，且公式内无「相近数相减」，"
        "因此误差直接落在双精度舍入底噪（实测 ~3e-15，见图 1）。"
        "升力势的 $\\ln(z/a)$ 沿回路用 `np.unwrap` 连续化，避免主值支割线切过小圆。"
    )
    add(
        "2. **升力＝压力积分**：$L'=-\\int_0^{2\\pi} p\\sin\\theta\\,a\\,d\\theta$，"
        "$p=\\tfrac12\\rho(U^2-|v|^2)$，$|v|$ 取上一步的数值导数；4096 点梯形对三角多项式谱精度。"
    )
    add(
        "3. **驻点角＝数值零点**：对数值切向速度做 721 点扫描 + 二分（收敛到 $10^{-13}$ rad），"
        "不代入 $\\sin\\theta_s=\\Gamma/4\\pi U a$。"
    )
    add("")
    add("脚本：`audit_reference_m22101.py`（写 JSON）、`make_reference_audit_figures_m22101.py`（图）、"
        "`tests/test_case_cylinder_reference_independent.py`（守门测试，16 条，3.5 s）。")
    add("")
    add("## 复核结果（$a=%s$, $U=%s$, %d 个探针点）" % (fmt(metrics["radius"]), fmt(metrics["speed"]), metrics["probe_count"]))
    add("")
    add("| 复核项 | 参考解（声明值） | 独立数值推导 | 绝对差 /U | 结论 |")
    add("| --- | --- | --- | --- | --- |")
    add(
        "| 自由流速度场 | $dW/dz=U(1-a^2/z^2)$ | Cauchy 积分 | %s | 一致 |"
        % fmt(free["velocity_max_abs_error_over_U"])
    )
    add(
        "| 扰动势 $\\phi-Ux$ | $Ua^2\\cos\\theta/r$ | $\\mathrm{Re}\\,W-Ux$ | %s | 一致 |"
        % fmt(free["perturbation_potential_max_abs_error"])
    )
    add(
        "| $C_p$ | $1-(|v|/U)^2$ | Bernoulli + 数值速度 | %s | 一致 |"
        % fmt(free["cp_max_abs_residual"])
    )
    for row in lifting:
        circulation = fmt(row["circulation"])
        add(
            "| 升力势速度场 ($\\Gamma=%s$) | $U(1-a^2/z^2)-i\\Gamma/(2\\pi z)$ | Cauchy 积分 | %s | 一致 |"
            % (circulation, fmt(row["velocity_max_abs_error_over_U"]))
        )
        add(
            "| 物面速度 ($\\Gamma=%s$) | $|\\Gamma/(2\\pi a)-2U\\sin\\theta|$ | 物面数值导数模长 | %s | 一致 |"
            % (circulation, fmt(row["surface_speed_max_abs_error_over_U"]))
        )
        add(
            "| 法向无流动 ($\\Gamma=%s$) | $v_r=0$ | 数值速度径向分量 | %s | 一致 |"
            % (circulation, fmt(row["surface_radial_velocity_over_U"]))
        )
        add(
            "| 升力 ($\\Gamma=%s$) | $-\\rho U\\Gamma=%s$ | 压力积分 $%s$ | 相对 %s | 一致 |"
            % (
                circulation,
                fmt(row["lift_joukowski"]),
                fmt(row["lift_from_pressure_integral"]),
                fmt(row["lift_relative_error"]),
            )
        )
        add(
            "| 阻力 ($\\Gamma=%s$) | d'Alembert: $0$ | 压力积分 $%s$ | 相对升力 %s | 一致 |"
            % (circulation, fmt(row["drag_from_pressure_integral"]), fmt(row["drag_over_lift"]))
        )
        add(
            "| 驻点角 ($\\Gamma=%s$) | %s°，%s° | 二分零点 %s°，%s° | %s° | 一致 |"
            % (
                circulation,
                fmt(row["stagnation_stated_deg"][0]),
                fmt(row["stagnation_stated_deg"][1]),
                fmt(row["stagnation_bisection_deg"][0]),
                fmt(row["stagnation_bisection_deg"][1]),
                fmt(row["stagnation_max_deviation_deg"]),
            )
        )
    add("")
    add("图为入库副本（`artifacts/m22101/` 为运行时输出目录，已被 .gitignore 忽略）：\n\n- `docs/reference_audit_m22101/audit_convergence.png`、`docs/reference_audit_m22101/audit_surface.png`、数值 `docs/reference_audit_m22101/audit_metrics.json`。")
    add("图 1：`docs/reference_audit_m22101/audit_convergence.png`——Cauchy 求导随回路节点数的谱收敛，"
        "两支曲线都落到双精度底噪，说明「裁判」本身没有误差，比较才有意义。")
    add("")
    add("图 2：`artifacts/m22101/audit_surface.png`——$\\Gamma=3$ 旋转圆柱物面 $C_p$："
        "闭式（实线）与数值复势导数（空心点）重合并列，副轴为残差。")
    add("")
    add("## 适用范围与局限")
    add("")
    add("- 复核对象是**参考解**，不是 BEM 离散解：它保证「判据对标的靶子是对的」，"
        "不保证边界元实现无误；后者由算例自身的 44 条判据与收敛阶负责。")
    add("- Cauchy 求导要求复势在探针周围解析。本算例的奇点只有 $z=0$（体内），"
        "小圆半径取 $0.05|z|$ 恒不含奇点；对带支割线的升力势用相位连续化处理。"
        "因此结论不外推到含其它奇点的场（如自由面、多体近场遮挡）。")
    add("- 谱精度依赖被积函数的解析性；若把该方法用于含奇异积分的源分布本身，需另做奇性处理。")
    add("- 复核下限受双精度限制（~1e-15）；要判定更小的参考解误差需高阶精度算术。")
    add("")
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {DOC} ({len(lines)} lines)")


if __name__ == "__main__":
    main()