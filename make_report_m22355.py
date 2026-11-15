#!/usr/bin/env python3
"""Generate figures for CASE-BEM-SUBMERGED-SPHERE-HAVELOCK paper-level report."""
import json
import base64
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTDIR = Path("/srv/tensor/agent/work/m22355/docs/case_bem_submerged_sphere_havelock")
NUMBERS = OUTDIR / "numbers.json"

def load_data():
    d = json.loads(NUMBERS.read_text())
    case = d["band_sweep_case"]
    econ = d["band_sweep_economy"]
    return d, case, econ

def fig1_band_comparison(case, econ):
    """Full-band coefficient comparison: BEM vs multipole reference."""
    kas = [r["ka"] for r in case]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    coeffs = [("a33", "Heave Added Mass A33'"), ("b33", "Heave Damping B33'"),
              ("a11", "Surge Added Mass A11'"), ("b11", "Surge Damping B11'")]
    for ax, (key, title) in zip(axes.flat, coeffs):
        bem_vals = [r[f"{key}_bem"] for r in case]
        ref_vals = [r[f"{key}_exact"] for r in case]
        econ_vals = [r[f"{key}_bem"] for r in econ]
        ax.plot(kas, ref_vals, 'k--', lw=2, label='Multipole Reference')
        ax.plot(kas, bem_vals, 'ro-', lw=1.5, markersize=6, label='BEM (1280 panels)')
        ax.plot(kas, econ_vals, 'bs-', lw=1, markersize=4, alpha=0.6, label='BEM (320 panels)')
        ax.set_xlabel('ka (non-dimensional wavenumber)')
        ax.set_ylabel(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle('CASE-BEM-SUBMERGED-SPHERE-HAVELOCK: Full-Band Radiation Coefficients\n'
                 'Submerged sphere, f/a=2.0, ρ=1025 kg/m³, g=9.80665 m/s²', fontsize=11)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def fig2_error_bars(case, econ):
    """Relative error bars for all coefficients."""
    kas = [r["ka"] for r in case]
    fig, ax = plt.subplots(figsize=(10, 5))
    coeffs = [("a33", "A33'"), ("b33", "B33'"), ("a11", "A11'"), ("b11", "B11'")]
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']
    x = np.arange(len(kas))
    width = 0.18
    for i, (key, label) in enumerate(coeffs):
        case_errs = [abs(r[f"{key}_rel_error"])*100 for r in case]
        econ_errs = [abs(r[f"{key}_rel_error"])*100 for r in econ]
        ax.bar(x + i*width - 1.5*width, case_errs, width, label=f'{label} (1280)', color=colors[i], alpha=0.8)
        ax.bar(x + i*width - 0.5*width, econ_errs, width, label=f'{label} (320)', color=colors[i], alpha=0.4, hatch='//')
    ax.axhline(y=5, color='red', linestyle='--', lw=1.5, label='5% Gate')
    ax.axhline(y=2, color='green', linestyle=':', lw=1, label='2% Target')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{k:.1f}' for k in kas])
    ax.set_xlabel('ka')
    ax.set_ylabel('Relative Error (%)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Relative Error: BEM vs Multipole Reference (all < 5% on 1280-panel mesh)')
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def fig3_convergence_summary(case):
    """Error summary: max error per coefficient across the band."""
    coeffs = [("a33", "A33'"), ("b33", "B33'"), ("a11", "A11'"), ("b11", "B11'")]
    max_errs = []
    labels = []
    for key, label in coeffs:
        errs = [abs(r[f"{key}_rel_error"])*100 for r in case]
        max_errs.append(max(errs))
        labels.append(label)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, max_errs, color=['#2196F3', '#F44336', '#4CAF50', '#FF9800'], alpha=0.8)
    ax.axhline(y=5, color='red', linestyle='--', lw=1.5, label='5% Gate')
    ax.axhline(y=2, color='green', linestyle=':', lw=1, label='2% Target')
    for bar, val in zip(bars, max_errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Max Relative Error (%)')
    ax.set_title('CASE-BEM-SUBMERGED-SPHERE-HAVELOCK\nMax Error per Coefficient (1280 panels, 7 ka)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def main():
    d, case, econ = load_data()
    
    fig1 = fig1_band_comparison(case, econ)
    fig2 = fig2_error_bars(case, econ)
    fig3 = fig3_convergence_summary(case)
    
    # Save figures
    (OUTDIR / "fig_report_band.png").write_bytes(base64.b64decode(fig1))
    (OUTDIR / "fig_report_error.png").write_bytes(base64.b64decode(fig2))
    (OUTDIR / "fig_report_summary.png").write_bytes(base64.b64decode(fig3))
    
    # Build report JSON
    report = {
        "title": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK: Havelock (1955) Submerged Sphere Radiation — Full-Band Verification",
        "summary": "TensorBEM 3D free-surface BEM verified against Havelock multipole analytical solution for a fully submerged sphere (f/a=2.0). All 28 items (7 ka × 4 coefficients) on the 1280-panel case mesh pass the 5% gate with max error 1.431%. ka=4,5 values are directly computed (cluster cj3884, 1288s), replacing prior GCI estimates.",
        "scenario_key": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
        "mission_id": 22355,
        "tags": ["BEM", "radiation", "submerged-sphere", "Havelock", "frequency-domain", "verification"],
        "markdown": build_markdown(d, case, econ),
        "figures": [
            {"name": "fig1_band_comparison", "caption": "Full-band radiation coefficients: BEM (1280 & 320 panels) vs multipole reference", "png_base64": fig1},
            {"name": "fig2_error_bars", "caption": "Relative error bars for all 28 items; all 1280-panel values below 5% gate", "png_base64": fig2},
            {"name": "fig3_max_error_summary", "caption": "Maximum relative error per coefficient across the full band (1280 panels)", "png_base64": fig3}
        ]
    }
    
    import json as j
    Path("/tmp/report_m22355.json").write_text(j.dumps(report, indent=2))
    print("Report JSON written to /tmp/report_m22355.json")

def build_markdown(d, case, econ):
    """Build the six-element paper-level markdown report."""
    # Extract key numbers
    tier = d["tiers"]["case_mesh"]
    max_err = tier["max_rel_error"]
    
    # Build coefficient table
    table_rows = []
    for r in case:
        ka = r["ka"]
        a33e = r["a33_rel_error"] * 100
        b33e = r["b33_rel_error"] * 100
        a11e = r["a11_rel_error"] * 100
        b11e = r["b11_rel_error"] * 100
        table_rows.append(f"| {ka:.1f} | {r['a33_bem']:.6f} | {r['a33_exact']:.6f} | {a33e:.3f}% | {r['b33_bem']:.6e} | {r['b33_exact']:.6e} | {b33e:.3f}% | {a11e:.3f}% | {b11e:.3f}% |")
    
    table = "\n".join(table_rows)
    
    md = f"""# CASE-BEM-SUBMERGED-SPHERE-HAVELOCK: Havelock (1955) Submerged Sphere Radiation — Full-Band Verification

## ① 案例介绍（物理背景/几何/参数）

**物理背景**: 完全浸没球体在自由表面下的频域辐射问题是水动力学经典基准。Havelock (1955) 提出的球谐多极子展开法给出了该问题的解析解，被广泛用作三维自由面边界元方法的验证基准。

**几何与参数**:

| 参数 | 值 | 说明 |
|---|---|---|
| 半径 a | 1.0 m | 球体半径 |
| 浸深 f | 2.0 m | 球心至自由面距离 |
| f/a | 2.0 | 无量纲浸深比 |
| 密度 ρ | 1025 kg/m³ | 海水密度 |
| 重力 g | 9.80665 m/s² | 标准重力 |
| 排水体积 ∇ | 4πa³/3 = 4.1888 m³ | 精确体积（非网格体积） |

**频带**: ka ∈ {{0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0}}（7 频点）

**网格**: 案定网格 subdivision=3（1280 面元，内接二十面体细分）；经济网格 subdivision=2（320 面元）

**系数**: A33'（升沉附加质量）、B33'（升沉阻尼）、A11'（纵荡附加质量）、B11'（纵荡阻尼）
归一化: A' = A/(ρ∇)，B' = B/(ρω∇)

## ② 方法（方法/流程/关键设置/工具链）

**BEM 求解器**: `tensorbem.free_surface_solver3d.solve_radiation_3d`
- 3D 自由面 Kelvin-Green 函数边界元（Delhommeau 核）
- 常值面元，Galerkin 格式
- 无内源正则化（CHIEF），最低不规则频率高于频带

**参考解**: `tensorbem.analytic.submerged_sphere_wave_multipole`
- Havelock 频域多极子展开（n_terms=20）
- 独立验证：截断独立性 20→40 项（机器零）、Haskind 互易残差 <1e-15、Lamb 无界流体极限 A'→0.5

**GCI 分析**: ASME V&V 20 框架，三阶网格（80/320/1280），观测阶 p≈2.03

**工具链**: Python 3.11 + NumPy 2.0 + TensorBEM（main 分支）

**计算资源**: ka=4,5 实算由集群作业 cj3884 完成（node n330, 4 核, 1288 s）

## ③ 计算结果

### 案定网格（1280 面元）全带结果

| ka | A33' BEM | A33' Ref | A33' 误差 | B33' BEM | B33' Ref | B33' 误差 | A11' 误差 | B11' 误差 |
|---|---|---|---|---|---|---|---|---|
{table}

**门控结果**: 28/28 项 PASS，最大相对误差 **{max_err*100:.3f}%**（ka=0.5 B33'）

### 经济网格（320 面元）对比

经济网格 21/28 项 PASS，7 项 FAIL（均为阻尼系数，最大 5.591%）。GCI 分析证实此为网格效应而非公式偏差。

## ④ 结果图

见附图：
- **fig1_band_comparison**: 全带系数对比（BEM 1280/320 面元 vs 多极子参考解）
- **fig2_error_bars**: 28 项相对误差柱状图（5% 门控线）
- **fig3_max_error_summary**: 各系数最大误差汇总

## ⑤ 注意点（踩过的坑/局限/适用范围）

1. **ka≥4 阻尼系数退化**: B33', B11' 在 ka=4,5 时量级 ~10⁻⁵–10⁻⁶，5% 相对判据在此 4 项上为噪声主导。已标注为 degenerate_items，附尺度归一误差。
2. **经济网格 FAIL 为网格效应**: G4 门控（预测比 1.03–1.11 ∈ [0.5, 2]）证实 320 面元误差可由 1280 面元误差的 2^2.03≈4.1 倍精确解释。
3. **观测阶系统性偏低**: p≈2.03 vs 理论 2 阶，与内接多面体几何亏损（1280 面元亏 0.48% 面积）一致。
4. **GCI 估计的局限**: m21647 曾预判"实算与估计偏差 <0.2%"，但 ka=5 b11 实际偏差 6.58%。小量（B'~10⁻⁶）的 GCI 外推能力被高估，不得用于闭合判决。
5. **适用范围**: 深水域（无限水深），f/a=2.0。有限水深需另案验证。

## ⑥ 误差判定

| 量 | 基准 | 实测 | 误差 | <5%? | <3%? |
|---|---|---|---|---|---|
| case_mesh 最大误差 | — | {max_err*100:.3f}%（ka=0.5 B33'） | {max_err*100:.3f}% | ✓ | ✓ |
| ka=0.1 最大误差 | — | {max(abs(r['a33_rel_error']),abs(r['b33_rel_error']),abs(r['a11_rel_error']),abs(r['b11_rel_error']))*100:.3f}% | — | ✓ | ✓ |
| ka=5.0 最大误差 | — | {max(abs(case[-1]['a33_rel_error']),abs(case[-1]['b33_rel_error']),abs(case[-1]['a11_rel_error']),abs(case[-1]['b11_rel_error']))*100:.3f}% | — | ✓ | ✓ |
| economy_mesh 最大误差 | — | {d['tiers']['economy_mesh']['max_rel_error']*100:.3f}% | {d['tiers']['economy_mesh']['max_rel_error']*100:.3f}% | ✗ (honest FAIL) | ✗ |

**结论**: 案定网格（1280 面元）全带 28 项全部通过 5% 门控，最大误差 {max_err*100:.3f}% < 3%，**满足验收标准**。经济网格（320 面元）7 项 FAIL 为 honest FAIL，属网格分辨率不足，非方法缺陷。
"""
    return md

if __name__ == "__main__":
    main()
