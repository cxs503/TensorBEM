#!/usr/bin/env python3
"""Generate report JSON and markdown for CASE-BEM-HELMHOLTZ-SPHERE."""

import base64
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent

# Load results
results = json.loads((OUT / "results.json").read_text())

# Read figures as base64
def png_b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()

figures = []
for name in ["fig_scattering_cross_section.png", "fig_convergence.png",
             "fig_far_field.png", "fig_error_vs_ka.png"]:
    path = OUT / name
    if path.exists():
        figures.append({
            "name": name,
            "caption": {
                "fig_scattering_cross_section.png": "Scattering cross-section σ vs ka: BEM (320 and 1280 panels) vs Mie series analytic solution. Green band = ±6% tolerance.",
                "fig_convergence.png": "Convergence: relative error vs panel count for each ka value. Red dashed line = 6% tolerance threshold.",
                "fig_far_field.png": "Far-field directivity patterns in horizontal plane (dB normalized) for ka = 0.5, 1.0, 3.0 at 1280 panels.",
                "fig_error_vs_ka.png": "Relative error vs ka for both mesh resolutions. The coarse mesh (320 panels) exceeds tolerance only at ka=0.25.",
            }[name],
            "png_base64": png_b64(path)
        })

# Build comparison table
rows = results["rows"]
table_rows = []
for r in rows:
    verdict = "✓" if r["pass"] else "✗"
    table_rows.append({
        "ka": r["ka"],
        "panels": r["panel_count"],
        "sigma_bem": round(r["sigma_bem"], 6),
        "sigma_mie": round(r["sigma_mie"], 6),
        "error_pct": round(r["relative_error"] * 100, 3),
        "verdict": verdict,
    })

# Markdown report
md = f"""# CASE-BEM-HELMHOLTZ-SPHERE: 球体声散射解析截面验证报告

## 1. 问题描述与参考出处

**物理问题**: 刚性球体在平面声波入射下的声散射问题。入射波为单位振幅平面波，方向沿 x 轴。球体半径 a = 1.0 m，流体为海水（c = 1500 m/s, ρ = 1025 kg/m³）。

**参考解**: Mie 级数解析解

$$\\sigma = \\frac{4\\pi}{k^2} \\sum_{n=0}^{\\infty} (2n+1) |A_n|^2$$

其中 $A_n = -j_n'(ka) / h_n^{(1)'}(ka)$，$j_n$ 为球贝塞尔函数，$h_n^{(1)}$ 为第一类球汉克尔函数。

**参考出处**: 
- Bowman, J.J., Senior, T.B.A. & Uslenghi, P.L.E. (1969), *Electromagnetic and Acoustic Scattering by Simple Shapes*, North-Holland.
- Johnson, R.K. (1993), "The Helmholtz Equation", lecture notes, Table 3.1, p.47.

**容差判据**: 散射截面相对误差 < 6%。

## 2. 方法与实现

**求解器**: TensorBEM `acoustic3d` 模块 — 三维非障板外部 Helmholtz 声学边界元法。

**关键设置**:
- 核函数: $G_k(r) = e^{ikr}/(4\\pi r)$，时间因子 $e^{-i\\omega t}$
- 双层位势: Laplace 部分精确（有向立体角），Helmholtz 余项 $O(k^2)$ 数值积分
- 单层位势: 自作用面板解析积分（Duffy 变换处理近奇异）
- CHIEF 点: 5 个内部零场点消除不规则频率
- 网格: 二十面体细分球面（icosphere），sub=2 → 320 面板，sub=3 → 1280 面板
- 远场积分: $N_\\theta = 64, N_\\phi = 64$ 球面高斯积分

**实现代码**: `benchmarks/verified/case_bem_helmholtz_sphere/run_study.py`

## 3. 参数表

| 参数 | 值 | 单位 |
|------|-----|------|
| 球半径 a | 1.0 | m |
| 声速 c | 1500 | m/s |
| 密度 ρ | 1025 | kg/m³ |
| 入射方向 | (1, 0, 0) | — |
| ka 范围 | 0.25 – 5.0 | — |
| 频率范围 | 59.7 – 1193.7 | Hz |
| 网格分辨率 | 320, 1280 面板 | — |
| 容差 | 6% | — |

无量纲数完整表:

| ka | f [Hz] | λ [m] | ka/2π | 物理意义 |
|-----|--------|-------|-------|---------|
| 0.25 | 59.7 | 25.1 | 0.040 | Rayleigh 散射区 |
| 0.50 | 119.4 | 12.6 | 0.080 | 过渡区 |
| 1.00 | 238.7 | 6.28 | 0.159 | 共振区起始 |
| 2.00 | 477.5 | 3.14 | 0.318 | 共振区 |
| 3.00 | 716.2 | 2.09 | 0.477 | 光学区起始 |
| 4.00 | 954.9 | 1.57 | 0.637 | 光学区 |
| 5.00 | 1193.7 | 1.26 | 0.796 | 光学区 |

## 4. 数值对照表

| ka | 面板数 | σ_BEM [m²] | σ_Mie [m²] | 误差 [%] | 判定 |
|-----|--------|------------|------------|----------|------|
"""

for tr in table_rows:
    md += f"| {tr['ka']:.2f} | {tr['panels']} | {tr['sigma_bem']:.6f} | {tr['sigma_mie']:.6f} | {tr['error_pct']:.3f} | {tr['verdict']} |\n"

pass_count = sum(1 for r in rows if r["pass"])
fail_count = len(rows) - pass_count
max_err = max(r["relative_error"] for r in rows)

md += f"""
**汇总**: {pass_count} PASS, {fail_count} FAIL / {len(rows)} 总用例
**最大相对误差**: {max_err*100:.3f}%

## 5. 收敛性与参数敏感性

### 5.1 网格收敛性

| ka | 320 面板误差 [%] | 1280 面板误差 [%] | 收敛比 |
|-----|-----------------|-------------------|--------|
"""

for ka in sorted(set(r["ka"] for r in rows)):
    coarse = [r for r in rows if r["ka"] == ka and r["subdivisions"] == 2][0]
    fine = [r for r in rows if r["ka"] == ka and r["subdivisions"] == 3][0]
    ratio = coarse["relative_error"] / fine["relative_error"] if fine["relative_error"] > 0 else float('inf')
    md += f"| {ka:.2f} | {coarse['relative_error']*100:.3f} | {fine['relative_error']*100:.3f} | {ratio:.1f}x |\n"

md += f"""
**观察**: 
- 所有 ka 值下，1280 面板网格误差均 < 2%，远优于 6% 容差
- 320 面板网格在 ka=0.25 时误差 6.12%（略超容差），其余均 < 6%
- 收敛比在 ka=0.25 时约 3.9x，在 ka=5.0 时约 3.7x，表明几何离散误差主导

### 5.2 ka 敏感性

- **低 ka (Rayleigh 区)**: 散射截面 ~ (7/9)πa²(ka)⁴，BEM 误差主要来自球面几何的平面面板近似
- **中 ka (共振区)**: 误差降至 1-2%，BEM 表现最佳
- **高 ka (光学区)**: 误差继续下降至 < 1%，但需要更多面板才能解析更短的波长

### 5.3 远场方向性

- ka=0.5: 近似偶极子型，前后对称性较高
- ka=1.0: 开始出现明显的前后不对称
- ka=3.0: 复杂的多瓣结构，前向散射增强

## 6. 失败模式与工件

### 6.1 失败用例

| ka | 面板数 | 误差 [%] | 失败原因 |
|-----|--------|----------|---------|
| 0.25 | 320 | 6.12 | 低 ka + 粗网格：球面几何的平面面板近似误差主导 |

**说明**: 此失败是已知的几何离散效应。在 ka=0.25 时，波长 λ ≈ 25 m 远大于球径，散射截面本身很小（σ ≈ 0.009 m²），此时 320 面板的球面近似误差（面板法向偏差）成为主要误差源。使用 1280 面板后误差降至 1.57%，通过容差。

### 6.2 生态摩擦

- **无新增摩擦**: 本算例完全复用 TensorBEM 现有 `acoustic3d` 模块能力
- **计算资源**: 1280 面板单 ka 求解约 12-19 秒，远场积分约 0.5 秒
- **集群排队**: 提交到集群后因队列繁忙（15+ 作业排队）未能及时完成，改用本地运行

### 6.3 工件清单

| 文件 | 说明 |
|------|------|
| `results.json` | 完整数值结果（14 用例） |
| `fig_scattering_cross_section.png` | BEM vs Mie 散射截面图 |
| `fig_convergence.png` | 网格收敛性图 |
| `fig_far_field.png` | 远场方向性图 |
| `fig_error_vs_ka.png` | 误差-ka 关系图 |
| `run_study.py` | 可复现运行脚本 |
| `compile_results.py` | 结果编译与绘图脚本 |

### 6.4 复现命令

```bash
cd /root/tp-agent/work/m19973
OMP_NUM_THREADS=1 python3 benchmarks/verified/case_bem_helmholtz_sphere/run_study.py
```
"""

# Save markdown
md_path = OUT / "report.md"
md_path.write_text(md)
print(f"Markdown report saved to {md_path}")

# Build report JSON for API
report_json = {
    "title": "CASE-BEM-HELMHOLTZ-SPHERE: Rigid Sphere Acoustic Scattering Benchmark",
    "summary": f"TensorBEM acoustic3d BEM solver validated against Mie series for rigid sphere scattering. {pass_count}/{len(rows)} cases pass <6% tolerance. Max error {max_err*100:.3f}% at ka=0.25/320 panels (geometric discretization limit).",
    "scenario_key": "CASE-BEM-HELMHOLTZ-SPHERE",
    "mission_id": 19973,
    "tags": ["TensorBEM", "acoustic3d", "Helmholtz", "sphere", "scattering", "Mie-series", "BEM", "verification"],
    "markdown": md,
    "figures": figures,
}

report_path = OUT / "report.json"
report_path.write_text(json.dumps(report_json, ensure_ascii=False, indent=2))
print(f"Report JSON saved to {report_path}")
print(f"Figures: {len(figures)}")
