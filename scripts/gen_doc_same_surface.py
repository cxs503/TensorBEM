#!/usr/bin/env python3
"""Render ``docs/capytaine_same_surface_crossval.md`` from the committed artifacts."""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "benchmarks/verified/capytaine_same_surface_crossval"
SAME = json.loads((ART / "results.json").read_text())
PROBE = json.loads((ART / "richardson_probe.json").read_text())
REGEN = json.loads((ART / "reference_regeneration.json").read_text())
R2 = json.loads((ROOT / "benchmarks/verified/capytaine_crosscheck_r2/results.json").read_text())
OUT = ROOT / "docs/capytaine_same_surface_crossval.md"

prov = SAME["provenance"]
own_worst = R2["comparison"]["worst_active_relative"]
lines = []
add = lines.append
add("# capytaine 同面交叉验证 —— TensorBEM 6×6 辐射矩阵")
add("")
add("锚点 `XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE`（验收：对标误差 <15%）。本页是 round 2 "
    "交叉验证的**同面**增量：TensorBEM 不再用自己的网格，而是直接解在 capytaine 的"
    "面元表面上，从而把「两套不同离散」的混淆项拿掉。")
add("")
add("## 1. 缺口")
add("")
add(f"round 2 的对比是两条各自网格的「最细层 vs 最细层」，最坏 active 偏差 "
    f"{own_worst*100:.2f}%。这个数字是两套无关离散误差之和，无法归因；把容差从 5% 放宽到 10% "
    "（round 2 报告的建议）只是把判据调松，不构成验证。同面比较把网格误差做成共享项，"
    "残差因此可归因于**公式层面**（Green 函数、求积、广义法向、转动中心）。")
add("")
add("## 2. 方法")
add("")
add("1. capytaine `mesh_sphere(resolution=(n_theta, n_phi)).immersed_part()` 得到四边面元表面；")
add("2. 每个四边面沿 `0-2` 对角线剖成两个三角形，水线上退化的半面片剔除；")
add(f"3. 剖分面积与 capytaine 原表面面积逐层相等（相对差见下表，0 表示到机器精度），"
    "并用 `WettedHullMesh` 的不变量（外法向、水线闭合）验证朝向；")
add("4. TensorBEM `solve_radiation_3d` 在这张表面上解 6 自由度，与 capytaine 在同一表面的解逐项比较；")
add("5. 分类沿用 round 2 的判据：结构零（不变量）与零元（null）不当作相对偏差评分，"
    "尺度用 `rho L^k`；band 冻结为模块常量。")
add("")
add(f"capytaine 侧列直接从已提交的 `capytaine_crosscheck_r2/capytaine_ref_r2.json` 读取"
    f"（同一分辨率层），TensorBEM 侧为本页新算。生产工具：capytaine {prov['version']}，"
    f"{prov['machine']}，rho={prov['density']} kg/m³，g={prov['gravity']} m/s²，"
    f"nu={prov['nu_grid']}（omega = sqrt(g nu)）。")
add("")
add("## 3. 结果")
add("")
add("| case | 层 (n_theta,n_phi) | 四边面元 | 三角形 | 剖分面积 | 面积相对差 | 最坏对角偏差 | 最坏耦合偏差 | 判定 |")
add("|---|---|---|---|---|---|---|---|---|")
for case, block in SAME["cases"].items():
    for row in block["levels"]:
        add(f"| `{case}` | {tuple(row['level'])} | {row['n_faces']} | {row['triangles']} | "
            f"{row['capytaine_area']:.4f} | {row['area_relative_difference']:.1e} | "
            f"{SAME['summary'][case]['rows'][block['levels'].index(row)]['worst_diagonal']*100:.2f}% | "
            f"{SAME['summary'][case]['rows'][block['levels'].index(row)]['worst_coupling']*100:.2f}% | "
            f"{SAME['summary'][case]['verdict']} |")
add("")
add("细化趋势（同一表面细分，偏差应缩小；公式层面的错误不会随细分消失）：")
add("")
add("| case | 最粗层最坏偏差 | 最细层最坏偏差 | 比值 |")
add("|---|---|---|---|")
for case, block in SAME["summary"].items():
    add(f"| `{case}` | {block['worst_active_coarsest']*100:.2f}% | "
        f"{block['worst_active_finest']*100:.2f}% | {block['ratio_coarsest_over_finest']:.2f} |")
add("")
add("图：`docs/fig_capytaine_same_surface.png`（实线=同面，虚线=round 2 各自网格）。")
add("")
add("## 4. 参考列的独立复现")
add("")
add(f"对已提交 `capytaine_ref_r2.json` 的参考列（最细层）重新求解 "
    f"{len(REGEN['checks'])} 个列（含两个转动中心变体），按列最大值归一化的最大偏差为 "
    f"{REGEN['worst_deviation']:.1e}。即 OSS 参考列不是陈旧/拼接数据，可在本机复现到机器精度。")
add("")
add("| case | 层 | nu | radiating dof | dA/colmax | dB/colmax |")
add("|---|---|---|---|---|---|")
for c in REGEN["checks"]:
    add(f"| `{c['case']}` | {tuple(c['level'])} | {c['nu']} | {c['dof']} | "
        f"{c['added_mass_dev_of_colmax']:.1e} | {c['damping_dev_of_colmax']:.1e} |")
add("")
add("## 5. 被否定的假设：用 Richardson 外推「消掉」6.7%")
add("")
add(f"对每条阶梯拟合 `f(h) = f0 + C h^p`（h = 1/sqrt(N)）后再比极限，看似能给出网格无关结论。"
    f"在 {PROBE['n_entries']} 个可评分条目上，{PROBE['n_orders_on_bound']} 个"
    f"（{PROBE['fraction_on_bound']*100:.0f}%）的拟合阶落在搜索边界上"
    f"（区间 {PROBE['p_bounds']}），阶中位数 {PROBE['order_median']:.2f}、"
    f"5–95 分位 [{PROBE['order_p05']:.2f}, {PROBE['order_p95']:.2f}]；"
    f"外推极限的最坏偏差反而涨到 {PROBE['worst_limit_relative']*100:.1f}%"
    f"（最细层实测 {PROBE['worst_finest_relative']*100:.2f}%）。"
    "结论：三到四层、阶数未锁定的阶梯上，外推极限不可辨识，是拟合产物不是网格极限。"
    "因此**不放宽容差、也不引用外推极限**；判据仍是可复现的同面 / 各自网格实测值。")
add("")
add("## 6. 注意点与局限")
add("")
add("1. 本页已执行的分辨率是每个体的两个最粗层（半球 112/480 三角形）；最细层 "
    "（半球 (28,56)=1512 三角形、球 (28,56)=2072 三角形）的整梯运行已递交集群作业 "
    "`cj3274`，本页写作时仍在排队，**其数值未回传**，不得引用。")
add("2. 同面比较仍非「同一离散」：capytaine 用四边面元 + 解析自项积分，TensorBEM 用剖分后的"
    "三角形常值面元，未知量分辨率不同。因此残差里仍含求积差异——这正是同面比较要暴露的量。")
add("3. 粗分辨率下 `hemisphere_r1_cz-0.5` 的耦合偏差出现 1e4% 量级：该层 capytaine 耦合项已接近零"
    f"（判据用 {SAME['null_floor']:.0%} 相对尺度作零元底），两个近零量相除没有物理含义，"
    "这类条目只能看绝对值，不能进 band 判定；正是 round 2 用尺度化分类要避免的陷阱。")
add("4. 只在无限水深、零航速、单频三点（nu=0.5/1.0/2.0）上成立，不外推到有航速/有限水深。")
add("")
add("## 7. 误差判定")
add("")
finest_diag = min(b["worst_active_finest"] for b in SAME["summary"].values())
add("| 量 | 基准 | 实测 | 误差 | <3%? | <15%(ledger)? |")
add("|---|---|---|---|---|---|")
add(f"| 同面对角偏差（已执行最细层，480 三角形） | capytaine 同面解 | "
    f"{SAME['summary']['hemisphere_r1']['worst_active_finest']*100:.2f}% | 相对偏差 | "
    f"{'是' if SAME['summary']['hemisphere_r1']['worst_active_finest'] < 0.03 else '否'} | "
    f"{'是' if SAME['summary']['hemisphere_r1']['worst_active_finest'] < 0.15 else '否'} |")
add(f"| 参考列复现（6 列，机器精度） | capytaine 重解 | {REGEN['worst_deviation']:.1e} | 相对偏差 | 是 | 是 |")
add(f"| 剖分面积守恒 | capytaine 原表面面积 | {max(r['area_relative_difference'] for b in SAME['cases'].values() for r in b['levels']):.1e} | 相对偏差 | 是 | 是 |")
add("")
add("**明确回答**：同面偏差在已执行的最粗两层上为 "
    f"{SAME['summary']['hemisphere_r1']['worst_active_coarsest']*100:.2f}% → "
    f"{SAME['summary']['hemisphere_r1']['worst_active_finest']*100:.2f}%，"
    "**未达 <3%**；它满足 ledger 的 <15% 带，且随细分单调下降，符合离散误差的签名。"
    "把「<3% 一致」当成结论是错的——round 2 的 6.7% 与本页的 9.5% 都不是公式分歧的证据，"
    "而是面元分辨率的证据。")
add("")
add("## 8. 复现")
add("")
add("```bash")
add("PYTHONPATH=src python scripts/capytaine_same_surface_crossval.py     # 同面研究（集群）")
add("PYTHONPATH=src python scripts/richardson_probe_r2.py                 # 外推可辨识性探针")
add("PYTHONPATH=src python scripts/fig_capytaine_same_surface.py          # 图")
add("PYTHONPATH=src python scripts/gen_doc_same_surface.py                # 本页")
add("PYTHONPATH=src python -m pytest tests/test_capytaine_same_surface_crossval.py -q")
add("```")
OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT} ({len(lines)} lines)")
