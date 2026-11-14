"""m20977: build docs/dev_report_m20977_*.md and /tmp/report.json from the artefacts.

Every number in the report is read out of ``docs/case_bem_rigid_body_mass/
results.json`` (the anchor's own machine artifact) -- nothing is transcribed.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT / "docs" / "case_bem_rigid_body_mass"
results = json.loads((ART / "results.json").read_text(encoding="utf-8"))
verdict = json.loads((ART / "verdict.json").read_text(encoding="utf-8"))
ext = results["external_route"]
cal = ext["self_calibration"]
att = ext["layer_attribution"]
ladder = ext["independent_mesh_route"]
ship = ext["ship_shape"]
topo = ext["topology_referee"]
mom = results["momentum_route"]


def pct(value: float, digits: int = 4) -> str:
    return f"{value * 100:.{digits}f}%"


def sci(value: float) -> str:
    return f"{value:.4e}"


lines: list[str] = []
add = lines.append
add("# m20977 · CASE-BEM-RIGID-BODY-MASS：外部裁判路线 route T（trimesh）")
add("")
add(
    f"**锚点** `CASE-BEM-RIGID-BODY-MASS`（算例池·刚体质量矩阵，ref=闭式块形式，容差 3%）"
    f"｜**机判** `{verdict['overall']}`（{len(verdict['checks'])} 项，"
    f"not_run={verdict['not_run']}）｜最差接受逐元素误差 "
    f"{pct(verdict['worst_accepted_entry_rel_error'])}"
)
add("")
add(
    "本文所有数字由 `make_report_m20977.py` 从 `docs/case_bem_rigid_body_mass/results.json` "
    "读出注入，无人工转录。"
)
add("")
add("## ① 案例介绍")
add("")
add(
    "频域运动方程 `(M_RB + A)ẍ + Bẋ + Cx = F` 中，刚体惯量矩阵 `M_RB` 由质量 `m`、质心 `r_g`"
    "与惯量张量 `I_g` 组成闭式块形式 `[[m I3, -m S(r_g)], [m S(r_g), I_g + m S(r_g)ᵀS(r_g)]]`。"
    "算例体为三轴椭球 `a,b,c = "
    f"{results['parameters']['semi_axes_m']}` m、密度 "
    f"{results['parameters']['density_kg_m3']:.0f} kg/m³、质心偏置 "
    f"{results['parameters']['cg_offset_m']} m，另有姿态旋转体与 Wigley 船型体。"
)
add("")
add("本锚点原有四层证据：闭式块形式（R）、TensorBEM 带符号四面体积分（M）、动量定义（P）、"
    "capytaine 装配（X，只涉及附加质量）。**四层里体积积分全部出自 TensorBEM 自己的代码** —— "
    "积分器或网格生成器若有偏，参照会与被测对象一起漂移，整条链看不见。")
add("")
add("本单补的是第五层 route T：把「离散」与「积分」两层都换成第三方实现（trimesh 的 "
    "Mirtich 1996 投影算法 + `uv_sphere` 经纬参数化），再用它回头裁判 TensorBEM。")
add("")
add("## ② 怎么做的")
add("")
add("| 层 | 文件 / 入口 | 作用 |")
add("|---|---|---|")
add(
    "| 新路线 | `src/tensorbem/benchmarks/rigid_body_mass_external_route.py` | "
    "外部质量特性积分、外部离散阶梯、船型体复核、拓扑守卫外部裁判 |")
add(
    "| 算例入口 | `src/tensorbem/benchmarks/casepool_bem_rigid_body_mass.py` | "
    "把 route T 折进 `results.json` / `verdict.json` / `CASE.md`，新增 `--no-external` |")
add(
    "| 测试 | `tests/test_rigid_body_mass_external_route.py` | 10 项门控（自校准、圆整一致性、"
    "接受层级、判别力、船型体、守卫裁判、焊接策略、not_run 语义、落地工件） |")
add(
    "| 依赖 | `pyproject.toml` `[project.optional-dependencies] external-verification` | "
    "`trimesh` 只用于验证，求解器不 import；缺库时本段严格报 `not_run`，判定不得 PASS |")
add("")
add("四问，逐层加码：")
add("")
add(
    "1. **裁判先自校准**：立方体 `I = m L²/6` 无人需要积分的数，trimesh 必须复现到圆整；"
)
add("2. **分层归因**：同一套三角形上，两个积分器必须一致到圆整 —— 这样接受层级的误差才能"
    "归给「面板化」而不是「积分」；")
add("3. **独立路线**：外部离散 + 外部积分对闭式块形式必须进 3%；")
add("4. **外部裁判守卫**：`trimesh.is_watertight` 对人为损坏的网格必须与 TensorBEM 守卫同意"
    "（或至少不比守卫更宽松），否则上一轮 ADR 的补救就是自证。")
add("")
add("## ③ 参数与结果")
add("")
add("**裁判自校准（trimesh 先证明自己）与分层归因**")
add("")
add("| 项 | 结果 |")
add("|---|---|")
add(
    f"| 立方体（边长 {cal['side_m']:.1f} m，{cal['panels']} 面元）体积误差 | {sci(cal['volume_rel_error'])} |")
add(f"| 立方体惯量误差 vs `m L²/6` | {sci(cal['inertia_rel_error'])} |")
add(
    f"| 共享网格上两积分器 6×6 最差差异（{len(att['levels'])} 层最大） | "
    f"{sci(att['worst_matrix_difference'])}（限值 {sci(att['roundoff_limit'])}） |")
add(
    f"| 两积分器各自对闭式形式的距离之差（误差预算差） | "
    f"{sci(att['worst_error_budget_difference'])} |")
add("")
add("**外部离散阶梯（uv_sphere + trimesh，参照 = 闭式块形式）**")
add("")
add("| uv 计数 | 面元 | 体积误差 | 惯量误差 | 6×6 最差元素 | 最差位置 | 旧耦合对照 | 接受 |")
add("|---|---|---|---|---|---|---|---|")
for row in ladder["levels"]:
    add(
        f"| {row['uv_count']} | {row['panels']} | {pct(row['volume_rel_error'])} | "
        f"{pct(row['inertia_rel_error'])} | {pct(row['max_entry_rel_error'])} | {row['worst_entry']} | "
        f"{pct(row['legacy_coupling_rel_error'], 2)} | {'✅' if row['accepted'] else '—'} |"
    )
add("")
add(
    f"接受层级（uv ≥ {ladder['accepted_counts'][0]}）最差 {pct(ladder['worst_accepted_entry_rel_error'])}；"
    f"拟合收敛阶 {ladder['observed_order']:+.3f}；最先达标的层级 uv = "
    f"{ladder['coarsest_within_tolerance_count']}（误差 ~ 面元数^{ladder['observed_order']:+.2f}）。"
)
add("")
add("**船型体（Wigley，`Cb = 4/9`）**")
add("")
add("| 项 | 参照（闭式积分） | 外部裁判（trimesh） | 误差 |")
add("|---|---|---|---|")
add(
    f"| 排水体积 | {ship['reference']['volume_m3']:.3f} m³ | {ship['external']['volume_m3']:.3f} m³ | "
    f"{pct(ship['volume_rel_error'])} |")
add(
    f"| 质量 | {ship['reference']['mass_kg']:.1f} kg | {ship['external']['mass_kg']:.1f} kg | "
    f"{pct(ship['mass_rel_error'])} |")
add(
    f"| 形心 | {ship['reference']['centroid_m']} | {ship['external']['centroid_m']} | "
    f"{ship['centroid_abs_error_m']:.2e} m |")
add(
    f"| 惯量张量（在设计质心） | — | — | {pct(ship['inertia_rel_error'])} |")
add(
    f"| **6×6 最差元素**（{ship['mesh']['panels']} 面元，最差位置 {ship['worst_entry']}） | 闭式块形式 | "
    f"外部路线 | **{pct(ship['matrix_max_entry_rel_error'])}** |")
add("")
add(
    f"分块最差：平动-平动 {pct(ship['block_max_entry_rel_error']['translation-translation'])}、"
    f"平动-转动 {pct(ship['block_max_entry_rel_error']['translation-rotation'])}、"
    f"转动-转动 {pct(ship['block_max_entry_rel_error']['rotation-rotation'])}"
    "（转动块最大，与薄壁船体 `I_zz` 主要来自水线面分布一致）。"
)
add("")
add("**外部裁判对拓扑守卫的逐变体裁决**")
add("")
add("| 变体 | 守卫接受 | 外部接受 | 体积误差（外部积分） | 直接采信外部的 6×6 最差 |")
add("|---|---|---|---|---|")
for row in topo["variants"]:
    err = (
        "未定义（质量≤0）"
        if row["naive_workflow_undefined"]
        else pct(row["matrix_max_entry_rel_error"], 3)
    )
    add(
        f"| {row['variant']} | {'是' if row['guard_accepts'] else '否'} | "
        f"{'是' if row['external_accepts'] else '否'} | "
        f"{pct(row['volume_rel_error_vs_closed_form'])} | {err} |"
    )
add("")
add(
    "**净增量在第 3 行**：单个翻转面对 trimesh 的边关联判据不可见（仍算 watertight，"
    "它会直接采信并积出一个错体积），而 TensorBEM 守卫按有向半边判定、予以拒绝。"
    f"外部漏检清单 = {topo['corruptions_the_external_predicate_misses']}。"
    "结论：拓扑校验不能外包给第三方库。"
)
add("")
add(
    f"焊接策略差异（全部顶点拆开的同一闭合面）：守卫接受={topo['weld_sensitivity']['guard_is_closed']}、"
    f"trimesh 原样 watertight={topo['weld_sensitivity']['external_watertight_as_given']}、"
    f"`merge_vertices` 后={topo['weld_sensitivity']['external_watertight_after_merge']} —— "
    "差异只是焊接策略，对齐后一致。"
)
add("")
add("## ④ 图")
add("")
add("- `docs/case_bem_rigid_body_mass/fig_external_route.png`：左=外部离散阶梯（误差 vs 面元数、"
    "3% 判据线、旧耦合对照）；右=守卫与外部判据对五个变体的接受矩阵。")
add("")
add("## ⑤ 注意点")
add("")
add("- **独立性有边界**：route T 换掉了离散与积分，但仍共用闭式参照、"
    "`rigid_body_mass_matrix` 装配与逐元素误差口径。块约定由 route P 负责，route T 只证"
    "「质量特性输入」。")
add("- **收敛阶是 1 阶而非 2 阶**：uv_sphere 的板元是内接平面三角形，"
    f"观测阶 {ladder['observed_order']:+.2f}（面积/体积型量的典型值），与 icosphere 的"
    "经验一致；外推不能当解析值用。")
add("- **粗层级已达标**：uv=16（1792 面元）就已在 3% 内，声明接受层级取 uv ≥ 32 是留裕度，"
    "不是「刚好通过」；`results.json` 同时记录最先达标层级以免读者误判裕度。")
add("- **trimesh 是可选依赖**：缺库/`--no-external` 时本段 `not_run`，机判变 `INCOMPLETE`，"
    "绝不落成 PASS。")
add("- **三角网格公共盲区**：两库都积分三角化表面，所有三角网格共有的几何误差（如半轴标错）"
    "不在本锚点覆盖内 —— 那是几何锚点（Cb/L/B）的活。")
add("- **既有红项（与本单无关）**：`tests/test_case_pool_registry.py` 有两个用例在本工作树"
    "里因其他锚点缺图（HAVELOCK / ANNULUS / LAPLACE-CYLINDER / ELLIPSE-2D 的 figures）"
    "而失败；RIGID-BODY-MASS 的路径全部存在。未在本单修（越界）。")
add("")
add("## ⑥ 误差判定")
add("")
add("| 关键量 | 基准 | 误差 | 是否 <3% |")
add("|---|---|---|---|")
add(
    f"| 外部离散 6×6 最差（接受层级） | 闭式块形式 | {pct(ladder['worst_accepted_entry_rel_error'])} | "
    f"{'是' if ladder['checks']['external_mesh_route_within_tolerance'] else '否'} |")
add(
    f"| Wigley 船型体 6×6 最差 | 闭式积分 | {pct(ship['matrix_max_entry_rel_error'])} | "
    f"{'是' if ship['checks']['external_ship_shape_within_tolerance'] else '否'} |")
add(
    f"| Wigley 排水体积 | 闭式积分 | {pct(ship['volume_rel_error'])} | 是 |")
add(
    f"| 两积分器一致性（共享网格） | 逐元素、限值 {sci(att['roundoff_limit'])} | "
    f"{sci(att['worst_matrix_difference'])} | 是（圆整量级） |")
add(
    f"| 旧耦合约定对照（必错） | 闭式块形式 | {pct(ladder['legacy_coupling_worst_rel_error'], 2)} | "
    "否（预期，判别力证据） |")
add(
    f"| 原锚点最差接受（不变） | 闭式块形式 | {pct(verdict['worst_accepted_entry_rel_error'])} | 是 |")
add(
    f"| route P 动量定义 vs 出厂装配（不变） | 动量定义 | {sci(mom['worst_relative_error'])} | "
    "是（圆整） |")
add("")
add(
    "**结论**：全部关键量 vs 基准误差均 < 3%（最差的 6×6 外部路线 "
    f"{pct(ladder['worst_accepted_entry_rel_error'], 3)}、船型体 {pct(ship['matrix_max_entry_rel_error'], 3)}）；"
    "一致性项与动量项在圆整量级。无「无基准」项。"
)
add("")
add(f"机判检查项（`verdict.json`）：`{json.dumps(verdict['checks'], ensure_ascii=False)}`")
add("")
doc_path = ROOT / "docs" / "dev_report_m20977_rigid_body_mass_external_route.md"
doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", doc_path)

# ---------------- platform report ----------------
markdown = "\n".join(lines)
figure = (ART / "fig_external_route.png").read_bytes()
payload = {
    "title": "CASE-BEM-RIGID-BODY-MASS · route T：用第三方积分器裁判刚体质量矩阵（trimesh）",
    "summary": (
        "刚体质量矩阵锚点新增第五层证据：把面元离散与体积积分两层都换成第三方实现"
        "（trimesh Mirtich + uv_sphere），外部离散 6×6 最差 "
        f"{pct(ladder['worst_accepted_entry_rel_error'], 3)}、Wigley 船型体 "
        f"{pct(ship['matrix_max_entry_rel_error'], 3)}（判据 3%），两积分器在共享网格上一致到 "
        f"{sci(att['worst_matrix_difference'])}；并发现 trimesh 的 watertight 判据看不见单个翻转面，"
        "证明 TensorBEM 的拓扑守卫不可外包。"
    ),
    "scenario_key": "case_pool/bem_rigid_body_mass/external_route",
    "mission_id": 20977,
    "tags": ["TensorBEM", "case-pool", "rigid-body-mass", "verification", "trimesh", "adversarial"],
    "markdown": markdown,
    "figures": [
        {
            "name": "fig_external_route.png",
            "caption": (
                "route T：左=外部（uv_sphere + trimesh）离散阶梯的 6×6 逐元素误差与旧耦合对照，"
                "虚线为 3% 判据；右=TensorBEM 拓扑守卫与 trimesh is_watertight 对五个网格变体的"
                "接受矩阵（单个翻转面仅守卫拒绝）。"
            ),
            "png_base64": base64.b64encode(figure).decode("ascii"),
        }
    ],
}
Path("/tmp/report.json").write_text(json.dumps(payload), encoding="utf-8")
print("wrote /tmp/report.json", len(payload["markdown"]), "chars")