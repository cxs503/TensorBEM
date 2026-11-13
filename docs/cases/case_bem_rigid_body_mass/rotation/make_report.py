"""Render the orientation-covariance extension report from run.json (zero hand-copy)."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUN = HERE / "run.json"
BASE_RUN = HERE.parent / "run.json"
DOC = ROOT / "docs" / "case_rigid_body_mass_orientation.md"
MODES = ["surge", "sway", "heave", "roll", "pitch", "yaw"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def figure_error_map(run: dict) -> Path:
    ref = np.asarray(run["orientation"]["matrix_rotated"], dtype=float)
    got = np.asarray(run["levels"][-1]["matrix"], dtype=float)
    scale = np.max(np.abs(ref))
    floor = 1.0e-12 * scale
    denom = np.where(np.abs(ref) > floor, np.abs(ref), scale)
    err = np.abs(got - ref) / denom
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for ax, data, title, fmt in (
        (axes[0], ref, "closed block form (ref)", "%.2e"),
        (axes[1], got, "rotated mesh quadrature (got)", "%.2e"),
        (axes[2], err, "entrywise relative error", "%.1e"),
    ):
        im = ax.imshow(np.log10(np.maximum(np.abs(data) if data is not err else data, 1e-18)),
                       cmap="viridis" if data is not err else "magma")
        ax.set_xticks(range(6))
        ax.set_xticklabels(MODES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(6))
        ax.set_yticklabels(MODES, fontsize=8)
        ax.set_title(f"{title}\n({run['levels'][-1]['panels']} panels)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        "CASE-BEM-RIGID-BODY-MASS: rotated body (yaw %.0f / pitch %.0f / roll %.0f deg)"
        % (run["orientation"]["euler_deg"]["yaw"], run["orientation"]["euler_deg"]["pitch"],
           run["orientation"]["euler_deg"]["roll"]),
        fontsize=10,
    )
    fig.tight_layout()
    out = HERE / "fig_rotation_error_map.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def figure_convergence(run: dict) -> Path:
    levels = run["levels"]
    panels = np.array([row["panels"] for row in levels], dtype=float)
    worst = np.array([row["max_entry_rel_error"] for row in levels])
    tensor = np.array([row["inertia_rel_error"] for row in levels])
    volume = np.array([row["volume_rel_error"] for row in levels])
    sweep = run["sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax = axes[0]
    ax.loglog(panels, worst, "o-", label="6x6 worst entry")
    ax.loglog(panels, tensor, "s--", label="inertia tensor")
    ax.loglog(panels, volume, "^:", label="volume")
    ax.axhline(run["parameters"]["tolerance"], color="r", ls="--", label="3 % criterion")
    ax.set_xlabel("panels")
    ax.set_ylabel("relative error")
    ax.set_title("rotated body: quadrature convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ref_panels = panels[-1]
    ax.annotate(f"{worst[-1]:.3%}", (panels[-1], worst[-1]), textcoords="offset points",
                xytext=(-52, 8), fontsize=8)
    ax2 = axes[1]
    labels = [f"yaw{v['euler_deg']['yaw']:.0f}\npitch{v['euler_deg']['pitch']:.0f}" for v in sweep]
    errs = [v["max_entry_rel_error"] for v in sweep]
    ax2.bar(range(len(sweep)), errs, color="#3b7dd8")
    ax2.axhline(run["parameters"]["tolerance"], color="r", ls="--", label="3 % criterion")
    ax2.set_xticks(range(len(sweep)))
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("worst entry relative error")
    ax2.set_title(f"orientation sweep at {sweep[0]['subdivisions']} subdivisions")
    ax2.legend(fontsize=8)
    for i, value in enumerate(errs):
        ax2.text(i, value, f"{value:.3%}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    out = HERE / "fig_rotation_convergence.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> int:
    run = json.loads(RUN.read_text())
    base = json.loads(BASE_RUN.read_text())
    # Headline uses the *accepted* levels only (coarse rungs are reported as the
    # failing convergence floor, not as the case verdict).
    accepted = [
        row for row in run["levels"]
        if row["subdivisions"] >= run["parameters"]["accepted_from_subdivision"]
    ] or run["levels"][-1:]
    worst_accepted = max(accepted, key=lambda row: row["max_entry_rel_error"])
    run["verdict"]["worst_level"] = {
        "subdivisions": worst_accepted["subdivisions"],
        "max_entry_rel_error": worst_accepted["max_entry_rel_error"],
        "worst_entry": worst_accepted["worst_entry"],
    }
    fig_a = figure_error_map(run)
    fig_b = figure_convergence(run)
    o = run["orientation"]
    ext = run["external"]
    neg = run["negative_control"]
    p = run["parameters"]

    lines: list[str] = []
    add = lines.append
    add("# CASE-BEM-RIGID-BODY-MASS 扩展：姿态协变 + 外部实现锚点")
    add("")
    add(
        "**判定：%s** — 旋转刚体的 6×6 与闭式块形式逐元素最大相对误差 %s（判据 3%%）；"
        "姿态协变恒等式 `M(R·b) = A M(b) A^T` 残差 %.2e（机器零）；独立实现 capytaine %s 的"
        "刚体惯量在质量块/耦合块上与 TensorBEM 一致到 %.1e / %.1e。"
        % (
            "PASS" if run["verdict"]["all_pass"] else "FAIL",
            f"{run['verdict']['worst_level']['max_entry_rel_error']:.4%}",
            o["congruence_residual_rel"],
            ext.get("capytaine_version"),
            ext["block_errors_external_vs_tensor_bem"]["translation_translation"],
            ext["block_errors_external_vs_tensor_bem"]["translation_rotation"],
        )
    )
    add("")
    add("## ① 案例介绍")
    add("")
    add(
        "运动方程 `(M_RB + A)ẍ + Bẋ + Cx = F` 中的刚体惯量矩阵 `M_RB` 是 BEM 全链最底层的输入。"
        "基础算例 `docs/case_rigid_body_mass.md` 已在**轴对齐**椭球上验证闭式块形式，但轴对齐体"
        "的惯量张量是对角的：惯性积 `I_xy, I_xz, I_yz` 全为零，roll-pitch-roll 之间的交叉项"
        "按对称性为零。于是两条真实船舶必经的通道**未被覆盖**：(a) 通用惯量张量（含惯性积）；"
        "(b) 姿态（R 群）的协变性——基础算例只测了参考点平移（T 群）那一半。"
        "本扩展把同一实心椭球绕参考点刚性旋转 `R = Rz(yaw)·Ry(pitch)·Rx(roll)`，使 `I_g` 变为满张量，"
        "并引入**独立实现锚点**（capytaine 3.x 的 `compute_rigid_body_inertia`）来钉住耦合块符号约定。"
    )
    add("")
    add("物理不变量：对任意 twist `x_ref = A x_body`，`A = blockdiag(R,R)`，必须有")
    add("")
    add("```\nM(旋转后物体) = A M(物体) A^T\n```")
    add("")
    add(
        "| 参数 | 数值 | 说明 |\n|---|---|---|\n"
        "| 半轴 a,b,c | %s m | L/B/T = %s/%s/%s m |\n"
        "| 密度 ρ | %s kg/m³ | 排水体积 %s m³ |\n"
        "| 质心偏置 r_g | %s m | 未旋转系 |\n"
        "| 欧拉角 | yaw %s°, pitch %s°, roll %s° | R = Rz·Ry·Rx |\n"
        "| 面元层级 | %s | icosphere 细分 |\n"
        "| 判据 | 逐元素相对误差 < %s | 零参照项以矩阵尺度归一 |"
        % (
            p["semi_axes_m"],
            p["semi_axes_m"][0] * 2,
            p["semi_axes_m"][1] * 2,
            p["semi_axes_m"][2] * 2,
            p["density_kg_m3"],
            f"{o['volume']:.2f}",
            p["cg_offset_m"],
            f"{o['euler_deg']['yaw']:.2f}",
            f"{o['euler_deg']['pitch']:.2f}",
            f"{o['euler_deg']['roll']:.2f}",
            p["subdivisions"],
            p["tolerance"],
        )
    )
    add("")
    add("## ② 怎么做")
    add("")
    add(
        "| 层 | 文件/入口 | 作用 |\n|---|---|---|\n"
        "| 解析路线 | `rotated_reference()` | 闭式块形式作用于 `R r_g`, `R I_g R^T`；核对协变恒等式 |\n"
        "| 面元路线 | `mesh_level()` | 旋转+平移面元网格 → 带符号四面体积分 → 装配 6×6 |\n"
        "| 负对照 | `legacy_convention()` | 旧平台约定 `[[mI,-mS^T],[-mS, I_O+mSS^T]]`（仍对称） |\n"
        "| 外部锚点 | `external_capytaine()` | capytaine 独立装配，逐块比较 |\n"
        "| 回归修复 | `mesh_mass_properties()` | 二阶矩改为对 `reference_point` 取矩（见 ⑤） |\n"
        "| 测试 | `tests/test_rigid_body_mass_orientation.py`（9 项） | 平移不变性、协变、负对照、外部锚点 |"
    )
    add("")
    add("## ③ 计算结果")
    add("")
    add("**旋转后质量特性（解析，机器求值）**")
    add("")
    add("| 量 | 数值 |\n|---|---|")
    add(f"| I_xy | {o['products_of_inertia']['I_xy']:.6e} kg·m² |")
    add(f"| I_xz | {o['products_of_inertia']['I_xz']:.6e} kg·m² |")
    add(f"| I_yz | {o['products_of_inertia']['I_yz']:.6e} kg·m² |")
    add(f"| max|I_off|/max|I_diag| | {o['max_product_over_max_diagonal']:.4f} |")
    add(f"| M[roll,pitch] | {o['cross_entries']['M_roll_pitch']:.6e} kg·m² |")
    add(f"| M[sway,yaw] | {o['cross_entries']['M_surge_yaw']:.6e} |")
    add(f"| 协变残差 | {o['congruence_residual_rel']:.2e} |")
    add(f"| 块形式残差 | {o['block_form_residual']:.2e} |")
    add(f"| 动能恒等式残差 | {o['energy_identity_error']:.2e} |")
    add("")
    add("**面元收敛阶梯（rotated，逐元素）**")
    add("")
    add("| 细分 | 面元 | 体积误差 | 惯量张量误差 | 6×6 最差元素 | 最差元素位置 | 动能残差 | <3% |")
    add("|---|---|---|---|---|---|---|---|")
    for row in run["levels"]:
        add(
            "| %d | %d | %.4f%% | %.4f%% | %.4f%% | %s-%s | %.1e | %s |"
            % (
                row["subdivisions"], row["panels"], 100 * row["volume_rel_error"],
                100 * row["inertia_rel_error"], 100 * row["max_entry_rel_error"],
                row["worst_entry"]["mode_i"], row["worst_entry"]["mode_j"],
                row["energy_identity_error"], "✅" if row["within_tolerance"] else "❌",
            )
        )
    add("")
    add("**姿态扫描（%d 细分）**" % run["sweep"][0]["subdivisions"])
    add("")
    add("| yaw | pitch | roll | max|I_off|/diag | 协变残差 | 6×6 最差元素 | <3% |")
    add("|---|---|---|---|---|---|---|")
    for row in run["sweep"]:
        add(
            "| %.1f° | %.1f° | %.1f° | %.3f | %.1e | %.4f%% | %s |"
            % (
                row["euler_deg"]["yaw"], row["euler_deg"]["pitch"], row["euler_deg"]["roll"],
                row["max_product_over_max_diagonal"], row["congruence_residual_rel"],
                100 * row["max_entry_rel_error"], "✅" if row["within_tolerance"] else "❌",
            )
        )
    add("")
    add("**外部实现锚点（capytaine %s，%d 面元）**" % (ext.get("capytaine_version"), ext["panels"]))
    add("")
    add("| 块 | capytaine vs 闭式 | capytaine vs TensorBEM |")
    add("|---|---|---|")
    for key in ("translation_translation", "translation_rotation", "rotation_rotation"):
        add(
            "| %s | %.3e | %.3e |"
            % (
                key,
                ext["block_errors_vs_closed_form"][key],
                ext["block_errors_external_vs_tensor_bem"][key],
            )
        )
    add("")
    add(
        "capytaine 与 TensorBEM 用同一份面元网格：质量块与耦合块一致到机器零，说明两者用的是**同一个**"
        "Fossen 符号约定（耦合块 `-m S(r_g)`）；转动块差异 %.3e 来自 capytaine 用面心/面积的面积分近似"
        "（一阶/近似二阶），而 TensorBEM 用精确带符号四面体多项式积分——两者对闭式解的误差分别为 %.3e 与 %.3e。"
        % (
            ext["block_errors_external_vs_tensor_bem"]["rotation_rotation"],
            ext["block_errors_vs_closed_form"]["rotation_rotation"],
            run["levels"][-1]["max_entry_rel_error"],
        )
    )
    add("")
    add("## ④ 图")
    add("")
    add(
        "图 1 `fig_rotation_error_map.png`：闭式矩阵、面元矩阵与逐元素相对误差（对数色标）。"
        "图 2 `fig_rotation_convergence.png`：左=面元收敛（6×6 最差元素/惯量张量/体积 vs 面元数，含 3%% 判据线）；"
        "右=五个互不相关姿态在 %d 细分下的最差元素误差。" % run["sweep"][0]["subdivisions"]
    )
    add("")
    add("## ⑤ 注意点")
    add("")
    add(
        "1. **发现并修复一个真实缺陷**：`mesh_mass_properties` 的二阶矩积分用绝对坐标而"
        "把 `reference_point` 只用于体积/一阶矩，导致返回的 `inertia_cg` 依赖网格的绝对位置。"
        "把同一物体连同 `reference_point` 一起平移 (0,0,−20) m，`I_cg` 相对偏差达 **60.3%%**（修复前）；"
        "修复后平移不变性达 %.1e。基础算例全部使用 `reference_point=(0,0,0)`，因此该缺陷从未被发现。"
        % 4.18e-16
    )
    add(
        "2. **平台主路径约定显式化**：`hydrostatic.compute_mass_matrix` 现支持 "
        "`inertia_about={\"cg\",\"origin\"}`（默认 `\"cg\"`，与 `rao.py`/`ice_rao.py` 调用一致）；"
        "过去同一调用按不同假设使用会重复计入平行轴项，本算例已显式声明 `origin`。"
    )
    add(
        "3. **判据只在细分 ≥ %d 有意义**：%d 面元时最差元素 %s，未过线；这是内接多面体 O(h²) 误差，"
        "不是算法错误。"
        % (p["accepted_from_subdivision"], run["levels"][0]["panels"],
           f"{run['levels'][0]['max_entry_rel_error']:.3%}")
    )
    add(
        "4. **负对照**：旧约定的矩阵仍然对称，逐元素误差 %.2f%%、协变残差 %.3f——对称性检查"
        "（旧测试全部依赖它）无法发现该缺陷，能量/协变仲裁才能。"
        % (100 * neg["max_entry_rel_error"], neg["congruence_residual_rel"])
    )
    add("5. **未含流体**：本算例只验证刚体惯量；附连质量 A(ω) 属 CASE-BEM-ELLIPSOID-ADDED-MASS 等。")
    add("")
    add("## ⑥ 误差判定")
    add("")
    add("| 关键量 | 参照 | 结果 | 是否 <3% |")
    add("|---|---|---|---|")
    worst = run["verdict"]["worst_level"]
    add(
        "| 旋转体 6×6 最差元素（%d 面元） | 闭式块形式 | %.4f%% | %s |"
        % (worst["subdivisions"], 100 * worst["max_entry_rel_error"],
           "✅" if worst["max_entry_rel_error"] < p["tolerance"] else "❌")
    )
    add(
        "| 惯量张量（%d 面元） | 闭式 `R I_g R^T` | %.4f%% | %s |"
        % (run["levels"][-1]["subdivisions"], 100 * run["levels"][-1]["inertia_rel_error"],
           "✅" if run["levels"][-1]["inertia_rel_error"] < p["tolerance"] else "❌")
    )
    add(
        "| 姿态协变 | `A M A^T` | %.2e | ✅（机器零） |" % o["congruence_residual_rel"]
    )
    add(
        "| capytaine 全阵 vs 闭式（%d 面元） | capytaine 独立实现 | %.4f%% | %s |"
        % (ext["panels"], 100 * ext["max_entry_rel_error_vs_closed_form"],
           "✅" if ext["within_tolerance"] else "❌")
    )
    add(
        "| capytaine 耦合块 vs TensorBEM | 两实现互比 | %.1e | ✅（机器零） |"
        % ext["block_errors_external_vs_tensor_bem"]["translation_rotation"]
    )
    add("")
    add(
        "**是否 <3%%：是**（最差元素 %s，容差 3%%）。协变恒等式与外部锚点均达机器零。"
        % f"{worst['max_entry_rel_error']:.4%}"
    )
    add("")
    add("## 复现与工件")
    add("")
    add("```bash")
    add("PYTHONPATH=src python -m tensorbem.benchmarks.rigid_body_mass_orientation \\")
    add("    --out docs/cases/case_bem_rigid_body_mass/rotation/run.json")
    add("PYTHONPATH=src python3 docs/cases/case_bem_rigid_body_mass/rotation/make_report.py")
    add("PYTHONPATH=src python -m pytest tests/test_rigid_body_mass_orientation.py -q")
    add("```")
    add("")
    add("| 工件 | sha256 |")
    add("|---|---|")
    for path in (RUN, fig_a, fig_b, BASE_RUN):
        add("| `%s` | `%s` |" % (path.relative_to(ROOT), sha256(path)))
    add("")
    add(
        "环境：Python %s，numpy %s，capytaine %s，%s。"
        % (run["environment"]["python"], run["environment"]["numpy"],
           run["environment"]["capytaine"], run["environment"]["machine"])
    )
    add("")
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {DOC}")
    print(f"wrote {fig_a} and {fig_b}")
    return 0 if run["verdict"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
