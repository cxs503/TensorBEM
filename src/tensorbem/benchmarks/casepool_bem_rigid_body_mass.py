"""Case-pool driver for the anchor ``CASE-BEM-RIGID-BODY-MASS`` (registry entry).

The anchor's two reference solutions already live in the repository:

* :mod:`tensorbem.benchmarks.case_rigid_body_mass` — closed-form block form vs a
  signed-tetrahedron quadrature over a panelised ellipsoid and a two-component
  box body, plus the RAO caller replay and the pre-fix ("legacy") control;
* :mod:`tensorbem.benchmarks.rigid_body_mass_orientation` — the same body
  rotated, i.e. a full (non-diagonal) inertia tensor, orientation covariance
  ``M(R b) = A M(b) A^T`` and the independent capytaine assembly.

What this driver adds is (a) the missing *machine* registration — the anchor was
answered by prose reports but no entry of
:mod:`tensorbem.benchmarks.case_pool_registry` could run it — and (b) a third
derivation route, **route "P"** (:func:`tensorbem.rigid_body_mass.
mass_matrix_from_momentum`), which assembles ``M_RB`` from the *definitions* of
linear and angular momentum instead of from the book's block form.

Why route "P" is not redundant
------------------------------
Every pre-existing check compares the assembly either with the closed block form
(1) itself or with capytaine's assembly, which encodes the same block
convention.  Route "P" re-derives the matrix from momentum conservation, so it
pins the **sign of the off-diagonal blocks** and the **placement of the
parallel-axis term** against physics rather than against a second transcription
of the same equation.  It is also strictly stronger than the kinetic-energy
identity used so far: the identity only sees the symmetric part ``sym(M)``,
while route "P" sees the whole matrix (the control section below exhibits an
antisymmetric corruption that the energy identity cannot see at all).

Criteria (predeclared, case pool tolerance ``< 3 %``)
----------------------------------------------------
* ``worst_accepted_entry_rel_error < 0.03`` over every entry of the 6x6 matrix,
  on the accepted mesh levels of both the base case and the rotated case;
* the pre-fix ("legacy coupling") control must *fail* the same criterion;
* route "P" agrees with the shipped assembly to round-off
  (``< 1e-9`` on the entry-relative metric) for four configurations: CG
  offset, reference-point offset, rotated body with a full inertia tensor, and
  a two-component body;
* route "P" *rejects* the sign-flipped coupling convention
  (``> 0.5`` entry-relative error), i.e. it has discriminating power;
* route "P" rejects an antisymmetric corruption that the kinetic-energy
  identity passes (``energy < 1e-9`` but ``momentum > 0.1``).

Run::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.casepool_bem_rigid_body_mass \
        --outdir docs/case_bem_rigid_body_mass
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.benchmarks import case_rigid_body_mass as base_case
from tensorbem.benchmarks import rigid_body_mass_orientation as orientation_case
from tensorbem.rigid_body_mass import (
    MODES,
    mass_matrix_from_momentum,
    momentum_residual,
    rigid_body_mass_matrix,
    skew_matrix,
)

__all__ = ["TOLERANCE", "momentum_section", "run", "write_doc", "main"]

TOLERANCE = 0.03
ROUNDOFF_LIMIT = 1.0e-9
DISCIPLINE = "CASE-BEM-RIGID-BODY-MASS"

DEFAULT_SUBDIVISIONS = (1, 2, 4)


def _rel(matrix_got: np.ndarray, matrix_ref: np.ndarray) -> dict[str, Any]:
    """Case-pool entry-relative metric (same floor convention as the base case)."""
    scale = float(np.max(np.abs(matrix_ref)))
    rel = np.abs(matrix_got - matrix_ref) / np.maximum(np.abs(matrix_ref), 1e-6 * scale)
    i, j = (int(v) for v in np.unravel_index(int(np.argmax(rel)), rel.shape))
    return {
        "max_relative_error": float(rel[i, j]),
        "worst_entry": [MODES[i], MODES[j]],
        "scale": scale,
    }


def momentum_section(reference: dict[str, Any]) -> dict[str, Any]:
    """Route "P": assemble ``M_RB`` from the momentum definitions and compare.

    ``reference`` is the closed-form ellipsoid configuration of the base case
    (``ellipsoid_reference()``: volume, mass, CG, inertia about the CG).
    """
    mass = float(reference["mass"])
    r_g = np.asarray(reference["center_of_gravity"], dtype=float)
    inertia = np.asarray(reference["inertia_about_cg"], dtype=float)

    rotation = orientation_case.rotation_matrix(*orientation_case.DEFAULT_ROTATION)
    r_rot = rotation @ r_g
    inertia_rot = rotation @ inertia @ rotation.T

    payloads: list[dict[str, Any]] = []

    def add(name: str, shipped: np.ndarray, trunk: dict[str, Any]) -> None:
        metric = _rel(shipped, mass_matrix_from_momentum(**trunk))
        payloads.append({"configuration": name, **metric})

    add(
        "cg-offset (axis-aligned ellipsoid, diagonal inertia)",
        rigid_body_mass_matrix(mass, r_g, inertia),
        dict(mass=mass, inertia_cg=inertia, center_of_gravity=r_g),
    )
    add(
        "reference-point offset o = (2, -1, 4) m",
        rigid_body_mass_matrix(mass, r_g, inertia, reference_point=(2.0, -1.0, 4.0)),
        dict(
            mass=mass,
            inertia_cg=inertia,
            center_of_gravity=r_g,
            reference_point=(2.0, -1.0, 4.0),
        ),
    )
    add(
        "rotated body (yaw 20 / pitch 15 / roll 25 deg, full inertia tensor)",
        rigid_body_mass_matrix(mass, r_rot, inertia_rot),
        dict(mass=mass, inertia_cg=inertia, center_of_gravity=r_g, rotation=rotation),
    )

    # two-component body: the base case's payload configuration, CG and inertia
    # composed analytically from the two boxes, then handed to both routes.
    components = base_case.case_components()
    composed = base_case.compose_analytic(components)
    add(
        "two-component box body (payload station x = %.1f m)" % 30.0,
        rigid_body_mass_matrix(
            float(composed["mass"]),
            np.asarray(composed["center_of_gravity"], dtype=float),
            np.asarray(composed["inertia_about_cg"], dtype=float),
        ),
        dict(
            mass=float(composed["mass"]),
            inertia_cg=np.asarray(composed["inertia_about_cg"], dtype=float),
            center_of_gravity=np.asarray(composed["center_of_gravity"], dtype=float),
        ),
    )

    worst_route = max(row["max_relative_error"] for row in payloads)

    # -- control 1: sign-flipped off-diagonal blocks (still a symmetric matrix,
    #    which is why a symmetry check alone cannot reject it).
    shipped = rigid_body_mass_matrix(mass, r_g, inertia)
    m = mass
    s = skew_matrix(r_g)
    flipped = shipped.copy()
    flipped[:3, 3:] = +m * s
    flipped[3:, :3] = -m * s
    flipped_residual = momentum_residual(flipped, mass, inertia, r_g)
    flipped_metric = _rel(flipped, mass_matrix_from_momentum(mass, inertia, r_g))

    # -- control 2: antisymmetric corruption. The kinetic-energy identity only
    #    sees 0.5 x^T M x, i.e. sym(M), so it must pass; route "P" must reject.
    corrupt = shipped.copy()
    corrupt[0, 1] = 1.0e6
    corrupt[1, 0] = -1.0e6
    from tensorbem.rigid_body_mass import energy_residual

    energy_blind = energy_residual(corrupt, mass, inertia, r_g)
    momentum_awake = momentum_residual(corrupt, mass, inertia, r_g)

    return {
        "route": "momentum definitions (4): p = m (v_O + w x r_g), h_O = I_g w + r_g x p",
        "function": "tensorbem.rigid_body_mass.mass_matrix_from_momentum",
        "configurations": payloads,
        "worst_relative_error": float(worst_route),
        "roundoff_limit": ROUNDOFF_LIMIT,
        "control_flipped_coupling": {
            "matrix_is_symmetric": bool(np.allclose(flipped, flipped.T)),
            "momentum_relative_error": float(flipped_metric["max_relative_error"]),
            "momentum_worst_entry": flipped_metric["worst_entry"],
            "energy_relative_error": float(energy_residual(flipped, mass, inertia, r_g)),
            "rejected": bool(flipped_metric["max_relative_error"] > 0.5),
        },
        "control_antisymmetric_corruption": {
            "magnitude_kg_m2": 1.0e6,
            "energy_relative_error": float(energy_blind),
            "momentum_relative_error": float(momentum_awake["max_relative_error"]),
            "energy_blind": bool(energy_blind < 1.0e-9),
            "momentum_detects": bool(momentum_awake["max_relative_error"] > 0.1),
        },
    }


def run(
    subdivisions: Sequence[int] = DEFAULT_SUBDIVISIONS,
    with_rao: bool = True,
) -> dict[str, Any]:
    """Execute the anchor: base case + orientation extension + route "P"."""
    start = time.perf_counter()
    base = base_case.run(list(subdivisions), with_rao=with_rao)
    orientation = orientation_case.run()

    reference = base["ellipsoid"]["reference"]
    momentum = momentum_section(reference)

    base_levels = [
        {
            "subdivisions": int(row["subdivisions"]),
            "panels": int(row["panels"]),
            "mesh_route_max_entry_rel_error": float(row["max_rel_error_from_mesh_properties"]),
            "max_entry_rel_error": float(row["max_rel_error"]),
            "control_max_entry_rel_error": float(row["control_max_rel_error"]),
            "kinetic_energy_error": float(row["kinetic_energy_error"]),
        }
        for row in base["levels"]
    ]
    accepted_from = int(base["parameters"]["accepted_from_subdivision"]) if (
        "accepted_from_subdivision" in base.get("parameters", {})
    ) else int(base["ellipsoid"]["ladder"][0].get("accepted_from", 3) or 3)

    ellipsoid_ladder = [
        {
            "panels": int(row["panels"]),
            "volume_rel_error": float(row["volume_rel_error"]),
            "inertia_cg_rel_error": float(row["inertia_cg_rel_error"]),
            "max_entry_rel_error": float(row["max_rel_error"]),
            "control_max_entry_rel_error": float(row["control_max_rel_error"]),
        }
        for row in base["ellipsoid"]["ladder"]
    ]
    orientation_levels = [
        {
            "subdivisions": int(row["subdivisions"]),
            "panels": int(row["panels"]),
            "max_entry_rel_error": float(row["max_entry_rel_error"]),
            "volume_rel_error": float(row["volume_rel_error"]),
            "within_tolerance": bool(row["within_tolerance"]),
        }
        for row in orientation["levels"]
    ]

    base_worst_accepted = max(
        level["max_entry_rel_error"] for level in base_levels if level["subdivisions"] >= accepted_from
    )
    orientation_worst_accepted = max(
        (
            level["max_entry_rel_error"]
            for level in orientation_levels
            if level["subdivisions"] >= int(orientation["parameters"]["accepted_from_subdivision"])
        ),
        default=float("inf"),
    )
    worst_accepted = max(base_worst_accepted, orientation_worst_accepted)

    cheapest_level = min(base_levels, key=lambda row: row["subdivisions"])
    controls = {
        "legacy_coupling_rejected": bool(
            cheapest_level["control_max_entry_rel_error"] > TOLERANCE
        ),
        "orientation_negative_control_rejected": bool(
            orientation["negative_control"]["max_entry_rel_error"] > TOLERANCE
        ),
    }
    checks = {
        "worst_accepted_entry_within_tolerance": bool(worst_accepted < TOLERANCE),
        "momentum_route_roundoff": bool(momentum["worst_relative_error"] < ROUNDOFF_LIMIT),
        "momentum_route_rejects_flipped_coupling": bool(
            momentum["control_flipped_coupling"]["rejected"]
        ),
        "momentum_route_beats_energy_blind_spot": bool(
            momentum["control_antisymmetric_corruption"]["energy_blind"]
            and momentum["control_antisymmetric_corruption"]["momentum_detects"]
        ),
        "base_case_passed": bool(base["passed"]),
        "orientation_case_passed": bool(orientation["verdict"]["all_pass"]),
        **controls,
    }
    all_pass = all(checks.values())

    return {
        "benchmark": DISCIPLINE,
        "subject": "6x6 rigid-body mass matrix: closed-form block form, mesh quadrature, "
        "rotated/full-tensor orientation and momentum-definition route",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "wall_time_s": time.perf_counter() - start,
        },
        "parameters": {
            "semi_axes_m": list(base_case.ELLIPSOID_SEMI_AXES),
            "density_kg_m3": float(base_case.ELLIPSOID_DENSITY),
            "cg_offset_m": list(base_case.ELLIPSOID_CENTRE),
            "mode_order": list(MODES),
            "subdivisions": [int(sd) for sd in subdivisions],
            "accepted_from_subdivision": accepted_from,
            "tolerance": TOLERANCE,
            "rotation_rad": [float(v) for v in orientation_case.DEFAULT_ROTATION],
            "rotation_order": "R = Rz(yaw) @ Ry(pitch) @ Rx(roll)",
            "reference": (
                "Fossen (2011) Handbook of Marine Craft Hydrodynamics and Motion "
                "Control, Sec. 3.3, eq. (3.44) block form; solid-ellipsoid second "
                "moments Lamb (1932) Ch. VI; the numbers are evaluated from those "
                "closed forms, none transcribed"
            ),
        },
        "reference": reference,
        "base_case": {
            "box_body": {
                "levels": base_levels,
                "worst_accepted_entry_rel_error": float(base_worst_accepted),
                "primary_max_entry_rel_error": float(base["primary"]["max_rel_error"]),
                "kinetic_energy_error": float(base["primary"]["kinetic_energy_error"]),
            },
            "ellipsoid": {"ladder": ellipsoid_ladder},
            "rao_caller": base.get("rao_integration"),
        },
        "orientation_case": {
            "levels": orientation_levels,
            "worst_accepted_entry_rel_error": float(orientation_worst_accepted),
            "congruence_residual_rel": float(
                orientation["orientation"]["congruence_residual_rel"]
            ),
            "external_status": orientation["external"]["status"],
            "checks": orientation["verdict"]["checks"],
        },
        "momentum_route": momentum,
        "verdict": {
            "overall": "PASS" if all_pass else "FAIL",
            "all_pass": bool(all_pass),
            "checks": checks,
            "worst_accepted_entry_rel_error": float(worst_accepted),
            "tolerance": TOLERANCE,
            "criterion": "entry-wise |got-ref|/max(|ref|, 1e-6 scale) < 3% on accepted "
            "mesh levels; route P must agree to round-off and must reject both controls",
        },
    }


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}e}"


def write_doc(payload: dict[str, Any], path: Path) -> Path:
    """Emit ``CASE.md`` from the artifact (every number injected, none typed)."""
    verdict = payload["verdict"]
    params = payload["parameters"]
    base = payload["base_case"]
    momentum = payload["momentum_route"]
    lines: list[str] = []
    add = lines.append
    add(f"# {DISCIPLINE} — 6×6 刚体质量矩阵的第三条独立路线（动量定义）与机器注册")
    add("")
    add(
        f"**判定：{verdict['overall']}** — 接受层级最大逐元素相对误差 "
        f"{verdict['worst_accepted_entry_rel_error']*100:.4f}%（判据 {verdict['tolerance']*100:.0f}%）；"
        f"动量定义路线（route P）与出厂装配在 4 种配置上一致到 {_fmt(momentum['worst_relative_error'])}"
        f"（圆整极限 {_fmt(momentum['roundoff_limit'])}）；route P 拒绝两种错误约定。"
        "本文所有数字由 `src/tensorbem/benchmarks/casepool_bem_rigid_body_mass.py` 从 "
        "`results.json` 注入，无人工转录。"
    )
    add("")
    add("## ① 案例介绍")
    add("")
    add(
        "运动方程 `(M_RB + A)ẍ + Bẋ + Cx = F` 中的刚体惯量矩阵 `M_RB` 是 TensorBEM 频域全链"
        "（辐射/绕射 → RAO → 短期预报）最底层的输入。广义坐标取在参考点（`(n, r×n)` 的取矩点）"
        "而质心偏离该点时，`M_RB` 呈闭式块形式 "
        "`[[m I3, -m S(r_g)], [m S(r_g), I_g + m S(r_g)ᵀ S(r_g)]]`。"
    )
    add("")
    add(
        "本锚点此前已由闭式块形式（route R）、面元带符号四面体积分（route M）与 capytaine 独立装配"
        "（route X）验证。本单补的是两件此前缺失的东西：**(a) 机器注册**——锚点有报告无注册表条目，"
        "`scripts/case_pool_run.py` 无法运行它；**(b) route P**——前三条路线全部以块形式或其转写为"
        "参照，没有任何一条从牛顿-欧拉动量定义反推符号约定。"
    )
    add("")
    add("## ② 怎么做的")
    add("")
    add("| 层 | 文件 / 入口 | 作用 |")
    add("|---|---|---|")
    add(
        "| 新路线 | `src/tensorbem/rigid_body_mass.py::momentum_map` / "
        "`mass_matrix_from_momentum` / `momentum_residual` | "
        "`p = m(v_O + ω×r_g)`、`h_O = I_g ω + r_g×p`；单位 twist 的广义动量为 `M_RB` 的列 |"
    )
    add(
        "| 算例入口 | `src/tensorbem/benchmarks/casepool_bem_rigid_body_mass.py` | "
        "基例 + 姿态扩展 + route P + 两项对照，落 `results.json`/`verdict.json`/`CASE.md` |"
    )
    add(
        "| 注册表 | `src/tensorbem/benchmarks/case_pool_registry.py` | "
        "锚点 → 驱动 → 工件 → 文档 的机器映射 |"
    )
    add(
        "| 测试 | `tests/test_rigid_body_mass_momentum.py` | route P 一致性与判别力；"
        "注册表条目完整性 |"
    )
    add("")
    add("## ③ 参数表")
    add("")
    add("| 参数 | 数值 |")
    add("|---|---|")
    add(f"| 半轴 a,b,c | {params['semi_axes_m']} m |")
    add(f"| 密度 ρ | {params['density_kg_m3']} kg/m³ |")
    add(f"| 质心偏置 r_g | {params['cg_offset_m']} m |")
    add(f"| 旋转（yaw, pitch, roll） | {[round(math.degrees(a), 3) for a in params['rotation_rad']]} deg |")
    add(f"| 面元细分 | {params['subdivisions']} |")
    add(f"| 接受层级自 | 细分 {params['accepted_from_subdivision']} |")
    add(f"| 判据 | 逐元素 < {params['tolerance']*100:.0f}% |")
    add(f"| 模式顺序 | {', '.join(params['mode_order'])} |")
    add("")
    add("## ④ 计算结果")
    add("")
    add("**route P 一致性（4 种配置，出厂装配 vs 动量定义）**")
    add("")
    add("| 配置 | 最大逐元素相对误差 | 最差位置 | 矩阵尺度 |")
    add("|---|---|---|---|")
    for row in momentum["configurations"]:
        add(
            f"| {row['configuration']} | {_fmt(row['max_relative_error'])} | "
            f"{row['worst_entry'][0]}–{row['worst_entry'][1]} | {_fmt(row['scale'], 3)} |"
        )
    add("")
    add("**基例面元收敛阶梯（椭球裁判配置，闭式 vs 四面体积分）**")
    add("")
    add("| 面元 | 体积误差 | 惯量张量误差 | 6×6 最差元素 | 旧约定对照 |")
    add("|---|---|---|---|---|")
    for row in base["ellipsoid"]["ladder"]:
        add(
            f"| {row['panels']} | {row['volume_rel_error']*100:.4f}% | "
            f"{row['inertia_cg_rel_error']*100:.4f}% | {row['max_entry_rel_error']*100:.4f}% | "
            f"{row['control_max_entry_rel_error']*100:.2f}% |"
        )
    add("")
    add("**旋转体（满惯量张量）收敛阶梯**")
    add("")
    add("| 细分 | 面元 | 6×6 最差元素 | 接受 |")
    add("|---|---|---|---|")
    for row in payload["orientation_case"]["levels"]:
        add(
            f"| {row['subdivisions']} | {row['panels']} | "
            f"{row['max_entry_rel_error']*100:.4f}% | "
            f"{'✅' if row['within_tolerance'] else '❌'} |"
        )
    add("")
    add("**对照（判别力）**")
    add("")
    add("| 对照 | 能量恒等式（只测 sym(M)） | route P | 判决 |")
    add("|---|---|---|---|")
    fc = momentum["control_flipped_coupling"]
    add(
        f"| 耦合块整体翻转（仍对称：{fc['matrix_is_symmetric']}） | "
        f"{fc['energy_relative_error']*100:.3f}% | {fc['momentum_relative_error']*100:.1f}% | "
        f"{'拒绝 ✅' if fc['rejected'] else '漏检 ❌'} |"
    )
    ac = momentum["control_antisymmetric_corruption"]
    add(
        f"| 反对称污染 {ac['magnitude_kg_m2']:.0e} kg·m² | "
        f"{ac['energy_relative_error']:.2e}（盲） | {ac['momentum_relative_error']*100:.1f}% | "
        f"{'拒绝 ✅' if ac['momentum_detects'] else '漏检 ❌'} |"
    )
    add("")
    add("## ⑤ 注意点")
    add("")
    add(
        "- **能量恒等式的盲区**：`0.5 xᵀ M x` 只看到 `sym(M)`，任意反对称污染对它完全不可见"
        "（上表实测能量残差为 0 量级）。route P 逐元素定义式的比较补上了这一盲区，"
        "这是它相对既有检查的净增量。"
    )
    add(
        "- **块符号翻转的归一化陷阱**：若以「整矩阵最大元素」归一，耦合块符号错误"
        "（~`m|r_g|`）会被惯性对角项（~`I_O`）稀释到 1%量级，恰好落在 3% 判据以内。"
        "故 route P 采用**逐元素**相对误差（`max(|exact_ij|, 1e-6·scale)`），与基例口径一致。"
    )
    add(
        "- **假设与适用范围**：均匀密度刚体、参考点为广义坐标取矩点；不覆盖附加质量 `A` 的约定"
        "（那是辐射问题的责任，见 `docs/case_bem_rigid_body_mass/` 中 RAO caller 回放一节）。"
    )
    add("")
    add("## ⑥ 误差判定")
    add("")
    add("| 关键量 | 基准 | 误差 | <3%？ |")
    add("|---|---|---|---|")
    add(
        f"| 6×6 最差元素（接受层级 = 细分 ≥ {params['accepted_from_subdivision']}） | "
        f"闭式块形式 | {verdict['worst_accepted_entry_rel_error']*100:.4f}% | "
        f"{'是' if verdict['worst_accepted_entry_rel_error'] < params['tolerance'] else '否'} |"
    )
    add(
        f"| route P vs 出厂装配（4 配置最差） | 动量定义 | {_fmt(momentum['worst_relative_error'])} | "
        f"是（远小于，属圆整） |"
    )
    add(
        f"| 旧耦合约定对照（几何上必错） | 闭式块形式 | "
        f"{base['box_body']['levels'][0]['control_max_entry_rel_error']*100:.2f}% | 否（预期，判别力证据） |"
    )
    add("")
    add(f"检查项：`{json.dumps(verdict['checks'], ensure_ascii=False)}`")
    add("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def plot(payload: dict[str, Any], path: Path) -> Path:
    """Two panels: refinement ladder vs the 3 % line, and the control magnitudes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ladder = payload["base_case"]["ellipsoid"]["ladder"]
    panels = [row["panels"] for row in ladder]
    axes[0].loglog(panels, [row["max_entry_rel_error"] for row in ladder], "o-",
                   label="shipped assembly (closed form vs mesh)")
    axes[0].loglog(panels, [row["control_max_entry_rel_error"] for row in ladder], "s--",
                   color="tab:red", label="pre-fix coupling control")
    axes[0].axhline(payload["parameters"]["tolerance"], color="k", ls=":", label="3 % criterion")
    axes[0].set_xlabel("panels")
    axes[0].set_ylabel("max entry relative error")
    axes[0].set_title("refinement ladder, solid ellipsoid")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    labels = ["route P\n(4 configs)", "flipped\ncoupling", "antisym.\ncorruption"]
    momentum = payload["momentum_route"]
    values = [
        max(momentum["worst_relative_error"], 1e-17),
        momentum["control_flipped_coupling"]["momentum_relative_error"],
        momentum["control_antisymmetric_corruption"]["momentum_relative_error"],
    ]
    colors = ["tab:green", "tab:red", "tab:orange"]
    bars = axes[1].bar(labels, values, color=colors)
    axes[1].set_yscale("log")
    axes[1].axhline(payload["momentum_route"]["roundoff_limit"], color="k", ls=":",
                    label="round-off limit 1e-9")
    axes[1].set_ylabel("momentum-route entry relative error")
    axes[1].set_title("route P: agreement and discriminating power")
    for bar, value in zip(bars, values):
        axes[1].annotate(f"{value:.2e}", (bar.get_x() + bar.get_width() / 2, value),
                         textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle(
        f"{DISCIPLINE}: r_g = {payload['parameters']['cg_offset_m']} m, "
        f"rho = {payload['parameters']['density_kg_m3']:.0f} kg/m3"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=Path("docs/case_bem_rigid_body_mass"))
    parser.add_argument("--subdivisions", type=int, nargs="+", default=list(DEFAULT_SUBDIVISIONS))
    parser.add_argument("--no-rao", action="store_true")
    args = parser.parse_args(argv)

    payload = run(args.subdivisions, with_rao=not args.no_rao)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (outdir / "verdict.json").write_text(
        json.dumps(payload["verdict"], indent=2) + "\n", encoding="utf-8"
    )
    write_doc(payload, outdir / "CASE.md")
    plot(payload, outdir / "fig_momentum_route.png")
    verdict = payload["verdict"]
    print(json.dumps({"verdict": verdict["overall"], "checks": verdict["checks"],
                      "worst_accepted_entry_rel_error":
                          verdict["worst_accepted_entry_rel_error"],
                      "outdir": str(outdir)}, indent=2))
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())