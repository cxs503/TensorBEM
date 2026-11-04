"""Publish the m3245 progress report to the platform report API.

All numbers come from ``outputs/m3245/free_surface3d_registry.json``; the figure
is the PNG produced by ``outputs/m3245/make_figures.py``.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]

data = json.loads(Path("outputs/m3245/free_surface3d_registry.json").read_text())
rows = {row["key"]: row for row in data["checks"]}
summary = data["summary"]
worst_value = max(
    abs(row["relative_deviation"]) for row in data["checks"] if row["comparison"] == "value"
)


def m(key: str, digits: int = 3) -> str:
    return f"{rows[key]['measured']:.{digits}g}"


def hulme_table() -> str:
    quantities = [
        ("A33'", "heave_added_mass_rel_error"),
        ("B33'", "heave_damping_rel_error"),
        ("A11'", "surge_added_mass_rel_error"),
        ("B11'", "surge_damping_rel_error"),
        ("|Fz|", "excitation_force_z_rel_error"),
    ]
    out = ["| N | " + " | ".join(q for q, _ in quantities) + " |", "|---:|" + "---:|" * len(quantities)]
    for n in (16, 64, 256):
        cells = [f"{rows[f'hulme_hemisphere_{n}_{q}']['measured']:.3e}" for _, q in quantities]
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def sphere_table() -> str:
    quantities = [
        ("A33'", "heave_added_mass_rel_error"),
        ("A11'", "surge_added_mass_rel_error"),
        ("|Fx|", "excitation_force_x_rel_error"),
        ("|Fz|", "excitation_force_z_rel_error"),
    ]
    out = ["| N | " + " | ".join(q for q, _ in quantities) + " |", "|---:|" + "---:|" * len(quantities)]
    for n in (20, 80, 320):
        cells = [f"{rows[f'submerged_sphere_wave_{n}_{q}']['measured']:.3e}" for _, q in quantities]
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def chief_table() -> str:
    keys = [
        ("未增广 BIE 条件数", "hulme_chief_ka2_58_unregularized_condition_number"),
        ("未增广 A33'", "hulme_chief_ka2_58_unregularized_a33"),
        ("未增广 |B33'|（符号为负）", "hulme_chief_ka2_58_unregularized_b33_magnitude"),
        ("5 点 CHIEF 条件数", "hulme_chief_ka2_58_chief_condition_number"),
        ("CHIEF A33'", "hulme_chief_ka2_58_chief_a33"),
        ("CHIEF B33'", "hulme_chief_ka2_58_chief_b33"),
    ]
    out = ["| 量 | 文档声明 | 实测 | 相对偏差 | 判定 |", "|---|---:|---:|---:|---|"]
    for label, key in keys:
        row = rows[key]
        out.append(
            f"| {label} | {row['declared']:.4g} | {row['measured']:.6g} | "
            f"{row['relative_deviation']:+.1e} | {row['verdict']} |"
        )
    return "\n".join(out)


markdown = f"""# 三维自由液面声明数字的可执行化：Hulme 半球 / Ka=2.58 CHIEF / 潜没球体（TensorBEM m3245）

## ① 案例介绍

**物理背景**：三维深水自由液面线性势流（Kelvin/Delhommeau Green 函数）下的浮体/潜体辐射与
绕射。本仓 `free_surface_solver3d` 的全部精度声明集中在 `docs/marine_bem_benchmarks.md` §4.10
表 10/表 11 与 §4.11 表 12，共 46 个数字，但注册表 `tensorbem.benchmarks.declared_numbers`
（m2136）只登记了二维圆柱、椭球、Ursell、内部谐波四族——这些三维自由液面数字**没有任何测试重算**，
m3059 收尾时明确把它列为「建议后续单独立项」。本单即该项。

| 项 | Hulme (1982) 漂浮半球 | Wang/Linton 潜没球体 | 内域水线共振对比 |
|---|---|---|---|
| 几何 | 半径 a=1 半球穿水面，`WettedHullMesh` | 半径 a=1，潜深 f/a=1.5，闭合三角形网格 | 同 Hulme 半球 |
| 频率 | Ka=1.0 | Ka=0.8 | Ka=2.58（第一轴对称内域水线特征值附近） |
| 密度/重力 | 1025 kg/m³ / 9.80665 m/s² | 同左 | 同左 |
| 面元 | 16 / 64 / 256（八面体递归细分） | 20 / 80 / 320（icosphere） | 256 |
| 真值来源 | 48 项球面无波势半解析多极子解 | 零数值求积的阶乘 + Ei(2νf) 闭式多极子系统 | 同一多极子解 |
| 判据量 | A33'/B33'/A11'/B11'/|Fz| 相对误差 + 观测阶 | A33'/A11'/|Fx|/|Fz| 相对误差 + 观测阶 | 条件数与 A33'/B33' 的符号与幅值 |

## ② 怎么做的

1. **注册表扩展**：在 `declared_numbers.py` 新增三个 benchmark 组
   （`hulme_hemisphere`、`hulme_chief`、`submerged_sphere_wave`），把 `run_refinement_study` /
   `run_irregular_frequency_study` 的每层每量拍平为 `panels<N>.<quantity>`，靠注册表自带的
   `(group, parameters)` 缓存让每条注册项共享同一次求解；共登记 **{summary['checked']} 行**
   （表 10：15 误差 + 4 观测阶；表 11：6；表 12：12 误差 + 4 观测阶；5 条上界型守恒断言；
   2 组单层 `_standard_guard` 行）。容差沿用注册表默认 1.5e-2（= 文档 2–3 位有效数字的舍入量级）。
2. **成本分层**：三维稠密自由液面求解秒级→分钟级，主体入 `heavy` 层（`TENSORBEM_HEAVY_TESTS=1`）；
   另为每族登记一条单层（64 面元半球 / 80 面元球体）`standard` 行，使默认 `pytest` 仍走通三维自由液面路径。
3. **回归钉桩**：`tests/test_declared_numbers.py` 新增默认层门与 HEAVY 层门；HEAVY 门复用同一次运行
   额外断言**物理方向**——未增广 BIE 在 Ka=2.58 的 B33' < 0（违背无源耗散性）、CHIEF 增广后 B33' > 0，
   并断言细网格误差严格小于粗网格（防止在未收敛族上引用观测阶）。
4. **零手抄**：`outputs/m3245/run_registry.py` 写出 `free_surface3d_registry.json`，
   `inject_docs.py` 从 JSON 注入 `docs/free_surface3d_registry.md`，`make_figures.py` 从 JSON 出图。

## ③ 计算结果

注册表本组 {summary['checked']} 行判定 **{summary['verdict']}**（{summary['passed']}/{summary['checked']}），
`value` 型最坏相对偏差 **{worst_value:.2e}**；重算合计 {data['elapsed_seconds']:.0f} s。

**Hulme 漂浮半球（Ka=1.0，表 10 复算）**

{hulme_table()}

**Wang/Linton 潜没球体（f/a=1.5, Ka=0.8，表 12 复算）**

{sphere_table()}

**Ka=2.58 内域水线共振：未增广 BIE vs 5 点三维 CHIEF（表 11 复算）**

{chief_table()}

观测阶复算：Hulme 256 面元 A33'/B33'/A11'/|Fz| = {m('hulme_hemisphere_256_observed_order_a33', 4)} /
{m('hulme_hemisphere_256_observed_order_b33', 4)} / {m('hulme_hemisphere_256_observed_order_a11', 4)} /
{m('hulme_hemisphere_256_observed_order_fz', 4)}；潜没球体 320 面元对应
{m('submerged_sphere_wave_320_observed_order_a33', 4)} /
{m('submerged_sphere_wave_320_observed_order_b33', 4)} /
{m('submerged_sphere_wave_320_observed_order_a11', 4)} /
{m('submerged_sphere_wave_320_observed_order_fz', 4)}，与文档声明一致。

## ④ 结果图

左图：三张表的收敛阶梯（实线 = Hulme 半球，虚线 = 潜没球体，点线 = N^-2 参考斜率）。
右图：Ka=2.58, N=256 处未增广 BIE（红）与 5 点三维 CHIEF（蓝）对半解析值（灰）的对比——
未增广解的 B33' 落在负值区，CHIEF 增广后回到正值。

## ⑤ 注意点

1. 未改任何求解器实现；本单是验证/注册表工作，不新增物理模型、不调精度参数。
2. 注册表 well-formedness 门只接受正的 `declared`，故未增广的**负** B33' 登记为幅值，符号由 HEAVY
   测试断言（代码注释与 docs 均写明，不是丢符号）。
3. 上界型行会使 `worst_relative_deviation` 出现大负偏差（上界 4e-3 vs 实测 3e-4），故本报告另报
   `value` 型最坏偏差。
4. 仍未纳入注册表：`kochin_hulme_convergence` 的 1024 面元 B11' 行（0.90%）——单次 1024 面元三维
   求解高一个量级，建议单独一单；有限水深三维 John 核仍是功能缺口，不在本单范围。

## ⑥ 误差判定（关键量 vs 基准，逐项回答「是否 <3%」）

| 关键量 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| Hulme 半球 A33'/B33'/A11'（256 面元） | 半解析多极子（Hulme 1982） | {m('hulme_hemisphere_256_heave_added_mass_rel_error')} / {m('hulme_hemisphere_256_heave_damping_rel_error')} / {m('hulme_hemisphere_256_surge_added_mass_rel_error')} | ✅ 是 |
| Hulme 半球 B11'（最慢收敛量，256 面元） | 同上 | {m('hulme_hemisphere_256_surge_damping_rel_error')} | ❌ 否（3.53%；1024 面元另证 0.90%） |
| Hulme 半球 \\|Fz\\|（256 面元） | 同上 | {m('hulme_hemisphere_256_excitation_force_z_rel_error')} | ✅ 是 |
| 潜没球体 A33'/A11'（320 面元） | 零求积多极子（Wang/Linton） | {m('submerged_sphere_wave_320_heave_added_mass_rel_error')} / {m('submerged_sphere_wave_320_surge_added_mass_rel_error')} | ✅ 是 |
| 潜没球体 \\|Fx\\|/\\|Fz\\|（320 面元） | 同上 | {m('submerged_sphere_wave_320_excitation_force_x_rel_error')} / {m('submerged_sphere_wave_320_excitation_force_z_rel_error')} | ❌ 否（3.10%/3.20%，略超） |
| CHIEF 正则化后 B33'（Ka=2.58） | 半解析精确 0.0631 | {m('hulme_chief_ka2_58_chief_b33', 4)} | ❌ 否（8.85%，但已消除非物理负阻尼） |
| 文档声明 vs 实测（{summary['checked']} 行注册项） | 文档 | 最坏 {worst_value:.2e}（`value` 型） | ✅ 是 |

**结论（honest）**：**文档与代码在 {summary['checked']} 行上一致**（最坏偏差 {worst_value:.1e}，即 3 位有效
数字的舍入量级），这些三维自由液面声明从本单起是**可执行的**、不再可能静默腐化。但**物理精度本身并非
全部 <3%**：Hulme B11' 在 256 面元为 3.53%、潜没球体波浪力在 320 面元为 3.10–3.20%、CHIEF 后 B33' 误差
8.85%。这些都是常数面元离散误差，文档原本已如实给出（B11' 另有 1024 面元 0.90% 证据），本单如实转载。

## 证据指针

- 代码：`src/tensorbem/benchmarks/declared_numbers.py`（三组 + 56 行）、
  `tests/test_declared_numbers.py`（默认层门 + HEAVY 层门）
- 工件：`outputs/m3245/free_surface3d_registry.json`（56 行原始判定）、
  `outputs/m3245/fig_3d_free_surface_registry.png`
- 文档：`docs/free_surface3d_registry.md`（程序化注入）、`docs/marine_bem_benchmarks.md` §4 末指针
- 报告：`dev_report_m3245_free_surface3d_registry.md`
"""

figure = Path("outputs/m3245/fig_3d_free_surface_registry.png").read_bytes()
payload = {
    "title": "TensorBEM: 三维自由液面声明数字可执行化（Hulme 半球 / Ka=2.58 CHIEF / 潜没球体，56 行注册 PASS）",
    "summary": (
        "把 docs/marine_bem_benchmarks.md 表 10–12 的 46 个三维自由液面数字纳入 declared_numbers 注册表"
        f"（共 {summary['checked']} 行，全部 PASS，value 型最坏偏差 {worst_value:.1e}），"
        "新增 standard 层单层守卫 + heavy 层物理方向门（未增广 BIE 负阻尼 / CHIEF 正阻尼）。"
        "如实记录精度仍有 3 处 >3%（B11' 3.53%、潜没球体波浪力 3.10–3.20%、CHIEF 后 B33' 8.85%）。"
    ),
    "scenario_key": "verification",
    "mission_id": 3245,
    "tags": ["TensorBEM", "verification-debt", "declared-numbers", "3D-free-surface", "Hulme", "CHIEF"],
    "markdown": markdown,
    "figures": [
        {
            "name": "fig_3d_free_surface_registry.png",
            "caption": (
                "左：三维自由液面收敛阶梯（Hulme 半球实线 / 潜没球体虚线 / N^-2 点线）；"
                "右：Ka=2.58, N=256 处未增广 BIE 与 5 点三维 CHIEF 对比，未增广 B33' 为负。"
            ),
            "png_base64": base64.b64encode(figure).decode("ascii"),
        }
    ],
}

body = json.dumps(payload).encode()
print("payload", len(body), "bytes")


def post(path: str, data: dict) -> dict:
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


login = post("/auth/login", {"username": USER, "password": PASS})
token = login.get("access_token") or login.get("token") or login.get("data", {}).get("access_token")
assert token, login

request = urllib.request.Request(
    BASE + "/tensor/reports",
    data=body,
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=120) as response:
    result = json.loads(response.read())
print(json.dumps(result, ensure_ascii=False)[:800])