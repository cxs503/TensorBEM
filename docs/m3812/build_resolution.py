"""Build the merge-resolution evidence for PR #128 (agent/m3181) onto main.

Three receipts are reconciled here, none of them hand-copied:

* ``docs/m3181/outputs/three_d_free_surface_registry.json`` -- the 63 cells
  recomputed by branch ``agent/m3181`` under *its* registry (groups
  ``hulme3d`` / ``hulme3d_chief`` / ``wang_linton3d``).
* ``docs/m3150/outputs/registry_3d_free_surface.json`` -- the 62 cells that
  m3150 had already registered on ``main`` (groups ``hulme_hemisphere`` /
  ``hulme_chief`` / ``submerged_sphere``).
* ``docs/m3812/outputs/merged_registry.json`` -- the merged registry run from
  the merged source tree.

Cells are matched on the *measured* value (a physical quantity of a fixed mesh
and frequency, so it is stable across the two implementations); the declared
value may legitimately differ when a branch chose a different rounding or a
different mesh-ratio convention, and those cases are listed explicitly instead
of being collapsed.

Run:  PYTHONPATH=src python3 docs/m3812/build_resolution.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
M3181 = ROOT / "docs/m3181/outputs/three_d_free_surface_registry.json"
M3150 = ROOT / "docs/m3150/outputs/registry_3d_free_surface.json"
MERGED = ROOT / "docs/m3812/outputs/merged_registry.json"
OUT = ROOT / "docs/m3812/merge_m3181_resolution.md"

#: relative tolerance used to pair two receipts on the same physical cell
MATCH = 1.0e-9


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pair(receipt: list[dict], target: list[dict], used: set[int]) -> list[tuple[dict, dict | None]]:
    """Pair each row of ``receipt`` with the unclaimed ``target`` row of equal measurement."""
    out: list[tuple[dict, dict | None]] = []
    for row in receipt:
        best: dict | None = None
        best_gap = MATCH
        for idx, other in enumerate(target):
            if idx in used:
                continue
            a, b = float(row["measured"]), float(other["measured"])
            if a == b:
                gap = 0.0
            elif b == 0.0:
                continue
            else:
                gap = abs(a - b) / abs(b)
            if gap <= best_gap:
                best, best_gap = other, gap
                if gap == 0.0:
                    break
        if best is not None:
            used.add(target.index(best))
        out.append((row, best))
    return out


def fmt(value: float) -> str:
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude < 1.0e-3 or magnitude >= 1.0e4:
        return f"{value:.4e}"
    return f"{value:.6g}"


def verdict_cell(row: dict) -> str:
    deviation = float(row["relative_deviation"])
    if deviation != deviation:  # NaN: cell pending a full merged run
        return f"{row['verdict']}"
    return f"{row['verdict']} ({deviation:+.2e})"


def main() -> None:
    m3181 = load(M3181)["checks"]
    m3150 = load(M3150)["checks"]
    if MERGED.exists():
        merged = load(MERGED)["checks"]
        merged_summary = load(MERGED)["summary"]
        merged_source = "`docs/m3812/outputs/merged_registry.json`（合并后源码的 standard 层全量回执）"
    else:
        # Fallback: the merged registry is the m3150 set plus four m3181-only
        # cells. Until the merged standard-tier receipt exists, take the 62
        # measured values from the m3150 receipt and the declared values (all 66)
        # from the merged source, so no number is transcribed by hand.
        sys.path.insert(0, str(ROOT / "src"))
        from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS

        declared = {row.key: row for row in DECLARED_NUMBERS}
        measured = {row["key"]: row for row in m3150}
        merged = []
        for key, entry in declared.items():
            if key in measured:
                merged.append(measured[key])
            elif entry.group in {"hulme_hemisphere", "hulme_chief", "submerged_sphere"}:
                merged.append(
                    {
                        "key": key,
                        "declared": entry.declared,
                        "measured": float("nan"),
                        "relative_deviation": float("nan"),
                        "verdict": "PENDING",
                        "source": entry.source,
                    }
                )
        merged_summary = {
            "checked": len(declared),
            "passed": sum(1 for r in merged if r["verdict"] == "PASS"),
            "note": "partial receipt (m3150 measured values + m3181-only cells pending)",
        }
        merged_source = (
            "fallback：66 格声明值直接取自合并后源码，测量值取 m3150 回执，"
            "m3181 独有的 4 格标注 PENDING（全量 standard 回执见 `docs/m3812/test_evidence.md`）"
        )

    merged_by_key = {row["key"]: row for row in merged}
    used_3150: set[int] = set()
    used_merged: set[int] = set()

    pairs_3181_3150 = pair(m3181, m3150, used_3150)
    merged_list = list(merged)
    pairs_3181_merged = pair(m3181, merged_list, used_merged)

    rows: list[str] = []
    convention: list[str] = []
    unmatched: list[str] = []
    for (row, sibling), (_, target) in zip(pairs_3181_3150, pairs_3181_merged):
        if target is None:
            unmatched.append(row["key"])
            continue
        differing = abs(float(row["declared"]) - float(target["declared"])) > 0.0
        rows.append(
            "| `{k3181}` | {d3181} | `{km}` | {dm} | {verdict} |".format(
                k3181=row["key"],
                d3181=fmt(float(row["declared"])),
                km=target["key"],
                dm=fmt(float(target["declared"])),
                verdict=verdict_cell(target),
            )
        )
        if differing:
            convention.append(
                "| `{k3181}` | {d3181} | `{km}` | {dm} | {why} |".format(
                    k3181=row["key"],
                    d3181=fmt(float(row["declared"])),
                    km=target["key"],
                    dm=fmt(float(target["declared"])),
                    why=explain(row["key"]),
                )
            )

    preserved = [
        (row, merged_by_key[key])
        for key in sorted(merged_by_key)
        for row in m3181
        if key in _PRESERVED and key in merged_by_key and row["key"] in _PRESERVED[key]
    ]
    extras = [row for row in merged if row["key"] not in {t["key"] for _, t in pairs_3181_merged if t}]

    summary = merged_summary
    m3181_summary = load(M3181)["summary"]
    m3150_summary = load(M3150)["summary"]

    text = f"""# PR #128（agent/m3181）→ main 三方合并解冲记录（m3812）

**仓**：TensorBEM · **工作树**：`/root/tp-agent/work/m3812` · **合并提交**：见 `git log -1`
**方向**：`origin/main`（m3150 等已入库）← `agent/m3181`（PR #128）
**冲突面**：3 个文件与 main 同时被改（2 个内容冲突 + 1 个自动合并需复核），其余 14 个文件为 m3181 单侧新增，原样并入。

本文件由 `docs/m3812/build_resolution.py` 从三份**程序化回执**生成，表中数字零手抄：
m3181 的 63 格回执（`docs/m3181/outputs/three_d_free_surface_registry.json`）、
m3150 已入库的 62 格回执（`docs/m3150/outputs/registry_3d_free_surface.json`）、
合并后源码跑出的注册表回执（`docs/m3812/outputs/merged_registry.json`）。

## 1. 冲突清单与逐文件解法

| 文件 | 状态 | 解法 |
|---|---|---|
| `src/tensorbem/benchmarks/declared_numbers.py` | 内容冲突 3 处（import 别名、表 10 注册项生成器、`_run_group` 分组派发） | 以 main 的 m3150 实现为骨架（三组 `hulme_hemisphere`/`hulme_chief`/`submerged_sphere`、`standard` 层、键名含面元数），并把 m3181 独有的 4 格补入：表 11 参考解 `A33'`/`B33'` 与内部 CHIEF 点数、表 12 的 20 面元机器零格（`kind="upper"`）。`_chief_rows` 增加 `exact_a33`/`exact_b33` 两个量的展开，使这 3 格有可复算的测量来源。m3181 的表驱动解析器（`_TABLE_HULME`/`_refinement_rows`）不保留：它与 main 的显式注册项覆盖同一批格子，保留会触发 `duplicate registry keys` 断言（`test_registry_is_well_formed`），属重复实现而非内容丢失——逐格等价性见 §2。 |
| `tests/test_declared_numbers.py` | 内容冲突 2 处（`declared>0` 断言注释、文档基准族集合中的组名） | 以 main 版本为骨架（组名 `hulme_hemisphere`/`hulme_chief`/`submerged_sphere`），并吸收 m3181 的判据：新增 `test_three_dimensional_free_surface_tables_are_fully_registered`（表 10/11/12 格子数 31/9/26、同表共用一次求解、CHIEF 阻尼符号、机器零格为上界）、`test_table_10_b11_order_uses_the_measured_mesh_ratio`（保留 m3181 发现的 `√N` 减半口径 1.62/1.88，并钉死文档采用的实测棱长比口径，外加二阶收敛断言）、`test_table_12_converges_at_second_order_and_machine_zero_coupling_holds`。 |
| `docs/marine_bem_benchmarks.md` | 自动合并（双方改同一批表） | git 自动合并成功，但存在**语义冲突**：m3150 的表 10/12 订正段与 m3181 新写的 §4.12 回执段并存。人工复核：表 10 观测阶列取 m3150 的实测棱长比口径（1.80/1.94），表 12 转动耦合取 2.15e-4/2.65e-5；在 §4.12 增补「合入订正」段，声明仓库内唯一可执行注册表为 m3150 三组、m3181 独有 4 格已并入、回执表保留原键名以备审计，并显式说明两处口径差异同源。 |
| 其余 14 个 m3181 文件（`docs/m3181/**`、`dev_report_m3181_*.md`） | 仅 m3181 有 | 原样并入，未删除、未改写。 |

## 2. m3181 的 63 格 → 合并注册表逐格对应

`docs/m3181/outputs/three_d_free_surface_registry.json` 判定：{m3181_summary['passed']}/{m3181_summary['checked']} PASS；
m3150 回执：{m3150_summary['passed']}/{m3150_summary['checked']} PASS；
合并后注册表：{summary['passed']}/{summary['checked']} PASS（{merged_source}）。
下表按「同一物理格 = 同一测量值」配对，最后一列为该格在合并注册表中的判定与相对偏差。

| m3181 键名 | m3181 声明 | 合并注册表键名 | 合并声明 | 合并判定（偏差） |
|---|---:|---|---:|---|
{chr(10).join(rows)}

未能在合并注册表中找到对应测量的 m3181 格：{("无" if not unmatched else "、".join(f"`{k}`" for k in unmatched))}。

### 2.1 声明值不同但物理格相同的行（口径/舍入差异，非数据丢失）

| m3181 键名 | m3181 声明 | 合并键名 | 合并声明 | 差异性质 |
|---|---:|---|---:|---|
{chr(10).join(convention)}

* `hulme_observed_order_b11_*`：m3181 按理想 \(\\sqrt{{N}}\) 减半折算（1.62 / 1.88），main 按实测平均棱长比 \(h_0/h_1=1.87\) 折算（1.80 / 1.94）。表 10 全表统一采用后者，故注册表登记后者的可复现值；前者在 `test_table_10_b11_order_uses_the_measured_mesh_ratio` 中被显式复算并断言仍在 1.62 附近，历史上没有丢。
* `wang_linton_rotational_added_mass_max_ratio_{80,320}`：m3181 按两位有效数字登记（2.2e-4 / 2.6e-5），main 的 m3150 订正要求可复现的四位有效数字（2.151e-4 / 2.6481e-5）。同一测量的两种精度处置，后者可回归，取后者。

## 3. 合并注册表相对 m3150 的 62 格新增（m3181 贡献）

| 新增键名 | 声明 | 来源 |
|---|---:|---|
{chr(10).join(f"| `{row['key']}` | {fmt(float(row['declared']))} | {row['source']} |" for row in extras)}


这 4 格正是 m3181 独有、m3150 未覆盖的表格单元：表 11 的参考解列与 CHIEF 点数（3 格）、表 12 的 20 面元机器零格（1 格，按 `1e-20` 上界判，实测落在 \(2.5\\times10^{{-32}}\)）。

## 4. 测试证据

见 `docs/m3812/test_evidence.md`（由实际 `pytest` 输出摘要注入）。
"""

    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} ({len(text.splitlines())} lines, {len(rows)} mapped cells, {len(extras)} extras)")


#: merged keys that carry a cell m3181 registered under a different name
_PRESERVED = {
    "hulme_chief_2_58_exact_a33": {"hulme_chief_exact_a33_dimless"},
    "hulme_chief_2_58_exact_b33": {"hulme_chief_exact_b33_dimless"},
    "hulme_chief_2_58_chief_point_count": {"hulme_chief_chief_regularized_chief_point_count"},
    "submerged_sphere_20_rotational_coupling": {"wang_linton_rotational_added_mass_max_ratio_20"},
}


def explain(key: str) -> str:
    if key.startswith("hulme_observed_order_b11"):
        return "观测阶口径：m3181 用 √N 减半，main/表 10 用实测棱长比"
    if key.startswith("wang_linton_rotational_added_mass_max_ratio"):
        return "有效数字：m3181 两位，main/m3150 订正为可复现四位"
    return "口径差异"


if __name__ == "__main__":
    main()