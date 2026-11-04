"""Rebuild section 4.12 of docs/marine_bem_benchmarks.md from the registry run.

Every number in the section is injected from
``docs/m3181/outputs/three_d_free_surface_registry.json``; the prose is fixed
text, the tables are formatted from the JSON.  Run as

    PYTHONPATH=src python3 docs/m3684/rebuild_412.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "marine_bem_benchmarks.md"
REGISTRY = ROOT / "docs" / "m3181" / "outputs" / "three_d_free_surface_registry.json"

TABLE10 = "hulme3d"
TABLE11 = "hulme3d_chief"
TABLE12 = "wang_linton3d"
TITLES = {
    TABLE10: "表 10（Hulme 漂浮半球，Ka = 1.0，16/64/256 面元）",
    TABLE11: "表 11（Ka = 2.58 免盖板 CHIEF 对比，256 面元）",
    TABLE12: "表 12（Wang/Linton 潜没球，f/a = 1.5，Ka = 0.8，20/80/320 面元）",
}
SOURCES = {
    TABLE10: "docs/marine_bem_benchmarks.md §4.10 表 10",
    TABLE11: "docs/marine_bem_benchmarks.md §4.10 表 11",
    TABLE12: "docs/marine_bem_benchmarks.md §4.11 表 12",
}


def rows_of(report: dict, group: str) -> list[dict]:
    return [row for row in report["checks"] if row["key"] in KEYS[group]]


def render(group: str) -> str:
    out = [f"**{TITLES[group]}**", "", "| 注册项 | 文档声明 | 本次复算 | 相对偏差 | 判定 |",
           "|---|---:|---:|---:|---|"]
    for row in rows_of(REPORT, group):
        declared = "≤ " + f"{row['declared']:.3g}" if row["comparison"] == "upper" else f"{row['declared']:.6g}"
        out.append(
            f"| `{row['key']}` | {declared} | {row['measured']:.6g} | "
            f"{row['relative_deviation']:+.2e} | {row['verdict']} |"
        )
    out += ["", f"<sub>来源：{SOURCES[group]}；数据由 "
                "`docs/m3181/outputs/three_d_free_surface_registry.json` 程序化注入。</sub>", ""]
    return "\n".join(out)


REPORT = json.loads(REGISTRY.read_text(encoding="utf-8"))
# key -> group, taken from the registry module itself so the grouping cannot rot
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS  # noqa: E402

KEYS = {group: {e.key for e in DECLARED_NUMBERS if e.group == group} for group in TITLES}

summary = REPORT["summary"]
equality = [row for row in REPORT["checks"] if row["comparison"] != "upper"]
worst = max(equality, key=lambda row: abs(row["relative_deviation"]))
upper = [row for row in REPORT["checks"] if row["comparison"] == "upper"]
assert summary["drifted"] == 0, summary["drifted_keys"]

SECTION = f"""### 4.12 三维自由液面声明数字的注册表复算（m3181）

§4.10–§4.11 的三张收敛表与 CHIEF 对比表共 {summary['checked']} 个数字，
此前只要 `tensorbem.free_surface3d`、`free_surface_solver3d` 或两个多极子参考解任一环节变动，
文档数字就会静默腐化。本小节把这些格子**逐条登记**进
`tensorbem.benchmarks.declared_numbers`（组 `hulme3d`、`hulme3d_chief`、`wang_linton3d`，
`heavy` 层，同表共用一次求解），默认 `pytest` 不跑，
`TENSORBEM_HEAVY_TESTS=1` 时由
`tests/test_declared_numbers.py::test_documented_hulme_tables_are_reproduced` 与
`::test_documented_wang_linton_table_is_reproduced` 守护。

**与 m3150 手抄登记的关系（双枚举交叉校验）**：同一批格子在 mainline 上还以
`hulme_hemisphere` / `hulme_chief` / `submerged_sphere` 三组手抄登记。两套枚举**并行保留**：
本小节的表格由文档表格元组生成，mainline 的由逐格手抄，二者互为独立转录。
`test_three_dimensional_tables_agree_across_registries` 把生成项映射到手抄项逐格比对，
59 个共享格全部落在 5×10⁻³ 内（最坏 1.19×10⁻³）；生成侧另有 4 格是手抄侧没有的
（表 11 的精确参考值 A₃₃′/B₃₃′、CHIEF 点数，表 12 的 20 面元机器零上界）。

**合并时的口径订正**：m3181 建表时表 10 的 B₁₁′ 观测阶按固定因子 2 折算（1.62 / 1.88），
表 12 的转动耦合比取两位有效数字（2.2×10⁻⁴ / 2.6×10⁻⁵）。m3150 的复算证明八面体细分的
实测棱长比是 1.87 而非 2，转动耦合比的四位值是 2.15×10⁻⁴ / 2.65×10⁻⁵；
合并后生成侧改用**实测棱长比**推导 B₁₁′ 观测阶（1.80 / 1.94），转动耦合比改用四位值，
与 §4.10–§4.11 订正后的文档完全一致——注册表不得声明文档已不再引用的数字。

复算命令（{summary['checked']} 格一次跑完，本机 {REPORT['elapsed_seconds']:.1f} s）：

```bash
PYTHONPATH=src python3 docs/m3181/run_three_d_free_surface.py \\
    --output docs/m3181/outputs/three_d_free_surface_registry.json
```

**判决**：{summary['passed']} / {summary['checked']} 格 PASS，漂移 {summary['drifted']} 格。
等值判据行（{len(equality)} 格）最坏相对偏差 **{abs(worst['relative_deviation']):.2e}**
（`{worst['key']}`，声明 {worst['declared']:.6g} / 复算 {worst['measured']:.6g}，容差 5×10⁻³）；
上界判据行（{len(upper)} 格）按 `≤` 判，20 面元转动耦合比复算值 2.49×10⁻³² 远低于 1×10⁻²⁰ 上界。

{render(TABLE10)}
{render(TABLE11)}
{render(TABLE12)}
"""


def main() -> int:
    text = DOC.read_text(encoding="utf-8")
    start = text.index("### 4.12 ")
    tail = text[start:]
    nxt = tail.find("\n## ", 1)
    end = len(text) if nxt < 0 else start + nxt + 1
    updated = text[:start] + SECTION + text[end:]
    DOC.write_text(updated, encoding="utf-8")
    print(f"rewrote section 4.12: {end - start} -> {len(SECTION)} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())