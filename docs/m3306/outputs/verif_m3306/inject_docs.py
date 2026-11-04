"""Inject the re-computed 3-D free-surface numbers into the benchmark document.

The block between the ``BEGIN``/``END`` markers is generated from
``outputs/verif_m3306/standard_tier.json``; re-running this script after a
re-solve refreshes the document without transcription.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = HERE.parents[1] / "docs" / "marine_bem_benchmarks.md"
BEGIN = "<!-- BEGIN m3306 declared-number pins -->"
END = "<!-- END m3306 declared-number pins -->"

GROUPS = (
    ("hulme_hemisphere", "表 10 · Hulme 漂浮半球（Ka=1, 64→256 面元）"),
    ("hulme_irregular", "表 11 · Ka=2.58 不规则频率与免盖板 CHIEF（256 面元）"),
    ("submerged_sphere", "表 12 · Wang/Linton 潜没球体（f/a=1.5, Ka=0.8, 80→320 面元）"),
)


def _fmt(value: float) -> str:
    if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
        return f"{value:.4e}"
    return f"{value:.5g}"


def build_block(report: dict) -> str:
    rows = report["checks"]
    new_rows = [row for row in rows if row["group"] in {g for g, _ in GROUPS}]
    lines = [BEGIN, ""]
    lines.append(
        "**注册表钉桩（m3306）**：上述三张表的 %d 个数字过去只由各基准测试模块"
        "独立断言，未纳入 `tensorbem.benchmarks.declared_numbers` 注册表——"
        "一次核函数或网格改动可以让文档格子悄悄漂移而无人发现。现按表逐格登记，"
        "复算命令 `python -m tensorbem.benchmarks.declared_numbers --tiers standard`"
        "（本次运行 %d 条全部 PASS，最坏相对偏差 %.2e），回归门为 "
        "`tests/test_declared_numbers.py`。" % (len(new_rows), len(rows),
                                               report["summary"]["worst_relative_deviation"])
    )
    lines.append("")
    for group, title in GROUPS:
        subset = [row for row in new_rows if row["group"] == group]
        if not subset:
            continue
        lines.append(f"**{title}**")
        lines.append("")
        lines.append("| 注册键 | 文档值 | 复算值 | 相对偏差 | 容差 | 判定 |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for row in subset:
            lines.append(
                "| `{}` | {} | {} | {:+.2e} | {:.1e} | {} |".format(
                    row["key"],
                    _fmt(row["declared"]),
                    _fmt(row["measured"]),
                    row["relative_deviation"],
                    row["tolerance"],
                    row["verdict"],
                )
            )
        lines.append("")
    lines.append(END)
    return "\n".join(lines)


def main() -> None:
    with open(HERE / "standard_tier.json") as handle:
        report = json.load(handle)
    block = build_block(report)
    text = DOC.read_text()
    if BEGIN in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        text = head + block + tail
    else:
        anchor = "python -m tensorbem.benchmarks.submerged_sphere_wave --subdivisions 0 1 2\n```\n"
        assert anchor in text, "anchor for the injection not found"
        text = text.replace(anchor, anchor + "\n" + block + "\n", 1)
    DOC.write_text(text)
    print("injected", DOC)


if __name__ == "__main__":
    main()