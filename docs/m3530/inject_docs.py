"""Inject the m3530 study tables into the markdown report (no hand-copied digits).

Rewrites everything between the ``<!-- BEGIN:generated -->`` and
``<!-- END:generated -->`` markers of ``near_surface_three_way.md`` from
``near_surface_three_way.json``.

Run: python docs/m3530/inject_docs.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
JSON_PATH = HERE / "near_surface_three_way.json"
MD_PATH = HERE / "near_surface_three_way.md"
BEGIN = "<!-- BEGIN:generated -->"
END = "<!-- END:generated -->"


def sci(value: float, digits: int = 2) -> str:
    if value == 0.0:
        return "0"
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{mantissa}\\times10^{{{int(exponent)}}}"


def main() -> None:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    rows = payload["rows"]
    lines: list[str] = []

    lines.append("### 参考值：Ursell 多极子闭式解（120 项）\n")
    lines.append(
        "| $f/a$ | $ka$ | $\\mu$ | $\\nu$ | $A$ [kg/m] | $B$ [kg/(m·s)] | 60↔120 项相对差 |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        reference = row["reference"]
        delta = max(row["reference_truncation_delta"].values())
        lines.append(
            f"| {row['depth_ratio']:.2f} | {row['ka']:.2f} | {reference['mu']:.6f} | "
            f"{reference['nu']:.6f} | {reference['added_mass']:.4f} | "
            f"{reference['damping']:.4f} | {sci(delta)} |"
        )

    lines.append("\n### 三条腿的对照（面板 32→256、MFS 48→192、解析参考）\n")
    header = ("| $f/a$ | $ka$ | 面元 | 面板 $\\varepsilon_A$ | 面板 $\\varepsilon_B$ | "
              "观测阶 ($A$) | 观测阶 ($B$) | MFS 源数 | MFS $\\varepsilon_A$ | MFS $\\varepsilon_B$ |")
    lines.append(header)
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        panel_rungs = sorted(row["panels"].items(), key=lambda item: int(item[0]))
        mfs_rungs = sorted(row["mfs"].items(), key=lambda item: int(item[0]))
        for index, (count, rung) in enumerate(panel_rungs):
            order_text = "—"
            if index > 0:
                coarse = panel_rungs[index - 1][1]
                ratio = coarse["panel_spacing"] / rung["panel_spacing"]
                order_text = (
                    f"{math.log(coarse['added_mass_error'] / rung['added_mass_error']) / math.log(ratio):.2f}"
                    " / "
                    f"{math.log(coarse['damping_error'] / rung['damping_error']) / math.log(ratio):.2f}"
                )
            mfs_text = "—"
            mfs_a = mfs_b = "—"
            if index < len(mfs_rungs):
                mfs_count, mfs_rung = mfs_rungs[index]
                mfs_text = mfs_count
                mfs_a = sci(mfs_rung["added_mass_error"])
                mfs_b = sci(mfs_rung["damping_error"])
            lines.append(
                f"| {row['depth_ratio']:.2f} | {row['ka']:.2f} | {count} | "
                f"{sci(rung['added_mass_error'])} | {sci(rung['damping_error'])} | "
                f"{order_text.split(' / ')[0] if order_text != '—' else '—'} | "
                f"{order_text.split(' / ')[1] if order_text != '—' else '—'} | "
                f"{mfs_text} | {mfs_a} | {mfs_b} |"
            )

    verdict = payload["verdict"]
    lines.append("\n### 判定量\n")
    lines.append("| 量 | 值 |")
    lines.append("|---|---:|")
    lines.append(
        f"| 面板 256 的最大 $\\varepsilon_A$（${verdict['panel_256_peak_error']['added_mass']['error']:.3e}$，"
        f"出现在 $f/a={verdict['panel_256_peak_error']['added_mass']['depth_ratio']}$、"
        f"$ka={verdict['panel_256_peak_error']['added_mass']['ka']}$） | "
        f"{sci(verdict['panel_256_peak_error']['added_mass']['error'])} |"
    )
    lines.append(
        f"| 面板 256 的最大 $\\varepsilon_B$ | "
        f"{sci(verdict['panel_256_peak_error']['damping']['error'])} |"
    )
    lines.append(
        f"| 面板 256 的最小 $\\varepsilon_A$ | {sci(verdict['panel_256_best_error']['added_mass']['error'])} |"
    )
    lines.append(
        f"| MFS 48 源在全部工作点的最大 $\\varepsilon_A$ | "
        f"{sci(verdict['mfs_48_worst_error']['added_mass'])} |"
    )
    lines.append(
        f"| MFS 48 源在全部工作点的最大 $\\varepsilon_B$ | "
        f"{sci(verdict['mfs_48_worst_error']['damping'])} |"
    )
    lines.append(
        f"| 256 面元下 $\\varepsilon_A<3\\%$ 的 $f/a$ | "
        f"{verdict['depth_ratios_meeting_3pct_at_256_panels']['added_mass']} |"
    )
    lines.append(
        f"| 256 面元下 $\\varepsilon_B<3\\%$ 的 $f/a$ | "
        f"{verdict['depth_ratios_meeting_3pct_at_256_panels']['damping']} |"
    )

    crux = next(row for row in rows
                if (row["depth_ratio"], row["ka"]) == (1.05, 1.0))
    crux_panels = sorted(crux["panels"].items(), key=lambda item: int(item[0]))
    lines.append("\n### 关键点 $f/a=1.05$、$ka=1.0$ 的完整阶梯（含 512 面元附加档）\n")
    lines.append("| 面元 | $h=2\\pi a/N$ | $\\varepsilon_A$ | $\\varepsilon_B$ | 条件数 | 相对残差 | 墙钟 [s] |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for count, rung in crux_panels:
        lines.append(
            f"| {count} | {rung['panel_spacing']:.5f} | {sci(rung['added_mass_error'])} | "
            f"{sci(rung['damping_error'])} | {rung['condition_number']:.2f} | "
            f"{sci(rung['relative_residual'])} | {rung['wall_time_s']:.1f} |"
        )

    block = "\n".join(lines)
    text = MD_PATH.read_text(encoding="utf-8")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    MD_PATH.write_text(f"{head}{BEGIN}\n{block}\n{END}{tail}", encoding="utf-8")
    print(f"injected {len(lines)} lines into {MD_PATH}")


if __name__ == "__main__":
    main()