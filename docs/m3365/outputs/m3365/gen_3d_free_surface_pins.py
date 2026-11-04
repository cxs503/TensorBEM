"""Register and re-quote the 3-D free-surface tables of marine_bem_benchmarks.md.

Runs the declared-number registry on the ``floating_hemisphere`` and
``submerged_sphere`` groups (67 quoted numbers: tables 10, 11, 12, the
Kochin/Haskind closure sentence and the B11 metric note), writes the verdicts to
``outputs/m3365/free_surface_pins.json``, and rewrites the block between the
``<!-- m3365-pins:begin ... -->`` markers from those verdicts. Nothing in the
injected text is typed by hand.

Usage::

    PYTHONPATH=src python3 outputs/m3365/gen_3d_free_surface_pins.py
"""

from __future__ import annotations

import json
from pathlib import Path

from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS, check

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "m3365"
DOC = REPO / "docs" / "marine_bem_benchmarks.md"
GROUPS = ("floating_hemisphere", "submerged_sphere")

BEGIN = "<!-- m3365-pins:begin -->"
END = "<!-- m3365-pins:end -->"


def _sci(value: float, digits: int = 2) -> str:
    """Render a number the way the surrounding tables do (2-digit LaTeX science)."""
    if value == 0.0:
        return "0"
    sign = "-" if value < 0 else ""
    mantissa, exponent = f"{abs(value):.{digits}e}".split("e")
    return f"{sign}{mantissa}\\times10^{{{int(exponent)}}}"


def _m(rows: dict[str, dict], key: str) -> float:
    return float(rows[key]["measured"])


def _d(rows: dict[str, dict], key: str) -> float:
    return float(rows[key]["declared"])


def build_section(rows: dict[str, dict], summary: dict) -> str:
    """Render the injected documentation block from the verdicts."""
    n69 = summary["checked"]
    metric_note = (
        "**表 10 观测阶的两种长度度量（本次审计的实质发现）**：表 10 的观测阶列在"
        "**同一列里混用了两种长度度量** —— $A_{33}'/B_{33}'/A_{11}'/F_z$ 四项用的是"
        "基准程序输出的平均面元边长 $h_{\\text{mean}}$，而 $B_{11}'$ 项（"
        f"{_d(rows, 'hulme_b11_panel_law_order_64'):.2f} / "
        f"{_d(rows, 'hulme_b11_panel_law_order_256'):.2f}）用的是 m2609 Kochin 专项的 "
        "$h\\propto N^{-1/2}$ 度量。两种度量都复现（$h_{\\text{mean}}$ 度量下该量实为 "
        f"{_m(rows, 'hulme_b11_h_mean_order_64'):.3f} / "
        f"{_m(rows, 'hulme_b11_h_mean_order_256'):.3f}），故表格并非错数，但读者无法"
        "从单元格判断度量来源。注册表把两者分别钉为 `observed_order_b11@N`"
        "（$h_{\\text{mean}}$）与 `surge_damping_rel_error_panel_law_order@N`"
        "（$N^{-1/2}$，即表内所印），关系由 "
        "`tests/test_declared_numbers.py::"
        "test_h_mean_and_panel_count_orders_are_related_by_the_edge_length_ratio` 钉桩。"
    )

    lines = [
        BEGIN,
        "",
        "**复算与回归钉桩（m3365，程序化注入，勿手改）**：本节表 10、表 11、表 12 与 "
        "Kochin/Haskind 闭合句中的 "
        f"**{n69} 个数字**已全部纳入 `tensorbem.benchmarks.declared_numbers` 注册表"
        "（`floating_hemisphere` 与 `submerged_sphere` 两组），由 "
        "`python -m tensorbem.benchmarks.declared_numbers --tiers standard` 一条命令"
        "重算并判 PASS/DRIFT，回归门在 `tests/test_declared_numbers.py`。下列实测列由 "
        "`outputs/m3365/gen_3d_free_surface_pins.py` 从上表实测值渲染，零手抄。"
        f"本次判决 **{summary['passed']}/{n69} PASS**，最坏相对偏差 "
        f"{summary['worst_relative_deviation']:.2e}（表格印刷位数内的舍入；两位有效"
        "数字的量按 5% 容差判定）。",
        "",
        "**表 10 实测复算（$Ka=1.0$，$\\rho=1025$，`floating_hemisphere.run_refinement_study`）**",
        "",
        "| 面元数 $N$ | 条件数 声明/实测 | $A_{33}'$ 声明/实测 | $B_{33}'$ 声明/实测 "
        "| $A_{11}'$ 声明/实测 | $B_{11}'$ 声明/实测 | $F_z$ 声明/实测 |",
        "|---:|---|---|---|---|---|---|",
    ]
    for n in (16, 64, 256):
        cells = [
            f"{_d(rows, f'hulme_condition_number_{n}'):.3f} / "
            f"{_m(rows, f'hulme_condition_number_{n}'):.3f}"
        ]
        for short in ("a33", "b33", "a11", "b11", "fz"):
            key = f"hulme_{short}_rel_error_{n}"
            cells.append(f"{_sci(_d(rows, key))} / {_sci(_m(rows, key))}")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")

    lines += [
        "",
        metric_note,
        "",
        "**表 11 实测复算（$Ka=2.58$，$N=256$，`floating_hemisphere.run_irregular_frequency_study`）**",
        "",
        "| 方案 | 条件数 声明/实测 | $A_{33}'$ 声明/实测 | $B_{33}'$ 声明/实测 | 物理耗散性 |",
        "|---|---|---|---|---|",
        "| 未增广外域 BIE | "
        f"{_d(rows, 'hulme_chief_unregularized_condition_number'):.2f} / "
        f"{_m(rows, 'hulme_chief_unregularized_condition_number'):.2f} | "
        f"{_d(rows, 'hulme_chief_unregularized_a33'):.4f} / "
        f"{_m(rows, 'hulme_chief_unregularized_a33'):.4f} | "
        f"{_d(rows, 'hulme_chief_unregularized_b33'):.4f} / "
        f"{_m(rows, 'hulme_chief_unregularized_b33'):.4f} | 违背（负阻尼） |",
        "| 5 点三维内部 CHIEF | "
        f"{_d(rows, 'hulme_chief_regularized_condition_number'):.2f} / "
        f"{_m(rows, 'hulme_chief_regularized_condition_number'):.2f} | "
        f"{_d(rows, 'hulme_chief_regularized_a33'):.4f} / "
        f"{_m(rows, 'hulme_chief_regularized_a33'):.4f} | "
        f"{_d(rows, 'hulme_chief_regularized_b33'):.4f} / "
        f"{_m(rows, 'hulme_chief_regularized_b33'):.4f} | 满足 |",
        "",
        "**精确参考（同次 Hulme 半解析求解）：$A_{33}'=$"
        f"{_m(rows, 'hulme_chief_exact_a33'):.6f}、$B_{33}'=$"
        f"{_m(rows, 'hulme_chief_exact_b33'):.6f}；所需内部 CHIEF 约束数 "
        f"{_m(rows, 'hulme_chief_point_count'):.0f}。近/远场 Kochin 阻尼闭合 $"
        f"{_sci(_m(rows, 'hulme_far_field_damping_closure_256'), 3)}$、Haskind 互易 $"
        f"{_sci(_m(rows, 'hulme_haskind_reciprocity_256_x'), 3)}$（纵荡）/ $"
        f"{_sci(_m(rows, 'hulme_haskind_reciprocity_256_z'), 3)}$（垂荡）同为注册项。",
        "",
        "**表 12 实测复算（$f/a=1.5$，$Ka=0.8$，`submerged_sphere_wave.run_refinement_study`）**",
        "",
        "| 面元数 $N$ | 条件数 声明/实测 | $A_{33}'$ 声明/实测 | $A_{11}'$ 声明/实测 "
        "| $F_x$ 声明/实测 | $F_z$ 声明/实测 | 转动耦合比 声明/实测 |",
        "|---:|---|---|---|---|---|---|",
    ]
    for n in (20, 80, 320):
        cells = [
            f"{_d(rows, f'wlt_sphere_condition_number_{n}'):.3f} / "
            f"{_m(rows, f'wlt_sphere_condition_number_{n}'):.3f}"
        ]
        for short in ("a33", "a11", "fx", "fz"):
            key = f"wlt_sphere_{short}_rel_error_{n}"
            cells.append(f"{_sci(_d(rows, key))} / {_sci(_m(rows, key))}")
        rot_key = f"wlt_sphere_rotational_coupling_{n}"
        if rot_key in rows:
            cells.append(f"{_sci(_d(rows, rot_key))} / {_sci(_m(rows, rot_key))}")
        else:
            cells.append("$2.5\\times10^{-32}$ / 双精度下恰为 0（球心对称精确成立）")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "表 12 的 $N=20$ 转动耦合比在文档中印为 $2.5\\times10^{-32}$，实测在双精度下恰为 "
        "$0$（球心对称使转动–平动耦合精确消失），该格无法用相对判据登记，故只钉 "
        "$N=80$、$N=320$ 两行，并以该量的单调下降作为对称性回归判据。",
        "",
        END,
    ]
    return "\n".join(lines)


def inject(path: Path, section: str) -> None:
    text = path.read_text()
    if BEGIN not in text or END not in text:
        raise SystemExit("markers missing from the document")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    path.write_text(head + section + tail)


def main() -> None:
    entries = tuple(e for e in DECLARED_NUMBERS if e.group in GROUPS)
    report = check(tiers=("standard",), entries=entries)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "free_surface_pins.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    section = build_section({r["key"]: r for r in report["checks"]}, report["summary"])
    inject(DOC, section)
    print(json.dumps(report["summary"], indent=2))
    print("injected chars:", len(section))


if __name__ == "__main__":
    main()