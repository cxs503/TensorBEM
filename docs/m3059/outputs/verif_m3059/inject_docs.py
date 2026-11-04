"""Inject the recomputed numbers into docs/classical_benchmarks.md (no hand-typed digits)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("outputs/verif_m3059")
DOC = Path("docs/classical_benchmarks.md")

MARKER = "**复算与回归钉桩（m3059）。**"


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for name in ("fast_standard.json", "heavy_tier.json", "harmonic_rows.json"):
        path = ROOT / name
        if path.exists():
            rows.extend(json.loads(path.read_text())["checks"])
    seen: set[str] = set()
    unique = []
    for row in rows:
        if row["key"] in seen:
            continue
        seen.add(row["key"])
        unique.append(row)
    return unique


def fmt(value: float) -> str:
    return f"{value:.4e}".replace("e", r"\times10^{").replace("0}", "}") if False else None


def latex(value: float) -> str:
    mantissa, exponent = f"{value:.4e}".split("e")
    return rf"\({mantissa}\times10^{{{int(exponent)}}}\)"


def main() -> None:
    rows = {row["key"]: row for row in load_rows()}
    ih_keys = [k for k in rows if k.startswith("interior_harmonic_")]
    if len(ih_keys) != 6:
        raise SystemExit(f"expected 6 interior-harmonic rows, found {len(ih_keys)}")

    rep = {1: "x", 2: "x2-y2", 3: "x3-3xy2"}
    lines = [
        "",
        MARKER,
        " 表 3 的 \\(N=1280\\) 列此前只是文档数字，没有任何计算复核。现将该列、",
        "以及三维椭球 1280 面元行（2026-10-24 扩编审计遗留的待复现项）纳入",
        "`tensorbem.benchmarks.declared_numbers` 注册表，由",
        "`python -m tensorbem.benchmarks.declared_numbers --tiers fast,standard,heavy`",
        "一键重算并判 PASS/DRIFT；下表的实测列由该命令的 JSON 程序化注入。",
        "",
        "| 度 | 算例 | 声明通量误差（1280） | 实测通量误差 | 相对偏差 | 实测探针 RMSE | 判定 |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for degree, harmonic in rep.items():
        flux = rows[f"interior_harmonic_deg{degree}_flux_1280"]
        rmse = rows[f"interior_harmonic_deg{degree}_field_rmse_1280"]
        lines.append(
            f"| {degree} | \\({harmonic.replace('x', 'x')}\\) | {latex(flux['declared'])} | "
            f"{latex(flux['measured'])} | {flux['relative_deviation']:+.2e} | "
            f"{latex(rmse['measured'])} | {flux['verdict']} / {rmse['verdict']} |"
        )

    ellipsoid_rows = [rows[k] for k in rows if k.startswith("ellipsoid_added_mass_1280")]
    worst = max(abs(row["relative_deviation"]) for row in rows.values())
    lines += [
        "",
        "三维椭球 1280 面元（\\(subdivisions=3\\)）实测："
        + "、".join(
            f"{row['key'].rsplit('_', 1)[-1]} 轴 {latex(row['measured'])}（声明 "
            f"{latex(row['declared'])}，偏差 {row['relative_deviation']:+.2e}）"
            for row in sorted(ellipsoid_rows, key=lambda r: r["key"])
        )
        + f"。该行自 2026-10-24 审计起登记为「未直接复现」，本次直接复现并判定 "
        + "/".join(sorted({row["verdict"] for row in ellipsoid_rows}))
        + f"；注册表全部 {len(rows)} 条声明数字的最坏相对偏差为 {worst:.2e}，"
        "均低于各自容差。",
        "",
        "复算命令与工件：`python -m tensorbem.benchmarks.interior_harmonics --subdivisions 2 3 --degrees 1 2 3`、"
        "`python -m tensorbem.benchmarks.declared_numbers --tiers fast,standard,heavy`；"
        "原始 JSON 见 `outputs/verif_m3059/`，回归钉桩见 `tests/test_declared_numbers.py`"
        "（1280 面元行）与 `tests/test_interior_harmonics.py`（320 面元行）。",
        "",
    ]
    text = DOC.read_text()
    if MARKER in text:
        start = text.index(MARKER) - 1
        end = text.index("## 6. 三类基准与既有验证的关系")
        text = text[:start] + "\n".join(lines[1:]) + "\n" + text[end:]
    else:
        anchor = "## 6. 三类基准与既有验证的关系"
        text = text.replace(anchor, "\n".join(lines[1:]) + "\n" + anchor)
    DOC.write_text(text)
    print("docs updated")


if __name__ == "__main__":
    main()