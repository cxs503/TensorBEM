"""Render the ice-navigation tutorial: numbers, criteria tables and figures.

Nothing in ``docs/ice_navigation_tutorial`` is typed by hand.  This script

1. computes every live chapter payload (:mod:`ice_tutorial.model`),
2. evaluates the machine-readable criteria registry (:mod:`ice_tutorial.criteria`),
3. writes ``artifacts/tutorial_numbers.json`` (single evidence bundle),
4. writes ``numbers/chXX.md`` (the human-readable tables),
5. draws ``figs/*.png`` with matplotlib (Agg backend),
6. injects the rendered blocks into the chapter markdown between the
   ``<!-- AUTO-...:chXX -->`` markers.

Running it twice produces byte-identical markdown, which is exactly what
``tests/test_ice_tutorial.py`` checks (rendering-fingerprint drift guard).

Usage
-----
    python examples/ice_tutorial/render.py                     # full, writes docs
    python examples/ice_tutorial/render.py --check             # verify only, no write
    python examples/ice_tutorial/render.py --no-bem --no-figs  # fast text-only pass
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = Path(__file__).resolve().parents[1]
for candidate in (str(_ROOT / "src"), str(_EXAMPLES)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from ice_tutorial import criteria as criteria_module  # noqa: E402
from ice_tutorial import model as model_module  # noqa: E402

DOCS_REL = Path("docs") / "ice_navigation_tutorial"

CHAPTER_FILES: dict[str, str] = {
    "ch01": "ch01_ice_regime.md",
    "ch02": "ch02_mass_loading_dispersion.md",
    "ch03": "ch03_wave_field_under_ice.md",
    "ch04": "ch04_added_mass_anchors.md",
    "ch05": "ch05_hull_rao_chain.md",
    "ch06": "ch06_ice_resistance.md",
    "ch07": "ch07_structural_ice_loads.md",
    "ch08": "ch08_operations_and_regulation.md",
    "ch09": "ch09_verification_protocol.md",
}

FIGURE_FILES: dict[str, str] = {
    "ch01": "fig_ch01_regimes.png",
    "ch02": "fig_ch02_dispersion.png",
    "ch03": "fig_ch03_wave_field.png",
    "ch04": "fig_ch04_added_mass.png",
}


# ---------------------------------------------------------------------------
# formatting helpers (single source of truth for rendered digits)
# ---------------------------------------------------------------------------
def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        if math.isinf(value):
            return "∞"
        if value != 0.0 and abs(value) < 1e-3:
            return f"{value:.2e}"
        return f"{value:.{digits}g}"
    return str(value)


def _md_table(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_criteria(chapter: str, rows: Sequence[Mapping[str, Any]]) -> str:
    """Markdown criteria table of one chapter (the 判据表)."""
    stats = criteria_module.summary(rows)
    head = (
        f"判据表（{chapter}）：可计算 {stats['passed'] + stats['failed']} 行、"
        f"其中通过 {stats['passed']} 行、未通过 {stats['failed']} 行；"
        f"文献/法规行 {stats['textual']} 行。"
    )
    table = _md_table(
        ["判据 ID", "判据（可检验陈述）", "阈值/区间", "本次实测", "结论", "证据", "出处"],
        [
            [
                row["id"],
                row["statement"],
                row["hold"],
                row["observed"],
                row["verdict"],
                row["evidence"],
                row["ref"],
            ]
            for row in rows
        ],
    )
    return f"{head}\n\n{table}\n"


def render_numbers(chapter: str, payload: Mapping[str, Any]) -> str:
    """Markdown number block (parameter table + data tables) of one chapter."""
    parts = [
        f"### {payload['title']} — 机器渲染数值表",
        "",
        "本表由 `examples/ice_tutorial/render.py` 从 `artifacts/tutorial_numbers.json` 生成，"
        "正文与表格均无手抄数值。",
        "",
        _md_table(
            ["参数", "取值", "单位", "说明"],
            [[p[0], p[1], p[2], p[3]] for p in payload["params"]],
        ),
    ]
    for table in payload["tables"]:
        parts += [
            "",
            f"**表 {table['key']}** — {table['caption']}",
            "",
            _md_table(table["columns"], table["rows"]),
        ]
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def figure_ch01(payload: Mapping[str, Any], path: Path) -> None:
    plt = _pyplot()
    series = payload["series"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    ax.plot(series["h"], series["mu"], "o-", color="#1f4e79", label=r"$\mu=\rho_i h_i/\rho_w$")
    ax.set_xlabel(r"冰厚 $h_i$ [m]")
    ax.set_ylabel(r"质量加载参数 $\mu$ [-]", color="#1f4e79")
    ax.tick_params(axis="y", labelcolor="#1f4e79")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(series["h"], series["cutoff_period"], "s--", color="#c0504d", label=r"$T_c$")
    ax2.set_ylabel(r"截止周期 $T_c$ [s]", color="#c0504d")
    ax2.tick_params(axis="y", labelcolor="#c0504d")
    ax.set_title("(a) 一年冰盖参数谱：$\\mu$ 与 $T_c$")

    ax = axes[1]
    ax.plot(series["h"], series["wavelength_ratio"], "^-", color="#4f6228")
    ax.set_xlabel(r"冰厚 $h_i$ [m]")
    ax.set_ylabel(r"$\lambda_{ice}/\lambda_{open}$ (T = 8 s) [-]")
    ax.grid(alpha=0.3)
    ax.set_title("(b) 冰下波长缩短比")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_ch02(payload: Mapping[str, Any], path: Path) -> None:
    plt = _pyplot()
    series = payload["series"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    ax.plot(series["k_dense"], series["omega_closed_dense"], "-", color="#1f4e79", label="Weitz-Keller 有限水深闭式解")
    ax.plot(series["k_dense"], series["omega_deep_dense"], ":", color="#4f6228", label="深水极限 $\\omega^2=gk/(1+\\mu k)$")
    ax.plot(series["k_dense"], series["omega_open_dense"], "--", color="#808080", label="敞水 $\\omega^2=gk\\tanh kH$")
    ax.plot(series["k"], series["omega_bem"], "o", mfc="none", color="#c0504d", label="垂直 BEM（32 面板/波长）")
    ax.set_xlabel("k [1/m]")
    ax.set_ylabel("$\\omega$ [rad/s]")
    ax.set_title("(a) 色散关系：闭式解 / 极限 / BEM")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.loglog(series["conv_ppw"], series["conv_error"], "o-", color="#c0504d")
    ax.set_xlabel("面板/波长 [-]")
    ax.set_ylabel("$|\\omega_{BEM}/\\omega_{closed}-1|$ [-]")
    ax.set_title("(b) 面板加密收敛（k = 0.1 1/m）")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_ch03(payload: Mapping[str, Any], path: Path) -> None:
    plt = _pyplot()
    series = payload["series"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    periods = [t for t, p in zip(series["period"], series["propagating"]) if p]
    ratios = [r for r, p in zip(series["wavelength_ratio"], series["propagating"]) if p]
    amps = [a for a, p in zip(series["amplitude_ratio"], series["propagating"]) if p]
    ax.plot(periods, ratios, "o-", color="#1f4e79", label=r"$\lambda_{ice}/\lambda_{open}$")
    ax.set_xlabel("周期 T [s]")
    ax.set_ylabel(r"$\lambda_{ice}/\lambda_{open}$ [-]", color="#1f4e79")
    ax.tick_params(axis="y", labelcolor="#1f4e79")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(periods, amps, "s--", color="#c0504d", label=r"$A_{ice}/A_{open}$")
    ax2.set_ylabel(r"$A_{ice}/A_{open}$ [-]", color="#c0504d")
    ax2.tick_params(axis="y", labelcolor="#c0504d")
    ax.axvline(series["cutoff_period"], color="k", ls=":", lw=1.0)
    ax.text(series["cutoff_period"] + 0.15, max(ratios) * 0.55, f"$T_c$ = {series['cutoff_period']:.3f} s", fontsize=7)
    ax.set_title("(a) 传播分支：波长缩短与能量通量幅值比")

    ax = axes[1]
    ev_periods = [t for t, p in zip(series["period"], series["propagating"]) if not p]
    efold = [e for e, p in zip(series["efold"], series["propagating"]) if not p]
    if ev_periods:
        ax.semilogy(ev_periods, efold, "o-", color="#4f6228")
        ax.set_ylabel("e 折长度 [m]")
    else:  # pragma: no cover - defensive
        ax.text(0.5, 0.5, "no evanescent sample", ha="center")
    ax.set_xlabel("周期 T [s]（$T<T_c$）")
    ax.set_title("(b) 消失波分支：衰减长度")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_ch04(payload: Mapping[str, Any], path: Path) -> None:
    plt = _pyplot()
    series = payload["series"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    lamb = series["lamb"]
    ax = axes[0]
    ax.plot(series["depth_ratio"], [v / lamb for v in series["depth_values"]], "-", color="#1f4e79", label="多极精确解")
    ax.plot(
        [r[0] for r in payload["tables"][0]["rows"]],
        [r[1] / lamb for r in payload["tables"][0]["rows"]],
        "o",
        mfc="none",
        color="#c0504d",
        label="BEM（168 / 352 面元）",
    )
    ax.axhline(1.0, color="k", ls=":", lw=1.0, label="Lamb 极限")
    ax.set_xlabel("d/R [-]")
    ax.set_ylabel(r"$A_{33}/A_{33}^{Lamb}$ [-]")
    ax.set_title("(a) 潜没深度：回到 Lamb 极限")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(series["omega"], [100.0 * c for c in series["contrast"]], "o-", color="#4f6228")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel(r"$(A_{33}^{ice}/A_{33}^{open}-1)\times 100$ [%]")
    ax.set_title("(b)【负结果】冰盖几乎不改变附加质量")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


FIGURE_FUNCTIONS = {
    "ch01": figure_ch01,
    "ch02": figure_ch02,
    "ch03": figure_ch03,
    "ch04": figure_ch04,
}


# ---------------------------------------------------------------------------
# injection
# ---------------------------------------------------------------------------
def inject(text: str, chapter: str, kind: str, body: str) -> str:
    """Replace the block between ``<!-- AUTO-<kind>:<chapter> -->`` markers."""
    start = f"<!-- AUTO-{kind}:{chapter} -->"
    end = f"<!-- /AUTO-{kind}:{chapter} -->"
    i = text.find(start)
    j = text.find(end)
    if i < 0 or j < 0 or j < i:
        raise ValueError(f"missing marker pair AUTO-{kind}:{chapter}")
    head = text[: i + len(start)]
    tail = text[j:]
    return f"{head}\n{body.rstrip()}\n{tail}"


def fingerprint(payloads: Mapping[str, Mapping[str, Any]], verdicts: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, str]:
    """A few key scalars whose rendered form must not change silently."""
    marks: dict[str, str] = {}
    for chapter, payload in payloads.items():
        for key in sorted(payload["metrics"]):
            marks[f"{chapter}.{key}"] = fmt(payload["metrics"][key])
    for chapter, rows in verdicts.items():
        failed = [r["id"] for r in rows if r["verdict"] == "未通过"]
        marks[f"{chapter}.failed"] = ",".join(failed) if failed else "none"
    return marks


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build(
    *,
    docs: Path,
    with_bem: bool = True,
    figures: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    artifacts = docs / "artifacts"
    numbers_dir = docs / "numbers"
    figs_dir = docs / "figs"

    payloads: dict[str, Any] = {}
    for chapter in model_module.COMPUTED_CHAPTERS:
        payloads[chapter] = model_module.compute(chapter, with_bem=with_bem)

    verdicts: dict[str, list[dict[str, Any]]] = {}
    for chapter in criteria_module.chapters():
        metrics = payloads.get(chapter, {}).get("metrics", {})
        verdicts[chapter] = criteria_module.evaluate(chapter, metrics)

    marks = fingerprint(payloads, verdicts)

    if write:
        artifacts.mkdir(parents=True, exist_ok=True)
        numbers_dir.mkdir(parents=True, exist_ok=True)
        figs_dir.mkdir(parents=True, exist_ok=True)

        bundle = {
            "generator": "examples/ice_tutorial/render.py",
            "with_bem": with_bem,
            "profiles": {k: {"params": v["params"], "tables": v["tables"], "metrics": v["metrics"]} for k, v in payloads.items()},
            "criteria": verdicts,
            "fingerprint": marks,
        }
        (artifacts / "tutorial_numbers.json").write_text(
            json.dumps(bundle, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )

    for chapter, payload in payloads.items():
        body = render_numbers(chapter, payload)
        if write:
            (numbers_dir / f"{chapter}.md").write_text(body, encoding="utf-8")
        if figures and chapter in FIGURE_FUNCTIONS:
            if write:
                FIGURE_FUNCTIONS[chapter](payload, figs_dir / FIGURE_FILES[chapter])
        chapter_path = docs / CHAPTER_FILES[chapter]
        if not chapter_path.exists():
            raise FileNotFoundError(chapter_path)
        text = chapter_path.read_text(encoding="utf-8")
        text = inject(text, chapter, "NUMBERS", body)
        text = inject(text, chapter, "CRITERIA", render_criteria(chapter, verdicts[chapter]))
        if write:
            chapter_path.write_text(text, encoding="utf-8")

    for chapter in model_module.TEXT_CHAPTERS:
        chapter_path = docs / CHAPTER_FILES[chapter]
        if not chapter_path.exists():
            raise FileNotFoundError(chapter_path)
        text = chapter_path.read_text(encoding="utf-8")
        text = inject(text, chapter, "CRITERIA", render_criteria(chapter, verdicts[chapter]))
        if write:
            chapter_path.write_text(text, encoding="utf-8")

    return {"payloads": payloads, "verdicts": verdicts, "fingerprint": marks}


def expected_blocks(*, docs: Path, with_bem: bool = True) -> dict[tuple[str, str], str]:
    """Recompute, without writing, the exact block text each marker should hold."""
    data = build(docs=docs, with_bem=with_bem, figures=False, write=False)
    blocks: dict[tuple[str, str], str] = {}
    for chapter, payload in data["payloads"].items():
        blocks[(chapter, "NUMBERS")] = render_numbers(chapter, payload)
    for chapter, rows in data["verdicts"].items():
        blocks[(chapter, "CRITERIA")] = render_criteria(chapter, rows)
    return blocks


def all_failed(verdicts: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[str]:
    out: list[str] = []
    for chapter, rows in verdicts.items():
        out += [f"{r['id']}: {r['statement']}" for r in rows if r["verdict"] == "未通过"]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=_ROOT / DOCS_REL)
    parser.add_argument("--no-bem", action="store_true", help="skip the BEM routes (fast text pass)")
    parser.add_argument("--no-figs", action="store_true", help="skip matplotlib output")
    parser.add_argument("--check", action="store_true", help="do not write, only verify criteria")
    args = parser.parse_args(argv)

    data = build(
        docs=args.docs,
        with_bem=not args.no_bem,
        figures=not args.no_figs,
        write=not args.check,
    )
    failed = all_failed(data["verdicts"])
    print(f"chapters computed: {', '.join(sorted(data['payloads']))}")
    for chapter, rows in data["verdicts"].items():
        stats = criteria_module.summary(rows)
        print(f"  {chapter}: pass={stats['passed']} fail={stats['failed']} text={stats['textual']}")
    if failed:
        print("FAILED CRITERIA:")
        for line in failed:
            print(" -", line)
        return 1
    print("all computable criteria pass")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())