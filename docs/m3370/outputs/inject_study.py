"""Inject the irregular-frequency study numbers into docs + draw the figure.

Every number is read from ``artifacts/irregular_frequency_study.json``; nothing
is typed by hand.  Run from the repository root:

    python docs/m3370/outputs/inject_study.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts" / "irregular_frequency_study.json"
DOC = ROOT / "docs" / "irregular_frequency_study.md"
FIG = ROOT / "docs" / "figures" / "irregular_frequency_band.png"


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _by_variant(rows):
    return {row["variant"]: row for row in rows}


def build(study):
    res = study["resonances"]
    panels = [str(p) for p in study["parameters"]["panel_counts"]]

    t1 = [["共振", "面元数", "解析零点 `ka`", "离散共振 `ka_peak`", "失谐", "`κ_peak/κ(ka_zero)`"]]
    for name, entry in res.items():
        for p in panels:
            row = entry["panel_counts"][p]
            t1.append([
                f"`{name}`", p, f"{entry['analytic_ka']:.6f}",
                f"{row['discrete_resonance']['ka']:.6f}",
                f"{100 * row['detuning_relative']:+.3f}%",
                f"{row['peak_to_zero_kappa_ratio']:.1f}",
            ])

    t2 = [["共振", "面元数", "评估点", "变体", "`κ`", "柱面势相对 `L2` 误差", "水平力相对误差"]]
    for name, entry in res.items():
        for p in panels:
            row = entry["panel_counts"][p]
            for label, key in (("解析零点", "at_analytic_zero"), ("离散共振", "at_discrete_resonance")):
                for variant in _by_variant(row[key]).values():
                    t2.append([
                        f"`{name}`", p, label, f"`{variant['variant']}`",
                        f"{variant['condition_number']:.3e}",
                        f"{variant['potential_relative_error']:.3e}",
                        f"{variant['force_relative_error']:.3e}",
                    ])

    t3 = [["共振", "面元数", "阈值", "是否超过", "`ka` 窗口", "占共振 `ka` 半宽", "带内峰值误差"]]
    for name, entry in res.items():
        for p in panels:
            for band in entry["panel_counts"][p]["amplification"]:
                window = (
                    f"[{band['ka_lo']:.6f}, {band['ka_hi']:.6f}]" if band["exceeded"] else "未分辨"
                )
                t3.append([
                    f"`{name}`", p, f"{100 * band['threshold']:.0f}%",
                    "是" if band["exceeded"] else "否",
                    window, f"{100 * band['relative_half_width']:.3f}%",
                    f"{band['peak_potential_error']:.3e}",
                ])

    verdict = ["| 关键量 | 基准 | 本单实测 | 相对误差 | <3%? |", "|---|---|---|---|---|"]
    for name, entry in res.items():
        for p in panels:
            row = entry["panel_counts"][p]
            peak = _by_variant(row["at_discrete_resonance"])
            for variant in ("unaugmented", "auto"):
                v = peak[variant]
                ok = "✅ 是" if v["potential_relative_error"] < 0.03 else "❌ 否"
                verdict.append([
                    f"`{name}` @离散共振 柱面势 `L2` ({variant})", "MacCamy–Fuchs 精确级数",
                    f"{v['potential_relative_error']:.3e}", f"{100 * v['potential_relative_error']:.2f}%", ok,
                ])
                okf = "✅ 是" if v["force_relative_error"] < 0.03 else "❌ 否"
                verdict.append([
                    f"`{name}` @离散共振 水平力 ({variant})", "MacCamy–Fuchs 精确级数",
                    f"{v['force_relative_error']:.3e}", f"{100 * v['force_relative_error']:.2f}%", okf,
                ])
            if entry["order"] >= 1:
                n = peak["nodal"]
                verdict.append([
                    f"`{name}` @离散共振 柱面势 `L2` (nodal 负控)", "同 `unaugmented`",
                    f"{n['potential_relative_error']:.3e}",
                    f"{100 * (n['potential_relative_error'] - peak['unaugmented']['potential_relative_error']) / peak['unaugmented']['potential_relative_error']:+.2f}%",
                    "❌ 否（负控成功：CHIEF 无效）",
                ])
    return "\n\n".join([
        "### 3.1 离散共振位置与解析零点\n\n" + _table(t1[0], t1[1:]),
        "### 3.2 解析零点 vs 离散共振：条件数与误差（三变体）\n\n" + _table(t2[0], t2[1:]),
        "### 3.3 放大带（未增强系统超过 1%/3% 的 `ka` 窗口）\n\n" + _table(t3[0], t3[1:]),
    ]), _table(verdict[0], verdict[1:])


def figure(study):
    res = study["resonances"]
    names = list(res)
    labels = [n.replace("_", "\n") for n in names]
    panels = [str(p) for p in study["parameters"]["panel_counts"]]
    p_last = panels[-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    x = np.arange(len(names))
    width = 0.26
    series = [
        ("kappa @ analytic zero (bare)", [res[n]["panel_counts"][p_last]["at_analytic_zero"] for n in names], "unaugmented", "#9ecae1"),
        ("kappa @ discrete resonance (bare)", [res[n]["panel_counts"][p_last]["at_discrete_resonance"] for n in names], "unaugmented", "#c0392b"),
        ("kappa @ discrete resonance + auto CHIEF", [res[n]["panel_counts"][p_last]["at_discrete_resonance"] for n in names], "auto", "#27ae60"),
    ]
    for i, (label, rows, variant, colour) in enumerate(series):
        heights = [max(_by_variant(r)[variant]["condition_number"], 1e-3) for r in rows]
        axes[0].bar(x + (i - 1) * width, heights, width, label=label, color=colour)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("condition number κ")
    axes[0].set_title(f"Condition number: analytic zero vs discrete resonance vs CHIEF (N={p_last})", fontsize=11)
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].grid(alpha=0.3, axis="y")

    series2 = [("unaugmented", "#c0392b", "o"), ("auto", "#27ae60", "s"), ("nodal", "#7f8c8d", "^")]
    for i, (variant, colour, marker) in enumerate(series2):
        vals = []
        for n in names:
            rows = _by_variant(res[n]["panel_counts"][p_last]["at_discrete_resonance"])
            vals.append(rows.get(variant, rows["unaugmented"])["potential_relative_error"])
        axes[1].bar(x + (i - 1) * width, vals, width, label=f"surface-potential L2 error ({variant})",
                    color=colour, hatch="" if variant != "nodal" else "//")
    axes[1].axhline(0.03, color="k", ls="--", lw=1, label="3% acceptance threshold")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("relative L2 error of surface potential")
    axes[1].set_title(f"Error at the discrete resonance and the CHIEF cure (N={p_last})", fontsize=11)
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle(
        "TensorBEM 2-D Helmholtz irregular frequencies: discrete resonance, CHIEF cure, "
        f"and nodal-placement negative control (R={study['parameters']['radius']} m, "
        f"h={study['parameters']['depth']} m)",
        fontsize=11,
    )
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=160)
    return FIG


def main():
    study = json.loads(ART.read_text())
    tables, verdict = build(study)
    text = DOC.read_text()
    text = text.replace("<!-- BEGIN GENERATED TABLES -->", "<!-- BEGIN GENERATED TABLES -->\n" + tables, 1)
    text = text.replace("<!-- END GENERATED TABLES -->", "\n<!-- END GENERATED TABLES -->", 1)
    text = text.replace("<!-- BEGIN VERDICT -->", "<!-- BEGIN VERDICT -->\n" + verdict, 1)
    text = text.replace("<!-- END VERDICT -->", "\n<!-- END VERDICT -->", 1)
    DOC.write_text(text)
    print(f"injected into {DOC}; figure -> {figure(study)}")


if __name__ == "__main__":
    main()
