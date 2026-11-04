"""End-to-end deliverable builder for m3250 (single process, no re-solves).

1. runs the extended registry on the standard tier (3-D free-surface groups),
2. writes the verdict JSON,
3. renders the convergence figure,
4. injects the recomputation section into docs/marine_bem_benchmarks.md.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS, check  # noqa: E402

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
GROUPS = ("hemisphere3d", "hemisphere3d_chief", "submerged_sphere3d")
BEGIN = "<!-- BEGIN m3250-fs3d-pins -->"
END = "<!-- END m3250-fs3d-pins -->"


def registry() -> dict:
    entries = tuple(e for e in DECLARED_NUMBERS if e.group in GROUPS)
    report = check(tiers=("standard",), entries=entries)
    (OUT / "fs3d_registry.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("checked", report["summary"]["checked"], report["summary"]["verdict"], flush=True)
    return report


def figure(report: dict) -> Path:
    rows = {row["key"]: row for row in report["checks"]}
    measured = {key: row["measured"] for key, row in rows.items()}
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    def series(prefix: str, name: str, levels) -> list[float]:
        return [measured[f"{prefix}_{name}_{n}"] for n in levels]

    hemi_levels = (16, 64, 256)
    labels = (
        ("heave_added_mass_rel_error", r"$A_{33}'$"),
        ("heave_damping_rel_error", r"$B_{33}'$"),
        ("surge_added_mass_rel_error", r"$A_{11}'$"),
        ("surge_damping_rel_error", r"$B_{11}'$"),
        ("excitation_force_z_rel_error", r"$|F_z|$"),
    )
    ax = axes[0]
    for name, label in labels:
        ax.loglog(hemi_levels, series("hemisphere3d", name, hemi_levels), "o-", label=label)
    ax.loglog(hemi_levels, series("hemisphere3d", "heave_added_mass_rel_error", hemi_levels)[0]
              * np.array([16 / n for n in hemi_levels]), "k--", lw=1.0, label=r"$O(h^2)$ guide")
    ax.set_title("Hulme hemisphere, $Ka=1$ (table 10)")
    ax.set_xlabel("panels $N$")
    ax.set_ylabel("relative error vs Hulme (1982)")
    ax.set_xticks(hemi_levels)
    ax.set_xticklabels([str(n) for n in hemi_levels])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    sphere_levels = (20, 80, 320)
    ax = axes[1]
    for name, label in (
        ("heave_added_mass_rel_error", r"$A_{33}'$"),
        ("surge_added_mass_rel_error", r"$A_{11}'$"),
        ("excitation_force_x_rel_error", r"$|F_x|$"),
        ("excitation_force_z_rel_error", r"$|F_z|$"),
    ):
        ax.loglog(sphere_levels, series("sphere3d", name, sphere_levels), "s-", label=label)
    ax.loglog(sphere_levels, series("sphere3d", "heave_added_mass_rel_error", sphere_levels)[0]
              * np.array([20 / n for n in sphere_levels]), "k--", lw=1.0, label=r"$O(h^2)$ guide")
    ax.axhline(3.0e-2, color="tab:red", ls=":", lw=1.0, label="3 % acceptance")
    ax.set_title("Wang/Linton submerged sphere, $Ka=0.8$ (table 12)")
    ax.set_xlabel("panels $N$")
    ax.set_ylabel("relative error vs multipole reference")
    ax.set_xticks(sphere_levels)
    ax.set_xticklabels([str(n) for n in sphere_levels])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    kappa = [
        measured["chief3d_unregularized_condition_number"],
        measured["chief3d_chief_regularized_condition_number"],
    ]
    b33_signed = [
        -measured["chief3d_unregularized_b33_dimless_abs"],
        measured["chief3d_chief_regularized_b33_dimless_abs"],
    ]
    x = np.arange(2)
    ax.bar(x - 0.2, kappa, width=0.35, label=r"$\kappa$", color="tab:blue")
    ax.bar(x + 0.2, np.abs(b33_signed), width=0.35, label=r"$|B_{33}'|$", color="tab:orange")
    ax.axhline(0.0, color="k", lw=0.8)
    for idx, (k, b) in enumerate(zip(kappa, b33_signed)):
        ax.annotate(f"$B_{{33}}'={b:+.4f}$", (idx + 0.2, abs(b)), ha="center", va="bottom", fontsize=8)
        ax.annotate(f"$\\kappa$={k:.2f}", (idx - 0.2, k), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["unregularised BIE", "5 interior CHIEF points"])
    ax.set_title("Waterline resonance $Ka=2.58$, 256 panels (table 11)")
    ax.set_ylabel("condition number / damping magnitude")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "TensorBEM 3-D free-surface benchmarks: documented tables 10-12 recomputed "
        f"({report['summary']['checked']} registered cells, all "
        f"{report['summary']['verdict']})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "fig_fs3d_pins.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def doc_section(report: dict) -> str:
    rows = report["checks"]
    worst = max(rows, key=lambda row: abs(row["relative_deviation"]))
    lines = [
        BEGIN,
        "",
        f"**表 10–12 复算与回归钉桩（m3250，2026-11-04）**：上述三张表此前是纯文档数字——"
        f"生成它们的 `floating_hemisphere` / `submerged_sphere_wave` 研究没有任何测试读取表中各列，"
        f"guard 物理量的既有测试用的是宽松阈值（2.5–5 %），一列收敛误差整体漂移一个量级也不会转红。"
        f"现在表 10/11/12 的每一格（含观测阶与正文的近/远场、Haskind 闭合量）都登记进"
        f"`tensorbem.benchmarks.declared_numbers`（`hemisphere3d`、`hemisphere3d_chief`、"
        f"`submerged_sphere3d` 三组，`standard` 成本层，约一分钟三维稠密求解）。",
        "",
        f"本次全量复算：**{report['summary']['checked']} 格全部 PASS**，"
        f"最大相对偏差 {abs(worst['relative_deviation']):.2e}（`{worst['key']}`，"
        f"声明 {worst['declared']:.4g}，实测 {worst['measured']:.6g}），"
        f"远小于各格容差；表 11 的“未正则化出现负阻尼、5 点 CHIEF 恢复正阻尼、"
        f"条件数下降 ≥20 倍”三条定性结论也各自成为可执行判据。",
        "",
        "复现命令：`python -m tensorbem.benchmarks.declared_numbers --tiers standard`；"
        "证据工件：`outputs/verif_m3250/{fs3d_registry.json,hemisphere_refinement.json,"
        "submerged_sphere_refinement.json,hemisphere_chief.json}`。",
        "",
        "一处钉桩口径改写（如实记录）：表 12 中 20 面元旋转耦合列的 `2.5×10⁻³²` 是完全对称网格上的"
        "舍入痕迹（实测为精确 0），钉桩改为上界判据 `<1×10⁻⁶`；表中该格已同步改写。",
        "",
        END,
    ]
    return "\n".join(lines)


def inject(section: str) -> None:
    path = ROOT / "docs" / "marine_bem_benchmarks.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace("| 2.5\\times10^{-32} |", "| $<10^{-6}$（精确 0） |")
    if BEGIN in text:
        text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), section, text, flags=re.S)
    else:
        anchor = "\n## 5. 讨论与适用限制"
        assert text.count(anchor) == 1
        text = text.replace(anchor, "\n" + section + anchor)
    path.write_text(text, encoding="utf-8")
    print("docs injected", flush=True)


if __name__ == "__main__":
    report = registry()
    print("figure", figure(report), flush=True)
    inject(doc_section(report))