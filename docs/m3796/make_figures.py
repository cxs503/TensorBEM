#!/usr/bin/env python3
"""Figures for the m3236 -> main union-merge report (PR #134 / mission 3796).

Three figures, all driven by files produced during the merge, none hand-typed:

fig1_registry_union.png   declared-number registry inventory per benchmark group,
                          split by the side that contributed the group
                          (read from the two parent blobs + the merged blob).
fig2_test_parity.png      pytest outcome counts, main baseline vs merged tree
                          (parsed from the two xdist logs).
fig3_dean_pinned.png      the eleven Dean table-7 rows that m3236 pinned:
                          relative deviation |measured-declared| of each row in
                          outputs/m3236/registry_fast.json, with the 3% reference
                          line drawn for the error-judgement section.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE_SRC = Path("/tmp/m3796_base_2251434/src/tensorbem/benchmarks/declared_numbers.py")
MERGED_SRC = Path("src/tensorbem/benchmarks/declared_numbers.py")
FAST_JSON = Path("outputs/m3236/registry_fast.json")
OUT = Path("docs/m3796/figures")


def registry_groups(path: Path) -> dict[str, str]:
    """key -> group for every DeclaredNumber(...) record in the module."""
    tree = ast.parse(path.read_text())
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", getattr(func, "attr", ""))
        if name != "DeclaredNumber":
            continue
        key = group = None
        for kw in node.keywords:
            if kw.arg == "key" and isinstance(kw.value, ast.Constant):
                key = kw.value.value
            elif kw.arg == "group" and isinstance(kw.value, ast.Constant):
                group = kw.value.value
        if key is not None:
            out[key] = group or "?"
    return out


def parse_pytest(log: Path) -> dict[str, int]:
    text = log.read_text(errors="replace")
    counts: dict[str, int] = {}
    for label in ("passed", "failed", "skipped", "error", "errors", "xfailed", "xpassed"):
        m = re.findall(rf"(\d+) {label}\b", text)
        if m:
            counts[label] = int(m[-1])
    counts["collected"] = len(re.findall(r"^tests/", text, re.MULTILINE)) or counts.get("passed", 0)
    return counts


def fig_registry_union() -> None:
    base = registry_groups(BASE_SRC)
    merged = registry_groups(MERGED_SRC)
    groups = sorted({g for g in merged.values()})
    main_only = [k for k in merged if k in base]
    new_keys = [k for k in merged if k not in base]
    counts_main = {g: sum(1 for k in main_only if merged[k] == g) for g in groups}
    counts_new = {g: sum(1 for k in new_keys if merged[k] == g) for g in groups}

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    xs = range(len(groups))
    ax.bar(xs, [counts_main[g] for g in groups], label=f"present on main (n={len(main_only)})",
           color="#3b6fa0")
    ax.bar(xs, [counts_new[g] for g in groups], bottom=[counts_main[g] for g in groups],
           label=f"added by agent/m3236 (n={len(new_keys)})", color="#c0504d")
    for i, g in enumerate(groups):
        total = counts_main[g] + counts_new[g]
        ax.text(i, total + 0.4, str(total), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel("DeclaredNumber records in the merged registry")
    ax.set_title("Union merge keeps both sides: registry inventory per benchmark group")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_registry_union.png", dpi=150)
    plt.close(fig)
    print("fig1:", len(main_only), "from main,", len(new_keys), "from m3236,", len(merged), "total")


def fig_test_parity() -> None:
    base = parse_pytest(Path("/tmp/m3796_base_xdist.log"))
    merged = parse_pytest(Path("/tmp/m3796_merged_xdist.log"))
    labels = ["passed", "failed", "skipped", "error"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    width = 0.38
    for offset, (name, data, colour) in enumerate(
        (("main baseline", base, "#3b6fa0"), ("merged (m3236+main)", merged, "#c0504d"))
    ):
        vals = [data.get(k, 0) for k in labels]
        xs = [i + (offset - 0.5) * width for i in range(len(labels))]
        bars = ax.bar(xs, vals, width=width, label=name, color=colour)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("tests")
    ax.set_title("Full suite parity: merged tree vs main baseline (same settings)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_test_parity.png", dpi=150)
    plt.close(fig)
    print("fig2:", base, merged)


def fig_dean_pinned() -> None:
    """Value rows: |measured-declared|/|declared|.  Upper-bound rows: measured/bound
    (must stay below 1 by construction; the declared bound is a guard, not a target,
    so a relative deviation there is meaningless -- it always approaches -100%)."""
    data = json.loads(FAST_JSON.read_text())
    rows = [c for c in data["checks"] if c.get("group") == "dean"]
    if not rows:
        raise SystemExit("no dean rows in the pinned artifact")
    rows.sort(key=lambda c: (c.get("comparison") != "value", c["key"]))
    names, heights, colours, kinds = [], [], [], []
    for r in rows:
        name = r["key"].replace("dean_", "").replace("_64", "")
        if r.get("comparison") == "upper":
            heights.append(r["measured"] / r["declared"])
            colours.append("#7f7f7f")
            kinds.append("upper")
            name += " (upper)"
        else:
            heights.append(abs(r["relative_deviation"]))
            colours.append("#3b6fa0")
            kinds.append("value")
        names.append(name)

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    xs = list(range(len(rows)))
    bars = ax.bar(xs, [max(h, 1e-19) for h in heights], color=colours)
    ax.axhline(1.0, color="#7f7f7f", ls="--", lw=1,
               label="upper-bound rows: acceptance is ratio < 1")
    ax.axhline(0.03, color="k", ls=":", lw=1, label="3% acceptance reference (value rows)")
    ax.axhline(0.015, color="r", ls="-.", lw=1, label="pinned value-row tolerance (1.5%)")
    for b, h, kind in zip(bars, heights, kinds):
        ax.text(b.get_x() + b.get_width() / 2, max(h, 1e-19) * 1.6,
                f"{h:.1e}", ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(1e-19, 10)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("dimensionless")
    ax.set_title("Dean table-7 rows pinned by m3236: reproduced deviation (blue) and\n"
                 "upper-bound guard ratio (grey), N=64 panels, ka=0.6")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_dean_pinned.png", dpi=150)
    plt.close(fig)
    worst_value = max(h for h, k in zip(heights, kinds) if k == "value")
    worst_upper = max(h for h, k in zip(heights, kinds) if k == "upper")
    print(f"fig3: {len(rows)} rows, worst value dev {worst_value:.3e}, "
          f"worst upper-bound ratio {worst_upper:.3e}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_registry_union()
    if Path("/tmp/m3796_merged_xdist.log").exists():
        fig_test_parity()
    fig_dean_pinned()


if __name__ == "__main__":
    main()