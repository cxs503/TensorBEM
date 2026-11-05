"""Figures for the m3838 merge report (agent/m3236 = m3236 + main).

Three figures, every number read from files in the tree (no hand typed data):

1. ``fig_registry_deviation.png`` -- fast-tier declared-number registry: measured
   versus declared, |relative deviation| per registered quantity, with the
   drift tolerance as a reference line (baseline comparison).
2. ``fig_dean_convergence.png`` -- Dean (1945) submerged-circular-cylinder
   diffraction, ka = 0.6: panel-count convergence of the discretisation error
   against the exact solution, and the machine-precision invariants.
3. ``fig_merge_union.png`` -- merge accounting: registry entries carried by each
   side of the merge and their union (no key lost on either side).

Run: ``python3 docs/m3838/make_figures.py`` (from the repository root).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
DATA = ROOT / "docs" / "m3236" / "outputs" / "m3236"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def fig_registry_deviation() -> dict:
    report = json.loads((DATA / "registry_fast.json").read_text())
    checks = report["checks"]
    checks.sort(key=lambda c: (c["group"], c["key"]))

    # Guarded rows are upper bounds on a machine-precision invariant: their
    # declared value is the guard, so a "deviation" from it is meaningless.
    guarded = {"dean_cylinder_reflection_64", "dean_cylinder_energy_balance_64",
               "dean_circular_polarisation_64"}

    def magnitude(c):
        if c["key"] in guarded:
            return max(abs(c["measured"]), 1e-18) / c["declared"]
        return abs(c["relative_deviation"])

    keys = [c["key"] for c in checks]
    vals = [magnitude(c) for c in checks]
    tols = [c["tolerance"] for c in checks]
    colors = ["#1f77b4" if c["key"] not in guarded else "#9467bd" for c in checks]

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    y = np.arange(len(keys))
    ax.barh(y, np.maximum(vals, 1e-18), color=colors, height=0.65)
    for yi, tol in zip(y, tols):
        ax.plot([tol, tol], [yi - 0.4, yi + 0.4], color="crimson", lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(keys, fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel("measured/declared magnitude  (-)")
    ax.set_title(
        "Fast-tier declared numbers: recomputation vs documented value\n"
        f"{report['summary']['passed']}/{report['summary']['checked']} PASS "
        "(red bars = declared drift tolerance)", fontsize=9
    )
    ax.invert_yaxis()
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color="#1f77b4", label="value row"),
                       Patch(color="#9467bd", label="guarded invariant (upper bound)"),
                       plt.Line2D([], [], color="crimson", label="tolerance")],
              fontsize=7, loc="lower right")
    fig.savefig(OUT / "fig_registry_deviation.png")
    plt.close(fig)
    return {"checked": report["summary"]["checked"],
            "worst_value_row_deviation": max(
                abs(c["relative_deviation"]) for c in checks if c["key"] not in guarded)}


def fig_dean_convergence() -> dict:
    panels = [32, 64, 128]
    rows = {n: json.loads((DATA / f"dean_{n}.json").read_text())["circular_cylinder"]
            for n in panels}
    series = {
        "panel transmission error $|T_{N}-T_{exact}|$": "panel_transmission_error",
        "panel force $F_x$ rel. error": "panel_force_x_relative_error",
        "MFS force $F_x$ rel. error": "mfs_force_x_relative_error",
        "Haskind vs diffraction $F_x$": "haskind_diffraction_force_x_relative_difference",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax = axes[0]
    for label, key in series.items():
        y = [rows[n][key] for n in panels]
        ax.loglog(panels, y, "o-", label=label, ms=4)
        slope = -np.polyfit(np.log(panels), np.log(y), 1)[0]
        ax.annotate(f"slope {slope:.2f}", (panels[-1], y[-1]), fontsize=6.5,
                    xytext=(4, -8), textcoords="offset points")
    ax.set_xlabel("panels  $N$")
    ax.set_ylabel("relative error  (-)")
    ax.set_title("Dean cylinder, $ka=0.6$: discretisation vs exact solution")
    ax.legend(fontsize=6.5)

    ax = axes[1]
    inv = {
        "panel reflection modulus $|R_N|$": "panel_reflection_modulus",
        "energy balance $||R|^2+|T|^2-1|$": "panel_energy_balance_error",
        "circular polarisation error": "panel_circular_polarisation_error",
        "MFS reflection modulus": "mfs_reflection_modulus",
    }
    for label, key in inv.items():
        y = [max(rows[n][key], 1e-18) for n in panels]
        ax.loglog(panels, y, "s-", label=label, ms=4)
    ax.axhline(1e-12, color="crimson", ls="--", lw=1.0,
               label="guard 1e-12 (exact R=0)")
    ax.set_xlabel("panels  $N$")
    ax.set_ylabel("error  (-)")
    ax.set_title("machine-precision invariants (exact: $R=0$, energy balance = 1)")
    ax.legend(fontsize=6.3)
    fig.savefig(OUT / "fig_dean_convergence.png")
    plt.close(fig)
    return {n: {k: rows[n][k] for k in series.values()} for n in panels}


def _keys(rev: str) -> set:
    text = subprocess.run(
        ["git", "show", f"{rev}:src/tensorbem/benchmarks/declared_numbers.py"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return set(re.findall(r'key="([^"]+)"', text))


def fig_merge_union() -> dict:
    ours = _keys("4f5b73e")          # merge base
    m3236 = _keys("origin/agent/m3236")
    main = _keys("origin/main")
    merged_text = (ROOT / "src" / "tensorbem" / "benchmarks" / "declared_numbers.py").read_text()
    merged = set(re.findall(r'key="([^"]+)"', merged_text))
    added_by_m3236 = m3236 - ours
    added_by_main = main - ours
    stats = {
        "merge_base": len(ours),
        "m3236_side": len(m3236),
        "main_side": len(main),
        "merged": len(merged),
        "added_by_m3236": len(added_by_m3236),
        "added_by_main": len(added_by_main),
        "lost_from_m3236": sorted(m3236 - merged),
        "lost_from_main": sorted(main - merged),
    }
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    labels = ["merge base\n(4f5b73e)", "agent/m3236\n(PR head)", "main\n(ecba798)",
              "merged\nbranch"]
    totals = [len(ours), len(m3236), len(main), len(merged)]
    common = [len(ours), len(ours), len(ours), len(merged - added_by_m3236 - added_by_main)]
    added = [0, len(added_by_m3236), len(added_by_main),
             len(added_by_m3236) + len(added_by_main)]
    x = np.arange(4)
    ax.bar(x, common, color="#1f77b4", label="shared with merge base")
    ax.bar(x, added, bottom=common, color="#ff7f0e", label="added by a side")
    for xi, t in zip(x, totals):
        ax.text(xi, t + 1, str(t), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("registered quantities")
    ax.set_title("merge semantics: union, no registry key lost", fontsize=9)
    ax.legend(fontsize=7)
    fig.savefig(OUT / "fig_merge_union.png")
    plt.close(fig)
    return stats


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    a = fig_registry_deviation()
    b = fig_dean_convergence()
    c = fig_merge_union()
    print(json.dumps({"registry": a, "dean_series": b, "merge": c}, indent=2, default=str))
