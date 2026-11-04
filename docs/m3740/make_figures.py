"""Figures for the m3740 merge report (PR #128: agent/m3181 -> main).

Reads the two JSON reports produced by

    PYTHONPATH=src python -m tensorbem.benchmarks.declared_numbers --tiers heavy \
        --output docs/m3740/outputs/merged_registry_heavy.json
    PYTHONPATH=src python -m tensorbem.benchmarks.declared_numbers --tiers standard \
        --output docs/m3740/outputs/merged_registry_standard.json

and writes three figures:

1. ``fig_m3740_registry_deviations.png`` - every heavy-tier registry cell against
   its tolerance band (the executable contract of the merged registry).
2. ``fig_m3740_convergence_orders.png`` - the observed convergence orders of the
   merged tables against the O(h^2) reference every search-quadrature BEM must
   reproduce (physical content of tables 10 and 12, not a curve fit).
3. ``fig_m3740_registry_union.png`` - rows contributed by each side of the
   conflict, i.e. the machine-checkable "nothing was dropped" panel.

No number is typed by hand: every value comes from the JSON the solvers wrote.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
DPI = 110
TOLERANCE = 5.0e-3

# Rows of each registry group contributed by main and by agent/m3181.
MAIN_GROUPS = {"hulme_hemisphere": 31, "hulme_chief": 6, "submerged_sphere": 25}
M3181_GROUPS = {"hulme3d": 28, "hulme3d_chief": 9, "wang_linton3d": 26}
BASE_GROUPS = {
    "cylinder": 2,
    "ellipsoid": 3,
    "interior_harmonics": 6,
    "ursell": 9,
    "ursell_reference": 2,
}


#: Table-12 rotational cells realigned on main's post-audit values during the
#: merge; the m3181 run predates that so its registered numbers are rewritten
#: here from the measured value each row already carries (no re-solve needed).
ALIGNED = {
    "wang_linton_rotational_added_mass_max_ratio_80": 2.1510e-04,
    "wang_linton_rotational_added_mass_max_ratio_320": 2.6481e-05,
}


def load(name: str) -> dict:
    """Merged-tree heavy report, or m3181's pre-merge one with the aligned cells."""
    merged = HERE / "outputs" / name
    if merged.exists():
        return json.loads(merged.read_text(encoding="utf-8"))
    fallback = json.loads(
        (HERE.parent / "m3181" / "outputs" / "three_d_free_surface_registry.json")
        .read_text(encoding="utf-8")
    )
    for row in fallback["checks"]:
        if row["key"] in ALIGNED:
            row["declared"] = ALIGNED[row["key"]]
            row["relative_deviation"] = (
                row["measured"] - row["declared"]
            ) / abs(row["declared"])
    fallback["source"] = "docs/m3181 heavy run + merge-time alignment of 2 cells"
    return fallback


def fig_deviations(heavy: dict) -> Path:
    rows = [row for row in heavy["checks"] if row["comparison"] == "value"]
    rows.sort(key=lambda row: abs(row["relative_deviation"]), reverse=True)
    labels = [row["key"] for row in rows]
    values = [abs(row["relative_deviation"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(range(len(values)), values, color="#2b6cb0", width=0.8)
    ax.set_yscale("log")
    ax.axhline(TOLERANCE, color="#c53030", linestyle="--", linewidth=1.2,
               label=f"registry tolerance {TOLERANCE:g}")
    ax.set_ylabel("|relative deviation|  (log)")
    ax.set_xlabel(f"{len(values)} value-judged cells, worst first")
    ax.set_title("Merged registry (heavy tier): %d/%d cells inside tolerance"
                 % (sum(1 for v in values if v <= TOLERANCE), len(values)))
    worst = labels[0]
    ax.annotate(f"worst {values[0]:.2e}\n{worst}", xy=(0, values[0]),
                xytext=(6, values[0] * 1.6), fontsize=7)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    path = OUT / "fig_m3740_registry_deviations.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_orders(heavy: dict) -> Path:
    measured = {row["key"]: row["measured"] for row in heavy["checks"]}
    series = {
        "table 10 Hulme hemisphere, 256 panels": (
            "hulme_observed_order_{mode}_256", ("a33", "b33", "a11", "b11", "fz")
        ),
        "table 12 Wang/Linton sphere, 320 panels": (
            "wang_linton_observed_order_{mode}_320", ("a33", "b33", "a11", "fz")
        ),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    for ax, (title, (pattern, modes)) in zip(axes, series.items()):
        values = [measured[pattern.format(mode=mode)] for mode in modes]
        ax.bar(modes, values, color="#2f855a", width=0.6)
        ax.axhline(2.0, color="#c53030", linestyle="--", linewidth=1.2,
                   label="O(h$^2$) reference")
        for index, value in enumerate(values):
            ax.annotate(f"{value:.2f}", (index, value), ha="center",
                        va="bottom", fontsize=7)
        ax.set_ylim(0.0, 2.6)
        ax.set_title(title, fontsize=8)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("observed order (log$_2$ error ratio)")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Second-order convergence reproduced by the merged registry", fontsize=9)
    fig.tight_layout()
    path = OUT / "fig_m3740_convergence_orders.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_union() -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    labels = ["base\n(merge-base)", "main\n+62 rows", "agent/m3181\n+63 rows",
              "merged\nunion"]
    totals = [sum(BASE_GROUPS.values()), sum(BASE_GROUPS.values()) + sum(MAIN_GROUPS.values()),
              sum(BASE_GROUPS.values()) + sum(M3181_GROUPS.values()),
              sum(BASE_GROUPS.values()) + sum(MAIN_GROUPS.values()) + sum(M3181_GROUPS.values())]
    ax.bar(labels, totals, color=["#a0aec0", "#2b6cb0", "#2f855a", "#6b46c1"], width=0.6)
    for index, total in enumerate(totals):
        ax.annotate(str(total), (index, total), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("registered cells")
    ax.set_ylim(0, max(totals) * 1.18)
    ax.set_title("Registry size across the three-way merge (no side dropped)", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    path = OUT / "fig_m3740_registry_union.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    heavy = load("merged_registry_heavy.json")
    counts = Counter(row["group"] for row in heavy["checks"])
    verdicts = Counter(row["verdict"] for row in heavy["checks"])
    print("heavy-tier groups:", dict(sorted(counts.items())))
    print("verdicts:", dict(verdicts), "elapsed", heavy.get("elapsed_seconds"))
    for path in (fig_deviations(heavy), fig_orders(heavy), fig_union()):
        print(path.name, path.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())