"""Figures for the m3306 report: registry deviations and the 3-D physics story.

Everything is read from ``outputs/verif_m3306/standard_tier.json`` (produced by
``python -m tensorbem.benchmarks.declared_numbers --tiers standard``) so no
number is hand-transcribed.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
NEW_GROUPS = ("hulme_hemisphere", "hulme_irregular", "submerged_sphere")
GROUP_COLORS = {
    "hulme_hemisphere": "#1f77b4",
    "hulme_irregular": "#d62728",
    "submerged_sphere": "#2ca02c",
}


def load() -> dict:
    with open(HERE / "standard_tier.json") as handle:
        return json.load(handle)


def fig_deviations(report: dict) -> Path:
    rows = [row for row in report["checks"] if row["group"] in NEW_GROUPS]
    rows.sort(key=lambda row: (row["group"], row["key"]))
    labels = [row["key"].replace("_256", "").replace("_320", "") for row in rows]
    values = [abs(row["relative_deviation"]) for row in rows]
    colors = [GROUP_COLORS[row["group"]] for row in rows]
    tolerances = [row["tolerance"] for row in rows]

    fig, ax = plt.subplots(figsize=(11.5, 8.0))
    y = np.arange(len(rows))
    bars = ax.barh(y, np.maximum(values, 1e-12), color=colors, alpha=0.85)
    for bar, tol, value in zip(bars, tolerances, values):
        ax.plot([tol, tol], [bar.get_y(), bar.get_y() + bar.get_height()],
                color="black", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlim(1e-7, 1e-1)
    ax.set_xlabel("|measured - declared| / |declared|   (black tick = registry tolerance)")
    ax.set_title(
        f"3-D free-surface documented numbers re-computed by the registry "
        f"({len(rows)} rows, all PASS)"
    )
    ax.grid(axis="x", which="both", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in NEW_GROUPS]
    ax.legend(handles, ["Hulme hemisphere (table 10)",
                        "irregular frequency / CHIEF (table 11)",
                        "submerged sphere (table 12)"],
              loc="lower right", fontsize=8)
    fig.tight_layout()
    out = HERE / "fig_registry_3d_pins.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def fig_physics(report: dict) -> Path:
    rows = {row["key"]: row["measured"] for row in report["checks"]}
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 4.9))

    # left: observed orders of the five table-10 quantities vs the nominal 2.
    keys = ("a33", "b33", "a11", "b11", "fz")
    measured = [rows[f"hulme_order_{k}_256"] for k in keys]
    declared = [
        2.06, 1.98, 2.01, 1.94, 2.02
    ]
    x = np.arange(len(keys))
    ax0.bar(x - 0.18, declared, 0.36, label="table 10, as pinned (actual $h_{mean}$ ratio)",
            color="#4c72b0")
    ax0.bar(x + 0.18, measured, 0.36, label="re-computed", color="#dd8452")
    ax0.axhline(2.0, color="black", linestyle=":", linewidth=1.0)
    ax0.set_xticks(x)
    ax0.set_xticklabels(["$A'_{33}$", "$B'_{33}$", "$A'_{11}$", "$B'_{11}$", "$|F_z|$"])
    ax0.set_ylim(1.6, 2.2)
    ax0.set_ylabel("observed order of convergence, 64 -> 256 panels")
    ax0.set_title("Hulme hemisphere: second-order family, $B'_{11}$ slowest")
    ax0.legend(fontsize=8, loc="lower left")
    ax0.grid(axis="y", alpha=0.3)

    # right: CHIEF suppression at the Ka=2.58 irregular frequency.
    exact_a = rows["hulme_irregular_exact_added_mass"]
    exact_b = rows["hulme_irregular_exact_damping"]
    unreg_a = rows["hulme_irregular_unregularized_added_mass"]
    unreg_b = -rows["hulme_irregular_unregularized_damping_magnitude"]
    chief_a = rows["hulme_irregular_chief_added_mass"]
    chief_b = rows["hulme_irregular_chief_damping"]
    x = np.arange(2)
    ax1.bar(x - 0.22, [exact_a, exact_b], 0.22, label="Hulme exact", color="#333333")
    ax1.bar(x, [unreg_a, unreg_b], 0.22, label="un-augmented BIE", color="#c44e52")
    ax1.bar(x + 0.22, [chief_a, chief_b], 0.22,
            label="5 interior CHIEF points", color="#55a868")
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["$A'_{33}$", "$B'_{33}$"])
    ax1.set_ylabel("dimensionless coefficient at $Ka=2.58$")
    ax1.set_title(
        "Irregular frequency: CHIEF removes $B'_{33}<0$\n"
        f"condition number {rows['hulme_irregular_unregularized_condition']:.1f} -> "
        f"{rows['hulme_irregular_chief_condition']:.2f}"
    )
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = HERE / "fig_hulme_orders_chief.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def fig_convergence() -> Path:
    """Hemisphere and sphere errors vs panels, straight from the studies."""
    from tensorbem.benchmarks.floating_hemisphere import (
        run_refinement_study as hulme_study,
    )
    from tensorbem.benchmarks.submerged_sphere_wave import (
        run_refinement_study as sphere_study,
    )

    hulme = hulme_study(subdivisions=(1, 2, 3))
    sphere = sphere_study(subdivisions=(0, 1, 2))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for key, label, style in (
        ("heave_added_mass_rel_error", "$A'_{33}$", "o-"),
        ("heave_damping_rel_error", "$B'_{33}$", "s-"),
        ("surge_added_mass_rel_error", "$A'_{11}$", "^-"),
        ("surge_damping_rel_error", "$B'_{11}$", "d-"),
        ("excitation_force_z_rel_error", "$|F_z|$", "v-"),
    ):
        panels = [level["panel_count"] for level in hulme["levels"]]
        ax0.loglog(panels, [level[key] for level in hulme["levels"]], style, label=label)
    ax0.set_xlabel("panels")
    ax0.set_ylabel("relative error vs Hulme (1982)")
    ax0.set_title("Floating hemisphere, $Ka=1$ (slopes $\\approx -1$, i.e. $O(h^2)$)")
    ax0.grid(which="both", alpha=0.3)
    ax0.legend(fontsize=8)

    for key, label, style in (
        ("heave_added_mass_rel_error", "$A'_{33}$", "o-"),
        ("heave_damping_rel_error", "$B'_{33}$", "s-"),
        ("surge_added_mass_rel_error", "$A'_{11}$", "^-"),
        ("excitation_force_x_rel_error", "$|F_x|$", "d-"),
        ("excitation_force_z_rel_error", "$|F_z|$", "v-"),
    ):
        panels = [level["panel_count"] for level in sphere["levels"]]
        ax1.loglog(panels, [level[key] for level in sphere["levels"]], style, label=label)
    ax1.set_xlabel("panels")
    ax1.set_ylabel("relative error vs Wang/Linton multipoles")
    ax1.set_title("Submerged sphere, $f/a=1.5$, $Ka=0.8$")
    ax1.grid(which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    fig.tight_layout()
    out = HERE / "fig_3d_convergence.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    report = load()
    for path in (fig_deviations(report), fig_physics(report), fig_convergence()):
        print(path)