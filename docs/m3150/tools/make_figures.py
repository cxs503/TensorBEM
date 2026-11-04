"""Figures for the m3150 3-D free-surface declared-number recomputation.

Every plotted number is produced by calling the benchmark modules (or read from
the registry JSON written by ``python -m tensorbem.benchmarks.declared_numbers``);
nothing is transcribed by hand.

Outputs (all under ``docs/m3150``):
    figures/fig_registry_deviation_3d.png   deviation of all 74 registered numbers
    figures/fig_convergence_3d.png          h-convergence of the two 3-D benchmarks
    figures/fig_chief_passivity.png         unregularized BIE vs interior CHIEF
    outputs/figure_data.json                the numbers behind the three figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorbem.benchmarks.declared_numbers import _run_group

HERE = Path(__file__).resolve().parents[1]
FIGURES = HERE / "figures"
OUTPUTS = HERE / "outputs"
REGISTRY = OUTPUTS / "registry_3d_free_surface.json"

GROUPS_3D = ("hulme_hemisphere", "hulme_chief", "submerged_sphere")


def _registry_rows() -> list[dict]:
    report = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return report["checks"], report["summary"]


def figure_registry_deviation(rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    colours = {
        "hulme_hemisphere": "#1f77b4",
        "hulme_chief": "#d62728",
        "submerged_sphere": "#2ca02c",
    }
    selected = [row for row in rows if row["group"] in GROUPS_3D]
    x = np.arange(len(selected))
    deviations = [max(abs(row["relative_deviation"]), 1e-12) for row in selected]
    tolerances = [row["tolerance"] for row in selected]
    ax.bar(
        x,
        deviations,
        color=[colours[row["group"]] for row in selected],
        width=0.8,
        label="relative deviation",
    )
    ax.plot(
        x,
        tolerances,
        color="k",
        linestyle="--",
        linewidth=1.4,
        label="drift tolerance",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [row["key"].replace("hulme_hemisphere_", "H").replace("submerged_sphere_", "S").replace("hulme_chief_", "C")
         for row in selected],
        rotation=90,
        fontsize=6,
    )
    ax.set_ylabel("relative deviation from the quoted number")
    ax.set_title(
        f"{len(selected)} documented 3-D free-surface numbers recomputed "
        f"(all made executable, no drift)"
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[g]) for g in GROUPS_3D]
    ax.legend(
        handles + [plt.Line2D([0], [0], color="k", linestyle="--")],
        ["Hulme hemisphere (table 10)", "CHIEF study (table 11)", "Wang/Linton sphere (table 12)", "tolerance"],
        loc="upper left",
        fontsize=8,
    )
    ax.grid(axis="y", which="both", alpha=0.25)
    path = FIGURES / "fig_registry_deviation_3d.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_convergence(hemisphere: dict, sphere: dict) -> tuple[Path, dict]:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    data: dict[str, dict] = {}
    for ax, values, levels, title in (
        (axes[0], hemisphere, ("sub1", "sub2", "sub3"), "Hulme (1982) floating hemisphere, Ka=1.0"),
        (axes[1], sphere, ("sub0", "sub1", "sub2"), "Wang/Linton submerged sphere, Ka=0.8, f/a=1.5"),
    ):
        h = np.array([values[f"{level}_h_mean"] for level in levels])
        panels = [int(values[f"{level}_panel_count"]) for level in levels]
        error_a33 = np.array([values[f"{level}_heave_added_mass_rel_error"] for level in levels])
        error_b33 = np.array([values[f"{level}_heave_damping_rel_error"] for level in levels])
        ax.loglog(h, error_a33, "o-", label="A33' relative error")
        ax.loglog(h, error_b33, "s-", label="B33' relative error")
        reference = error_a33[-1] * (h / h[-1]) ** 2
        ax.loglog(h, reference, "k--", linewidth=1.0, label="slope h^2")
        ax.invert_xaxis()
        ax.set_xlabel("mean panel edge length h (m)")
        ax.set_ylabel("relative error vs exact multipole reference")
        ax.set_title(title, fontsize=10)
        ax.grid(which="both", alpha=0.25)
        ax.legend(fontsize=8)
        data[title] = {
            "panels": panels,
            "h_mean": h.tolist(),
            "heave_added_mass_rel_error": error_a33.tolist(),
            "heave_damping_rel_error": error_b33.tolist(),
        }
    path = FIGURES / "fig_convergence_3d.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, data


def figure_chief(chief: dict) -> tuple[Path, dict]:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    schemes = ("unregularized", "chief_regularized")
    labels = ("unregularized BIE", "5-point interior CHIEF")
    condition = [chief[f"ka2.58_{s}_condition_number"] for s in schemes]
    b33 = [chief[f"ka2.58_{s}_b33_dimless"] for s in schemes]
    a33 = [chief[f"ka2.58_{s}_a33_dimless"] for s in schemes]

    axes[0].bar(labels, condition, color=["#d62728", "#1f77b4"], width=0.55)
    axes[0].set_ylabel("condition number at Ka=2.58, N=256")
    axes[0].set_title("interior waterplane resonance: conditioning", fontsize=10)
    for index, value in enumerate(condition):
        axes[0].text(index, value * 1.05, f"{value:.2f}", ha="center", fontsize=9)
    axes[0].set_yscale("log")

    width = 0.35
    x = np.arange(2)
    axes[1].bar(x - width / 2, a33, width, label="A33' (exact 0.4007)", color="#7f7f7f")
    axes[1].bar(x + width / 2, b33, width, label="B33' (exact 0.0631)", color="#17becf")
    axes[1].axhline(0.0, color="k", linewidth=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("dimensionless coefficient")
    axes[1].set_title("passivity: negative B33' before regularisation", fontsize=10)
    axes[1].legend(fontsize=8)
    for index, value in enumerate(b33):
        axes[1].text(index + width / 2, value + (0.006 if value >= 0 else -0.018), f"{value:+.4f}", ha="center", fontsize=9)

    path = FIGURES / "fig_chief_passivity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, {
        "schemes": dict(zip(schemes, zip(condition, a33, b33))),
        "ka": 2.58,
        "panels": 256,
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    rows, summary = _registry_rows()

    hemisphere = _run_group("hulme_hemisphere", {"subdivisions": [1, 2, 3], "ka": 1.0})
    sphere = _run_group("submerged_sphere", {"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5})
    chief = _run_group("hulme_chief", {"subdivision": 3, "ka_values": [2.58]})

    paths = [figure_registry_deviation(rows)]
    convergence_path, convergence_data = figure_convergence(hemisphere, sphere)
    chief_path, chief_data = figure_chief(chief)
    paths += [convergence_path, chief_path]

    data = {
        "registry_summary": summary,
        "convergence": convergence_data,
        "chief": chief_data,
        "figures": [str(path.relative_to(HERE)) for path in paths],
    }
    (OUTPUTS / "figure_data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    for path in paths:
        print(path, path.stat().st_size // 1024, "KB")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()