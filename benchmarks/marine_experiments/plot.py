"""Regenerate measured-versus-predicted curves from the committed case outputs."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREEN = "#19628c", "#d07031", "#267f65"


def read(case, name="results"):
    with (ROOT / case / (name + ".csv")).open() as stream:
        return list(csv.DictReader(stream))


def values(rows, key):
    return np.array([float(row[key]) for row in rows])


def prediction(ax, rows, key, label, color, style):
    # Experiments at multiple amplitudes share a single linear model prediction.
    x = values(rows, "lambda_L")
    x, idx = np.unique(x, return_index=True)
    ax.plot(x, values(rows, key)[idx], style, color=color, label=label, lw=1.6)


def hull_panel(ax, rows, ylabel, reflected_key=None):
    prediction(ax, rows, "predicted", "Model", BLUE, "-")
    if reflected_key:
        prediction(ax, rows, reflected_key, "Model + bow reflection", ORANGE, "--")
    ax.scatter(
        values(rows, "lambda_L"),
        values(rows, "observed"),
        s=22,
        facecolor="white",
        edgecolor="#182f3a",
        lw=1.0,
        label="Experiment",
        zorder=5,
    )
    ax.axhline(0, color="#8c9393", lw=0.55)
    ax.set(xlabel="Wavelength / ship length", ylabel=ylabel)


def render_wigley():
    case = "journee_added_resistance"
    rows = [r for r in read(case) if r["panels"] == "480"]
    groups = sorted(
        {(r["hull"], float(r["Fn"])) for r in rows},
        key=lambda item: (item[1] == 0, item[0], item[1]),
    )
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.6), layout="constrained")
    for ax, (hull, fn) in zip(axes.flat, groups):
        group = [r for r in rows if r["hull"] == hull and float(r["Fn"]) == fn]
        hull_panel(ax, group, "Added resistance coefficient", "with_reflection")
        ax.set_title(f"Wigley {hull}   ·   Fn = {fn:.2f}", loc="left")
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(
        *axes.flat[0].get_legend_handles_labels(), loc="center", frameon=False
    )
    fig.suptitle(
        "Regular head waves   |   Journée towing-tank measurements", fontsize=16
    )
    fig.savefig(ROOT / case / "comparison.png", dpi=124)
    plt.close(fig)


def render_kvlcc2():
    case = "kvlcc2_head_seas"
    rows = [r for r in read(case) if r["panels"] == "1118"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained")
    for ax, (key, label) in zip(
        axes,
        [
            ("heave", "Heave / amplitude"),
            ("pitch", "Pitch / wave slope"),
            ("resistance", "Added resistance coefficient"),
        ],
    ):
        group = [r for r in rows if r["quantity"] == key]
        hull_panel(ax, group, label)
        if key == "resistance":
            prediction(
                ax,
                [r for r in rows if r["quantity"] == "resistance_plus_reflection"],
                "predicted",
                "Model + bow reflection",
                ORANGE,
                "--",
            )
        ax.set_title(key.capitalize(), loc="left")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("KVLCC2   |   Head seas at 15.5 kn full scale", fontsize=15)
    fig.savefig(ROOT / case / "comparison.png", dpi=125)
    plt.close(fig)


def render_floe():
    case = "floe_transmission"
    rows = [r for r in read(case) if r["panels"] == "72"]
    fig, axes = plt.subplots(
        4, 3, figsize=(10, 9.3), sharex=True, sharey=True, layout="constrained"
    )
    for row, (material, quantity) in enumerate(
        (("PP", "T"), ("PP", "R"), ("PVC", "T"), ("PVC", "R"))
    ):
        for col, period in enumerate((0.8, 0.9, 1.0)):
            ax = axes[row, col]
            group = [
                r
                for r in rows
                if r["material"] == material and float(r["period"]) == period
            ]
            key = "observed_energy_" + quantity
            for config, marker, color, label in (
                ("free", "o", BLUE, "Free"),
                ("moored", "+", ORANGE, "Moored"),
                ("barriers", "s", GREEN, "Moored + barriers"),
            ):
                data = [r for r in group if r["configuration"] == config]
                ax.plot(
                    values(data, "ka" if quantity == "T" else "ka_R"),
                    values(data, key),
                    marker=marker,
                    mfc="none",
                    color=color,
                    linestyle="none",
                    ms=5,
                    label=label,
                )
            ax.axhline(
                float(group[0]["predicted_energy_" + quantity]),
                color="#25343b",
                lw=1.3,
                label="Linear plate",
            )
            ax.set_ylim(-0.06, 1.12)
            if col == 0:
                ax.set_ylabel(f"{material}   ·   energy {quantity}")
            if row == 0:
                ax.set_title(f"Period {period:.1f} s", loc="left")
            if row == 3:
                ax.set_xlabel("Incident wave steepness ka")
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Floating plates   |   Transmitted and reflected wave energy", fontsize=15
    )
    fig.savefig(ROOT / case / "comparison.png", dpi=124)
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.15,
        }
    )
    render_wigley()
    render_kvlcc2()
    render_floe()
