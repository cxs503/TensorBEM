"""Plot the two transfer hulls against every original Table 8 wave run."""

import csv
import json

import matplotlib.pyplot as plt

from .assess import ROOT, LIMITS
from .cross_hull_motions import read_hulls


def plot(root=ROOT):
    trials = read_hulls(root / "transfer_motions.csv")
    with (root / "transfer_coefficients.csv").open() as stream:
        predictions = list(csv.DictReader(stream))
    with (root / "transfer_trials.csv").open() as stream:
        errors = list(csv.DictReader(stream))
    limit = json.loads(LIMITS.read_text())["phase_max_deg"]
    fig, axes = plt.subplots(3, 2, figsize=(9, 8), sharex="col", layout="constrained")
    colors = {"heave": "#14668a", "pitch": "#a74621"}
    for col, hull in enumerate(("I", "IV")):
        tank = trials[trials["hull"] == hull]
        pred = [r for r in predictions if r["hull"] == hull]
        for i, name in enumerate(("heave", "pitch")):
            ax = axes[i, col]
            ax.plot(
                [float(r["lambda_L"]) for r in pred],
                [float(r[name]) for r in pred],
                color=colors[name],
                label="Endpoint prediction",
            )
            ax.scatter(
                tank["lambda_L"],
                tank[name],
                marker="o",
                s=32,
                facecolors="none",
                edgecolors="#18202a",
                label="Tank runs",
                zorder=3,
            )
            ax.set_ylabel(f"{name.capitalize()} amplitude (–)")
            ax.set_ylim(bottom=0)
            pts = [
                r
                for r in errors
                if r["hull"] == hull
                and r["quantity"] == name
                and r["phase_scored"] == "1"
            ]
            axes[2, col].scatter(
                [float(r["lambda_L"]) for r in pts],
                [float(r["wrapped_error_deg"]) for r in pts],
                color=colors[name],
                marker=("o", "x")[i],
                s=25,
                label=name.capitalize(),
                alpha=0.85,
            )
        axes[0, col].set_title(
            f"Wigley {hull} · Fn = 0.30 · {len(tank)} runs", loc="left"
        )
        ax = axes[2, col]
        ax.axhspan(-limit, limit, color="#eaf1ea", zorder=0)
        ax.axhline(0, color="#708080", linewidth=0.7)
        ax.set(
            xlabel="Wavelength / model length",
            ylabel="Phase error (degrees)",
            ylim=(-70, 70),
        )
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[2, 0].legend(frameon=False, fontsize=8, ncol=2)
    for ax in axes.flat:
        ax.grid(alpha=0.18)
    fig.savefig(root / "transfer_curves.png", dpi=145)
    plt.close(fig)


if __name__ == "__main__":
    plot()
