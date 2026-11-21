"""Plot the withheld Fn = 0.30 free-motion trials and the three prediction routes."""

import csv

import matplotlib.pyplot as plt
import numpy as np

from .assess import ROOT, table


def plot():
    with (ROOT / "candidate_motions.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if float(r["Fn"]) == 0.30]
    obs = table(ROOT, "motions")
    obs = obs[obs["Fn"] == 0.30]
    fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for j, dof in enumerate(("heave", "pitch")):
        for i, field in enumerate((dof, dof + "_deg")):
            ax = axs[i, j]

            def display(y):
                return (y + 180) % 360 - 180 if i and dof == "heave" else y

            ax.scatter(
                obs["lambda_L"],
                display(obs[field]),
                facecolors="none",
                edgecolors="black",
                label="Tank (20 runs)",
                zorder=5,
                s=30,
            )
            for name, label, color, style in (
                ("baseline", "3-D uniform current", "#a04b20", ":"),
                ("strip_20x32", "2-D sections + speed terms", "#376598", "--"),
                (
                    "endpoint_radiation_load",
                    "Endpoint tank calibration",
                    "#16754b",
                    "-",
                ),
            ):
                curve = [r for r in rows if r["model"] == name]
                x = np.array([float(r["lambda_L"]) for r in curve])
                y = display(np.array([float(r[field]) for r in curve]))
                if i:
                    cut = np.flatnonzero(np.abs(np.diff(y)) > 180) + 1
                    x, y = np.insert(x, cut, np.nan), np.insert(y, cut, np.nan)
                ax.plot(x, y, style, color=color, label=label, lw=1.8)
            ax.grid(alpha=0.2)
            if i:
                ax.set_xlabel("Wavelength / model length")
                ax.set_ylabel("Phase (degrees)")
            else:
                ax.set_title(dof.capitalize())
                ax.set_ylabel("Dimensionless amplitude")
    fig.suptitle("Wigley III  ·  withheld speed Fn = 0.30")
    fig.legend(
        *axs[0, 0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(ROOT / "holdout.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot()
