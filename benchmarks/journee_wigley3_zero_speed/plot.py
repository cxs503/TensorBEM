"""Render amplitude and signed phase on the finest grid against all measurements."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def read_table(name):
    return np.genfromtxt(HERE / (name + ".csv"), delimiter=",", names=True)


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    measured = read_table("table10")
    for column, dof in enumerate(("heave", "pitch")):
        for row, field in enumerate((dof, dof + "_deg")):
            ax = axes[row, column]

            def display(value):
                # Continuous branch around -90 degrees for pitch.
                return (
                    (value + 90) % 360 - 90
                    if row and dof == "heave"
                    else ((value + 180) % 360 - 180 if row else value)
                )

            ax.scatter(
                measured["lambda_L"],
                display(measured[field]),
                marker="o",
                facecolor="none",
                edgecolor="black",
                label="Journée (13 runs)",
                zorder=3,
            )
            for solver, label, style in [
                ("tensorbem", "TensorBEM", "-"),
                ("capytaine", "Capytaine 3.0", "--"),
            ]:
                records = read_table(solver)
                curve = records[records["panels"] == 760]
                ax.plot(curve["lambda_L"], display(curve[field]), style, label=label)
            ax.grid(alpha=0.23)
            if row:
                ax.set_xlabel("Wavelength / model length")
                ax.set_ylabel("Phase (degrees)")
            else:
                ax.set_title(dof.title())
                ax.set_ylabel("Dimensionless amplitude")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "comparison.png", dpi=145)
