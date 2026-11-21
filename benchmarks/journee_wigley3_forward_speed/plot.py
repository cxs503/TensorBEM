"""Plot every towing-tank run and both finest-mesh solutions."""

import matplotlib.pyplot as plt
import numpy as np

from .assess import ROOT, table


def draw():
    motions = table(ROOT, "motions")
    fig, axes = plt.subplots(4, 3, figsize=(11, 9), sharex=True, sharey="row")
    for col, fn in enumerate(np.unique(motions["Fn"])):
        measured = motions[motions["Fn"] == fn]
        axes[0, col].set_title(f"Fn = {fn:.2f}   ·   {len(measured)} tank runs")
        for row, field in enumerate(("heave", "pitch", "heave_deg", "pitch_deg")):
            ax = axes[row, col]

            def display(values):
                if field == "heave_deg":
                    return (values + 180) % 360 - 180
                return values % 360 if field == "pitch_deg" else values

            ax.scatter(
                measured["lambda_L"],
                display(measured[field]),
                label="Journée tank",
                s=22,
                facecolors="none",
                edgecolors="#171717",
                linewidths=0.9,
                zorder=3,
            )
            for source, label, style, color in [
                ("tensorbem", "TensorBEM", "-", "#085c86"),
                ("capytaine", "Capytaine 3.0", "--", "#ca5523"),
            ]:
                records = table(ROOT, source)
                curve = records[
                    (records["Fn"] == fn)
                    & (records["panels"] == max(records["panels"]))
                ]
                x, y = curve["lambda_L"], display(curve[field])
                if field.endswith("_deg"):
                    jumps = np.flatnonzero(abs(np.diff(y)) > 180) + 1
                    x, y = np.insert(x, jumps, np.nan), np.insert(y, jumps, np.nan)
                ax.plot(x, y, style, color=color, lw=1.5, label=label)
            ax.grid(alpha=0.23)
            if col == 0:
                ax.set_ylabel(
                    (
                        "Heave amplitude  |z|/a",
                        "Pitch amplitude  |θ|L/(2πa)",
                        "Heave phase (deg)",
                        "Pitch phase (deg)",
                    )[row]
                )
            if row == 3:
                ax.set_xlabel("Wavelength / model length")
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Wigley III in head seas at forward speed", y=0.991, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(ROOT / "comparison.png", dpi=145)
    plt.close(fig)


if __name__ == "__main__":
    draw()
