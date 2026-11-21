"""Experimental and computed S-175 oblique-wave response curves."""

import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .import_data import ROOT


def draw():
    with (ROOT / "results.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["panels"] == "1372"]
    fig, axes = plt.subplots(3, 3, figsize=(10, 8.1), layout="constrained")
    for axs, (q, headings) in zip(
        axes,
        (("heave", (180, 90, 60)), ("roll", (150, 90, 30)), ("pitch", (180, 90, 30))),
    ):
        for ax, h in zip(axs, headings):
            group = sorted(
                (r for r in rows if r["quantity"] == q and int(r["heading_deg"]) == h),
                key=lambda r: float(r["omega_sqrt_L_g"]),
            )
            x = np.array([float(r["omega_sqrt_L_g"]) for r in group])
            unique, idx = np.unique(x, return_index=True)
            ax.plot(
                unique,
                [float(group[i]["predicted"]) for i in idx],
                color="#195a82",
                label="TensorBEM",
            )
            if q == "roll":
                ax.plot(
                    unique,
                    [float(group[i]["roll_damping_sensitivity"]) for i in idx],
                    "--",
                    color="#bd6c2d",
                    label="Roll damping sensitivity",
                )
            for lab, marker, col in (
                ("IHI", "s", "#7b4776"),
                ("SRI", "^", "#3d453d"),
                ("SHI", "o", "#28765e"),
            ):
                data = [r for r in group if r["laboratory"] == lab]
                if data:
                    ax.scatter(
                        [float(r["omega_sqrt_L_g"]) for r in data],
                        [float(r["observed"]) for r in data],
                        s=20,
                        marker=marker,
                        facecolors="none",
                        edgecolor=col,
                        label=lab,
                        zorder=5,
                    )
            ax.set_title(f"{q.capitalize()}   ·   {h}°", loc="left", fontsize=10)
            ax.set_xlabel("ω √(L/g)", fontsize=9)
            ax.set_ylabel(
                "Motion / wave amplitude" if q == "heave" else "Angle / wave slope",
                fontsize=8,
            )
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.16)
    handles = {}
    for ax in axes.flat:
        for artist, label in zip(*ax.get_legend_handles_labels()):
            handles[label] = artist
    fig.legend(
        handles.values(),
        handles.keys(),
        loc="outside lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("S-175 containership   |   Regular waves at Fn = 0.275")
    fig.savefig(ROOT / "comparison.png", dpi=126)
    plt.close(fig)


if __name__ == "__main__":
    draw()
