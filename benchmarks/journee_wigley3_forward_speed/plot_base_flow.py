"""Plot holdout motions and forced-coefficient evidence at Fn = 0.30."""

import matplotlib.pyplot as plt

from .base_flow_study import ROOT, load_coefficients, predictions
from .assess import table
from .speed_parity import parity_motions


def plot():
    curves = predictions(ROOT, load_coefficients(ROOT / "double_body_coefficients.csv"))
    geometry = curves["double_body_both"]
    baseline = curves["uniform"]
    assisted, _ = parity_motions()
    motion = table(ROOT, "motions")
    motion = motion[motion["Fn"] == 0.3]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7))
    for ax, name in zip(axes[0], ("heave", "pitch")):
        ax.scatter(
            motion["lambda_L"],
            motion[name],
            label="Tank",
            marker="o",
            s=29,
            facecolors="none",
            edgecolors="black",
            zorder=4,
        )
        for curve, label, color, style in (
            (baseline, "Uniform current", "#a25638", ":"),
            (geometry, "Double body", "#155e90", "-"),
            (assisted, "Endpoint tank + speed parity", "#29764d", "--"),
        ):
            data = curve[(curve["Fn"] == 0.3) & (curve["panels"] == 760)]
            ax.plot(
                data["lambda_L"], data[name], style, color=color, label=label, lw=1.8
            )
        ax.set(
            xlabel="Wavelength / model length",
            ylabel="Dimensionless amplitude",
            title=name.capitalize(),
        )
    forced = load_coefficients(ROOT / "double_body_forced.csv")
    for ax, code in zip(axes[1], ("A53", "B55")):
        rows = [r for r in forced if r["coefficient"] == code and float(r["Fn"]) == 0.3]
        ax.scatter(
            [float(r["w_nd"]) for r in rows],
            [float(r["measured"]) for r in rows],
            s=29,
            facecolors="none",
            edgecolors="black",
            zorder=4,
        )
        native = {
            float(r["w_nd"]): [float(r["uniform"]), float(r["db760"])] for r in rows
        }
        x = sorted(native)
        ax.plot(x, [native[t][0] for t in x], ":", color="#a25638", lw=1.8)
        ax.plot(x, [native[t][1] for t in x], "-", color="#155e90", lw=1.8)
        ax.set(
            title=code,
            xlabel="Forced frequency  ω / √(g/L)",
            ylabel="Dimensionless coefficient",
        )
    for ax in axes.flat:
        ax.grid(alpha=0.22)
    fig.suptitle("Wigley III at Fn = 0.30")
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(ROOT / "double_body_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot()
