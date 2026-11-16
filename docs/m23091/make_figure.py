"""Report figure for m23091: which DOF owns each damping minimum.

Reads the two C0 negative controls and the committed coarse sweep, and draws
the per-DOF minimum of the radiation-damping diagonal.  The point of the figure
is the annotation: the note used to print ``-0.000681`` as *pitch*, but that
value belongs to **yaw**, and the real pitch minimum (``-1.463``, the worst
number in the sweep) was published without a DOF next to it.

Usage::

    PYTHONPATH=src python docs/m23091/make_figure.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent
DOFS = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def diagonal(path: pathlib.Path, dof: str) -> np.ndarray:
    """One DOF's radiation-damping diagonal over the band (a 1-D slice)."""
    damping = np.load(path)["damping"]
    return damping[:, DOFS.index(dof), DOFS.index(dof)]


def minima(path: pathlib.Path) -> dict[str, float]:
    return {name: float(diagonal(path, name).min()) for name in DOFS}


def main() -> int:
    n315 = minima(ROOT / "docs/casepool_bem_cummins_wigley/evidence_cluster_n315_sweep_coarse.npz")
    committed = minima(ROOT / "benchmarks/verified/casepool_bem_cummins_wigley/sweep_coarse.npz")
    worst = min(n315, key=n315.get)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), width_ratios=(1.35, 1.0))
    colours = ["#c0392b" if v < 0 else "#7f8c8d" for v in n315.values()]

    ax = axes[0]
    bars = ax.bar(list(n315), list(n315.values()), color=colours)
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_ylabel("min over band of diag(B)  [symlog]")
    ax.set_title(
        "C0 negative control (n315, coarse 12x4): negative radiation damping\n"
        "red = impossible (creating energy) -> the run is not a solution"
    )
    for bar, (name, value) in zip(bars, n315.items()):
        ax.annotate(
            f"{value:.4g}",
            (bar.get_x() + bar.get_width() / 2, value),
            textcoords="offset points",
            xytext=(0, -14 if value < 0 else 6),
            ha="center",
            fontsize=8.5,
        )
    ax.annotate(
        f"worst DOF = {worst} = {n315[worst]:.4g}\n(used to be printed unlabelled;\n"
        f"{n315['yaw']:.3g} was printed as 'pitch')",
        xy=(DOFS.index(worst), n315[worst]),
        xytext=(0.42, 0.16),
        textcoords="axes fraction",
        fontsize=8.6,
        arrowprops=dict(arrowstyle="->", color="#c0392b"),
        color="#c0392b",
    )

    ax = axes[1]
    omega = np.load(ROOT / "docs/casepool_bem_cummins_wigley/evidence_cluster_n315_sweep_coarse.npz")["omega"]
    pitch_n315 = diagonal(ROOT / "docs/casepool_bem_cummins_wigley/evidence_cluster_n315_sweep_coarse.npz", "pitch")
    pitch_ok = diagonal(ROOT / "benchmarks/verified/casepool_bem_cummins_wigley/sweep_coarse.npz", "pitch")
    ax.semilogx(omega, pitch_ok, "o-", ms=3, color="#27ae60", label="committed coarse (C0 pass)")
    ax.semilogx(omega, pitch_n315, "s-", ms=3, color="#c0392b", label="n315 coarse (C0 fail)")
    ax.axhline(0.0, color="black", lw=1.0)
    ax.fill_between(
        omega,
        0.0,
        pitch_n315.min() * 1.05,
        where=pitch_n315 < 0.0,
        color="#c0392b",
        alpha=0.12,
    )
    ax.set_xlabel("ω [rad/s]")
    ax.set_ylabel("B_pitch(ω)")
    ax.set_title(
        "Same code, same mesh, same band: pitch radiation damping\n"
        f"n315 goes negative at {int((pitch_n315 < 0).sum())} of {omega.size} nodes "
        f"(min {pitch_n315.min():.3g})"
    )
    ax.legend(fontsize=8.5, loc="upper left")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.tight_layout()
    png = OUT / "dof_attribution.png"
    fig.savefig(png, dpi=150)
    (OUT / "dof_attribution.json").write_text(
        json.dumps(
            {
                "n315_coarse_12x4": n315,
                "committed_coarse_12x4": committed,
                "n315_worst_dof": worst,
                "n315_worst_value": n315[worst],
                "mislabel": {
                    "published_as_pitch": n315["yaw"],
                    "true_pitch": n315["pitch"],
                    "true_yaw": n315["yaw"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(json.loads((OUT / "dof_attribution.json").read_text()), indent=2))
    print("figure:", png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())