"""Figures for the round-2 survivor report (docs/m2856).

Both panels are programmatic: nothing is hand-copied.

* ``fig_round2_gradients.png`` -- the kernel site.  Left: the analytic vertical
  source derivative against a central difference of the public Green function
  along a sweep of horizontal offset, with the depth path isolated.  Right: the
  absolute error of the analytic derivative relative to the finite difference.
* ``fig_round2_mutants.png`` -- the verdicts of
  ``docs/m2856/tools/verify_round2_mutants.py`` read straight out of
  ``docs/m2856/outputs/round2_mutants.json``: survived-before / killed-after per
  audit survivor.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorbem.free_surface import green_function, green_gradient

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def gradient_figure() -> Path:
    nu = 0.6
    field_z, source_z = -0.9, -1.1
    source_x = -0.4
    step = 1.0e-6
    offsets = np.linspace(-2.0, 2.0, 81)

    analytic = np.array(
        [
            complex(green_gradient(source_x + d, field_z, source_x, source_z, nu)[1])
            for d in offsets
        ]
    )
    finite = np.array(
        [
            complex(
                green_function(source_x + d, field_z, source_x, source_z + step, nu)
                - green_function(source_x + d, field_z, source_x, source_z - step, nu)
            )
            / (2.0 * step)
            for d in offsets
        ]
    )
    error = np.abs(analytic - finite) / np.maximum(np.abs(finite), 1e-12)

    figure, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(9.0, 3.6), constrained_layout=True
    )
    ax_left.plot(offsets, finite.real, "o", ms=3.0, label="central difference", alpha=0.75)
    ax_left.plot(offsets, analytic.real, "-", lw=1.4, label="analytic (green_gradient)")
    ax_left.set_xlabel("horizontal offset $X = x - x_0$ (m)")
    ax_left.set_ylabel(r"$\mathrm{Re}\,\partial G/\partial z_0$ (1/m)")
    ax_left.set_title(r"$\nu=0.6$, $z=-0.9$, $z_0=-1.1$")
    ax_left.legend(fontsize=8)
    ax_left.grid(alpha=0.3)

    ax_right.semilogy(offsets, error, "-", lw=1.2, color="tab:red")
    ax_right.axhline(1e-6, ls="--", lw=1.0, color="0.4", label="FD truncation floor")
    ax_right.set_xlabel("horizontal offset $X$ (m)")
    ax_right.set_ylabel("relative deviation")
    ax_right.set_title("analytic vs central difference (step $10^{-6}$ m)")
    ax_right.legend(fontsize=8)
    ax_right.grid(alpha=0.3)

    path = OUT / "fig_round2_gradients.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path, float(error.max()), float(error.min())


def verdict_figure() -> tuple[Path, dict]:
    payload = json.loads((OUT / "round2_mutants.json").read_text(encoding="utf-8"))
    rows = payload["rows"]
    labels = [row["key"].replace("tensorbem.", "") for row in rows]
    before = [0 if row["before"]["verdict"] == "survived" else 1 for row in rows]
    after = [0 if row["after"]["verdict"] == "survived" else 1 for row in rows]

    figure, axis = plt.subplots(figsize=(9.0, 3.6), constrained_layout=True)
    positions = np.arange(len(rows))
    axis.barh(positions - 0.18, [1 - v for v in before], height=0.34,
              color="0.6", label="killed BEFORE (pre-existing suite)")
    axis.barh(positions + 0.18, [1 - v for v in after], height=0.34,
              color="tab:blue", label="killed AFTER (round-2 guardrails)")
    for index, (b, a) in enumerate(zip(before, after)):
        axis.text(0.02, index - 0.18, "survived" if not b else "killed",
                  va="center", fontsize=8)
        axis.text(0.02, index + 0.18, "survived" if not a else "killed",
                  va="center", fontsize=8, color="white" if a else "black")
    axis.set_yticks(positions, labels, fontsize=8)
    axis.set_xlim(0, 1.25)
    axis.set_xticks([])
    axis.set_xlabel("verdict per audit survivor")
    axis.set_title(
        f"round-2 closure: {payload['closed']}/{payload['total']} survivors now killed"
    )
    axis.legend(fontsize=8, loc="lower right")
    axis.invert_yaxis()
    path = OUT / "fig_round2_mutants.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path, payload


def main() -> None:
    path, worst, best = gradient_figure()
    print(f"{path.name}: max rel deviation {worst:.3e}, min {best:.3e}")
    path2, payload = verdict_figure()
    print(f"{path2.name}: closed {payload['closed']}/{payload['total']}")
    for item in (path, path2):
        print(f"{item.name} {item.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()