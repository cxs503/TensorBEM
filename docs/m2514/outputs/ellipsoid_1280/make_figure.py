"""Regenerate the ellipsoid-1280 verification figure from the raw run JSONs.

Nothing in the figure is typed by hand: every curve point is read from the
committed raw benchmark outputs (subdivisions_2.json, subdivisions_3.json) or
from the values recorded in the 2026-10-24 expansion audit for the 5120-panel
case (expand_audit_tensorbem.md section 2.3).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

# 5120-panel (subdivisions=4) row as recorded by the 2026-10-24 expansion
# audit (expand_audit_tensorbem.md section 2.3); not re-run here because the
# machine was above the heavy-compute load threshold when this study started.
AUDIT_5120 = {
    "x": 1.0512e-3,
    "y": 1.1275e-3,
    "z": 1.2525e-3,
}

# Values quoted in README.md (Milestone 3) and docs/classical_benchmarks.md
# section 3, table at 1280 panels.
DOCUMENTED_1280 = {
    "x": 4.1679e-3,
    "y": 4.4621e-3,
    "z": 4.9359e-3,
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    sub2 = load(HERE / "subdivisions_2.json")
    sub3 = load(HERE / "subdivisions_3.json")

    def errors(report: dict) -> dict[str, float]:
        return {
            row["motion_axis"]: row["added_mass_relative_error"]
            for row in report["axes"]
        }

    e320 = errors(sub2)
    e1280 = errors(sub3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # Left: convergence family, log-log.
    panels = [320, 1280, 5120]
    for axis, color in zip("xyz", ("tab:blue", "tab:orange", "tab:green")):
        values = [e320[axis], e1280[axis], AUDIT_5120[axis]]
        ax1.loglog(panels, values, "o-", color=color, label=axis, ms=5)
        ax1.loglog(
            [1280, 1280],
            [DOCUMENTED_1280[axis] * 0.8, DOCUMENTED_1280[axis] * 1.25],
            "|", color=color, ms=14, mew=2.5,
        )
    ax1.set_xlabel("panels")
    ax1.set_ylabel("added-mass relative error")
    ax1.set_title("Translating ellipsoid (1.5, 1.0, 0.75)\nerror vs panels; bars = documented 1280 values")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    # Right: measured vs documented at 1280 panels, per axis.
    import numpy as np

    axes_names = ["x", "y", "z"]
    x = np.arange(3)
    width = 0.38
    ax2.bar(x - width / 2, [e1280[a] for a in axes_names], width,
             label="measured (this run)", color="tab:blue")
    ax2.bar(x + width / 2, [DOCUMENTED_1280[a] for a in axes_names], width,
            label="documented", color="tab:gray")
    for i, a in enumerate(axes_names):
        dev = (e1280[a] - DOCUMENTED_1280[a]) / DOCUMENTED_1280[a]
        ax2.annotate(
            f"{dev:+.1e}",
            (x[i] + width / 2, DOCUMENTED_1280[a]),
            textcoords="offset points", xytext=(0, 4),
            ha="center", fontsize=8,
        )
    ax2.set_xticks(x, axes_names)
    ax2.set_ylabel("relative error at 1280 panels")
    ax2.set_title("Measured vs documented (deviation annotated)")
    ax2.legend()

    fig.tight_layout()
    out = HERE / "ellipsoid_1280_verification.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
