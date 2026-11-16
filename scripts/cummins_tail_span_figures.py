"""Figure: C9 带顶斜率 vs 拟合跨度（数据全部来自 tail_span_sensitivity.json）。

Each panel is one committed artifact; each line is one DOF's decay exponent ``p`` as
the fit span grows from 3 samples to the whole tail band.  The dashed line is the
algebraic-decay bar ``2.05``: a curve that crosses it means the C9 verdict flips
inside a single artifact, i.e. the verdict belongs to the estimator, not the data.
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

WORKTREE = pathlib.Path(__file__).resolve().parent.parent
CASE_DIR = WORKTREE / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
SOURCE = CASE_DIR / "tail_span_sensitivity.json"
OUTPUT = WORKTREE / "outputs" / "m23070" / "tail_span_sensitivity.png"

TITLES = {
    "sweep_extended_probe_coarse.npz": "probe coarse: 192 panels, 12-20 rad/s",
    "sweep_extended_probe_medium.npz": "probe medium: 480 panels, 12-20 rad/s",
    "sweep_medium.npz": "committed: 480 panels, 0.2-12 rad/s",
}


def main() -> int:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    blocks = payload["artifacts"]
    figure, axes = plt.subplots(1, len(blocks), figsize=(5.0 * len(blocks), 4.2), sharey=True)
    if len(blocks) == 1:
        axes = [axes]
    for axis, block in zip(axes, blocks):
        for row in block["per_dof"]:
            spans, values = [], []
            for key, fit in row["slope_by_span"].items():
                exponent = fit["decay_exponent"]
                if exponent is None or not np.isfinite(exponent):
                    continue
                spans.append(fit["decades"])
                values.append(exponent)
            if not spans:
                continue
            axis.plot(spans, values, marker="o", markersize=3.5, label=row["dof_name"])
        axis.axhline(block["slope_bar"], color="0.3", linestyle="--", linewidth=1.0)
        axis.set_title(TITLES.get(block["artifact"], block["artifact"]), fontsize=9)
        axis.set_xlabel("fit span [decades]")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("band-top decay exponent $p$  ($B \\propto \\omega^{-p}$)")
    axes[-1].annotate(
        f"bar $p={blocks[0]['slope_bar']}$", xy=(0.02, blocks[0]["slope_bar"]),
        xycoords=("axes fraction", "data"), fontsize=8, color="0.3", va="bottom",
    )
    axes[0].legend(fontsize=7, ncol=2)
    figure.suptitle(
        "C9 band-top slope is not identifiable: the verdict flips with the fit span",
        fontsize=10,
    )
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=150)
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())