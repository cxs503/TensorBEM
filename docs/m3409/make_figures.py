"""Figures for docs/finite_depth3d_regime.md (m3409)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("docs/m3409/figures")
OUT.mkdir(parents=True, exist_ok=True)
DATA = json.loads(Path("docs/m3409/outputs/finite_depth3d_regime.json").read_text())

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

# (a) long-wave collapse
shallow = [r for r in DATA["shallow_water_limit"]["rows"] if r["r_over_h"] == 5.0]
k0h = np.array([r["k0h"] for r in shallow])
dev = np.array([r["relative_deviation"] for r in shallow])
grid = np.logspace(math.log10(k0h.min()), math.log10(k0h.max()), 100)
axes[0].loglog(grid, grid**2 / 12.0, "--", color="0.5", label=r"$(k_0h)^2/12$ (closed form)")
for ratio, marker in ((5.0, "o"), (10.0, "s")):
    rows = [r for r in DATA["shallow_water_limit"]["rows"] if r["r_over_h"] == ratio]
    axes[0].loglog(
        [r["k0h"] for r in rows],
        [r["relative_deviation"] for r in rows],
        marker,
        ms=5,
        label=rf"$R/h={ratio:g}$ (measured)",
    )
axes[0].axhline(1e-2, color="crimson", lw=0.8, ls=":")
axes[0].text(0.022, 1.3e-2, "1 %", color="crimson", fontsize=8)
axes[0].set_xlabel(r"$k_0 h$")
axes[0].set_ylabel(r"$|G/G_{\rm long\ wave} - 1|$")
axes[0].set_title(r"Long-wave collapse $G\to \frac{i}{4h}H_0^{(1)}(k_0R)$")
axes[0].legend(fontsize=8, loc="lower right")
axes[0].grid(True, which="both", alpha=0.3)

# (b) deep-water collapse, h ladder at fixed nu
deep = DATA["deep_water_limit"]["rows"]
k0h_d = np.array([r["k0h"] for r in deep])
dev_d = np.array([r["relative_deviation"] for r in deep])
axes[1].loglog(k0h_d, dev_d, "o-", color="tab:blue", label="measured (fixed $\\nu=50$)")
axes[1].axvspan(700.0, k0h_d.max() * 1.5, color="tab:orange", alpha=0.15)
axes[1].text(
    900.0,
    1e-2,
    "old code:\n$G=\\mathrm{nan}$\n($\\cosh$ overflow)",
    color="tab:orange",
    fontsize=8,
)
axes[1].axhline(1e-2, color="crimson", lw=0.8, ls=":")
axes[1].set_xlabel(r"$k_0 h$")
axes[1].set_ylabel(r"$|G/G_{\rm deep} - 1|$")
axes[1].set_title("Deep-water collapse (log-scaled wave term)")
axes[1].legend(fontsize=8, loc="lower left")
axes[1].grid(True, which="both", alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / "fig_finite_depth3d_regime.png", dpi=170)
print("wrote", OUT / "fig_finite_depth3d_regime.png")