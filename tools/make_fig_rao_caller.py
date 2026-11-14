"""Figure for anchor CASE-BEM-RIGID-BODY-MASS: end-to-end caller replay."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = Path("docs/case_bem_rigid_body_mass")
data = json.loads((ART / "results.json").read_text())
rao = data["rao_caller"]
modes = rao["modes"]
idx = np.arange(len(modes))

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
w = 0.38
axes[0].bar(idx - w / 2, rao["rao_shipped"], w, label="shipped assembly", color="tab:green")
axes[0].bar(idx + w / 2, rao["rao_with_legacy_coupling"], w,
            label="pre-fix coupling injected", color="tab:red")
axes[0].set_yscale("log")
axes[0].set_xticks(idx, modes, rotation=20)
axes[0].set_ylabel("|RAO| (m/m or rad/m)")
axes[0].set_title(f"delivered RAO, omega = {rao['omega_rad_s']:.2f} rad/s, {rao['panels']} panels")
axes[0].grid(True, axis="y", which="both", alpha=0.3)
axes[0].legend(fontsize=8)

change = np.asarray(rao["rao_change_fraction"], dtype=float)
bars = axes[1].bar(idx, np.maximum(change, 1e-16), color="tab:orange")
axes[1].axhline(data["parameters"]["tolerance"], color="k", ls=":", label="3 % anchor tolerance")
axes[1].set_yscale("log")
axes[1].set_xticks(idx, modes, rotation=20)
axes[1].set_ylabel("relative RAO change")
axes[1].set_title(
    f"discriminating power; replay residual "
    f"{rao['replay_rel_residual']:.1e} (< {rao['replay_roundoff_limit']:.0e})"
)
for bar, value in zip(bars, change):
    axes[1].annotate(f"{value*100:.2f}%", (bar.get_x() + bar.get_width() / 2, value),
                     textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
axes[1].grid(True, axis="y", which="both", alpha=0.3)
axes[1].legend(fontsize=8)
fig.suptitle("CASE-BEM-RIGID-BODY-MASS: seakeeping caller replay (tensorbem.rao)")
fig.tight_layout()
out = ART / "fig_rao_caller.png"
fig.savefig(out, dpi=130)
print(out, out.stat().st_size)
