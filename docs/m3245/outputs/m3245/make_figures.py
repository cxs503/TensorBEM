"""Figures for the 3-D free-surface registry rows (m3245).

Left: the documented refinement ladders, straight from the registry run JSON.
Right: the Ka=2.58 irregular-frequency comparison, where the unregularized
boundary integral equation returns a negative heave damping and the 5-point 3-D
CHIEF augmentation restores a positive one.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorbem.analytic import hulme_hemisphere_radiation

DATA = Path("outputs/m3245/free_surface3d_registry.json")
FIG = Path("outputs/m3245/fig_3d_free_surface_registry.png")

report = json.loads(DATA.read_text())
rows = {row["key"]: row for row in report["checks"]}


def ladder(group: str, quantity: str) -> tuple[list[int], list[float]]:
    panels, errors = [], []
    for key, row in rows.items():
        if row["group"] != group:
            continue
        if not key.endswith(quantity):
            continue
        if "standard_guard" in key:
            continue
        count = int(key.split(f"_{quantity}")[0].split("_")[-1])
        panels.append(count)
        errors.append(row["measured"])
    order = np.argsort(panels)
    return [panels[i] for i in order], [errors[i] for i in order]


fig, (ax_err, ax_chief) = plt.subplots(1, 2, figsize=(10.5, 4.0))

hulme_curves = {
    "A33'": "heave_added_mass_rel_error",
    "B33'": "heave_damping_rel_error",
    "A11'": "surge_added_mass_rel_error",
    "B11'": "surge_damping_rel_error",
    "|Fz|": "excitation_force_z_rel_error",
}
for label, quantity in hulme_curves.items():
    panels, errors = ladder("hulme_hemisphere", quantity)
    ax_err.loglog(panels, errors, "o-", label=f"Hulme hemisphere {label}")

sphere_curves = {
    "A33'": "heave_added_mass_rel_error",
    "A11'": "surge_added_mass_rel_error",
    "|Fx|": "excitation_force_x_rel_error",
    "|Fz|": "excitation_force_z_rel_error",
}
for label, quantity in sphere_curves.items():
    panels, errors = ladder("submerged_sphere_wave", quantity)
    ax_err.loglog(panels, errors, "s--", alpha=0.75, label=f"submerged sphere {label}")

reference_panels = np.array([16.0, 320.0])
ax_err.loglog(
    reference_panels,
    1.5e-1 * (reference_panels / 16.0) ** -2.0,
    "k:",
    linewidth=1.0,
    label=r"slope $N^{-2}$",
)
ax_err.set_xlabel("panels N")
ax_err.set_ylabel("relative error vs semi-analytical reference")
ax_err.set_title("3-D free-surface refinement (registry rows)")
ax_err.grid(True, which="both", alpha=0.3)
ax_err.legend(fontsize=7, ncol=2)

ka = 2.58
ref = hulme_hemisphere_radiation(ka, radius=1.0, density=1025.0)
labels = ["A33'", "B33'"]
exact = [ref.heave_added_mass_dimless, ref.heave_damping_dimless]
without = [
    rows["hulme_chief_ka2_58_unregularized_a33"]["measured"],
    -rows["hulme_chief_ka2_58_unregularized_b33_magnitude"]["measured"],
]
with_chief = [
    rows["hulme_chief_ka2_58_chief_a33"]["measured"],
    rows["hulme_chief_ka2_58_chief_b33"]["measured"],
]
x = np.arange(len(labels))
width = 0.26
ax_chief.bar(x - width, exact, width, label="semi-analytical (Hulme 1982)", color="0.35")
ax_chief.bar(x, without, width, label="BIE, no regularization", color="tab:red")
ax_chief.bar(x + width, with_chief, width, label="5-point 3-D CHIEF", color="tab:blue")
ax_chief.axhline(0.0, color="k", linewidth=0.8)
ax_chief.set_xticks(x)
ax_chief.set_xticklabels(labels)
ax_chief.set_ylabel("dimensionless coefficient")
ax_chief.set_title(
    "Interior-waterline resonance Ka=2.58, N=256\n"
    "unregularized B33' = "
    f"{without[1]:+.3f} (passivity violated); CHIEF "
    f"{with_chief[1]:+.3f}"
)
ax_chief.legend(fontsize=7)
ax_chief.grid(True, axis="y", alpha=0.3)

fig.tight_layout()
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=85)
print("wrote", FIG, FIG.stat().st_size, "bytes")