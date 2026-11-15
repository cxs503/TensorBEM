"""Audit the CASE-BEM-CUMMINS-WIGLEY A_inf anchor and publish the artifacts.

Runs entirely off the committed sweeps (no BEM frequency sweep is re-solved):
the image-method A_inf ladder plus the replay of both committed sweeps through
the extended gate set.  Writes ``anchor_audit.json`` next to the case artifact,
plus the report figure.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from tensorbem.benchmarks import case_bem_cummins_wigley as case

OUT = pathlib.Path("benchmarks/verified/casepool_bem_cummins_wigley")
DOCS = pathlib.Path("docs/m21471_cummins_anchor")
DOCS.mkdir(parents=True, exist_ok=True)

LADDER = ((12, 4), (20, 6), (28, 8))

self_check = case.anchor_self_check()
ladder = []
for n_sections, n_levels in LADDER:
    anchor = case.added_mass_inf_anchor(n_sections, n_levels, self_check=self_check)
    ladder.append(
        {
            "n_sections": n_sections,
            "n_levels": n_levels,
            "panel_count": anchor["panel_count"],
            "doubled_panel_count": anchor["doubled_panel_count"],
            "condition_number": anchor["condition_number"],
            "solver_relative_residual_l2": anchor["solver_relative_residual_l2"],
            "added_mass_inf": anchor["added_mass_inf"],
        }
    )

replays = {}
for level in ("coarse", "medium"):
    replay = case.gates_from_artifact(OUT / f"sweep_{level}.npz")
    replays[level] = replay

production = replays["medium"]
rows = {
    "C3": production["gates"]["C3"]["values"]["per_dof"],
    "C3a": production["gates"]["C3a"]["values"]["per_dof"],
    "C7": production["gates"]["C7"]["values"]["per_dof"],
}

convergence = {}
for index, name in enumerate(case.DOF_NAMES):
    values = [row["added_mass_inf"][name] for row in ladder]
    convergence[name] = {
        "values": values,
        "coarse_to_medium_relative_change": abs(values[1] - values[0]) / abs(values[1]),
        "medium_to_fine_relative_change": abs(values[2] - values[1]) / abs(values[2]),
        "richardson_estimate": (
            values[2] + (values[2] - values[1]) / (4.0 - 1.0)
            if abs(values[1]) > 0
            else float("nan")
        ),
    }

payload = {
    "case": case.CASE_SLUG,
    "mission": "m21471",
    "purpose": (
        "Anchor the A_inf that gate C3 fits as a free parameter with the "
        "independent antisymmetric-image solve, and re-judge the KK criterion "
        "with zero free parameters (C3a)."
    ),
    "tolerances": case.TOLERANCES,
    "anchor_self_check": self_check,
    "anchor_ladder": ladder,
    "anchor_convergence": convergence,
    "verdicts": {
        level: {
            "verdict": replay["verdict"],
            "C3": {
                "passed": replay["gates"]["C3"]["passed"],
                "worst_dof": replay["gates"]["C3"]["values"]["worst_dof"],
                "worst_relative_residual": replay["gates"]["C3"]["values"][
                    "worst_relative_residual"
                ],
                "free_parameters": 1,
            },
            "C3a": {
                "passed": replay["gates"]["C3a"]["passed"],
                "worst_dof": replay["gates"]["C3a"]["values"]["worst_dof"],
                "worst_relative_residual": replay["gates"]["C3a"]["values"][
                    "worst_relative_residual"
                ],
                "free_parameters": 0,
            },
            "C7": {
                "passed": replay["gates"]["C7"]["passed"],
                "worst_dof": replay["gates"]["C7"]["values"]["worst_dof"],
                "worst_relative_to_added_mass_scale": replay["gates"]["C7"]["values"][
                    "worst_relative_to_added_mass_scale"
                ],
                "worst_relative_to_anchor": replay["gates"]["C7"]["values"][
                    "worst_relative_to_anchor"
                ],
            },
        }
        for level, replay in replays.items()
    },
    "per_dof_medium": [
        {
            "dof": name,
            "added_mass_inf_fitted": rows["C7"][index]["added_mass_inf_fitted"],
            "added_mass_inf_anchor": rows["C7"][index]["added_mass_inf_anchor"],
            "relative_to_anchor": rows["C7"][index]["relative_to_anchor"],
            "relative_to_added_mass_scale": rows["C7"][index][
                "relative_to_added_mass_scale"
            ],
            "kk_residual_fitted": rows["C3"][index]["max_relative_residual"],
            "kk_residual_anchored": rows["C3a"][index]["max_relative_residual"],
            "truncation_budget_max_relative": rows["C3"][index][
                "truncation_budget_max_relative"
            ],
            "n_decidable": rows["C3"][index]["n_decidable"],
            "n_samples": rows["C3"][index]["n_samples"],
            "worst_sample_omega": rows["C3a"][index]["worst_sample_omega"],
        }
        for index, name in enumerate(case.DOF_NAMES)
    ],
}
(OUT / "anchor_audit.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

# --------------------------------------------------------------------- figures
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
names = list(case.DOF_NAMES)
x = np.arange(len(names))
fitted = [rows["C7"][i]["added_mass_inf_fitted"] for i in range(6)]
anchored = [rows["C7"][i]["added_mass_inf_anchor"] for i in range(6)]
axes[0].bar(x - 0.2, fitted, 0.4, label=r"band closure fit ($C3$)", color="#c44e52")
axes[0].bar(x + 0.2, anchored, 0.4, label="image solve (anchor)", color="#4c72b0")
axes[0].set_yscale("symlog", linthresh=1e-1)
axes[0].set_xticks(x, names, rotation=30)
axes[0].set_ylabel(r"$A_\infty$ (diagonal, case units)")
axes[0].set_title(r"$A_\infty$: fitted free parameter vs physical anchor")
axes[0].legend(fontsize=7)
axes[0].grid(alpha=0.3, axis="y")

anchored_res = [100.0 * rows["C3a"][i]["max_relative_residual"] for i in range(6)]
fitted_res = [100.0 * rows["C3"][i]["max_relative_residual"] for i in range(6)]
axes[1].bar(x - 0.2, fitted_res, 0.4, label=r"$A_\infty$ fitted ($C3$)", color="#dd8452")
axes[1].bar(x + 0.2, anchored_res, 0.4, label=r"$A_\infty$ anchored ($C3a$)", color="#55a868")
axes[1].axhline(100.0 * case.TOLERANCES["kk"], color="crimson", ls="--", lw=1.0,
                label="case criterion 10%")
axes[1].set_xticks(x, names, rotation=30)
axes[1].set_ylabel("max KK residual [% of $\\max|A_{ii}|$]")
axes[1].set_title("Kramers-Kronig closure, free parameter vs anchored")
axes[1].legend(fontsize=7)
axes[1].grid(alpha=0.3, axis="y")

for index, name in enumerate(names):
    row = rows["C3a"][index]
    axes[2].semilogx(row["sample_omega"], 100.0 * np.asarray(row["relative_residual"]),
                     "-o", ms=2.8, lw=1.2, label=name)
axes[2].axhline(100.0 * case.TOLERANCES["kk"], color="crimson", ls="--", lw=1.0)
axes[2].set_xlabel(r"$\omega$ [rad/s]")
axes[2].set_ylabel("KK residual [% of $\\max|A_{ii}|$]")
axes[2].set_title("Anchored closure residual vs frequency (medium mesh)")
axes[2].legend(fontsize=6.5, ncol=2)
axes[2].grid(alpha=0.3, which="both")
fig.suptitle("CASE-BEM-CUMMINS-WIGLEY: independent $A_\\infty$ anchor and zero-free-parameter KK closure")
fig.tight_layout()
fig.savefig(DOCS / "fig_m21471_cummins_anchor.png", dpi=145)
plt.close(fig)
print("wrote", OUT / "anchor_audit.json", DOCS / "fig_m21471_cummins_anchor.png")
print(json.dumps(payload["verdicts"], indent=1))
print(json.dumps(payload["per_dof_medium"], indent=1))