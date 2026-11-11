"""Render the MacCamy-Fuchs anchor figures from the committed record.

The report ``benchmarks/verified/bem_maccamy_fuchs/report.md`` links a
three-panel figure ``figures/maccamy_fuchs_convergence.png``.  A committed
``*.png`` under ``benchmarks/`` is **not** a durable invariant of this
repository: the platform push gate drops ``*.png`` outside ``docs/`` (see
``docs/dev_report_m13806_push_gate_figure_integrity.md``), so the linked image
never reaches ``main``.  What *is* durable is that the figure can be re-rendered
from the committed numbers alone.  This module is that renderer: it takes the
parsed ``results.json`` of the anchor and writes the three panels, so both the
report build and the acceptance test work from the same code path (no hand
drawing, no hand-typed values).

Panels (as described in §4 of the report):

* (a) ``h``-convergence of the horizontal force error with an ``O(h^2)``
  reference slope anchored on the finest level;
* (b) effective inertia coefficient ``C_M(ka)`` computed vs exact, from the
  Morison long-wave limit to the short-wave regime;
* (c) irregular-frequency suppression: condition number without and with the
  CHIEF interior null-field augmentation at the ``J_0`` and ``J_1`` resonances.
"""

from __future__ import annotations

import pathlib
from typing import Any

FIGURE_NAME = "maccamy_fuchs_convergence.png"


def render_convergence_figure(
    results: dict[str, Any],
    out_dir: pathlib.Path | str,
    name: str = FIGURE_NAME,
) -> pathlib.Path:
    """Write the three-panel figure into ``out_dir`` and return its path.

    ``results`` is the parsed content of
    ``benchmarks/verified/bem_maccamy_fuchs/results.json``; every plotted number
    is read from it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    levels = results["refinement"]["levels"]
    h = [float(level["h_max"]) for level in levels]
    err = [abs(float(level["force_x_relative_error"])) for level in levels]
    potential_err = [abs(float(level["surface_potential_relative_l2_error"])) for level in levels]

    sweep = results["frequency_sweep"]
    ka = [float(row["ka"]) for row in sweep]
    cm_computed = [float(row["computed_effective_cm"]) for row in sweep]
    cm_exact = [float(row["exact_effective_cm"]) for row in sweep]

    irregular = results["irregular_frequency_study"]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), dpi=150)

    # ---- (a) h-convergence with an O(h^2) reference ---------------------
    orders = results["refinement"].get("observed_orders") or {}
    force_orders = [value for value in orders.get("force_x_relative_error", []) or []
                    if value is not None]
    order = float(force_orders[-1]) if force_orders else 2.0
    reference = [err[-1] * (hi / h[-1]) ** 2 for hi in h]
    axes[0].loglog(h, err, "o-", label="horizontal force, $|\\Delta F|/|F|$")
    axes[0].loglog(h, potential_err, "s-", label="surface potential, $L_2$")
    axes[0].loglog(h, reference, "k--", lw=1, label="reference slope $h^{2}$")
    axes[0].set_xlabel("$h_{max}$  [m]")
    axes[0].set_ylabel("relative error  [-]")
    axes[0].set_title(f"(a) h-convergence (last order {order:.3f})")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    # ---- (b) effective inertia coefficient vs ka ------------------------
    axes[1].semilogx(ka, cm_computed, "o-", label="$C_M$ (BEM)")
    axes[1].semilogx(ka, cm_exact, "k--", lw=1, label="$C_M$ (exact)")
    axes[1].axhline(2.0, ls=":", c="tab:green", lw=1, label="Morison long-wave limit")
    axes[1].set_xlabel("$ka$")
    axes[1].set_ylabel("$C_M$  [-]")
    axes[1].set_title("(b) frequency regime sweep (64 panels)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")

    # ---- (c) irregular-frequency suppression ----------------------------
    labels = [row["resonance"].replace("_resonance", "") for row in irregular]
    unaugmented = [float(row["unaugmented_condition_number"]) for row in irregular]
    chief = [float(row["chief_condition_number"]) for row in irregular]
    positions = range(len(irregular))
    width = 0.36
    axes[2].bar([p - width / 2 for p in positions], unaugmented, width,
                color="tab:red", label="$\\kappa$ without CHIEF")
    axes[2].bar([p + width / 2 for p in positions], chief, width,
                color="tab:blue", label="$\\kappa$ with CHIEF")
    axes[2].set_yscale("log")
    axes[2].set_xticks(list(positions))
    axes[2].set_xticklabels(labels, fontsize=8)
    for p, value in zip(positions, unaugmented):
        axes[2].annotate(f"{value:.1f}", (p - width / 2, value), ha="center",
                         va="bottom", fontsize=8)
    for p, value in zip(positions, chief):
        axes[2].annotate(f"{value:.2f}", (p + width / 2, value), ha="center",
                         va="bottom", fontsize=8)
    axes[2].set_ylabel("condition number  [-]")
    axes[2].set_title("(c) irregular-frequency suppression")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = out / name
    fig.savefig(path)
    plt.close(fig)
    return path