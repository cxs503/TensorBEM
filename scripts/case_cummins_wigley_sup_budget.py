#!/usr/bin/env python3
"""Per-DOF KK tail budget for CASE-BEM-CUMMINS-WIGLEY under the kernel-sup bound.

Both budget routes are evaluated on the *committed* sweeps so that the numbers
are reproducible without a new BEM solve:

* closed form (``tensorbem.causality.high_band_truncation_budget``, needs p>2.05)
* kernel-sup envelope (``tensorbem.kk_kernel_sup_bound``, needs only p>1)

Writes ``docs/m22027_kk_sup_budget/sup_budget.json`` + the figure.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorbem.benchmarks import case_bem_cummins_wigley as case  # noqa: E402
from tensorbem.causality import high_band_truncation_budget, local_tail_exponent  # noqa: E402
from tensorbem.kk_kernel_sup_bound import kernel_sup_tail_budget  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
OUT = ROOT / "docs" / "m22027_kk_sup_budget"
BUDGET_BAR = float(case.TOLERANCES["kk_budget"])


def analyse(label: str) -> dict:
    sweep = case.load_sweep(ARTIFACT / f"sweep_{label}.npz")
    anchor = case.anchor_for_sweep(sweep)
    gate = case.kk_gate(
        sweep,
        added_mass_inf=anchor["added_mass_inf_diagonal"],
        identifier="C3a",
    )
    omega = np.asarray(sweep.omega, dtype=float)
    rows = []
    for index, name in enumerate(case.DOF_NAMES):
        row = gate["values"]["per_dof"][index]
        damping = np.abs(np.asarray(sweep.damping[:, index, index], dtype=float))
        added = np.abs(np.asarray(sweep.added_mass[:, index, index], dtype=float))
        scale = float(np.max(added))
        samples = np.asarray(row["sample_omega"], dtype=float)
        exponent, _ = local_tail_exponent(omega, damping)
        closed = high_band_truncation_budget(omega, damping, samples)
        sup = kernel_sup_tail_budget(omega, damping, samples)
        observed = row["max_relative_residual"]
        closed_rel = (
            float(np.max(closed.budget / scale)) if closed.admissible else float("inf")
        )
        sup_rel = float(np.max(sup.budget / scale)) if sup.admissible else float("inf")
        rows.append(
            {
                "dof": index,
                "dof_name": name,
                "measured_tail_exponent": float(exponent),
                "amplitude_scale_max_abs_A": scale,
                "n_samples": int(samples.size),
                "observed_kk_residual_relative": float(observed),
                "closed_form_budget_relative": closed_rel,
                "closed_form_decidable_all_samples": bool(closed.admissible),
                "sup_bound_budget_relative": sup_rel,
                "sup_bound_admissible": bool(sup.admissible),
                "decidable_under_closed_form": bool(
                    closed.admissible and closed_rel <= BUDGET_BAR
                ),
                "decidable_under_sup_bound": bool(
                    sup.admissible and sup_rel <= BUDGET_BAR
                ),
                "sup_bound_reason": sup.reason,
            }
        )
    return {
        "label": label,
        "mesh": [sweep.n_sections, sweep.n_levels],
        "panels": int(sweep.panel_count),
        "band": [float(omega[0]), float(omega[-1])],
        "budget_bar": BUDGET_BAR,
        "rows": rows,
    }


def figure(audits: list[dict]) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    medium = next(a for a in audits if a["label"] == "medium")
    names = [r["dof_name"] for r in medium["rows"]]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    colors = plt.cm.tab10(np.linspace(0, 1, 6))

    ax = axes[0]
    for j, audit in enumerate(audits):
        values = [r["measured_tail_exponent"] for r in audit["rows"]]
        ax.bar(np.arange(6) + 0.25 * (j - 0.5), values, width=0.25,
               color=[colors[k] for k in range(6)], alpha=0.9 if j else 0.55,
               edgecolor="k", linewidth=0.5)
    ax.axhline(2.05, color="k", ls="--", lw=1.2, label="closed-form bar p>2.05")
    ax.axhline(1.05, color="crimson", ls=":", lw=1.4, label="kernel-sup bar p>1.05")
    ax.set_xticks(range(6)); ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("measured damping tail exponent p")
    ax.set_title("A: where each budget route stops being usable")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    width = 0.2
    for j, key in enumerate(("closed_form_budget_relative", "sup_bound_budget_relative")):
        vals = [min(r[key], 1e3) for r in medium["rows"]]
        ax.bar(np.arange(6) + (j - 1.5) * width, vals, width=width,
               label=key.replace("_relative", ""), log=True,
               color=["0.6", "steelblue"][j])
    ax.axhline(BUDGET_BAR, color="crimson", ls="--", lw=1.2,
               label=f"budget bar {BUDGET_BAR:.0%} of max|A|")
    ax.set_xticks(range(6)); ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("tail budget / max|A_diag|")
    ax.set_title("B: tail budget on the committed medium sweep")
    ax.legend(fontsize=8)

    ax = axes[2]
    for i, r in enumerate(medium["rows"]):
        ax.scatter(r["sup_bound_budget_relative"], r["observed_kk_residual_relative"],
                   s=70, color=colors[i], label=r["dof_name"])
        ax.annotate(r["dof_name"], (r["sup_bound_budget_relative"],
                    r["observed_kk_residual_relative"]), textcoords="offset points",
                    xytext=(5, 3), fontsize=8)
    ax.axhline(BUDGET_BAR, color="crimson", ls="--", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sup-bound tail budget / max|A|")
    ax.set_ylabel("observed KK residual / max|A|")
    ax.set_title("C: residual must sit under the budget to be attributable")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    target = OUT / "fig_m22027_kk_sup_budget.png"
    fig.savefig(target, dpi=150)
    return str(target)


def main() -> int:
    audits = [analyse(label) for label in ("coarse", "medium")]
    target = OUT / "fig_m22027_kk_sup_budget.png"
    path = figure(audits)
    payload = {
        "case": case.CASE_SLUG,
        "mission": "m22027",
        "purpose": (
            "Decide how much of the KK certification gap on surge/sway/yaw is a real "
            "data limitation and how much is the closed-form route's p>2.05 bar, by "
            "re-judging the same committed sweeps against a distribution-free "
            "kernel-sup bound that only needs p>1."
        ),
        "budget_bar": BUDGET_BAR,
        "audits": audits,
        "figure": path,
        "cluster_jobs": {
            "extended_band_sweep": "cj3855 (60x16, 0.2..20 rad/s, 55 nodes, 25 shards)",
            "extended_anchor_ladder": "cj3856 (44x14 + 60x16 image anchors)",
            "probe_node_54": "cj3854",
            "status": "queued at submission time; artifacts not yet available",
        },
    }
    (OUT / "sup_budget.json").write_text(json.dumps(payload, indent=1) + "\n")
    for audit in audits:
        print(f"[{audit['label']}] mesh {audit['mesh']} panels {audit['panels']}")
        for r in audit["rows"]:
            print(
                f"  {r['dof_name']:6s} p={r['measured_tail_exponent']:5.2f} "
                f"closed={'finite' if r['closed_form_decidable_all_samples'] else 'INF':>6s} "
                f"closed_budget={r['closed_form_budget_relative']:9.3g} "
                f"sup_budget={r['sup_bound_budget_relative']:9.3g} "
                f"residual={r['observed_kk_residual_relative']:7.4f} "
                f"decidable_sup={r['decidable_under_sup_bound']}"
            )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())