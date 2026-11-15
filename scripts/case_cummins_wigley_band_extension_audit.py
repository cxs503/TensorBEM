"""Audit the extended-band (0.2..20 rad/s) sweep of CASE-BEM-CUMMINS-WIGLEY.

Reads the committed medium sweep (0.2..12 rad/s, 480 panels) and the extended
one produced by ``case_cummins_wigley_extended_band.py`` (0.2..20 rad/s, 3840
panels) and answers one question: does the wider band turn surge/sway/yaw from
"empirically consistent" into "certified"?

Writes
------
``benchmarks/verified/casepool_bem_cummins_wigley/extended_band_audit.json``
    every number quoted by the report, plus the per-DOF tables
``docs/m21533_cummins_band_extension/``
    the figures (png)
"""
from __future__ import annotations

import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorbem.benchmarks import case_bem_cummins_wigley as case  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
DOCS = ROOT / "docs" / "m21533_cummins_band_extension"
AMPLITUDE_DOF = (2, 3, 4)  # heave, roll, pitch: certified on the committed band


def _load_anchor(path: pathlib.Path, key: tuple[int, int]) -> dict:
    payload = json.loads(path.read_text())
    for rung in payload["rungs"]:
        if (rung["n_sections"], rung["n_levels"]) == key:
            case._ANCHOR_CACHE[key] = rung
            return rung
    raise SystemExit(f"{path} has no anchor rung for mesh {key}")


def replay(sweep: case.Sweep, anchor: dict) -> dict:
    """Re-judge every gate that reads ``A``/``B`` on a given sweep."""
    case._ANCHOR_CACHE[(sweep.n_sections, sweep.n_levels)] = anchor
    gates: dict = {}
    gates["C0"] = case.solver_validity_gate(sweep)
    gates["C2"] = case.panel_adequacy_gate(sweep)
    gates["C3"] = case.kk_gate(sweep)
    gates["C3a"] = case.kk_gate(
        sweep,
        added_mass_inf=anchor["added_mass_inf_diagonal"],
        identifier="C3a",
    )
    gates["C7"] = case.added_mass_inf_gate(
        sweep, anchor, closure_rows=gates["C3"]["values"]["per_dof"]
    )
    gates["C4"] = case.round_trip_gate(sweep)
    gates["C5"] = case.free_decay_gate(sweep)
    gates["C6a"] = case.negative_control_gate(sweep)
    return gates


def _figure(medium: case.Sweep, extended: case.Sweep, audit: dict) -> str:
    DOCS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    colors = plt.cm.tab10(np.linspace(0, 1, 6))

    ax = axes[0, 0]
    for index, name in enumerate(case.DOF_NAMES):
        ax.loglog(extended.omega, extended.damping[:, index, index], "-", lw=1.2,
                  color=colors[index], label=name)
        ax.loglog(medium.omega, medium.damping[:, index, index], "o", ms=2.6,
                  color=colors[index])
    ax.axvline(case.BAND[1], color="k", ls="--", lw=0.9)
    ax.text(case.BAND[1] * 1.05, ax.get_ylim()[0] * 3, "committed band top", fontsize=8)
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel(r"$B_{ii}$ [kg/s]")
    ax.set_title("radiation damping: solid = 3840-panel extended band, dots = committed 480-panel")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[0, 1]
    rows = audit["C9"]["values"]["per_dof"]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["exponent_at_12"] for r in rows], width=0.4,
           label="slope at 12 rad/s")
    ax.bar(x + 0.2, [r["exponent_at_20"] for r in rows], width=0.4,
           label="slope at 20 rad/s")
    ax.axhline(2.05, color="crimson", ls="--", lw=1.0, label=r"$p>2.05$ needed")
    ax.set_xticks(x)
    ax.set_xticklabels([r["dof_name"] for r in rows])
    ax.set_ylabel("measured log-log damping slope $p$")
    ax.set_title("does the damping reach algebraic decay? (truncation budget finite iff $p>2.05$)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 0]
    for index, name in enumerate(case.DOF_NAMES):
        rows_ref = audit["medium"]["C3a"]["values"]["per_dof"][index]
        rows_ext = audit["extended"]["C3a"]["values"]["per_dof"][index]
        grid = np.asarray(rows_ext["sample_omega"], dtype=float)
        ax.semilogy(grid, np.clip(rows_ext["relative_residual"], 1e-12, None), "-",
                    lw=1.1, color=colors[index], label=f"{name} ext")
        ax.semilogy(np.asarray(rows_ref["sample_omega"], dtype=float),
                    np.clip(rows_ref["relative_residual"], 1e-12, None), "o",
                    ms=2.6, color=colors[index], alpha=0.55)
    ax.axhline(case.TOLERANCES["kk"], color="crimson", ls="--", lw=1.0,
               label="10 % case criterion")
    ax.set_xlabel(r"sample $\omega$ [rad/s]")
    ax.set_ylabel(r"$|A_{BEM}-A_{KK}|/\max|A|$")
    ax.set_title("anchored KK closure (zero free parameters): line = extended, dots = committed")
    ax.legend(ncol=2, fontsize=7)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1, 1]
    for index, name in enumerate(case.DOF_NAMES):
        ax.plot(extended.omega, extended.added_mass[:, index, index] /
                max(np.max(np.abs(extended.added_mass[:, index, index])), 1e-300),
                "-", lw=1.1, color=colors[index], label=name)
        ax.plot(medium.omega, medium.added_mass[:, index, index] /
                max(np.max(np.abs(medium.added_mass[:, index, index])), 1e-300),
                "o", ms=2.4, color=colors[index])
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel(r"$A_{ii}/\max|A_{ii}|$")
    ax.set_title("added mass, normalised per DOF (mesh convergence on the shared band)")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    target = DOCS / "fig_m21533_cummins_band_extension.png"
    fig.savefig(target, dpi=140)
    plt.close(fig)
    return str(target.relative_to(ROOT))


def main() -> int:
    medium = case.load_sweep(OUT / "sweep_medium.npz")
    extended = case.load_sweep(OUT / "sweep_extended.npz")
    anchor_medium = _load_anchor(OUT / "anchor_audit.json", (20, 6))
    anchor_extended = _load_anchor(OUT / "anchor_ladder_extended.json", (60, 16))

    gates_medium = replay(medium, anchor_medium)
    gates_extended = replay(extended, anchor_extended)

    consistency = case.extension_consistency_gate(medium, extended)
    decidability = case.truncation_decidability_gate(
        medium,
        extended,
        closure_rows=gates_extended["C3a"]["values"]["per_dof"],
        reference_rows=gates_medium["C3a"]["values"]["per_dof"],
    )
    gates_extended["C8"] = consistency
    gates_extended["C9"] = decidability
    verdict_extended = case.case_verdict(gates_extended)
    verdict_medium_published = case.gates_from_artifact(OUT / "sweep_medium.npz")["verdict"]

    comparison = []
    for index, name in enumerate(case.DOF_NAMES):
        ref = gates_medium["C3a"]["values"]["per_dof"][index]
        ext = gates_extended["C3a"]["values"]["per_dof"][index]
        comparison.append(
            {
                "dof_name": name,
                "exponent_at_12": ref["measured_tail_exponent"],
                "exponent_at_20": ext["measured_tail_exponent"],
                "budget_admissible_at_12": ref["truncation_budget_admissible"],
                "budget_admissible_at_20": ext["truncation_budget_admissible"],
                "n_decidable_at_12": ref["n_decidable"],
                "n_decidable_at_20": ext["n_decidable"],
                "n_samples": ext["n_samples"],
                "worst_anchored_residual_at_12": ref["max_relative_residual_decidable"],
                "worst_anchored_residual_at_20": ext["max_relative_residual_decidable"],
                "worst_all_samples_at_12": ref["max_relative_residual"],
                "worst_all_samples_at_20": ext["max_relative_residual"],
                "worst_omega_at_20": ext["worst_sample_omega"],
            }
        )

    payload = {
        "case": case.CASE_SLUG,
        "mission": "m21533",
        "purpose": (
            "Widen the Cummins/Wigley band from 12 to 20 rad/s on a panel-resolved "
            "(60x16, 3840 panels) hull so that the truncation budget of surge, sway "
            "and yaw can become finite, and re-judge the zero-free-parameter KK "
            "criterion on the extended sweep."
        ),
        "tolerances": case.TOLERANCES,
        "bands": {
            "committed": {"band": [0.2, 12.0], "panels": medium.panel_count,
                          "mesh": [medium.n_sections, medium.n_levels],
                          "n_omega": int(medium.omega.size),
                          "max_panel_scale": medium.max_panel_scale},
            "extended": {"band": [0.2, 20.0], "panels": extended.panel_count,
                         "mesh": [extended.n_sections, extended.n_levels],
                         "n_omega": int(extended.omega.size),
                         "max_panel_scale": extended.max_panel_scale},
        },
        "anchor_extended": {
            "mesh": [anchor_extended["n_sections"], anchor_extended["n_levels"]],
            "doubled_panel_count": anchor_extended["doubled_panel_count"],
            "condition_number": anchor_extended["condition_number"],
            "solver_relative_residual_l2": anchor_extended["solver_relative_residual_l2"],
            "added_mass_inf": anchor_extended["added_mass_inf"],
            "self_check": anchor_extended["self_check"],
        },
        "medium": {
            "verdict": verdict_medium_published,
            "C3a": gates_medium["C3a"],
            "C7": gates_medium["C7"],
        },
        "extended": {
            "verdict": verdict_extended,
            "gates": {
                key: gates_extended[key]
                for key in ("C0", "C2", "C3", "C3a", "C7", "C4", "C5", "C6a")
            },
        },
        "C8": consistency,
        "C9": decidability,
        "comparison": comparison,
        "figure": _figure(medium, extended, {
            "C9": decidability,
            "medium": {"C3a": gates_medium["C3a"]},
            "extended": {"C3a": gates_extended["C3a"]},
        }),
    }
    target = OUT / "extended_band_audit.json"
    target.write_text(json.dumps(payload, indent=1, default=float) + "\n")
    print(target)

    print("\nextended verdict:", json.dumps(verdict_extended))
    print("\nC8:", consistency["status"])
    print("C9:", decidability["status"])
    for row in comparison:
        print(
            f"{row['dof_name']:>6}: p12={row['exponent_at_12']:6.3f} "
            f"p20={row['exponent_at_20']:6.3f} "
            f"decidable {row['n_decidable_at_12']}->{row['n_decidable_at_20']}"
            f"/{row['n_samples']} "
            f"worst_anchored {100 * row['worst_anchored_residual_at_20']:6.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())