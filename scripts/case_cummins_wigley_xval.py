#!/usr/bin/env python3
"""Run gate C10 (external capytaine anchor) on the committed Wigley sweep."""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorbem.benchmarks import case_bem_cummins_wigley as case
from tensorbem.benchmarks import case_bem_cummins_wigley_xval as xval

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "fig_case_bem_cummins_wigley_xval.png"


def figure(study, reference, sweep) -> str:
    omega = np.asarray(sweep.omega, dtype=float)
    names = case.DOF_NAMES
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
    for dof, ax in enumerate(axes[0]):
        block = reference["levels"]["medium"]["variants"]["lid"]
        ref_a = np.asarray(block["added_mass"], dtype=float)[:, dof, dof]
        ref_b = np.asarray(block["damping"], dtype=float)[:, dof, dof]
        ax.plot(omega, sweep.added_mass[:, dof, dof], "k-", lw=1.6, label="A$_%d$$_%d$ TensorBEM" % (dof + 1, dof + 1))
        ax.plot(omega, ref_a, "C1--", lw=1.4, label="A capytaine (same panels)")
        ax.plot(omega, sweep.damping[:, dof, dof], "C0-", lw=1.2, label="B TensorBEM")
        ax.plot(omega, ref_b, "C2:", lw=1.4, label="B capytaine")
        ax.set_title(names[dof])
        ax.set_yscale("symlog", linthresh=1.0)
        ax.grid(alpha=0.3)
        ax.set_xlabel("$\\omega$ [rad/s]")
        if dof == 0:
            ax.legend(fontsize=6.5, loc="best")
    ax = axes[1][0]
    per = np.asarray(study["compare"]["medium"]["lid"]["worst_frequency_deviation"], dtype=float)
    bare = np.asarray(xval.compare(sweep, reference, variant="no_lid")["per_frequency_deviation"], dtype=float)
    lid = np.asarray(xval.compare(sweep, reference, variant="lid")["per_frequency_deviation"], dtype=float)
    ax.semilogy(omega, 100 * bare, "C3.-", lw=1.2, label="no lid")
    ax.semilogy(omega, 100 * lid, "C0.-", lw=1.2, label="lid")
    ax.axhline(100 * case.TOLERANCES["kk"], color="r", ls="--", lw=1.0, label="case 10%")
    ax.axhline(100 * xval.TARGET_FRACTION, color="grey", ls=":", lw=1.0, label="3% bar")
    ax.set_ylabel("max cell deviation [%]")
    ax.set_xlabel("$\\omega$ [rad/s]")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    ax.set_title("cross-code deviation vs $\\omega$")
    ax = axes[1][1]
    for k, (level, row) in enumerate(study["convergence"]["ladder"].items()):
        ax.bar(k, 100 * row["worst"]["relative"], color="C%d" % k)
        ax.text(k, 100 * row["worst"]["relative"], " %d panels" % row["panel_count"], ha="center", va="bottom", fontsize=7)
    ax.axhline(100 * case.TOLERANCES["kk"], color="r", ls="--", lw=1.0)
    ax.set_xticks(range(len(study["convergence"]["ladder"])))
    ax.set_xticklabels(list(study["convergence"]["ladder"]))
    ax.set_ylabel("worst deviation [%]")
    ax.set_title("panel ladder (capytaine)")
    ax = axes[1][2]
    rel = np.asarray(study["compare"]["medium"]["lid"]["worst"]["relative"])
    mat_a = np.asarray(xval.compare(sweep, reference)["relative_matrices"]["added_mass"], dtype=float)
    mat_b = np.asarray(xval.compare(sweep, reference)["relative_matrices"]["damping"], dtype=float)
    combined = np.maximum(mat_a, mat_b)
    im = ax.imshow(100 * combined, cmap="magma")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(names, rotation=45, fontsize=7); ax.set_yticklabels(names, fontsize=7)
    ax.set_title("worst cell deviation [%]\n(rows=force dof, cols=motion dof)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG, dpi=140)
    plt.close(fig)
    return FIG.name


def main() -> int:
    reference = xval.load_reference()
    sweep = case.load_sweep(xval.SWEEP_PATH)
    study = xval.run_study(reference, sweep)
    name = figure(study, reference, sweep)
    paths = xval.write_outputs(study, figure=name)
    gate = study["gate"]
    print(f"C10 {gate['status']}: worst {100*gate['values']['worst']['relative']:.2f}% "
          f"({gate['values']['worst']['dof_to']}<-{gate['values']['worst']['dof_from']}, "
          f"omega={gate['values']['worst_frequency_rad_s']:.4f}) "
          f"identical_panels={gate['values']['identical_panels']}")
    print(f"cells within 3%: {gate['values']['cells_within_target']}/{gate['values']['scored_cells']}")
    a = study["attribution"]
    print(f"attribution supported={a['supported']} flagged={a['flagged']} repaired={a['repaired']} "
          f"max no_lid {100*a['max_deviation_no_lid']:.2f}% -> lid {100*a['max_deviation_lid']:.2f}%")
    for k, v in paths.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
