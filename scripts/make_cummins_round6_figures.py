"""Generate figures for CASE-BEM-CUMMINS-WIGLEY round 6 report.

Produces:
1. Added mass and damping curves for all 6 DOFs
2. Memory kernel decay for heave
3. Free decay comparison (time-domain vs half-power)
4. G3 pointwise error distribution
5. Gate summary bar chart
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str = "artifacts/case_bem_cummins_wigley_round6/results_closure.json") -> dict:
    with open(path) as f:
        return json.load(f)


def load_points(path: str = "artifacts/case_bem_cummins_wigley_round6/points", level: str = "coarse") -> dict:
    """Load all point results."""
    import glob
    points = {}
    for p in sorted(glob.glob(f"{path}/{level}_omega_*.json")):
        with open(p) as f:
            d = json.load(f)
        points[d["omega"]] = d
    return points


def fig_radiation_curves(results: dict, points: dict, outdir: pathlib.Path):
    """Figure 1: Added mass and damping vs frequency for all 6 DOFs."""
    dof_labels = ["surge", "sway", "heave", "roll", "pitch", "yaw"]
    omega = np.array(results["omega_grid"])

    fig, axes = plt.subplots(2, 6, figsize=(18, 5), sharex=True)
    fig.suptitle("Radiation Added Mass and Damping — Wigley III Hull (coarse mesh, 68 nodes)", fontsize=12)

    for j, dof in enumerate(dof_labels):
        pd = results["per_dof"][dof]
        a_inf = pd["added_mass_inf"]

        # Load damping and added mass from points
        a_vals = []
        b_vals = []
        for w in omega:
            key = round(float(w), 9)
            if key in points:
                p = points[key]
                a_vals.append(p["added_mass"][j][j])
                b_vals.append(p["damping"][j][j])

        ax_a = axes[0, j]
        ax_b = axes[1, j]

        ax_a.plot(omega, a_vals, 'b-', linewidth=1.5, label='A(ω)')
        ax_a.axhline(a_inf, color='r', linestyle='--', linewidth=0.8, label=f'A∞={a_inf:.1f}')
        ax_a.set_ylabel('Added mass')
        ax_a.set_title(dof.capitalize())
        ax_a.legend(fontsize=7)
        ax_a.grid(True, alpha=0.3)

        ax_b.plot(omega, b_vals, 'g-', linewidth=1.5, label='B(ω)')
        ax_b.set_ylabel('Damping')
        ax_b.set_xlabel('ω (rad/s)')
        ax_b.legend(fontsize=7)
        ax_b.grid(True, alpha=0.3)

    plt.tight_layout()
    out = outdir / "fig_radiation_curves.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[fig] -> {out}")
    return out


def fig_kernel_and_decay(results: dict, points: dict, outdir: pathlib.Path):
    """Figure 2: Heave memory kernel and free decay comparison."""
    from tensorbem.benchmarks import cummins_wigley_kk_causality as case
    from tensorbem import time_domain as td
    from tensorbem.causality import fit_added_mass_asymptote

    # Load heave data
    omega = np.array(results["omega_grid"])
    hydro = results["geometry"]
    mass = hydro["displaced_mass"]
    c33 = hydro["C33"]

    a_vals = []
    b_vals = []
    for w in omega:
        key = round(float(w), 9)
        if key in points:
            p = points[key]
            a_vals.append(p["added_mass"][2][2])
            b_vals.append(p["damping"][2][2])

    omega = np.array(omega)
    added = np.array(a_vals)
    damping = np.array(b_vals)

    fit = fit_added_mass_asymptote(omega, added)
    a_inf = float(fit.added_mass_inf)

    tau = np.arange(0.0, 40.0, 0.02)
    kernel = td.radiation_memory_kernel(omega, damping, tau, low_frequency_model="zero")

    decay = td.cummins_free_decay(
        kernel, mass=mass, restoring=c33, added_mass_inf=a_inf,
        initial_displacement=0.01, dt=0.02, duration=40.0
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Heave Memory Kernel and Free Decay — Wigley III Hull", fontsize=12)

    # Kernel
    ax = axes[0]
    ax.plot(kernel.tau, kernel.values, 'b-', linewidth=1.5)
    ax.set_xlabel('τ (s)')
    ax.set_ylabel('K(τ)')
    ax.set_title('Memory Kernel K(t)')
    ax.grid(True, alpha=0.3)

    # Free decay
    ax = axes[1]
    ax.plot(decay.time, decay.displacement, 'b-', linewidth=1.5)
    ax.set_xlabel('t (s)')
    ax.set_ylabel('z(t)')
    ax.set_title('Free Decay (time domain)')
    ax.grid(True, alpha=0.3)

    # RAO peak
    ax = axes[2]
    rao = np.zeros_like(omega)
    for i, w in enumerate(omega):
        denom = -w**2 * (mass + added[i]) + 1j * w * damping[i] + c33
        rao[i] = 1.0 / abs(denom)
    ax.plot(omega, rao, 'b-', linewidth=1.5)
    peak_idx = np.argmax(rao)
    ax.axvline(omega[peak_idx], color='r', linestyle='--', linewidth=0.8,
               label=f'ω_peak={omega[peak_idx]:.2f}')
    ax.set_xlabel('ω (rad/s)')
    ax.set_ylabel('|H(ω)|')
    ax.set_title('Response Amplitude Operator')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = outdir / "fig_kernel_decay.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[fig] -> {out}")
    return out


def fig_gate_summary(results: dict, outdir: pathlib.Path):
    """Figure 3: Gate summary bar chart with tolerances."""
    gates = results["gates"]
    gate_names = ["G1\nCross-solver", "G2\nKernel@0", "G3\nKramers-Kronig",
                  "G4\nFree decay", "G5\nRound-trip"]
    rel_errs = []
    tols = []
    passed = []
    colors = []

    for key, name in zip(["G1_cross_solver", "G2_kernel_at_zero", "G3_kramers_kronig",
                           "G4_free_decay", "G5_round_trip"], gate_names):
        info = gates[key]
        rel_errs.append(info.get("rel_err", 0) or 0)
        tols.append(info.get("tol", 0) or 0)
        passed.append(info.get("passed", False))
        colors.append("green" if info.get("passed") else "red")

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(gate_names))
    width = 0.35

    bars = ax.bar(x - width/2, rel_errs, width, label='Relative error', color=colors, alpha=0.7)
    ax.bar(x + width/2, tols, width, label='Tolerance', color='gray', alpha=0.3, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(gate_names, fontsize=9)
    ax.set_ylabel('Relative error')
    ax.set_title('CASE-BEM-CUMMINS-WIGLEY Gate Summary (coarse mesh, 68 nodes)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, rel_errs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2%}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out = outdir / "fig_gate_summary.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[fig] -> {out}")
    return out


def fig_g4_comparison(results: dict, outdir: pathlib.Path):
    """Figure 4: G4 zeta comparison (decay vs half-power) per DOF."""
    dof_labels = ["surge", "sway", "heave", "roll", "pitch", "yaw"]
    zeta_decay = []
    zeta_hp = []

    for dof in dof_labels:
        g4 = results["per_dof"][dof]["G4"]
        zeta_decay.append(g4.get("zeta_decay") or 0)
        zeta_hp.append(g4.get("zeta_half_power") or 0)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(dof_labels))
    width = 0.35

    ax.bar(x - width/2, zeta_decay, width, label='ζ from free decay', color='blue', alpha=0.7)
    ax.bar(x + width/2, zeta_hp, width, label='ζ from half-power', color='orange', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in dof_labels])
    ax.set_ylabel('Damping ratio ζ')
    ax.set_title('G4: Free Decay vs Half-Power Damping Ratio (per DOF)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = outdir / "fig_g4_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[fig] -> {out}")
    return out


def main():
    outdir = pathlib.Path("artifacts/case_bem_cummins_wigley_round6")
    results = load_results()
    points = load_points()

    fig_radiation_curves(results, points, outdir)
    fig_kernel_and_decay(results, points, outdir)
    fig_gate_summary(results, outdir)
    fig_g4_comparison(results, outdir)

    print("[fig] all figures generated")


if __name__ == "__main__":
    main()
