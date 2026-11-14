#!/usr/bin/env python3
"""CASE-BEM-CRACK-SIF benchmark runner: produces artifacts and report data.

Usage:
    python run_benchmark_crack_sif.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from tensorbem.crack_sif import (
    analytical_cod,
    analytical_sif,
    convergence_study,
    solve_crack_sif,
)

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------
SIGMA = 100.0e6       # 100 MPa remote tension
A = 0.01              # 10 mm half crack length
YOUNG = 2.0e11        # 200 GPa (steel)
POISSON = 0.3


def main(output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Primary solve (Chebyshev, exact) ----
    result = solve_crack_sif(
        sigma=SIGMA, a=A, young=YOUNG, poisson=POISSON,
        n_elements=200, method="chebyshev",
    )

    # ---- Convergence study (constant DD) ----
    n_list = [10, 20, 40, 80, 160, 320, 640]
    conv_results = convergence_study(SIGMA, A, YOUNG, POISSON, n_list, method="constant")

    # ---- JSON artifact ----
    artifact = {
        "case": "CASE-BEM-CRACK-SIF",
        "description": "Centre-crack Mode-I stress intensity factor",
        "reference": "Tada, Paris & Irwin (2000), Eq. 2.1a",
        "parameters": {
            "sigma_MPa": SIGMA / 1e6,
            "a_mm": A * 1e3,
            "E_GPa": YOUNG / 1e9,
            "nu": POISSON,
        },
        "primary_solve": {
            "method": "chebyshev",
            "n_elements": result.n_elements,
            "K_ref_MPa_sqrt_m": result.sif_ref / 1e6 * math.sqrt(1),  # already in Pa·sqrt(m)
            "K_got_MPa_sqrt_m": result.sif_got / 1e6 * math.sqrt(1),
            "error_pct": result.error_pct,
            "chebyshev_c0": float(result.chebyshev_coefficients[0]) if result.chebyshev_coefficients is not None else None,
        },
        "convergence_study": [
            {
                "n_elements": r.n_elements,
                "K_got_MPa_sqrt_m": r.sif_got / 1e6 * math.sqrt(1),
                "error_pct": r.error_pct,
                "condition_number": r.condition_number,
                "residual_l2": r.residual_l2,
            }
            for r in conv_results
        ],
        "verdict": "PASS" if result.error_pct < 5.0 else "FAIL",
    }

    # Convert SIF to proper units (Pa·sqrt(m) -> MPa·sqrt(m))
    artifact["primary_solve"]["K_ref_MPa_sqrt_m"] = result.sif_ref / 1e6
    artifact["primary_solve"]["K_got_MPa_sqrt_m"] = result.sif_got / 1e6
    for item in artifact["convergence_study"]:
        item["K_got_MPa_sqrt_m"] = item["K_got_MPa_sqrt_m"]

    json_path = output_dir / "case_bem_crack_sif.json"
    json_path.write_text(json.dumps(artifact, indent=2))
    print(f"JSON artifact: {json_path}")

    # ---- Figure 1: COD comparison ----
    fig, ax = plt.subplots(figsize=(8, 5))
    x_norm = result.cod_profile_x / A
    cod_got_norm = result.cod_profile_got / (4 * SIGMA / result.e_prime * A)
    cod_ref_norm = result.cod_profile_ref / (4 * SIGMA / result.e_prime * A)

    ax.plot(x_norm, cod_ref_norm, 'k-', linewidth=2, label='Analytical: $4\\sigma/E\' \\sqrt{a^2-x^2}$')
    ax.plot(x_norm, cod_got_norm, 'r.', markersize=3, label='TensorBEM (Chebyshev)')
    ax.set_xlabel('$x/a$')
    ax.set_ylabel('$D_y(x) / (4\\sigma a/E\')$')
    ax.set_title('CASE-BEM-CRACK-SIF: Crack Opening Displacement Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = output_dir / "cod_profile.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Figure: {fig_path}")

    # ---- Figure 2: Convergence ----
    fig, ax = plt.subplots(figsize=(8, 5))
    n_vals = [r.n_elements for r in conv_results]
    errors = [r.error_pct for r in conv_results]
    ax.semilogx(n_vals, errors, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=5.0, color='r', linestyle='--', alpha=0.7, label='5% tolerance')
    ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.7, label='1% tolerance')
    ax.set_xlabel('Number of elements')
    ax.set_ylabel('SIF error (%)')
    ax.set_title('CASE-BEM-CRACK-SIF: Convergence (constant DD method)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path2 = output_dir / "convergence.png"
    fig.savefig(fig_path2, dpi=150)
    plt.close(fig)
    print(f"Figure: {fig_path2}")

    # ---- Figure 3: SIF comparison bar chart ----
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ['Reference\n(Tada et al.)', 'TensorBEM\n(Chebyshev)']
    values = [result.sif_ref / 1e6, result.sif_got / 1e6]
    colors = ['#2196F3', '#4CAF50']
    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.set_ylabel('$K_I$ (MPa·√m)')
    ax.set_title('CASE-BEM-CRACK-SIF: SIF Comparison')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig_path3 = output_dir / "sif_comparison.png"
    fig.savefig(fig_path3, dpi=150)
    plt.close(fig)
    print(f"Figure: {fig_path3}")

    # ---- Repro script ----
    repro_script = f'''#!/usr/bin/env python3
"""One-line reproduction of CASE-BEM-CRACK-SIF."""
import sys; sys.path.insert(0, "src")
from tensorbem.crack_sif import solve_crack_sif
r = solve_crack_sif(sigma={SIGMA}, a={A}, young={YOUNG}, poisson={POISSON}, method="chebyshev")
print(f"K_ref = {{r.sif_ref:.6f}} Pa·√m")
print(f"K_got = {{r.sif_got:.6f}} Pa·√m")
print(f"Error = {{r.error_pct:.4e}}%")
assert r.error_pct < 5.0, f"FAIL: error {{r.error_pct:.2f}}% >= 5%"
print("PASS")
'''
    repro_path = output_dir / "reproduce.sh"
    repro_path.write_text(repro_script)
    repro_path.chmod(0o755)
    print(f"Repro script: {repro_path}")

    print(f"\\nVerdict: {artifact['verdict']}")
    print(f"K_ref = {result.sif_ref:.6f} Pa·√m = {result.sif_ref/1e6:.4f} MPa·√m")
    print(f"K_got = {result.sif_got:.6f} Pa·√m = {result.sif_got/1e6:.4f} MPa·√m")
    print(f"Error = {result.error_pct:.4e}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("artifacts"))
    args = parser.parse_args()
    main(args.output_dir)
