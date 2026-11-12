"""Benchmark runner for seakeeping verification suite.

Generates figures and JSON report for the seakeeping verification surface.

Run:
    python -m tensorbem.verification.seakeeping.benchmark_runner
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from tensorbem.verification.seakeeping.hemisphere_anchor import (
    compute_hemisphere_zero_frequency_anchor,
    compute_hemisphere_radiation_study,
)
from tensorbem.verification.seakeeping.diffraction_anchor import (
    compute_diffraction_anchor,
    compute_diffraction_sweep,
    compute_diffraction_convergence,
)
from tensorbem.verification.seakeeping.fk_excitation import (
    compute_fk_vs_bem_comparison,
)
from tensorbem.verification.seakeeping.wigley_rao import (
    compute_wigley_rao_with_limits,
    WIGLEY3_FULL_L, WIGLEY3_FULL_B, WIGLEY3_FULL_T,
)
from tensorbem.verification.seakeeping.mesh_convergence import (
    run_wigley_mesh_convergence,
)


def generate_figures(output_dir: Path) -> list[dict]:
    """Generate verification figures and return figure metadata."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    figures = []

    # Figure 1: Hemisphere radiation - A33 and B33 vs Ka
    print("Generating Figure 1: Hemisphere radiation vs Hulme (1982)...")
    study = compute_hemisphere_radiation_study(
        radius=1.0, density=1025.0, subdivisions=3,
        ka_values=(0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0),
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    kas = [l["ka"] for l in study["levels"]]
    ax1.plot(kas, [l["a33_hulme_dimless"] for l in study["levels"]], "b-o", label="Hulme (1982)")
    ax1.plot(kas, [l["a33_bem_dimless"] for l in study["levels"]], "r-s", label="TensorBEM BEM")
    ax1.set_xlabel("Ka")
    ax1.set_ylabel("A33' (dimensionless)")
    ax1.set_title("Heave Added Mass: BEM vs Hulme (1982)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(kas, [l["b33_hulme_dimless"] for l in study["levels"]], "b-o", label="Hulme (1982)")
    ax2.plot(kas, [l["b33_bem_dimless"] for l in study["levels"]], "r-s", label="TensorBEM BEM")
    ax2.set_xlabel("Ka")
    ax2.set_ylabel("B33' (dimensionless)")
    ax2.set_title("Heave Damping: BEM vs Hulme (1982)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({
        "name": "hemisphere_radiation_comparison",
        "caption": "Hemisphere heave added mass and damping vs Hulme (1982) reference",
        "png_base64": base64.b64encode(buf.getvalue()).decode(),
    })

    # Figure 2: Diffraction excitation - surge and heave vs Hulme
    print("Generating Figure 2: Diffraction excitation vs Hulme (1982)...")
    diff_sweep = compute_diffraction_sweep(
        ka_values=(0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0),
        subdivisions=2,
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    kas = list(diff_sweep.ka_values)
    ax1.plot(kas, [abs(l.bem_fx) for l in diff_sweep.levels], "r-s", label="BEM surge")
    ax1.plot(kas, [abs(l.hulme_fx) for l in diff_sweep.levels], "b-o", label="Hulme surge")
    ax1.set_xlabel("Ka")
    ax1.set_ylabel("|F_x| (N)")
    ax1.set_title("Surge Excitation: BEM vs Hulme (1982)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(kas, [abs(l.bem_fz) for l in diff_sweep.levels], "r-s", label="BEM heave")
    ax2.plot(kas, [abs(l.hulme_fz) for l in diff_sweep.levels], "b-o", label="Hulme heave")
    ax2.set_xlabel("Ka")
    ax2.set_ylabel("|F_z| (N)")
    ax2.set_title("Heave Excitation: BEM vs Hulme (1982)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({
        "name": "diffraction_excitation_comparison",
        "caption": "Hemisphere surge and heave diffraction excitation vs Hulme (1982) reference",
        "png_base64": base64.b64encode(buf.getvalue()).decode(),
    })

    # Figure 3: FK vs BEM excitation force
    print("Generating Figure 3: FK vs BEM excitation force...")
    fk_results = compute_fk_vs_bem_comparison(
        length=100.0, beam=20.0, draft=8.0,
        omega_values=(0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0),
        n_length=15, n_beam=8, n_draft=6,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    omegas_fk = [r["omega"] for r in fk_results]
    fk_mags = [r["fk_magnitude"] for r in fk_results]
    bem_mags = [r["bem_magnitude"] for r in fk_results]
    ax.plot(omegas_fk, fk_mags, "b-o", label="Froude-Krylov (closed-form)")
    ax.plot(omegas_fk, bem_mags, "r-s", label="TensorBEM BEM diffraction")
    ax.set_xlabel("ω (rad/s)")
    ax.set_ylabel("|F₃| (N)")
    ax.set_title("Heave Excitation Force: FK vs BEM (Box Barge L=100m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({
        "name": "fk_vs_bem_excitation",
        "caption": "Froude-Krylov closed-form vs BEM diffraction heave excitation force",
        "png_base64": base64.b64encode(buf.getvalue()).decode(),
    })

    # Figure 4: Wigley III RAO spectrum
    print("Generating Figure 4: Wigley III RAO spectrum...")
    rao_result = compute_wigley_rao_with_limits(nx=12, nz=6)
    fig, ax = plt.subplots(figsize=(8, 5))
    omegas = rao_result["spectrum"]["omegas"]
    rao_amps = rao_result["spectrum"]["heave_rao_amplitude"]
    ax.plot(omegas, rao_amps, "b-o", label="Heave RAO")
    ax.axhline(y=1.0, color="g", linestyle="--", alpha=0.5, label="Long-wave limit (1.0)")
    ax.axhline(y=0.2, color="r", linestyle="--", alpha=0.5, label="High-freq threshold (0.2)")
    ax.set_xlabel("ω (rad/s)")
    ax.set_ylabel("Heave RAO (m/m)")
    ax.set_title(f"Wigley III Heave RAO (L={WIGLEY3_FULL_L}m, nx=12, nz=6)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(rao_amps) * 1.1)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({
        "name": "wigley_rao_spectrum",
        "caption": f"Wigley III heave RAO frequency response (L={WIGLEY3_FULL_L}m)",
        "png_base64": base64.b64encode(buf.getvalue()).decode(),
    })

    # Figure 5: Mesh convergence
    print("Generating Figure 5: Mesh convergence...")
    conv = run_wigley_mesh_convergence(
        omega=1.0,
        nx_values=[10, 15, 20, 30],
        nz_values=[5, 8, 10, 15],
        L=WIGLEY3_FULL_L, B=WIGLEY3_FULL_B, T=WIGLEY3_FULL_T,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    panels = conv.panel_counts
    rao_vals = conv.heave_rao_amplitudes
    ax.plot(panels, rao_vals, "b-o", label="Heave RAO at ω=1.0")
    ax.set_xlabel("Panel count")
    ax.set_ylabel("Heave RAO (m/m)")
    ax.set_title("Mesh Convergence: Wigley III Heave RAO")
    ax.legend()
    ax.grid(True, alpha=0.3)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({
        "name": "mesh_convergence",
        "caption": "Mesh convergence of Wigley III heave RAO at ω=1.0 rad/s",
        "png_base64": base64.b64encode(buf.getvalue()).decode(),
    })

    return figures


def run_full_verification(output_dir: Path) -> dict:
    """Run the full seakeeping verification suite and return results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TensorBEM Seakeeping Verification Suite")
    print("=" * 60)

    # 1. Hemisphere radiation anchor
    print("\n[1/5] Hemisphere zero-frequency added-mass anchor...")
    hemi_anchor = compute_hemisphere_zero_frequency_anchor(
        radius=1.0, density=1025.0, subdivisions=3, ka_low=0.01,
    )
    hemi_study = compute_hemisphere_radiation_study(
        radius=1.0, density=1025.0, subdivisions=3,
        ka_values=(0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0),
    )
    print(f"  A33 displaced mass: {hemi_anchor.displaced_mass:.2f} kg")
    print(f"  A33 BEM at Ka=0.01: {hemi_anchor.a33_bem_dim:.2f} kg")
    print(f"  A33 vs Hulme error: {hemi_anchor.a33_vs_hulme_rel_error*100:.2f}%")

    # 2. Diffraction excitation anchor
    print("\n[2/5] Hemisphere diffraction excitation anchor...")
    diff_anchor = compute_diffraction_anchor(ka=1.0, subdivisions=2)
    diff_sweep = compute_diffraction_sweep(
        ka_values=(0.3, 0.5, 0.8, 1.0, 1.5, 2.0),
        subdivisions=2,
    )
    diff_conv = compute_diffraction_convergence(
        ka=1.0, subdivisions=(1, 2, 3),
    )
    print(f"  Surge error at Ka=1.0: {diff_anchor.surge_rel_error*100:.2f}%")
    print(f"  Heave error at Ka=1.0: {diff_anchor.heave_rel_error*100:.2f}%")
    print(f"  Haskind heave error: {diff_anchor.haskind_heave_rel_error*100:.2f}%")
    print(f"  Finest heave error (sub=3): {diff_conv.finest_heave_error*100:.2f}%")

    # 3. FK excitation
    print("\n[3/5] Froude-Krylov excitation force...")
    fk_results = compute_fk_vs_bem_comparison(
        length=100.0, beam=20.0, draft=8.0,
        omega_values=(0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0),
        n_length=15, n_beam=8, n_draft=6,
    )
    long_wave_fk = [r for r in fk_results if r["long_wave_limit"]]
    max_fk_error = max((r["magnitude_rel_error"] for r in long_wave_fk), default=0)
    if long_wave_fk:
        print(f"  FK vs BEM max error (kL<0.3): {max_fk_error*100:.2f}%")

    # 4. Wigley RAO
    print("\n[4/5] Wigley III RAO with limit validation...")
    rao_result = compute_wigley_rao_with_limits(nx=12, nz=6)
    lw = rao_result["limit_validation"]
    print(f"  Long-wave RAO mean: {lw['long_wave_rao_mean']:.4f} (pass: {lw['long_wave_pass']})")
    print(f"  High-freq RAO: {lw['high_freq_rao']:.4f} (pass: {lw['high_freq_pass']})")

    # 5. Mesh convergence
    print("\n[5/5] Mesh convergence study...")
    conv = run_wigley_mesh_convergence(
        omega=1.0,
        nx_values=[10, 15, 20, 30],
        nz_values=[5, 8, 10, 15],
        L=WIGLEY3_FULL_L, B=WIGLEY3_FULL_B, T=WIGLEY3_FULL_T,
    )
    print(f"  Converged: {conv.converged}")
    if conv.max_relative_change is not None:
        print(f"  Max relative change: {conv.max_relative_change:.4f}")

    # Generate figures
    print("\nGenerating figures...")
    figures = generate_figures(output_dir)

    # Save figures to files
    for fig in figures:
        fig_path = output_dir / f"{fig['name']}.png"
        import base64
        with open(fig_path, "wb") as f:
            f.write(base64.b64decode(fig["png_base64"]))
        print(f"  Saved: {fig_path}")

    # Build report
    report = {
        "title": "TensorBEM 3D Potential Flow Seakeeping Verification Surface",
        "mission_id": 16442,
        "scenario_key": "seakeeping-verification-surface",
        "tags": ["seakeeping", "radiation", "diffraction", "RAO", "verification"],
        "verification_results": {
            "hemisphere_anchor": {
                "displaced_mass": hemi_anchor.displaced_mass,
                "bem_a33_at_ka_001": hemi_anchor.a33_bem_dim,
                "vs_hulme_rel_error": hemi_anchor.a33_vs_hulme_rel_error,
                "panel_count": hemi_anchor.panel_count,
                "pass": hemi_anchor.a33_vs_hulme_rel_error < 0.05,
            },
            "diffraction_anchor": {
                "ka": diff_anchor.ka,
                "surge_rel_error": diff_anchor.surge_rel_error,
                "heave_rel_error": diff_anchor.heave_rel_error,
                "haskind_heave_rel_error": diff_anchor.haskind_heave_rel_error,
                "haskind_surge_rel_error": diff_anchor.haskind_surge_rel_error,
                "finest_heave_error": diff_conv.finest_heave_error,
                "finest_surge_error": diff_conv.finest_surge_error,
                "observed_order_heave": diff_conv.observed_order_heave,
                "observed_order_surge": diff_conv.observed_order_surge,
                "pass": (
                    diff_anchor.heave_rel_error < 0.10
                    and diff_anchor.surge_rel_error < 0.30
                    and diff_anchor.haskind_heave_rel_error < 0.05
                ),
            },
            "fk_excitation": {
                "long_wave_max_error": max_fk_error,
                "pass": max_fk_error < 0.20,
            },
            "wigley_rao": {
                "long_wave_rao_mean": lw["long_wave_rao_mean"],
                "long_wave_pass": lw["long_wave_pass"],
                "high_freq_rao": lw["high_freq_rao"],
                "high_freq_pass": lw["high_freq_pass"],
            },
            "mesh_convergence": {
                "converged": conv.converged,
                "max_relative_change": conv.max_relative_change,
                "pass": conv.converged,
            },
        },
        "figures": [{"name": f["name"], "caption": f["caption"]} for f in figures],
    }

    # Save JSON report
    report_path = output_dir / "seakeeping_verification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")

    return report


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent.parent.parent / "artifacts" / "seakeeping_verification"
    report = run_full_verification(output_dir)
    print("\n" + "=" * 60)
    print("Verification Summary:")
    vr = report["verification_results"]
    print(f"  Hemisphere anchor: {'PASS' if vr['hemisphere_anchor']['pass'] else 'FAIL'}")
    print(f"  Diffraction anchor: {'PASS' if vr['diffraction_anchor']['pass'] else 'FAIL'}")
    print(f"  FK excitation: {'PASS' if vr['fk_excitation']['pass'] else 'FAIL'}")
    print(f"  Wigley RAO long-wave: {'PASS' if vr['wigley_rao']['long_wave_pass'] else 'FAIL'}")
    print(f"  Wigley RAO high-freq: {'PASS' if vr['wigley_rao']['high_freq_pass'] else 'FAIL'}")
    print(f"  Mesh convergence: {'PASS' if vr['mesh_convergence']['pass'] else 'FAIL'}")
    print("=" * 60)
