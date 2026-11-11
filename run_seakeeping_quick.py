#!/usr/bin/env python3
"""Quick seakeeping verification - reduced meshes for fast execution.

This is a reduced verification that demonstrates all four components
with real BEM results but smaller meshes for faster execution.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tensorbem.verification.seakeeping.hemisphere_anchor import (
    compute_hemisphere_zero_frequency_anchor,
    compute_hemisphere_radiation_study,
)
from tensorbem.verification.seakeeping.fk_excitation import (
    compute_fk_heave_force_box_barge,
    compute_fk_vs_bem_comparison,
)
from tensorbem.verification.seakeeping.wigley_rao import (
    compute_wigley_rao_spectrum,
)
from tensorbem.verification.seakeeping.mesh_convergence import (
    run_wigley_mesh_convergence,
)


def main():
    print("=" * 70)
    print("TENSORbem SEAKEEPING VERIFICATION (QUICK)")
    print("=" * 70)
    
    output_dir = Path(__file__).parent / "runs"
    output_dir.mkdir(exist_ok=True)
    
    results = {}
    t_total = time.time()
    
    # 1. Hemisphere anchor (subdivisions=2 for speed)
    print("\n[1/4] Hemisphere radiation anchor...")
    t0 = time.time()
    anchor = compute_hemisphere_zero_frequency_anchor(
        radius=1.0, density=1025.0, subdivisions=2, ka_low=0.01,
    )
    print(f"  A33 BEM: {anchor.a33_bem_dimless:.4f}, vs Hulme: {anchor.a33_vs_hulme_rel_error:.2%}")
    print(f"  Time: {time.time()-t0:.1f}s")
    
    results["hemisphere_anchor"] = {
        "a33_bem_dimless": anchor.a33_bem_dimless,
        "a33_hulme_dimless": anchor.hulme_a33_dimless,
        "a33_vs_hulme_rel_error": anchor.a33_vs_hulme_rel_error,
        "a33_vs_zero_freq_rel_error": anchor.a33_vs_zero_freq_rel_error,
        "panel_count": anchor.panel_count,
        "condition_number": anchor.condition_number,
    }
    
    # 2. FK excitation (reduced frequencies and mesh)
    print("\n[2/4] Froude-Krylov excitation...")
    t0 = time.time()
    fk_bem = compute_fk_vs_bem_comparison(
        length=50.0, beam=10.0, draft=4.0,
        omega_values=(0.2, 0.3, 0.4),
        n_length=8, n_beam=4, n_draft=3,
    )
    print(f"  kL range: [{fk_bem[0]['kL']:.3f}, {fk_bem[-1]['kL']:.3f}]")
    errors = [f"{r['magnitude_rel_error']:.2%}" for r in fk_bem]
    print(f"  FK vs BEM errors: {errors}")
    print(f"  Time: {time.time()-t0:.1f}s")
    
    results["fk_excitation"] = fk_bem
    
    # 3. Wigley RAO (model scale L=3m, reduced frequencies)
    print("\n[3/4] Wigley III RAO (model scale L=3m)...")
    t0 = time.time()
    rao = compute_wigley_rao_spectrum(
        omega_values=[0.5, 1.0, 1.5, 2.0, 2.5],
        nx=10, nz=5,
        L=3.0, B=0.3, T=0.1875,
    )
    rao_strs = [f"{r:.3f}" for r in rao.heave_rao_amplitude]
    print(f"  Heave RAO: {rao_strs}")
    print(f"  kL range: [{rao.kL_values[0]:.3f}, {rao.kL_values[-1]:.3f}]")
    print(f"  Long-wave RAO mean: {rao.long_wave_rao_mean:.4f}")
    print(f"  High-freq RAO: {rao.high_freq_rao:.4f}")
    print(f"  Time: {time.time()-t0:.1f}s")
    
    results["wigley_rao"] = {
        "length": rao.length,
        "beam": rao.beam,
        "draft": rao.draft,
        "omegas": rao.omegas.tolist(),
        "kL_values": rao.kL_values.tolist(),
        "heave_rao_amplitude": rao.heave_rao_amplitude.tolist(),
        "heave_rao_phase_deg": rao.heave_rao_phase.tolist(),
        "heave_added_mass": rao.heave_added_mass.tolist(),
        "heave_damping": rao.heave_damping.tolist(),
        "heave_excitation_magnitude": rao.heave_excitation_magnitude.tolist(),
        "long_wave_rao_mean": rao.long_wave_rao_mean,
        "long_wave_rao_std": rao.long_wave_rao_std,
        "high_freq_rao": rao.high_freq_rao,
        "panel_count": rao.panel_count,
    }
    
    # 4. Mesh convergence (2 levels only)
    print("\n[4/4] Mesh convergence...")
    t0 = time.time()
    conv = run_wigley_mesh_convergence(
        omega=1.0,
        nx_values=[8, 12],
        nz_values=[4, 6],
        L=3.0, B=0.3, T=0.1875,
    )
    print(f"  Panel counts: {conv.panel_counts}")
    rao_strs = [f"{r:.4f}" for r in conv.heave_rao_amplitudes]
    print(f"  Heave RAO: {rao_strs}")
    changes = [f"{c:.4f}" if c is not None else "-" for c in conv.relative_changes]
    print(f"  Relative changes: {changes}")
    print(f"  Converged: {conv.converged}")
    print(f"  Time: {time.time()-t0:.1f}s")
    
    results["mesh_convergence"] = {
        "omega": conv.omega,
        "nx_values": conv.nx_values,
        "nz_values": conv.nz_values,
        "panel_counts": conv.panel_counts,
        "heave_rao_amplitudes": conv.heave_rao_amplitudes,
        "relative_changes": conv.relative_changes,
        "max_relative_change": conv.max_relative_change,
        "converged": conv.converged,
    }
    
    # Save results
    json_path = output_dir / "seakeeping_verification_quick.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"Total time: {time.time()-t_total:.1f}s")
    print(f"Results saved to: {json_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
