#!/usr/bin/env python3
"""
Ellipsoid added mass benchmark: BEM vs Lamb's analytical solution.

This script runs the TensorBEM ellipsoid added mass benchmark for multiple
mesh refinement levels and compares against Lamb's closed-form solution.

Reference: Lamb, H. (1932). Hydrodynamics, 5th ed. Cambridge University Press.
           Section 114, equations for ellipsoid shape factors alpha_i.
           The sum rule: alpha_1 + alpha_2 + alpha_3 = 2 (exact).
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tensorbem.benchmarks.ellipsoid_added_mass import run
from tensorbem.analytic import ellipsoid_shape_factors, ellipsoid_added_mass, ellipsoid_volume

def main():
    # Ellipsoid geometry: tri-axial with semi-axes a=1.5, b=1.0, c=0.75
    semi_axes = (1.5, 1.0, 0.75)
    density = 1025.0  # kg/m^3 (seawater)
    speed = 1.0  # m/s
    
    print("=" * 80)
    print("ELLIPSOID ADDED MASS BENCHMARK: TensorBEM vs Lamb (1932)")
    print("=" * 80)
    print(f"\nGeometry: Tri-axial ellipsoid")
    print(f"  Semi-axes: a={semi_axes[0]}, b={semi_axes[1]}, c={semi_axes[2]} m")
    print(f"  Aspect ratios: a/b={semi_axes[0]/semi_axes[1]:.3f}, b/c={semi_axes[1]/semi_axes[2]:.3f}")
    print(f"  Volume: {ellipsoid_volume(semi_axes):.6f} m^3")
    print(f"  Fluid density: {density} kg/m^3")
    
    # Lamb's shape factors (analytical)
    shape_factors = ellipsoid_shape_factors(semi_axes)
    print(f"\nLamb's shape factors (analytical):")
    print(f"  α₁ = {shape_factors[0]:.10f}")
    print(f"  α₂ = {shape_factors[1]:.10f}")
    print(f"  α₃ = {shape_factors[2]:.10f}")
    print(f"  Sum  = {sum(shape_factors):.10f} (exact: 2.0)")
    
    # Analytical added mass
    exact_added_mass = ellipsoid_added_mass(semi_axes, density)
    print(f"\nAnalytical added mass (Lamb 1932, §114):")
    print(f"  A₁₁ = {exact_added_mass[0]:.6f} kg")
    print(f"  A₂₂ = {exact_added_mass[1]:.6f} kg")
    print(f"  A₃₃ = {exact_added_mass[2]:.6f} kg")
    
    # BEM convergence study
    print("\n" + "=" * 80)
    print("BEM CONVERGENCE STUDY")
    print("=" * 80)
    
    results = []
    for subdivisions in [0, 1, 2]:
        print(f"\n--- Subdivision level {subdivisions} ---")
        result = run(subdivisions, semi_axes, speed, density)
        results.append(result)
        
        panels = result['panels']
        print(f"  Panels: {panels}")
        print(f"  h_max: {result['h_max']:.6f} m")
        print(f"  Volume ratio: {result['volume_ratio']:.6f}")
        
        for ax_data in result['axes']:
            axis = ax_data['motion_axis']
            am_bem = ax_data['added_mass']
            am_exact = ax_data['exact_added_mass']
            rel_err = ax_data['added_mass_relative_error']
            flux_err = ax_data['normal_derivative_relative_l2_error']
            
            print(f"  {axis}-axis: A_{axis}{axis} = {am_bem:.4f} kg "
                  f"(exact: {am_exact:.4f}, error: {rel_err*100:.3f}%)")
            print(f"           flux L2 error: {flux_err*100:.3f}%")
    
    # Save results
    output = {
        "benchmark": "ellipsoid_added_mass_lamb1932",
        "geometry": {
            "semi_axes": list(semi_axes),
            "aspect_ratios": [semi_axes[0]/semi_axes[1], semi_axes[1]/semi_axes[2]],
            "volume": ellipsoid_volume(semi_axes),
        },
        "fluid_density": density,
        "lamb_shape_factors": {
            "alpha_1": float(shape_factors[0]),
            "alpha_2": float(shape_factors[1]),
            "alpha_3": float(shape_factors[2]),
            "sum": float(sum(shape_factors)),
        },
        "analytical_added_mass": {
            "A_11": float(exact_added_mass[0]),
            "A_22": float(exact_added_mass[1]),
            "A_33": float(exact_added_mass[2]),
        },
        "bem_convergence": [],
    }
    
    for result in results:
        bem_data = {
            "subdivisions": result['subdivisions'],
            "panels": result['panels'],
            "h_max": result['h_max'],
            "volume_ratio": result['volume_ratio'],
            "axes": {},
        }
        for ax_data in result['axes']:
            axis = ax_data['motion_axis']
            bem_data["axes"][axis] = {
                "added_mass": ax_data['added_mass'],
                "exact_added_mass": ax_data['exact_added_mass'],
                "relative_error": ax_data['added_mass_relative_error'],
                "flux_l2_error": ax_data['normal_derivative_relative_l2_error'],
            }
        output["bem_convergence"].append(bem_data)
    
    # Write JSON
    output_path = Path("artifacts/ellipsoid_benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    final = results[-1]
    print(f"Finest mesh: {final['panels']} panels (subdivision level {final['subdivisions']})")
    print("\nAdded mass errors (BEM vs Lamb):")
    for ax_data in final['axes']:
        axis = ax_data['motion_axis']
        err = ax_data['added_mass_relative_error']
        status = "✓ PASS" if err < 0.05 else "✗ FAIL"
        print(f"  {axis}-axis: {err*100:.3f}% {status}")
    
    all_pass = all(ax['added_mass_relative_error'] < 0.05 for ax in final['axes'])
    print(f"\nOverall: {'✓ ALL PASS (< 5% tolerance)' if all_pass else '✗ FAIL'}")
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
