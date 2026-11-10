#!/usr/bin/env python
"""Quick cathodic protection verification study (m9204).

Runs the essential verification checks:
1. Analytical benchmarks (plate, cylinder)
2. Design case DNV criteria
3. Anode current scale
4. Charge conservation
5. Mesh convergence (coarse/medium/fine)
"""

import sys
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from tensorbem.cp_galvanic import (
    BoxHull,
    box_hull_mesh,
    partition_surface,
    solve_galvanic_protection,
    dnv_envelope,
    grid_convergence_index,
    plate_linear_benchmark,
    cylinder_log_benchmark,
    sphere_exterior_benchmark,
    SEAWATER_CONDUCTIVITY,
    PROTECTION_POTENTIAL,
    OVERPROTECTION_POTENTIAL,
    MESH_LEVELS,
)

output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/cp_verification")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("TensorBEM CP Verification (m9204) - Quick Study")
print("=" * 60)

hull = BoxHull(length=100.0, breadth=20.0, draft=10.0)
criteria = {}

# 1. Analytical benchmarks
print("\n[1] Analytical benchmarks...")
plate = plate_linear_benchmark(n_horizontal=8, n_vertical=2)
print(f"  Plate L2: {plate['potential_rel_l2']:.6e} (< 1%: {plate['potential_rel_l2'] < 0.01})")
criteria["a_plate"] = {
    "metric": "plate L2 error",
    "value": plate["potential_rel_l2"],
    "threshold": 0.01,
    "pass": plate["potential_rel_l2"] < 0.01,
}

cyl = cylinder_log_benchmark(n_theta=32, n_radial=6, n_axial=1)
print(f"  Cylinder L2: {cyl['potential_rel_l2']:.6e} (< 2%: {cyl['potential_rel_l2'] < 0.02})")
criteria["b_cylinder"] = {
    "metric": "cylinder L2 error",
    "value": cyl["potential_rel_l2"],
    "threshold": 0.02,
    "pass": cyl["potential_rel_l2"] < 0.02,
}

sphere = sphere_exterior_benchmark(subdivisions=2)
print(f"  Sphere L2: {sphere['potential_rel_l2']:.6e}")

# 2. Design case DNV criteria
print("\n[2] Design case DNV criteria (beta=0.05, N=4)...")
mesh, groups = box_hull_mesh(hull, *MESH_LEVELS["medium"])
partition = partition_surface(mesh, groups, beta=0.05, n_anodes=4)
result = solve_galvanic_protection(mesh, partition)
envelope = dnv_envelope(result)
print(f"  phi_min={envelope['phi_min']:.4f}V, phi_max={envelope['phi_max']:.4f}V")
print(f"  window_ok={envelope['window_ok']}")
criteria["c_dnv_window"] = {
    "metric": "DNV window",
    "phi_min": envelope["phi_min"],
    "phi_max": envelope["phi_max"],
    "window_ok": envelope["window_ok"],
    "pass": envelope["window_ok"],
}

# 3. Anode current scale
print("\n[3] Anode current scale...")
for beta in [0.01, 0.2]:
    partition = partition_surface(mesh, groups, beta=beta, n_anodes=4)
    result = solve_galvanic_protection(mesh, partition)
    threshold = 10.0 if beta == 0.01 else 200.0
    passed = result.anode_current_total < threshold
    print(f"  beta={beta}: I={result.anode_current_total:.2f}A (< {threshold}A: {passed})")
    if beta == 0.01:
        criteria["d_beta_001"] = {"metric": "I_anode at beta=0.01", "value": result.anode_current_total, "threshold": threshold, "pass": passed}
    else:
        criteria["d_beta_02"] = {"metric": "I_anode at beta=0.2", "value": result.anode_current_total, "threshold": threshold, "pass": passed}

# 4. Charge conservation
print("\n[4] Charge conservation...")
max_imbalance = 0.0
for beta in [0.01, 0.05, 0.1, 0.2]:
    partition = partition_surface(mesh, groups, beta=beta, n_anodes=4)
    result = solve_galvanic_protection(mesh, partition)
    max_imbalance = max(max_imbalance, abs(result.charge_imbalance))
print(f"  Max charge imbalance: {max_imbalance:.2e} (< 0.1%: {max_imbalance < 0.001})")
criteria["f_charge"] = {"metric": "max charge imbalance", "value": max_imbalance, "threshold": 0.001, "pass": max_imbalance < 0.001}

# 5. Mesh convergence
print("\n[5] Mesh convergence...")
conv_results = {}
for name, spec in MESH_LEVELS.items():
    m, g = box_hull_mesh(hull, *spec)
    p = partition_surface(m, g, beta=0.05, n_anodes=4)
    r = solve_galvanic_protection(m, p)
    conv_results[name] = {
        "n_panels": m.panel_count,
        "cell_size_m": float(np.sqrt(np.mean(m.areas))),
        "anode_current_A": r.anode_current_total,
    }
    print(f"  {name}: panels={m.panel_count}, I={r.anode_current_total:.4f}A")

gci = grid_convergence_index(
    conv_results["fine"]["anode_current_A"],
    conv_results["medium"]["anode_current_A"],
    conv_results["coarse"]["anode_current_A"],
    conv_results["fine"]["cell_size_m"],
    conv_results["medium"]["cell_size_m"],
    conv_results["coarse"]["cell_size_m"],
)
gci_pct = gci.get("gci_fine_percent", 100)
print(f"  GCI: {gci_pct:.4f}% (< 2%: {gci_pct < 2.0})")
criteria["e_gci"] = {"metric": "GCI anode current", "value": gci_pct, "threshold": 2.0, "pass": gci_pct < 2.0, "gci_detail": gci}

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
all_pass = all(c.get("pass", False) for c in criteria.values())
for name, c in criteria.items():
    status = "PASS" if c.get("pass") else "FAIL"
    val = c.get("value", "N/A")
    print(f"  {name}: {status} (value={val})")
print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAIL'}")

# Save results
results = {
    "study": "cathodic_protection_verification_m9204",
    "hull": {"length_m": hull.length, "breadth_m": hull.breadth, "draft_m": hull.draft},
    "criteria": criteria,
    "overall_pass": all_pass,
    "mesh_convergence": conv_results,
    "gci": gci,
}
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {output_dir / 'results.json'}")
