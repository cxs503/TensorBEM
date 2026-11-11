"""Run a partial study with key ka values to produce real artifacts."""
import sys, json, time
sys.path.insert(0, 'src')
from pathlib import Path
from tensorbem.benchmarks.drift_far_field_3d_study import run_case, radius_study
from tensorbem.drift_far_field3d import local_loglog_slope, slope_band, panel_richardson

# Key ka values that demonstrate the physics
ka_values = [0.3, 1.0, 2.0]
levels = [("base", 16, 6, 1.35), ("fine", 32, 12, 1.25)]

print(f"Running partial study with ka={ka_values}")
t0 = time.time()

cases = {}
for name, n_theta, n_z, growth in levels:
    print(f"  Level {name}: {n_theta}x{n_z}...")
    cases[name] = []
    for ka in ka_values:
        print(f"    ka={ka}...", end=" ", flush=True)
        result = run_case(ka, n_theta, n_z, growth, n_azimuth=90)
        cases[name].append(result)
        print(f"done ({result['seconds']:.1f}s)")

# Radius scan at ka=1.0
print("  Radius scan at ka=1.0...")
rad_study = radius_study(ka=1.0, n_theta=16, n_z=6, growth=1.35, n_azimuth=90)

# Panel ladder at ka=1.0
print("  Panel ladder at ka=1.0...")
ladder = [(16, 6, 1.35), (24, 9, 1.30), (32, 12, 1.25)]
ladder_rows = [run_case(1.0, nt, nz, g, n_azimuth=90) for nt, nz, g in ladder]

elapsed = time.time() - t0
print(f"Completed in {elapsed:.1f}s")

# Assemble study dict
study = {
    "geometry": {
        "body": "bottom-mounted vertical cylinder",
        "radius_m": 1.0,
        "depth_policy": "5/ka",
        "density_kg_m3": 1025.0,
        "gravity_m_s2": 9.81,
        "wave_amplitude_m": 1.0,
        "wave_direction_rad": 0.0,
    },
    "levels": levels,
    "ka_values": ka_values,
    "cases": cases,
    "radius_scan": rad_study,
    "ladder_ka": 1.0,
    "ladder": ladder_rows,
    "seconds": elapsed,
}

# Compute slopes
for level_name in ["base", "fine"]:
    ka_list = [c["ka"] for c in cases[level_name]]
    far_list = [c["far_field_x"] for c in cases[level_name]]
    exact_list = [c["exact_far_x"] for c in cases[level_name]]
    study[f"{level_name}_local_slope"] = local_loglog_slope(ka_list, far_list)
    study[f"exact_local_slope_{level_name}"] = local_loglog_slope(ka_list, exact_list)

# Panel Richardson
panels = [row["panels"] for row in ladder_rows]
far_ladder = [row["far_field_x"] for row in ladder_rows]
try:
    rich = panel_richardson(panels, far_ladder)
    study["panel_richardson"] = rich
except Exception as e:
    study["panel_richardson"] = {"error": str(e)}

# Gates (simplified)
fine = cases["fine"]
gates = {
    "near_vs_far_within_3pct_fine_mesh": {
        "pass": all(c["near_vs_far_gap"] <= 0.03 for c in fine),
        "max_gap": max(c["near_vs_far_gap"] for c in fine),
    },
    "far_vs_exact_within_3pct_fine_mesh": {
        "pass": all(c["far_vs_exact"] <= 0.03 for c in fine),
        "max_error": max(c["far_vs_exact"] for c in fine),
    },
    "energy_identity_residual_le_1e-3": {
        "pass": all(c["energy_residual"] <= 1e-3 for c in fine),
        "max_residual": max(c["energy_residual"] for c in fine),
    },
}
study["gates"] = gates

# Save
outdir = Path("artifacts/drift_far_field_3d")
outdir.mkdir(parents=True, exist_ok=True)
(outdir / "study.json").write_text(json.dumps(study, indent=2, default=float))
print(f"Saved to {outdir}/study.json")
print("Gates:", json.dumps(gates, indent=2, default=float))
