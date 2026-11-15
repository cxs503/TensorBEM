"""Probe: A33/B33 vs mode_count on a small sphere mesh (truncation convergence)."""
import sys, time
import numpy as np

sys.path.insert(0, "src")
from tensorbem.geometry3d import submerged_sphere_mesh
from tensorbem.finite_depth_solver3d import (
    assemble_finite_depth_operators_3d,
    solve_radiation_finite_depth_3d,
    mode_count_for,
)

R = 1.0
SUB = 2.0
sub = int(sys.argv[1]) if len(sys.argv) > 1 else 1
mesh = submerged_sphere_mesh(sub, radius=R, submergence=SUB)
print("sub", sub, "panels", mesh.panel_count, flush=True)
for depth_ratio, ka in ((5.0, 1.0), (12.0, 0.3), (12.0, 1.0)):
    h = depth_ratio * R
    nu = ka / R
    auto = mode_count_for(mesh, h)
    print(f"h/R={depth_ratio} ka={ka} auto_modes={auto}", flush=True)
    for M in (16, 24, 32, 48, 64, 96, 128):
        t0 = time.time()
        try:
            res = solve_radiation_finite_depth_3d(
                mesh, nu, h, modes=[0, 2], mode_count=M, density=1025.0
            )
            print(f"  M={M:4d} am_h={res.added_mass_heave:.6g} b_h={res.damping_heave:.6g} "
                  f"t={time.time()-t0:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  M={M:4d} FAIL {type(exc).__name__}: {exc} t={time.time()-t0:.1f}s", flush=True)