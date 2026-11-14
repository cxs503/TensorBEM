import time, math
import numpy as np
from tensorbem.geometry3d import submerged_sphere_mesh
from tensorbem.finite_depth_solver3d import solve_radiation_finite_depth_3d, mode_count_for

R = 1.0
depth = 6.0
g = 9.80665
for sub in (1, 2):
    mesh = submerged_sphere_mesh(sub, radius=R, submergence=2.5)
    print("sub", sub, "faces", mesh.panel_count, "auto modes", mode_count_for(mesh, depth))
    for mc in (16, 32):
        k = 1.0
        nu = k * math.tanh(k * depth)
        t0 = time.time()
        res = solve_radiation_finite_depth_3d(mesh, nu, depth, modes=[2], density=1000.0, gravity=g, mode_count=mc)
        print(f"  mc={mc} A={res.added_mass[0,0]:.4f} B={res.damping[0,0]:.4f} cond={res.condition_number:.2e} resid={res.relative_residual:.2e} t={time.time()-t0:.1f}s")
