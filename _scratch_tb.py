import math, time
import numpy as np
from tensorbem.geometry3d import submerged_sphere_mesh
from tensorbem.finite_depth_solver3d import solve_radiation_finite_depth_3d
R=1.0; f=1.5; rho=1025.0; g=9.80665
for sub in (1,2):
    mesh = submerged_sphere_mesh(sub, radius=R, submergence=f)
    for mc in (32,64):
        t0=time.time()
        r = solve_radiation_finite_depth_3d(mesh, 1.0, 20.0, modes=[0,1,2], density=rho, gravity=g, mode_count=mc)
        print(f"sub={sub} panels={mesh.panel_count} mc={mc} A11={r.added_mass[0,0]:.2f} A33={r.added_mass[2,2]:.2f} B11={r.damping[0,0]:.2f} B33={r.damping[2,2]:.2f} cond={r.condition_number:.2f} t={time.time()-t0:.1f}s")
