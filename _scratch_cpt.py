import math, time
import numpy as np
import capytaine as cpt
from tensorbem.geometry3d import submerged_sphere_mesh

mesh = submerged_sphere_mesh(2, radius=1.0, submergence=2.5)
print("faces", mesh.panel_count)
verts = np.asarray(mesh.vertices, float); faces = np.asarray(mesh.faces, int)
cm = cpt.Mesh(vertices=verts, faces=faces)
body = cpt.FloatingBody(mesh=cm, dofs=cpt.rigid_body_dofs(rotation_center=(0,0,-2.5)))
depth = 6.0
g = 9.80665
for k in (0.5, 1.0, 2.0):
    nu = k*math.tanh(k*depth); omega = math.sqrt(g*nu)
    pb = cpt.RadiationProblem(body=body, omega=omega, water_depth=depth, radiating_dof="Heave", rho=1000.0, g=g)
    t0=time.time()
    res = cpt.BEMSolver().solve(pb)
    print(f"k={k} A={res.added_mass['Heave']:.4f} B={res.radiation_damping['Heave']:.4f} t={time.time()-t0:.1f}s")
