import math, time
import numpy as np
import capytaine as cpt
from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.geometry3d import submerged_sphere_mesh

R=1.0; f=1.5; rho=1025.0; g=9.80665
for sub in (0,1,2):
    mesh = submerged_sphere_mesh(sub, radius=R, submergence=f)
    verts=np.asarray(mesh.vertices,float); faces=np.asarray(mesh.faces,int)
    cm = cpt.Mesh(vertices=verts, faces=faces)
    body = cpt.FloatingBody(mesh=cm, dofs=cpt.rigid_body_dofs(rotation_center=(0,0,-f)))
    for depth in (20.0,):
        pb = cpt.RadiationProblem(body=body, wavenumber=1.0, water_depth=depth, radiating_dof="Heave", rho=rho, g=g)
        t0=time.time(); res = cpt.BEMSolver().solve(pb)
        print(f"sub={sub} panels={mesh.panel_count} h={depth} A33={res.added_mass['Heave']:.2f} B33={res.radiation_damping['Heave']:.2f} t={time.time()-t0:.1f}s")
an = submerged_sphere_wave_multipole(1.0, radius=R, submergence=f, density=rho, gravity=g)
print("analytic A33=%.4f B33=%.4f"%(an.added_mass[2,2], an.damping[2,2]))
