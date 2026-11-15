import math, time, json, sys
import numpy as np, capytaine as cpt
from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.geometry3d import submerged_sphere_mesh

g, rho, a, f, V = 9.80665, 1025.0, 1.0, 2.0, 4.0/3.0*math.pi
out = []
def emit(s):
    print(s, flush=True)
    out.append(s)

for ka in (1.0, 5.0):
    om = math.sqrt(g*ka)
    ref = submerged_sphere_wave_multipole(ka, radius=a, submergence=f, density=rho, gravity=g)
    for sub in (3, 4):
        m = submerged_sphere_mesh(sub, radius=a, submergence=f)
        me = cpt.Mesh(vertices=np.asarray(m.vertices, float), faces=np.asarray(m.faces, int))
        body = cpt.FloatingBody(mesh=me, dofs=cpt.rigid_body_dofs())
        for label, gr in (("low_freq", cpt.Delhommeau(gf_singularities='low_freq')),
                          ("both", cpt.Delhommeau(gf_singularities='both'))):
            t0 = time.perf_counter()
            res = {}
            for dof, keys in (("Heave", ("a33", "b33")), ("Surge", ("a11", "b11"))):
                r = cpt.BEMSolver(green_function=gr).solve(cpt.RadiationProblem(
                    body=body, omega=om, water_depth=math.inf, radiating_dof=dof, rho=rho, g=g))
                res[keys[0]] = float(r.added_mass[dof])/(rho*V)
                res[keys[1]] = float(r.radiation_damping[dof])/(rho*om*V)
            ex = {"a33": ref.heave_added_mass_dimless, "b33": ref.heave_damping_dimless,
                  "a11": ref.surge_added_mass_dimless, "b11": ref.surge_damping_dimless}
            dev = {k: (res[k]-ex[k])/abs(ex[k]) for k in ex}
            emit(f"ka={ka} sub={sub} gf={label:8s} " +
                 " ".join(f"{k}={v*100:+.4f}%" for k, v in dev.items()) +
                 f" t={time.perf_counter()-t0:.1f}s")
json.dump(out, open('/tmp/gfprobe.json', 'w'))