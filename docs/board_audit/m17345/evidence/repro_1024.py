"""Audit probe (m17345): 1024-panel level for ka in {0.5, 2.0} - the level the m17278 doc
claims was 'submitted to cluster cj1476' but never delivered. Compares against the same
in-repo multipole reference used by the m17278 study."""
import json, sys, time
import numpy as np
from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.free_surface_solver3d import solve_diffraction_3d
from tensorbem.geometry3d import hemisphere_hull_mesh

DENS, G, R, A = 1025.0, 9.80665, 1.0, 1.0
out = []
for ka in (0.5, 2.0):
    ref = hulme_hemisphere_radiation(ka, radius=R, density=DENS, gravity=G)
    mesh = hemisphere_hull_mesh(4, radius=R)
    t0 = time.time()
    res = solve_diffraction_3d(mesh, nu=ka, amplitude=A, wave_angle=0.0, density=DENS, gravity=G)
    got_x, got_z = abs(res.excitation_force[0]), abs(res.excitation_force[2])
    ref_x, ref_z = abs(ref.excitation_force_x), abs(ref.excitation_force_z)
    out.append({"ka": ka, "panels": int(mesh.faces.shape[0]), "cond": float(res.condition_number),
                "resid": float(res.relative_residual), "wall_s": round(time.time()-t0, 2),
                "Fx": {"ref": ref_x, "got": got_x, "err": (got_x-ref_x)/ref_x},
                "Fz": {"ref": ref_z, "got": got_z, "err": (got_z-ref_z)/ref_z}})
    print(json.dumps(out[-1]))
json.dump(out, open("artifacts/m17345/hulme_1024_probe.json", "w"), indent=1)
