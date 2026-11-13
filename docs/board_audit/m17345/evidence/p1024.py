import json, sys, time
from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.free_surface_solver3d import solve_diffraction_3d
from tensorbem.geometry3d import hemisphere_hull_mesh
DENS,G,R,A=1025.0,9.80665,1.0,1.0
kas=[float(x) for x in sys.argv[1:]]
rows=[]
for ka in kas:
    ref=hulme_hemisphere_radiation(ka,radius=R,density=DENS,gravity=G)
    mesh=hemisphere_hull_mesh(4,radius=R); t0=time.time()
    res=solve_diffraction_3d(mesh,nu=ka,amplitude=A,wave_angle=0.0,density=DENS,gravity=G)
    gx,gz=abs(res.excitation_force[0]),abs(res.excitation_force[2])
    rx,rz=abs(ref.excitation_force_x),abs(ref.excitation_force_z)
    rows.append({"ka":ka,"panels":int(mesh.faces.shape[0]),"cond":float(res.condition_number),
      "resid":float(res.relative_residual),"wall_s":round(time.time()-t0,1),
      "Fx_ref":rx,"Fx_got":gx,"Fx_err":(gx-rx)/rx,"Fz_ref":rz,"Fz_got":gz,"Fz_err":(gz-rz)/rz})
    print(json.dumps(rows[-1]),flush=True)
json.dump(rows,open("p1024_%s.json"%("_".join(str(k) for k in kas)),"w"),indent=1)
