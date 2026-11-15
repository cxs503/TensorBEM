"""Probe: (a) eigenseries convergence for small horizontal separation,
(b) actual minimum target<->quadrature-point horizontal separation on the
cross-validation meshes, and the mode count the solver therefore needs."""
import sys
import numpy as np

sys.path.insert(0, "src")
from tensorbem.geometry3d import submerged_sphere_mesh
from tensorbem.finite_depth3d import green_function_3d
from tensorbem.finite_depth_solver3d import quadrature_points, _modes_for, mode_count_for

nu, h = 0.5, 5.0
print("== (a) kernel convergence vs horizontal separation ==")
for R in (1.0, 0.3, 0.1, 0.03, 0.01):
    tgt = np.array([R, 0.0, -2.3])
    src = np.array([0.0, 0.0, -1.95])
    ref = complex(green_function_3d(tgt, src, nu, h, modes=4000)[()])
    line = [f"R={R:<6} ref={ref.real:.10f}"]
    for M in (32, 64, 128, 256, 512, 1024):
        g = complex(green_function_3d(tgt, src, nu, h, modes=M)[()])
        line.append(f"M{M}:{(g-ref).real:+.2e}")
    print(" ".join(line), flush=True)

print("\n== (b) mesh separations (sub, panels, min centroid->qp horiz, _modes_for) ==")
for sub in (1, 2, 3):
    mesh = submerged_sphere_mesh(sub, radius=1.0, submergence=2.0)
    qpts, _, _ = quadrature_points(mesh)
    horiz = np.hypot(
        mesh.centroids[:, None, 0] - qpts[None, :, 0],
        mesh.centroids[:, None, 1] - qpts[None, :, 1],
    )
    row_min = np.where(horiz > 0.0, horiz, 0.0).min(axis=1)
    auto = mode_count_for(mesh, h)
    need = np.array([_modes_for(h, r) for r in row_min])
    print(f"sub={sub} panels={mesh.panel_count} auto_modes={auto} "
          f"row_min horiz: min={row_min.min():.2e} median={np.median(row_min):.3e} "
          f"max={row_min.max():.2e}", flush=True)
    print(f"   modes needed (uncapped) min={need.min()} median={int(np.median(need))} "
          f"max={need.max()}; rows needing >auto: {(need > auto).sum()}/{need.size}",
          flush=True)