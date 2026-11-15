"""Probe: locate non-finite entries of the assembled operator vs mode cap."""
import sys
import numpy as np

sys.path.insert(0, "src")
from tensorbem.geometry3d import submerged_sphere_mesh
from tensorbem.finite_depth_solver3d import (
    assemble_finite_depth_operators_3d,
    mode_count_for,
    quadrature_points,
    _modes_for,
)

sub = int(sys.argv[1]) if len(sys.argv) > 1 else 1
h, ka = 5.0, 1.0
nu = ka
mesh = submerged_sphere_mesh(sub, radius=1.0, submergence=2.0)
print("panels", mesh.panel_count, "auto", mode_count_for(mesh, h), flush=True)
for M in (8, 16, 32, 48, 64, 96):
    for kernel in ("eigenseries",):
        try:
            ops = assemble_finite_depth_operators_3d(mesh, nu, h, mode_count=M, kernel=kernel)
            S, D = ops.single_layer, ops.double_layer
            print(f"M={M:3d} {kernel:11s} nan_S={np.isnan(S).sum()} inf_S={np.isinf(S).sum()} "
                  f"nan_D={np.isnan(D).sum()} inf_D={np.isinf(D).sum()} "
                  f"max|S|={np.nanmax(np.abs(S)):.3e} max|D|={np.nanmax(np.abs(D)):.3e}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"M={M:3d} {kernel:11s} FAIL {type(exc).__name__}: {exc}", flush=True)