import sys
sys.path.insert(0, "src")
import numpy as np
from tensorbem.geometry3d import TriangleMesh, icosphere
from tensorbem import shape_gradient as sg

mesh = icosphere(0)  # 20 panels, 12 vertices
P = mesh.panel_count
k, axis = 3, 0
dS, dD = sg.vertex_operator_derivative(mesh, k, order=8)
h = 1e-6
vp = mesh.vertices.copy(); vp[k, axis] += h
vm = mesh.vertices.copy(); vm[k, axis] -= h
Sp, Dp = sg.assemble_quadrature_operators(TriangleMesh(vp, mesh.faces))
Sm, Dm = sg.assemble_quadrature_operators(TriangleMesh(vm, mesh.faces))
fdS = (Sp - Sm) / (2 * h)
fdD = (Dp - Dm) / (2 * h)
print("dS err", np.max(np.abs(dS[:, :, axis] - fdS)), "scale", np.max(np.abs(fdS)))
print("dD err", np.max(np.abs(dD[:, :, axis] - fdD)), "scale", np.max(np.abs(fdD)))
i, j = np.unravel_index(np.argmax(np.abs(dS[:, :, axis] - fdS)), (P, P))
print("worst dS at", i, j, dS[i, j, axis], fdS[i, j])
i, j = np.unravel_index(np.argmax(np.abs(dD[:, :, axis] - fdD)), (P, P))
print("worst dD at", i, j, dD[i, j, axis], fdD[i, j])

# which rows/cols are bad?
err_S = np.max(np.abs(dS[:, :, axis] - fdS), axis=1)
print("bad rows (S):", np.argsort(err_S)[-5:], err_S[np.argsort(err_S)[-5:]])
err_D = np.max(np.abs(dD[:, :, axis] - fdD), axis=1)
print("bad rows (D):", np.argsort(err_D)[-5:], err_D[np.argsort(err_D)[-5:]])
err_Dc = np.max(np.abs(dD[:, :, axis] - fdD), axis=0)
print("bad cols (D):", np.argsort(err_Dc)[-5:], err_Dc[np.argsort(err_Dc)[-5:]])