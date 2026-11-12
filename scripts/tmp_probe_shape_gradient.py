import sys, time
sys.path.insert(0, "src")
import numpy as np
from tensorbem.geometry3d import icosphere
from tensorbem.solver3d import assemble_surface_matrices, compute_added_mass_matrix_3d
from tensorbem import shape_gradient as sg

mesh = icosphere(1)  # 80 panels
print("panels", mesh.panel_count, "vertices", mesh.vertex_count)

S_ref, D_ref = assemble_surface_matrices(mesh, quadrature_order=8)
S, D = sg.assemble_quadrature_operators(mesh, order=8)
print("max|S-S_ref|", np.max(np.abs(S - S_ref)), "scale", np.max(np.abs(S_ref)))
print("max|D-D_ref|", np.max(np.abs(D - D_ref)), "scale", np.max(np.abs(D_ref)))
print("rel", np.max(np.abs(S - S_ref))/np.max(np.abs(S_ref)), np.max(np.abs(D - D_ref))/np.max(np.abs(D_ref)))
print("rowsum D", np.max(np.abs(D.sum(axis=1)+0.5)), np.max(np.abs(D_ref.sum(axis=1)+0.5)))

# panel metric derivative vs FD
v = mesh.vertices[mesh.faces[0]]
area, n, dA, dN = sg.panel_metric_derivatives(v)
h = 1e-6
for k in range(3):
    vp = v.copy(); vp[k] += h
    vm = v.copy(); vm[k] -= h
    a_p, n_p, _, _ = sg.panel_metric_derivatives(vp)
    a_m, n_m, _, _ = sg.panel_metric_derivatives(vm)
    for axis in range(3):
        vp = v.copy(); vp[k, axis] += h
        vm = v.copy(); vm[k, axis] -= h
        a_p, n_p, _, _ = sg.panel_metric_derivatives(vp)
        a_m, n_m, _, _ = sg.panel_metric_derivatives(vm)
        print("area d/dv[%d][%d]" % (k, axis), dA[k][axis], (a_p - a_m) / (2*h),
              "  normal max err", np.max(np.abs(dN[k][:, axis] - (n_p - n_m) / (2*h))))

# added mass vs analytic and gradient vs FD
r = 1.0
for sub in (1, 2):
    m = icosphere(sub)
    out = sg.added_mass_gradient(m, dof="surge")
    vol = 4/3*np.pi*r**3
    print(f"P={m.panel_count} A11={out.value:.6f} analytic={0.5*vol:.6f} "
          f"rel={abs(out.value-0.5*vol)/(0.5*vol):.3e} cond={out.condition_number:.2f} "
          f"cost_ratio={out.cost_ratio:.2f} fwd={out.forward_seconds:.3f} grad={out.gradient_seconds:.3f}")

m = icosphere(1)
g = sg.added_mass_gradient(m, dof="surge")
val, fd, h = sg.finite_difference_gradient(m, dof="surge", step=1e-5)
print("value", g.value, val)
print(sg.gradient_agreement(g.gradient, fd))
print("translation invariance", np.abs(g.gradient.sum(axis=0)), "norm", np.linalg.norm(g.gradient))
print("fd translation invariance", np.abs(fd.sum(axis=0)))
# homogeneity: gradient dotted with uniform scaling direction
c = m.vertices.mean(axis=0)
scale_dir = m.vertices - c
pred = 3.0 * g.value / np.mean(np.linalg.norm(scale_dir, axis=1))
meas = np.sum(g.gradient * scale_dir)
print("homogeneity pred", pred, "meas", meas, "rel", abs(meas-pred)/abs(pred))