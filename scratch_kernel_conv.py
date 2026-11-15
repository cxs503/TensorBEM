"""Probe: kernel-level partial-sum convergence of the John eigenseries."""
import sys
import numpy as np

sys.path.insert(0, "src")
from tensorbem.finite_depth3d import green_function_3d, depth_modes, bessel_k0
from tensorbem.finite_depth import evanescent_wavenumbers, mode_norms
from tensorbem.finite_depth3d import polish_evanescent_roots

nu = 0.5          # omega^2/g
h = 5.0           # depth
k0 = None
from tensorbem.finite_depth import finite_depth_wavenumber
k0 = finite_depth_wavenumber(nu, h)
print("nu", nu, "h", h, "k0", k0, "k0*h", k0 * h)

# one target on the sphere bottom-ish and source on the same sphere
tgt = np.array([0.10, 0.0, -2.30])
src = np.array([0.0, 0.05, -1.95])
R = np.hypot(tgt[0] - src[0], tgt[1] - src[1])
print("horizontal R", R, "3D r", np.linalg.norm(tgt - src))

for M in (16, 32, 48, 64, 96, 128, 192, 256, 512):
    g = complex(green_function_3d(tgt, src, nu, h, modes=M)[()])
    print(f"M={M:4d} G={g.real:+.12e} {g.imag:+.12e}")

# evanescent roots and norms sanity
mu = polish_evanescent_roots(evanescent_wavenumbers(nu, h, 12), nu, h)
print("mu", mu)
print("residual mu*tan(mu*h)+nu", mu * np.tan(mu * h) + nu)
print("norms", mode_norms(nu, h, mu))