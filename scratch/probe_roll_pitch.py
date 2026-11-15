"""Feasibility probe: can C5 (free decay) be scored on roll and pitch?"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from tensorbem import time_domain as td
from tensorbem.benchmarks import case_bem_cummins_wigley as case

L, B, T = case.LENGTH, case.BEAM, case.DRAFT
rho, g = case.DENSITY, case.GRAVITY
vol = case.VOLUME_ANALYTIC
m = rho * vol
z_b = -3.0 * T / 8.0
KB = z_b + T
KG = 0.5 * T
i_xx_wp = 4.0 * B**3 * L / 105.0
i_yy_wp = B * L**3 / 30.0
BM_T, BM_L = i_xx_wp / vol, i_yy_wp / vol
GM_T, GM_L = KB + BM_T - KG, KB + BM_L - KG
C44 = rho * g * vol * GM_T
C55 = rho * g * vol * GM_L

x = np.linspace(-L / 2, L / 2, 400)
z = np.linspace(-T, 0.0, 200)
X, Z = np.meshgrid(x, z, indexing="ij")
y = 0.5 * B * (1.0 - (2.0 * X / L) ** 2) * (1.0 - (Z / T) ** 2)
dx, dz = x[1] - x[0], z[1] - z[0]
vol_num = float(np.sum(2.0 * y) * dx * dz)
i_xx = rho * float(np.sum((2.0 / 3.0) * y**3 + 2.0 * y * Z**2) * dx * dz)
i_yy = rho * float(np.sum(2.0 * y * X**2 + 2.0 * y * Z**2) * dx * dz)
print(f"vol_numeric={vol_num:.6f} vs {vol:.6f}  m={m:.4f} kg")
print(f"Ixx={i_xx:.5f}  Iyy={i_yy:.5f} kg m^2  (m*B^2/12={m*B**2/12:.5f})")
print(f"GM_T={GM_T:.5f} GM_L={GM_L:.4f}  C44={C44:.3f} C55={C55:.1f}")

sweep = case.load_sweep("benchmarks/verified/casepool_bem_cummins_wigley/sweep_medium.npz")
C33 = rho * g * (2.0 * L * B / 3.0)

for dof, inertia, stiff in ((2, m, C33), (3, i_xx, C44), (4, i_yy, C55)):
    A = sweep.added_mass[:, dof, dof]
    Bdw = sweep.damping[:, dof, dof]
    print(f"\n--- dof={case.DOF_NAMES[dof]}  minB={Bdw.min():.4g} maxB={Bdw.max():.4g}")
    if Bdw.min() < 0:
        print("negative damping -> undecidable")
        continue
    closure = case.kk_added_mass_closure(sweep.omega, A, Bdw)
    a_inf = float(closure.added_mass_inf)
    wn = math.sqrt(stiff / (inertia + a_inf))
    for _ in range(50):
        A_at = float(np.interp(wn, sweep.omega, A))
        wn = math.sqrt(stiff / (inertia + A_at))
    b_at = float(np.interp(wn, sweep.omega, Bdw))
    zeta_fd = b_at / (2.0 * (inertia + A_at) * wn)
    print(
        f"A_inf_fit={a_inf:.4g} A(wn)={A_at:.4g} wn={wn:.4f} rad/s "
        f"in_band={sweep.omega[0] <= wn <= sweep.omega[-1]} zeta_fd={zeta_fd:.5f}"
    )
    if not (sweep.omega[0] <= wn <= sweep.omega[-1]):
        continue
    t0 = time.time()
    tau = np.arange(0.0, case.TAU_MAX + 0.5 * case.DT, case.DT)
    kernel = td.radiation_memory_kernel(sweep.omega, Bdw, tau)
    decay = td.cummins_free_decay(
        kernel, mass=inertia, restoring=stiff, added_mass_inf=a_inf,
        initial_displacement=1.0, dt=case.DT, duration=case.DURATION,
    )
    zeta, period, npeaks = case._decay_zeta(
        np.asarray(decay.time), np.asarray(decay.displacement)
    )
    rel = abs(zeta - zeta_fd) / zeta_fd if zeta_fd else float("nan")
    print(
        f"zeta_decay={zeta:.5f} period={period:.4f} npeaks={npeaks} "
        f"rel={rel:.3%} elapsed={time.time() - t0:.1f}s"
    )