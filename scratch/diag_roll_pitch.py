"""Diagnose the roll decay discrepancy: which numerical knob moves zeta_decay?"""
import math
import sys

import numpy as np

sys.path.insert(0, "src")
from tensorbem import time_domain as td
from tensorbem.benchmarks import case_bem_cummins_wigley as case

L, B, T = case.LENGTH, case.BEAM, case.DRAFT
rho, g = case.DENSITY, case.GRAVITY
vol = case.VOLUME_ANALYTIC
m = rho * vol
KB = -3.0 * T / 8.0 + T
KG = 0.5 * T
GM_T = KB + (4.0 * B**3 * L / 105.0) / vol - KG
GM_L = KB + (B * L**3 / 30.0) / vol - KG
C44 = rho * g * vol * GM_T
C55 = rho * g * vol * GM_L

x = np.linspace(-L / 2, L / 2, 400)
z = np.linspace(-T, 0.0, 200)
X, Z = np.meshgrid(x, z, indexing="ij")
y = 0.5 * B * (1.0 - (2.0 * X / L) ** 2) * (1.0 - (Z / T) ** 2)
dx, dz = x[1] - x[0], z[1] - z[0]
i_xx = rho * float(np.sum((2.0 / 3.0) * y**3 + 2.0 * y * Z**2) * dx * dz)
i_yy = rho * float(np.sum(2.0 * y * X**2 + 2.0 * y * Z**2) * dx * dz)

sweep = case.load_sweep("benchmarks/verified/casepool_bem_cummins_wigley/sweep_medium.npz")


def measure(dof, inertia, stiff, *, dt, duration, tau_max, low_model="cubic", a_inf_mode="fit",
            floor=case.DECAY_AMPLITUDE_FLOOR):
    A = sweep.added_mass[:, dof, dof]
    Bd = sweep.damping[:, dof, dof]
    closure = case.kk_added_mass_closure(sweep.omega, A, Bd)
    a_inf = float(closure.added_mass_inf) if a_inf_mode == "fit" else float(a_inf_mode)
    wn = math.sqrt(stiff / (inertia + a_inf))
    for _ in range(80):
        wn = math.sqrt(stiff / (inertia + float(np.interp(wn, sweep.omega, A))))
    a_at = float(np.interp(wn, sweep.omega, A))
    b_at = float(np.interp(wn, sweep.omega, Bd))
    zeta_fd = b_at / (2.0 * (inertia + a_at) * wn)
    tau = np.arange(0.0, tau_max + 0.5 * dt, dt)
    kernel = td.radiation_memory_kernel(sweep.omega, Bd, tau, low_frequency_model=low_model)
    decay = td.cummins_free_decay(
        kernel, mass=inertia, restoring=stiff, added_mass_inf=a_inf,
        initial_displacement=1.0, dt=dt, duration=duration,
    )
    zeta, period, npeaks = case._decay_zeta(
        np.asarray(decay.time), np.asarray(decay.displacement), amplitude_floor=floor
    )
    return zeta, zeta_fd, period, npeaks, wn, a_at, b_at


base = dict(dt=case.DT, duration=case.DURATION, tau_max=case.TAU_MAX)
print("=== roll sensitivity ===")
for label, kw in [
    ("baseline", base),
    ("dt=0.0005", {**base, "dt": 0.0005}),
    ("duration=80", {**base, "duration": 80.0}),
    ("tau_max=60", {**base, "tau_max": 60.0}),
    ("low_model=zero", {**base, "low_model": "zero"}),
    ("floor=1e-3", {**base, "floor": 1.0e-3}),
]:
    z, zf, p, n, wn, aa, ba = measure(3, i_xx, C44, **kw)
    print(f"{label:16s} zeta_decay={z:.5f} zeta_fd={zf:.5f} rel={abs(z-zf)/zf:7.3%} period={p:.4f} peaks={n} wn={wn:.3f} A={aa:.4f} B={ba:.4f}")

print("=== pitch sensitivity ===")
for label, kw in [("baseline", base), ("dt=0.0005", {**base, "dt": 0.0005}), ("low_model=zero", {**base, "low_model": "zero"})]:
    z, zf, p, n, wn, aa, ba = measure(4, i_yy, C55, **kw)
    print(f"{label:16s} zeta_decay={z:.5f} zeta_fd={zf:.5f} rel={abs(z-zf)/zf:7.3%} period={p:.4f} peaks={n} wn={wn:.3f}")

print("=== roll: does inertia choice matter (it should not) ===")
for scale in (0.5, 1.0, 2.0, 4.0):
    z, zf, p, n, wn, aa, ba = measure(3, scale * i_xx, C44, **base)
    print(f"Ixx x{scale:4.1f} zeta_decay={z:.5f} zeta_fd={zf:.5f} rel={abs(z-zf)/zf:7.3%} wn={wn:.3f}")

print("=== roll: check A(omega) dispersion across the band (relative to A(wn)) ===")
A = sweep.added_mass[:, 3, 3]
print("omega:", np.array2string(sweep.omega, precision=2))
print("A44  :", np.array2string(A, precision=3))
print("B44  :", np.array2string(sweep.damping[:, 3, 3], precision=3))
A4 = sweep.added_mass[:, 4, 4]
print("A55  :", np.array2string(A4, precision=3))
print("B55  :", np.array2string(sweep.damping[:, 4, 4], precision=3))

print("=== synthetic: constant-B oscillator through the same kernel pipeline ===")
# a damping curve that is flat near resonance isolates the kernel/dispersion effect
for dof, inertia, stiff in ((3, i_xx, C44), (4, i_yy, C55)):
    Bd = sweep.damping[:, dof, dof].copy()
    A = sweep.added_mass[:, dof, dof]
    flat = np.full_like(Bd, float(np.interp(7.0, sweep.omega, Bd)))
    closure = case.kk_added_mass_closure(sweep.omega, A, Bd)
    a_inf = float(closure.added_mass_inf)
    wn = math.sqrt(stiff / (inertia + a_inf))
    for _ in range(80):
        wn = math.sqrt(stiff / (inertia + float(np.interp(wn, sweep.omega, A))))
    a_at = float(np.interp(wn, sweep.omega, A))
    b_at = float(np.interp(wn, sweep.omega, flat))
    zeta_fd = b_at / (2.0 * (inertia + a_at) * wn)
    tau = np.arange(0.0, case.TAU_MAX + 0.5 * case.DT, case.DT)
    kernel = td.radiation_memory_kernel(sweep.omega, flat, tau)
    decay = td.cummins_free_decay(kernel, mass=inertia, restoring=stiff, added_mass_inf=a_inf,
                                  initial_displacement=1.0, dt=case.DT, duration=case.DURATION)
    z, p, n = case._decay_zeta(np.asarray(decay.time), np.asarray(decay.displacement))
    print(f"{case.DOF_NAMES[dof]:6s} flat-B: zeta_decay={z:.5f} zeta_fd={zeta_fd:.5f} rel={abs(z-zeta_fd)/zeta_fd:7.3%} period={p:.4f}")