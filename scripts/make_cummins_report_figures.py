"""Generate paper-quality figures for the Cummins memory kernel closure campaign (m11546)."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import json
import os

from tensorbem.benchmarks import cummins_memory_closure as campaign
from tensorbem import time_domain as td
from tensorbem import causality

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'artifacts')
os.makedirs(OUT, exist_ok=True)

# ── Build the sweep ──────────────────────────────────────────────────────
sweep = campaign.submerged_sphere_sweep(points=150, ka_max=6.0)

# ── Figure 1: Radiation coefficients A(ω) and B(ω) ──────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(sweep.omega, sweep.added_mass, 'b-', linewidth=1.5, label='A(ω) [multipole]')
ax1.axhline(sweep.added_mass_inf, color='r', linestyle='--', linewidth=1.0,
            label=f'A(∞) = {sweep.added_mass_inf:.1f} kg')
ax1.axhline(sweep.added_mass_inf_anchor, color='g', linestyle=':', linewidth=1.0,
            label=f'A(∞) anchor = ρV/2 = {sweep.added_mass_inf_anchor:.1f} kg')
ax1.set_xlabel('ω [rad/s]', fontsize=12)
ax1.set_ylabel('Added mass A(ω) [kg]', fontsize=12)
ax1.set_title('(a) Frequency-dependent added mass', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.plot(sweep.omega, sweep.damping, 'b-', linewidth=1.5, label='B(ω) [multipole]')
ax2.axvline(sweep.damping_peak_omega, color='r', linestyle='--', linewidth=1.0,
            label=f'Peak at ω = {sweep.damping_peak_omega:.2f} rad/s')
ax2.set_xlabel('ω [rad/s]', fontsize=12)
ax2.set_ylabel('Radiation damping B(ω) [kg/s]', fontsize=12)
ax2.set_title('(b) Radiation damping coefficient', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig1_radiation_coefficients.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig1 done")

# ── Figure 2: Memory kernel K(τ) ────────────────────────────────────────
tau = np.arange(0.0, 30.0, 0.01)
K_obj = td.radiation_memory_kernel(sweep.omega, sweep.damping, tau)
K_vals = K_obj(tau)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(tau, K_vals, 'b-', linewidth=1.5, label='K(τ)')
ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlabel('τ [s]', fontsize=12)
ax.set_ylabel('K(τ) [kg/s²]', fontsize=12)
ax.set_title('Radiation memory kernel K(τ) = (2/π) ∫₀^∞ B(ω) cos(ωτ) dω', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 30)

K0 = float(K_obj.value_at_zero)
ax.annotate(f'K(0) = {K0:.1f} kg/s²',
            xy=(0, K0), xytext=(3, K0*0.9),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='red'),
            color='red')
ax.annotate(f'Exponential decay\n(99% by τ ≈ 1.1 s)',
            xy=(1.1, K0*0.01), xytext=(5, K0*0.3),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='green'),
            color='green')

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig2_memory_kernel.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig2 done")

# ── Figure 3: Round-trip closure (B → K → B, A) ────────────────────────
dt = 0.02
tau_max_rt = 200.0
tau_grid = np.arange(0.0, tau_max_rt, dt)
K_rt = td.radiation_memory_kernel(sweep.omega, sweep.damping, tau_grid)
K_rt_vals = K_rt(tau_grid)

omega_recon = sweep.omega.copy()
B_recon = np.zeros_like(omega_recon)
A_recon_minus_ainf = np.zeros_like(omega_recon)
for i, w in enumerate(omega_recon):
    cos_vals = K_rt_vals * np.cos(w * tau_grid)
    sin_vals = K_rt_vals * np.sin(w * tau_grid)
    B_recon[i] = np.trapezoid(cos_vals, tau_grid)
    if w > 0:
        A_recon_minus_ainf[i] = -np.trapezoid(sin_vals, tau_grid) / w

A_recon = A_recon_minus_ainf + sweep.added_mass_inf

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(omega_recon, sweep.damping, 'b-', linewidth=1.5, label='B(ω) original')
ax1.plot(omega_recon, B_recon, 'r--', linewidth=1.2, label='B(ω) from K(τ)')
ax1.set_xlabel('ω [rad/s]', fontsize=12)
ax1.set_ylabel('B(ω) [kg/s]', fontsize=12)
ax1.set_title('(a) Damping round-trip closure', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

B_err = np.abs(B_recon - sweep.damping) / np.maximum(np.abs(sweep.damping), 1e-10)
ax2.semilogy(omega_recon, B_err * 100, 'b-', linewidth=1.2, label='|ΔB/B| (%)')
ax2.axhline(2.0, color='r', linestyle='--', linewidth=1.0, label='2% threshold')
ax2.set_xlabel('ω [rad/s]', fontsize=12)
ax2.set_ylabel('Relative error [%]', fontsize=12)
ax2.set_title('(b) Round-trip error envelope', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(1e-4, 10)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig3_round_trip.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig3 done")

# ── Figure 4: Kramers-Kronig causality check ────────────────────────────
omega_kk = np.linspace(sweep.omega[1]*1.5, sweep.omega[-2]*0.8, 25)
omega_kk_safe = []
min_sep = 0.02 * np.mean(np.diff(sweep.omega))
for w in omega_kk:
    diffs = np.abs(sweep.omega - w)
    if np.min(diffs) > min_sep:
        omega_kk_safe.append(w)
omega_kk_safe = np.array(omega_kk_safe)

A_kk = np.array([causality.kkr_added_mass(sweep.omega, sweep.damping, w,
                                           added_mass_inf=sweep.added_mass_inf)
                 for w in omega_kk_safe])
A_orig_interp = np.interp(omega_kk_safe, sweep.omega, sweep.added_mass)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(sweep.omega, sweep.added_mass - sweep.added_mass_inf, 'b-', linewidth=1.5,
         label='A(ω) - A(∞) [multipole]')
ax1.plot(omega_kk_safe, A_kk - sweep.added_mass_inf, 'r--', linewidth=1.2,
         label='KK prediction from B(ω)')
ax1.set_xlabel('ω [rad/s]', fontsize=12)
ax1.set_ylabel('A(ω) - A(∞) [kg]', fontsize=12)
ax1.set_title('(a) Added mass: data vs Kramers-Kronig', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

denom = np.maximum(np.abs(A_orig_interp - sweep.added_mass_inf), 1e-10)
kk_err = np.abs(A_kk - A_orig_interp) / denom
ax2.semilogy(omega_kk_safe, kk_err * 100, 'b-', linewidth=1.2, label='|ΔA/A| (%)')
ax2.axhline(2.0, color='r', linestyle='--', linewidth=1.0, label='2% threshold')
ax2.set_xlabel('ω [rad/s]', fontsize=12)
ax2.set_ylabel('KK relative error [%]', fontsize=12)
ax2.set_title('(b) Kramers-Kronig causality error', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(1e-3, 100)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig4_kk_causality.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig4 done")

# ── Figure 5: Free decay comparison ─────────────────────────────────────
fd = campaign.criterion_free_decay(sweep, dt_list=(0.02, 0.01, 0.005), duration=40.0)

dt_fine = 0.005
tau_max_fd = 300.0
tau_grid_fd = np.arange(0.0, tau_max_fd, dt_fine)
kernel_fd = td.radiation_memory_kernel(sweep.omega, sweep.damping, tau_grid_fd)

mass = fd['values']['mass']
restoring = fd['values']['restoring']

result = td.cummins_free_decay(
    kernel_fd,
    mass=mass,
    restoring=restoring,
    added_mass_inf=sweep.added_mass_inf,
    initial_displacement=1.0,
    dt=dt_fine,
    duration=40.0,
)

zeta = fd['values']['zeta_decay']
omega_n = sweep.damping_peak_omega
envelope = np.exp(-zeta * omega_n * result.time)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(result.time, result.displacement, 'b-', linewidth=1.0, label='Cummins convolution x(t)')
ax.plot(result.time, envelope, 'r--', linewidth=1.0, label=f'exp(-ζωₙt), ζ = {zeta:.5f}')
ax.plot(result.time, -envelope, 'r--', linewidth=1.0)
ax.set_xlabel('Time t [s]', fontsize=12)
ax.set_ylabel('Displacement x(t) [m]', fontsize=12)
ax.set_title('Free decay from Cummins equation (no external force)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 40)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig5_free_decay.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig5 done")

# ── Figure 6: Error envelope (constant-A approximation) ─────────────────
env = campaign.error_envelope(sweep, ratios=(0.5, 0.75, 1.0, 1.5, 2.0), samples=2001)
ratios = [row['omega_p_over_omega_n'] for row in env['rows']]
amp_errs = [row['peak_amplitude_error'] for row in env['rows']]
phase_errs = [row['phase_error_degrees'] for row in env['rows']]
zeta_errs = [row['damping_ratio_relative_error'] for row in env['rows']]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(ratios, [abs(e)*100 for e in amp_errs], 'bo-', linewidth=1.5, markersize=8,
         label='|Peak amplitude error|')
ax1.plot(ratios, [abs(e) for e in phase_errs], 'rs-', linewidth=1.5, markersize=8,
         label='|Phase error| [deg]')
ax1.axvline(1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5, label='ω_p = ω_n')
ax1.set_xlabel('ω_p / ω_n', fontsize=12)
ax1.set_ylabel('Error', fontsize=12)
ax1.set_title('(a) Constant-A(∞) approximation error', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(ratios, [e*100 for e in zeta_errs], 'go-', linewidth=1.5, markersize=8,
         label='Damping ratio error')
ax2.axhline(5.0, color='r', linestyle='--', linewidth=1.0, label='5% threshold')
ax2.axvline(1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.set_xlabel('ω_p / ω_n', fontsize=12)
ax2.set_ylabel('ζ relative error [%]', fontsize=12)
ax2.set_title('(b) Damping ratio error vs frequency ratio', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig6_error_envelope.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig6 done")

# ── Figure 7: Two-sided kernel (causality visualization) ────────────────
tau_pos = np.arange(0.0, 10.0, 0.01)
K_pos_obj = td.radiation_memory_kernel(sweep.omega, sweep.damping, tau_pos)
K_pos = K_pos_obj(tau_pos)
tau_neg = -tau_pos[::-1]
K_neg = np.zeros_like(tau_neg)
tau_full = np.concatenate([tau_neg, tau_pos[1:]])
K_full = np.concatenate([K_neg, K_pos[1:]])

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(tau_full, K_full, 'b-', linewidth=1.5, label='K(τ)')
ax.axvline(0, color='k', linewidth=0.8, linestyle='-', alpha=0.5)
ax.axhline(0, color='k', linewidth=0.5)
ax.fill_between(tau_neg, 0, K_neg, alpha=0.2, color='red', label='τ < 0: K = 0 (causality)')
ax.set_xlabel('τ [s]', fontsize=12)
ax.set_ylabel('K(τ) [kg/s²]', fontsize=12)
ax.set_title('Two-sided memory kernel: causality requires K(τ < 0) = 0', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-5, 10)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig7_causality.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig7 done")

print("\nAll figures generated successfully in", OUT)
print("Files:", sorted(os.listdir(OUT)))
