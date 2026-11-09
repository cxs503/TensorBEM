"""Generate verification figures for the Wigley III / spheroid study."""
import sys
sys.path.insert(0, '/root/tp-agent/work/m7078')

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
with open('/root/tp-agent/work/m7078/docs/m7078/wigley3_verification.json') as f:
    data = json.load(f)

outdir = Path('/root/tp-agent/work/m7078/docs/m7078/figures')
outdir.mkdir(parents=True, exist_ok=True)

# Extract data
rad = data['radiation']
omegas = [r['omega'] for r in rad]
A33 = [r['A33'] for r in rad]
B33 = [r['B33'] for r in rad]
A44 = [r['A44'] for r in rad]
B44 = [r['B44'] for r in rad]

diff = data['diffraction']
rao = data['rao']

# ============================================================================
# Figure 1: Added mass and damping
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Heave added mass
ax = axes[0, 0]
ax.plot(omegas, A33, 'bo-', linewidth=2, markersize=6, label='BEM (spheroid)')
ax.set_xlabel('ω (rad/s)', fontsize=12)
ax.set_ylabel('A₃₃ (kg)', fontsize=12)
ax.set_title('Heave Added Mass', fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend()

# Heave damping
ax = axes[0, 1]
ax.plot(omegas, B33, 'ro-', linewidth=2, markersize=6, label='BEM (spheroid)')
ax.set_xlabel('ω (rad/s)', fontsize=12)
ax.set_ylabel('B₃₃ (kg/s)', fontsize=12)
ax.set_title('Heave Radiation Damping', fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend()

# Pitch added mass
ax = axes[1, 0]
ax.plot(omegas, A44, 'go-', linewidth=2, markersize=6, label='BEM (spheroid)')
ax.set_xlabel('ω (rad/s)', fontsize=12)
ax.set_ylabel('A₄₄ (kg⋅m²)', fontsize=12)
ax.set_title('Pitch Added Mass', fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend()

# Pitch damping
ax = axes[1, 1]
ax.plot(omegas, B44, 'mo-', linewidth=2, markersize=6, label='BEM (spheroid)')
ax.set_xlabel('ω (rad/s)', fontsize=12)
ax.set_ylabel('B₄₄ (kg⋅m²/s)', fontsize=12)
ax.set_title('Pitch Radiation Damping', fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend()

fig.suptitle('Radiation Forces: Spheroid (Cb=0.543, V=0.0916 m³)', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(outdir / 'fig1_radiation_forces.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {outdir / 'fig1_radiation_forces.png'}")

# ============================================================================
# Figure 2: Diffraction forces vs wave angle
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

beta_values = [0.0, 30.0, 60.0, 90.0]
for idx, beta in enumerate(beta_values):
    ax = axes[idx // 2, idx % 2]
    beta_data = [d for d in diff if d['beta_deg'] == beta]
    omegas_d = [d['omega'] for d in beta_data]
    F3 = [d['F3'] for d in beta_data]
    F5 = [d['F5'] for d in beta_data]
    
    ax.plot(omegas_d, F3, 'b-o', label='|F₃| (heave)', linewidth=2)
    ax.plot(omegas_d, F5, 'r-s', label='|F₅| (pitch)', linewidth=2)
    ax.set_xlabel('ω (rad/s)', fontsize=11)
    ax.set_ylabel('Force (N/m)', fontsize=11)
    ax.set_title(f'β = {beta}°', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

fig.suptitle('Wave Excitation Forces vs Wave Direction', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(outdir / 'fig2_diffraction_forces.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {outdir / 'fig2_diffraction_forces.png'}")

# ============================================================================
# Figure 3: RAO
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

rao_omegas = [r['omega'] for r in rao]
rao3 = [r['rao3'] for r in rao]
rao5 = [r['rao5'] for r in rao]

ax = axes[0]
ax.plot(rao_omegas, rao3, 'bo-', linewidth=2, markersize=8, label='BEM heave RAO')
ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Perfect tracking (|RAO|=1)')
ax.set_xlabel('ω (rad/s)', fontsize=12)
ax.set_ylabel('|RAO₃| (m/m)', fontsize=12)
ax.set_title('Heave RAO (head seas β=0°)', fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim([0, 1.2])

ax = axes[1]
ax.plot(rao_omegas, np.degrees(rao5), 'ro-', linewidth=2, markersize=8, label='BEM pitch RAO')
ax.set_xlabel('ω (rad/s)', fontsize=12)
ax.set_ylabel('|RAO₅| (deg/m)', fontsize=12)
ax.set_title('Pitch RAO (head seas β=0°)', fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend()

fig.suptitle('Response Amplitude Operators', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(outdir / 'fig3_rao.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {outdir / 'fig3_rao.png'}")

# ============================================================================
# Figure 4: Non-dimensional coefficients
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Non-dimensionalize: A33' = A33 / (rho * V), B33' = B33 / (rho * omega * V)
V = data['geometry']['volume']
rho = 1025.0

A33_nd = [a / (rho * V) for a in A33]
B33_nd = [b / (rho * om * V) for b, om in zip(B33, omegas)]

ax = axes[0]
ax.plot(omegas, A33_nd, 'bo-', linewidth=2, markersize=6)
ax.set_xlabel('ω (rad/s)', fontsize=12)
ax.set_ylabel('A₃₃ / (ρV)', fontsize=12)
ax.set_title('Non-dimensional Heave Added Mass', fontsize=13)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(omegas, B33_nd, 'ro-', linewidth=2, markersize=6)
ax.set_xlabel('ω (rad/s)', fontsize=12)
ax.set_ylabel('B₃₃ / (ρωV)', fontsize=12)
ax.set_title('Non-dimensional Heave Damping', fontsize=13)
ax.grid(True, alpha=0.3)

fig.suptitle('Non-dimensional Radiation Coefficients', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(outdir / 'fig4_nondimensional.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {outdir / 'fig4_nondimensional.png'}")

print("\nAll figures generated successfully.")
