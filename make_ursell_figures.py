"""Generate Ursell cylinder radiation benchmark figures."""
import json
import math
import base64
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorbem.analytic import ursell_cylinder_added_mass

# Load convergence data
with open('runs/ursell_cyl_rad_ka06.json') as f:
    conv_data = json.load(f)

# Load ka sweep data
with open('runs/ursell_cyl_rad_ka_sweep.json') as f:
    sweep_data = json.load(f)

# Reference parameters
radius = 1.0
submergence = 2.0
density = 1025.0
gravity = 9.80665

# Figure 1: Convergence study
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

panels = [r['panels'] for r in conv_data]
a_errors = [r['panel_added_mass_relative_error'] for r in conv_data]
b_errors = [r['panel_damping_relative_error'] for r in conv_data]

# Added mass convergence
ax = axes[0]
ax.loglog(panels, a_errors, 'bo-', linewidth=2, markersize=8, label='BEM (panel)')
# Reference line for O(h) convergence
ref_panels = np.array([32, 256])
ref_err = a_errors[0] * (ref_panels[0] / ref_panels)
ax.loglog(ref_panels, ref_err, 'k--', linewidth=1, alpha=0.5, label='O(h) reference')
ax.set_xlabel('Number of panels')
ax.set_ylabel('Relative error (added mass)')
ax.set_title('Convergence: Added Mass (ka=0.6)')
ax.legend()
ax.grid(True, alpha=0.3)

# Damping convergence
ax = axes[1]
ax.loglog(panels, b_errors, 'ro-', linewidth=2, markersize=8, label='BEM (panel)')
ref_err = b_errors[0] * (ref_panels[0] / ref_panels)
ax.loglog(ref_panels, ref_err, 'k--', linewidth=1, alpha=0.5, label='O(h) reference')
ax.set_xlabel('Number of panels')
ax.set_ylabel('Relative error (damping)')
ax.set_title('Convergence: Damping (ka=0.6)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
buf.seek(0)
fig1_base64 = base64.b64encode(buf.read()).decode('utf-8')
plt.close()

# Figure 2: Error vs ka
fig, ax = plt.subplots(figsize=(8, 5))

ka_vals = [r['ka'] for r in sweep_data]
a_errs = [r['panel_added_mass_relative_error'] for r in sweep_data]
b_errs = [r['panel_damping_relative_error'] for r in sweep_data]

ax.plot(ka_vals, a_errs, 'bo-', linewidth=2, markersize=8, label='Added mass error')
ax.plot(ka_vals, b_errs, 'rs-', linewidth=2, markersize=8, label='Damping error')
ax.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='5% tolerance')
ax.axhline(y=0.03, color='g', linestyle='--', alpha=0.5, label='3% tolerance')
ax.set_xlabel('Dimensionless wavenumber ka')
ax.set_ylabel('Relative error')
ax.set_title('Error vs ka (128 panels, a/f=0.5)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
buf.seek(0)
fig2_base64 = base64.b64encode(buf.read()).decode('utf-8')
plt.close()

# Figure 3: Added mass and damping vs ka (reference vs BEM)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ka_fine = np.linspace(0.1, 1.2, 50)
a_ref_fine = []
b_ref_fine = []
for ka in ka_fine:
    omega = math.sqrt(ka * gravity / radius)
    a, b = ursell_cylinder_added_mass(radius, submergence, omega, density, gravity=gravity)
    a_ref_fine.append(a)
    b_ref_fine.append(b)

ax = axes[0]
ax.plot(ka_fine, a_ref_fine, 'k-', linewidth=2, label='Ursell (reference)')
ax.plot(ka_vals, [r['reference_added_mass'] for r in sweep_data], 'ko', markersize=6)
ax.plot(ka_vals, [r['panel_added_mass'] for r in sweep_data], 'bo', markersize=8, label='BEM (128 panels)')
ax.set_xlabel('ka')
ax.set_ylabel('Added mass A [kg/m]')
ax.set_title('Added Mass vs ka')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(ka_fine, b_ref_fine, 'k-', linewidth=2, label='Ursell (reference)')
ax.plot(ka_vals, [r['reference_damping'] for r in sweep_data], 'ko', markersize=6)
ax.plot(ka_vals, [r['panel_damping'] for r in sweep_data], 'ro', markersize=8, label='BEM (128 panels)')
ax.set_xlabel('ka')
ax.set_ylabel('Damping B [kg/(m·s)]')
ax.set_title('Damping vs ka')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
buf.seek(0)
fig3_base64 = base64.b64encode(buf.read()).decode('utf-8')
plt.close()

# Save figures
import os
os.makedirs('docs/figures', exist_ok=True)

with open('docs/figures/ursell_cyl_rad_convergence.png', 'wb') as f:
    f.write(base64.b64decode(fig1_base64))
with open('docs/figures/ursell_cyl_rad_error_vs_ka.png', 'wb') as f:
    f.write(base64.b64decode(fig2_base64))
with open('docs/figures/ursell_cyl_rad_coefficients_vs_ka.png', 'wb') as f:
    f.write(base64.b64decode(fig3_base64))

# Save base64 for report
figures = {
    'convergence': fig1_base64,
    'error_vs_ka': fig2_base64,
    'coefficients_vs_ka': fig3_base64
}
with open('runs/ursell_figures_base64.json', 'w') as f:
    json.dump(figures, f)

print('Figures generated successfully')
print(f'  docs/figures/ursell_cyl_rad_convergence.png')
print(f'  docs/figures/ursell_cyl_rad_error_vs_ka.png')
print(f'  docs/figures/ursell_cyl_rad_coefficients_vs_ka.png')
