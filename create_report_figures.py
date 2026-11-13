#!/usr/bin/env python3
"""
Generate figures for the ellipsoid added mass benchmark report.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
with open("artifacts/ellipsoid_added_mass_sub2.json") as f:
    result_sub2 = json.load(f)

# Create figure directory
fig_dir = Path("artifacts/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

# Figure 1: Ellipsoid geometry with mesh
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': '3d'})

# Generate ellipsoid points for smooth surface
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 30)
x = 1.5 * np.outer(np.cos(u), np.sin(v))
y = 1.0 * np.outer(np.sin(u), np.sin(v))
z = 0.75 * np.outer(np.ones_like(u), np.cos(v))

ax.plot_surface(x, y, z, color='lightblue', alpha=0.3, edgecolor='none')

# Add semi-axes labels
ax.quiver(0, 0, 0, 1.5, 0, 0, color='r', linewidth=2, arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, 1.0, 0, color='g', linewidth=2, arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, 0, 0.75, color='b', linewidth=2, arrow_length_ratio=0.1)

ax.text(1.6, 0, 0, 'a=1.5m', color='r', fontsize=12, fontweight='bold')
ax.text(0, 1.1, 0, 'b=1.0m', color='g', fontsize=12, fontweight='bold')
ax.text(0, 0, 0.8, 'c=0.75m', color='b', fontsize=12, fontweight='bold')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Tri-axial Ellipsoid Geometry\nSemi-axes: a=1.5m, b=1.0m, c=0.75m', 
             fontsize=14, fontweight='bold')
ax.set_box_aspect([1.5, 1.0, 0.75])

plt.tight_layout()
plt.savefig(fig_dir / "ellipsoid_geometry.png", dpi=150, bbox_inches='tight')
print("✓ Created ellipsoid_geometry.png")
plt.close()

# Figure 2: Added mass comparison (bar chart)
fig, ax = plt.subplots(figsize=(10, 6))

axes_names = ['X-axis (surge)', 'Y-axis (sway)', 'Z-axis (heave)']
bem_values = [ax_data['added_mass'] for ax_data in result_sub2['axes']]
exact_values = [ax_data['exact_added_mass'] for ax_data in result_sub2['axes']]
errors = [ax_data['added_mass_relative_error'] * 100 for ax_data in result_sub2['axes']]

x = np.arange(len(axes_names))
width = 0.35

bars1 = ax.bar(x - width/2, exact_values, width, label='Lamb (1932) analytical', 
               color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, bem_values, width, label='TensorBEM (320 panels)', 
               color='coral', alpha=0.8)

ax.set_xlabel('Motion axis', fontsize=12, fontweight='bold')
ax.set_ylabel('Added mass (kg)', fontsize=12, fontweight='bold')
ax.set_title('Ellipsoid Added Mass: BEM vs Lamb Analytical Solution', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(axes_names)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add error percentages on bars
for i, (bar, err) in enumerate(zip(bars2, errors)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{err:.2f}%\nerror',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            color='darkred')

plt.tight_layout()
plt.savefig(fig_dir / "added_mass_comparison.png", dpi=150, bbox_inches='tight')
print("✓ Created added_mass_comparison.png")
plt.close()

# Figure 3: Convergence study
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Data from different subdivision levels
subdivisions = [0, 1, 2]
panels = [20, 80, 320]
h_max = [0.9747, 0.4874, 0.2437]  # Approximate values

# Errors for each axis at each level
errors_x = [20.855, 6.162, 1.631]
errors_y = [21.126, 6.473, 1.737]
errors_z = [22.541, 7.020, 1.908]

# Left plot: Error vs panel count
ax1.semilogy(panels, errors_x, 'ro-', linewidth=2, markersize=8, label='X-axis (surge)')
ax1.semilogy(panels, errors_y, 'gs-', linewidth=2, markersize=8, label='Y-axis (sway)')
ax1.semilogy(panels, errors_z, 'b^-', linewidth=2, markersize=8, label='Z-axis (heave)')
ax1.axhline(y=5.0, color='k', linestyle='--', linewidth=2, label='5% tolerance')
ax1.set_xlabel('Number of panels', fontsize=12, fontweight='bold')
ax1.set_ylabel('Relative error (%)', fontsize=12, fontweight='bold')
ax1.set_title('Mesh Convergence: Added Mass Error', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Right plot: Error vs mesh size
ax2.loglog(h_max, errors_x, 'ro-', linewidth=2, markersize=8, label='X-axis')
ax2.loglog(h_max, errors_y, 'gs-', linewidth=2, markersize=8, label='Y-axis')
ax2.loglog(h_max, errors_z, 'b^-', linewidth=2, markersize=8, label='Z-axis')

# Add convergence rate line (slope = 2 for second-order)
h_ref = np.array([1.0, 0.2])
err_ref = 20.0 * (h_ref / h_ref[0])**2
ax2.loglog(h_ref, err_ref, 'k--', linewidth=1.5, alpha=0.5, label='O(h²) reference')

ax2.set_xlabel('Maximum panel size h_max (m)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Relative error (%)', fontsize=12, fontweight='bold')
ax2.set_title('Convergence Rate', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / "convergence_study.png", dpi=150, bbox_inches='tight')
print("✓ Created convergence_study.png")
plt.close()

# Figure 4: Shape factors validation
fig, ax = plt.subplots(figsize=(8, 6))

shape_factors = result_sub2['shape_factors']
axis_labels = ['α₁ (x-axis)', 'α₂ (y-axis)', 'α₃ (z-axis)']
colors = ['steelblue', 'coral', 'seagreen']

bars = ax.bar(axis_labels, shape_factors, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, value in zip(bars, shape_factors):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.6f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add sum annotation
ax.text(0.5, 0.95, f'Sum: {sum(shape_factors):.10f}\n(Exact: 2.0)',
        transform=ax.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top', horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_ylabel('Shape factor value', fontsize=12, fontweight='bold')
ax.set_title("Lamb's Shape Factors for Tri-axial Ellipsoid\n" + 
             r"$\alpha_1 + \alpha_2 + \alpha_3 = 2$ (exact)", 
             fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.2)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / "shape_factors.png", dpi=150, bbox_inches='tight')
print("✓ Created shape_factors.png")
plt.close()

print("\n✓ All figures created successfully in artifacts/figures/")
