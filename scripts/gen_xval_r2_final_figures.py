#!/usr/bin/env python3
"""Generate publication-quality summary figures for XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE.

Uses English labels to avoid CJK font issues.
"""

import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / 'benchmarks' / 'verified' / 'capytaine_crosscheck_r2'
DOCS = ROOT / 'docs'

# Load artifacts
with open(ARTIFACT_DIR / 'results.json') as f:
    results = json.load(f)
with open(ARTIFACT_DIR / 'independent_adjudication.json') as f:
    adjudication = json.load(f)

CASES = ['hemisphere_r1', 'hemisphere_r1_cz-0.5', 'sphere_r1_z-0.5', 'sphere_r1_z-0.5_c0']
CASE_LABELS = ['Hemi R=1\n(ctr)', 'Hemi R=1\n(off-ctr)', 'Sphere R=1\n(ctr)', 'Sphere R=1\n(WL)']
NU_GRID = [0.5, 1.0, 2.0]
DOF_ORDER = ['surge', 'sway', 'heave', 'roll', 'pitch', 'yaw']

def load_case_comparison(case_name):
    comp = results.get('comparison', {})
    return comp.get('cases', {}).get(case_name, {})

def load_adjudication_case(case_name):
    return adjudication.get('cases', {}).get(case_name, {})

# ---------------------------------------------------------------------------
# Figure 1: Summary dashboard (4-panel)
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

# --- Panel 1: Worst active deviation by case ---
ax1 = fig.add_subplot(gs[0, 0])
worst_devs = []
ledger_pass = []
for case in CASES:
    adj = load_adjudication_case(case)
    worst_devs.append(adj.get('worst_active_relative', 0) * 100)
    ledger_pass.append(adj.get('n_over_ledger_band', 0) == 0)

colors = ['#2ecc71' if p else '#e74c3c' for p in ledger_pass]
bars = ax1.bar(range(len(CASES)), worst_devs, color=colors, edgecolor='white', linewidth=0.5)
ax1.axhline(y=15, color='#e74c3c', linestyle='--', linewidth=2, label='Ledger 15% limit')
ax1.axhline(y=10, color='#f39c12', linestyle='--', linewidth=1.5, label='Off-diag 10% band')
ax1.axhline(y=5, color='#3498db', linestyle='--', linewidth=1.5, label='Diagonal 5% band')
ax1.set_xticks(range(len(CASES)))
ax1.set_xticklabels(CASE_LABELS, fontsize=9)
ax1.set_ylabel('Worst active rel. deviation (%)', fontsize=10)
ax1.set_title('(a) Worst Active Deviation by Case', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8, loc='upper right')
ax1.set_ylim(0, 18)
for i, (bar, val) in enumerate(zip(bars, worst_devs)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# --- Panel 2: Entry classification budget ---
ax2 = fig.add_subplot(gs[0, 1])
n_active, n_null, n_invariant, n_conflict = [], [], [], []
for case in CASES:
    adj = load_adjudication_case(case)
    counts = adj.get('counts', {})
    n_active.append(counts.get('active', 0))
    n_null.append(counts.get('null', 0))
    n_invariant.append(counts.get('invariant', 0))
    n_conflict.append(counts.get('conflict', 0))

x = np.arange(len(CASES))
width = 0.2
ax2.bar(x - 1.5*width, n_active, width, label='active', color='#3498db')
ax2.bar(x - 0.5*width, n_null, width, label='null', color='#95a5a6')
ax2.bar(x + 0.5*width, n_invariant, width, label='invariant', color='#2ecc71')
ax2.bar(x + 1.5*width, n_conflict, width, label='conflict', color='#e74c3c')
ax2.set_xticks(x)
ax2.set_xticklabels(CASE_LABELS, fontsize=9)
ax2.set_ylabel('Entry count', fontsize=10)
ax2.set_title('(b) Entry Classification Budget', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)

# --- Panel 3: Deviation by frequency ---
ax3 = fig.add_subplot(gs[1, 0])
nu_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
for ci, case in enumerate(CASES):
    comp = load_case_comparison(case)
    entries = comp.get('entries', [])
    nu_devs = {nu: [] for nu in NU_GRID}
    for e in entries:
        if e.get('class') == 'active':
            nu_devs[e['nu']].append(abs(e.get('relative', 0)) * 100)
    
    means = [np.mean(nu_devs[nu]) if nu_devs[nu] else 0 for nu in NU_GRID]
    ax3.plot(NU_GRID, means, 'o-', color=nu_colors[ci], linewidth=2,
             label=CASE_LABELS[ci].replace('\n', ' '), markersize=6)

ax3.axhline(y=5, color='#3498db', linestyle='--', linewidth=1.5, alpha=0.7)
ax3.axhline(y=10, color='#f39c12', linestyle='--', linewidth=1.5, alpha=0.7)
ax3.axhline(y=15, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7)
ax3.set_xlabel('Frequency nu (m^-1)', fontsize=10)
ax3.set_ylabel('Mean active deviation (%)', fontsize=10)
ax3.set_title('(c) Deviation vs Frequency', fontsize=11, fontweight='bold')
ax3.legend(fontsize=7, loc='upper left')
ax3.set_xticks(NU_GRID)

# --- Panel 4: Ledger acceptance gauge ---
ax4 = fig.add_subplot(gs[1, 1])
ledger = results.get('ledger_acceptance', {})
threshold = ledger.get('threshold', 0.15) * 100
worst = ledger.get('worst_active_relative', 0) * 100
satisfied = ledger.get('satisfied', False)

theta = np.linspace(0, np.pi, 100)
gauge_r = 0.8
ax4.plot(gauge_r * np.cos(theta), gauge_r * np.sin(theta), 'k-', linewidth=2)
ax4.plot([0, gauge_r], [0, 0], 'k-', linewidth=2)

for start, end, color in [(0, 5/18*np.pi, '#2ecc71'), 
                           (5/18*np.pi, 10/18*np.pi, '#f39c12'),
                           (10/18*np.pi, 15/18*np.pi, '#e67e22'),
                           (15/18*np.pi, np.pi, '#e74c3c')]:
    ax4.fill_between([gauge_r * np.cos(start), gauge_r * np.cos(end)],
                     [0, 0], [gauge_r * np.sin(start), gauge_r * np.sin(end)],
                     color=color, alpha=0.3)

needle_angle = np.pi * (worst / 20)
ax4.plot([0, gauge_r * 0.9 * np.cos(needle_angle)],
         [0, gauge_r * 0.9 * np.sin(needle_angle)],
         'r-', linewidth=3)
ax4.plot(0, 0, 'ro', markersize=8)

ax4.text(0, -0.15, f'Worst: {worst:.2f}%', ha='center', fontsize=12, fontweight='bold')
ax4.text(0, -0.3, f'Threshold: {threshold:.0f}%', ha='center', fontsize=10)
ax4.text(0, -0.45, 'PASS' if satisfied else 'FAIL', ha='center', 
         fontsize=14, fontweight='bold', color='#2ecc71' if satisfied else '#e74c3c')

ax4.text(-0.7, 0.5, '0-5%', fontsize=7, ha='center')
ax4.text(-0.3, 0.7, '5-10%', fontsize=7, ha='center')
ax4.text(0.3, 0.7, '10-15%', fontsize=7, ha='center')
ax4.text(0.7, 0.5, '>15%', fontsize=7, ha='center')

ax4.set_xlim(-1.1, 1.1)
ax4.set_ylim(-0.6, 1.1)
ax4.set_aspect('equal')
ax4.axis('off')
ax4.set_title('(d) Ledger Acceptance Gauge', fontsize=11, fontweight='bold')

fig.suptitle('XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE -- Cross-Validation Summary', 
             fontsize=14, fontweight='bold', y=0.98)

outpath = DOCS / 'fig_xval_r2_final_summary.png'
fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {outpath}')

# ---------------------------------------------------------------------------
# Figure 2: Matrix comparison detail (added mass, nu=1, hemisphere_r1)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

DOF_LABELS = ['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw']

comp = load_case_comparison('hemisphere_r1')
entries = comp.get('entries', [])

tb_am = np.zeros((6, 6))
cpt_am = np.zeros((6, 6))
diff_am = np.zeros((6, 6))

for e in entries:
    if e.get('nu') == 1.0 and e.get('quantity') == 'added_mass':
        i = DOF_ORDER.index(e['row'])
        j = DOF_ORDER.index(e['column'])
        tb_am[i, j] = e.get('tensorbem', 0)
        cpt_am[i, j] = e.get('capytaine', 0)
        if cpt_am[i, j] != 0:
            diff_am[i, j] = abs(e.get('relative', 0)) * 100

im0 = axes[0].imshow(tb_am, cmap='Blues', aspect='equal', vmin=0, vmax=1300)
axes[0].set_xticks(range(6))
axes[0].set_yticks(range(6))
axes[0].set_xticklabels(DOF_LABELS, rotation=45, ha='right', fontsize=8)
axes[0].set_yticklabels(DOF_LABELS, fontsize=8)
axes[0].set_title('TensorBEM Added Mass (nu=1)', fontsize=10, fontweight='bold')
for i in range(6):
    for j in range(6):
        val = tb_am[i, j]
        if val > 1:
            axes[0].text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=6, 
                        color='white' if val > 800 else 'black')
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(cpt_am, cmap='Blues', aspect='equal', vmin=0, vmax=1300)
axes[1].set_xticks(range(6))
axes[1].set_yticks(range(6))
axes[1].set_xticklabels(DOF_LABELS, rotation=45, ha='right', fontsize=8)
axes[1].set_yticklabels(DOF_LABELS, fontsize=8)
axes[1].set_title('capytaine Added Mass (nu=1)', fontsize=10, fontweight='bold')
for i in range(6):
    for j in range(6):
        val = cpt_am[i, j]
        if val > 1:
            axes[1].text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=6,
                        color='white' if val > 800 else 'black')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(diff_am, cmap='RdYlGn_r', aspect='equal', vmin=0, vmax=10)
axes[2].set_xticks(range(6))
axes[2].set_yticks(range(6))
axes[2].set_xticklabels(DOF_LABELS, rotation=45, ha='right', fontsize=8)
axes[2].set_yticklabels(DOF_LABELS, fontsize=8)
axes[2].set_title('Relative Deviation (%)', fontsize=10, fontweight='bold')
for i in range(6):
    for j in range(6):
        val = diff_am[i, j]
        if val > 0.01:
            axes[2].text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6)
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

fig.suptitle('hemisphere_r1 Added Mass Matrix Comparison (nu = 1 m^-1)', fontsize=12, fontweight='bold')
plt.tight_layout()

outpath2 = DOCS / 'fig_xval_r2_matrix_detail.png'
fig.savefig(outpath2, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {outpath2}')

print('All figures generated successfully.')
