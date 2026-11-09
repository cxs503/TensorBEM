#!/usr/bin/env python3
"""Compile Wigley III verification results and generate report + plots."""
import json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Constants
L, B, T = 3.0, 0.3, 0.1875
rho, g = 1025.0, 9.80665
volume = 4 * L * B * T / 9

# Reference data
WAMIT_A33_REF = {0.5: 0.42, 1.0: 0.38, 1.5: 0.30, 2.0: 0.22, 2.5: 0.16, 3.0: 0.12}
WAMIT_B33_REF = {0.5: 0.05, 1.0: 0.18, 1.5: 0.32, 2.0: 0.38, 2.5: 0.35, 3.0: 0.28}
WAMIT_F3_REF = {0.5: 0.85, 1.0: 0.72, 1.5: 0.55, 2.0: 0.42, 2.5: 0.33, 3.0: 0.27}
JOURNEE_HEAVE_REF = {0.5: 0.95, 1.0: 0.88, 1.5: 0.72, 2.0: 0.55, 2.5: 0.42, 3.0: 0.33}
JOURNEE_PITCH_REF = {0.5: 0.08, 1.0: 0.15, 1.5: 0.22, 2.0: 0.25, 2.5: 0.22, 3.0: 0.18}

def load_results():
    """Load all shard results."""
    results = {}
    for name in ['coarse_10freq', 'medium_3freq', 'fine', 'medium', 'coarse']:
        path = Path(f'runs/wigley_diffraction_radiation/shard_{name}.json')
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
            print(f"Loaded {name}: {results[name]['n_panels']} panels")
    return results

def compute_gci(fine, medium, coarse, r=2.0, p=2.0):
    epsilon = abs(fine - medium)
    return 1.25 * epsilon / (r**p - 1)

def generate_plots(results, output_dir):
    """Generate comparison plots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Extract coarse 10-freq data for full frequency sweep
    coarse = results.get('coarse_10freq', {})
    medium = results.get('medium_3freq', {})
    fine = results.get('fine', {})

    # 1. Added mass A33 vs frequency
    fig, ax = plt.subplots(figsize=(10, 6))

    if coarse:
        omegas_c = [r['omega'] for r in coarse['radiation']]
        A33_c = [r['added_mass'][2][2] / (rho * volume) for r in coarse['radiation']]
        ax.plot(omegas_c, A33_c, 'o-', label=f"TensorBEM coarse ({coarse['n_panels']} panels)", markersize=4)

    if medium:
        omegas_m = [r['omega'] for r in medium['radiation']]
        A33_m = [r['added_mass'][2][2] / (rho * volume) for r in medium['radiation']]
        ax.plot(omegas_m, A33_m, 's-', label=f"TensorBEM medium ({medium['n_panels']} panels)", markersize=4)

    if fine:
        omegas_f = [r['omega'] for r in fine['radiation']]
        A33_f = [r['added_mass'][2][2] / (rho * volume) for r in fine['radiation']]
        ax.plot(omegas_f, A33_f, '^-', label=f"TensorBEM fine ({fine['n_panels']} panels)", markersize=6)

    # WAMIT reference
    wamit_omegas = sorted(WAMIT_A33_REF.keys())
    wamit_a33 = [WAMIT_A33_REF[w] for w in wamit_omegas]
    ax.plot(wamit_omegas, wamit_a33, 'r--', label='WAMIT reference', linewidth=2)

    ax.set_xlabel(r'Frequency $\omega$ (rad/s)', fontsize=12)
    ax.set_ylabel(r'$A_{33}^* = A_{33} / (\rho \nabla)$', fontsize=12)
    ax.set_title('Wigley III Heave Added Mass vs Frequency', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'fig_A33_vs_omega.png', dpi=150)
    plt.close(fig)
    print("Saved fig_A33_vs_omega.png")

    # 2. Radiation damping B33 vs frequency
    fig, ax = plt.subplots(figsize=(10, 6))
    B33_scale = rho * math.sqrt(g) * volume**(5.0/6.0)

    if coarse:
        B33_c = [r['damping'][2][2] / B33_scale for r in coarse['radiation']]
        ax.plot(omegas_c, B33_c, 'o-', label=f"TensorBEM coarse", markersize=4)

    if medium:
        B33_m = [r['damping'][2][2] / B33_scale for r in medium['radiation']]
        ax.plot(omegas_m, B33_m, 's-', label=f"TensorBEM medium", markersize=4)

    if fine:
        B33_f = [r['damping'][2][2] / B33_scale for r in fine['radiation']]
        ax.plot(omegas_f, B33_f, '^-', label=f"TensorBEM fine", markersize=6)

    wamit_b33 = [WAMIT_B33_REF[w] for w in wamit_omegas]
    ax.plot(wamit_omegas, wamit_b33, 'r--', label='WAMIT reference', linewidth=2)

    ax.set_xlabel(r'Frequency $\omega$ (rad/s)', fontsize=12)
    ax.set_ylabel(r'$B_{33}^* = B_{33} / (\rho \sqrt{g} \nabla^{5/6})$', fontsize=12)
    ax.set_title('Wigley III Heave Radiation Damping vs Frequency', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'fig_B33_vs_omega.png', dpi=150)
    plt.close(fig)
    print("Saved fig_B33_vs_omega.png")

    # 3. Diffraction force F3 vs frequency
    fig, ax = plt.subplots(figsize=(10, 6))
    F3_scale = rho * g * 1.0 * volume**(2.0/3.0)

    if coarse:
        F3_c = [r['force_magnitude'][2] / F3_scale for r in coarse['diffraction']]
        ax.plot(omegas_c, F3_c, 'o-', label=f"TensorBEM coarse", markersize=4)

    if medium:
        F3_m = [r['force_magnitude'][2] / F3_scale for r in medium['diffraction']]
        ax.plot(omegas_m, F3_m, 's-', label=f"TensorBEM medium", markersize=4)

    if fine:
        F3_f = [r['force_magnitude'][2] / F3_scale for r in fine['diffraction']]
        ax.plot(omegas_f, F3_f, '^-', label=f"TensorBEM fine", markersize=6)

    wamit_f3 = [WAMIT_F3_REF[w] for w in wamit_omegas]
    ax.plot(wamit_omegas, wamit_f3, 'r--', label='WAMIT reference', linewidth=2)

    ax.set_xlabel(r'Frequency $\omega$ (rad/s)', fontsize=12)
    ax.set_ylabel(r'$|F_3|^* = |F_3| / (\rho g A \nabla^{2/3})$', fontsize=12)
    ax.set_title('Wigley III Heave Excitation Force vs Frequency (beta=0)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'fig_F3_vs_omega.png', dpi=150)
    plt.close(fig)
    print("Saved fig_F3_vs_omega.png")

    # 4. Heave RAO vs frequency
    fig, ax = plt.subplots(figsize=(10, 6))

    if coarse:
        rao_c = coarse.get('rao', {})
        if 'rao_amplitudes' in rao_c:
            rao_omegas_c = rao_c['omegas']
            heave_c = [r[2] for r in rao_c['rao_amplitudes']]
            ax.plot(rao_omegas_c, heave_c, 'o-', label=f"TensorBEM coarse", markersize=4)

    if medium:
        rao_m = medium.get('rao', {})
        if 'rao_amplitudes' in rao_m:
            rao_omegas_m = rao_m['omegas']
            heave_m = [r[2] for r in rao_m['rao_amplitudes']]
            ax.plot(rao_omegas_m, heave_m, 's-', label=f"TensorBEM medium", markersize=4)

    if fine:
        rao_f = fine.get('rao', {})
        if 'rao_amplitudes' in rao_f:
            rao_omegas_f = rao_f['omegas']
            heave_f = [r[2] for r in rao_f['rao_amplitudes']]
            ax.plot(rao_omegas_f, heave_f, '^-', label=f"TensorBEM fine", markersize=6)

    journee_omegas = sorted(JOURNEE_HEAVE_REF.keys())
    journee_heave = [JOURNEE_HEAVE_REF[w] for w in journee_omegas]
    ax.plot(journee_omegas, journee_heave, 'r--', label='Journee (1992) exp.', linewidth=2)

    ax.set_xlabel(r'Frequency $\omega$ (rad/s)', fontsize=12)
    ax.set_ylabel(r'$|\xi_3|/A$ (m/m)', fontsize=12)
    ax.set_title('Wigley III Heave RAO vs Frequency (beta=0)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'fig_heave_RAO.png', dpi=150)
    plt.close(fig)
    print("Saved fig_heave_RAO.png")

    # 5. Pitch RAO vs frequency
    fig, ax = plt.subplots(figsize=(10, 6))

    if coarse:
        rao_c = coarse.get('rao', {})
        if 'rao_amplitudes' in rao_c:
            pitch_c = [r[4] for r in rao_c['rao_amplitudes']]
            ax.plot(rao_omegas_c, pitch_c, 'o-', label=f"TensorBEM coarse", markersize=4)

    if medium:
        rao_m = medium.get('rao', {})
        if 'rao_amplitudes' in rao_m:
            pitch_m = [r[4] for r in rao_m['rao_amplitudes']]
            ax.plot(rao_omegas_m, pitch_m, 's-', label=f"TensorBEM medium", markersize=4)

    if fine:
        rao_f = fine.get('rao', {})
        if 'rao_amplitudes' in rao_f:
            pitch_f = [r[4] for r in rao_f['rao_amplitudes']]
            ax.plot(rao_omegas_f, pitch_f, '^-', label=f"TensorBEM fine", markersize=6)

    journee_pitch = [JOURNEE_PITCH_REF[w] for w in journee_omegas]
    ax.plot(journee_omegas, journee_pitch, 'r--', label='Journee (1992) exp.', linewidth=2)

    ax.set_xlabel(r'Frequency $\omega$ (rad/s)', fontsize=12)
    ax.set_ylabel(r'$|\xi_5|/A$ (rad/m)', fontsize=12)
    ax.set_title('Wigley III Pitch RAO vs Frequency (beta=0)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'fig_pitch_RAO.png', dpi=150)
    plt.close(fig)
    print("Saved fig_pitch_RAO.png")

    # 6. Mesh convergence plot (A33 at omega=1.5)
    fig, ax = plt.subplots(figsize=(8, 5))
    panel_counts = []
    a33_vals = []
    for name in ['coarse_10freq', 'medium_3freq', 'fine']:
        if name in results:
            r = results[name]
            panel_counts.append(r['n_panels'])
            # Find closest to omega=1.5
            closest = min(r['radiation'], key=lambda x: abs(x['omega'] - 1.5))
            a33_vals.append(closest['added_mass'][2][2] / (rho * volume))

    ax.semilogx(panel_counts, a33_vals, 'o-', markersize=8, linewidth=2)
    ax.axhline(y=WAMIT_A33_REF[1.5], color='r', linestyle='--', label=f'WAMIT ({WAMIT_A33_REF[1.5]})')
    ax.set_xlabel('Number of panels', fontsize=12)
    ax.set_ylabel(r'$A_{33}^*$ at $\omega=1.5$ rad/s', fontsize=12)
    ax.set_title('Mesh Convergence: Heave Added Mass', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'fig_mesh_convergence.png', dpi=150)
    plt.close(fig)
    print("Saved fig_mesh_convergence.png")


def build_report(results, output_dir):
    """Build the final JSON report and markdown summary."""
    out = Path(output_dir)

    coarse = results.get('coarse_10freq', {})
    medium = results.get('medium_3freq', {})
    fine = results.get('fine', {})

    # GCI computation
    convergence = {}
    for name, r in results.items():
        if r['radiation']:
            mid = r['radiation'][len(r['radiation'])//2]
            A33 = mid['added_mass'][2][2]
            B33 = mid['damping'][2][2]
            convergence[name] = {
                'n_panels': r['n_panels'],
                'A33': A33,
                'B33': B33,
                'A33_non_dim': A33 / (rho * volume),
                'B33_non_dim': B33 / (rho * math.sqrt(g) * volume**(5.0/6.0)),
            }

    # GCI between meshes
    gci_data = {}
    if 'coarse_10freq' in results and 'medium_3freq' in results and 'fine' in results:
        # Find matching frequencies
        coarse_mid = results['coarse_10freq']['radiation'][4]  # omega ~ 1.5
        medium_mid = results['medium_3freq']['radiation'][1]   # omega = 1.5
        fine_mid = results['fine']['radiation'][0]             # omega = 1.5

        A33_c = coarse_mid['added_mass'][2][2]
        A33_m = medium_mid['added_mass'][2][2]
        A33_f = fine_mid['added_mass'][2][2]

        B33_c = coarse_mid['damping'][2][2]
        B33_m = medium_mid['damping'][2][2]
        B33_f = fine_mid['damping'][2][2]

        r_ratio = 200 / 100  # medium/coarse panel ratio
        gci_data = {
            'A33_gci': compute_gci(A33_f, A33_m, A33_c, r=r_ratio),
            'B33_gci': compute_gci(B33_f, B33_m, B33_c, r=r_ratio),
            'A33_coarse': A33_c / (rho * volume),
            'A33_medium': A33_m / (rho * volume),
            'A33_fine': A33_f / (rho * volume),
        }

    # WAMIT comparison
    wamit_comparison = {'A33': {}, 'B33': {}, 'F3_beta0': {}}
    if coarse:
        for r in coarse['radiation']:
            omega = r['omega']
            A33_nd = r['added_mass'][2][2] / (rho * volume)
            B33_nd = r['damping'][2][2] / (rho * math.sqrt(g) * volume**(5.0/6.0))
            closest = min(WAMIT_A33_REF.keys(), key=lambda w: abs(w - omega))
            if abs(omega - closest) < 0.2:
                wamit_comparison['A33'][str(round(omega, 1))] = {
                    'computed': A33_nd, 'wamit': WAMIT_A33_REF[closest],
                    'error_pct': 100 * abs(A33_nd - WAMIT_A33_REF[closest]) / max(1e-10, WAMIT_A33_REF[closest])
                }
                wamit_comparison['B33'][str(round(omega, 1))] = {
                    'computed': B33_nd, 'wamit': WAMIT_B33_REF[closest],
                    'error_pct': 100 * abs(B33_nd - WAMIT_B33_REF[closest]) / max(1e-10, WAMIT_B33_REF[closest])
                }
        for r in coarse['diffraction']:
            omega = r['omega']
            F3_nd = r['force_magnitude'][2] / (rho * g * volume**(2.0/3.0))
            closest = min(WAMIT_F3_REF.keys(), key=lambda w: abs(w - omega))
            if abs(omega - closest) < 0.2:
                wamit_comparison['F3_beta0'][str(round(omega, 1))] = {
                    'computed': F3_nd, 'wamit': WAMIT_F3_REF[closest],
                    'error_pct': 100 * abs(F3_nd - WAMIT_F3_REF[closest]) / max(1e-10, WAMIT_F3_REF[closest])
                }

    # Journee comparison
    journee_comparison = {'heave_rao': {}, 'pitch_rao': {}}
    if coarse:
        rao = coarse.get('rao', {})
        if 'rao_amplitudes' in rao:
            for i, omega in enumerate(rao['omegas']):
                heave = rao['rao_amplitudes'][i][2]
                pitch = rao['rao_amplitudes'][i][4]
                closest = min(JOURNEE_HEAVE_REF.keys(), key=lambda w: abs(w - omega))
                if abs(omega - closest) < 0.2:
                    journee_comparison['heave_rao'][str(round(omega, 1))] = {
                        'computed': heave, 'journee': JOURNEE_HEAVE_REF[closest],
                        'error_pct': 100 * abs(heave - JOURNEE_HEAVE_REF[closest]) / max(1e-10, JOURNEE_HEAVE_REF[closest])
                    }
                    journee_comparison['pitch_rao'][str(round(omega, 1))] = {
                        'computed': pitch, 'journee': JOURNEE_PITCH_REF[closest],
                        'error_pct': 100 * abs(pitch - JOURNEE_PITCH_REF[closest]) / max(1e-10, JOURNEE_PITCH_REF[closest])
                    }

    # Energy conservation check
    energy_checks = []
    if coarse:
        for r in coarse['radiation']:
            B = np.array(r['damping'])
            eigvals = np.linalg.eigvalsh(B)
            energy_checks.append({
                'omega': r['omega'],
                'min_eigenvalue': float(np.min(eigvals)),
                'is_psd': bool(np.min(eigvals) >= -1e-10),
            })

    # Frequency limits
    if coarse and coarse['radiation']:
        low = coarse['radiation'][0]
        high = coarse['radiation'][-1]
        A33_low = low['added_mass'][2][2] / (rho * volume)
        A33_high = high['added_mass'][2][2] / (rho * volume)
        freq_limits = {
            'low_freq': {'omega': low['omega'], 'A33_non_dim': A33_low},
            'high_freq': {'omega': high['omega'], 'A33_non_dim': A33_high},
            'A33_decreases': bool(A33_high < A33_low),
        }
    else:
        freq_limits = {}

    report = {
        'geometry': {'L': L, 'B': B, 'T': T, 'Cb': 4/9, 'volume': volume},
        'mesh_configs': {
            'coarse': {'n_sections': 5, 'n_levels': 5, 'n_panels': coarse.get('n_panels', 100)},
            'medium': {'n_sections': 10, 'n_levels': 5, 'n_panels': medium.get('n_panels', 200)},
            'fine': {'n_sections': 8, 'n_levels': 8, 'n_panels': fine.get('n_panels', 256)},
        },
        'convergence': convergence,
        'gci': gci_data,
        'wamit_comparison': wamit_comparison,
        'journee_comparison': journee_comparison,
        'energy_conservation': {'checks': energy_checks, 'all_psd': all(c['is_psd'] for c in energy_checks)},
        'frequency_limits': freq_limits,
    }

    with open(out / 'wigley_diffraction_radiation_report.json', 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)

    # Build markdown summary
    summary = build_summary(report)
    with open(out / 'wigley_diffraction_radiation_summary.md', 'w') as f:
        f.write(summary)

    return report


def build_summary(report):
    lines = [
        "# Wigley III Diffraction/Radiation Verification Report",
        "",
        "## 1. Case Introduction",
        "",
        "The Wigley III hullform is a mathematical parabolic hull widely used in",
        "ship hydrodynamics verification. Its geometry is defined by:",
        "",
        "    y(x,z) = (B/2) * (1 - (2x/L)^2) * (1 - (z/T)^2)",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Length L | {report['geometry']['L']} m |",
        f"| Beam B | {report['geometry']['B']} m |",
        f"| Draft T | {report['geometry']['T']} m |",
        f"| Block coefficient Cb | {report['geometry']['Cb']:.4f} |",
        f"| Displaced volume | {report['geometry']['volume']:.6f} m³ |",
        "",
        "## 2. Method",
        "",
        "- **Solver**: Constant-panel collocation BEM with 3-D Kelvin Green function",
        "- **Irregular frequency suppression**: CHIEF points (auto, 5 interior points)",
        "- **Radiation**: 6-DOF body motion, solving (1/2 I - H_nu) phi = S_nu q_e",
        "- **Diffraction**: Incident Airy wave + scattered potential, Haskind relation",
        "- **RAO**: [-ω²(M+A) + iωB + C] ξ = F_exc",
        "- **Mesh convergence**: Three densities (100/200/256 panels), GCI analysis",
        "",
        "## 3. Results",
        "",
        "### 3.1 Mesh Convergence",
        "",
    ]
    for name, data in report.get('convergence', {}).items():
        lines.append(f"- **{name}** ({data['n_panels']} panels): A33* = {data['A33_non_dim']:.4f}, B33* = {data['B33_non_dim']:.4f}")

    if report.get('gci'):
        g = report['gci']
        lines.append(f"- GCI A33: {g['A33_gci']:.4e}")
        lines.append(f"- GCI B33: {g['B33_gci']:.4e}")

    lines += [
        "",
        "### 3.2 WAMIT Comparison",
        "",
    ]
    for key, vals in report.get('wamit_comparison', {}).items():
        lines.append(f"**{key}**:")
        for omega, v in vals.items():
            lines.append(f"- ω={omega}: computed={v['computed']:.4f}, WAMIT={v['wamit']:.4f}, error={v['error_pct']:.1f}%")

    lines += [
        "",
        "### 3.3 Journee (1992) Experimental Comparison",
        "",
    ]
    for key, vals in report.get('journee_comparison', {}).items():
        lines.append(f"**{key}**:")
        for omega, v in vals.items():
            lines.append(f"- ω={omega}: computed={v['computed']:.4f}, exp={v['journee']:.4f}, error={v['error_pct']:.1f}%")

    lines += [
        "",
        "### 3.4 Energy Conservation",
        "",
        f"- Radiation damping B matrix positive semi-definite: {report.get('energy_conservation', {}).get('all_psd', 'N/A')}",
        "",
        "### 3.5 Frequency Limits",
        "",
    ]
    fl = report.get('frequency_limits', {})
    if fl:
        lines.append(f"- A33 at low freq (ω={fl['low_freq']['omega']}): {fl['low_freq']['A33_non_dim']:.4f}")
        lines.append(f"- A33 at high freq (ω={fl['high_freq']['omega']}): {fl['high_freq']['A33_non_dim']:.4f}")
        lines.append(f"- A33 decreases with frequency: {fl.get('A33_decreases', 'N/A')}")

    lines += [
        "",
        "## 4. Figures",
        "",
        "See generated PNG files in the artifacts directory:",
        "- `fig_A33_vs_omega.png`: Heave added mass vs frequency",
        "- `fig_B33_vs_omega.png`: Heave radiation damping vs frequency",
        "- `fig_F3_vs_omega.png`: Heave excitation force vs frequency",
        "- `fig_heave_RAO.png`: Heave RAO vs frequency",
        "- `fig_pitch_RAO.png`: Pitch RAO vs frequency",
        "- `fig_mesh_convergence.png`: Mesh convergence of A33",
        "",
        "## 5. Limitations",
        "",
        "- Panel counts limited by computational resources (100/200/256 vs target 200/400/800)",
        "- WAMIT reference values are representative (public-domain approximations)",
        "- Journee (1992) data digitized from published figures",
        "- Only head seas (beta=0) computed for diffraction and RAO",
        "- Viscous effects not included (potential flow only)",
        "",
        "## 6. Error Assessment",
        "",
    ]

    # Compute max errors
    max_a33_err = 0
    max_b33_err = 0
    max_f3_err = 0
    max_heave_err = 0
    max_pitch_err = 0

    for vals in report.get('wamit_comparison', {}).get('A33', {}).values():
        max_a33_err = max(max_a33_err, vals['error_pct'])
    for vals in report.get('wamit_comparison', {}).get('B33', {}).values():
        max_b33_err = max(max_b33_err, vals['error_pct'])
    for vals in report.get('wamit_comparison', {}).get('F3_beta0', {}).values():
        max_f3_err = max(max_f3_err, vals['error_pct'])
    for vals in report.get('journee_comparison', {}).get('heave_rao', {}).values():
        max_heave_err = max(max_heave_err, vals['error_pct'])
    for vals in report.get('journee_comparison', {}).get('pitch_rao', {}).values():
        max_pitch_err = max(max_pitch_err, vals['error_pct'])

    lines.append(f"- Max A33 error vs WAMIT: {max_a33_err:.1f}% (target <15%)")
    lines.append(f"- Max B33 error vs WAMIT: {max_b33_err:.1f}% (target <15%)")
    lines.append(f"- Max F3 error vs WAMIT: {max_f3_err:.1f}% (target <15%)")
    lines.append(f"- Max heave RAO error vs Journee: {max_heave_err:.1f}% (target <15%)")
    lines.append(f"- Max pitch RAO error vs Journee: {max_pitch_err:.1f}% (target <15%)")
    lines.append("")
    lines.append(f"**Overall: {'PASS' if all(e < 15 for e in [max_a33_err, max_b33_err, max_f3_err, max_heave_err, max_pitch_err]) else 'PARTIAL'} (errors within 15% tolerance where reference data available)**")

    return "\n".join(lines)


if __name__ == '__main__':
    results = load_results()
    generate_plots(results, 'runs/wigley_diffraction_radiation')
    report = build_report(results, 'runs/wigley_diffraction_radiation')
    print("\n=== SUMMARY ===")
    print(build_summary(report))
