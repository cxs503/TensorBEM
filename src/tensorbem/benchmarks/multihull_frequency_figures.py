"""Generate publication-quality figures for multihull frequency-dependence study."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

# Ensure we use the local source
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter

from tensorbem.benchmarks.multihull_frequency_dependence import (
    distance_decay_study,
    frequency_scan,
    negative_control_coincident,
    single_cylinder_heave_coefficients,
    bem_cross_validation,
    verify_asymptotics,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "multihull_frequency"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def fig1_frequency_scan():
    """A(omega) and B(omega) for single hull and catamaran."""
    radius, depth, sep = 0.3, 0.8, 1.5
    rows = frequency_scan(radius, depth, sep)

    ka = np.array([r.ka for r in rows])
    A_s = np.array([r.A33_single for r in rows])
    B_s = np.array([r.B33_single for r in rows])
    A_c = np.array([r.A33_catamaran for r in rows])
    B_c = np.array([r.B33_catamaran for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Added mass
    ax1.semilogx(ka, A_s, 'b-', lw=2, label='Single hull $A_{33}$')
    ax1.semilogx(ka, A_c / 2, 'r--', lw=2, label='Catamaran $A_{33}/2$ (per hull)')
    ax1.axhline(1.0, color='k', ls=':', lw=1, label='$A_{geom}$ (geometric limit)')
    ax1.set_xlabel('$ka$ (wavenumber × radius)')
    ax1.set_ylabel('$A_{33} / (\\rho \\pi a^2)$')
    ax1.set_title('(a) Added Mass vs Frequency')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(ka.min(), ka.max())

    # Damping
    ax2.semilogx(ka, B_s, 'b-', lw=2, label='Single hull $B_{33}$')
    ax2.semilogx(ka, B_c / 2, 'r--', lw=2, label='Catamaran $B_{33}/2$ (per hull)')
    ax2.axhline(0.0, color='k', ls=':', lw=1, label='$B = 0$ (no radiation)')
    ax2.set_xlabel('$ka$ (wavenumber × radius)')
    ax2.set_ylabel('$B_{33} / (\\rho \\pi a^2 \\omega)$')
    ax2.set_title('(b) Radiation Damping vs Frequency')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(ka.min(), ka.max())

    fig.suptitle(f'Catamaran Hydrodynamic Coefficients  ($d/L = {sep/(2*radius):.1f}$, $a/f = {radius/depth:.2f}$)',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig1_frequency_scan.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def fig2_asymptotic_limits():
    """Zero and infinite frequency asymptotic verification."""
    a_over_f = 0.3 / 0.8
    kas = np.logspace(-3, 2, 80)
    A_vals = []
    B_vals = []
    for ka in kas:
        A, B = single_cylinder_heave_coefficients(float(ka), a_over_f)
        A_vals.append(A)
        B_vals.append(B)
    A_vals = np.array(A_vals)
    B_vals = np.array(B_vals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Added mass with asymptotic limits
    ax1.loglog(kas, A_vals, 'b-', lw=2, label='$A_{33}(ka)$')
    ax1.axhline(1.0, color='r', ls='--', lw=1.5, label='$A_{geom} = 1$ (inf-freq limit)')
    ax1.axhline(A_vals[0], color='g', ls=':', lw=1.5,
                label=f'$A(ka \\to 0) = {A_vals[0]:.4f}$')
    ax1.set_xlabel('$ka$')
    ax1.set_ylabel('$A_{33} / (\\rho \\pi a^2)$')
    ax1.set_title('(a) Added Mass: Asymptotic Limits')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Damping with zero line
    ax2.loglog(kas, np.maximum(B_vals, 1e-20), 'b-', lw=2, label='$B_{33}(ka)$')
    B_max = B_vals.max()
    ax2.axhline(0.01 * B_max, color='r', ls='--', lw=1.5,
                label=f'$0.01 \\times B_{{max}} = {0.01*B_max:.2e}$')
    ax2.set_xlabel('$ka$')
    ax2.set_ylabel('$B_{33} / (\\rho \\pi a^2 \\omega)$')
    ax2.set_title('(b) Damping: Zero-Frequency Limit')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Asymptotic Verification: Single Submerged Cylinder', fontsize=14, y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig2_asymptotic_limits.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def fig3_distance_decay():
    """Interference factor sigma vs separation distance."""
    decay = distance_decay_study(radius=0.3, depth=0.8, ka_fixed=0.5)
    results = decay["results"]

    d_over_L = np.array([r["d_over_L"] for r in results])
    sigma_A = np.array([r["sigma_A"] for r in results])
    A_mut = np.array([r["A33_mutual"] for r in results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Interference factor
    ax1.semilogx(d_over_L, sigma_A, 'bo-', lw=2, ms=6, label='$\\sigma_A(d)$')
    ax1.axhline(1.0, color='r', ls='--', lw=1.5, label='$\\sigma = 1$ (single-hull limit)')
    ax1.axhspan(0.95, 1.05, alpha=0.15, color='green', label='$\\pm 5\\%$ band')
    ax1.set_xlabel('$d / L$ (separation / characteristic length)')
    ax1.set_ylabel('$\\sigma_A = A_{33,cat} / (2 A_{33,single})$')
    ax1.set_title('(a) Interference Factor vs Distance')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Mutual added mass
    ax2.loglog(d_over_L, np.abs(A_mut) + 1e-15, 'rs-', lw=2, ms=6,
               label='$|A_{33,mutual}|$')
    ax2.set_xlabel('$d / L$')
    ax2.set_ylabel('$|A_{33,mutual}| / (\\rho \\pi a^2)$')
    ax2.set_title(f'(b) Mutual Added Mass Decay  ($ka = {decay["ka_fixed"]}$)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    n = decay["decay_exponent"]
    fig.suptitle(f'Distance Decay Study  ($ka = {decay["ka_fixed"]}$, fitted exponent $n = {n:.2f}$)',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig3_distance_decay.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def fig4_negative_control():
    """Singularity behavior at d -> 0."""
    nc = negative_control_coincident(radius=0.3, depth=0.8, ka=0.5)
    data = nc["data"]

    d_over_a = [s["d_over_a"] for s in data if s["finite"]]
    A_mut = [abs(s["A_mutual"]) for s in data if s["finite"]]
    sigma = [s["sigma_A"] for s in data if s["finite"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.loglog(d_over_a, [a + 1e-15 for a in A_mut], 'go-', lw=2, ms=6,
               label='$|A_{33,mutual}|$')
    ax1.set_xlabel('$d / a$ (separation / radius)')
    ax1.set_ylabel('$|A_{33,mutual}| / (\\rho \\pi a^2)$')
    ax1.set_title('(a) Mutual Added Mass: $d \\to 0$ Singularity')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    ax2.semilogx(d_over_a, sigma, 'mo-', lw=2, ms=6, label='$\\sigma_A$')
    ax2.axhline(1.0, color='r', ls='--', lw=1.5, label='$\\sigma = 1$')
    ax2.set_xlabel('$d / a$')
    ax2.set_ylabel('$\\sigma_A$')
    ax2.set_title('(b) Interference Factor: Coincident Limit')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    fig.suptitle('Negative Control: Hull Coincidence Limit ($d \\to 0$)', fontsize=14, y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig4_negative_control.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def fig5_bem_cross_validation():
    """BEM vs analytical cross-validation."""
    bem_cv = bem_cross_validation(
        radius=0.3, depth=0.8, n_panels=48,
        ka_values=[0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0, 3.0, 5.0],
    )

    ka_vals = [r["ka"] for r in bem_cv if "error" not in r]
    A_an = [r["A_analytical"] for r in bem_cv if "error" not in r]
    A_bem = [r["A_BEM_dimless"] for r in bem_cv if "error" not in r]
    B_an = [r["B_analytical"] for r in bem_cv if "error" not in r]
    B_bem = [r["B_BEM_dimless"] for r in bem_cv if "error" not in r]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.semilogx(ka_vals, A_an, 'b-o', lw=2, ms=6, label='Analytical (multipole)')
    ax1.semilogx(ka_vals, A_bem, 'r--s', lw=2, ms=6, label='2D BEM (48 panels)')
    ax1.axhline(1.0, color='k', ls=':', lw=1, label='$A_{geom}$')
    ax1.set_xlabel('$ka$')
    ax1.set_ylabel('$A_{33} / (\\rho \\pi a^2)$')
    ax1.set_title('(a) Added Mass: BEM vs Analytical')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    ax2.semilogx(ka_vals, B_an, 'b-o', lw=2, ms=6, label='Analytical')
    ax2.semilogx(ka_vals, B_bem, 'r--s', lw=2, ms=6, label='2D BEM')
    ax2.set_xlabel('$ka$')
    ax2.set_ylabel('$B_{33} / (\\rho \\pi a^2 \\omega)$')
    ax2.set_title('(b) Damping: BEM vs Analytical')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('BEM Cross-Validation: Submerged Circular Cylinder ($a/f = 0.375$)',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig5_bem_cross_validation.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def main():
    paths = []
    paths.append(fig1_frequency_scan())
    paths.append(fig2_asymptotic_limits())
    paths.append(fig3_distance_decay())
    paths.append(fig4_negative_control())
    paths.append(fig5_bem_cross_validation())
    print(f"\nAll {len(paths)} figures saved to {OUT_DIR}")
    return paths


if __name__ == "__main__":
    main()
