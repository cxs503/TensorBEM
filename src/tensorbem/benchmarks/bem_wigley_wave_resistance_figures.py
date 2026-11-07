"""Generate figures for Wigley wave resistance verification."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..wigley import michell_wave_resistance, wigley_hull_properties
from .wigley_verification import converged_wave_resistance


def plot_wave_resistance_comparison(
    results: dict, output_path: Path
) -> None:
    """Plot wave resistance vs Froude number: reference vs production."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cases = results["cases"]
    froude_numbers = [c["froude_number"] for c in cases]
    Rw_ref = [c["Rw_reference"] for c in cases]
    Rw_prod = [c["Rw_production"] for c in cases]
    Cw_ref = [c["Cw_reference"] for c in cases]
    Cw_prod = [c["Cw_production"] for c in cases]
    
    # Left panel: Wave resistance (N)
    ax = axes[0]
    ax.plot(froude_numbers, Rw_ref, 'o-', label='Reference (converged)', linewidth=2, markersize=8)
    ax.plot(froude_numbers, Rw_prod, 's--', label='Production (n=256)', linewidth=2, markersize=8)
    ax.set_xlabel('Froude number $F_n$', fontsize=12)
    ax.set_ylabel('Wave resistance $R_w$ (N)', fontsize=12)
    ax.set_title('Wigley III Wave Resistance vs Michell Theory', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.15, 0.45)
    
    # Right panel: Wave resistance coefficient
    ax = axes[1]
    ax.plot(froude_numbers, [1e3 * c for c in Cw_ref], 'o-', label='Reference', linewidth=2, markersize=8)
    ax.plot(froude_numbers, [1e3 * c for c in Cw_prod], 's--', label='Production', linewidth=2, markersize=8)
    ax.set_xlabel('Froude number $F_n$', fontsize=12)
    ax.set_ylabel('Wave resistance coefficient $10^3 C_w$', fontsize=12)
    ax.set_title('Wave Resistance Coefficient', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.15, 0.45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_quadrature_convergence(
    results: dict, output_path: Path
) -> None:
    """Plot quadrature convergence for each Froude number."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    cases = results["cases"]
    
    for idx, case in enumerate(cases):
        ax = axes[idx]
        Fr = case["froude_number"]
        Rw_ref = case["Rw_reference"]
        
        quad_levels = case["quadrature_levels"]
        n_quad = [q["n_quad"] for q in quad_levels]
        Rw = [q["Rw"] for q in quad_levels]
        
        # Plot relative error
        rel_error = [abs(r - Rw_ref) / Rw_ref * 100 for r in Rw]
        
        ax.semilogy(n_quad, rel_error, 'o-', linewidth=2, markersize=8, color='C0')
        ax.axhline(y=5.0, color='r', linestyle='--', linewidth=1.5, label='5% criterion')
        ax.set_xlabel('Number of quadrature points $n_{quad}$', fontsize=11)
        ax.set_ylabel('Relative error (%)', fontsize=11)
        ax.set_title(f'$F_n = {Fr:.2f}$', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(100, 600)
        
        # Add GCI annotation
        gci = case["gci"]["gci_fine"] * 100
        ax.text(0.95, 0.95, f'GCI = {gci:.4f}%', 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_hull_geometry(
    results: dict, output_path: Path
) -> None:
    """Plot Wigley hull geometry."""
    L = results["geometry"]["L"]
    B = results["geometry"]["B"]
    T = results["geometry"]["T"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Generate hull surface
    nx, nz = 50, 25
    x_vals = np.linspace(-L/2, L/2, nx)
    z_vals = np.linspace(-T, 0, nz)
    X, Z = np.meshgrid(x_vals, z_vals, indexing='ij')
    Y = 0.5 * B * (1 - (2*X/L)**2) * (1 - (Z/T)**2)
    
    # Left panel: Body plan (stations)
    ax = axes[0]
    stations = np.linspace(-L/2, L/2, 11)
    for x in stations:
        z = np.linspace(-T, 0, 50)
        y = 0.5 * B * (1 - (2*x/L)**2) * (1 - (z/T)**2)
        y = np.maximum(y, 0)
        ax.plot(y, z, 'b-', linewidth=1.5)
        ax.plot(-y, z, 'b-', linewidth=1.5)
    ax.axhline(y=0, color='k', linewidth=1)
    ax.set_xlabel('Half-breadth $y$ (m)', fontsize=12)
    ax.set_ylabel('Height $z$ (m)', fontsize=12)
    ax.set_title('Body Plan', fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Right panel: Waterline profile
    ax = axes[1]
    waterlines = np.linspace(-T, 0, 7)
    for z in waterlines:
        x = np.linspace(-L/2, L/2, 100)
        y = 0.5 * B * (1 - (2*x/L)**2) * (1 - (z/T)**2)
        y = np.maximum(y, 0)
        ax.plot(x, y, 'g-', linewidth=1.5)
        ax.plot(x, -y, 'g-', linewidth=1.5)
    ax.axvline(x=0, color='k', linewidth=1, linestyle='--')
    ax.set_xlabel('Longitudinal position $x$ (m)', fontsize=12)
    ax.set_ylabel('Half-breadth $y$ (m)', fontsize=12)
    ax.set_title('Waterline Profile', fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main(
    results_path: Path = Path("benchmarks/verified/bem_wigley_wave_resistance/results.json"),
    output_dir: Path = Path("benchmarks/verified/bem_wigley_wave_resistance/figures"),
) -> None:
    """Generate all figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_path) as f:
        results = json.load(f)
    
    plot_wave_resistance_comparison(results, output_dir / "wave_resistance_comparison.png")
    plot_quadrature_convergence(results, output_dir / "quadrature_convergence.png")
    plot_hull_geometry(results, output_dir / "hull_geometry.png")
    
    print(f"Figures written to {output_dir}")


if __name__ == "__main__":
    main()
