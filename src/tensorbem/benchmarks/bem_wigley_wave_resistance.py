"""Verified benchmark: Wigley hull wave resistance vs Michell theory.

This module establishes a verification case for the Wigley hull wave resistance
at three Froude numbers (Fr=0.2, 0.3, 0.4) against the converged Michell
thin-ship theory reference.

Verification strategy:
1. Geometry: Wigley III parabolic hullform with closed-form properties
2. Wave resistance: Kochin far-field path (Michell theory) vs converged reference
3. Quadrature convergence: GCI analysis for the Michell quadrature
4. Criteria: Rw error < 5%, GCI < 3%

The Kochin path is the verification path (analytical, converged quadrature).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..wigley import (
    michell_wave_resistance,
    wigley_hull_properties,
)
from .wigley_verification import converged_wave_resistance


def compute_gci(
    values: list[float], refinement_ratio: float = 2.0, safety_factor: float = 1.25
) -> dict[str, float]:
    """Compute Grid Convergence Index for three mesh levels.
    
    Parameters
    ----------
    values : list[float]
        Three values from coarsest to finest mesh.
    refinement_ratio : float
        Mesh refinement ratio (default 2.0).
    safety_factor : float
        GCI safety factor (default 1.25).
    
    Returns
    -------
    dict
        GCI metrics including order, gci_fine, rel_change.
    """
    if len(values) != 3:
        raise ValueError("GCI requires exactly 3 mesh levels")
    
    f1, f2, f3 = values[2], values[1], values[0]  # finest to coarsest
    r = refinement_ratio
    
    eps12 = abs((f2 - f1) / f1) if abs(f1) > 1e-30 else 1e-10
    eps23 = abs((f3 - f2) / f2) if abs(f2) > 1e-30 else 1e-10
    
    if eps12 < 1e-15 or eps23 < 1e-15:
        return {"order": 2.0, "gci_fine": 0.0, "rel_change": 0.0, "refinement": r}
    
    # Estimate order of convergence
    s = math.log(eps23 / eps12) / math.log(r)
    s = max(s, 0.1)  # clamp to avoid division by zero
    
    # GCI for fine mesh
    gci_fine = safety_factor * eps12 / (r**s - 1.0)
    
    return {
        "order": s,
        "gci_fine": gci_fine,
        "rel_change": eps12,
        "refinement": r,
    }


def run_wigley_wave_resistance_verification(
    L: float = 10.0,
    B: float = 1.0,
    T: float = 0.625,
    froude_numbers: tuple[float, ...] = (0.2, 0.3, 0.4),
    rho: float = 1025.0,
    g: float = 9.80665,
) -> dict[str, Any]:
    """Run the Wigley wave resistance verification study.
    
    Parameters
    ----------
    L, B, T : float
        Hull dimensions (length, beam, draft).
    froude_numbers : tuple[float]
        Froude numbers to test.
    rho : float
        Water density.
    g : float
        Gravitational acceleration.
    
    Returns
    -------
    dict
        Complete verification results.
    """
    # Hull properties
    props = wigley_hull_properties(L, B, T)
    
    cases = []
    for Fr in froude_numbers:
        U = Fr * math.sqrt(g * L)
        
        # Reference (converged Michell)
        Rw_ref = converged_wave_resistance(L, B, T, U, rho=rho, g=g)
        
        # Quadrature convergence study (Kochin path)
        quad_levels = [128, 256, 512]
        quad_results = []
        Rw_quad_list = []
        
        for n_quad in quad_levels:
            Rw = michell_wave_resistance(L, B, T, U, rho=rho, g=g, n_quad=n_quad)
            quad_results.append({
                "n_quad": n_quad,
                "Rw": Rw,
            })
            Rw_quad_list.append(Rw)
        
        # Production value (256 quadrature points)
        Rw_prod = michell_wave_resistance(L, B, T, U, rho=rho, g=g, n_quad=256)
        
        # GCI for quadrature convergence
        gci = compute_gci(Rw_quad_list)
        
        # Reference area for Cw
        S_ref = L * T
        Cw_ref = Rw_ref / (0.5 * rho * U**2 * S_ref)
        Cw_prod = Rw_prod / (0.5 * rho * U**2 * S_ref)
        
        # Error metrics
        rel_error = abs(Rw_prod - Rw_ref) / Rw_ref if Rw_ref > 1e-30 else 0.0
        
        case = {
            "froude_number": Fr,
            "speed": U,
            "Rw_reference": Rw_ref,
            "Rw_production": Rw_prod,
            "rel_error": rel_error,
            "Cw_reference": Cw_ref,
            "Cw_production": Cw_prod,
            "quadrature_levels": quad_results,
            "gci": gci,
            "criteria": {
                "wave_resistance_error": {
                    "measured": rel_error,
                    "limit": 0.05,
                    "passed": rel_error < 0.05,
                },
                "gci": {
                    "measured": gci["gci_fine"],
                    "limit": 0.03,
                    "passed": gci["gci_fine"] < 0.03,
                },
            },
        }
        cases.append(case)
    
    # Overall criteria check
    all_passed = all(
        c["criteria"]["wave_resistance_error"]["passed"] 
        and c["criteria"]["gci"]["passed"]
        for c in cases
    )
    
    return {
        "study": "bem_wigley_wave_resistance",
        "hull": "Wigley III (parabolic)",
        "geometry": {
            "L": L,
            "B": B,
            "T": T,
            "L_over_B": L / B,
            "B_over_T": B / T,
        },
        "fluid": {
            "rho": rho,
            "g": g,
        },
        "reference": {
            "volume": props["volume"],
            "Cb": props["Cb"],
            "Cm": props["Cm"],
            "Cp": props["Cp"],
            "waterplane_area": props["waterplane_area"],
            "midship_section_area": props["midship_section_area"],
        },
        "method": {
            "reference": "Converged Michell thin-ship theory (composite Gauss-Legendre, 480 subintervals, 20 nodes each)",
            "production": "Standard Michell quadrature (256 Gauss-Legendre nodes, lambda=1+s^2 substitution)",
            "quadrature_convergence": "Three levels: n_quad = 128, 256, 512",
        },
        "cases": cases,
        "all_criteria_passed": all_passed,
        "evidence_level": "REAL (converged Michell theory reference; two independent quadratures)",
    }


def main(output_dir: str | Path = "benchmarks/verified/bem_wigley_wave_resistance") -> None:
    """Run verification and write results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_wigley_wave_resistance_verification()
    
    # Write results JSON
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results written to {results_path}")
    print(f"All criteria passed: {results['all_criteria_passed']}")
    
    for case in results["cases"]:
        Fr = case["froude_number"]
        err = case["rel_error"]
        gci = case["gci"]["gci_fine"]
        print(f"  Fr={Fr:.2f}: Rw error={err:.4%}, GCI={gci:.4%}")


if __name__ == "__main__":
    main()
