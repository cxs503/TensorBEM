"""Cathodic protection verification module.

This module provides verification tests for ship hull cathodic protection
current distribution using 3-D Laplace BEM, following DNV-RP-B401 criteria.

See :mod:`tensorbem.cathodic_protection` for the solver implementation.
"""

from tensorbem.cathodic_protection import (
    BulkCarrierHull,
    generate_closed_hull_mesh,
    assign_coating_damage,
    solve_cathodic_protection,
    evaluate_exterior_potential,
    check_laplace_residual,
    compute_dnv_anode_demand,
    run_verification_study,
    PROTECTION_POTENTIAL,
    ANODE_OPEN_CIRCUIT_POTENTIAL,
    SEAWATER_CONDUCTIVITY,
)

__all__ = [
    "BulkCarrierHull",
    "generate_closed_hull_mesh",
    "assign_coating_damage",
    "solve_cathodic_protection",
    "evaluate_exterior_potential",
    "check_laplace_residual",
    "compute_dnv_anode_demand",
    "run_verification_study",
    "PROTECTION_POTENTIAL",
    "ANODE_OPEN_CIRCUIT_POTENTIAL",
    "SEAWATER_CONDUCTIVITY",
]
