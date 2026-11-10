"""Seakeeping verification suite for TensorBEM 3-D potential-flow BEM.

This module implements the first comprehensive seakeeping verification surface
for TensorBEM, covering:

1. **Hemisphere radiation added-mass analytical anchor** — a=1 m hemisphere,
   A₃₃ = 2/3·ρ·π·a³ at zero frequency, validated against Hulme (1982).
2. **Froude-Krylov excitation force closed-form** — box barge L=100 m / B=20 m / T=8 m,
   F₃ FK closed-form solution vs BEM diffraction at long-wave limit.
3. **Wigley III RAO frequency response** — ω ∈ [0.05, 3.0] rad/s sweep,
   validating long-wave limit (kL < 0.3 → heave RAO ∈ [0.95, 1.05]) and
   high-frequency limit (ω = 3.0 → heave RAO < 0.2).
4. **Mesh convergence** — N×M from 10×5 to 40×20, RAO convergence < 2%.

References
----------
- Hulme, A. (1982). "The wave forces acting on a floating hemisphere."
  J. Fluid Mech. 121, 443–463.
- Journee, J.M.J. (1992). "Experiments and Calculations on 4 Wigley Hull
  Forms in Head Waves." Report 09-V-09, Delft University of Technology.
- Newman, J.N. (1977). "Marine Hydrodynamics." MIT Press.
"""

from .hemisphere_anchor import (
    HemisphereAnchorResult,
    compute_hemisphere_zero_frequency_anchor,
    compute_hemisphere_radiation_study,
)
from .fk_excitation import (
    FKExcitationResult,
    compute_fk_heave_force_box_barge,
    compute_fk_vs_bem_comparison,
)
from .wigley_rao import (
    WigleyRAOStudy,
    compute_wigley_rao_spectrum,
    compute_wigley_rao_with_limits,
)
from .mesh_convergence import (
    MeshConvergenceStudy,
    run_wigley_mesh_convergence,
)

__all__ = [
    "HemisphereAnchorResult",
    "compute_hemisphere_zero_frequency_anchor",
    "compute_hemisphere_radiation_study",
    "FKExcitationResult",
    "compute_fk_heave_force_box_barge",
    "compute_fk_vs_bem_comparison",
    "WigleyRAOStudy",
    "compute_wigley_rao_spectrum",
    "compute_wigley_rao_with_limits",
    "MeshConvergenceStudy",
    "run_wigley_mesh_convergence",
]
