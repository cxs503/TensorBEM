"""TensorBEM: transparent boundary-element methods for scientific computing."""

from .geometry import PanelMesh, regular_polygon
from .geometry3d import TriangleMesh, icosphere
from .solver import LaplaceDirichletResult, solve_laplace_dirichlet
from .solver3d import (
    LaplaceSurfaceResult,
    assemble_surface_matrices,
    evaluate_surface_field,
    solve_laplace_surface,
)

__all__ = [
    "PanelMesh",
    "regular_polygon",
    "LaplaceDirichletResult",
    "solve_laplace_dirichlet",
    "TriangleMesh",
    "icosphere",
    "LaplaceSurfaceResult",
    "solve_laplace_surface",
    "assemble_surface_matrices",
    "evaluate_surface_field",
]
__version__ = "0.2.0"
