"""TensorBEM: transparent boundary-element methods for scientific computing."""

from .geometry import PanelMesh, regular_polygon
from .solver import LaplaceDirichletResult, solve_laplace_dirichlet

__all__ = [
    "PanelMesh",
    "regular_polygon",
    "LaplaceDirichletResult",
    "solve_laplace_dirichlet",
]
__version__ = "0.1.0"
