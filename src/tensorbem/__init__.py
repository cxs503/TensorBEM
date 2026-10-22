"""TensorBEM: transparent boundary-element methods for scientific computing."""

from .analytic import (
    SOLID_HARMONICS,
    ellipse_added_mass,
    ellipse_translation_trace_scale,
    ellipsoid_added_mass,
    ellipsoid_shape_factors,
    ellipsoid_translation_trace_scale,
    ellipsoid_volume,
    prolate_spheroid_shape_factors,
    solid_harmonic,
    solid_harmonic_degree,
    solid_harmonic_names,
    exponential_integral,
    ursell_cylinder_added_mass,
    ursell_cylinder_coefficients,
)
from .geometry import PanelMesh, ellipse_mesh, regular_polygon
from .clearance import SurfaceClearanceReport, diagnose_surface_clearance
from .geometry3d import (
    TriangleMesh,
    ellipsoid_mesh,
    icosphere,
    project_to_ellipsoid,
)
from .recovery import recover_tangent_gradient
from .solver import (
    LaplaceDirichletResult,
    assemble_matrices,
    evaluate_interior_field,
    solve_laplace_dirichlet,
    solve_laplace_exterior_dirichlet,
)
from .solver3d import (
    LaplaceSurfaceResult,
    assemble_surface_matrices,
    evaluate_representation_surface_velocity_at_panels,
    evaluate_surface_field,
    evaluate_surface_velocity,
    evaluate_surface_velocity_at_panels,
    solve_laplace_surface,
)

__all__ = [
    "PanelMesh",
    "regular_polygon",
    "ellipse_mesh",
    "LaplaceDirichletResult",
    "solve_laplace_dirichlet",
    "solve_laplace_exterior_dirichlet",
    "assemble_matrices",
    "evaluate_interior_field",
    "TriangleMesh",
    "icosphere",
    "ellipsoid_mesh",
    "project_to_ellipsoid",
    "SurfaceClearanceReport",
    "diagnose_surface_clearance",
    "LaplaceSurfaceResult",
    "solve_laplace_surface",
    "assemble_surface_matrices",
    "evaluate_surface_field",
    "evaluate_surface_velocity",
    "evaluate_surface_velocity_at_panels",
    "evaluate_representation_surface_velocity_at_panels",
    "recover_tangent_gradient",
    "SOLID_HARMONICS",
    "solid_harmonic",
    "solid_harmonic_names",
    "exponential_integral",
    "ursell_cylinder_added_mass",
    "ursell_cylinder_coefficients",
    "solid_harmonic_degree",
    "ellipse_added_mass",
    "ellipse_translation_trace_scale",
    "ellipsoid_shape_factors",
    "prolate_spheroid_shape_factors",
    "ellipsoid_added_mass",
    "ellipsoid_volume",
    "ellipsoid_translation_trace_scale",
]
__version__ = "0.3.0"
