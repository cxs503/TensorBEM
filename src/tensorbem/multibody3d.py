"""Multi-body frequency-domain hydrodynamics in three dimensions.

Why this module exists
----------------------
:mod:`tensorbem.free_surface_solver3d` solves the radiation problem for *one*
wetted hull (:class:`~tensorbem.geometry3d.WettedHullMesh`): the Kelvin
free-surface Green function is used directly, so the solution is the exact
linearised answer for a prescribed frequency ``nu = omega**2 / g``, but there is
no way to ask the two questions a multihull design actually poses:

1. **What is the hydrodynamic interaction between the hulls?**  Two identical
   hulls side by side do not behave like two isolated hulls: the wave field
   radiated by one is felt by the other, and the interference factor
   ``sigma = A_pair / (n_hull * A_single)`` is what a catamaran or trimaran
   design trades against.
2. **What are the two frequency limits ``omega -> 0`` and ``omega -> infinity``?**
   A frequency-domain Green-function solver approaches neither limit
   convergently: ``nu -> 0`` degenerates the kernel, and ``nu -> infinity`` makes
   the kernel oscillate on a scale shorter than any affordable panel.  The
   limits are classical *potential* problems, not frequency-domain ones, and are
   cheapest to solve directly.

This module supplies both: :func:`radiation_sweep` runs the existing 3-D
radiation solver on any (possibly disconnected) wetted surface, and
:func:`static_added_mass` solves the two asymptotic potential problems exactly
with a plain Laplace double/single-layer collocation -- the same machinery as
:func:`tensorbem.solver3d.compute_added_mass_matrix_3d`, generalised to a
prescribed free-surface parity.

Formulation of the asymptotes
-----------------------------
In the frequency domain with the time factor ``exp(-i omega t)``, the linearised
free-surface condition for the radiation potential is

    dphi/dz = nu * phi,   nu = omega**2 / g,   on z = 0.

The two limits therefore are

======================  =======================  ==========================
limit                   free-surface condition   parity of ``phi`` in z
======================  =======================  ==========================
``omega -> 0``          ``dphi/dz = 0``          even, ``sigma = +1``
(rigid wall, ``A(0)``)
``omega -> infinity``   ``phi = 0``              odd,  ``sigma = -1``
(pressure release, ``A(inf)``)
======================  =======================  ==========================

Both are half-space problems, and both are solved here by *continuation*: the
wetted hull is closed by its mirror image about ``z = 0`` into a closed body
``D`` (a "double body"), and the exterior Neumann problem

    laplace(phi_j) = 0  in R^3 \\ D,      dphi_j/dn = q_j  on dD,
    q_j(M x) = sigma * q_j(x),            M: (x, y, z) -> (x, y, -z),

is solved for every requested rigid mode ``j``.  For a body that is symmetric
about ``z = 0`` the solution inherits ``phi_j(M x) = sigma phi_j(x)``, i.e. it
satisfies ``dphi/dz = 0`` (``sigma = +1``) or ``phi = 0`` (``sigma = -1``) on
``z = 0``, so restricted to ``z < 0`` it *is* the required half-space solution.
The added mass is then the half-body integral

    A_ij = -rho * int_{Gamma_B} phi_j n_i dS,

taken over the original (upper) panels only, with ``n_i`` the generalised
normal of mode ``i``.  Because ``phi`` is continued symmetrically and the
integrand ``phi_j n_i`` is even in ``z`` for every parity, this equals half of
the double-body integral; the code integrates the half body directly so no
factor is hidden.

Two independent properties are used as self-checks and are reported on every
solve:

* the **parity residual** ``max |phi(M x) - sigma phi(x)|`` over mirrored panel
  pairs: the continuation is built in, and the residual measures how exactly the
  discretisation honours it (the mirrored collocation rows differ only by the
  quadrature error on near-singular panel pairs, so the residual sits at
  ``1e-4 .. 1e-2`` on the meshes used here -- a diagnostic, not a proof);
* the **closure volume**, which must be twice the wetted-hull displacement
  (``Cb * L * B * T``), so a broken mirror weld cannot pass unnoticed.

Because the Laplace problem has no spurious modes (irregular frequencies of the
direct BIE belong to the Helmholtz interior Dirichlet problem), the collocation
system is well conditioned for any hull, and the two asymptotes need no
frequency sweep: they are exact for the discretisation, ``O(h^2)`` in the
panel-averaged added mass like :func:`~tensorbem.solver3d.compute_added_mass_matrix_3d`.

What the module does *not* do
-----------------------------
No mean drift, no forward-speed terms (the Kelvin kernel here is the zero-speed
deep-water kernel), no hull-to-hull viscosity or wave breaking, and no
diffraction interaction between hulls (that is a separate scattering problem).
The interference factors are linear-potential-theory quantities.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .free_surface_solver3d import Radiation3DResult, solve_radiation_3d
from .geometry3d import SurfaceMesh3D, TriangleMesh, WettedHullMesh
from .solver3d import assemble_surface_matrices
from .wigley3_hull import (
    WIGLEY3_ALPHA,
    WIGLEY3_B,
    WIGLEY3_L,
    WIGLEY3_T,
    wigley3_block_coefficient,
    wigley3_displaced_volume,
    wigley3_wetted_hull_mesh,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: Rigid-body mode indices (surge, sway, heave, roll, pitch, yaw).
SURGE = 0
SWAY = 1
HEAVE = 2
ROLL = 3
PITCH = 4
YAW = 5

#: Modes used by the multihull study: transverse drift, vertical heave, pitch.
DEFAULT_MODES: tuple[int, ...] = (SWAY, HEAVE, PITCH)

_DEFAULT_DENSITY = 1025.0
_DEFAULT_GRAVITY = 9.80665

#: Parity sign ``sigma`` of each asymptotic free-surface condition.
CONTINUATION_PARITY = {"rigid_wall": 1.0, "pressure_release": -1.0}


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def wigley_hull_mesh(
    nx: int = 12,
    nz: int = 6,
    *,
    L: float = WIGLEY3_L,
    B: float = WIGLEY3_B,
    T: float = WIGLEY3_T,
    alpha: float = WIGLEY3_ALPHA,
) -> WettedHullMesh:
    """Wigley III wetted hull, thin wrapper around the generator."""
    return wigley3_wetted_hull_mesh(nx=nx, nz=nz, L=L, B=B, T=T, alpha=alpha)


def translated_hull(mesh: WettedHullMesh, dx: float = 0.0, dy: float = 0.0) -> WettedHullMesh:
    """Rigid translation of a wetted hull in the calm waterplane.

    Only horizontal translations are allowed: a vertical shift would move the
    waterline off ``z = 0`` and break the wetted-hull contract.
    """
    if not (math.isfinite(dx) and math.isfinite(dy)):
        raise ValueError("dx and dy must be finite")
    if dx == 0.0 and dy == 0.0:
        return mesh
    shifted = mesh.vertices + np.array([float(dx), float(dy), 0.0])
    return WettedHullMesh(shifted, mesh.faces)


def combined_hull_mesh(meshes: Sequence[WettedHullMesh]) -> WettedHullMesh:
    """Union of disconnected wetted hulls into one boundary surface.

    The BEM collocation does not need connected components, so two or three
    hulls are simply concatenated; the waterline loop test of
    :class:`~tensorbem.geometry3d.WettedHullMesh` then verifies that every open
    edge still lies on ``z = 0`` and that the loops close.
    """
    if len(meshes) == 0:
        raise ValueError("at least one hull is required")
    vertices: list[FloatArray] = []
    faces: list[IntArray] = []
    offset = 0
    for mesh in meshes:
        if not isinstance(mesh, SurfaceMesh3D):
            raise TypeError("each hull must be a SurfaceMesh3D / WettedHullMesh")
        vertices.append(np.asarray(mesh.vertices, dtype=float))
        faces.append(np.asarray(mesh.faces, dtype=np.int64) + offset)
        offset += int(mesh.vertices.shape[0])
    return WettedHullMesh(np.vstack(vertices), np.vstack(faces))


def multihull_hulls(
    n_hulls: int = 2,
    gap_over_beam: float = 0.5,
    *,
    nx: int = 12,
    nz: int = 6,
    L: float = WIGLEY3_L,
    B: float = WIGLEY3_B,
    T: float = WIGLEY3_T,
    alpha: float = WIGLEY3_ALPHA,
) -> list[WettedHullMesh]:
    """The individual hulls of a side-by-side arrangement, centreline last.

    ``gap_over_beam`` is the *clear* distance between neighbouring hull sides in
    units of the beam (the hulls touch at ``0``), so the centre-to-centre
    spacing is ``B * (1 + gap_over_beam)``.  Hulls are placed symmetrically
    about ``y = 0`` with the centre hull of a trimaran on the centreline.
    """
    if int(n_hulls) < 2:
        raise ValueError("n_hulls must be >= 2")
    if not math.isfinite(gap_over_beam) or gap_over_beam < 0.0:
        raise ValueError("gap_over_beam must be a finite non-negative number")
    pitch = float(B) * (1.0 + float(gap_over_beam))
    centre = 0.5 * (int(n_hulls) - 1)
    return [
        translated_hull(
            wigley_hull_mesh(nx=nx, nz=nz, L=L, B=B, T=T, alpha=alpha),
            dy=(index - centre) * pitch,
        )
        for index in range(int(n_hulls))
    ]


def side_by_side_multihull(
    n_hulls: int = 2,
    gap_over_beam: float = 0.5,
    *,
    nx: int = 12,
    nz: int = 6,
    L: float = WIGLEY3_L,
    B: float = WIGLEY3_B,
    T: float = WIGLEY3_T,
    alpha: float = WIGLEY3_ALPHA,
) -> WettedHullMesh:
    """Identical Wigley hulls on parallel tracks, ``n_hulls`` abreast."""
    return combined_hull_mesh(
        multihull_hulls(
            n_hulls,
            gap_over_beam,
            nx=nx,
            nz=nz,
            L=L,
            B=B,
            T=T,
            alpha=alpha,
        )
    )


def multihull_chief_points(hulls: Sequence[WettedHullMesh]) -> FloatArray:
    """CHIEF null-field points for a multi-body surface, one set *per hull*.

    The library's ``chief_points="auto"`` heuristic places its rows near the
    area-weighted centre of the surface it is given.  For a union of two or three
    hulls that centre sits in the *gap between* the hulls, outside every body,
    and the null-field identity is then invalid: the solution keeps the
    irregular-frequency spikes (the symptom is a negative diagonal added mass at
    a few frequencies).  Placing the rows on each hull separately is what makes
    the multi-body solve usable.
    """
    from .free_surface_solver3d import _auto_chief_points_3d

    if len(hulls) == 0:
        raise ValueError("at least one hull is required")
    return np.vstack([_auto_chief_points_3d(hull) for hull in hulls])


def mirror_closure(mesh: WettedHullMesh) -> TriangleMesh:
    """Close a wetted hull by its mirror image about ``z = 0``.

    Every waterline vertex is welded to its own image, so the returned
    :class:`~tensorbem.geometry3d.TriangleMesh` is watertight: its panel count
    is twice the wetted hull's, its volume twice the displacement, and the
    first ``mesh.panel_count`` faces are the original (upper) panels, in the
    original order -- which is what makes the mode-data continuation and the
    half-body added-mass integral below index-exact.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    count = int(vertices.shape[0])
    mirrored = np.column_stack((vertices[:, 0], vertices[:, 1], -vertices[:, 2]))
    unique, inverse = np.unique(
        np.round(np.vstack((vertices, mirrored)), 12), axis=0, return_inverse=True
    )
    lower = inverse[count:][faces][:, ::-1]
    closure = np.vstack((inverse[:count][faces], lower))
    return TriangleMesh(unique, closure)


def generalized_normals(
    mesh: SurfaceMesh3D,
    modes: Sequence[int] = DEFAULT_MODES,
    center: ArrayLike = (0.0, 0.0, 0.0),
) -> FloatArray:
    """Panel-wise generalised normal velocities ``n_i`` for rigid modes.

    ``(n_1, n_2, n_3) = n`` is the outward unit normal and
    ``(n_4, n_5, n_6) = (x - center) x n`` the rotational part, so a unit
    velocity amplitude in mode ``i`` gives ``dphi/dn = n_i`` on the body.
    """
    mode_array = np.asarray(list(modes), dtype=np.int64)
    if mode_array.ndim != 1 or mode_array.size == 0:
        raise ValueError("modes must be a non-empty 1-D sequence")
    if np.any((mode_array < 0) | (mode_array > 5)):
        raise ValueError("each mode index must be in range 0..5")
    origin = np.asarray(center, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("center must be a finite 3-vector")
    normals = np.asarray(mesh.normals, dtype=float)
    relative = np.asarray(mesh.centroids, dtype=float) - origin[None, :]
    full = np.hstack((normals, np.cross(relative, normals)))
    return full[:, mode_array]


def mesh_displacement(mesh: SurfaceMesh3D) -> float:
    """Displaced volume of a hull closed by its calm waterplane."""
    return float(np.sum(np.einsum("ij,ij->i", mesh.centroids, mesh.normals) * mesh.areas)) / 3.0


# ---------------------------------------------------------------------------
# asymptotic potential problems
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StaticAddedMassResult:
    """Added mass of the two frequency asymptotes of the radiation problem.

    ``continuation`` selects the free-surface condition: ``"rigid_wall"`` is the
    ``omega -> 0`` limit (``dphi/dz = 0``), ``"pressure_release"`` the
    ``omega -> infinity`` limit (``phi = 0``).
    """

    modes: IntArray
    continuation: str
    added_mass: FloatArray
    potentials: FloatArray
    condition_number: float
    relative_residual: float
    parity_error: float
    mesh_displacement: float
    closure_volume: float


def static_added_mass(
    mesh: WettedHullMesh,
    *,
    modes: Sequence[int] = DEFAULT_MODES,
    continuation: str = "rigid_wall",
    center: ArrayLike = (0.0, 0.0, 0.0),
    density: float = _DEFAULT_DENSITY,
    quadrature_order: int = 8,
) -> StaticAddedMassResult:
    """Added mass at ``omega -> 0`` or ``omega -> infinity`` (exact BVP solve).

    Solves the exterior Neumann problem for the mirrored (closed) double body
    with the mode data continued by the parity ``sigma`` of the requested
    free-surface limit, then evaluates ``A_ij = -rho int phi_j n_i dS`` over the
    wetted hull, exactly as described in the module docstring.

    Parameters
    ----------
    mesh:
        Wetted hull (or union of hulls) piercing ``z = 0``.
    modes:
        Rigid-body mode indices ``0..5`` to solve for.
    continuation:
        ``"rigid_wall"`` (``sigma = +1``, zero-frequency limit) or
        ``"pressure_release"`` (``sigma = -1``, infinite-frequency limit).
    center:
        Reference point for the rotational modes.
    density, quadrature_order:
        Fluid density and the Duffy/solid-angle quadrature order.
    """
    if continuation not in CONTINUATION_PARITY:
        raise ValueError(
            f"continuation must be one of {sorted(CONTINUATION_PARITY)}, got {continuation!r}"
        )
    sigma = CONTINUATION_PARITY[continuation]
    if isinstance(density, (bool, np.bool_)) or not math.isfinite(float(density)) or float(density) <= 0.0:
        raise ValueError("density must be a positive finite number")
    rho = float(density)

    mode_array = np.asarray(list(modes), dtype=np.int64)
    upper_count = int(mesh.faces.shape[0])
    closure = mirror_closure(mesh)
    single, double = assemble_surface_matrices(closure, quadrature_order=quadrature_order)

    data = generalized_normals(mesh, mode_array, center=center)
    continued = np.vstack((data, sigma * data))

    system = 0.5 * np.eye(closure.panel_count) - double
    rhs = single @ (-continued)
    condition = float(np.linalg.cond(system))
    potentials = np.linalg.solve(system, rhs)

    residuals = np.linalg.norm(system @ potentials - rhs, axis=0)
    rhs_norms = np.linalg.norm(rhs, axis=0)
    relative_residual = float(
        np.max(np.where(rhs_norms > 0.0, residuals / rhs_norms, residuals))
    )

    upper = potentials[:upper_count]
    lower = potentials[upper_count:]
    scale = float(np.max(np.abs(upper))) if upper.size else 1.0
    parity_error = float(np.max(np.abs(lower - sigma * upper)) / max(scale, 1.0e-300))

    areas = np.asarray(mesh.areas, dtype=float)
    added_mass = -rho * ((areas[:, None] * data).T @ upper)

    return StaticAddedMassResult(
        modes=mode_array,
        continuation=continuation,
        added_mass=np.asarray(added_mass, dtype=float),
        potentials=np.asarray(upper, dtype=float),
        condition_number=condition,
        relative_residual=relative_residual,
        parity_error=parity_error,
        mesh_displacement=mesh_displacement(mesh),
        closure_volume=float(
            np.sum(
                np.einsum("ij,ij->i", closure.centroids, closure.normals) * closure.areas
            )
            / 3.0
        ),
    )


# ---------------------------------------------------------------------------
# frequency-domain radiation and interference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RadiationRow:
    """Added mass and damping of one configuration at one frequency."""

    label: str
    n_hulls: int
    gap_over_beam: float | None
    omega: float
    nu: float
    added_mass: FloatArray
    damping: FloatArray
    condition_number: float
    relative_residual: float
    panel_count: int


def radiation_row(
    mesh: WettedHullMesh,
    nu: float,
    *,
    label: str,
    n_hulls: int = 1,
    gap_over_beam: float | None = None,
    modes: Sequence[int] = DEFAULT_MODES,
    center: ArrayLike = (0.0, 0.0, 0.0),
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    chief_points: ArrayLike | str | None = "auto",
) -> RadiationRow:
    """One frequency-domain radiation solve for a multihull configuration."""
    result = solve_radiation_3d(
        mesh,
        float(nu),
        modes=modes,
        center=center,
        density=density,
        gravity=gravity,
        chief_points=chief_points,
    )
    return RadiationRow(
        label=str(label),
        n_hulls=int(n_hulls),
        gap_over_beam=None if gap_over_beam is None else float(gap_over_beam),
        omega=float(result.omega),
        nu=float(nu),
        added_mass=np.asarray(result.added_mass, dtype=float),
        damping=np.asarray(result.damping, dtype=float),
        condition_number=float(result.condition_number),
        relative_residual=float(result.relative_residual),
        panel_count=int(mesh.faces.shape[0]),
    )


def radiation_sweep(
    meshes: dict[str, tuple[WettedHullMesh, int, float | None]],
    nu_values: Iterable[float],
    *,
    modes: Sequence[int] = DEFAULT_MODES,
    center: ArrayLike = (0.0, 0.0, 0.0),
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> list[RadiationRow]:
    """Radiation solves for every configuration at every frequency.

    ``meshes`` maps a label to ``(mesh, n_hulls, gap_over_beam)``.
    """
    rows: list[RadiationRow] = []
    for nu in nu_values:
        for label, (mesh, n_hulls, gap) in meshes.items():
            rows.append(
                radiation_row(
                    mesh,
                    float(nu),
                    label=label,
                    n_hulls=n_hulls,
                    gap_over_beam=gap,
                    modes=modes,
                    center=center,
                    density=density,
                    gravity=gravity,
                )
            )
    return rows


def interference_factor(
    pair_value: float, single_value: float, n_hulls: int
) -> float:
    """``sigma = pair / (n_hulls * single)`` with a guard on the denominator."""
    denominator = float(n_hulls) * float(single_value)
    if denominator == 0.0 or not math.isfinite(denominator):
        return float("nan")
    return float(pair_value) / denominator


def far_field_damping_check(
    mesh: WettedHullMesh,
    nu: float,
    *,
    modes: Sequence[int] = DEFAULT_MODES,
    center: ArrayLike = (0.0, 0.0, 0.0),
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    n_azimuth: int = 72,
    chief_points: ArrayLike | str | None = None,
    interior_panels: ArrayLike | str | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Near-field and far-field radiation damping at one frequency.

    The near-field value comes from the pressure integral on the body, the
    far-field value from the Kochin functions through the Haskind energy
    relation (:func:`tensorbem.free_surface_solver3d.far_field_damping_matrix_3d`).
    Equality of the two is an independent statement of energy conservation: the
    damping equals the energy flux carried away by the radiated waves.

    ``chief_points`` / ``interior_panels`` are forwarded to the radiation solve
    and default to ``None`` (no null-field augmentation).  A multi-body surface
    must pass :func:`multihull_chief_points` explicitly: the library's
    ``"auto"`` heuristic puts the rows at the area-weighted centre of the union,
    which for two hulls abreast lies *between* them, i.e. outside every body,
    where the null-field identity does not hold.
    """
    from .free_surface_solver3d import far_field_damping_matrix_3d

    radiation: Radiation3DResult = solve_radiation_3d(
        mesh,
        float(nu),
        modes=modes,
        center=center,
        density=density,
        gravity=gravity,
        chief_points=chief_points,
        interior_panels=interior_panels,
    )
    near = np.asarray(radiation.damping, dtype=float)
    far = far_field_damping_matrix_3d(
        mesh,
        radiation,
        density=density,
        gravity=gravity,
        n_azimuth=n_azimuth,
    )
    return near, far


def heave_natural_frequency(
    single_mesh: WettedHullMesh,
    *,
    center: ArrayLike = (0.0, 0.0, 0.0),
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> dict[str, float]:
    """Heave natural frequency of a single hull from its own hydrostatics.

    ``omega_n = sqrt(rho g A_wp / (m + A_33(omega -> 0)))`` with ``m`` the
    displaced mass and ``A_33(0)`` the zero-frequency (rigid-wall) heave added
    mass.  Using the zero-frequency added mass (rather than a guessed value) is
    the usual convention for the non-dimensional frequency
    ``omega / omega_n`` of a heave response curve.
    """
    static = static_added_mass(
        single_mesh,
        modes=(HEAVE,),
        continuation="rigid_wall",
        center=center,
        density=density,
    )
    heave_added_mass = float(static.added_mass[0, 0])
    waterplane = waterplane_area(single_mesh)
    mass = float(density) * mesh_displacement(single_mesh)
    stiffness = float(density) * float(gravity) * waterplane
    omega_n = math.sqrt(stiffness / (mass + heave_added_mass))
    return {
        "omega_n": omega_n,
        "heave_added_mass_zero_frequency": heave_added_mass,
        "waterplane_area": waterplane,
        "displaced_mass": mass,
        "heave_stiffness": stiffness,
        "mesh_displacement": mesh_displacement(single_mesh),
    }


def waterplane_area(mesh: WettedHullMesh) -> float:
    """Waterplane area from the closed waterline loop of a wetted hull."""
    edges = np.asarray(mesh.waterline_edges, dtype=np.int64)
    if edges.size == 0:
        raise ValueError("the hull has no waterline loop (submerged body?)")
    start = np.asarray(mesh.vertices, dtype=float)[edges[:, 0]]
    end = np.asarray(mesh.vertices, dtype=float)[edges[:, 1]]
    area = 0.5 * float(np.sum(start[:, 0] * end[:, 1] - end[:, 0] * start[:, 1]))
    return abs(area)