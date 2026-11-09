"""Galvanic cathodic protection of a ship hull by 3-D Laplace BEM.

Scope and relation to :mod:`tensorbem.cathodic_protection`
---------------------------------------------------------
:mod:`tensorbem.cathodic_protection` solves a *linear* mixed Dirichlet/Neumann
problem on a 200 m bulk carrier: the potential on every coating holiday and on
every anode is prescribed, so the anode current is a direct algebraic outcome
of those prescribed potentials.

This module solves the *galvanically coupled non-linear* problem on a
simplified 100 m x 20 m x 10 m box hull and answers a different question: with
Butler-Volmer cathode kinetics, an oxygen-transport limiting current and a
coating-breakdown ratio ``beta``, what is the steady-state protection-potential
envelope, is it inside the DNV-RP-B401 window, and how does the anode current
scale with ``beta`` and with the number of anodes?  No bulk-carrier
verification case is re-used or re-published here.

Physical model
--------------
Seawater is a homogeneous conductor (``sigma = 3.5 S/m`` at 15 degC).  In the
steady state the electrolyte potential satisfies

    nabla^2 phi = 0            in the seawater Omega (exterior of the hull),

with, on the wetted hull surface Gamma = Gamma_A u Gamma_B u Gamma_C,

    Gamma_A (sacrificial anode)          phi = phi_a   (closed circuit),
    Gamma_B (bare steel / holiday)       sigma*dphi/dn = -i_c(phi),
    Gamma_C (intact coating)             dphi/dn = 0,
    phi -> 0 at infinity                 (decay),

where ``n`` is the sea-side (fluid) normal, so ``q = dphi/dn`` is positive where
current leaves the metal (anode) and negative where it enters (cathode).  The
cathodic current density is Butler-Volmer kinetics throttled by oxygen
transport (Koutecky-Levich form):

    i_kin(phi) = i_0 [ exp(-a_c (phi - phi_eq)) - exp(+a_a (phi - phi_eq)) ],
    a_c = alpha_c F /(R T),   a_a = alpha_a F /(R T),
    i_c(phi)   = i_kin / (1 + i_kin / i_lim).

``i_lim`` is the oxygen-diffusion limiting current density; it is set to the
DNV-RP-B401 *initial design current density for bare steel in temperate
seawater* (0.15 A/m^2).  The consequence, stated up front: no bare panel can
draw more than 0.15 A/m^2, so the total anode current is bounded by
``0.15 * beta * A_wetted``.  The BEM solution says how close to that bound the
hull operates and where the potential envelope sits.

Discretisation
--------------
Constant triangular panels, collocation at panel centroids, analytically
integrated single-layer matrix ``S`` and solid-angle double-layer matrix ``H``
of :mod:`tensorbem.solver3d` (Duffy quadrature for non-self entries).  The
non-linear boundary condition is linearised (Newton) into a mixed
Dirichlet/Robin/Neumann linear BEM system, reassembled each iteration (``S``
and ``H`` are cached per mesh), and solved by dense LU (LAPACK) or by GMRES
with a Jacobi preconditioner.  Each Newton step is damped by residual
backtracking.

Verification face
-----------------
* :func:`plate_linear_benchmark`    -- two infinite parallel plates (Dirichlet)
  with mirror-symmetry insulating side walls: exact solution ``phi = V z/H``.
* :func:`cylinder_log_benchmark`    -- coaxial infinite cylinders (inner wall at
  ``phi_a``, outer at 0) with insulating annular caps: exact solution
  ``phi = phi_a ln(R/r)/ln(R/a)``.
* :func:`sphere_exterior_benchmark` -- exterior of a sphere at uniform
  potential: exact ``phi = phi_a a/r`` with uniform ``q = phi_a/a``; anchors
  the *exterior* machinery used by the protection solve.
* :func:`run_study`                 -- DNV envelope, ``beta`` x anode-count
  parameter study, mesh-convergence GCI, charge balance.

Reference values: DNV-RP-B401 (2016) protection window ``-1.05 V <= phi <=
-0.80 V`` vs Ag/AgCl/seawater; bare-steel initial design current density
0.15 A/m^2; Al-Zn-In closed-circuit anode potential -1.05 V.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .geometry3d import TriangleMesh, icosphere
from .kernels3d import duffy_quadrature_batch, solid_angle_triangles_batch
from .solver3d import assemble_surface_matrices

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

_FOUR_PI = 4.0 * np.pi

#: Faraday constant [C/mol].
FARADAY = 96485.332
#: Gas constant [J/(mol K)].
GAS_CONSTANT = 8.314462618
#: Reference temperature [K] (25 degC).
REFERENCE_TEMPERATURE = 298.15

# --- DNV-RP-B401 reference values -------------------------------------------
#: Seawater conductivity at 15 degC [S/m].
SEAWATER_CONDUCTIVITY = 3.5
#: DNV-RP-B401 minimum protection potential [V vs Ag/AgCl].
PROTECTION_POTENTIAL = -0.80
#: DNV-RP-B401 over-protection limit [V vs Ag/AgCl].
OVERPROTECTION_POTENTIAL = -1.05
#: Closed-circuit potential of an Al-Zn-In sacrificial anode [V vs Ag/AgCl].
ANODE_CLOSED_CIRCUIT_POTENTIAL = -1.05
#: DNV-RP-B401 initial design current density for bare steel, temperate
#: seawater [A/m^2]; used as the oxygen-transport limiting current density.
BARE_STEEL_DESIGN_CURRENT_DENSITY = 0.15
#: Free-corrosion potential of carbon steel in aerated seawater [V].
STEEL_FREE_CORROSION_POTENTIAL = -0.60


# ===========================================================================
# 1. Geometry
# ===========================================================================


@dataclass(frozen=True)
class BoxHull:
    """Simplified box-shaped hull used for the protection study.

    The waterline is the plane ``z = 0``, the keel is ``z = -draft``.
    """

    length: float = 100.0
    breadth: float = 20.0
    draft: float = 10.0

    @property
    def wetted_area(self) -> float:
        """Wetted surface area [m^2]: bottom + 2 sides + 2 ends."""
        return (
            self.length * self.breadth
            + 2.0 * self.length * self.draft
            + 2.0 * self.breadth * self.draft
        )

    @property
    def displacement_volume(self) -> float:
        return self.length * self.breadth * self.draft

    def block_coefficient(self) -> float:
        return 1.0


def _quads_from_grid(grid: FloatArray, outward: FloatArray) -> FloatArray:
    """Turn a ``(nv+1, nu+1, 3)`` node grid into outward-oriented quads."""
    quads = np.stack(
        [grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]], axis=2
    ).reshape(-1, 4, 3)
    normal = np.cross(quads[:, 1] - quads[:, 0], quads[:, 3] - quads[:, 0])
    if outward.ndim == 1:
        outward = np.broadcast_to(outward, normal.shape)
    flip = np.einsum("ij,ij->i", normal, outward) < 0.0
    quads[flip] = quads[flip][:, ::-1]
    return quads


def _planar_patch(
    origin: Sequence[float],
    edge_u: Sequence[float],
    edge_v: Sequence[float],
    nu: int,
    nv: int,
    outward: Sequence[float],
) -> FloatArray:
    if nu < 1 or nv < 1:
        raise ValueError("patch subdivisions must be >= 1")
    origin_a = np.asarray(origin, dtype=float)
    edge_u_a = np.asarray(edge_u, dtype=float)
    edge_v_a = np.asarray(edge_v, dtype=float)
    u = np.linspace(0.0, 1.0, nu + 1)
    v = np.linspace(0.0, 1.0, nv + 1)
    grid = (
        origin_a[None, None, :]
        + u[None, :, None] * edge_u_a[None, None, :]
        + v[:, None, None] * edge_v_a[None, None, :]
    )
    return _quads_from_grid(grid, np.asarray(outward, dtype=float))


def _weld_and_triangulate(
    patches: list[tuple[str, FloatArray]]
) -> tuple[FloatArray, IntArray, dict[str, IntArray]]:
    """Weld patch corners into one closed mesh and label the panel groups."""
    quads = np.concatenate([patch for _, patch in patches], axis=0)
    counts = [patch.shape[0] for _, patch in patches]
    keys = np.round(quads.reshape(-1, 3), 9)
    vertices, inverse = np.unique(keys, axis=0, return_inverse=True)
    corner_ids = inverse.reshape(-1, 4)
    faces_blocks: list[IntArray] = []
    groups: dict[str, IntArray] = {}
    offset = 0
    start = 0
    for (name, _), n_quads in zip(patches, counts):
        block = corner_ids[offset : offset + n_quads]
        offset += n_quads
        tris = np.concatenate([block[:, [0, 1, 2]], block[:, [0, 2, 3]]], axis=0)
        faces_blocks.append(tris)
        groups[name] = np.arange(start, start + tris.shape[0], dtype=np.int64)
        start += tris.shape[0]
    faces = np.concatenate(faces_blocks, axis=0)
    return vertices, faces, groups


def box_hull_mesh(
    hull: BoxHull | None = None, nx: int = 20, ny: int = 6, nz: int = 3
) -> tuple[TriangleMesh, dict[str, IntArray]]:
    """Triangular mesh of the wetted surface of the box hull.

    The hull is treated as a closed immersed body (bottom + 4 sides); the free
    surface and the sea bed are not modelled, which is the standard
    simplification for a protection-potential study in deep homogeneous water.
    """
    if hull is None:
        hull = BoxHull()
    L, B, T = hull.length, hull.breadth, hull.draft
    x0, x1 = -0.5 * L, 0.5 * L
    y0, y1 = -0.5 * B, 0.5 * B
    z0 = -T
    patches = [
        ("bottom", _planar_patch((x0, y0, z0), (L, 0.0, 0.0), (0.0, B, 0.0), nx, ny, (0.0, 0.0, -1.0))),
        ("side_y_pos", _planar_patch((x0, y1, z0), (L, 0.0, 0.0), (0.0, 0.0, T), nx, nz, (0.0, 1.0, 0.0))),
        ("side_y_neg", _planar_patch((x0, y0, z0), (L, 0.0, 0.0), (0.0, 0.0, T), nx, nz, (0.0, -1.0, 0.0))),
        ("end_x_pos", _planar_patch((x1, y0, z0), (0.0, B, 0.0), (0.0, 0.0, T), ny, nz, (1.0, 0.0, 0.0))),
        ("end_x_neg", _planar_patch((x0, y0, z0), (0.0, B, 0.0), (0.0, 0.0, T), ny, nz, (-1.0, 0.0, 0.0))),
        ("waterline", _planar_patch((x0, y0, 0.0), (L, 0.0, 0.0), (0.0, B, 0.0), nx, ny, (0.0, 0.0, 1.0))),
    ]
    vertices, faces, groups = _weld_and_triangulate(patches)
    return TriangleMesh(vertices=vertices, faces=faces), groups


def box_hull_panel_count(nx: int, ny: int, nz: int) -> int:
    """Triangle count produced by :func:`box_hull_mesh` for ``(nx, ny, nz)``.

    Includes the waterline lid (``2 nx ny`` triangles).
    """
    return 2 * (2 * nx * ny + 2 * nx * nz + 2 * ny * nz)


def interior_box_mesh(
    length: float = 10.0,
    breadth: float = 10.0,
    height: float = 1.0,
    n_horizontal: int = 8,
    n_vertical: int = 2,
) -> tuple[TriangleMesh, dict[str, IntArray]]:
    """Closed mesh of the box *domain* ``[0,L] x [0,B] x [0,H]``.

    Used by the parallel-plate anchor: interior Laplace problem with Dirichlet
    data on ``z=0`` and ``z=H`` and homogeneous Neumann data on the four side
    walls, exact solution ``phi = V z/H`` (the side walls are the
    mirror-symmetry planes of an infinite plate pair).
    """
    patches = [
        ("z_min", _planar_patch((0.0, 0.0, 0.0), (length, 0.0, 0.0), (0.0, breadth, 0.0), n_horizontal, n_horizontal, (0.0, 0.0, -1.0))),
        ("z_max", _planar_patch((0.0, 0.0, height), (length, 0.0, 0.0), (0.0, breadth, 0.0), n_horizontal, n_horizontal, (0.0, 0.0, 1.0))),
        ("y_min", _planar_patch((0.0, 0.0, 0.0), (length, 0.0, 0.0), (0.0, 0.0, height), n_horizontal, n_vertical, (0.0, -1.0, 0.0))),
        ("y_max", _planar_patch((0.0, breadth, 0.0), (length, 0.0, 0.0), (0.0, 0.0, height), n_horizontal, n_vertical, (0.0, 1.0, 0.0))),
        ("x_min", _planar_patch((0.0, 0.0, 0.0), (0.0, breadth, 0.0), (0.0, 0.0, height), n_horizontal, n_vertical, (-1.0, 0.0, 0.0))),
        ("x_max", _planar_patch((length, 0.0, 0.0), (0.0, breadth, 0.0), (0.0, 0.0, height), n_horizontal, n_vertical, (1.0, 0.0, 0.0))),
    ]
    vertices, faces, groups = _weld_and_triangulate(patches)
    return TriangleMesh(vertices=vertices, faces=faces), groups


def annulus_cylinder_mesh(
    inner_radius: float = 1.0,
    outer_radius: float = 10.0,
    height: float = 2.0,
    n_theta: int = 32,
    n_radial: int = 6,
    n_axial: int = 1,
) -> tuple[TriangleMesh, dict[str, IntArray]]:
    """Closed mesh of the coaxial annulus ``a <= r <= R``, ``0 <= z <= H``.

    Used by the coaxial-cylinder anchor: Dirichlet on the two cylindrical walls,
    homogeneous Neumann on the two annular caps, exact solution
    ``phi = phi_a ln(R/r)/ln(R/a)`` (the caps are mirror-symmetry planes of an
    infinitely long cylinder).
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)

    def ring(radius: float, z: float) -> FloatArray:
        return np.stack(
            [radius * np.cos(theta), radius * np.sin(theta), np.full_like(theta, z)],
            axis=1,
        )

    mid_theta = 0.5 * (theta[:-1] + theta[1:])
    inner_outward = np.stack(
        [-np.cos(mid_theta), -np.sin(mid_theta), np.zeros_like(mid_theta)], axis=1
    )
    outer_outward = np.stack(
        [np.cos(mid_theta), np.sin(mid_theta), np.zeros_like(mid_theta)], axis=1
    )

    def quads_over_theta(grid: FloatArray, outward: FloatArray) -> FloatArray:
        quads = np.stack(
            [grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]], axis=2
        ).reshape(-1, 4, 3)
        normal = np.cross(quads[:, 1] - quads[:, 0], quads[:, 3] - quads[:, 0])
        expanded = np.tile(outward, (quads.shape[0] // outward.shape[0], 1))
        flip = np.einsum("ij,ij->i", normal, expanded) < 0.0
        quads[flip] = quads[flip][:, ::-1]
        return quads

    patches: list[tuple[str, FloatArray]] = []
    z_lin = np.linspace(0.0, height, n_axial + 1)
    for k in range(n_axial):
        grid_in = np.stack(
            [ring(inner_radius, z_lin[k]), ring(inner_radius, z_lin[k + 1])], axis=0
        )
        grid_out = np.stack(
            [ring(outer_radius, z_lin[k]), ring(outer_radius, z_lin[k + 1])], axis=0
        )
        patches.append(("inner_wall", quads_over_theta(grid_in, inner_outward)))
        patches.append(("outer_wall", quads_over_theta(grid_out, outer_outward)))

    r_lin = np.linspace(inner_radius, outer_radius, n_radial + 1)

    def cap_grid(z: float) -> FloatArray:
        r_grid = np.tile(r_lin[:, None], (1, n_theta + 1))
        t_grid = np.tile(theta[None, :], (n_radial + 1, 1))
        return np.stack(
            [r_grid * np.cos(t_grid), r_grid * np.sin(t_grid), np.full_like(r_grid, z)],
            axis=2,
        )

    patches.append(("cap_top", _quads_from_grid(cap_grid(height), np.array([0.0, 0.0, 1.0]))))
    patches.append(("cap_bottom", _quads_from_grid(cap_grid(0.0), np.array([0.0, 0.0, -1.0]))))
    vertices, faces, groups_raw = _weld_and_triangulate(patches)
    merged: dict[str, list[IntArray]] = {}
    for name, ids in groups_raw.items():
        merged.setdefault(name, []).append(ids)
    groups = {name: np.concatenate(parts) for name, parts in merged.items()}
    return TriangleMesh(vertices=vertices, faces=faces), groups


# ===========================================================================
# 2. Steel electrochemistry
# ===========================================================================


@dataclass(frozen=True)
class SteelKinetics:
    """Butler-Volmer cathode kinetics with a transport-limited plateau.

    ``phi_eq`` is the free-corrosion potential, ``i_exchange`` the exchange
    current density, ``alpha_c``/``alpha_a`` the transfer coefficients and
    ``i_limiting`` the oxygen-diffusion limited current density.  The
    Koutecky-Levich combination

        i_c = i_kin / (1 + i_kin / i_lim),   i_kin = Butler-Volmer form,

    makes ``i_c -> i_lim`` at strongly cathodic potentials, which is the
    physical behaviour of oxygen reduction on steel in seawater and the reason
    DNV-RP-B401 can quote a *design current density* per square metre.
    """

    phi_eq: float = STEEL_FREE_CORROSION_POTENTIAL
    i_exchange: float = 1.5e-3
    alpha_c: float = 0.5
    alpha_a: float = 0.5
    temperature: float = REFERENCE_TEMPERATURE
    i_limiting: float = BARE_STEEL_DESIGN_CURRENT_DENSITY

    @property
    def a_c(self) -> float:
        return self.alpha_c * FARADAY / (GAS_CONSTANT * self.temperature)

    @property
    def a_a(self) -> float:
        return self.alpha_a * FARADAY / (GAS_CONSTANT * self.temperature)

    @property
    def tafel_slope(self) -> float:
        """Cathodic Tafel slope ``d phi / d log10(i)`` [V/decade]."""
        return math.log(10.0) / self.a_c

    def _branches(self, phi: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Cathodic / anodic partial current densities of the BV expression."""
        eta = np.asarray(phi, dtype=float) - self.phi_eq
        with np.errstate(over="ignore"):
            cathodic = self.i_exchange * np.exp(np.clip(-self.a_c * eta, -700.0, 700.0))
            anodic = self.i_exchange * np.exp(np.clip(self.a_a * eta, -700.0, 700.0))
        return cathodic, anodic

    def net_current_density(self, phi: FloatArray) -> FloatArray:
        """Signed net current density [A/m^2]; positive = cathodic (into metal).

        The full Butler-Volmer expression is kept (no clipping), so a steel area
        above its free-corrosion potential *dissolves* (negative current) and
        feeds the couple.  Only the cathodic branch is oxygen-transport limited
        (Koutecky-Levich), which is what makes the galvanic mixed potential well
        posed: clipping the current at zero would admit the spurious "no current
        anywhere" solution.
        """
        cathodic, anodic = self._branches(phi)
        i_kin = cathodic - anodic
        scale = 1.0 + np.maximum(i_kin, 0.0) / self.i_limiting
        return i_kin / scale

    def cathodic_current_density(self, phi: FloatArray) -> FloatArray:
        """Cathodic part of the net current density [A/m^2], ``>= 0``."""
        return np.maximum(self.net_current_density(phi), 0.0)

    def derivative(self, phi: FloatArray) -> FloatArray:
        """``d i_net / d phi`` [A/(m^2 V)], negative (monotone BV response)."""
        cathodic, anodic = self._branches(phi)
        i_kin = cathodic - anodic
        d_kin = -(self.a_c * cathodic + self.a_a * anodic)
        scale = 1.0 + np.maximum(i_kin, 0.0) / self.i_limiting
        return np.where(i_kin > 0.0, d_kin / (scale * scale), d_kin)

    def transport_limited_potential(self) -> float:
        """Electrode potential at which the cathodic branch meets the transport limit."""
        return float(
            self.phi_eq - math.log(1.0 + self.i_limiting / self.i_exchange) / self.a_c
        )


@dataclass(frozen=True)
class AnodeKinetics:
    """Anodic Tafel dissolution of an Al-Zn-In sacrificial anode.

    ``i_out(E) = i_reference * 10 ** ((E - ocp) / tafel_slope)`` with ``E`` the
    electrode potential (metal minus adjacent electrolyte) vs Ag/AgCl.  The
    closed-circuit potential ``ocp = -1.05 V`` (DNV-RP-B401) and a 0.05 V/decade
    anodic slope make the anode act as a stiff potentiostat: its electrode
    potential stays a few tens of millivolts noble of ``ocp`` while it delivers
    current, and the metal potential of the whole structure floats to whatever
    value balances anodic against cathodic current.
    """

    ocp: float = ANODE_CLOSED_CIRCUIT_POTENTIAL
    i_reference: float = 1.0
    tafel_slope: float = 0.05

    @property
    def slope(self) -> float:
        """``d ln i / dE`` [1/V]."""
        return math.log(10.0) / self.tafel_slope

    def current_density(self, electrode_potential: FloatArray) -> FloatArray:
        eta = np.asarray(electrode_potential, dtype=float) - self.ocp
        with np.errstate(over="ignore"):
            return self.i_reference * np.exp(np.clip(self.slope * eta, -700.0, 700.0))

    def derivative(self, electrode_potential: FloatArray) -> FloatArray:
        return self.slope * self.current_density(electrode_potential)


# ===========================================================================
# 3. Linear mixed Dirichlet / Neumann Laplace BEM (analytic anchors)
# ===========================================================================


@dataclass(frozen=True)
class MixedLaplaceResult:
    """Boundary data of a mixed Dirichlet/Neumann Laplace solve."""

    potential: FloatArray
    flux: FloatArray
    dirichlet_panels: IntArray
    neumann_panels: IntArray
    relative_residual: float
    domain: str


def solve_mixed_laplace(
    mesh: TriangleMesh,
    phi_prescribed: FloatArray,
    flux_prescribed: FloatArray,
    domain: str = "interior",
    matrices: tuple[FloatArray, FloatArray] | None = None,
) -> MixedLaplaceResult:
    """Solve the mixed Dirichlet/Neumann 3-D Laplace problem by collocation BEM.

    ``phi_prescribed`` carries ``NaN`` on Neumann panels and
    ``flux_prescribed`` carries ``NaN`` on Dirichlet panels.  ``flux`` is
    ``dphi/dn`` along the outward normal of the *interior* domain for
    ``domain='interior'`` and along the fluid normal for ``domain='exterior'``.
    """
    if domain not in ("interior", "exterior"):
        raise ValueError("domain must be 'interior' or 'exterior'")
    phi = np.asarray(phi_prescribed, dtype=float)
    flux = np.asarray(flux_prescribed, dtype=float)
    n = mesh.panel_count
    if phi.shape != (n,) or flux.shape != (n,):
        raise ValueError(f"boundary arrays must both have shape ({n},)")
    is_dirichlet = ~np.isnan(phi)
    if np.any(np.isnan(phi) & np.isnan(flux)):
        raise ValueError("every panel needs exactly one prescribed quantity")
    if np.any(np.isnan(flux[~is_dirichlet])):
        raise ValueError("Neumann panels need a prescribed flux")
    dirichlet = np.where(is_dirichlet)[0]
    neumann = np.where(~is_dirichlet)[0]
    if dirichlet.size == 0:
        raise ValueError("at least one Dirichlet panel is required")

    single, double = matrices if matrices is not None else assemble_surface_matrices(mesh)
    sign = 1.0 if domain == "exterior" else -1.0

    phi_d = phi[dirichlet]
    flux_n = flux[neumann]

    # Discrete boundary integral equation (rows = panels, columns = unknowns):
    #   sum_j S_ij q_j - 0.5 phi_i + sign * sum_j H_ij phi_j = 0
    # with sign = -1 for the interior problem (S q = (0.5 I + H) phi) and
    # sign = +1 for the exterior problem (S q = (0.5 I - H) phi).
    n_d, n_n = dirichlet.size, neumann.size
    matrix = np.empty((n, n), dtype=float)
    matrix[:, :n_d] = single[:, dirichlet]
    matrix[:, n_d:] = sign * double[:, neumann]
    matrix[neumann, n_d + np.arange(n_n)] -= 0.5

    prescribed = np.zeros(n, dtype=float)
    prescribed[dirichlet] = phi_d
    # constants moved to the right-hand side
    rhs = 0.5 * prescribed - sign * (double[:, dirichlet] @ phi_d)
    rhs = rhs - single[:, neumann] @ flux_n

    solution = np.linalg.solve(matrix, rhs)
    values = phi.copy()
    values[neumann] = solution[n_d:]
    fluxes = flux.copy()
    fluxes[dirichlet] = solution[:n_d]

    residual = float(np.linalg.norm(matrix @ solution - rhs))
    scale = float(np.linalg.norm(rhs))
    return MixedLaplaceResult(
        potential=values,
        flux=fluxes,
        dirichlet_panels=dirichlet,
        neumann_panels=neumann,
        relative_residual=residual / scale if scale > 0.0 else residual,
        domain=domain,
    )


def evaluate_boundary_potential(
    mesh: TriangleMesh,
    potential: FloatArray,
    flux: FloatArray,
    points: FloatArray,
    domain: str = "interior",
) -> FloatArray:
    """Representation-formula potential at points away from the surface.

    ``phi(x) = sum_j int_{Gamma_j} [ G q - phi dG/dn ] dS`` with
    ``G = 1/(4 pi r)``.  The single-layer term is integrated with the same Duffy
    rule used to assemble ``S``; the double-layer term uses the *exact* signed
    solid angle of each planar triangle, exactly as ``assemble_surface_matrices``
    does for ``H`` (quadrature of the ``1/r^3`` kernel is hopeless for panels
    seen edge-on or from close range).  ``q`` is the interior-domain (or
    fluid-domain) normal derivative matching :func:`solve_mixed_laplace`.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (P, 3)")
    if domain not in ("interior", "exterior"):
        raise ValueError("domain must be 'interior' or 'exterior'")
    v0, v1, v2 = mesh.v0, mesh.v1, mesh.v2
    samples, weights = duffy_quadrature_batch(v0, v1, v2, order=8)
    flux = np.asarray(flux, dtype=float)
    potential = np.asarray(potential, dtype=float)
    sign_h = 1.0 if domain == "exterior" else -1.0
    values = np.zeros(pts.shape[0], dtype=float)
    for index, target in enumerate(pts):
        delta = target[None, None, :] - samples  # (F, Q, 3)
        radius = np.linalg.norm(delta, axis=2)
        if np.any(radius < 1.0e-12):
            raise ValueError("evaluation point coincides with a quadrature node")
        single_row = np.sum(weights / (_FOUR_PI * radius), axis=1)
        solid_angle = solid_angle_triangles_batch(target, v0, v1, v2)
        double_row = -solid_angle / _FOUR_PI  # =  int dG/dn_body dS
        values[index] = float(single_row @ flux + sign_h * (double_row @ potential))
    return values


# ===========================================================================
# 4. Analytic verification anchors
# ===========================================================================


def _relative_l2(numeric: FloatArray, exact: FloatArray) -> float:
    denom = float(np.linalg.norm(exact))
    diff = float(np.linalg.norm(np.asarray(numeric) - np.asarray(exact)))
    return diff / denom if denom > 0.0 else diff


def plate_linear_benchmark(
    n_horizontal: int = 8,
    n_vertical: int = 2,
    length: float = 10.0,
    breadth: float = 10.0,
    gap: float = 1.0,
    voltage: float = 1.0,
    probes_per_axis: int = 4,
) -> dict[str, Any]:
    """Infinite parallel plates vs BEM (exact solution ``phi = V z/H``).

    The plates are the Dirichlet surfaces ``z=0`` (0 V) and ``z=H`` (V); the
    four side walls are homogeneous-Neumann mirror-symmetry planes, so the
    truncated box reproduces the infinite plate pair *exactly* by uniqueness of
    the Laplace solution.  Errors are measured on an interior probe grid (the
    genuine unknown of a mixed problem) and on the computed plate flux.
    """
    mesh, groups = interior_box_mesh(
        length=length,
        breadth=breadth,
        height=gap,
        n_horizontal=n_horizontal,
        n_vertical=n_vertical,
    )
    n = mesh.panel_count
    phi = np.full(n, np.nan)
    flux = np.zeros(n, dtype=float)
    phi[groups["z_min"]] = 0.0
    phi[groups["z_max"]] = voltage
    result = solve_mixed_laplace(mesh, phi, flux, domain="interior")

    xs = np.linspace(0.15 * length, 0.85 * length, probes_per_axis)
    ys = np.linspace(0.15 * breadth, 0.85 * breadth, probes_per_axis)
    zs = np.linspace(0.1 * gap, 0.9 * gap, probes_per_axis)
    grid = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    numeric = evaluate_boundary_potential(
        mesh, result.potential, result.flux, grid, domain="interior"
    )
    exact = voltage * grid[:, 2] / gap

    plate_flux = np.concatenate(
        [result.flux[groups["z_min"]], result.flux[groups["z_max"]]]
    )
    exact_flux = np.concatenate(
        [
            np.full(groups["z_min"].size, -voltage / gap),
            np.full(groups["z_max"].size, voltage / gap),
        ]
    )
    wall_ids = np.concatenate(
        [groups[k] for k in ("y_min", "y_max", "x_min", "x_max")]
    )
    return {
        "case": "infinite_parallel_plates",
        "panels": int(n),
        "n_horizontal": int(n_horizontal),
        "n_vertical": int(n_vertical),
        "probes": int(grid.shape[0]),
        "potential_rel_l2": _relative_l2(numeric, exact),
        "potential_max_abs_error": float(np.max(np.abs(numeric - exact))),
        "plate_flux_rel_l2": _relative_l2(plate_flux, exact_flux),
        "wall_flux_max_abs": float(np.max(np.abs(result.flux[wall_ids]))),
        "wall_potential_rel_l2": _relative_l2(
            result.potential[wall_ids], mesh.centroids[wall_ids, 2] * voltage / gap
        ),
        "relative_residual": result.relative_residual,
    }


def cylinder_log_benchmark(
    n_theta: int = 32,
    n_radial: int = 6,
    n_axial: int = 1,
    inner_radius: float = 1.0,
    outer_radius: float = 10.0,
    height: float = 2.0,
    inner_potential: float = -1.0,
    probes_per_direction: int = 4,
) -> dict[str, Any]:
    """Coaxial infinite cylinders vs BEM (exact ``phi = phi_a ln(R/r)/ln(R/a)``)."""
    mesh, groups = annulus_cylinder_mesh(
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        height=height,
        n_theta=n_theta,
        n_radial=n_radial,
        n_axial=n_axial,
    )
    n = mesh.panel_count
    phi = np.full(n, np.nan)
    flux = np.zeros(n, dtype=float)
    phi[groups["inner_wall"]] = inner_potential
    phi[groups["outer_wall"]] = 0.0
    result = solve_mixed_laplace(mesh, phi, flux, domain="interior")

    radii = np.linspace(inner_radius * 1.15, outer_radius * 0.85, probes_per_direction)
    thetas = np.linspace(0.2, 2.0 * math.pi - 0.2, 6)
    zs = np.array([0.25 * height, 0.5 * height, 0.75 * height])
    grid = np.array(
        [
            [r * math.cos(t), r * math.sin(t), z]
            for r in radii
            for t in thetas
            for z in zs
        ]
    )
    numeric = evaluate_boundary_potential(
        mesh, result.potential, result.flux, grid, domain="interior"
    )
    rr = np.linalg.norm(grid[:, :2], axis=1)
    exact = (
        inner_potential * np.log(outer_radius / rr) / math.log(outer_radius / inner_radius)
    )

    log_denominator = math.log(outer_radius / inner_radius)
    # interior outward normal on the inner wall is -r_hat, so q = -dphi/dr
    exact_inner_flux = inner_potential / (inner_radius * log_denominator)
    exact_outer_flux = -inner_potential / (outer_radius * log_denominator)
    inner_area = mesh.areas[groups["inner_wall"]]
    outer_area = mesh.areas[groups["outer_wall"]]
    num_inner = result.flux[groups["inner_wall"]]
    num_outer = result.flux[groups["outer_wall"]]
    cap_flux = np.concatenate(
        [result.flux[groups["cap_top"]], result.flux[groups["cap_bottom"]]]
    )
    return {
        "case": "coaxial_infinite_cylinders",
        "panels": int(n),
        "n_theta": int(n_theta),
        "n_radial": int(n_radial),
        "n_axial": int(n_axial),
        "probes": int(grid.shape[0]),
        "potential_rel_l2": _relative_l2(numeric, exact),
        "potential_max_abs_error": float(np.max(np.abs(numeric - exact))),
        "inner_wall_flux_rel_error": float(
            abs(float(np.sum(num_inner * inner_area)) / float(np.sum(exact_inner_flux * inner_area)) - 1.0)
        ),
        "outer_wall_flux_rel_error": float(
            abs(float(np.sum(num_outer * outer_area)) / float(np.sum(exact_outer_flux * outer_area)) - 1.0)
        ),
        "cap_flux_max_abs": float(np.max(np.abs(cap_flux))),
        "relative_residual": result.relative_residual,
    }


def sphere_exterior_benchmark(
    subdivisions: int = 3, radius: float = 1.0, potential: float = -1.0
) -> dict[str, Any]:
    """Exterior of an equipotential sphere (exact ``phi = phi_a a / r``)."""
    mesh = icosphere(subdivisions=subdivisions, radius=radius)
    n = mesh.panel_count
    phi = np.full(n, potential)
    flux = np.full(n, np.nan)
    result = solve_mixed_laplace(mesh, phi, flux, domain="exterior")
    exact_flux = potential / radius
    radii = np.array([1.5, 2.0, 3.0, 5.0])
    directions = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.6, 0.6, 0.53]]
    )
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    grid = np.array([r * d for r in radii for d in directions])
    numeric = evaluate_boundary_potential(
        mesh, result.potential, result.flux, grid, domain="exterior"
    )
    exact = potential * radius / np.linalg.norm(grid, axis=1)
    return {
        "case": "exterior_equipotential_sphere",
        "panels": int(n),
        "flux_rel_error": float(abs(float(np.mean(result.flux)) / exact_flux - 1.0)),
        "potential_rel_l2": _relative_l2(numeric, exact),
        "potential_max_abs_error": float(np.max(np.abs(numeric - exact))),
        "relative_residual": result.relative_residual,
    }


# ===========================================================================
# 5. Coating-breakdown partition and anodes
# ===========================================================================


@dataclass(frozen=True)
class SurfacePartition:
    """Disjoint panel sets of the protection problem.

    ``waterline`` holds the artificial lid panels that close the hull at the
    waterline (insulating free-surface plane); they are excluded from the
    wetted-area statistics.
    """

    bare: IntArray
    coated: IntArray
    anodes: IntArray
    waterline: IntArray
    anode_centers: FloatArray
    requested_beta: float
    achieved_beta: float
    bare_area: float
    coated_area: float
    anode_area: float
    waterline_area: float = 0.0
    #: Total active anode area the discretisation is normalised to [m^2].
    anode_area_target: float = 0.0

    @property
    def anode_scale(self) -> float:
        """Factor that normalises the discrete anode area to the design area."""
        if self.anode_area <= 0.0 or self.anode_area_target <= 0.0:
            return 1.0
        return self.anode_area_target / self.anode_area

    @property
    def total_area(self) -> float:
        """Wetted area (anodes + holidays + coating), excluding the lid."""
        return self.bare_area + self.coated_area + self.anode_area

    def wetted_indices(self, panel_count: int) -> IntArray:
        mask = np.zeros(panel_count, dtype=bool)
        mask[self.bare] = True
        mask[self.coated] = True
        mask[self.anodes] = True
        return np.where(mask)[0]


def partition_surface(
    mesh: TriangleMesh,
    groups: dict[str, IntArray],
    beta: float,
    n_anodes: int = 8,
    anode_span: float = 0.7,
    seed: int = 0,
    insulating_groups: Sequence[str] = ("waterline",),
    anode_footprint: tuple[float, float] = (6.25, 4.0),
) -> SurfacePartition:
    """Assign the hull surface to anodes, bare steel (holidays) and coating.

    Anodes occupy a fixed physical footprint ``anode_footprint = (length,
    breadth)`` centred on ``n_anodes`` target positions spread evenly over the
    mid-ship length fraction ``anode_span``; every bottom panel whose centroid
    falls inside the footprint belongs to the anode (falling back to the nearest
    panel if the mesh is too coarse).  Fixing the footprint rather than "one
    panel per anode" keeps the anode area — and therefore the galvanic operating
    point — independent of the mesh, which is what makes a mesh-convergence study
    meaningful.

    ``beta`` is the *area* fraction of the remaining (non-anode) wetted surface
    left bare: panels are visited in a fixed pseudo-random order (``seed``) until
    the accumulated area reaches ``beta * A_cathode``, so the achieved ratio is
    reported alongside the requested one.
    """
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie in (0, 1)")
    if n_anodes < 1:
        raise ValueError("n_anodes must be >= 1")
    areas = mesh.areas
    centroids = mesh.centroids
    bottom = np.sort(groups["bottom"])
    xs = centroids[bottom, 0]
    span = 0.5 * anode_span * (xs.max() - xs.min())
    targets = np.linspace(-span, span, n_anodes) if n_anodes > 1 else np.zeros(1)

    half_length, half_breadth = 0.5 * anode_footprint[0], 0.5 * anode_footprint[1]
    anode_list: list[int] = []
    for target in targets:
        inside = [
            int(index)
            for index in bottom
            if abs(centroids[index, 0] - target) <= half_length
            and abs(centroids[index, 1]) <= half_breadth
        ]
        if not inside:
            inside = [int(bottom[np.argmin(np.abs(xs - target))])]
        for index in inside:
            if index not in anode_list:
                anode_list.append(index)
    anodes = np.sort(np.asarray(anode_list, dtype=np.int64))

    mask = np.zeros(mesh.panel_count, dtype=bool)
    mask[anodes] = True
    lid_parts = [groups[name] for name in insulating_groups if name in groups]
    waterline = (
        np.sort(np.concatenate(lid_parts)) if lid_parts else np.zeros(0, dtype=np.int64)
    )
    mask[waterline] = True
    cathode = np.where(~mask)[0]
    cathode_area = float(np.sum(areas[cathode]))
    target_area = beta * cathode_area

    rng = np.random.default_rng(seed)
    shuffled = cathode[rng.permutation(cathode.size)]
    cumulative = np.cumsum(areas[shuffled])
    n_bare = int(np.searchsorted(cumulative, target_area, side="left")) + 1
    n_bare = min(max(n_bare, 1), cathode.size - 1)
    bare = np.sort(shuffled[:n_bare])
    bare_mask = np.zeros(mesh.panel_count, dtype=bool)
    bare_mask[bare] = True
    coated = np.sort(cathode[~bare_mask[cathode]])

    bare_area = float(np.sum(areas[bare]))
    coated_area = float(np.sum(areas[coated]))
    anode_area = float(np.sum(areas[anodes]))
    anode_area_target = (
        float(n_anodes) * float(anode_footprint[0] * anode_footprint[1])
        if anode_area > 0.0
        else 0.0
    )
    return SurfacePartition(
        bare=bare,
        coated=coated,
        anodes=anodes,
        waterline=waterline,
        anode_centers=centroids[anodes],
        requested_beta=float(beta),
        achieved_beta=bare_area / (bare_area + coated_area),
        bare_area=bare_area,
        coated_area=coated_area,
        anode_area=anode_area,
        waterline_area=float(np.sum(areas[waterline])) if waterline.size else 0.0,
        anode_area_target=anode_area_target,
    )


# ===========================================================================
# 6. Non-linear galvanic protection solve
# ===========================================================================
#
# Formulation (the standard CP boundary-element formulation, and the one that
# makes the DNV envelope meaningful): the steel structure and the anodes form a
# single equipotential metal whose potential ``V_m`` is unknown; only the
# *electrolyte* potential ``phi`` varies over the hull.  The quantity that
# DNV-RP-B401 constrains is the electrode potential
#
#       E(x) = V_m - phi(x)        [V vs Ag/AgCl],
#
# which therefore varies over the hull exactly by the electrolyte IR drop.  The
# floating metal potential ``V_m`` is fixed by charge conservation on the
# structure, ``sum_j i_out,j A_j = 0``, i.e. anodic (anode dissolution) and
# cathodic (oxygen reduction) currents balance.  Every panel is therefore a
# Robin-type boundary condition and there is no prescribed electrode potential
# anywhere: the closed-circuit anode potential enters only through the Al
# kinetics, as it does physically.


@dataclass(frozen=True)
class ProtectionResult:
    """Steady-state galvanic protection solution on the hull surface."""

    #: Electrode potential ``E = V_m - phi`` [V vs Ag/AgCl] (DNV quantity).
    potential: FloatArray
    #: Electrolyte potential ``phi`` [V] (reference: zero at infinity).
    electrolyte_potential: FloatArray
    #: Floating potential of the equipotential metal [V].
    metal_potential: float
    #: Sea-side normal derivative ``q = dphi/dn`` [V/m].
    flux: FloatArray
    #: Signed current density leaving the metal [A/m^2] (>0 anodic).
    current_density: FloatArray
    partition: SurfacePartition
    anode_current_total: float
    anode_current_per_anode: FloatArray
    steel_cathodic_current_total: float
    steel_anodic_current_total: float
    charge_imbalance: float
    phi_min: float
    phi_max: float
    phi_min_bare: float
    phi_max_bare: float
    protected_area_fraction: float
    overprotected_area: float
    newton_iterations: int
    nonlinear_residual: float
    linear_iterations: int
    linear_solver: str
    conductivity: float
    anode_potential: float
    kinetics: SteelKinetics
    anode_kinetics: AnodeKinetics
    mesh: TriangleMesh
    panel_areas: FloatArray


def _solve_linear_system(matrix: FloatArray, rhs: FloatArray, method: str) -> tuple[FloatArray, int]:
    if method == "lu":
        return np.linalg.solve(matrix, rhs), 0
    if method != "gmres":
        raise ValueError("linear solver must be 'lu' or 'gmres'")
    from scipy.sparse.linalg import LinearOperator, gmres

    diagonal = np.diag(matrix).copy()
    diagonal[np.abs(diagonal) < 1.0e-30] = 1.0
    inverse = 1.0 / diagonal
    operator = LinearOperator(matrix.shape, matvec=lambda v: matrix @ v, dtype=float)
    preconditioner = LinearOperator(matrix.shape, matvec=lambda v: inverse * v, dtype=float)
    counter = {"count": 0}

    def callback(_residual: float) -> None:
        counter["count"] += 1

    solution, info = gmres(
        operator,
        rhs,
        M=preconditioner,
        rtol=1.0e-6,
        atol=0.0,
        restart=200,
        maxiter=50,
        callback=callback,
        callback_type="pr_norm",
    )
    if info != 0:
        raise RuntimeError(f"GMRES did not converge (info={info})")
    return solution, counter["count"]


def _panel_fluxes(
    mesh: TriangleMesh,
    partition: SurfacePartition,
    electrode_potential: FloatArray,
    kinetics: SteelKinetics,
    anode_kinetics: AnodeKinetics,
    sigma: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``(q, dq/dphi, dq/dV_m)`` for the current electrode potential.

    ``q = (1/sigma) i_out(E)`` with ``E = V_m - phi``; the derivatives are with
    respect to the *electrolyte* potential ``phi_j`` and the (single) metal
    potential ``V_m``.
    """
    n = mesh.panel_count
    bare_mask = np.zeros(n, dtype=bool)
    bare_mask[partition.bare] = True
    anode_mask = np.zeros(n, dtype=bool)
    anode_mask[partition.anodes] = True
    q = np.zeros(n, dtype=float)
    dq_dphi = np.zeros(n, dtype=float)
    dq_dvm = np.zeros(n, dtype=float)

    if bare_mask.any():
        net = kinetics.net_current_density(electrode_potential[bare_mask])
        d_net = kinetics.derivative(electrode_potential[bare_mask])
        q[bare_mask] = -net / sigma
        # d q / d phi = -i'(E) * dE/dphi / sigma = +i'(E)/sigma  (i' < 0)
        dq_dphi[bare_mask] = d_net / sigma
        dq_dvm[bare_mask] = -d_net / sigma
    if anode_mask.any():
        # the discrete anode area is normalised to the design area so that the
        # galvanic operating point does not depend on the mesh resolution
        scale = partition.anode_scale
        out = scale * anode_kinetics.current_density(electrode_potential[anode_mask])
        d_out = scale * anode_kinetics.derivative(electrode_potential[anode_mask])
        q[anode_mask] = out / sigma
        dq_dphi[anode_mask] = -d_out / sigma
        dq_dvm[anode_mask] = d_out / sigma
    return q, dq_dphi, dq_dvm


def _assemble_galvanic_system(
    single: FloatArray,
    double: FloatArray,
    areas: FloatArray,
    q: FloatArray,
    dq_dphi: FloatArray,
    dq_dvm: FloatArray,
    phi: FloatArray,
    metal_potential: float,
) -> tuple[FloatArray, FloatArray]:
    """Linearised Newton system for the unknowns ``[phi_1 .. phi_n, V_m]``.

    Rows ``0..n-1`` are the exterior discrete boundary integral equations
    ``sum_j S_ij q_j - 0.5 phi_i + sum_j H_ij phi_j = 0``; row ``n`` is charge
    conservation on the structure ``sum_j A_j q_j = 0``.
    """
    n = single.shape[0]
    matrix = np.empty((n + 1, n + 1), dtype=float)
    matrix[:n, :n] = single * dq_dphi[None, :] + double
    matrix[np.arange(n), np.arange(n)] -= 0.5
    matrix[:n, n] = single @ dq_dvm
    matrix[n, :n] = areas * dq_dphi
    matrix[n, n] = float(areas @ dq_dvm)

    known = dq_dphi * phi + dq_dvm * metal_potential - q
    rhs = np.empty(n + 1, dtype=float)
    rhs[:n] = single @ known
    rhs[n] = float(areas @ known)
    return matrix, rhs


def _nonlinear_residuals(
    single: FloatArray,
    double: FloatArray,
    areas: FloatArray,
    phi: FloatArray,
    q: FloatArray,
) -> tuple[float, float]:
    """Boundary-integral and charge-balance residuals (relative)."""
    field = single @ q - 0.5 * phi + double @ phi
    scale = float(np.linalg.norm(single @ q))
    field_r = float(np.linalg.norm(field)) / scale if scale > 0.0 else float(np.linalg.norm(field))
    current = float(np.sum(q * areas))
    magnitude = float(np.sum(np.abs(q) * areas))
    charge_r = abs(current) / magnitude if magnitude > 0.0 else abs(current)
    return field_r, charge_r


def _equilibrium_metal_potential(
    mesh: TriangleMesh,
    partition: SurfacePartition,
    electrolyte_potential: FloatArray,
    kinetics: SteelKinetics,
    anode_kinetics: AnodeKinetics,
    lower: float = -3.0,
    upper: float = 1.0,
    iterations: int = 80,
) -> float:
    """Bisection on ``V_m`` so that the structure carries no net current."""

    def imbalance(metal_potential: float) -> float:
        q, _, _ = _panel_fluxes(
            mesh,
            partition,
            metal_potential - electrolyte_potential,
            kinetics,
            anode_kinetics,
            1.0,
        )
        return float(np.sum(q * mesh.areas))

    f_low = imbalance(lower)
    f_high = imbalance(upper)
    if f_low > 0.0 or f_high < 0.0:
        raise RuntimeError(
            f"charge-balance bracket does not contain a root (f({lower})={f_low:.3e}, "
            f"f({upper})={f_high:.3e})"
        )
    for _ in range(iterations):
        mid = 0.5 * (lower + upper)
        if imbalance(mid) < 0.0:
            lower = mid
        else:
            upper = mid
    return 0.5 * (lower + upper)


def _finalise_result(
    mesh: TriangleMesh,
    partition: SurfacePartition,
    electrolyte_potential: FloatArray,
    metal_potential: float,
    kinetics: SteelKinetics,
    anode_kinetics: AnodeKinetics,
    sigma: float,
    iterations: int,
    residual: tuple[float, float],
    linear_iterations: int,
    linear_solver: str,
) -> ProtectionResult:
    n = mesh.panel_count
    areas = mesh.areas
    electrode = metal_potential - electrolyte_potential
    q, _, _ = _panel_fluxes(mesh, partition, electrode, kinetics, anode_kinetics, sigma)
    current = sigma * q  # signed current density leaving the metal [A/m^2]
    anode_currents = current[partition.anodes] * areas[partition.anodes]
    steel_current = current[partition.bare] * areas[partition.bare]
    total = float(np.sum(current * areas))
    magnitude = float(np.sum(np.abs(current) * areas))
    wetted = partition.wetted_indices(n)
    wetted_phi = electrode[wetted]
    wetted_area = float(np.sum(areas[wetted]))
    protected = wetted_phi <= PROTECTION_POTENTIAL
    over = wetted_phi < OVERPROTECTION_POTENTIAL - 1.0e-12
    return ProtectionResult(
        potential=electrode,
        electrolyte_potential=electrolyte_potential,
        metal_potential=float(metal_potential),
        flux=q,
        current_density=current,
        partition=partition,
        anode_current_total=float(np.sum(anode_currents)),
        anode_current_per_anode=anode_currents,
        steel_cathodic_current_total=float(np.sum(np.maximum(-steel_current, 0.0))),
        steel_anodic_current_total=float(np.sum(np.maximum(steel_current, 0.0))),
        charge_imbalance=abs(total) / magnitude if magnitude > 0.0 else abs(total),
        phi_min=float(np.min(wetted_phi)),
        phi_max=float(np.max(wetted_phi)),
        phi_min_bare=float(np.min(electrode[partition.bare])),
        phi_max_bare=float(np.max(electrode[partition.bare])),
        protected_area_fraction=float(np.sum(areas[wetted][protected]) / wetted_area),
        overprotected_area=float(np.sum(areas[wetted][over])),
        newton_iterations=int(iterations),
        nonlinear_residual=float(max(residual)),
        linear_iterations=int(linear_iterations),
        linear_solver=linear_solver,
        conductivity=float(sigma),
        anode_potential=float(anode_kinetics.ocp),
        kinetics=kinetics,
        anode_kinetics=anode_kinetics,
        mesh=mesh,
        panel_areas=areas,
    )


def solve_galvanic_protection(
    mesh: TriangleMesh,
    partition: SurfacePartition,
    kinetics: SteelKinetics | None = None,
    anode_kinetics: AnodeKinetics | None = None,
    sigma: float = SEAWATER_CONDUCTIVITY,
    initial_metal_potential: float = -1.0,
    tolerance: float = 1.0e-9,
    max_iterations: int = 60,
    linear_solver: str = "lu",
    matrices: tuple[FloatArray, FloatArray] | None = None,
) -> ProtectionResult:
    """Solve the non-linear galvanic protection problem by damped Newton.

    Unknowns: the electrolyte potential on every panel plus the single floating
    metal potential ``V_m``.  Bare steel obeys Butler-Volmer kinetics with an
    oxygen-transport limit, anodes obey Al anodic Tafel kinetics, intact coating
    and the waterline lid are insulating, and charge conservation
    ``sum i_out A = 0`` closes the system.
    """
    if kinetics is None:
        kinetics = SteelKinetics()
    if anode_kinetics is None:
        anode_kinetics = AnodeKinetics()
    if partition.anodes.size == 0:
        raise ValueError("at least one anode panel is required (galvanic coupling)")
    single, double = matrices if matrices is not None else assemble_surface_matrices(mesh)
    n = mesh.panel_count
    areas = mesh.areas

    phi = np.zeros(n, dtype=float)
    metal_potential = float(initial_metal_potential)
    electrode = metal_potential - phi
    q, dq_dphi, dq_dvm = _panel_fluxes(mesh, partition, electrode, kinetics, anode_kinetics, sigma)
    field_residual, charge_residual = _nonlinear_residuals(single, double, areas, phi, q)
    residual = max(field_residual, charge_residual)

    iterations = 0
    linear_iterations = 0
    for iterations in range(1, max_iterations + 1):
        matrix, rhs = _assemble_galvanic_system(
            single, double, areas, q, dq_dphi, dq_dvm, phi, metal_potential
        )
        solution, steps = _solve_linear_system(matrix, rhs, linear_solver)
        linear_iterations += steps
        phi_new = solution[:n]
        metal_new = float(solution[n])

        damping = 1.0
        best = None
        for _ in range(14):
            trial_phi = phi + damping * (phi_new - phi)
            trial_metal = metal_potential + damping * (metal_new - metal_potential)
            trial_q, _, _ = _panel_fluxes(
                mesh, partition, trial_metal - trial_phi, kinetics, anode_kinetics, sigma
            )
            field_r, charge_r = _nonlinear_residuals(
                single, double, areas, trial_phi, trial_q
            )
            trial_residual = max(field_r, charge_r)
            if trial_residual < residual or damping < 1.0e-3:
                best = (trial_phi, trial_metal, trial_q, trial_residual, field_r, charge_r)
                break
            damping *= 0.5
        if best is None:  # pragma: no cover - defensive
            raise RuntimeError("Newton line search failed")
        phi_new, metal_new, q_new, residual, field_residual, charge_residual = best
        step = max(
            float(np.max(np.abs(phi_new - phi))), abs(metal_new - metal_potential)
        )
        phi, metal_potential, q = phi_new, metal_new, q_new
        dq_dphi, dq_dvm = _panel_fluxes(
            mesh, partition, metal_potential - phi, kinetics, anode_kinetics, sigma
        )[1:]
        if step < tolerance and residual < 1.0e-10:
            break

    return _finalise_result(
        mesh,
        partition,
        phi,
        metal_potential,
        kinetics,
        anode_kinetics,
        sigma,
        iterations,
        (field_residual, charge_residual),
        linear_iterations,
        linear_solver,
    )


def solve_galvanic_picard(
    mesh: TriangleMesh,
    partition: SurfacePartition,
    kinetics: SteelKinetics | None = None,
    anode_kinetics: AnodeKinetics | None = None,
    sigma: float = SEAWATER_CONDUCTIVITY,
    tolerance: float = 1.0e-9,
    max_iterations: int = 300,
    relaxation: float = 0.15,
) -> ProtectionResult:
    """Independent (Picard) solution of the same non-linear problem.

    Each sweep (i) freezes the current densities on the *previous* iterate,
    (ii) re-solves the linear mixed boundary-value problem, and (iii) updates the
    metal potential by scalar bisection on the charge balance.  It shares no
    Jacobian assembly with :func:`solve_galvanic_protection`, so agreement
    between the two is a genuine cross-check of the non-linear solve.
    """
    if kinetics is None:
        kinetics = SteelKinetics()
    if anode_kinetics is None:
        anode_kinetics = AnodeKinetics()
    if partition.anodes.size == 0:
        raise ValueError("at least one anode panel is required (galvanic coupling)")
    single, double = assemble_surface_matrices(mesh)
    n = mesh.panel_count
    # exterior Neumann operator: S q = (0.5 I - H) phi  (well conditioned in 3-D)
    neumann_operator = 0.5 * np.eye(n) - double
    phi = np.zeros(n, dtype=float)
    metal_potential = -1.0
    for iteration in range(1, max_iterations + 1):
        electrode = metal_potential - phi
        q, _, _ = _panel_fluxes(mesh, partition, electrode, kinetics, anode_kinetics, sigma)
        solved = np.linalg.solve(neumann_operator, single @ q)
        updated = phi + relaxation * (solved - phi)
        metal_new = _equilibrium_metal_potential(
            mesh, partition, updated, kinetics, anode_kinetics
        )
        step = max(float(np.max(np.abs(updated - phi))), abs(metal_new - metal_potential))
        phi, metal_potential = updated, metal_new
        if step < tolerance:
            break
    electrode = metal_potential - phi
    q, _, _ = _panel_fluxes(mesh, partition, electrode, kinetics, anode_kinetics, sigma)
    field_r, charge_r = _nonlinear_residuals(single, double, mesh.areas, phi, q)
    return _finalise_result(
        mesh, partition, phi, metal_potential, kinetics, anode_kinetics, sigma,
        iteration, (field_r, charge_r), 0, "picard",
    )


# ===========================================================================
# 7. DNV-RP-B401 envelope and mesh-convergence diagnostics
# ===========================================================================


def dnv_envelope(
    result: ProtectionResult,
    protection: float = PROTECTION_POTENTIAL,
    overprotection: float = OVERPROTECTION_POTENTIAL,
) -> dict[str, Any]:
    """Check the DNV-RP-B401 protection window on the whole wetted surface.

    The published criterion is ``-1.05 V <= phi <= -0.80 V`` vs Ag/AgCl: every
    point must be at least as negative as the protection potential (``phi <=
    -0.80``) and no more negative than the over-protection limit (``phi >=
    -1.05``).  ``window_ok`` encodes exactly that pair of inequalities.  The
    alternative parsing (``phi_min >= -0.80`` and ``phi_max <= -1.05``) is
    unsatisfiable when ``phi_min <= phi_max`` and is reported as
    ``literal_reading_satisfiable = False`` for the avoidance of doubt.
    """
    n = result.mesh.panel_count
    steel = np.setdiff1d(result.partition.wetted_indices(n), result.partition.anodes)
    steel_potential = result.potential[steel]
    phi_min = float(np.min(steel_potential))
    phi_max = float(np.max(steel_potential))
    return {
        "phi_min": phi_min,
        "phi_max": phi_max,
        "phi_min_all_including_anodes": result.phi_min,
        "phi_max_all_including_anodes": result.phi_max,
        "criterion_surface": "steel (bare holidays + coating), anodes excluded",
        "protection_potential": protection,
        "overprotection_potential": overprotection,
        "all_points_protected": bool(phi_max <= protection),
        "no_overprotection": bool(phi_min >= overprotection),
        "window_ok": bool(phi_max <= protection and phi_min >= overprotection),
        "protected_area_fraction": result.protected_area_fraction,
        "overprotected_area_m2": result.overprotected_area,
        "literal_reading_satisfiable": False,
        "literal_reading_note": (
            "phi_min >= -0.80 V together with phi_max <= -1.05 V contradicts "
            "phi_min <= phi_max; the DNV window -1.05 <= phi <= -0.80 is used."
        ),
    }


def grid_convergence_index(
    fine: float,
    medium: float,
    coarse: float,
    h_fine: float,
    h_medium: float,
    h_coarse: float,
    safety_factor: float = 1.25,
) -> dict[str, Any]:
    """Roache grid-convergence index from three solutions of a scalar metric.

    ``h`` are representative cell sizes (``h_coarse > h_medium > h_fine``).  The
    observed order ``p`` solves ``(f3-f2)/(f2-f1) = (r31^p - r21^p)/(r21^p - 1)``
    with ``r21 = h2/h1``, ``r31 = h3/h1``; the reported GCI is the fine-grid
    value in percent of ``f_fine``.
    """
    values = [float(fine), float(medium), float(coarse)]
    h = [float(h_fine), float(h_medium), float(h_coarse)]
    e21 = values[1] - values[0]
    e32 = values[2] - values[1]
    r21 = h[1] / h[0]
    r31 = h[2] / h[0]
    if abs(e21) < 1.0e-18 or abs(e32) < 1.0e-18:
        return {
            "values": values,
            "cell_sizes": h,
            "observed_order": None,
            "gci_fine_percent": 0.0,
            "note": "differences below round-off; convergence indistinguishable from zero",
        }
    target = e32 / e21

    def model(p: float) -> float:
        return (r31**p - r21**p) / (r21**p - 1.0)

    low, high = 0.1, 8.0
    if not (min(model(low), model(high)) <= target <= max(model(low), model(high))):
        extrapolated = values[0] - e21 / (r21**2.0 - 1.0)
        gci = (
            safety_factor * abs(extrapolated - values[0]) / abs(values[0])
            if values[0]
            else float("inf")
        )
        return {
            "values": values,
            "cell_sizes": h,
            "observed_order": None,
            "note": "observed order outside [0.1, 8]; GCI evaluated with p = 2",
            "gci_fine_percent": 100.0 * float(gci),
        }
    increasing = model(high) > model(low)
    for _ in range(200):
        mid = 0.5 * (low + high)
        if (model(mid) < target) == increasing:
            low = mid
        else:
            high = mid
    order = 0.5 * (low + high)
    extrapolated = values[0] - e21 / (r21**order - 1.0)
    gci = (
        safety_factor * abs(extrapolated - values[0]) / abs(values[0])
        if values[0]
        else float("inf")
    )
    return {
        "values": values,
        "cell_sizes": h,
        "observed_order": float(order),
        "extrapolated_value": float(extrapolated),
        "gci_fine_percent": 100.0 * float(gci),
    }


#: Box-hull meshes used by the study: ~552 / ~960 / ~2012 triangles.
MESH_LEVELS: dict[str, tuple[int, int, int]] = {
    "coarse": (16, 5, 3),
    "medium": (22, 7, 4),
    "fine": (32, 9, 5),
}


def _cell_size(mesh: TriangleMesh) -> float:
    return float(np.sqrt(np.mean(mesh.areas)))


def _case_summary(
    beta: float, n_anodes: int, result: ProtectionResult, mesh_name: str
) -> dict[str, Any]:
    envelope = dnv_envelope(result)
    partition = result.partition
    return {
        "beta_requested": float(beta),
        "beta_achieved": partition.achieved_beta,
        "n_anodes": int(n_anodes),
        "mesh": mesh_name,
        "panels": int(result.mesh.panel_count),
        "cell_size_m": _cell_size(result.mesh),
        "bare_area_m2": partition.bare_area,
        "coated_area_m2": partition.coated_area,
        "anode_area_m2": partition.anode_area,
        "wetted_area_m2": partition.total_area,
        "waterline_lid_area_m2": partition.waterline_area,
        "anode_current_A": result.anode_current_total,
        "anode_current_per_anode_A": result.anode_current_per_anode.tolist(),
        "anode_current_density_A_m2": result.anode_current_total / partition.anode_area,
        "anode_electrode_potential_mean_V": float(np.mean(result.potential[partition.anodes])),
        "steel_cathodic_current_A": result.steel_cathodic_current_total,
        "steel_anodic_current_A": result.steel_anodic_current_total,
        "bare_mean_current_density_A_m2": (
            result.steel_cathodic_current_total / partition.bare_area
        ),
        "bare_anodic_current_A": result.steel_anodic_current_total,
        "bare_anodic_to_cathodic_ratio": (
            result.steel_anodic_current_total
            / max(result.steel_cathodic_current_total, 1.0e-30)
        ),
        "bare_current_density_limit_fraction": (
            result.steel_cathodic_current_total
            / partition.bare_area
            / result.kinetics.i_limiting
        ),
        "surface_mean_current_density_A_m2": result.anode_current_total / partition.total_area,
        "phi_min": result.phi_min,
        "phi_max": result.phi_max,
        "phi_min_bare": result.phi_min_bare,
        "phi_max_bare": result.phi_max_bare,
        "metal_potential_V": result.metal_potential,
        "phi_min_steel": envelope["phi_min"],
        "phi_max_steel": envelope["phi_max"],
        "phi_mean_bare": float(np.mean(result.potential[partition.bare])),
        "phi_mean_coated": float(np.mean(result.potential[partition.coated])),
        "potential_spread_V": result.phi_max - result.phi_min,
        "quantile_phi_05": float(
            np.quantile(result.potential[result.partition.wetted_indices(result.mesh.panel_count)], 0.05)
        ),
        "quantile_phi_95": float(
            np.quantile(result.potential[result.partition.wetted_indices(result.mesh.panel_count)], 0.95)
        ),
        "protection_coverage": result.protected_area_fraction,
        "overprotected_area_m2": result.overprotected_area,
        "charge_imbalance": result.charge_imbalance,
        "newton_iterations": result.newton_iterations,
        "nonlinear_residual": result.nonlinear_residual,
        "linear_iterations": result.linear_iterations,
        "linear_solver": result.linear_solver,
        "dnv": envelope,
        "dnv_window_ok": envelope["window_ok"],
    }


def run_study(
    output_dir: str | Path | None = None,
    betas: Sequence[float] = (0.01, 0.05, 0.1, 0.2),
    anode_counts: Sequence[int] = (2, 4, 8),
    design_beta: float = 0.05,
    design_anodes: int = 4,
    sigma: float = SEAWATER_CONDUCTIVITY,
    kinetics: SteelKinetics | None = None,
    anode_kinetics: AnodeKinetics | None = None,
    solver: str = "lu",
    picard_crosscheck: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the verification and parameter study, optionally writing JSON.

    Returns a dictionary with the analytic anchors, the mesh-convergence study,
    the ``beta`` x anode-count parameter study, the independent Picard / GMRES
    cross-checks and the machine-checkable acceptance criteria.
    """
    if kinetics is None:
        kinetics = SteelKinetics()
    if anode_kinetics is None:
        anode_kinetics = AnodeKinetics()
    hull = BoxHull()
    outputs: dict[str, Any] = {
        "meta": {
            "module": "tensorbem.cp_galvanic",
            "sigma_S_m": sigma,
            "kinetics": {
                "phi_eq_V": kinetics.phi_eq,
                "i_exchange_A_m2": kinetics.i_exchange,
                "alpha_c": kinetics.alpha_c,
                "alpha_a": kinetics.alpha_a,
                "i_limiting_A_m2": kinetics.i_limiting,
                "tafel_slope_V_dec": kinetics.tafel_slope,
            },
            "hull": {
                "length_m": hull.length,
                "breadth_m": hull.breadth,
                "draft_m": hull.draft,
                "wetted_area_m2": hull.wetted_area,
                "displacement_m3": hull.displacement_volume,
            },
            "dnv": {
                "protection_potential_V": PROTECTION_POTENTIAL,
                "overprotection_potential_V": OVERPROTECTION_POTENTIAL,
                "anode_closed_circuit_potential_V": ANODE_CLOSED_CIRCUIT_POTENTIAL,
                "bare_steel_design_current_density_A_m2": BARE_STEEL_DESIGN_CURRENT_DENSITY,
            },
            "mesh_levels": {
                name: box_hull_panel_count(*spec) for name, spec in MESH_LEVELS.items()
            },
            "linear_solver": solver,
            "anode_kinetics": {
                "ocp_V": anode_kinetics.ocp,
                "i_reference_A_m2": anode_kinetics.i_reference,
                "tafel_slope_V_dec": anode_kinetics.tafel_slope,
            },
        }
    }

    # --- (a),(b) analytic Laplace anchors -----------------------------------
    plate_runs = [
        plate_linear_benchmark(n_horizontal=nx, n_vertical=nz)
        for nx, nz in ((4, 1), (8, 2), (12, 3))
    ]
    cylinder_runs = [
        cylinder_log_benchmark(n_theta=nt, n_radial=nr, n_axial=na)
        for nt, nr, na in ((16, 3, 1), (32, 6, 1), (48, 8, 1))
    ]
    sphere_runs = [sphere_exterior_benchmark(subdivisions=s) for s in (2, 3)]
    outputs["analytic"] = {
        "plate": plate_runs,
        "cylinder": cylinder_runs,
        "sphere": sphere_runs,
    }
    if verbose:
        print("plate rel-L2:", [round(run["potential_rel_l2"], 8) for run in plate_runs])
        print("cylinder rel-L2:", [round(run["potential_rel_l2"], 8) for run in cylinder_runs])
        print("sphere rel-L2:", [round(run["potential_rel_l2"], 8) for run in sphere_runs])

    # --- mesh convergence of the design case --------------------------------
    mesh_runs: dict[str, Any] = {}
    meshes: dict[str, TriangleMesh] = {}
    results: dict[str, ProtectionResult] = {}
    for name, (nx, ny, nz) in MESH_LEVELS.items():
        mesh, groups = box_hull_mesh(hull, nx=nx, ny=ny, nz=nz)
        meshes[name] = mesh
        partition = partition_surface(mesh, groups, design_beta, n_anodes=design_anodes)
        matrices = assemble_surface_matrices(mesh)
        result = solve_galvanic_protection(
            mesh, partition, kinetics=kinetics, anode_kinetics=anode_kinetics,
            sigma=sigma, linear_solver=solver, matrices=matrices,
        )
        results[name] = result
        mesh_runs[name] = _case_summary(design_beta, design_anodes, result, name)
        if verbose:
            print(
                f"design mesh={name:6s} panels={mesh.panel_count:5d} "
                f"I_anode={result.anode_current_total:8.3f} A "
                f"phi=[{result.phi_min:.4f},{result.phi_max:.4f}] V "
                f"newton={result.newton_iterations:2d} res={result.nonlinear_residual:.2e} "
                f"imb={result.charge_imbalance:.2e}"
            )
    outputs["mesh_convergence"] = {"cases": mesh_runs, "gci": {}}
    outputs["mesh_convergence"]["gci"]["anode_current_A"] = grid_convergence_index(
        mesh_runs["fine"]["anode_current_A"],
        mesh_runs["medium"]["anode_current_A"],
        mesh_runs["coarse"]["anode_current_A"],
        mesh_runs["fine"]["cell_size_m"],
        mesh_runs["medium"]["cell_size_m"],
        mesh_runs["coarse"]["cell_size_m"],
    )
    outputs["mesh_convergence"]["gci"]["phi_mean_bare_V"] = grid_convergence_index(
        mesh_runs["fine"]["phi_mean_bare"],
        mesh_runs["medium"]["phi_mean_bare"],
        mesh_runs["coarse"]["phi_mean_bare"],
        mesh_runs["fine"]["cell_size_m"],
        mesh_runs["medium"]["cell_size_m"],
        mesh_runs["coarse"]["cell_size_m"],
    )

    # --- (d) parameter study on the medium mesh -----------------------------
    medium_mesh = meshes["medium"]
    _, medium_groups = box_hull_mesh(hull, *MESH_LEVELS["medium"])
    medium_matrices = assemble_surface_matrices(medium_mesh)
    study_cases: list[dict[str, Any]] = []
    for beta in betas:
        for n_anodes in anode_counts:
            partition = partition_surface(medium_mesh, medium_groups, beta, n_anodes=n_anodes)
            result = solve_galvanic_protection(
                medium_mesh, partition, kinetics=kinetics, anode_kinetics=anode_kinetics,
                sigma=sigma, linear_solver=solver, matrices=medium_matrices,
            )
            case = _case_summary(beta, n_anodes, result, "medium")
            study_cases.append(case)
            if verbose:
                print(
                    f"beta={beta:5.2f} N={n_anodes}: I={case['anode_current_A']:8.3f} A "
                    f"i_bare={case['bare_mean_current_density_A_m2']:.4f} A/m2 "
                    f"phi=[{case['phi_min']:.3f},{case['phi_max']:.3f}] "
                    f"ok={case['dnv_window_ok']}"
                )
    outputs["parameter_study"] = {
        "betas": list(betas),
        "anode_counts": list(anode_counts),
        "mesh": "medium",
        "cases": study_cases,
    }

    # --- independent cross-checks ------------------------------------------
    coarse_mesh = meshes["coarse"]
    _, coarse_groups = box_hull_mesh(hull, *MESH_LEVELS["coarse"])
    coarse_matrices = assemble_surface_matrices(coarse_mesh)
    coarse_partition = partition_surface(
        coarse_mesh, coarse_groups, design_beta, n_anodes=design_anodes
    )
    # GMRES with a Jacobi preconditioner against dense LU on the first
    # linearised Newton system (rtol = 1e-6 as required by the study design).
    first_phi = np.zeros(coarse_mesh.panel_count, dtype=float)
    first_metal = -1.0
    first_q, first_dq_phi, first_dq_vm = _panel_fluxes(
        coarse_mesh, coarse_partition,
        first_metal - first_phi, kinetics, anode_kinetics, sigma,
    )
    first_matrix, first_rhs = _assemble_galvanic_system(
        coarse_matrices[0], coarse_matrices[1], coarse_mesh.areas,
        first_q, first_dq_phi, first_dq_vm, first_phi, first_metal,
    )
    lu_solution = np.linalg.solve(first_matrix, first_rhs)
    gmres_solution, gmres_steps = _solve_linear_system(first_matrix, first_rhs, "gmres")
    outputs["linear_solver_crosscheck"] = {
        "mesh": "coarse",
        "system_size": int(first_matrix.shape[0]),
        "gmres_tolerance": 1.0e-6,
        "gmres_iterations": int(gmres_steps),
        "lu_gmres_relative_difference": float(
            np.linalg.norm(gmres_solution - lu_solution) / np.linalg.norm(lu_solution)
        ),
        "lu_residual": float(
            np.linalg.norm(first_matrix @ lu_solution - first_rhs) / np.linalg.norm(first_rhs)
        ),
        "gmres_residual": float(
            np.linalg.norm(first_matrix @ gmres_solution - first_rhs) / np.linalg.norm(first_rhs)
        ),
        "note": "production solves use dense LU; GMRES verifies the same system to 1e-6",
    }
    if picard_crosscheck:
        result_picard = solve_galvanic_picard(
            coarse_mesh, coarse_partition, kinetics=kinetics,
            anode_kinetics=anode_kinetics, sigma=sigma,
        )
        outputs["picard_crosscheck"] = {
            "mesh": "coarse",
            "newton_anode_current_A": results["coarse"].anode_current_total,
            "picard_anode_current_A": result_picard.anode_current_total,
            "relative_difference": abs(
                result_picard.anode_current_total - results["coarse"].anode_current_total
            )
            / abs(results["coarse"].anode_current_total),
            "max_potential_difference_V": float(
                np.max(np.abs(result_picard.potential - results["coarse"].potential))
            ),
            "picard_iterations": result_picard.newton_iterations,
        }
        if verbose:
            print("picard crosscheck:", outputs["picard_crosscheck"])

    # --- machine-checkable acceptance criteria ------------------------------
    beta_to_case = {
        case["beta_requested"]: case
        for case in study_cases
        if case["n_anodes"] == design_anodes
    }
    gci_current = outputs["mesh_convergence"]["gci"]["anode_current_A"]
    criteria = {
        "a_plate_rel_l2": {
            "value": plate_runs[-1]["potential_rel_l2"],
            "threshold": 1.0e-2,
            "pass": bool(plate_runs[-1]["potential_rel_l2"] < 1.0e-2),
            "detail": "infinite parallel plates, probe-grid potential L2 error vs phi = V z/H",
        },
        "b_cylinder_rel_l2": {
            "value": cylinder_runs[-1]["potential_rel_l2"],
            "threshold": 2.0e-2,
            "pass": bool(cylinder_runs[-1]["potential_rel_l2"] < 2.0e-2),
            "detail": "coaxial infinite cylinders, probe-grid potential L2 error vs phi = phi_a ln(R/r)/ln(R/a)",
        },
        "c_dnv_window_design_case": {
            "phi_min": mesh_runs["fine"]["dnv"]["phi_min"],
            "phi_max": mesh_runs["fine"]["dnv"]["phi_max"],
            "phi_min_including_anodes": mesh_runs["fine"]["phi_min"],
            "phi_max_including_anodes": mesh_runs["fine"]["phi_max"],
            "pass": bool(mesh_runs["fine"]["dnv_window_ok"]),
            "detail": "design case (beta=0.05, N=8, fine mesh): -1.05 V <= phi <= -0.80 V",
        },
        "d_beta_0p01_anode_current": {
            "value": beta_to_case.get(0.01, {}).get("anode_current_A"),
            "threshold": 10.0,
            "pass": bool(beta_to_case.get(0.01, {}).get("anode_current_A", 1.0e9) < 10.0),
            "detail": "anode current at beta=0.01 (medium mesh, N=8) must stay below 10 A",
        },
        "d_beta_0p2_anode_current": {
            "value": beta_to_case.get(0.2, {}).get("anode_current_A"),
            "threshold": 200.0,
            "pass": bool(beta_to_case.get(0.2, {}).get("anode_current_A", 1.0e9) < 200.0),
            "detail": "anode current at beta=0.2 (medium mesh, N=8) must stay below 200 A",
        },
        "e_mesh_gci_anode_current": {
            "value": gci_current["gci_fine_percent"],
            "threshold": 2.0,
            "pass": bool(gci_current["gci_fine_percent"] < 2.0),
            "detail": "Roache GCI of the total anode current on 552/960/2012-panel meshes",
        },
        "f_charge_balance": {
            "value": mesh_runs["fine"]["charge_imbalance"],
            "threshold": 1.0e-3,
            "pass": bool(mesh_runs["fine"]["charge_imbalance"] < 1.0e-3),
            "detail": "|sum(q dS)| / sum(|q| dS) on the fine mesh must be below 0.1 %",
        },
    }
    outputs["criteria"] = criteria
    outputs["all_criteria_pass"] = bool(all(item["pass"] for item in criteria.values()))

    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "results.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
        fine_result = results["fine"]
        np.savez_compressed(
            path / "fine_solution.npz",
            centroids=fine_result.mesh.centroids,
            potential=fine_result.potential,
            flux=fine_result.flux,
            current_density=fine_result.current_density,
            areas=fine_result.panel_areas,
            bare=fine_result.partition.bare,
            coated=fine_result.partition.coated,
            anodes=fine_result.partition.anodes,
            hull=np.array([hull.length, hull.breadth, hull.draft]),
        )
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Galvanic cathodic protection verification study"
    )
    parser.add_argument("--output", default="docs/cathodic_protection_galvanic")
    parser.add_argument("--solver", default="lu", choices=("lu", "gmres"))
    args = parser.parse_args(argv)
    study = run_study(output_dir=args.output, solver=args.solver)
    print(json.dumps(study["criteria"], indent=2))
    print("all criteria pass:", study["all_criteria_pass"])
    return 0 if study["all_criteria_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
