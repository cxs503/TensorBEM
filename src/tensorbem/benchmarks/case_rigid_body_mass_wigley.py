"""CASE-BEM-RIGID-BODY-MASS (Wigley hull): ship-form 6x6 rigid-body mass matrix.

The case-pool anchor ``CASE-BEM-RIGID-BODY-MASS`` is pinned by
:mod:`tensorbem.benchmarks.case_rigid_body_mass` on a **solid ellipsoid**.  An
ellipsoid is convenient but it is not a ship: its block coefficient is
``pi/6`` by construction, it has no waterplane, no fine ends and no keel, so the
geometry gate the ecosystem requires (``Cb / L / B`` validated before any
tolerance statement) cannot be exercised on it.

This module extends the same anchor to the one hull form with ship-like fine
ends whose mass properties are available in **closed form**: the parabolic
Wigley hull,

    y(x, z) = (B/2) (1 - (2x/L)^2) (1 - (z/T)^2),   x in [-L/2, L/2],
                                                    z in [-T, 0],

parameterised by ``x = -L/2 + s L``, ``z = -t T``, ``s, t in [0, 1]`` so that
``y = 2 B s (1-s) (1-t^2)``.  Everything below is an elementary integral of that
parabola; nothing is copied from a table and nothing is fitted.

Closed-form mass properties (``ref``)
-------------------------------------
With ``rho`` the uniform density and ``V = 4 L B T / 9`` (hence
``Cb = V/(L B T) = 4/9``):

===================  ===================================================
quantity             closed form
===================  ===================================================
volume               ``V = 4 L B T / 9``
centroid             ``x_B = 0``, ``z_B = -3T/8``
``int y^2 dV``       ``48 B^2 V / 1225``
``int x^2 dV``       ``L^2 V / 20``
``int z^2 dV``       ``T^2 V / 5``            (about ``z = 0``)
``I_xx`` (roll)      ``rho V (48 B^2/1225 + 19 T^2/320)``  (about the CG)
``I_yy`` (pitch)     ``rho V (L^2/20 + 19 T^2/320)``
``I_zz`` (yaw)       ``rho V (L^2/20 + 48 B^2/1225)``
products of inertia  identically zero (two symmetry planes)
===================  ===================================================

``19 T^2/320 = T^2/5 - (3T/8)^2`` is the parallel-axis shift from the
waterplane ``z = 0`` to the centroid ``z_B``; the length and beam factors
separate because the half-breadth is a product of a parabola in ``x`` and one
in ``z``.  The reduction is recorded so that the numbers can be re-derived and
not merely believed:

* ``int_0^1 s (1-s) ds = 1/6``, ``int_0^1 s^2 (1-s)^2 ds = 1/30``,
  ``int_0^1 s^3 (1-s)^3 ds = 1/140``;
* ``int_0^1 (1-t^2) dt = 2/3``, ``int_0^1 t^2 (1-t^2) dt = 2/15``,
  ``int_0^1 (1-t^2)^3 dt = 16/35``;
* ``int x^2 dV = 4 B L^3 T (1/30) (2/3) + L^2 ... = B L^3 T / 45 = L^2 V / 20``
  (with ``x`` measured from midships, where the ``L^2/4`` shift cancels).

The reference matrix is the closed block form of the rigid-body kinetics
(Fossen 2011, *Handbook of Marine Craft Hydrodynamics and Motion Control*,
section 3.3.1, eqs. (3.44)-(3.53); same partition in Newman 1977 section 4.5),

    M_RB = [[ m I3 , -m S(r_g) ],
            [ m S(r_g) , I_g + m (|r_g|^2 I3 - r_g r_g^T) ]],

with ``S(r)`` the cross-product matrix.  It is assembled here **by hand** from
the closed-form mass properties above, i.e. without calling
:func:`tensorbem.rigid_body_mass.rigid_body_mass_matrix`, so that the
assembler is checked rather than reused.

Three independent routes
------------------------
``R`` (closed form)   the reduction above, evaluated in float64.
``Q`` (quadrature)    a 3-D Gauss-Legendre integral of the same physical
                      definitions over the parametric domain
                      ``(s, eta, t)`` with Jacobian ``L T 2 B s (1-s) (1-t^2)``;
                      the integrands are polynomials, so the quadrature is
                      exact to round-off and any disagreement means a wrong
                      closed form.
``M`` (BEM mesh)      :func:`tensorbem.wigley_hydrostatics.wigley_wetted_hull_mesh`
                      closed with a planar waterplane lid, integrated by
                      :func:`tensorbem.rigid_body_mass.mesh_mass_properties`
                      (signed tetrahedra, exact for a polyhedron), and then
                      assembled through the **platform** path
                      :func:`tensorbem.hydrostatic.compute_mass_matrix` -- the
                      function :mod:`tensorbem.rao` and :mod:`tensorbem.ice_rao`
                      call.  ``M`` is the only route that carries a
                      discretisation error, and that error is what the 3 %
                      case-pool criterion measures.

Acceptance
----------
Entry-wise ``|got - ref| / max(|ref|, 1e-6 max|ref|) < 3 %`` over all 36
entries at the accepted mesh level; the geometry gate ``|Cb_m - 4/9|``,
``|L_m - L| / L`` and ``|B_m - B| / B`` also < 3 % at that level (a mass matrix
of a mis-meshed hull is worthless no matter how self-consistent it is); the
kinetic-energy identity for random twists at round-off on the mesh properties;
and a negative control -- the pre-m17324 transposed coupling block -- that must
break the criterion, otherwise the benchmark cannot discriminate.

Run::

    python -m tensorbem.benchmarks.case_rigid_body_mass_wigley \
        --out docs/cases/case_bem_rigid_body_mass_wigley/run.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from tensorbem.hydrostatic import compute_mass_matrix
from tensorbem.rigid_body_mass import mesh_mass_properties, rigid_body_mass_matrix
from tensorbem.wigley_hydrostatics import wigley_wetted_hull_mesh

from .case_rigid_body_mass import (
    MODES,
    kinetic_energy_error,
    legacy_coupling_matrix,
    relative_error_matrix,
)

__all__ = [
    "DENSITY",
    "HULL",
    "TOLERANCE",
    "WigleyHull",
    "closed_form_mass_properties",
    "hand_block_matrix",
    "mesh_route",
    "quadrature_mass_properties",
    "run",
    "solid_mesh",
    "wigley_halfbreadth",
]

FloatArray = NDArray[np.float64]

#: Case-pool tolerance: per-entry relative error < 3 %.
TOLERANCE = 0.03

#: Water density (ITTC sea water, kg/m^3).
DENSITY = 1025.0

#: Mesh ladder: (n_sections, n_levels) of the wetted-surface grid per side.
MESH_LADDER: tuple[tuple[int, int], ...] = ((6, 3), (12, 6), (24, 12), (48, 24), (96, 48))

REFERENCE = {
    "form": "M_RB = [[m I3, -m S(r_g)], [m S(r_g), I_g + m (|r_g|^2 I3 - r_g r_g^T)]]",
    "source": (
        "Fossen (2011) Handbook of Marine Craft Hydrodynamics and Motion "
        "Control, Wiley, section 3.3.1 (rigid-body kinetics), eqs. (3.44)-(3.53); "
        "same block partition in Newman (1977) Marine Hydrodynamics section 4.5. "
        "Mass properties of the parabolic Wigley hull are elementary integrals of "
        "y = (B/2)(1-(2x/L)^2)(1-(z/T)^2) evaluated in this module (zero hand-copy); "
        "the reduction is listed in the module docstring and re-derived by an "
        "independent Gauss-Legendre route."
    ),
    "numbers": "evaluated by this module from the cited closed forms (zero hand-copy)",
}


class WigleyHull:
    """Geometry, mass model and centre of gravity of the case body.

    Parameters
    ----------
    length, beam, draft:
        ``L``, ``B``, ``T`` of the parabolic Wigley hull (m), measured at the
        calm waterline (``z = 0``) and the design draft (``z = -T``).
    density:
        Uniform density of the body (kg/m^3).
    cg_fractions:
        Centre of gravity as fractions of ``(L, B, T)``, measured from the
        reference point (midships, centreline, waterline).  The default
        ``(0.05, 0.0, -0.45)`` is a ship-like design offset (CG aft of
        midships and below the waterline) that makes every block of ``M_RB``
        non-trivial: the coupling block sees ``r_g`` in ``x`` and ``z``.
    """

    def __init__(
        self,
        length: float = 120.0,
        beam: float = 24.0,
        draft: float = 8.0,
        density: float = DENSITY,
        cg_fractions: Sequence[float] = (0.05, 0.0, -0.45),
    ) -> None:
        for name, value in (("length", length), ("beam", beam), ("draft", draft)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(density) or density <= 0.0:
            raise ValueError("density must be finite and positive")
        self.length = float(length)
        self.beam = float(beam)
        self.draft = float(draft)
        self.density = float(density)
        fractions = np.asarray(cg_fractions, dtype=float).reshape(3)
        if not np.all(np.isfinite(fractions)):
            raise ValueError("cg_fractions must be finite")
        self.cg = fractions * np.array([self.length, self.beam, self.draft])

    # -- helpers ---------------------------------------------------------
    @property
    def block_coefficient(self) -> float:
        """``Cb = 4/9`` of the parabolic Wigley hull (exact, independent of mesh)."""
        return 4.0 / 9.0

    def halfbreadth(self, s: FloatArray, t: FloatArray) -> FloatArray:
        """``y = 2 B s (1-s) (1-t^2)``; ``s`` along the length, ``t`` down the draft."""
        return wigley_halfbreadth(s, t, self.beam)

    def to_dict(self) -> dict[str, Any]:
        return {
            "length_m": self.length,
            "beam_m": self.beam,
            "draft_m": self.draft,
            "density_kg_m3": self.density,
            "cg_m": [float(value) for value in self.cg],
            "cg_fractions_of_LBT": [
                float(self.cg[0] / self.length),
                float(self.cg[1] / self.beam),
                float(self.cg[2] / self.draft),
            ],
            "block_coefficient_analytic": self.block_coefficient,
            "length_beam_ratio": self.length / self.beam,
            "beam_draft_ratio": self.beam / self.draft,
            "halfbreadth": "y = 2 B s (1-s) (1-t^2), x = -L/2 + s L, z = -t T",
            "mode_order": list(MODES),
        }


#: The case body: a 120 m Wigley hull with a ship-like CG offset.
HULL = WigleyHull()


def wigley_halfbreadth(s: FloatArray, t: FloatArray, beam: float) -> FloatArray:
    """Half-breadth ``2 B s (1-s) (1-t^2)`` of the parabolic Wigley hull."""
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)
    return 2.0 * beam * s * (1.0 - s) * (1.0 - t * t)


def transfer_inertia(
    inertia: ArrayLike, mass: float, from_point: ArrayLike, to_point: ArrayLike
) -> FloatArray:
    """Parallel-axis transfer of an inertia tensor between two points.

    ``I_to = I_from + m (|d|^2 I3 - d d^T)`` with ``d = to_point - from_point``.
    The case body is a uniform-density hull whose mass model places the centre
    of gravity at the design point ``hull.cg``; the mesh/discretised inertia is
    first obtained about its own centroid and then transferred to that same
    point, so the reference and the mesh route describe the *same* ten mass
    numbers up to discretisation error.
    """
    i_from = np.asarray(inertia, dtype=float)
    if i_from.shape != (3, 3):
        raise ValueError("inertia must be (3, 3)")
    d = np.asarray(to_point, dtype=float).reshape(3) - np.asarray(
        from_point, dtype=float
    ).reshape(3)
    return i_from + float(mass) * (float(d @ d) * np.eye(3) - np.outer(d, d))


# ---------------------------------------------------------------------------
# route R: closed form
# ---------------------------------------------------------------------------
def closed_form_mass_properties(hull: WigleyHull = HULL) -> dict[str, Any]:
    """Mass properties of the Wigley solid from the closed-form integrals (route R).

    Returns a dict with ``volume``, ``block_coefficient``, ``centroid``,
    ``mass``, ``inertia_cg`` (3x3, about the centroid), the three second-moment
    integrals and the individual terms of each diagonal entry so the reduction
    can be audited term by term.
    """
    L, B, T = hull.length, hull.beam, hull.draft
    volume = 4.0 * L * B * T / 9.0
    centroid = np.array([0.0, 0.0, -3.0 * T / 8.0])
    integral_x2 = volume * L * L / 20.0
    integral_y2 = volume * (48.0 * B * B / 1225.0)
    integral_z2 = volume * (T * T / 5.0)
    shift = volume * (3.0 * T / 8.0) ** 2  # parallel axis: z = 0 -> z_B
    inertia_cg = hull.density * np.diag(
        [
            integral_y2 + integral_z2 - shift,
            integral_x2 + integral_z2 - shift,
            integral_x2 + integral_y2,
        ]
    )
    mass = hull.density * volume
    inertia_at_cg = transfer_inertia(inertia_cg, mass, centroid, hull.cg)
    return {
        "volume": volume,
        "block_coefficient": volume / (L * B * T),
        "centroid": centroid,
        "mass": mass,
        "inertia_cg": inertia_cg,
        "inertia_at_design_cg": inertia_at_cg,
        "cg_transfer_offset": centroid - np.asarray(hull.cg, dtype=float),
        "second_moments": {
            "int_x2_dV": integral_x2,
            "int_y2_dV": integral_y2,
            "int_z2_dV": integral_z2,
            "int_z2_dV_about_centroid": integral_z2 - shift,
        },
        "inertia_terms": {
            "I_xx": [integral_y2 * hull.density, (integral_z2 - shift) * hull.density],
            "I_yy": [integral_x2 * hull.density, (integral_z2 - shift) * hull.density],
            "I_zz": [integral_x2 * hull.density, integral_y2 * hull.density],
        },
        "radii_of_gyration": np.sqrt(
            np.diag(inertia_cg) / (hull.density * volume)
        ).tolist(),
        "products_of_inertia": [0.0, 0.0, 0.0],
    }


# ---------------------------------------------------------------------------
# route Q: 3-D Gauss-Legendre quadrature of the definitions
# ---------------------------------------------------------------------------
def quadrature_mass_properties(
    hull: WigleyHull = HULL, order: int = 24
) -> dict[str, Any]:
    """Volume, centroid and inertia by Gauss-Legendre quadrature over ``(s, eta, t)``.

    The map is ``x = -L/2 + s L``, ``z = -t T``, ``y = eta * y_max(s, t)`` with
    Jacobian ``L T y_max``; over ``s, t in [0, 1]`` and ``eta in [-1, 1]`` this
    covers the solid exactly once.  Every integrand (volume, first and second
    moments) is a polynomial of degree <= 3 in ``s, t`` and <= 2 in ``eta``
    times the Jacobian, so ``order = 24`` Gauss points per direction integrate
    them to round-off: this route is an *independent evaluation of the
    definition*, not a second approximation of it.
    """
    if order < 4:
        raise ValueError("order must be at least 4")
    nodes_s, weights_s = np.polynomial.legendre.leggauss(order)
    grid_s = 0.5 * (nodes_s + 1.0)
    grid_t = 0.5 * (nodes_s + 1.0)
    weight_s = 0.5 * weights_s
    weight_t = 0.5 * weights_s
    weight_eta = weights_s

    S, T, E = np.meshgrid(grid_s, grid_t, nodes_s, indexing="ij")
    W = weight_s[:, None, None] * weight_t[None, :, None] * weight_eta[None, None, :]
    y_max = wigley_halfbreadth(S, T, hull.beam)
    jacobian = hull.length * hull.draft * y_max
    x = -0.5 * hull.length + S * hull.length
    z = -T * hull.draft
    y = E * y_max

    weight = W * jacobian
    volume = float(np.sum(weight))
    centroid = np.array(
        [
            float(np.sum(weight * x)) / volume,
            float(np.sum(weight * y)) / volume,
            float(np.sum(weight * z)) / volume,
        ]
    )
    int_x2 = float(np.sum(weight * x * x))
    int_y2 = float(np.sum(weight * y * y))
    int_z2 = float(np.sum(weight * z * z))
    shift = float(volume * centroid[2] ** 2)
    inertia_cg = hull.density * np.diag(
        [int_y2 + int_z2 - shift, int_x2 + int_z2 - shift, int_x2 + int_y2]
    )
    mass = hull.density * volume
    return {
        "volume": volume,
        "block_coefficient": volume / (hull.length * hull.beam * hull.draft),
        "centroid": centroid,
        "mass": mass,
        "inertia_cg": inertia_cg,
        "inertia_at_design_cg": transfer_inertia(inertia_cg, mass, centroid, hull.cg),
        "second_moments": {
            "int_x2_dV": int_x2,
            "int_y2_dV": int_y2,
            "int_z2_dV": int_z2,
            "int_z2_dV_about_centroid": int_z2 - shift,
        },
        "quadrature_order": order,
    }


# ---------------------------------------------------------------------------
# route M: closed BEM mesh + signed-tetrahedron integration
# ---------------------------------------------------------------------------
def solid_mesh(hull: WigleyHull = HULL, n_sections: int = 48, n_levels: int = 24):
    """Closed (watertight) triangulation of the Wigley solid.

    The wetted surface comes from
    :func:`tensorbem.wigley_hydrostatics.wigley_wetted_hull_mesh` (the generator
    used by the hydrostatics benchmarks); its single boundary is the calm
    waterline loop, which is closed here with a triangle fan to the planform
    centroid ``(0, 0, 0)``.  The planform ``|y| <= 2 B s (1-s)`` is star-shaped
    about that point, so the fan is a valid triangulation of the waterplane and
    the closed surface encloses the displaced volume with outward normals.

    Returns
    -------
    vertices, faces, meta :
        ``(V, 3)`` float array, ``(F, 3)`` int array and a dict with the
        wetted/lid panel counts and the waterline vertex count.
    """
    wetted = wigley_wetted_hull_mesh(hull.length, hull.beam, hull.draft, n_sections, n_levels)
    vertices = np.asarray(wetted.vertices, dtype=float).copy()
    faces = np.asarray(wetted.faces, dtype=np.int64)

    interior = np.where(np.abs(vertices[:, 1]) > 1.0e-12)[0]
    x_min = float(vertices[interior, 0].min()) if interior.size else 0.0
    x_max = float(vertices[interior, 0].max()) if interior.size else 0.0
    apex_index = len(vertices)
    vertices = np.vstack([vertices, np.array([[0.0, 0.0, 0.0]])])
    del x_min, x_max

    lid_edges = np.asarray(wetted.waterline_edges, dtype=np.int64)
    lid_faces = []
    for edge in lid_edges:
        i, j = int(edge[0]), int(edge[1])
        if vertices[i, 2] < -1.0e-12 or vertices[j, 2] < -1.0e-12:
            continue  # only the waterline belongs to the lid
        lid_faces.append((apex_index, j, i))
    if not lid_faces:
        raise ValueError("wetted mesh has no waterline edges to close")
    lid = np.asarray(lid_faces, dtype=np.int64)

    closed_faces = np.vstack([faces, lid])
    # The waterline half-edges are oriented along the wetted surface (outward,
    # i.e. into the fluid), so each lid triangle must traverse its edge the
    # other way round -- ``(apex, j, i)``, not ``(apex, i, j)`` -- otherwise the
    # lid half-edges coincide with the wetted ones and the surface is not
    # watertight.  The winding is still probed below and flipped if the enclosed
    # volume comes out negative.
    probe = mesh_mass_properties(vertices, closed_faces, hull.density)
    if float(probe["volume"]) <= 0.0:
        lid = lid[:, ::-1]
        closed_faces = np.vstack([faces, lid])
        probe = mesh_mass_properties(vertices, closed_faces, hull.density)
        if float(probe["volume"]) <= 0.0:
            raise ValueError("could not orient the waterplane lid outward")
    return vertices, closed_faces, {
        "wetted_panels": int(faces.shape[0]),
        "lid_panels": int(lid.shape[0]),
        "waterline_vertices": int(np.unique(lid[:, 1:]).size),
        "n_sections": int(n_sections),
        "n_levels": int(n_levels),
    }


def mesh_route(hull: WigleyHull = HULL, n_sections: int = 48, n_levels: int = 24) -> dict[str, Any]:
    """Mass properties and 6x6 mass matrix from the closed BEM mesh (route M).

    The inertia tensor is integrated with signed tetrahedra
    (:func:`tensorbem.rigid_body_mass.mesh_mass_properties`) and the matrix is
    then assembled by the **platform** entry point
    :func:`tensorbem.hydrostatic.compute_mass_matrix` -- the same call
    :mod:`tensorbem.rao` makes -- so the route measures the assembled matrix,
    not a re-implementation of it.
    """
    vertices, faces, meta = solid_mesh(hull, n_sections, n_levels)
    props = mesh_mass_properties(vertices, faces, hull.density)
    mass = float(props["mass"])
    centroid = np.asarray(props["centroid"], dtype=float)
    inertia_cg = np.asarray(props["inertia_cg"], dtype=float)
    # Same mass model as the reference: CG at the design point ``hull.cg``, the
    # mesh inertia transferred there by the parallel-axis theorem.
    inertia_at_cg = transfer_inertia(inertia_cg, mass, centroid, hull.cg)
    matrix = compute_mass_matrix(mass, hull.cg, inertia_at_cg, inertia_about="cg")
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    return {
        "mesh": meta,
        "mass": mass,
        "volume": float(props["volume"]),
        "block_coefficient": float(props["volume"])
        / (hull.length * hull.beam * hull.draft),
        "centroid": centroid,
        "inertia_cg": inertia_cg,
        "inertia_at_design_cg": inertia_at_cg,
        "matrix": matrix,
        "bbox": {"length": float(hi[0] - lo[0]), "beam": float(hi[1] - lo[1])},
        "energy_identity_error": kinetic_energy_error(
            matrix, mass, hull.cg, inertia_at_cg
        ),
    }


# ---------------------------------------------------------------------------
# reference assembly (hand-written, no assembler reuse)
# ---------------------------------------------------------------------------
def hand_block_matrix(
    mass: float, center_of_gravity: Sequence[float], inertia_cg: ArrayLike
) -> FloatArray:
    """Assemble ``M_RB`` by hand from the closed block form.

    Written out explicitly -- ``S(r)`` built entry by entry, the two off-diagonal
    blocks as ``-m S`` and ``+m S``, the lower-right block as
    ``I_g + m (|r|^2 I - r r^T)`` -- so that comparing it with
    :func:`tensorbem.rigid_body_mass.rigid_body_mass_matrix` checks the
    assembler instead of re-using it.
    """
    m = float(mass)
    x, y, z = (float(value) for value in np.asarray(center_of_gravity, dtype=float).reshape(3))
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    inertia = np.asarray(inertia_cg, dtype=float)
    if inertia.shape != (3, 3):
        raise ValueError("inertia_cg must be (3, 3)")
    r = np.array([x, y, z])
    lower_right = inertia + m * (float(r @ r) * np.eye(3) - np.outer(r, r))
    matrix = np.zeros((6, 6), dtype=float)
    matrix[:3, :3] = m * np.eye(3)
    matrix[:3, 3:] = -m * skew
    matrix[3:, :3] = m * skew
    matrix[3:, 3:] = lower_right
    return matrix


ArrayLike = Sequence[Sequence[float]] | NDArray[np.floating]


# ---------------------------------------------------------------------------
# study driver
# ---------------------------------------------------------------------------
def run(
    ladder: Sequence[tuple[int, int]] = MESH_LADDER,
    hull: WigleyHull = HULL,
    quadrature_order: int = 24,
) -> dict[str, Any]:
    """Run the Wigley-hull rigid-body mass-matrix case.

    Returns a JSON-serialisable dict with the three routes, the geometry gate,
    the per-entry error ladder, the energy identity, the negative control and a
    single ``verdict`` block (``all_pass`` plus per-check booleans).
    """
    closed = closed_form_mass_properties(hull)
    quadrature = quadrature_mass_properties(hull, quadrature_order)

    ref_matrix = hand_block_matrix(
        closed["mass"], hull.cg, closed["inertia_at_design_cg"]
    )
    assembled = rigid_body_mass_matrix(
        closed["mass"], hull.cg, closed["inertia_at_design_cg"], inertia_about="cg"
    )
    assembler_residual = float(
        np.max(np.abs(assembled - ref_matrix)) / np.max(np.abs(ref_matrix))
    )
    energy_ref = kinetic_energy_error(
        ref_matrix, closed["mass"], hull.cg, closed["inertia_at_design_cg"]
    )

    closed_vs_quadrature = {
        "volume_rel": abs(quadrature["volume"] - closed["volume"]) / closed["volume"],
        "centroid_abs_rel": float(
            np.max(np.abs(quadrature["centroid"] - closed["centroid"]))
            / max(hull.length, 1.0)
        ),
        "inertia_rel": float(
            np.max(
                np.abs(
                    quadrature["inertia_at_design_cg"] - closed["inertia_at_design_cg"]
                )
            )
            / np.max(np.abs(closed["inertia_at_design_cg"]))
        ),
    }

    levels: list[dict[str, Any]] = []
    for n_sections, n_levels in ladder:
        route = mesh_route(hull, n_sections, n_levels)
        errors = relative_error_matrix(ref_matrix, route["matrix"])
        levels.append(
            {
                "n_sections": int(n_sections),
                "n_levels": int(n_levels),
                "panels": int(route["mesh"]["wetted_panels"] + route["mesh"]["lid_panels"]),
                "wetted_panels": route["mesh"]["wetted_panels"],
                "volume": route["volume"],
                "volume_rel_error": abs(route["volume"] - closed["volume"]) / closed["volume"],
                "block_coefficient": route["block_coefficient"],
                "block_coefficient_rel_error": abs(
                    route["block_coefficient"] - closed["block_coefficient"]
                )
                / closed["block_coefficient"],
                "length_rel_error": abs(route["bbox"]["length"] - hull.length) / hull.length,
                "beam_rel_error": abs(route["bbox"]["beam"] - hull.beam) / hull.beam,
                "centroid_abs_error_m": float(
                    np.max(np.abs(route["centroid"] - closed["centroid"]))
                ),
                "inertia_rel_error": float(
                    np.max(
                        np.abs(
                            route["inertia_at_design_cg"] - closed["inertia_at_design_cg"]
                        )
                    )
                    / np.max(np.abs(closed["inertia_at_design_cg"]))
                ),
                "max_entry_rel_error": float(errors.max()),
                "worst_entry": {
                    "mode_i": MODES[int(np.unravel_index(np.argmax(errors), errors.shape)[0])],
                    "mode_j": MODES[int(np.unravel_index(np.argmax(errors), errors.shape)[1])],
                },
                "within_tolerance": bool(errors.max() < TOLERANCE),
                "energy_identity_error": route["energy_identity_error"],
            }
        )

    accepted = [level for level in levels if level["within_tolerance"]]
    accepted_level = accepted[0] if accepted else None
    pass_level = levels[-1]

    # Negative control: the pre-m17324 transposed coupling block.
    legacy = legacy_coupling_matrix(
        closed["mass"], hull.cg, closed["inertia_at_design_cg"]
    )
    legacy_errors = relative_error_matrix(ref_matrix, legacy)
    negative_control = {
        "variant": "pre-m17324 transposed coupling block (-m S^T in the upper-right)",
        "max_entry_rel_error": float(legacy_errors.max()),
        "energy_identity_error": kinetic_energy_error(
            legacy, closed["mass"], hull.cg, closed["inertia_at_design_cg"]
        ),
        "detected": bool(legacy_errors.max() > TOLERANCE),
    }

    # Discretisation error measured on a property the 3 % gate can hold.
    checks = {
        "closed_form_matches_quadrature": bool(
            closed_vs_quadrature["volume_rel"] < 1.0e-10
            and closed_vs_quadrature["inertia_rel"] < 1.0e-10
        ),
        "assembler_matches_hand_block": bool(assembler_residual < 1.0e-12),
        "reference_energy_identity_roundoff": bool(energy_ref < 1.0e-12),
        "geometry_gate_at_accepted_level": bool(
            accepted_level is not None
            and accepted_level["block_coefficient_rel_error"] < TOLERANCE
            and accepted_level["length_rel_error"] < TOLERANCE
            and accepted_level["beam_rel_error"] < TOLERANCE
        ),
        "mesh_route_within_tolerance": bool(accepted_level is not None),
        "mesh_energy_identity_roundoff": bool(pass_level["energy_identity_error"] < 1.0e-12),
        "negative_control_detected": bool(negative_control["detected"]),
        "error_decreases_under_refinement": bool(
            all(
                levels[i + 1]["max_entry_rel_error"] <= levels[i]["max_entry_rel_error"] + 1.0e-6
                for i in range(len(levels) - 1)
            )
        ),
    }

    return {
        "benchmark": "CASE-BEM-RIGID-BODY-MASS.wigley-ship-form",
        "subject": "6x6 rigid-body mass matrix of a parabolic Wigley hull (analytically known Cb, L, B)",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "parameters": {
            "hull": hull.to_dict(),
            "tolerance": TOLERANCE,
            "mesh_ladder": [[int(a), int(b)] for a, b in ladder],
            "quadrature_order": int(quadrature_order),
            "accepted_level": None if accepted_level is None else {
                "n_sections": accepted_level["n_sections"],
                "n_levels": accepted_level["n_levels"],
                "panels": accepted_level["panels"],
            },
        },
        "reference": {
            **REFERENCE,
            "volume": closed["volume"],
            "block_coefficient": closed["block_coefficient"],
            "centroid": closed["centroid"].tolist(),
            "mass": closed["mass"],
            "inertia_about_centroid": closed["inertia_cg"].tolist(),
            "inertia_about_design_cg": closed["inertia_at_design_cg"].tolist(),
            "second_moments": closed["second_moments"],
            "inertia_terms": closed["inertia_terms"],
            "radii_of_gyration": closed["radii_of_gyration"],
            "matrix": ref_matrix.tolist(),
        },
        "routes": {
            "R_closed_form": {
                "volume": closed["volume"],
                "centroid": closed["centroid"].tolist(),
                "inertia_about_design_cg": closed["inertia_at_design_cg"].tolist(),
            },
            "Q_quadrature": {
                "volume": quadrature["volume"],
                "centroid": quadrature["centroid"].tolist(),
                "inertia_about_design_cg": quadrature["inertia_at_design_cg"].tolist(),
                "order": quadrature["quadrature_order"],
                "rel_difference": closed_vs_quadrature,
            },
            "A_assembler": {
                "entry": "tensorbem.rigid_body_mass.rigid_body_mass_matrix",
                "rel_residual_vs_hand_block": assembler_residual,
                "energy_identity_error": energy_ref,
            },
        },
        "levels": levels,
        "negative_control": negative_control,
        "verdict": {"all_pass": all(checks.values()), "checks": checks},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/cases/case_bem_rigid_body_mass_wigley/run.json"),
        help="where to write the JSON result",
    )
    parser.add_argument("--length", type=float, default=HULL.length)
    parser.add_argument("--beam", type=float, default=HULL.beam)
    parser.add_argument("--draft", type=float, default=HULL.draft)
    parser.add_argument("--density", type=float, default=HULL.density)
    args = parser.parse_args(argv)

    hull = WigleyHull(args.length, args.beam, args.draft, args.density)
    report = run(hull=hull)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    verdict = report["verdict"]
    print(json.dumps({"out": str(args.out), "all_pass": verdict["all_pass"],
                      "checks": verdict["checks"]}, indent=2))
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())