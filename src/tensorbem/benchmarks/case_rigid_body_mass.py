"""CASE-BEM-RIGID-BODY-MASS: the 6x6 rigid-body mass matrix against its closed form.

Case pool anchor ``CASE-BEM-RIGID-BODY-MASS``.  The quantity under test is the
rigid-body (inertia) part of the 6-DOF equations of motion that every
frequency-domain seakeeping solve in TensorBEM assembles next to the radiation
matrices:

    ( -w^2 (M + A(w)) + i w B(w) + C ) xi = F_exc ,
    M = M_RB .

Reference (``ref``)
-------------------
Closed-form block form of the rigid-body mass matrix,

    M_RB = [[ m I_3 , -m S(r_g) ],
            [ m S(r_g) ,  I_g + m S(r_g) S(r_g)^T ]] ,

with ``S(r)`` the skew-symmetric matrix of ``r``.  This is the form given by
Fossen (2011), *Handbook of Marine Craft Hydrodynamics and Motion Control*,
Wiley, section 3.3.1 "Rigid-body kinetics", eqs. (3.44)-(3.53) (the block
partition of ``M_RB`` and the parallel-axis shift ``I_b = I_g + m S^T(r_g)
S(r_g)``); the same block structure appears in Newman (1977) *Marine
Hydrodynamics* section 4.5 and in Journee & Massie (2001) *Offshore
Hydromechanics* section 3.2.  Every number quoted by this case is *computed*
from those closed forms by the code below (zero hand-copy); the case prints the
book section it evaluates.

Because the closed form is analytic, the numerical target for a specified body
is obtained from two independent derivations of its mass properties:

* route ``R`` (analytic): solid-box second moments (``m(L^2+B^2)/12`` etc.) plus
  the parallel-axis theorem, composed over the body's components;
* route ``M`` (mesh): exact surface integration of a triangulated closed surface
  by the divergence theorem,
  ``V = (1/3) int_S r.n dS``, ``int_V r dV = (1/4) int_S r (r.n) dS``,
  ``int_V x_i x_j dV = (1/5) int_S x_i x_j (r.n) dS``, with a degree-3 triangle
  rule (exact for these integrands).

They agree to round-off for a polyhedral body, which is the geometry check the
ecosystem requires (Cb/L/B style validation) before any tolerance judgement.

Unit under test (``got``)
-------------------------
``tensorbem.hydrostatic.compute_mass_matrix(m, r_g, I_g)`` -- the function the
seakeeping solver (``tensorbem.rao``), the ice RAO path
(``tensorbem.ice_rao``) and the verification harnesses call.

Acceptance
----------
Entry-wise ``|got - ref| / |ref| < 3%`` (case-pool criterion) over all 36
entries, plus the kinetic-energy identity
``0.5 x^T M x == 0.5 m |v_o + w x r_g|^2 + 0.5 w^T I_g w`` for random
``x = (v_o, w)``.  A negative control replays the pre-fix coupling block
(``-m S^T`` in the upper-right) and must fail the criterion, otherwise the
benchmark has no discriminating power.

Run::

    python -m tensorbem.benchmarks.case_rigid_body_mass --out runs/m_case_rigid_body_mass

"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from tensorbem.hydrostatic import compute_mass_matrix

__all__ = [
    "BoxComponent",
    "TOLERANCE",
    "case_components",
    "closed_form_mass_matrix",
    "compose_analytic",
    "kinetic_energy_error",
    "main",
    "mesh_mass_properties",
    "relative_error_matrix",
    "run",
]

TOLERANCE = 0.03  # case pool: < 3 %
MODES = ("surge", "sway", "heave", "roll", "pitch", "yaw")
WATER_DENSITY = 1025.0
GRAVITY = 9.80665

#: Where the closed form is taken from (printed into every artifact).
REFERENCE = {
    "form": "M_RB = [[m I3, -m S(r_g)], [m S(r_g), I_g + m S(r_g) S(r_g)^T]]",
    "source": (
        "Fossen (2011) Handbook of Marine Craft Hydrodynamics and Motion "
        "Control, Wiley, section 3.3.1 (rigid-body kinetics), eqs. (3.44)-(3.53); "
        "same block partition in Newman (1977) Marine Hydrodynamics section 4.5 "
        "and Journee & Massie (2001) Offshore Hydromechanics section 3.2"
    ),
    "numbers": (
        "evaluated by this module from the cited closed forms and the case "
        "geometry below (zero hand-copy)"
    ),
}


# ---------------------------------------------------------------------------
# geometry / mass model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BoxComponent:
    """An axis-aligned solid box of uniform density (a mass-model component)."""

    name: str
    length: float
    beam: float
    height: float
    center: tuple[float, float, float]
    density: float

    @property
    def volume(self) -> float:
        return self.length * self.beam * self.height

    @property
    def mass(self) -> float:
        return self.density * self.volume

    def inertia_about_centroid(self) -> np.ndarray:
        """Solid-box second moments about its own centroid (diagonal)."""
        m, a, b, c = self.mass, self.length, self.beam, self.height
        return np.diag(
            [
                m * (b * b + c * c) / 12.0,
                m * (a * a + c * c) / 12.0,
                m * (a * a + b * b) / 12.0,
            ]
        )


def case_components(payload_x: float = 30.0) -> tuple[BoxComponent, BoxComponent]:
    """Barge + deck payload: a body with non-zero x_g, y_g, z_g and I_xy.

    The hull is a 120 m x 20 m x 10 m box (bottom ``z = -8`` m, deck ``z = +2``
    m) of relative density 0.55, so it floats at a draft well inside the depth
    once the payload is aboard.  The payload is a 20 m x 10 m x 4 m machinery
    box standing on the deck, off the centreline, with 8 % of the hull mass.
    ``payload_x`` slides it fore/aft: that is the parameter ladder used for the
    sensitivity study, and it moves ``x_g`` without touching anything else.
    """
    hull = BoxComponent(
        name="hull",
        length=120.0,
        beam=20.0,
        height=10.0,
        center=(0.0, 0.0, -3.0),
        density=0.55 * WATER_DENSITY,
    )
    payload_density = 0.08 * hull.mass / (20.0 * 10.0 * 4.0)
    payload = BoxComponent(
        name="payload",
        length=20.0,
        beam=10.0,
        height=4.0,
        center=(float(payload_x), 4.0, 4.0),
        density=payload_density,
    )
    return hull, payload


def skew(r: Sequence[float]) -> np.ndarray:
    """Skew-symmetric matrix ``S(r)`` with ``S(r) x = r x x``."""
    x, y, z = (float(v) for v in r)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


# ---------------------------------------------------------------------------
# route R: analytic composite (closed form)
# ---------------------------------------------------------------------------
def compose_analytic(components: Iterable[BoxComponent]) -> dict[str, Any]:
    """Mass, centre of gravity and inertia tensors of a composite body."""
    comps = list(components)
    volumes = np.array([c.volume for c in comps])
    masses = np.array([c.mass for c in comps])
    centres = np.array([c.center for c in comps], dtype=float)

    total_volume = float(volumes.sum())
    total_mass = float(masses.sum())
    r_g = (masses[:, None] * centres).sum(axis=0) / total_mass

    inertia_cg = np.zeros((3, 3))
    for comp, m in zip(comps, masses):
        d = np.asarray(comp.center, dtype=float) - r_g
        inertia_cg += comp.inertia_about_centroid() + m * (
            float(d @ d) * np.eye(3) - np.outer(d, d)
        )

    s = skew(r_g)
    inertia_origin = inertia_cg + total_mass * (s @ s.T)
    return {
        "mass": total_mass,
        "volume": total_volume,
        "center_of_gravity": r_g.tolist(),
        "inertia_about_cg": inertia_cg.tolist(),
        "inertia_about_origin": inertia_origin.tolist(),
        "components": [
            {
                "name": c.name,
                "length": c.length,
                "beam": c.beam,
                "height": c.height,
                "center": list(c.center),
                "density": c.density,
                "volume": c.volume,
                "mass": c.mass,
            }
            for c in comps
        ],
    }


def closed_form_mass_matrix(
    mass: float, center_of_gravity: Sequence[float], inertia_cg: np.ndarray
) -> np.ndarray:
    """The reference block form (Fossen 2011 eq. 3.51), written out by hand."""
    m = float(mass)
    s = skew(center_of_gravity)
    # upper-right -m*S, lower-left -m*S^T (= +m*S), as in Fossen eq. 3.51.
    return np.block(
        [
            [m * np.eye(3), -m * s],
            [-m * s.T, np.asarray(inertia_cg, dtype=float) + m * (s @ s.T)],
        ]
    )


# ---------------------------------------------------------------------------
# route M: mesh integration by the divergence theorem
# ---------------------------------------------------------------------------
def _box_quads(component: BoxComponent, subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed triangulated surface of a box, ``subdivisions`` per edge."""
    half = np.array([component.length, component.beam, component.height]) / 2.0
    centre = np.asarray(component.center, dtype=float)
    n = max(int(subdivisions), 1)

    nodes: list[np.ndarray] = []
    index: dict[tuple[float, float, float], int] = {}
    tris: list[tuple[int, int, int]] = []

    def node(p: np.ndarray) -> int:
        key = (float(p[0]), float(p[1]), float(p[2]))
        if key not in index:
            index[key] = len(nodes)
            nodes.append(p)
        return index[key]

    for axis in range(3):
        for sign in (-1.0, 1.0):
            u_axis, v_axis = [a for a in range(3) if a != axis]
            u = np.linspace(-half[u_axis], half[u_axis], n + 1)
            v = np.linspace(-half[v_axis], half[v_axis], n + 1)
            for i in range(n):
                for j in range(n):
                    corners = []
                    for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        p = np.zeros(3)
                        p[axis] = sign * half[axis]
                        p[u_axis] = u[i + du]
                        p[v_axis] = v[j + dv]
                        corners.append(node(centre + p))
                    a, b, c, d = corners
                    tris.append((a, b, c))
                    tris.append((a, c, d))
    return np.asarray(nodes), np.asarray(tris, dtype=int)


def _orient_outward(nodes: np.ndarray, tris: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Flip triangles whose normal points toward the box centre."""
    a = nodes[tris[:, 0]]
    b = nodes[tris[:, 1]]
    c = nodes[tris[:, 2]]
    normal = np.cross(b - a, c - a)
    outward = np.einsum("ij,ij->i", normal, a - centre)
    flipped = tris.copy()
    flipped[outward < 0.0] = tris[outward < 0.0][:, ::-1]
    return flipped


# degree-3 triangle rule (exact for polynomials up to degree 3):
# centroid weight -9/16, three points at (3/5, 1/5, 1/5) weight 25/48 each.
_Q3_WEIGHTS = np.array([-9.0 / 16.0, 25.0 / 48.0, 25.0 / 48.0, 25.0 / 48.0])
_Q3_BARY = np.array(
    [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [3.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0],
        [1.0 / 5.0, 3.0 / 5.0, 1.0 / 5.0],
        [1.0 / 5.0, 1.0 / 5.0, 3.0 / 5.0],
    ]
)


def mesh_mass_properties(
    components: Sequence[BoxComponent], subdivisions: int = 2
) -> dict[str, Any]:
    """Volume, centre of gravity and inertia of a closed triangulated surface.

    Exact (to round-off) for polyhedral bodies: the divergence-theorem
    identities are integrated with a rule exact for the degree of the
    integrands, and every panel is planar and lies on the analytic surface.
    Components may have different densities; each is integrated separately and
    combined with the parallel-axis theorem.
    """
    total_volume = 0.0
    total_mass = 0.0
    mass_moment = np.zeros(3)
    inertia_origin = np.zeros((3, 3))
    panels = 0
    for component in components:
        props = _single_box_properties(component, subdivisions)
        total_volume += props["volume"]
        total_mass += props["mass"]
        mass_moment += props["mass"] * np.asarray(props["center_of_gravity"])
        inertia_origin += np.asarray(props["inertia_about_origin"])
        panels += props["panels"]

    r_g = mass_moment / total_mass
    s = skew(r_g)
    inertia_cg = inertia_origin - total_mass * (s @ s.T)
    return {
        "volume": total_volume,
        "mass": total_mass,
        "center_of_gravity": r_g.tolist(),
        "inertia_about_cg": inertia_cg.tolist(),
        "inertia_about_origin": inertia_origin.tolist(),
        "panels": panels,
    }


def _single_box_properties(component: BoxComponent, subdivisions: int) -> dict[str, Any]:
    nodes, tris = _box_quads(component, subdivisions)
    tris = _orient_outward(nodes, tris, np.asarray(component.center, dtype=float))
    a, b, c = (nodes[tris[:, k]] for k in range(3))
    area_vec = 0.5 * np.cross(b - a, c - a)

    points = np.einsum("qb,tbj->qtj", _Q3_BARY, np.stack([a, b, c], axis=1))
    r_dot_n = np.einsum("qtj,tj->qt", points, area_vec)
    w = _Q3_WEIGHTS[:, None]

    volume = float((w * r_dot_n).sum() / 3.0)
    first = (w[..., None] * points * r_dot_n[:, :, None]).sum(axis=(0, 1)) / 4.0
    second = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            second[i, j] = (w * points[:, :, i] * points[:, :, j] * r_dot_n).sum() / 5.0

    rho = component.density
    mass = rho * volume
    centre = first / volume
    inertia_origin = rho * (float(np.trace(second)) * np.eye(3) - second)
    s = skew(centre)
    inertia_cg = inertia_origin - mass * (s @ s.T)
    return {
        "volume": volume,
        "mass": mass,
        "center_of_gravity": centre.tolist(),
        "inertia_about_cg": inertia_cg.tolist(),
        "inertia_about_origin": inertia_origin.tolist(),
        "panels": int(tris.shape[0]),
    }


# ---------------------------------------------------------------------------
# second configuration: the platform referee row (tensor_benchmark_cases id=57)
# ---------------------------------------------------------------------------
#: Geometry of the referee row registered for this anchor by audit m17342:
#: solid ellipsoid a, b, c = 60, 12, 8 m, rho = 1025 kg/m^3, r_g = (5, 0, -3) m,
#: ladder 20 / 80 / 320 / 1280 / 5120 panels (= 20*4**subdivisions).
ELLIPSOID_SEMI_AXES = (60.0, 12.0, 8.0)
ELLIPSOID_DENSITY = 1025.0
ELLIPSOID_CENTRE = (5.0, 0.0, -3.0)


def ellipsoid_reference(
    semi_axes: Sequence[float] = ELLIPSOID_SEMI_AXES,
    density: float = ELLIPSOID_DENSITY,
    centre: Sequence[float] = ELLIPSOID_CENTRE,
) -> dict[str, Any]:
    """Closed-form mass properties of a uniform solid ellipsoid (route R')."""
    a, b, c = (float(v) for v in semi_axes)
    volume = 4.0 / 3.0 * math.pi * a * b * c
    mass = density * volume
    inertia_cg = np.diag(
        [mass * (b * b + c * c) / 5.0, mass * (a * a + c * c) / 5.0, mass * (a * a + b * b) / 5.0]
    )
    return {
        "volume": volume,
        "mass": mass,
        "center_of_gravity": [float(v) for v in centre],
        "inertia_about_cg": inertia_cg.tolist(),
    }


def tetra_mass_properties(
    vertices: np.ndarray, faces: np.ndarray, density: float
) -> dict[str, Any]:
    """Mass properties of a closed triangle mesh by signed-tetrahedron quadrature.

    Exact for a polyhedron: each face is joined to the origin, and for a
    tetrahedron with vertices ``0, a, b, c`` (volume ``V``) the exact integrals
    are ``int x_j x_k dV = V/20 (sum_p v_pj v_pk + (sum_p v_pj)(sum_p v_pk))``.
    """
    verts = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)
    a, b, c = (verts[tris[:, k]] for k in range(3))
    vol6 = np.einsum("ij,ij->i", a, np.cross(b, c))
    tet_volume = vol6 / 6.0
    volume = float(tet_volume.sum())

    corners = np.stack([np.zeros_like(a), a, b, c], axis=1)  # (n_tri, 4, 3)
    corner_sum = corners.sum(axis=1)  # (n_tri, 3)
    first = (tet_volume[:, None] * corner_sum / 4.0).sum(axis=0)

    second = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            pair = (corners[:, :, i] * corners[:, :, j]).sum(axis=1)
            second[i, j] = (
                tet_volume * (pair + corner_sum[:, i] * corner_sum[:, j]) / 20.0
            ).sum()

    mass = density * volume
    centre = first / volume
    inertia_origin = density * (float(np.trace(second)) * np.eye(3) - second)
    s = skew(centre)
    inertia_cg = inertia_origin - mass * (s @ s.T)
    return {
        "volume": volume,
        "mass": mass,
        "center_of_gravity": centre.tolist(),
        "inertia_about_cg": inertia_cg.tolist(),
        "inertia_about_origin": inertia_origin.tolist(),
        "panels": int(tris.shape[0]),
    }


def ellipsoid_ladder(subdivisions: Sequence[int] = (0, 1, 2, 3, 4)) -> list[dict[str, Any]]:
    """Referee-row configuration: mesh quadrature mass properties vs closed form."""
    from tensorbem.geometry3d import ellipsoid_mesh

    ref = ellipsoid_reference()
    ref_matrix = closed_form_mass_matrix(
        ref["mass"], ref["center_of_gravity"], np.asarray(ref["inertia_about_cg"])
    )
    rows = []
    for sub in subdivisions:
        base = ellipsoid_mesh(int(sub), np.asarray(ELLIPSOID_SEMI_AXES))
        vertices = np.asarray(base.vertices, dtype=float) + np.asarray(ELLIPSOID_CENTRE)
        props = tetra_mass_properties(vertices, base.faces, ELLIPSOID_DENSITY)
        got_matrix = compute_mass_matrix(
            props["mass"],
            np.asarray(props["center_of_gravity"]),
            np.asarray(props["inertia_about_cg"]),
        )
        control_matrix = legacy_coupling_matrix(
            props["mass"],
            np.asarray(props["center_of_gravity"]),
            np.asarray(props["inertia_about_cg"]),
        )
        error = relative_error_matrix(ref_matrix, got_matrix)
        control = relative_error_matrix(ref_matrix, control_matrix)
        rows.append(
            {
                "subdivisions": int(sub),
                "panels": props["panels"],
                "volume_rel_error": abs(props["volume"] - ref["volume"]) / ref["volume"],
                "mass_rel_error": abs(props["mass"] - ref["mass"]) / ref["mass"],
                "cg_abs_error": float(
                    np.max(
                        np.abs(np.asarray(props["center_of_gravity"]) - np.asarray(ref["center_of_gravity"]))
                    )
                ),
                "inertia_cg_rel_error": float(
                    np.max(
                        np.abs(
                            np.asarray(props["inertia_about_cg"])
                            - np.asarray(ref["inertia_about_cg"])
                        )
                        / np.max(np.abs(ref["inertia_about_cg"]))
                    )
                ),
                "max_rel_error": float(error.max()),
                "control_max_rel_error": float(control.max()),
                "passed": bool(error.max() < TOLERANCE),
                "control_passed": bool(control.max() < TOLERANCE),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# comparison helpers
# ---------------------------------------------------------------------------
def relative_error_matrix(ref: np.ndarray, got: np.ndarray) -> np.ndarray:
    """Entry-wise ``|got-ref| / max(|ref|, scale)`` with ``scale`` = max |ref|.

    Entries that are zero in the reference (surge-sway of a body with no
    z-coupling, for instance) are judged on the absolute error divided by the
    largest reference entry, so the ratio stays finite and meaningful.
    """
    ref = np.asarray(ref, dtype=float)
    got = np.asarray(got, dtype=float)
    scale = float(np.max(np.abs(ref)))
    denominator = np.maximum(np.abs(ref), 1e-6 * scale)
    return np.abs(got - ref) / denominator


def kinetic_energy_error(
    matrix: np.ndarray,
    mass: float,
    center_of_gravity: Sequence[float],
    inertia_cg: np.ndarray,
    samples: int = 256,
    seed: int = 20261113,
) -> float:
    """Max relative error of ``0.5 x^T M x`` against the rigid-body kinetic energy."""
    rng = np.random.default_rng(seed)
    r_g = np.asarray(center_of_gravity, dtype=float)
    inertia_cg = np.asarray(inertia_cg, dtype=float)
    worst = 0.0
    for _ in range(samples):
        v = rng.normal(size=3)
        w = rng.normal(size=3)
        v_g = v + np.cross(w, r_g)
        energy = 0.5 * mass * float(v_g @ v_g) + 0.5 * float(w @ inertia_cg @ w)
        quadratic = 0.5 * float(np.concatenate([v, w]) @ matrix @ np.concatenate([v, w]))
        worst = max(worst, abs(quadratic - energy) / max(abs(energy), 1e-12))
    return worst


def legacy_coupling_matrix(
    mass: float, center_of_gravity: Sequence[float], inertia_cg: np.ndarray
) -> np.ndarray:
    """The pre-fix convention (``-m S^T`` in the upper-right block) as a control."""
    m = float(mass)
    s = skew(center_of_gravity)
    return np.block(
        [
            [m * np.eye(3), -m * s.T],
            [-m * s, np.asarray(inertia_cg, dtype=float) + m * (s @ s.T)],
        ]
    )  # noqa: E501 - kept verbatim as the pre-fix convention


# ---------------------------------------------------------------------------
# the study
# ---------------------------------------------------------------------------
def _evaluate(
    components: Sequence[BoxComponent], subdivisions: int
) -> dict[str, Any]:
    analytic = compose_analytic(components)
    mesh = mesh_mass_properties(components, subdivisions)

    r_g = np.asarray(analytic["center_of_gravity"])
    inertia_cg = np.asarray(analytic["inertia_about_cg"])
    mass = float(analytic["mass"])

    ref = closed_form_mass_matrix(mass, r_g, inertia_cg)
    got = compute_mass_matrix(mass, r_g, inertia_cg)
    got_from_mesh = compute_mass_matrix(
        float(mesh["mass"]),
        np.asarray(mesh["center_of_gravity"]),
        np.asarray(mesh["inertia_about_cg"]),
    )
    control = legacy_coupling_matrix(mass, r_g, inertia_cg)

    # geometry / mass-property agreement between the two independent routes
    geometry = {
        "mass_rel_error": abs(mesh["mass"] - mass) / mass,
        "volume_rel_error": abs(mesh["volume"] - analytic["volume"]) / analytic["volume"],
        "cg_abs_error": float(
            np.max(np.abs(np.asarray(mesh["center_of_gravity"]) - r_g))
        ),
        "inertia_cg_rel_error": float(
            np.max(
                np.abs(np.asarray(mesh["inertia_about_cg"]) - inertia_cg)
                / np.max(np.abs(inertia_cg))
            )
        ),
    }

    error = relative_error_matrix(ref, got)
    error_mesh = relative_error_matrix(ref, got_from_mesh)
    error_control = relative_error_matrix(ref, control)

    return {
        "subdivisions": int(subdivisions),
        "panels": mesh["panels"],
        "analytic": analytic,
        "mesh": mesh,
        "ref_matrix": ref.tolist(),
        "got_matrix": got.tolist(),
        "control_matrix": control.tolist(),
        "rel_error_matrix": error.tolist(),
        "rel_error_from_mesh_properties": error_mesh.tolist(),
        "control_rel_error_matrix": error_control.tolist(),
        "max_rel_error": float(error.max()),
        "max_rel_error_loc": [int(i) for i in np.unravel_index(error.argmax(), error.shape)],
        "max_rel_error_from_mesh_properties": float(error_mesh.max()),
        "control_max_rel_error": float(error_control.max()),
        "geometry_agreement": geometry,
        "kinetic_energy_error": kinetic_energy_error(got, mass, r_g, inertia_cg),
        "control_kinetic_energy_error": kinetic_energy_error(control, mass, r_g, inertia_cg),
        "coupling_block_norm": float(np.linalg.norm(ref[:3, 3:])),
        "mass": mass,
        "center_of_gravity": r_g.tolist(),
        "inertia_about_cg": inertia_cg.tolist(),
        "passed": bool(error.max() < TOLERANCE),
        "control_passed": bool(error_control.max() < TOLERANCE),
    }


def _payload_ladder() -> list[dict[str, Any]]:
    """Slide the payload fore/aft: coupling magnitude and error vs x_g."""
    rows = []
    for x in (0.0, 5.0, 15.0, 30.0, 45.0, 55.0):
        result = _evaluate(case_components(payload_x=x), subdivisions=2)
        rows.append(
            {
                "payload_x": x,
                "x_g": result["center_of_gravity"][0],
                "coupling_block_norm": result["coupling_block_norm"],
                "max_rel_error": result["max_rel_error"],
                "control_max_rel_error": result["control_max_rel_error"],
                "kinetic_energy_error": result["kinetic_energy_error"],
                "passed": result["passed"],
                "control_passed": result["control_passed"],
            }
        )
    return rows


def _rao_integration_check() -> dict[str, Any]:
    """Prove that the seakeeping solver really consumes this matrix.

    ``tensorbem.rao.compute_rao_at_frequency`` assembles
    ``Z = -w^2 (M + A) + i w B + C`` from ``compute_mass_matrix`` and then
    solves for the RAO.  Re-running the *same* solver with the pre-fix coupling
    block injected at that single call site isolates the matrix's effect on the
    delivered RAO numbers (this is an integration test: nothing else changes).
    """
    from tensorbem import rao as rao_module
    from tensorbem.geometry3d import box_hull_mesh

    components = case_components()
    hull = components[0]
    analytic = compose_analytic(components)
    # The RAO solver needs a body in Archimedes equilibrium on its wetted mesh,
    # so the same barge is cut at the draft implied by the full mass model
    # (mass = rho * L * B * T_eq), carrying the composite mass properties of
    # route R (hull + deck payload).
    mass = float(analytic["mass"])
    r_g = np.asarray(analytic["center_of_gravity"], dtype=float)
    inertia_cg = np.asarray(analytic["inertia_about_cg"], dtype=float)
    draft = mass / (WATER_DENSITY * hull.length * hull.beam)
    mesh = box_hull_mesh(length=hull.length, beam=hull.beam, draft=draft,
                         n_length=4, n_beam=3, n_draft=3)
    omega = 0.6

    reference = rao_module.compute_rao_at_frequency(
        mesh, omega, 0.0, mass, r_g, inertia=inertia_cg, chief_points="auto"
    )

    original = rao_module.compute_mass_matrix
    legacy = legacy_coupling_matrix(mass, r_g, inertia_cg)
    rao_module.compute_mass_matrix = lambda *a, **k: legacy  # type: ignore[assignment]
    try:
        control = rao_module.compute_rao_at_frequency(
            mesh, omega, 0.0, mass, r_g, inertia=inertia_cg, chief_points="auto"
        )
    finally:
        rao_module.compute_mass_matrix = original  # type: ignore[assignment]

    amp = np.abs(reference.rao)
    amp_control = np.abs(control.rao)
    scale = np.maximum(amp, 1e-30)
    change = np.abs(amp_control - amp) / scale

    # the same call, but with the reference block form injected, must reproduce
    # the shipped result to round-off (proves the caller uses this matrix and
    # nothing else from the mass model)
    ref_matrix = closed_form_mass_matrix(mass, r_g, inertia_cg)
    rao_module.compute_mass_matrix = lambda *a, **k: ref_matrix  # type: ignore[assignment]
    try:
        injected = rao_module.compute_rao_at_frequency(
            mesh, omega, 0.0, mass, r_g, inertia=inertia_cg, chief_points="auto"
        )
    finally:
        rao_module.compute_mass_matrix = original  # type: ignore[assignment]
    replay = float(np.max(np.abs(np.abs(injected.rao) - amp) / scale))

    return {
        "caller": "tensorbem.rao.compute_rao_at_frequency (impedance -w^2(M+A)+iwB+C)",
        "caller_line": "src/tensorbem/rao.py: m = compute_mass_matrix(mass, center_of_gravity, inertia)",
        "omega": omega,
        "panels": int(mesh.faces.shape[0]),
        "rao_shipped": amp.tolist(),
        "rao_with_legacy_coupling": amp_control.tolist(),
        "rao_change_fraction": change.tolist(),
        "max_rao_change": float(change.max()),
        "max_rao_change_mode": MODES[int(change.argmax())],
        "replay_with_injected_ref_matrix": replay,
        "modes": list(MODES),
    }


def run(
    subdivisions: Sequence[int] = (1, 2, 4),
    *,
    with_rao: bool = True,
) -> dict[str, Any]:
    """Execute the case: geometry validation, ref-vs-got table, ladders."""
    levels = [_evaluate(case_components(), int(n)) for n in subdivisions]
    primary = levels[-1]
    result: dict[str, Any] = {
        "anchor": "CASE-BEM-RIGID-BODY-MASS",
        "case": "rigid-body 6x6 mass matrix of a barge + deck payload",
        "reference": REFERENCE,
        "modes": list(MODES),
        "tolerance": TOLERANCE,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "geometry": {
            "water_density": WATER_DENSITY,
            "gravity": GRAVITY,
            "components": primary["analytic"]["components"],
            "mass": primary["mass"],
            "center_of_gravity": primary["center_of_gravity"],
            "displaced_volume": primary["mass"] / WATER_DENSITY,
            "waterplane_area": 120.0 * 20.0,
            "equilibrium_draft": primary["mass"] / (WATER_DENSITY * 120.0 * 20.0),
            "hull_depth": 10.0,
            "hull_bottom_z": -8.0,
            "comment": (
                "equilibrium draft is a geometry consistency check (it must stay "
                "inside the 10 m hull depth); the rigid-body mass matrix does not "
                "depend on the waterline"
            ),
        },
        "levels": levels,
        "primary": primary,
        "payload_ladder": _payload_ladder(),
        "ellipsoid": {
            "semi_axes": list(ELLIPSOID_SEMI_AXES),
            "density": ELLIPSOID_DENSITY,
            "center_of_gravity": list(ELLIPSOID_CENTRE),
            "reference": ellipsoid_reference(),
            "ladder": ellipsoid_ladder(),
            "comment": (
                "configuration of the platform referee row tensor_benchmark_cases "
                "id=57 recorded for this anchor; the mesh route here is an "
                "independent signed-tetrahedron quadrature, so the ladder "
                "reproduces the row's published numbers"
            ),
        },
        "passed": bool(all(level["passed"] for level in levels)),
        "control_passed": bool(any(level["control_passed"] for level in levels)),
    }
    if with_rao:
        result["rao_integration"] = _rao_integration_check()
    return result


def _format_table(matrix: np.ndarray, errors: np.ndarray) -> str:
    lines = []
    header = f"{'mode':<7}" + "".join(f"{m:>16}" for m in MODES)
    lines.append(header)
    for i, label in enumerate(MODES):
        row = f"{label:<7}"
        for j in range(6):
            mark = "ok" if errors[i, j] < TOLERANCE else "XX"
            row += f"{matrix[i, j]:>14.4g}/{errors[i, j]*100:4.1f}{mark}"
        lines.append(row)
    return "\n".join(lines)


def plot(result: dict[str, Any], path: Path) -> Path:
    """Two-panel figure: error map of the shipped matrix, and the x_g ladder."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    primary = result["primary"]
    error = np.asarray(primary["rel_error_matrix"])
    control = np.asarray(primary["control_rel_error_matrix"])
    image = axes[0].imshow(
        np.maximum(error, 1e-18), cmap="viridis", norm=matplotlib.colors.LogNorm(vmin=1e-18, vmax=1.0)
    )
    for i in range(6):
        for j in range(6):
            axes[0].text(
                j, i, f"{error[i, j]:.0e}", ha="center", va="center",
                color="white" if error[i, j] < 1e-3 else "black", fontsize=7,
            )
    axes[0].set_xticks(range(6), MODES, rotation=45)
    axes[0].set_yticks(range(6), MODES)
    axes[0].set_title(
        f"|got-ref|/|ref| after fix (max {primary['max_rel_error']:.2e}); "
        f"control max {primary['control_max_rel_error']:.2f}"
    )
    fig.colorbar(image, ax=axes[0], fraction=0.046, label="relative error")

    ladder = result["payload_ladder"]
    x = [row["x_g"] for row in ladder]
    axes[1].semilogy(x, [row["max_rel_error"] for row in ladder], "o-", label="shipped M_RB")
    axes[1].semilogy(
        x, [max(row["control_max_rel_error"], 1e-18) for row in ladder], "s--",
        label="pre-fix coupling block",
    )
    axes[1].axhline(TOLERANCE, color="red", ls=":", label="3% criterion")
    axes[1].axhline(1.0, color="grey", ls=":", lw=0.8)
    axes[1].set_xlabel("payload station $x_g$ [m]")
    axes[1].set_ylabel("max entry relative error [-]")
    axes[1].set_title("CG-offset ladder: the coupling block is exercised")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/case_rigid_body_mass"))
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--no-rao", action="store_true", help="skip the RAO caller check")
    args = parser.parse_args(argv)

    result = run(args.subdivisions, with_rao=not args.no_rao)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    primary = result["primary"]
    ref = np.asarray(primary["ref_matrix"])
    got = np.asarray(primary["got_matrix"])
    errors = np.asarray(primary["rel_error_matrix"])
    report = [
        "CASE-BEM-RIGID-BODY-MASS - 6x6 rigid-body mass matrix vs closed-form block form",
        f"ref: {REFERENCE['source']}",
        "",
        "geometry: " + ", ".join(
            f"{c['name']} {c['length']}x{c['beam']}x{c['height']} m @ {c['center']} rho={c['density']:.3f}"
            for c in result["geometry"]["components"]
        ),
        f"mass = {result['geometry']['mass']:.6f} kg, r_g = {result['geometry']['center_of_gravity']}",
        f"equilibrium draft = {result['geometry']['equilibrium_draft']:.4f} m (inside the 8 m box)",
        "",
        "ref vs got (value/err%), ok = |err| < 3%:",
        _format_table(ref, errors),
        "",
        f"max relative error (shipped) = {primary['max_rel_error']:.3e} "
        f"at {MODES[primary['max_rel_error_loc'][0]]}-{MODES[primary['max_rel_error_loc'][1]]}",
        f"max relative error (pre-fix control) = {primary['control_max_rel_error']:.3e}",
        f"kinetic energy identity error (shipped) = {primary['kinetic_energy_error']:.3e}",
        f"kinetic energy identity error (control) = {primary['control_kinetic_energy_error']:.3e}",
        "geometry agreement analytic vs mesh: " + json.dumps(primary["geometry_agreement"]),
        "",
        "refinement ladder (subdivisions -> panels, max err):",
    ]
    for level in result["levels"]:
        report.append(
            f"  sub={level['subdivisions']:>2} panels={level['panels']:>4} "
            f"max_err={level['max_rel_error']:.3e} control={level['control_max_rel_error']:.3e} "
            f"energy={level['kinetic_energy_error']:.3e}"
        )
    report.append("")
    report.append("x_g ladder (payload station): x_g [m], |coupling| , max err, control err")
    for row in result["payload_ladder"]:
        report.append(
            f"  x_payload={row['payload_x']:>5.1f} x_g={row['x_g']:>6.3f} "
            f"|M_12|={row['coupling_block_norm']:.4e} err={row['max_rel_error']:.3e} "
            f"control={row['control_max_rel_error']:.3e}"
        )
    ell = result.get("ellipsoid")
    if ell:
        report += [
            "",
            "referee-row configuration (tensor_benchmark_cases id=57): solid ellipsoid "
            f"a,b,c={ell['semi_axes']} m, rho={ell['density']} kg/m3, r_g={ell['center_of_gravity']} m",
            "  panels  volume err   inertia_cg err   max entry err   control max entry err",
        ]
        for row in ell["ladder"]:
            report.append(
                f"  {row['panels']:>6}  {row['volume_rel_error']*100:9.4f}%  "
                f"{row['inertia_cg_rel_error']*100:13.4f}%  {row['max_rel_error']*100:13.4f}%  "
                f"{row['control_max_rel_error']*100:19.1f}%"
            )
    if "rao_integration" in result:
        rao = result["rao_integration"]
        report += [
            "",
            f"RAO caller check ({rao['caller']}, omega={rao['omega']}, {rao['panels']} panels):",
            "  |RAO| shipped     : " + ", ".join(f"{v:.6g}" for v in rao["rao_shipped"]),
            "  |RAO| legacy block: " + ", ".join(f"{v:.6g}" for v in rao["rao_with_legacy_coupling"]),
            f"  max change = {rao['max_rao_change']*100:.3f}% in {rao['max_rao_change_mode']}",
            f"  replay with injected ref matrix: max |diff| = {rao['replay_with_injected_ref_matrix']:.3e}",
        ]
    report += ["", f"passed = {result['passed']} (control passed = {result['control_passed']})"]

    text = "\n".join(report) + "\n"
    (out / "report.txt").write_text(text)
    plot(result, out / "rigid_body_mass_error.png")
    print(text)
    return 0 if result["passed"] and not result["control_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())