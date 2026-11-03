"""Analytic verification of the hydrostatic restoring module (``tensorbem.hydrostatic``).

Why this benchmark exists
-------------------------
The hydrostatic layer added for ship seakeeping (:mod:`tensorbem.hydrostatic`,
:mod:`tensorbem.rao`) turns a wetted-hull panel mesh into

    A_wp, grad (displaced volume), KB, BM_T = I_T/grad, BM_L = I_L/grad,
    GM_T = KB + BM_T - KG, GM_L = KB + BM_L - KG,

and from those into the restoring matrix ``C[2,2] = rho g A_wp``,
``C[3,3] = rho g grad GM_T``, ``C[4,4] = rho g grad GM_L``.  Its original
acceptance record reported a *single* coarse-mesh case (floating hemisphere,
``subdivisions=2``) with 2.5 % waterplane-area and 8.8 % displaced-volume error
and labelled both "PASS" without any mesh-refinement evidence.  A number that
large is not a property of the method, it is a property of one grid, and it
cannot tell a discretisation artefact from a modelling error.

This benchmark closes that gap the way every other TensorBEM benchmark does:
against **closed-form reference values** for bodies whose hydrostatics are
textbook exact -- a rectangular box, an upright circular cylinder and a floating
hemisphere -- on a refinement sequence, reporting the observed convergence order
and the restoring-matrix relative errors (geometric quantities and the 6x6
matrix entries are compared separately, because ``C`` inherits both the
waterplane error and the volume error and they can cancel).

Reference formulas (all exact, no series truncation; ``z = 0`` is the calm
waterplane, ``T`` the draft, ``KG`` the centre-of-gravity height above the keel)

Rectangular box, waterplane ``L x B``, vertical sides:

    grad = L B T,            A_wp = L B,           KB = T/2,
    I_T  = L B^3 / 12,       BM_T = B^2 / (12 T),
    I_L  = B L^3 / 12,       BM_L = L^2 / (12 T).

Upright circular cylinder, radius ``R``, vertical sides:

    grad = pi R^2 T,         A_wp = pi R^2,        KB = T/2,
    I_T  = I_L = pi R^4 / 4, BM_T = BM_L = R^2 / (4 T).

Floating hemisphere of radius ``R`` (flat face on the waterplane, so the centre
of the sphere sits at the waterline and the keel at ``z = -R``):

    grad = 2 pi R^3 / 3,     A_wp = pi R^2,        KB = R - 3R/8 = 5R/8,
    I_T  = I_L = pi R^4 / 4, BM_T = BM_L = 3 R / 8, KM = KB + BM = R.

Conventions match :mod:`tensorbem.hydrostatic`: the transverse moment used for
roll is ``I_xx = int y^2 dA`` (about the x axis) and the longitudinal moment
used for pitch is ``I_yy = int x^2 dA`` (about the y axis).

Run::

    python -m tensorbem.benchmarks.hydrostatic_analytic --cases box cylinder hemisphere \\
        --output hydrostatic_verification.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..geometry3d import WettedHullMesh, hemisphere_hull_mesh
from ..hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_hydrostatic_restoring_matrix,
    compute_waterplane_properties,
)

DEFAULT_DENSITY = 1025.0
DEFAULT_GRAVITY = 9.80665


# --------------------------------------------------------------------------- #
# analytic references
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnalyticHydrostatics:
    """Closed-form hydrostatic properties of a floating body."""

    name: str
    displaced_volume: float
    waterplane_area: float
    kb: float
    bm_transverse: float
    bm_longitudinal: float
    transverse_moment: float
    longitudinal_moment: float

    @property
    def km_transverse(self) -> float:
        return self.kb + self.bm_transverse

    @property
    def km_longitudinal(self) -> float:
        return self.kb + self.bm_longitudinal

    def gm_transverse(self, kg: float) -> float:
        return self.km_transverse - kg

    def gm_longitudinal(self, kg: float) -> float:
        return self.km_longitudinal - kg


def box_analytic(length: float, beam: float, draft: float) -> AnalyticHydrostatics:
    """Closed-form hydrostatics of a wall-sided box, ``L x B`` waterplane."""
    volume = length * beam * draft
    return AnalyticHydrostatics(
        name="box",
        displaced_volume=volume,
        waterplane_area=length * beam,
        kb=0.5 * draft,
        bm_transverse=beam**2 / (12.0 * draft),
        bm_longitudinal=length**2 / (12.0 * draft),
        transverse_moment=length * beam**3 / 12.0,
        longitudinal_moment=beam * length**3 / 12.0,
    )


def cylinder_analytic(radius: float, draft: float) -> AnalyticHydrostatics:
    """Closed-form hydrostatics of an upright circular cylinder."""
    volume = math.pi * radius**2 * draft
    second_moment = 0.25 * math.pi * radius**4
    return AnalyticHydrostatics(
        name="cylinder",
        displaced_volume=volume,
        waterplane_area=math.pi * radius**2,
        kb=0.5 * draft,
        bm_transverse=second_moment / volume,
        bm_longitudinal=second_moment / volume,
        transverse_moment=second_moment,
        longitudinal_moment=second_moment,
    )


def hemisphere_analytic(radius: float) -> AnalyticHydrostatics:
    """Closed-form hydrostatics of a hemisphere floating flat-face-up."""
    volume = 2.0 * math.pi * radius**3 / 3.0
    second_moment = 0.25 * math.pi * radius**4
    return AnalyticHydrostatics(
        name="hemisphere",
        displaced_volume=volume,
        waterplane_area=math.pi * radius**2,
        kb=radius - 0.375 * radius,
        bm_transverse=second_moment / volume,
        bm_longitudinal=second_moment / volume,
        transverse_moment=second_moment,
        longitudinal_moment=second_moment,
    )


# --------------------------------------------------------------------------- #
# wetted-hull mesh builders (wall-sided bodies the analytic references cover)
# --------------------------------------------------------------------------- #
def _weld(vertices: Sequence[Sequence[float]], faces: Sequence[Sequence[int]]):
    """Merge coincident vertices and drop the index indirection."""
    coords = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=np.int64)
    if coords.size == 0:
        raise ValueError("empty vertex list")
    keys = np.round(coords, 12)
    _, first, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    order = np.argsort(first)
    remap = np.empty_like(order)
    remap[order] = np.arange(order.size)
    merged = coords[np.sort(first)]
    return merged, remap[inverse][tris]


def _flip_if_inward(vertices: np.ndarray, faces: np.ndarray):
    """Return faces ordered so the enclosed polyhedron has positive volume."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    signed = float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)
    if signed < 0.0:
        faces = faces[:, ::-1].copy()
    return faces


def box_hull_mesh(
    length: float, beam: float, draft: float, n_x: int = 1, n_y: int = 1, n_z: int = 1
) -> WettedHullMesh:
    """Wetted surface of a wall-sided box (bottom + four sides, open at ``z=0``).

    The mesh is a genuinely *faceted* box, so for this body both the waterplane
    polygon and the displaced volume are exact for every refinement level: the
    box case is the zero-error control that isolates waterline-tracing and
    volume-integration bugs from discretisation error.
    """
    for name, value, minimum in (("n_x", n_x, 1), ("n_y", n_y, 1), ("n_z", n_z, 1)):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value in (("length", length), ("beam", beam), ("draft", draft)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")

    hx, hy = 0.5 * length, 0.5 * beam
    tz = -draft
    verts: list[list[float]] = []
    faces: list[list[int]] = []

    def add_quad(p0, p1, p2, p3):
        base = len(verts)
        verts.extend([p0, p1, p2, p3])
        faces.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])

    # bottom panel grid, outward normal -z
    for i in range(n_x):
        for j in range(n_y):
            x0, x1 = -hx + length * i / n_x, -hx + length * (i + 1) / n_x
            y0, y1 = -hy + beam * j / n_y, -hy + beam * (j + 1) / n_y
            add_quad([x0, y0, tz], [x0, y1, tz], [x1, y1, tz], [x1, y0, tz])

    # side panels: sweep z from keel to waterline, outward normal in the side plane
    for k in range(n_z):
        z0 = tz + draft * k / n_z
        z1 = tz + draft * (k + 1) / n_z
        for i in range(n_x):  # +-y faces
            x0, x1 = -hx + length * i / n_x, -hx + length * (i + 1) / n_x
            add_quad([x0, -hy, z0], [x1, -hy, z0], [x1, -hy, z1], [x0, -hy, z1])
            add_quad([x1, hy, z0], [x0, hy, z0], [x0, hy, z1], [x1, hy, z1])
        for j in range(n_y):  # +-x faces
            y0, y1 = -hy + beam * j / n_y, -hy + beam * (j + 1) / n_y
            add_quad([hx, y0, z0], [hx, y1, z0], [hx, y1, z1], [hx, y0, z1])
            add_quad([-hx, y1, z0], [-hx, y0, z0], [-hx, y0, z1], [-hx, y1, z1])

    merged, tris = _weld(verts, faces)
    tris = _flip_if_inward(merged, tris)
    return WettedHullMesh(merged, tris)


def cylinder_hull_mesh(
    radius: float, draft: float, n_theta: int = 24, n_z: int = 1
) -> WettedHullMesh:
    """Wetted surface of an upright circular cylinder (fan bottom + side quads)."""
    if not isinstance(n_theta, (int, np.integer)) or n_theta < 3:
        raise ValueError("n_theta must be an integer >= 3")
    if not isinstance(n_z, (int, np.integer)) or n_z < 1:
        raise ValueError("n_z must be an integer >= 1")
    for name, value in (("radius", radius), ("draft", draft)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")

    thetas = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    verts: list[list[float]] = []
    faces: list[list[int]] = []

    # bottom fan, outward normal -z
    centre = len(verts)
    verts.append([0.0, 0.0, -draft])
    ring = []
    for theta in thetas:
        ring.append(len(verts))
        verts.append([radius * math.cos(theta), radius * math.sin(theta), -draft])
    for i in range(n_theta):
        faces.append([centre, ring[(i + 1) % n_theta], ring[i]])

    # side quads swept from keel to waterline, outward radial normal
    for k in range(n_z):
        z0 = -draft + draft * k / n_z
        z1 = -draft + draft * (k + 1) / n_z
        lower, upper = [], []
        for theta in thetas:
            lower.append(len(verts))
            verts.append([radius * math.cos(theta), radius * math.sin(theta), z0])
        for theta in thetas:
            upper.append(len(verts))
            verts.append([radius * math.cos(theta), radius * math.sin(theta), z1])
        for i in range(n_theta):
            j = (i + 1) % n_theta
            base = len(faces)
            faces.append([lower[i], lower[j], upper[j]])
            faces.append([lower[i], upper[j], upper[i]])

    merged, tris = _weld(verts, faces)
    tris = _flip_if_inward(merged, tris)
    return WettedHullMesh(merged, tris)


def annulus_hull_mesh(
    outer_radius: float,
    inner_radius: float,
    draft: float,
    n_theta: int = 24,
    n_z: int = 1,
) -> WettedHullMesh:
    """Wetted surface of a wall-sided hollow float with a moonpool.

    The body is a vertical annular cylinder: outer wall, inner moonpool wall and
    a bottom ring, open at the waterplane. The inner wall normals point *into*
    the moonpool fluid, so the waterline consists of two loops of opposite
    orientation. This is the smallest geometry that exercises the multiply
    connected waterplane branch of
    :func:`tensorbem.hydrostatic.compute_waterplane_properties`.
    """
    if not isinstance(n_theta, (int, np.integer)) or n_theta < 3:
        raise ValueError("n_theta must be an integer >= 3")
    if not isinstance(n_z, (int, np.integer)) or n_z < 1:
        raise ValueError("n_z must be an integer >= 1")
    for name, value in (
        ("outer_radius", outer_radius),
        ("inner_radius", inner_radius),
        ("draft", draft),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")
    if inner_radius >= outer_radius:
        raise ValueError("inner_radius must be smaller than outer_radius")

    thetas = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    verts: list[list[float]] = []
    faces: list[list[int]] = []

    def ring(radius: float, z: float) -> list[int]:
        ids = []
        for theta in thetas:
            ids.append(len(verts))
            verts.append([radius * math.cos(theta), radius * math.sin(theta), z])
        return ids

    outer_bottom = ring(outer_radius, -draft)
    inner_bottom = ring(inner_radius, -draft)
    outer_top = ring(outer_radius, 0.0)
    inner_top = ring(inner_radius, 0.0)

    # bottom annulus, outward normal -z
    for i in range(n_theta):
        j = (i + 1) % n_theta
        faces.append([outer_bottom[i], inner_bottom[i], inner_bottom[j]])
        faces.append([outer_bottom[i], inner_bottom[j], outer_bottom[j]])

    for k in range(n_z):  # side walls swept from keel to waterline
        z0 = -draft + draft * k / n_z
        z1 = -draft + draft * (k + 1) / n_z
        ol, il = ring(outer_radius, z0), ring(inner_radius, z0)
        ou, iu = ring(outer_radius, z1), ring(inner_radius, z1)
        for i in range(n_theta):
            j = (i + 1) % n_theta
            # outer wall: outward radial normal
            faces.append([ol[i], ol[j], ou[j]])
            faces.append([ol[i], ou[j], ou[i]])
            # inner (moonpool) wall: normal points towards the axis
            faces.append([il[j], il[i], iu[i]])
            faces.append([il[j], iu[i], iu[j]])

    merged, tris = _weld(verts, faces)
    tris = _flip_if_inward(merged, tris)
    return WettedHullMesh(merged, tris)


def twin_box_hull_mesh(
    length: float,
    beam: float,
    draft: float,
    gap: float,
    n_x: int = 1,
    n_y: int = 1,
) -> WettedHullMesh:
    """Two identical wall-sided boxes side by side: a two-loop waterplane."""
    if not math.isfinite(gap) or gap <= 0.0:
        raise ValueError("gap must be a positive finite number")
    offset = 0.5 * beam + 0.5 * gap
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for sign in (-1.0, 1.0):
        single = box_hull_mesh(length, beam, draft, n_x, n_y, 1)
        base = len(verts)
        verts.extend((single.vertices + np.array([0.0, sign * offset, 0.0])).tolist())
        faces.extend((single.faces + base).tolist())
    merged, tris = _weld(verts, faces)
    tris = _flip_if_inward(merged, tris)
    return WettedHullMesh(merged, tris)


def annulus_analytic(outer_radius: float, inner_radius: float, draft: float) -> AnalyticHydrostatics:
    """Closed-form hydrostatics of a wall-sided annular float (moonpool)."""
    waterplane_area = math.pi * (outer_radius**2 - inner_radius**2)
    volume = waterplane_area * draft
    second_moment = 0.25 * math.pi * (outer_radius**4 - inner_radius**4)
    return AnalyticHydrostatics(
        name="annulus",
        displaced_volume=volume,
        waterplane_area=waterplane_area,
        kb=0.5 * draft,
        bm_transverse=second_moment / volume,
        bm_longitudinal=second_moment / volume,
        transverse_moment=second_moment,
        longitudinal_moment=second_moment,
    )


def twin_box_analytic(
    length: float, beam: float, draft: float, gap: float
) -> AnalyticHydrostatics:
    """Closed-form hydrostatics of two identical boxes separated by ``gap``."""
    offset = 0.5 * beam + 0.5 * gap
    waterplane_area = 2.0 * length * beam
    volume = waterplane_area * draft
    own_moment = length * beam**3 / 12.0
    # transverse moment about the combined centreline (y = 0 by symmetry)
    transverse = 2.0 * (own_moment + length * beam * offset**2)
    longitudinal = 2.0 * beam * length**3 / 12.0
    return AnalyticHydrostatics(
        name="twin_box",
        displaced_volume=volume,
        waterplane_area=waterplane_area,
        kb=0.5 * draft,
        bm_transverse=transverse / volume,
        bm_longitudinal=longitudinal / volume,
        transverse_moment=transverse,
        longitudinal_moment=longitudinal,
    )


# --------------------------------------------------------------------------- #
# case definitions and evaluation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Case:
    """One refinement family: mesh builder + closed-form reference + KG."""

    name: str
    levels: tuple[int, ...]
    builder: object
    reference: AnalyticHydrostatics
    kg: float  # centre of gravity height above the keel


def _scale_parameter(analytic: AnalyticHydrostatics) -> tuple[float, float]:
    """Return ``(characteristic_length, panel_area)`` for the convergence fit.

    A wall-sided body is refined isotropically, so the metric is the mean panel
    edge length ``h`` derived from the wetted-surface area divided by the panel
    count: ``h = sqrt(2 A / F)`` for triangles of area ``A/F``.
    """
    return 0.0, 0.0


def evaluate_case(case: Case, level: int, density: float, gravity: float) -> dict:
    """Compute hydrostatics at one refinement level and compare with the reference."""
    hull = case.builder(level)
    ref = case.reference

    wp_area, wp_centre, wp_moments = compute_waterplane_properties(hull)
    volume, cb = compute_displaced_volume_and_buoyancy(hull)
    mass = density * volume
    cg = np.array([wp_centre[0], wp_centre[1], case.kg], dtype=float)  # z from keel?
    # ``kg`` is measured above the keel; the module works in the z=0-centred frame.
    z_keel = float(np.min(hull.vertices[:, 2]))
    cg = np.array([0.0, 0.0, z_keel + case.kg], dtype=float)

    props = compute_hydrostatic_restoring_matrix(
        hull, mass, cg, density=density, gravity=gravity
    )
    kb = float(cb[2] - z_keel)
    bm_t = float(wp_moments[0] / volume) if volume > 0.0 else math.nan
    bm_l = float(wp_moments[1] / volume) if volume > 0.0 else math.nan
    gm_t = kb + bm_t - case.kg
    gm_l = kb + bm_l - case.kg
    c = props.restoring_matrix

    # Convergence metric: the mean waterline edge length. Every quantity computed
    # here is a waterline integral (A_wp, I_T, I_L) or follows the waterline
    # polygonisation (the displaced volume of a wall-sided body is A_wp * T), so
    # the waterline segment length -- not the mean panel size -- is the parameter
    # that controls the error, and it is comparable across body families whose
    # meshes refine differently (a cylinder doubles n_theta at fixed vertical
    # subdivision, an icosphere refines isotropically).
    va = hull.vertices[hull.waterline_edges[:, 0]]
    vb = hull.vertices[hull.waterline_edges[:, 1]]
    h = float(np.mean(np.linalg.norm(vb - va, axis=1)))

    def rel(value: float, reference: float) -> float:
        return float(value - reference) / float(reference)

    return {
        "level": int(level),
        "panel_count": int(hull.panel_count),
        "h": h,
        "waterplane_area": float(wp_area),
        "waterplane_area_rel": rel(wp_area, ref.waterplane_area),
        "volume": float(volume),
        "volume_rel": rel(volume, ref.displaced_volume),
        "kb": kb,
        "kb_rel": rel(kb, ref.kb),
        "transverse_moment": float(wp_moments[0]),
        "transverse_moment_rel": rel(wp_moments[0], ref.transverse_moment),
        "longitudinal_moment": float(wp_moments[1]),
        "longitudinal_moment_rel": rel(wp_moments[1], ref.longitudinal_moment),
        "bm_transverse": bm_t,
        "bm_transverse_rel": rel(bm_t, ref.bm_transverse),
        "gm_transverse": gm_t,
        "gm_transverse_rel": rel(gm_t, ref.gm_transverse(case.kg)),
        "gm_longitudinal": gm_l,
        "gm_longitudinal_rel": rel(gm_l, ref.gm_longitudinal(case.kg)),
        "c_heave": float(c[2, 2]),
        "c_heave_rel": rel(c[2, 2], density * gravity * ref.waterplane_area),
        "c_roll": float(c[3, 3]),
        "c_roll_rel": rel(
            c[3, 3],
            density * gravity * ref.displaced_volume * ref.gm_transverse(case.kg),
        ),
        "c_pitch": float(c[4, 4]),
        "c_pitch_rel": rel(
            c[4, 4],
            density * gravity * ref.displaced_volume * ref.gm_longitudinal(case.kg),
        ),
    }


def observed_order(levels: Iterable[dict], key: str) -> list[float]:
    """Observed convergence orders ``p = log(e1/e2) / log(h1/h2)`` between levels."""
    rows = list(levels)
    orders = []
    for a, b in zip(rows[:-1], rows[1:]):
        e1, e2 = abs(a[key]), abs(b[key])
        if e1 <= 0.0 or e2 <= 0.0 or a["h"] == b["h"]:
            orders.append(math.nan)
        else:
            orders.append(float(math.log(e1 / e2) / math.log(a["h"] / b["h"])))
    return orders


def build_cases(beam: float = 1.0, draft: float = 0.5) -> list[Case]:
    """Default three-body verification family (box, cylinder, hemisphere)."""
    length = 4.0 * beam
    box_ref = box_analytic(length, beam, draft)
    cyl_ref = cylinder_analytic(0.5 * beam, draft)
    hem_ref = hemisphere_analytic(0.5 * beam)

    return [
        Case(
            name="box",
            levels=(1, 2, 4, 8),
            builder=lambda n: box_hull_mesh(length, beam, draft, n, n, 1),
            reference=box_ref,
            kg=0.5 * draft,
        ),
        Case(
            name="cylinder",
            levels=(8, 16, 32, 64),
            builder=lambda n: cylinder_hull_mesh(0.5 * beam, draft, n, 2),
            reference=cyl_ref,
            kg=0.5 * draft,
        ),
        Case(
            name="hemisphere",
            levels=(2, 3, 4, 5),
            builder=lambda n: hemisphere_hull_mesh(n, 0.5 * beam),
            reference=hem_ref,
            kg=0.5 * (0.5 * beam),
        ),
        Case(
            name="twin_box",
            levels=(1, 2, 4, 8),
            builder=lambda n: twin_box_hull_mesh(
                length=2.0, beam=0.5, draft=0.4, gap=1.0, n_x=n, n_y=n
            ),
            reference=twin_box_analytic(2.0, 0.5, 0.4, 1.0),
            kg=0.5 * 0.4,
        ),
        Case(
            name="moonpool_annulus",
            levels=(8, 16, 32, 64),
            builder=lambda n: annulus_hull_mesh(0.5, 0.25, draft, n, 2),
            reference=annulus_analytic(0.5, 0.25, draft),
            kg=0.5 * draft,
        ),
    ]


def run_study(
    cases: Sequence[Case] | None = None,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> dict:
    """Run the full refinement study and return a JSON-serialisable summary."""
    cases = list(cases) if cases is not None else build_cases()
    results = {
        "density": density,
        "gravity": gravity,
        "cases": [],
    }
    for case in cases:
        rows = [evaluate_case(case, level, density, gravity) for level in case.levels]
        reference = asdict(case.reference)
        reference["km_transverse"] = case.reference.km_transverse
        reference["km_longitudinal"] = case.reference.km_longitudinal
        results["cases"].append(
            {
                "name": case.name,
                "kg": case.kg,
                "reference": reference,
                "gm_transverse": case.reference.gm_transverse(case.kg),
                "gm_longitudinal": case.reference.gm_longitudinal(case.kg),
                "levels": rows,
                "observed_orders": {
                    key: observed_order(rows, key)
                    for key in (
                        "waterplane_area_rel",
                        "volume_rel",
                        "transverse_moment_rel",
                        "gm_transverse_rel",
                        "c_roll_rel",
                    )
                },
            }
        )
    return results


def run_rao_long_wave(
    subdivisions: int = 2,
    radius: float = 0.5,
    omegas: Sequence[float] = (0.6, 0.9, 1.2),
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> dict:
    """Check the long-wave heave limit ``|xi_3| -> 1`` of the RAO assembly.

    In the limit ``k a -> 0`` the wave excitation force on a surface-piercing body
    tends to the hydrostatic force ``rho g A_wp zeta_a`` and the inertial term
    ``-omega^2 (M + A)`` vanishes against the restoring term, so the heave RAO
    must tend to unity whatever the panelisation. The check therefore validates
    the *assembly* of ``C``, ``A`` and ``F_exc`` (a missing ``rho g`` or a factor
    two in the excitation is caught immediately) independently of the
    discretisation error quantified in the refinement study.
    """
    from ..geometry3d import hemisphere_hull_mesh
    from ..rao import compute_rao_at_frequency

    hull = hemisphere_hull_mesh(subdivisions, radius)
    cg = np.array([0.0, 0.0, -radius + 0.5 * radius])
    props = compute_hydrostatic_restoring_matrix(hull, 0.0, cg, density, gravity)
    mass = density * props.displaced_volume

    rows = []
    for omega in omegas:
        result = compute_rao_at_frequency(
            hull,
            float(omega),
            0.0,
            mass,
            cg,
            hydrostatic_props=props,
            density=density,
            gravity=gravity,
        )
        rows.append(
            {
                "omega": float(omega),
                "ka": float(omega) ** 2 / gravity * radius,
                "heave_rao": float(result.rao_amplitude[2]),
                "excitation_over_hydrostatic": float(
                    abs(result.excitation_force[2])
                    / (density * gravity * props.waterplane_area)
                ),
                "impedance_residual": float(result.impedance_residual),
            }
        )
    return {
        "subdivisions": int(subdivisions),
        "panels": int(hull.panel_count),
        "radius": float(radius),
        "rows": rows,
    }


def _plot(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for case in results["cases"]:
        rows = case["levels"]
        h = [r["h"] for r in rows]
        axes[0].loglog(h, [abs(r["waterplane_area_rel"]) for r in rows], "o-", label=f"{case['name']} A_wp")
        axes[0].loglog(h, [abs(r["volume_rel"]) for r in rows], "s--", label=f"{case['name']} grad")
        axes[1].loglog(h, [abs(r["gm_transverse_rel"]) for r in rows], "o-", label=f"{case['name']} GM_T")
        axes[1].loglog(h, [abs(r["c_roll_rel"]) for r in rows], "s--", label=f"{case['name']} C33")

    for ax, title in zip(axes, ("waterplane area / displaced volume", "transverse GM / roll restoring")):
        ax.set_title(title)
        ax.set_xlabel("panel edge length h [m]")
        ax.set_ylabel("relative error")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("TensorBEM hydrostatic analytic verification")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_markdown(results: dict) -> str:
    """Render the study as a markdown table block (numbers read from the JSON)."""
    lines = []
    for case in results["cases"]:
        ref = case["reference"]
        lines.append(f"### {case['name']} (KG = {case['kg']:.4f} m, reference values exact)")
        lines.append("")
        lines.append(
            "| panels | h [m] | A_wp err | vol err | I_T err | GM_T err | C33 err |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for row in case["levels"]:
            lines.append(
                "| {p} | {h:.4f} | {a:+.3e} | {v:+.3e} | {i:+.3e} | {g:+.3e} | {c:+.3e} |".format(
                    p=row["panel_count"],
                    h=row["h"],
                    a=row["waterplane_area_rel"],
                    v=row["volume_rel"],
                    i=row["transverse_moment_rel"],
                    g=row["gm_transverse_rel"],
                    c=row["c_roll_rel"],
                )
            )
        lines.append("")
        orders = case["observed_orders"]
        lines.append(
            "observed order (A_wp, vol, I_T, GM_T, C33) = ({:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f})".format(
                *[orders[k][-1] for k in (
                    "waterplane_area_rel",
                    "volume_rel",
                    "transverse_moment_rel",
                    "gm_transverse_rel",
                    "c_roll_rel",
                )]
            )
        )
        lines.append("")
        lines.append(
            "analytic: A_wp={:.6f} m^2, grad={:.6f} m^3, KB={:.6f} m, BM_T={:.6f} m, GM_T={:.6f} m".format(
                ref["waterplane_area"],
                ref["displaced_volume"],
                ref["kb"],
                ref["bm_transverse"],
                case["gm_transverse"],
            )
        )
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=None, help="JSON output path")
    parser.add_argument("--figure", type=Path, default=None, help="PNG convergence plot")
    parser.add_argument("--markdown", type=Path, default=None, help="markdown table dump")
    parser.add_argument(
        "--with-rao",
        action="store_true",
        help="also run the long-wave heave RAO limit check (a few BEM solves)",
    )
    args = parser.parse_args(argv)

    results = run_study()
    if args.with_rao:
        results["rao_long_wave"] = run_rao_long_wave()
    payload = json.dumps(results, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    if args.figure is not None:
        _plot(results, args.figure)
    table = render_markdown(results)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(table + "\n")
    print(table)
    return results


if __name__ == "__main__":  # pragma: no cover
    main()