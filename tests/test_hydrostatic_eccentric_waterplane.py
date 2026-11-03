"""Eccentric (off-origin) waterplanes: closed-form verification and mutation closure.

Why this file exists
--------------------
The hydrostatic suite verifies boxes, cylinders, hemispheres, twin pontoons and
annular floats -- but **every one of those waterplanes is centred on the
coordinate origin**.  Two consequences were recorded as open items by the
mutation audit (`docs/mutation_audit_tensorbem.md`, m2582):

* A multiply/divide swap in the first-moment sum of
  :func:`tensorbem.hydrostatic.compute_waterplane_properties`
  (``x_moment += ... / 6.0`` -> ``* 6.0``) survived the whole test suite,
  because for a centred waterplane the first moment is identically zero and the
  factor cancels out of the centroid.
* The product-of-inertia component ``I_xy`` was never asserted on a case where
  it is non-zero, so its parallel-axis shift ``I_xy - A c_x c_y`` was unverified.

Both are genuine verification debt, not just missing test coverage: the
waterplane centroid and ``I_xy`` feed roll/pitch stiffness and any heave-roll /
heave-pitch coupling of a real (asymmetric) hull.

What is asserted here
---------------------
1. **Closed-form references** for three off-origin convex waterplanes:

   * an axis-aligned rectangle translated to ``x in [0.5, 2.5]``,
     ``y in [-0.5, 1.5]`` -- first moments non-zero, ``I_xy = 0``;
   * a right triangle with legs 2.0 (x) and 1.2 (y) translated to
     ``(0.3, 0.2)`` -- ``I_xx = A h^2/18``, ``I_yy = A b^2/18``,
     ``I_xy = -A b h/36`` about the centroid;
   * an oblique parallelogram spanned by ``u = (2.0, 0.4)`` and
     ``v = (0.6, 1.5)`` from the corner ``(1.0, -1.0)`` -- ``I_xy != 0``, so the
     product-of-inertia shift is exercised, including its sign.

   A second, independent reference (exact triangle-moment fan triangulation with
   the 3-point quadratic-degree rule) is computed in the test file for the same
   polygons, so an error shared by both closed forms is not possible.

2. **Two-loop pooling with a non-zero pooled centroid.**  The existing twin-hull
   test uses symmetric pontoon offsets, so the pooled centroid is ``(0, 0)`` and
   the ``A_i * d^2`` parallel-axis terms cancel pairwise.  Here two dissimilar
   prisms sit at dissimilar offsets, so area-weighted pooling is pinned for
   area, centroid, both second moments and the product of inertia.

3. **Internal consistency between two independent integrations.**  For a
   wall-sided prism the centre of buoyancy (3-D tetrahedron sum over the wetted
   panels) must sit vertically above the waterplane centroid (2-D Green's
   theorem over the waterline).  The two integrals share no code.

4. **Reference-origin convention.**  The module reports second moments about the
   *pooled waterplane centroid*, so ``C[2,2]``, ``C[3,3]`` and ``C[4,4]`` of a
   body must be invariant under a lateral translation of the body.  That
   invariance is what makes the reported restoring matrix correct for an origin
   whose vertical axis pierces the waterplane centroid; see
   ``docs/hydrostatic_eccentric_waterplane.md`` for the coupling terms that are
   *not* modelled when the centroid is offset from the rotation axis.

5. **Defensive branches.**  The zero-waterplane and zero-volume early returns are
   unreachable through the validating ``WettedHullMesh`` constructor (a wetted
   hull with a zero-area waterline encloses zero volume, which the constructor
   rejects), so they are pinned at contract level with a duck-typed stand-in.

Everything is exact (faceted prisms, straight waterlines), so the tolerances are
machine-precision plus the mesh-node rounding, not discretisation allowances.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.geometry3d import WettedHullMesh
from tensorbem.hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_hydrostatic_restoring_matrix,
    compute_waterplane_properties,
)

DENSITY = 1025.0
GRAVITY = 9.80665


# --------------------------------------------------------------------------- #
# mesh construction
# --------------------------------------------------------------------------- #
def prism_mesh(polygon, draft: float) -> WettedHullMesh:
    """Wall-sided prism: a simple CCW waterplane polygon extruded to ``z = -draft``.

    The wetted surface is the bottom face (fan triangulation, normal ``-z``) plus
    the side panels, leaving the top open at ``z = 0`` where the waterline lives.
    Both the waterplane polygon and the displaced volume are therefore exact for
    the meshed body -- this is a zero-discretisation-error control, not a
    convergence study.
    """
    poly = np.asarray(polygon, dtype=float)
    count = poly.shape[0]
    verts = [[x, y, -draft] for x, y in poly] + [[x, y, 0.0] for x, y in poly]
    faces: list[list[int]] = []
    for i in range(1, count - 1):  # bottom, seen from -z the fan is clockwise
        faces.append([0, i + 1, i])
    for i in range(count):  # sides, outward normals
        j = (i + 1) % count
        faces.append([i, j, count + j])
        faces.append([i, count + j, count + i])
    return WettedHullMesh(np.asarray(verts, dtype=float), np.asarray(faces, dtype=int))


def merge_meshes(meshes: list[WettedHullMesh]) -> WettedHullMesh:
    """Merge disjoint wetted hulls into one multi-loop wetted mesh."""
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for hull in meshes:
        vertices.append(hull.vertices)
        faces.append(hull.faces + offset)
        offset += hull.vertices.shape[0]
    return WettedHullMesh(np.vstack(vertices), np.vstack(faces))


# --------------------------------------------------------------------------- #
# independent reference: exact triangle-moment fan triangulation
# --------------------------------------------------------------------------- #
def fan_reference(polygon):
    """Area, centroid and centroidal moments by fan triangulation of the polygon.

    Each triangle contributes its exact degree-2 moments through the three-point
    edge-midpoint rule (exact for quadratic integrands), then everything is
    pooled and shifted to the pooled centroid.  This is a different algorithm
    from the library's edge-wise Green's theorem sums.
    """
    poly = np.asarray(polygon, dtype=float)
    origin = poly[0]
    area = mx = my = i_xx = i_yy = i_xy = 0.0
    for i in range(1, len(poly) - 1):
        tri = np.array([origin, poly[i], poly[i + 1]])
        tri_area = 0.5 * (
            (tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1])
            - (tri[2, 0] - tri[0, 0]) * (tri[1, 1] - tri[0, 1])
        )
        mids = 0.5 * (tri + np.roll(tri, -1, axis=0))
        area += tri_area
        mx += tri_area / 3.0 * mids[:, 0].sum()
        my += tri_area / 3.0 * mids[:, 1].sum()
        i_xx += tri_area / 3.0 * (mids[:, 1] ** 2).sum()
        i_yy += tri_area / 3.0 * (mids[:, 0] ** 2).sum()
        i_xy += tri_area / 3.0 * (mids[:, 0] * mids[:, 1]).sum()
    cx, cy = mx / area, my / area
    return (
        area,
        np.array([cx, cy]),
        np.array([i_xx - area * cy**2, i_yy - area * cx**2, i_xy - area * cx * cy]),
    )


def pool_reference(parts):
    """Pool several ``(area, centroid, centroidal_moments)`` triples."""
    area = sum(part[0] for part in parts)
    centroid = sum(part[0] * part[1] for part in parts) / area
    moments = np.zeros(3)
    for part_area, part_centroid, part_moments in parts:
        delta = part_centroid - centroid
        moments[0] += part_moments[0] + part_area * delta[1] ** 2
        moments[1] += part_moments[1] + part_area * delta[0] ** 2
        moments[2] += part_moments[2] + part_area * delta[0] * delta[1]
    return area, centroid, moments


# waterplane polygons (counter-clockwise)
OFFSET_RECTANGLE = [(0.5, -0.5), (2.5, -0.5), (2.5, 1.5), (0.5, 1.5)]
OFFSET_TRIANGLE = [(0.3, 0.2), (2.3, 0.2), (0.3, 1.4)]
OBLIQUE_PARALLELOGRAM = [(1.0, -1.0), (3.0, -0.6), (3.6, 0.9), (1.6, 0.5)]
SMALL_OFFSET_BOX = [(4.0, 0.5), (5.0, 0.5), (5.0, 1.0), (4.0, 1.0)]


# --------------------------------------------------------------------------- #
# 1. closed-form waterplane properties, off-origin
# --------------------------------------------------------------------------- #
def test_offset_rectangle_waterplane_matches_closed_form() -> None:
    """x in [0.5, 2.5], y in [-0.5, 1.5]: A = 4, centre (1.5, 0.5), I_xx = I_yy = 4/3.

    The waterplane does not straddle the origin, so its first moments are exactly
    4*1.5 = 6 and 4*0.5 = 2.  A defect that scales the first-moment sum (the
    ``/ 6.0`` -> ``* 6.0`` mutant) moves the centroid by a factor 36 and is caught
    here, while a swap inside a centred case is invisible.
    """
    hull = prism_mesh(OFFSET_RECTANGLE, draft=1.3)
    area, centroid, moments = compute_waterplane_properties(hull)

    length_x, length_y = 2.0, 2.0
    exact_area = length_x * length_y
    assert area == pytest.approx(exact_area, rel=1e-14)
    assert centroid == pytest.approx([1.5, 0.5], abs=1e-13)
    assert moments[0] == pytest.approx(length_x * length_y**3 / 12.0, rel=1e-13)
    assert moments[1] == pytest.approx(length_y * length_x**3 / 12.0, rel=1e-13)
    assert moments[2] == pytest.approx(0.0, abs=1e-13)

    reference = fan_reference(OFFSET_RECTANGLE)
    assert area == pytest.approx(reference[0], rel=1e-13)
    assert centroid == pytest.approx(reference[1], abs=1e-13)
    assert moments == pytest.approx(reference[2], abs=1e-13)


def test_offset_right_triangle_waterplane_matches_closed_form() -> None:
    """Legs b = 2.0 (x), h = 1.2 (y): A = 1.2, centroid (0.96667, 0.6).

    Centroidal moments of a right triangle are ``A h^2 / 18`` (about the centroid
    line parallel to the x axis) and ``A b^2 / 18``; the product of inertia is
    ``-A b h / 36``, i.e. a *named* sign that a symmetric case cannot test.
    """
    hull = prism_mesh(OFFSET_TRIANGLE, draft=0.9)
    area, centroid, moments = compute_waterplane_properties(hull)

    b, h = 2.0, 1.2
    exact_area = 0.5 * b * h
    assert area == pytest.approx(exact_area, rel=1e-14)
    assert centroid == pytest.approx(
        [(0.3 + 2.3 + 0.3) / 3.0, (0.2 + 0.2 + 1.4) / 3.0], abs=1e-13
    )
    assert moments[0] == pytest.approx(exact_area * h**2 / 18.0, rel=1e-13)
    assert moments[1] == pytest.approx(exact_area * b**2 / 18.0, rel=1e-13)
    assert moments[2] == pytest.approx(-exact_area * b * h / 36.0, rel=1e-13)

    reference = fan_reference(OFFSET_TRIANGLE)
    assert moments == pytest.approx(reference[2], abs=1e-13)


def test_oblique_waterplane_product_of_inertia_matches_closed_form() -> None:
    """Parallelogram spanned by u = (2, 0.4), v = (0.6, 1.5) from (1, -1).

    For a parallelogram, ``I_xx = A (u_y^2 + v_y^2)/12``,
    ``I_yy = A (u_x^2 + v_x^2)/12`` and ``I_xy = A (u_x u_y + v_x v_y)/12`` about
    the centroid.  ``I_xy = 0.391`` is non-zero and of one sign, so both the
    product-of-inertia sum and its parallel-axis shift
    ``I_xy - A c_x c_y`` are pinned; a sign slip in the shift changes
    ``I_xy`` by 2*A*c_x*c_y = -0.46, far outside tolerance.
    """
    hull = prism_mesh(OBLIQUE_PARALLELOGRAM, draft=1.1)
    area, centroid, moments = compute_waterplane_properties(hull)

    u = np.array([2.0, 0.4])
    v = np.array([0.6, 1.5])
    exact_area = abs(u[0] * v[1] - u[1] * v[0])
    assert area == pytest.approx(exact_area, rel=1e-14)
    assert centroid == pytest.approx(
        np.mean(np.asarray(OBLIQUE_PARALLELOGRAM, dtype=float), axis=0), abs=1e-13
    )
    assert moments[0] == pytest.approx(exact_area * (u[1] ** 2 + v[1] ** 2) / 12.0, rel=1e-13)
    assert moments[1] == pytest.approx(exact_area * (u[0] ** 2 + v[0] ** 2) / 12.0, rel=1e-13)
    assert moments[2] == pytest.approx(
        exact_area * (u[0] * u[1] + v[0] * v[1]) / 12.0, rel=1e-13
    )
    assert abs(moments[2]) > 0.3  # the assertion above would be vacuous at I_xy = 0

    reference = fan_reference(OBLIQUE_PARALLELOGRAM)
    assert moments == pytest.approx(reference[2], abs=1e-13)


# --------------------------------------------------------------------------- #
# 2. multi-loop pooling with a non-zero pooled centroid
# --------------------------------------------------------------------------- #
def test_two_loop_pooling_with_offset_centroid() -> None:
    """A 2x2 prism at (1.5, 0.5) plus a 1x0.5 prism at (4.5, 0.75).

    Areas 4 and 0.5, pooled centroid (1.6111..., 0.5277...).  Both ``A_i d_i^2``
    parallel-axis contributions are non-zero and unequal, so a pooling bug that
    is invisible for symmetric twin hulls (where ``d_1 = -d_2`` and the pooled
    centroid is the origin) shows up immediately in area, centroid, ``I_xx``,
    ``I_yy`` and ``I_xy``.
    """
    hull = merge_meshes(
        [
            prism_mesh(OFFSET_RECTANGLE, draft=1.3),
            prism_mesh(SMALL_OFFSET_BOX, draft=0.4),
        ]
    )
    area, centroid, moments = compute_waterplane_properties(hull)

    reference = pool_reference(
        [fan_reference(OFFSET_RECTANGLE), fan_reference(SMALL_OFFSET_BOX)]
    )
    assert area == pytest.approx(reference[0], rel=1e-13)
    assert centroid == pytest.approx(reference[1], abs=1e-13)
    assert moments == pytest.approx(reference[2], abs=1e-13)
    assert centroid[0] > 1.5  # pooled centroid is pulled towards the small box
    assert area == pytest.approx(4.5, rel=1e-13)


# --------------------------------------------------------------------------- #
# 3. coupling block of the restoring matrix on an eccentric body
# --------------------------------------------------------------------------- #
def test_centre_of_buoyancy_sits_over_the_waterplane_centroid() -> None:
    """Wall-sided prism: CB_x, CB_y equal the waterplane centroid; volume = A * draft.

    The centre of buoyancy is a 3-D tetrahedron sum over the wetted panels and the
    centroid a 2-D Green's theorem sum over the waterline, so agreement is a
    cross-check between two independent integrals rather than a restatement of
    either one.
    """
    for polygon, draft in (
        (OFFSET_RECTANGLE, 1.3),
        (OFFSET_TRIANGLE, 0.9),
        (OBLIQUE_PARALLELOGRAM, 1.1),
    ):
        hull = prism_mesh(polygon, draft=draft)
        area, centroid, _ = compute_waterplane_properties(hull)
        volume, buoyancy_centre = compute_displaced_volume_and_buoyancy(hull)

        assert volume == pytest.approx(area * draft, rel=1e-13)
        assert buoyancy_centre[:2] == pytest.approx(centroid, abs=1e-13)
        assert buoyancy_centre[2] == pytest.approx(-0.5 * draft, rel=1e-13)


def test_restoring_matrix_is_invariant_under_lateral_translation() -> None:
    """Translating a floating body must not change C[2,2], C[3,3], C[4,4].

    The module reports waterplane moments about the pooled centroid and the
    metacentric radii ``I / volume`` from those, so heave, roll and pitch stiffness
    are properties of the body alone.  A missing centroid shift (or a shift with
    the wrong sign, i.e. ``+A c^2`` instead of ``-A c^2``) would make the roll and
    pitch stiffness depend on where the user put the origin -- a defect the
    centred test cases cannot see.
    """
    draft = 1.4
    base = [(0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (0.0, 1.0)]
    shifted = [(x + 2.7, y - 1.9) for x, y in base]

    mass = 3.0 * 1.0 * draft * DENSITY  # not needed for equality, but physical
    cg = np.array([0.0, 0.0, -0.5 * draft])
    centred = compute_hydrostatic_restoring_matrix(prism_mesh(base, draft), mass, cg)
    moved_cg = np.array([2.7, -1.9, -0.5 * draft])
    moved = compute_hydrostatic_restoring_matrix(prism_mesh(shifted, draft), mass, moved_cg)

    assert moved.waterplane_center == pytest.approx(
        np.asarray(centred.waterplane_center) + np.array([2.7, -1.9]), abs=1e-12
    )
    for row, column in ((2, 2), (3, 3), (4, 4)):
        assert moved.restoring_matrix[row, column] == pytest.approx(
            centred.restoring_matrix[row, column], rel=1e-12
        )


def test_eccentric_box_restoring_matrix_matches_closed_form() -> None:
    """Off-origin box barge against ``rho g A``, ``rho g I_xx``, ``rho g I_yy``.

    With the centre of gravity at the centre of buoyancy the metacentric heights
    reduce to the metacentric radii, so ``C[3,3] = rho g grad BM_T = rho g I_xx``
    and ``C[4,4] = rho g I_yy`` exactly -- the waterplane moments that the
    off-origin cases above verify now propagate to the stiffness matrix.
    """
    polygon = [(2.4, 1.1), (8.4, 1.1), (8.4, 3.1), (2.4, 3.1)]  # 6.0 x 2.0
    draft = 1.5
    hull = prism_mesh(polygon, draft=draft)
    area, centroid, moments = compute_waterplane_properties(hull)
    volume, buoyancy_centre = compute_displaced_volume_and_buoyancy(hull)

    mass = DENSITY * volume
    cg = np.array([centroid[0], centroid[1], buoyancy_centre[2]])
    props = compute_hydrostatic_restoring_matrix(hull, mass, cg, density=DENSITY)

    assert props.restoring_matrix[2, 2] == pytest.approx(
        DENSITY * GRAVITY * area, rel=1e-13
    )
    assert props.restoring_matrix[3, 3] == pytest.approx(
        DENSITY * GRAVITY * moments[0], rel=1e-13
    )
    assert props.restoring_matrix[4, 4] == pytest.approx(
        DENSITY * GRAVITY * moments[1], rel=1e-13
    )
    # ... and equal to the textbook box formulas for this draft
    assert props.restoring_matrix[3, 3] == pytest.approx(
        DENSITY * GRAVITY * 6.0 * 2.0**3 / 12.0, rel=1e-13
    )
    assert props.restoring_matrix[4, 4] == pytest.approx(
        DENSITY * GRAVITY * 2.0 * 6.0**3 / 12.0, rel=1e-13
    )
    # A body whose waterplane centroid is off the origin has *no* modelled
    # heave-roll / heave-pitch coupling in this module; the magnitude of the
    # omitted term is quantified in docs/hydrostatic_eccentric_waterplane.md.
    assert props.restoring_matrix[2, 4] == 0.0
    assert props.restoring_matrix[2, 3] == 0.0


# --------------------------------------------------------------------------- #
# 4. defensive branches, pinned at contract level
# --------------------------------------------------------------------------- #
class _StubHull:
    """Attribute stand-in for ``WettedHullMesh`` (the consumers duck-type)."""

    def __init__(self, vertices, *, waterline_edges=None, faces=None) -> None:
        self.vertices = np.asarray(vertices, dtype=float)
        self.waterline_edges = (
            np.zeros((0, 2), dtype=int)
            if waterline_edges is None
            else np.asarray(waterline_edges, dtype=int)
        )
        self.faces = np.zeros((0, 3), dtype=int) if faces is None else np.asarray(faces, dtype=int)


def test_zero_area_waterline_returns_zero_properties() -> None:
    """A degenerate (collinear) waterline encloses no area: 0, zeros(2), zeros(3).

    Such a mesh cannot be built through ``WettedHullMesh`` -- zero waterplane area
    means zero displaced volume, which the constructor rejects -- so the early
    return in ``compute_waterplane_properties`` is defensive.  It is pinned here
    because the mutation audit showed the ``0.0 -> 1.0`` mutant surviving: a
    spurious 1 m^2 waterplane would inject ``rho g = 1.0e4 N/m`` of heave
    stiffness into a body that has none.
    """
    stub = _StubHull(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.3, -1.0]],
        waterline_edges=[[0, 1], [1, 2], [2, 0]],
    )
    area, centroid, moments = compute_waterplane_properties(stub)
    assert area == 0.0
    assert np.asarray(centroid).shape == (2,)
    assert np.asarray(moments).shape == (3,)
    assert not np.any(centroid)
    assert not np.any(moments)


def test_no_waterline_returns_zero_properties() -> None:
    """A fully submerged body has no waterline edges: zero area, zeros(2), zeros(3)."""
    stub = _StubHull([[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
    area, centroid, moments = compute_waterplane_properties(stub)
    assert area == 0.0
    assert np.asarray(centroid).shape == (2,)
    assert np.asarray(moments).shape == (3,)


def test_zero_volume_face_list_returns_zero_buoyancy() -> None:
    """A face list whose signed tetrahedron volumes cancel returns (0.0, zeros(3)).

    The pair of oppositely oriented triangles encloses nothing, so the centre of
    buoyancy must be the origin *of length three*; the mutation audit showed the
    ``np.zeros(3) -> np.zeros(4)`` mutant surviving in this branch.
    """
    vertices = np.array(
        [[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, -1.0]]
    )
    faces = np.array([[0, 1, 2], [3, 2, 1]])
    volume, centre = compute_displaced_volume_and_buoyancy(_StubHull(vertices, faces=faces))
    assert volume == 0.0
    assert np.asarray(centre).shape == (3,)
    assert not np.any(centre)


def test_empty_face_list_returns_zero_buoyancy() -> None:
    """No wetted panels at all: (0.0, zeros(3)) -- the other defensive branch."""
    stub = _StubHull([[0.0, 0.0, -1.0]])
    volume, centre = compute_displaced_volume_and_buoyancy(stub)
    assert volume == 0.0
    assert np.asarray(centre).shape == (3,)
    assert not np.any(centre)