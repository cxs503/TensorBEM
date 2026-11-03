"""Guardrails for the waterplane / displaced-volume degeneracies and the
eccentric-waterplane centroid.

Why this file exists
--------------------
``docs/mutation_audit_tensorbem.md`` §5 registers four surviving mutants as
*real* coverage gaps rather than equivalent mutants.  Two of them live in
:mod:`tensorbem.hydrostatic`:

* ``hydrostatic.py:236`` — the waterplane first moment ``x_moment`` has its
  ``/ 6.0`` swapped for ``* 6.0`` and every test stays green, because all
  existing waterplane examples (barge, hemisphere, spheroid) are symmetric
  about ``x = 0`` (and ``y = 0``), so the first moment is ~0 and a wrong
  normalisation is invisible;
* ``hydrostatic.py:255`` and ``hydrostatic.py:314`` — the two degenerate
  sentinel returns (``0.0`` for the waterplane area, ``np.zeros(3)`` for the
  centre of buoyancy) are never exercised, so their shape *and* value are
  unpinned.

The tests below pin them with cases that have closed-form answers:

1. an **eccentric** wall-sided barge whose waterplane is ``x in [0, 2]``,
   ``y in [0, 1]``: centroid ``(1, 0.5)``, ``I_xx = L B^3 / 12 = 1/6``,
   ``I_yy = B L^3 / 12 = 2/3``, ``I_xy = 0`` — exact to round-off, so the
   distinction between ``/6`` and ``*6`` is a factor 36 in the centroid;
2. a **fully submerged** body (no waterline at all): the waterplane sentinel
   must be exactly ``0.0`` / ``np.zeros(2)`` / ``np.zeros(3)`` and the
   restoring matrix must stay finite;
3. a **sub-threshold** hull (all dimensions ``1e-9``, ``|V| < 1e-14``): the
   displaced-volume and waterplane sentinels must be exactly zero, with the
   documented array shapes.

None of these are new physics; they are regression fences for branches that
the mutation audit proved were dead weight.
"""

import numpy as np
import pytest

from tensorbem.geometry3d import (
    WettedHullMesh,
    box_hull_mesh,
    submerged_sphere_mesh,
)
from tensorbem.hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_hydrostatic_restoring_matrix,
    compute_waterplane_properties,
)

DENSITY = 1025.0
GRAVITY = 9.80665


def _translate(hull: WettedHullMesh, shift) -> WettedHullMesh:
    """Return the same hull shifted in the horizontal plane."""
    return WettedHullMesh(hull.vertices + np.asarray(shift, dtype=float), hull.faces)


class TestEccentricWaterplaneCentroid:
    """The waterplane first moment must survive a horizontal offset."""

    @pytest.mark.parametrize("shift", [(1.0, 0.5), (0.0, 0.0), (-0.7, 0.3)])
    def test_centroid_and_second_moments_of_an_offset_barge(self, shift):
        length, beam, draft = 2.0, 1.0, 1.0
        hull = box_hull_mesh(
            length, beam, draft, n_length=3, n_beam=2, n_draft=2
        )
        shifted = _translate(hull, (shift[0], shift[1], 0.0))

        area, centroid, moments = compute_waterplane_properties(shifted)

        # Area is translation invariant and exact for a wall-sided barge.
        assert area == pytest.approx(length * beam, rel=1e-13)
        # Second moments about the *pooled centroid* are translation
        # invariant: I_xx = L B^3 / 12, I_yy = B L^3 / 12, I_xy = 0.
        assert moments[0] == pytest.approx(length * beam**3 / 12.0, rel=1e-12)
        assert moments[1] == pytest.approx(beam * length**3 / 12.0, rel=1e-12)
        assert moments[2] == pytest.approx(0.0, abs=1e-14)

        # The centroid follows the offset.  A swapped factor 6 (the surviving
        # mutant) changes it by a factor 36, i.e. far outside this tolerance.
        assert centroid == pytest.approx(
            np.asarray(shift, dtype=float), abs=1e-12
        )

    def test_offset_barge_restoring_matrix_is_translation_invariant(self):
        """C must not depend on where the hull sits in the horizontal plane."""
        hull = box_hull_mesh(2.0, 1.0, 1.0, n_length=2, n_beam=2, n_draft=2)
        volume, _ = compute_displaced_volume_and_buoyancy(hull)
        mass = DENSITY * volume
        cog = np.array([0.0, 0.0, -0.5])

        base = compute_hydrostatic_restoring_matrix(hull, mass, cog)
        moved = compute_hydrostatic_restoring_matrix(
            _translate(hull, (1.0, 0.5, 0.0)), mass, cog + np.array([1.0, 0.5, 0.0])
        )
        assert np.allclose(
            base.restoring_matrix, moved.restoring_matrix, rtol=1e-12, atol=0.0
        )
        # C[2,2] = rho g A_wp, with A_wp = L B = 2.0 m^2 for this barge.
        assert moved.restoring_matrix[2, 2] == pytest.approx(
            DENSITY * GRAVITY * 2.0 * 1.0, rel=1e-12
        )


class TestDegenerateSentinels:
    """Unreachable-by-accident branches: pin value *and* shape."""

    def test_fully_submerged_body_has_no_waterplane(self):
        sphere = submerged_sphere_mesh(2, radius=1.0, submergence=5.0)
        # Closing a submerged sphere needs no waterline lid: every directed
        # edge is matched, so the waterline edge list is empty.
        hull = WettedHullMesh(sphere.vertices, sphere.faces)
        assert hull.waterline_edges.shape == (0, 2)
        area, centroid, moments = compute_waterplane_properties(hull)

        assert area == 0.0
        assert isinstance(centroid, np.ndarray) and centroid.shape == (2,)
        assert isinstance(moments, np.ndarray) and moments.shape == (3,)
        assert np.array_equal(centroid, np.zeros(2))
        assert np.array_equal(moments, np.zeros(3))

        # The restoring matrix must stay finite: a fully submerged body has no
        # free-surface restoring, so the heave term is exactly zero.
        volume, cob = compute_displaced_volume_and_buoyancy(hull)
        assert volume > 0.0
        props = compute_hydrostatic_restoring_matrix(
            hull, DENSITY * volume, cob.copy()
        )
        assert np.isfinite(props.restoring_matrix).all()
        # No waterplane -> no free-surface heave restoring, exactly zero.
        assert props.restoring_matrix[2, 2] == 0.0

    def test_sub_threshold_waterline_loop_hits_the_area_sentinel(self):
        """A waterline loop of negligible area must fall through to the sentinel.

        This is the *second* degenerate return (``hydrostatic.py:255``): the
        edge list is non-empty (so the empty-waterline branch is not taken),
        but every traced loop encloses less than ``1e-14`` m^2, so the pooled
        area stays zero.  Such a loop appears physically as a pair of
        nearly-coincident waterline edges, e.g. on a hull whose waterline
        is numerically pinched at a sharp end.
        """

        class _SliverHull:
            vertices = np.array(
                [[0.0, 0.0, 0.0], [1.0e-8, 0.0, 0.0], [0.0, 1.0e-8, 0.0]]
            )
            waterline_edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)

        area, centre, moments = compute_waterplane_properties(_SliverHull())
        assert area == 0.0
        assert isinstance(centre, np.ndarray) and centre.shape == (2,)
        assert isinstance(moments, np.ndarray) and moments.shape == (3,)
        assert np.array_equal(centre, np.zeros(2))
        assert np.array_equal(moments, np.zeros(3))

    def test_zero_volume_hull_hits_the_volume_sentinel(self):
        """``|V| < 1e-14`` must return exactly ``(0.0, np.zeros(3))``.

        A collapsed (zero-thickness) surface is the numerical situation this
        guard exists for: it is produced by a scaling bug or by a mesh whose
        panels all lie in one plane.  :class:`WettedHullMesh` itself refuses to
        build such a mesh (positive volume is enforced), so the guard is
        exercised through a minimal duck-typed hull with the same
        ``vertices``/``faces`` interface -- which is exactly what the function
        consumes.
        """

        class _FlatHull:
            vertices = np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            )
            faces = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int64)

        volume, centre = compute_displaced_volume_and_buoyancy(_FlatHull())
        assert volume == 0.0
        assert isinstance(centre, np.ndarray) and centre.shape == (3,)
        assert np.array_equal(centre, np.zeros(3))

    def test_empty_panel_list_hits_the_same_sentinel(self):
        """The zero-face guard returns the identical ``(0.0, np.zeros(3))``."""

        class _EmptyHull:
            vertices = np.zeros((0, 3))
            faces = np.zeros((0, 3), dtype=np.int64)

        volume, centre = compute_displaced_volume_and_buoyancy(_EmptyHull())
        assert volume == 0.0
        assert centre.shape == (3,)
        assert np.array_equal(centre, np.zeros(3))
