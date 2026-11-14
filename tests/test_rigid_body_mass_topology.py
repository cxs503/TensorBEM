"""Topology guard of the signed-tetrahedron mass-property route.

``mesh_mass_properties`` is the *independent* reference of the case-pool anchor
``CASE-BEM-RIGID-BODY-MASS``: it re-derives volume, centroid and second moments
from the panelised surface without touching the closed block form.  That only
holds when the surface is closed and consistently oriented.  Before the guard
was added the function accepted an open patch, a flipped triangle and a
duplicated triangle without a word, returning a volume wrong at the percent
level -- the same order as the panelisation error the anchor resolves, so the
defect could not be seen from the acceptance metric.  These tests pin the guard
and keep the pre-fix ("legacy") behaviour as a control that must stay wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.geometry3d import ellipsoid_mesh
from tensorbem.rigid_body_mass import (
    mesh_mass_properties,
    mesh_topology_report,
    solid_ellipsoid_mass_properties,
)

AXES = (60.0, 12.0, 8.0)
DENSITY = 1025.0
SUBDIVISIONS = 3


def ellipsoid():
    return ellipsoid_mesh(SUBDIVISIONS, np.asarray(AXES))


def legacy_volume(vertices, faces, reference_point=(0.0, 0.0, 0.0)):
    """Pre-fix signed-tetrahedron sum, written out independently of the module.

    This is the *un*guarded integration: it is what the function used to return
    for any input whose total signed volume happened to be positive.
    """
    verts = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)
    p0 = np.asarray(reference_point, dtype=float)
    a, b, c = (verts[tris[:, k]] - p0 for k in range(3))
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


class TestRejectsInvalidSurfaces:
    def test_open_patch_is_rejected(self):
        mesh = ellipsoid()
        with pytest.raises(ValueError, match="not a closed"):
            mesh_mass_properties(mesh.vertices, mesh.faces[:-1], DENSITY)

    def test_single_flipped_triangle_is_rejected(self):
        mesh = ellipsoid()
        faces = mesh.faces.copy()
        faces[0] = faces[0][::-1]
        report = mesh_topology_report(mesh.vertices, faces)
        assert report["orientation_mismatches"] == 3
        with pytest.raises(ValueError, match="not a closed"):
            mesh_mass_properties(mesh.vertices, faces, DENSITY)

    def test_duplicated_triangle_is_rejected(self):
        mesh = ellipsoid()
        faces = np.vstack([mesh.faces, mesh.faces[:1]])
        report = mesh_topology_report(mesh.vertices, faces)
        assert report["nonmanifold_edges"] == 3
        with pytest.raises(ValueError, match="not a closed"):
            mesh_mass_properties(mesh.vertices, faces, DENSITY)

    def test_inward_orientation_is_rejected(self):
        mesh = ellipsoid()
        with pytest.raises(ValueError):
            mesh_mass_properties(mesh.vertices, mesh.faces[:, ::-1], DENSITY)

    def test_out_of_range_face_index_is_rejected(self):
        mesh = ellipsoid()
        faces = mesh.faces.copy()
        faces[0, 0] = len(mesh.vertices)
        with pytest.raises(ValueError, match="out of range"):
            mesh_topology_report(mesh.vertices, faces)


class TestReportShape:
    def test_closed_mesh_reports_clean(self):
        mesh = ellipsoid()
        report = mesh_topology_report(mesh.vertices, mesh.faces)
        assert report["is_closed"] is True
        assert report["open_edges"] == 0
        assert report["nonmanifold_edges"] == 0
        assert report["orientation_mismatches"] == 0
        assert report["degenerate_faces"] == 0
        assert report["boundary_length"] == 0.0
        assert report["n_faces"] == len(mesh.faces)
        assert report["welded_vertices"] == 0

    def test_open_patch_reports_its_boundary(self):
        mesh = ellipsoid()
        report = mesh_topology_report(mesh.vertices, mesh.faces[:-1])
        assert report["is_closed"] is False
        assert report["open_edges"] == 3
        assert report["boundary_length"] > 0.0


class TestWelding:
    def test_coincident_vertices_are_welded(self):
        """A surface whose every triangle owns a private copy of its nodes."""
        mesh = ellipsoid()
        verts = mesh.vertices[mesh.faces].reshape(-1, 3)
        faces = np.arange(len(verts)).reshape(-1, 3)
        report = mesh_topology_report(verts, faces)
        assert report["is_closed"] is True
        assert report["welded_vertices"] == len(verts) - len(mesh.vertices)
        guarded = mesh_mass_properties(verts, faces, DENSITY)
        direct = mesh_mass_properties(mesh.vertices, mesh.faces, DENSITY)
        assert guarded["volume"] == pytest.approx(direct["volume"], rel=1e-12)

    def test_near_coincident_seam_needs_tolerance(self):
        mesh = ellipsoid()
        verts = mesh.vertices.copy()
        # Duplicate the mesh with a 1e-12 shift and connect the halves through
        # the shifted nodes only: exact welding cannot see the seam, a 1e-9
        # tolerance can.  (Both halves are closed on their own, so this tests
        # the parameter rather than a defect.)
        shifted = verts + 1e-12
        report = mesh_topology_report(shifted, mesh.faces)
        assert report["is_closed"] is True  # shift is rigid, topology unchanged
        report = mesh_topology_report(
            np.vstack([verts, shifted]),
            np.vstack([mesh.faces, mesh.faces + len(verts)]),
            weld_tolerance=1e-9,
        )
        assert report["welded_vertices"] == len(verts)

    def test_invalid_weld_tolerance_is_rejected(self):
        mesh = ellipsoid()
        with pytest.raises(ValueError, match="weld_tolerance"):
            mesh_topology_report(mesh.vertices, mesh.faces, weld_tolerance=0.0)


class TestLegacyBlindSpotControl:
    """The pre-fix sum stays wrong: the guard is what makes the route sound."""

    @pytest.mark.parametrize(
        "corruption",
        ["open", "flipped", "duplicated"],
    )
    def test_legacy_sum_is_finite_and_wrong(self, corruption):
        mesh = ellipsoid()
        faces = mesh.faces
        if corruption == "open":
            faces = faces[:-1]
        elif corruption == "flipped":
            faces = faces.copy()
            faces[0] = faces[0][::-1]
        else:
            faces = np.vstack([faces, faces[:1]])
        reference = solid_ellipsoid_mass_properties(AXES, DENSITY)
        error = abs(legacy_volume(mesh.vertices, faces) - reference["volume"]) / reference["volume"]
        # finite, plausible (sub-3%: it would have passed the anchor criterion)
        # and wrong -- this is exactly why a tolerance test cannot catch it.
        assert np.isfinite(error)
        assert 1e-3 < error < 0.03