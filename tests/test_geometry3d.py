import numpy as np
import pytest

from tensorbem.geometry3d import TriangleMesh, icosphere


@pytest.mark.parametrize("level,panels,vertices", [(0, 20, 12), (1, 80, 42), (2, 320, 162)])
def test_icosphere_counts_geometry_and_quality(level, panels, vertices):
    mesh = icosphere(level)
    assert mesh.panel_count == panels
    assert mesh.vertex_count == vertices
    radii = np.linalg.norm(mesh.vertices, axis=1)
    assert np.allclose(radii, 1.0)
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0)
    assert np.all(np.einsum("ij,ij->i", mesh.normals, mesh.centroids) > 0.0)
    assert mesh.surface_area < 4.0 * np.pi
    assert mesh.volume < 4.0 / 3.0 * np.pi
    quality = mesh.quality
    assert quality["min_edge_length"] > 0.0
    assert quality["max_edge_length"] >= quality["min_edge_length"]
    assert quality["edge_length_ratio"] >= 1.0
    assert 0.0 < quality["min_angle_degrees"] < quality["max_angle_degrees"] < 180.0
    assert quality["max_aspect_ratio"] >= 1.0
    assert np.isfinite(list(quality.values())).all()


def test_icosphere_refinement_improves_area_and_volume():
    coarse, medium, fine = icosphere(0), icosphere(1), icosphere(2)
    assert coarse.surface_area < medium.surface_area < fine.surface_area
    assert coarse.volume < medium.volume < fine.volume
    assert fine.surface_area / (4.0 * np.pi) > medium.surface_area / (4.0 * np.pi)


def test_icosphere_scales_with_radius():
    base = icosphere(1, radius=1.0)
    scaled = icosphere(1, radius=2.5)
    assert np.allclose(np.linalg.norm(scaled.vertices, axis=1), 2.5)
    assert np.isclose(scaled.surface_area, 2.5**2 * base.surface_area)
    assert np.isclose(scaled.volume, 2.5**3 * base.volume)
    assert np.isclose(scaled.h_max, 2.5 * base.h_max)


def test_mesh_rejects_open_or_misoriented_surfaces():
    mesh = icosphere(0)
    with pytest.raises(ValueError, match="closed|directed"):
        TriangleMesh(mesh.vertices, mesh.faces[:-1])

    flipped = mesh.faces.copy()
    flipped[0] = flipped[0][::-1]
    with pytest.raises(ValueError, match="closed|directed|outward"):
        TriangleMesh(mesh.vertices, flipped)

    with pytest.raises(ValueError, match="outward"):
        TriangleMesh(mesh.vertices, mesh.faces[:, ::-1])


def test_mesh_rejects_invalid_indices_and_degenerate_faces():
    mesh = icosphere(0)
    repeated = mesh.faces.copy()
    repeated[0, 1] = repeated[0, 0]
    with pytest.raises(ValueError, match="distinct"):
        TriangleMesh(mesh.vertices, repeated)

    out_of_range = mesh.faces.copy()
    out_of_range[0, 0] = mesh.vertex_count
    with pytest.raises(ValueError, match="range"):
        TriangleMesh(mesh.vertices, out_of_range)

    with pytest.raises(ValueError, match="integer"):
        TriangleMesh(mesh.vertices, mesh.faces.astype(float))


def test_mesh_validates_shapes_and_finiteness():
    mesh = icosphere(0)
    with pytest.raises(ValueError, match=r"shape \(V, 3\)"):
        TriangleMesh(mesh.vertices[:, :2], mesh.faces)
    with pytest.raises(ValueError, match=r"shape \(F, 3\)"):
        TriangleMesh(mesh.vertices, mesh.faces[:, :2])
    with pytest.raises(ValueError, match="finite"):
        bad = mesh.vertices.copy()
        bad[0, 0] = np.nan
        TriangleMesh(bad, mesh.faces)
    with pytest.raises(ValueError, match="at least four vertices"):
        TriangleMesh(mesh.vertices[:3], mesh.faces[:4])
    with pytest.raises(ValueError, match="at least four faces"):
        TriangleMesh(mesh.vertices, mesh.faces[:3])


@pytest.mark.parametrize("subdivisions", [-1, 1.5, True])
def test_icosphere_rejects_invalid_subdivisions(subdivisions):
    with pytest.raises(ValueError):
        icosphere(subdivisions)


@pytest.mark.parametrize("radius", [0.0, -1.0, np.nan])
def test_icosphere_rejects_invalid_radius(radius):
    with pytest.raises(ValueError):
        icosphere(0, radius=radius)
