import numpy as np
import pytest

from tensorbem.geometry import PanelMesh, regular_polygon


def test_regular_polygon_geometry_is_consistent():
    mesh = regular_polygon(16, radius=2.0)
    assert mesh.panel_count == 16
    assert np.all(mesh.lengths > 0.0)
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0)
    assert np.isclose(mesh.perimeter, np.sum(mesh.lengths))


def test_rejects_clockwise_polygon():
    with pytest.raises(ValueError, match="CCW"):
        PanelMesh(np.array([[0., 0.], [0., 1.], [1., 0.]]))


def test_rejects_degenerate_panel_count():
    with pytest.raises(ValueError):
        regular_polygon(2)



@pytest.mark.parametrize(("size", "origin"), [(1.0, 2.0**45), (1e-130, 0.0), (1e130, 0.0)])
def test_rectangle_geometry_under_translation_and_scaling(size, origin, record_property):
    vertices = np.array([[0, 0], [4, 0], [4, 2], [0, 2]]) * size + origin
    mesh = PanelMesh(vertices)
    targets = np.array([[1, 1], [5, 1], [4, 1], [7, 6]]) * size + origin
    np.testing.assert_array_equal(mesh.contains_point(targets[[0, 1, 3]]), [True, False, False])
    clearance_error = max(abs(mesh.boundary_clearance(targets) / size - [1, 1, 0, 5]))
    area_error = abs(mesh.area / size / size / 8 - 1)
    record_property("normalized_clearance_max_error", float(clearance_error))
    record_property("area_relative_error", float(area_error))
    assert clearance_error < 2e-15
    assert area_error < 5e-16
    np.testing.assert_allclose(mesh.lengths / size, [4, 2, 4, 2], rtol=1e-15)
    with pytest.raises(ValueError, match="CCW"):
        PanelMesh(vertices[::-1])
