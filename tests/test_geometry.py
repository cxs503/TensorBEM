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
