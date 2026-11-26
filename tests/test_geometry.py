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


@pytest.mark.parametrize("offset,scale", [(0., 1.), (2.**36, 1.), (0., 2.**450), (0., 2.**-450)])
def test_oblique_panel_clearance_on_and_next_to_edge(offset, scale, record_property):
    # First edge has direction (3,7). Perturb only y at its exact midpoint,
    # so the perpendicular distance is 3*|delta_y|/sqrt(58).
    vertices = offset + scale*np.array([[0.,0.], [3.,7.], [-6.,4.]])
    mesh = PanelMesh(vertices)
    mid = offset + scale*np.array([1.5,3.5])
    targets = np.tile(mid, (3,1))
    targets[1:,1] = np.nextafter(mid[1], [-np.inf, np.inf])
    expected = abs(targets[1:,1]-mid[1])*(3/np.sqrt(58.))
    actual = mesh.boundary_clearance(targets)
    record_property("on_edge_distance_per_scale", float(actual[0]/scale))
    record_property("max_neighbour_relative_error", float(max(abs(actual[1:]/expected-1))))
    assert actual[0] == 0.
    np.testing.assert_allclose(actual[1:], expected, atol=0., rtol=2e-15)


@pytest.mark.parametrize("direction", [(3,7), (3,-7), (-3,7), (-3,-7)])
@pytest.mark.parametrize("scale", [1., 2.**450, 2.**-450])
def test_ray_cast_resolves_adjacent_points_at_oblique_midpoint(direction, scale, record_property):
    edge = np.asarray(direction, dtype=float)*scale
    midpoint = np.array([.25, .125])*scale
    # Counterclockwise triangle: increasing y enters the triangle for dx>0.
    normal = np.array([-edge[1], edge[0]])
    mesh = PanelMesh(np.array([midpoint-edge, midpoint+edge, midpoint+normal]))
    points = np.tile(midpoint, (2, 1))
    points[:, 1] = np.nextafter(midpoint[1], [-np.inf, np.inf])
    expected = np.array([direction[0] < 0, direction[0] > 0])
    measured = mesh.contains_point(points)
    record_property("classification_errors", int(np.count_nonzero(expected != measured)))
    np.testing.assert_array_equal(measured, expected)
