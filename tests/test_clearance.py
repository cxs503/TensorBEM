import numpy as np
import pytest

from tensorbem.clearance import diagnose_surface_clearance
from tensorbem.geometry3d import icosphere


def test_clearance_report_distinguishes_far_and_near_points():
    mesh = icosphere(1)
    points = np.array([[3.0, 0.0, 0.0], mesh.centroids[0]])
    with pytest.warns(RuntimeWarning, match="clearance layer"):
        report = diagnose_surface_clearance(mesh, points, warning_factor=1.0)
    assert report.minimum_distance.shape == (2,)
    assert report.minimum_distance[0] > report.minimum_distance[1]
    assert report.clearance_ratio[0] > 1.0
    assert report.unsafe_indices == (1,)
    assert report.threshold_distance == pytest.approx(mesh.h_max)
    assert report.to_dict()["unsafe_count"] == 1
    assert report.to_dict()["point_count"] == 2


def test_clearance_policy_raise_and_ignore_are_deterministic():
    mesh = icosphere(0)
    points = mesh.centroids[:1]
    with pytest.raises(ValueError, match="inside the configured clearance layer"):
        diagnose_surface_clearance(mesh, points, warning_factor=0.5, policy="raise")
    report = diagnose_surface_clearance(
        mesh, points, warning_factor=0.5, policy="ignore"
    )
    assert report.unsafe_indices == (0,)
    assert report.clearance_ratio[0] == pytest.approx(0.0)


def test_clearance_rejects_invalid_inputs():
    mesh = icosphere(0)
    with pytest.raises(ValueError, match="shape"):
        diagnose_surface_clearance(mesh, np.array([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        diagnose_surface_clearance(mesh, np.array([[np.nan, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="warning_factor"):
        diagnose_surface_clearance(mesh, np.array([[3.0, 0.0, 0.0]]), warning_factor=-1.0)
    with pytest.raises(ValueError, match="policy"):
        diagnose_surface_clearance(mesh, np.array([[3.0, 0.0, 0.0]]), policy="silent")
