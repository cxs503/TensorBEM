"""Order of the recovery operator on its own, pinned to the measured value.

``tensorbem.recovery`` used to describe the least-squares patch fit as
*second-order accurate*, and the only convergence tests measured the composite
solve+recovery pipeline, where the order of the two stages cannot be separated.
The isolated measurement
(:mod:`tensorbem.benchmarks.recovery_order`, audited in
``docs/m3692/recovery_order_audit.md``) shows the operator is **first order**
asymptotically: the relative L2 error of the recovered tangential gradient
against the panel-plane projection of the analytic gradient falls by a factor
1.9-2.6 per mesh halving, not by 4.

The fit is exact for affine data and linear in the samples, so its error is
exactly the gradient it returns for the ``O(h**2)`` curvature residual; the
first order comes from the third moment of the patch offsets (patch
asymmetry).  These tests pin that mechanism as well, so a future change of the
patch or of the fit that silently restores second order (or degrades to zero
order) fails here instead of only in a document.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from tensorbem.benchmarks.recovery_order import (
    FIELD_SPECS,
    main,
    predicted_fit_error,
    project,
    study,
    targets,
)
from tensorbem.geometry3d import icosphere
from tensorbem.recovery import recover_tangent_gradient

#: Coarsest mesh of the refinement ladder used by the order tests.  The whole
#: ladder (320/1280/5120 panels) costs about two seconds.
LADDER = (2, 3, 4)


@pytest.fixture(scope="module")
def quadratic() -> dict:
    return study(LADDER, ("x_times_y",))["rows"][0]


@pytest.fixture(scope="module")
def cubic() -> dict:
    return study(LADDER, ("cubic_x3",))["rows"][0]


def _orders(row: dict, key: str = "panel_target") -> list[float]:
    return [entry["observed_order"] for entry in row["orders"][key]]


def test_the_recovery_operator_is_first_order_and_not_second_order(quadratic):
    errors = [level["panel_target"] for level in quadratic["levels"]]
    assert errors == sorted(errors, reverse=True)
    for order in _orders(quadratic):
        # First order: a factor 2 per halving.  Second order would need 2.0
        # here (order 2); the band is wide enough for the pre-asymptotic
        # coarse level and narrow enough to exclude it.
        assert 0.9 < order < 1.4
    for entry in quadratic["orders"]["panel_target"]:
        assert entry["error_ratio"] < 3.0


def test_the_cubic_control_shows_the_same_order(cubic):
    for order in _orders(cubic):
        assert 0.9 < order < 1.5


def test_first_order_constant_is_stable_on_the_fine_meshes(quadratic):
    # error/h settles near 0.20; a second-order operator would have this
    # constant fall by 2 per level.
    ratios = [
        level["panel_target"] / level["h_mean"] for level in quadratic["levels"]
    ]
    assert 0.18 < ratios[-1] < 0.22
    assert 0.18 < ratios[-2] < 0.22
    assert ratios[0] > ratios[-1]


def test_fit_error_is_reproduced_by_the_curvature_residual(quadratic):
    # The prediction is computed from q_j alone, without the library routine.
    for level in quadratic["levels"]:
        assert level["prediction_residual"] < 1.0e-9


def test_the_order_does_not_come_from_the_reference_definition(quadratic):
    # The panel-plane and smooth-surface references differ at O(h); both give
    # the same order, so the first order is the operator's, not the faceting.
    for entry in quadratic["orders"]["target_definition_gap"]:
        assert 0.85 < entry["observed_order"] < 1.15
    assert _orders(quadratic, "smooth_target") == pytest.approx(
        _orders(quadratic, "panel_target"), abs=0.05
    )


def test_predicted_fit_error_is_the_affine_defect_of_the_patch():
    # Sanity check of the prediction itself on one mesh: it must reproduce the
    # library output to round-off on every field, not just on the L2 average.
    mesh = icosphere(2, radius=1.0)
    for name, (value, _) in FIELD_SPECS.items():
        panel_target, _ = targets(mesh, name)
        measured = recover_tangent_gradient(mesh, value(mesh.centroids)) - panel_target
        predicted = predicted_fit_error(mesh, name)
        scale = float(np.max(np.abs(measured)))
        assert float(np.max(np.abs(predicted - measured))) < 1.0e-10 * scale


def test_the_driver_writes_the_table(tmp_path):
    out = tmp_path / "recovery_order.json"
    main(["--subdivisions", "1", "--json", str(out)])
    payload = json.loads(out.read_text())
    assert payload["study"] == "recovery_order"
    assert payload["panels"] == [80]
    level = payload["rows"][0]["levels"][0]
    assert level["relative_l2" if "relative_l2" in level else "panel_target"] > 0.0
    assert np.isfinite(level["prediction_residual"])