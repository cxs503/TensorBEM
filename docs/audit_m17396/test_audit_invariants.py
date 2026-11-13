#!/usr/bin/env python3
"""Re-runnable invariants behind the m17396 audit verdict (fast: ~2 s).

Run:  python3 -m pytest docs/audit_m17396/test_audit_invariants.py -q

Each test pins one claim made in AUDIT_REPORT.md, so a future reader can re-check the verdict
mechanically instead of trusting the prose.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import mpmath as mp  # noqa: E402

from tensorbem import irregular_frequency as irr  # noqa: E402
from tensorbem.benchmarks.irregular_frequency_cylinder_exact import (  # noqa: E402
    exact_cylinder_modes,
)
from tensorbem.diffraction_drift_maturity import vertical_cylinder_hull_mesh  # noqa: E402


def _pred(radius: float = 1.0, ratio: float = 0.5, n_theta: int = 32):
    mesh = vertical_cylinder_hull_mesh(radius, radius * ratio, n_theta=n_theta, n_levels=3)
    return mesh, irr.predict_irregular_frequencies(mesh)


def test_bessel_literals_are_the_tabulated_zeros_to_machine_precision():
    """The library constants are correct data (but that is a lookup, not a prediction)."""
    mp.mp.dps = 30
    for order, literal in ((0, irr._BESSEL_J01), (1, irr._BESSEL_J11), (2, irr._BESSEL_J21)):
        zero = float(mp.findroot(lambda x: mp.besselj(order, x), mp.mpf("2.4") + order))
        assert abs(literal - zero) < 1e-14


def test_headline_error_is_exactly_the_waterplane_radius_error():
    """nu_pred = j/R_wp and nu_ref = j/a share the Bessel constant, so err == a/R_wp - 1 for every mode."""
    for ratio in (0.5, 1.0):
        for n_theta in (16, 32):
            _, pred = _pred(ratio=ratio, n_theta=n_theta)
            ident = 1.0 / pred.waterplane_radius - 1.0
            err = np.array([irr._BESSEL_J01, irr._BESSEL_J11, irr._BESSEL_J21]) / pred.waterplane_radius - \
                np.array([irr._BESSEL_J01, irr._BESSEL_J11, irr._BESSEL_J21])
            err = err / np.array([irr._BESSEL_J01, irr._BESSEL_J11, irr._BESSEL_J21])
            assert np.allclose(err, ident, rtol=1e-12, atol=0.0)


def test_residual_follows_the_inscribed_polygon_model():
    """Relative error tracks pi^2/(3 N^2) (polygonal area deficit), not any Bessel quantity."""
    for n_theta in (16, 32):
        mesh, pred = _pred(n_theta=n_theta)
        err = abs(1.0 / pred.waterplane_radius - 1.0)
        model = math.pi ** 2 / (3.0 * n_theta ** 2)
        assert abs(err - model) / model < 0.03
        area = math.pi  # radius 1
        wp = 0.5 * abs(float(np.sum(
            (lambda xy: xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))(
                mesh.vertices[np.unique(mesh.waterline_edges.ravel())][:, :2]))))
        assert wp < area  # inscribed polygon always under-estimates the disk


def test_public_api_cannot_reach_higher_radial_zeros():
    """j_{0,2} = 5.520078... is not reachable: the API caps at the three first-order zeros."""
    _, pred = _pred()
    wide = irr.predict_irregular_frequencies(vertical_cylinder_hull_mesh(1.0, 0.5, n_theta=32, n_levels=3),
                                             n_modes=6)
    assert len(wide.mode_labels) == 3
    mp.mp.dps = 30
    j02 = float(mp.findroot(lambda x: mp.besselj(0, x), mp.mpf("5.5")))
    assert all(abs(v - j02) > 0.05 for v in wide.predicted_nu)  # j21/R=5.19 is merely near j02=5.52


def test_free_surface_route_is_far_above_the_shipped_predictor():
    """Route B gap is 18.6-70.6 %: the shipped formula is the rigid-lid eigenvalue, not the free-surface one."""
    gaps = []
    for ratio in (0.5, 1.0):
        for mode in exact_cylinder_modes(1.0, ratio, angular_orders=(0, 1, 2)):
            gaps.append(abs(mode.nu - mode.radial_wavenumber) / mode.nu)
    assert 0.18 < min(gaps) and max(gaps) < 0.72