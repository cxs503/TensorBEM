"""Verification tests for cambered Joukowski sections (see docs/lifting_camber.md)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks.lifting_camber import ARTIFACT, kutta_residual
from tensorbem.lifting import (
    JoukowskiAirfoil,
    camber_line,
    glauert_thin_aerofoil_lift,
    solve_lifting_flow,
)

THICKNESS = 0.1
RADIUS = 1.0
ALPHA = 4.0


def _foil(camber: float, thickness: float = THICKNESS, alpha: float = ALPHA):
    return JoukowskiAirfoil(thickness, RADIUS, alpha, camber=camber)


def test_symmetric_reference_is_bit_identical():
    """Adding ``camber`` must not move any symmetric-section number."""
    foil = _foil(0.0)
    assert foil.circle_radius == 1.1
    assert foil.chord == pytest.approx(4.033333333333333, rel=1e-15)
    assert foil.leading_edge == pytest.approx((-2.033333333333333, 0.0), rel=1e-15)
    assert foil.chord_line_angle_deg == 0.0
    assert foil.cusp_angle == 0.0
    assert foil.exact_circulation() == pytest.approx(0.9642442720014428, rel=1e-15)
    assert foil.decay_offset() == pytest.approx(THICKNESS * math.cos(math.radians(ALPHA)))


@pytest.mark.parametrize("camber", [-0.15, 0.0, 0.05, 0.1, 0.2])
def test_cusp_stays_on_the_circle_and_at_the_trailing_edge(camber):
    foil = _foil(camber)
    cusp = foil.map_point(foil.transform_radius)
    assert abs(cusp - complex(*foil.trailing_edge)) < 1e-12
    assert abs(complex(*foil.circle_centre) + foil.circle_radius * complex(
        math.cos(foil.cusp_angle), math.sin(foil.cusp_angle)
    ) - foil.transform_radius) < 1e-12
    mesh = foil.mesh(32)
    assert mesh.vertices[0] == pytest.approx(foil.trailing_edge, abs=1e-12)


@pytest.mark.parametrize("camber", [-0.15, 0.0, 0.05, 0.1, 0.2])
def test_cusp_is_a_stagnation_point_for_every_camber(camber):
    """``dF/dzeta = 0`` at the cusp is what defines the circulation."""
    foil = _foil(camber)
    assert kutta_residual(foil) < 1e-12


@pytest.mark.parametrize("camber", [0.0, 0.05, 0.1, 0.2])
def test_zero_lift_incidence_is_the_cusp_angle(camber):
    foil = _foil(camber)
    zero_lift = math.degrees(foil.cusp_angle)
    assert foil.exact_circulation() == 0.0 or abs(
        _foil(camber, alpha=zero_lift).exact_circulation()
    ) < 1e-12
    # camber lifts at zero incidence: the zero-lift angle is negative
    assert (zero_lift < 0.0) == (camber > 0.0)
    assert (zero_lift > 0.0) == (camber < 0.0)


def test_camber_increases_lift_monotonically():
    lifts = [_foil(camber).exact_lift_coefficient() for camber in (0.0, 0.05, 0.1, 0.2)]
    assert all(later > earlier for earlier, later in zip(lifts, lifts[1:]))


def test_glauert_route_converges_to_the_closed_form_as_the_section_thins():
    gaps = []
    for thickness in (0.05, 0.025):
        foil = _foil(thickness, thickness=thickness)
        thin = glauert_thin_aerofoil_lift(foil, stations=201)
        gaps.append(abs(foil.exact_lift_coefficient() - thin["lift_coefficient"])
                    / abs(thin["lift_coefficient"]))
    assert gaps[1] < gaps[0] * 0.65, gaps
    assert gaps[1] < 0.08, gaps


def test_camber_line_of_a_symmetric_section_is_the_chord():
    line = camber_line(_foil(0.0), stations=101)
    assert np.max(np.abs(line["eta"])) < 2.0e-3 * line["chord"]
    assert line["camber_ratio"] < 5.0e-4


def test_measured_camber_ratio_grows_with_the_geometric_offset():
    ratios = [camber_line(_foil(camber), stations=201)["camber_ratio"]
              for camber in (0.0, 0.05, 0.1)]
    assert ratios[0] < 1e-3 < ratios[1] < ratios[2]


def test_bem_closure_converges_on_a_cambered_section():
    """Pinned values from docs/lifting_camber.md (64 panels, camber 0.1)."""
    foil = _foil(0.1)
    result = solve_lifting_flow(foil, 64)
    exact_circulation = -foil.exact_circulation()
    error = (result.circulation_ccw - exact_circulation) / abs(exact_circulation)
    assert error == pytest.approx(0.1322, abs=5e-4)
    assert result.kutta_residual < 1e-12
    assert abs(result.lift_coefficient_from_circulation
               - result.lift_coefficient_from_pressure) < 0.02
    assert result.circulation_ccw < 0.0  # counter-clockwise, lift up


def test_bem_is_worse_on_cambered_than_on_symmetric_at_the_same_mesh():
    """The honest finding: camber raises the error coefficient, rate unchanged."""
    symmetric = _foil(0.0)
    cambered = _foil(0.1)
    errors = []
    for foil in (symmetric, cambered):
        exact = -foil.exact_circulation()
        result = solve_lifting_flow(foil, 64)
        errors.append(abs((result.circulation_ccw - exact) / exact))
    assert errors[1] > 2.0 * errors[0]


@pytest.mark.skipif(not ARTIFACT.exists(), reason="benchmark artifact not generated")
def test_artifact_reproduces_the_documented_numbers():
    summary = json.loads(ARTIFACT.read_text())
    rows = summary["mesh_ladders"]["0.1"]
    row = next(item for item in rows if item["panels"] == 128)
    foil = _foil(0.1)
    exact = -foil.exact_circulation()
    result = solve_lifting_flow(foil, 128)
    assert result.circulation_ccw == pytest.approx(row["circulation_ccw"], rel=1e-12)
    assert (result.circulation_ccw - exact) / abs(exact) == pytest.approx(
        row["circulation_error"], rel=1e-12
    )
    heavy = summary["heavy"]
    assert heavy["lift_exact"] == pytest.approx(foil.exact_lift_coefficient(), rel=1e-12)
    document = Path("docs/lifting_camber.md")
    if document.exists():
        text = document.read_text()
        assert f"{row['circulation_ccw']:+.5f}" in text
        assert f"{heavy['circulation_ccw']:+.5f}" in text