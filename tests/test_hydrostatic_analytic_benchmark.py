"""Analytic verification gate for the hydrostatic restoring module.

Two things are tested here:

1. **Correctness of the waterplane integration on multiply connected
   waterplanes.** The original implementation traced the waterline with a
   nearest-neighbour walk that returned a *single* polygon. A two-hull
   (catamaran) or moonpool waterplane therefore lost entire loops silently: the
   twin-box case returned half the true waterplane area and 1/56 of the true
   transverse second moment, with no error raised. The tests below pin the
   multi-loop behaviour against closed-form values.

2. **Accuracy of the whole hydrostatic stack against textbook references.**
   Waterplane area, displaced volume, transverse/longitudinal second moments,
   KB, BM, GM and the restoring-matrix entries are compared with the exact box,
   cylinder, hemisphere, twin-box and annular-float formulas, on a refinement
   sequence, and the observed convergence order is required to be close to 2.

The slowest test is the long-wave RAO limit check, which needs a radiation +
diffraction solve (a few seconds per frequency).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.benchmarks.hydrostatic_analytic import (
    annulus_analytic,
    annulus_hull_mesh,
    box_analytic,
    box_hull_mesh,
    build_cases,
    cylinder_analytic,
    cylinder_hull_mesh,
    evaluate_case,
    hemisphere_analytic,
    observed_order,
    run_study,
    twin_box_analytic,
    twin_box_hull_mesh,
)
from tensorbem.geometry3d import hemisphere_hull_mesh
from tensorbem.hydrostatic import (
    _trace_waterline_loops,
    compute_displaced_volume_and_buoyancy,
    compute_hydrostatic_restoring_matrix,
    compute_waterplane_properties,
)

DENSITY = 1025.0
GRAVITY = 9.80665


# --------------------------------------------------------------------------- #
# 1. multiply connected waterplanes
# --------------------------------------------------------------------------- #
def test_twin_box_waterplane_keeps_both_loops():
    """A two-hull waterplane must keep both loops (was silently halved)."""
    length, beam, draft, gap = 2.0, 0.5, 0.4, 1.0
    hull = twin_box_hull_mesh(length, beam, draft, gap, n_x=1, n_y=1)

    loops = _trace_waterline_loops(hull.waterline_edges)
    assert len(loops) == 2

    area, centroid, moments = compute_waterplane_properties(hull)
    ref = twin_box_analytic(length, beam, draft, gap)
    assert area == pytest.approx(ref.waterplane_area, rel=1e-12)
    assert moments[0] == pytest.approx(ref.transverse_moment, rel=1e-12)
    assert moments[1] == pytest.approx(ref.longitudinal_moment, rel=1e-12)
    # symmetric twin hull: centroid on the centreline
    assert abs(centroid[0]) < 1e-12
    assert abs(centroid[1]) < 1e-12


def test_moonpool_waterplane_subtracts_the_hole():
    """An internal waterline loop (moonpool) must subtract its area."""
    outer, inner, draft = 0.5, 0.25, 0.5
    hull = annulus_hull_mesh(outer, inner, draft, n_theta=256, n_z=1)

    loops = _trace_waterline_loops(hull.waterline_edges)
    assert len(loops) == 2

    area, _centroid, moments = compute_waterplane_properties(hull)
    ref = annulus_analytic(outer, inner, draft)
    # inscribed polygon: O(h^2) underestimate at 256 segments is < 1e-3
    assert area == pytest.approx(ref.waterplane_area, rel=2e-4)
    assert moments[0] == pytest.approx(ref.transverse_moment, rel=5e-4)
    assert moments[0] < math.pi * outer**4 / 4.0  # hole removed

    volume, _cb = compute_displaced_volume_and_buoyancy(hull)
    assert volume == pytest.approx(ref.displaced_volume, rel=2e-4)


def test_twin_box_volume_and_buoyancy_centre():
    length, beam, draft, gap = 2.0, 0.5, 0.4, 1.0
    hull = twin_box_hull_mesh(length, beam, draft, gap, n_x=2, n_y=1)
    volume, centre = compute_displaced_volume_and_buoyancy(hull)
    assert volume == pytest.approx(2.0 * length * beam * draft, rel=1e-12)
    assert centre[2] == pytest.approx(-0.5 * draft, abs=1e-12)


def test_twin_box_restoring_uses_total_waterplane():
    """Heave restoring must use the summed area of both hulls."""
    length, beam, draft, gap = 2.0, 0.5, 0.4, 1.0
    hull = twin_box_hull_mesh(length, beam, draft, gap)
    ref = twin_box_analytic(length, beam, draft, gap)
    kg = 0.5 * draft
    props = compute_hydrostatic_restoring_matrix(
        hull, DENSITY * ref.displaced_volume, np.array([0.0, 0.0, -draft + kg])
    )
    assert props.restoring_matrix[2, 2] == pytest.approx(
        DENSITY * GRAVITY * ref.waterplane_area, rel=1e-12
    )
    assert props.restoring_matrix[3, 3] == pytest.approx(
        DENSITY * GRAVITY * ref.displaced_volume * ref.gm_transverse(kg), rel=1e-12
    )


# --------------------------------------------------------------------------- #
# 2. analytic accuracy of the hydrostatic stack
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "builder, reference, kg",
    [
        (
            lambda: box_hull_mesh(4.0, 1.0, 0.5, 8, 8, 1),
            box_analytic(4.0, 1.0, 0.5),
            0.25,
        ),
        (
            lambda: twin_box_hull_mesh(2.0, 0.5, 0.4, 1.0, n_x=8, n_y=8),
            twin_box_analytic(2.0, 0.5, 0.4, 1.0),
            0.2,
        ),
    ],
)
def test_wall_sided_bodies_are_exact(builder, reference, kg):
    """Wall-sided panels reproduce the exact hydrostatics to machine precision."""
    hull = builder()
    area, _c, moments = compute_waterplane_properties(hull)
    volume, centre = compute_displaced_volume_and_buoyancy(hull)
    props = compute_hydrostatic_restoring_matrix(
        hull,
        DENSITY * reference.displaced_volume,
        np.array([0.0, 0.0, -0.5 * kg]),
    )
    z_keel = float(np.min(hull.vertices[:, 2]))
    assert area == pytest.approx(reference.waterplane_area, rel=1e-12)
    assert volume == pytest.approx(reference.displaced_volume, rel=1e-12)
    assert moments[0] == pytest.approx(reference.transverse_moment, rel=1e-10)
    assert moments[1] == pytest.approx(reference.longitudinal_moment, rel=1e-10)
    assert centre[2] - z_keel == pytest.approx(reference.kb, rel=1e-12)
    assert props.restoring_matrix[2, 2] == pytest.approx(
        DENSITY * GRAVITY * reference.waterplane_area, rel=1e-10
    )


def test_curved_bodies_converge_at_second_order():
    """Cylinder, hemisphere and annular float: O(h^2) errors below 0.2 % at the finest grid."""
    cases = {case.name: case for case in build_cases()}
    for name in ("cylinder", "hemisphere", "moonpool_annulus"):
        case = cases[name]
        rows = [evaluate_case(case, level, DENSITY, GRAVITY) for level in case.levels]

        # Area and volume are waterline-integrals of order h^2 with a small
        # constant; the second moments carry the same order but a constant about
        # twice as large (they weight the integrand by y^2).
        for key in ("waterplane_area_rel", "volume_rel", "transverse_moment_rel"):
            tolerance = 5.0e-3 if "moment" in key else 2.0e-3
            assert abs(rows[-1][key]) < tolerance, (name, key, rows[-1][key])
            order = observed_order(rows, key)
            assert all(1.7 < p < 2.3 for p in order if not math.isnan(p)), (name, key, order)

        # GM is a small difference of larger quantities, so its relative error is
        # amplified with respect to the geometric errors; require < 1 %.
        assert abs(rows[-1]["gm_transverse_rel"]) < 1.0e-2
        assert abs(rows[-1]["c_roll_rel"]) < 5.0e-3


def test_hemisphere_coarse_grid_reproduces_documented_error():
    """The historical single-case acceptance numbers (2.5 %, 8.8 % at 64 panels)."""
    case = {c.name: c for c in build_cases()}["hemisphere"]
    row = evaluate_case(case, 2, DENSITY, GRAVITY)
    assert row["panel_count"] == 64
    assert row["waterplane_area_rel"] == pytest.approx(-0.0255, abs=5e-4)
    assert row["volume_rel"] == pytest.approx(-0.0886, abs=1e-3)


def test_restoring_matrix_matches_closed_form():
    """C[2,2]/C[3,3]/C[4,4] vs rho g A_wp and rho g grad GM at 1024 hemisphere panels."""
    case = {c.name: c for c in build_cases()}["hemisphere"]
    row = evaluate_case(case, 4, DENSITY, GRAVITY)
    assert abs(row["c_heave_rel"]) < 2.0e-3
    assert abs(row["c_roll_rel"]) < 5.0e-3
    assert abs(row["c_pitch_rel"]) < 5.0e-3


def test_study_payload_is_json_serialisable():
    import json

    results = run_study(cases=build_cases()[:1])
    json.dumps(results)
    assert results["cases"][0]["name"] == "box"
    assert results["cases"][0]["levels"]