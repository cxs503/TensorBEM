"""Anchor regression tests for CASE-BEM-CATHODIC-SPHERE.

Ledger criterion: ``|BEM-CATHODIC-SPHERE| ref = 4 pi sigma a phi_a`` (DNV-RP-B401
isolated-anode spreading resistance ``R = 1 / (4 pi sigma a)``), tolerance < 5 %.

These tests pin the four things the anchor is worth:

1. the absolute current scale against the closed form at the finest level used
   in CI (icosphere level 2, 320 panels: 1.30 %), plus second-order convergence;
2. the non-dimensional scaling ``I / (4 pi sigma a phi) = 1`` (conductivity,
   radius and potential are not swapped or double-counted);
3. the two current routes -- surface panel integral vs far-field (Gauss-law)
   recovery -- and their agreement (a missing ``sigma`` or a ``4 pi`` defect in
   the exterior representation breaks this);
4. the benchmark CLI printing a single-level run (it used to crash on the
   ``None`` observed-order placeholder).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tensorbem.benchmarks import cathodic_protection_sphere as sphere_bench
from tensorbem.cathodic_protection import (
    SEAWATER_CONDUCTIVITY,
    far_field_current,
    fibonacci_sphere_directions,
    verify_conducting_sphere,
    verify_conducting_sphere_routes,
)

LEDGER_TOLERANCE = 0.05


@pytest.fixture(scope="module")
def sphere_levels():
    return verify_conducting_sphere(subdivisions=(1, 2))


def test_current_scale_within_ledger_tolerance(sphere_levels):
    """Finest CI level reproduces 4 pi sigma a phi_a inside the 5 % ledger band."""
    finest = sphere_levels["levels"][-1]
    exact = 4.0 * math.pi * SEAWATER_CONDUCTIVITY * 1.0 * (-1.0)
    assert finest["current_rel_error"] < LEDGER_TOLERANCE
    assert finest["max_potential_rel_error"] < LEDGER_TOLERANCE
    assert finest["current_A"] == pytest.approx(exact, rel=LEDGER_TOLERANCE)
    assert sphere_levels["exact_current_A"] == pytest.approx(exact)
    # spreading resistance of the same closed form
    assert sphere_levels["exact_spreading_resistance_ohm"] == pytest.approx(
        1.0 / (4.0 * math.pi * SEAWATER_CONDUCTIVITY * 1.0)
    )


def test_second_order_convergence(sphere_levels):
    """Nested icospheres double the panel count: p = log(e1/e2)/log(2) ~ 2."""
    assert sphere_levels["observed_order_finest_pair"] == pytest.approx(2.0, abs=0.2)


def test_unscaled_current_defect_is_pinned(sphere_levels):
    """Without the sigma factor of Ohm's law the current is low by 1/sigma."""
    assert sphere_levels["unscaled_current_error_percent"] == pytest.approx(
        100.0 * (1.0 - 1.0 / SEAWATER_CONDUCTIVITY), rel=1e-12
    )


@pytest.mark.parametrize(
    "conductivity,radius,potential",
    [(3.5, 1.0, -1.0), (2.0, 1.0, -1.0), (3.5, 0.5, -1.0)],
)
def test_nondimensional_scaling(conductivity, radius, potential):
    """``I / (4 pi sigma a phi)`` must be 1 regardless of sigma, a and phi."""
    report = verify_conducting_sphere(
        subdivisions=(2,),
        radius=radius,
        potential=potential,
        conductivity=conductivity,
    )
    exact = 4.0 * math.pi * conductivity * radius * potential
    get = report["levels"][0]["current_A"]
    assert get / exact == pytest.approx(1.0, abs=LEDGER_TOLERANCE)
    # and the potential scaling is linear (same case, doubled magnitude)
    doubled = verify_conducting_sphere(
        subdivisions=(2,),
        radius=radius,
        potential=2.0 * potential,
        conductivity=conductivity,
    )
    assert doubled["levels"][0]["current_A"] == pytest.approx(2.0 * get, rel=1e-12)


def test_two_current_routes_agree():
    """Surface panel integral and far-field monopole recover the same current."""
    routes = verify_conducting_sphere_routes(subdivisions=2, n_directions=16)
    exact = 4.0 * math.pi * SEAWATER_CONDUCTIVITY * 1.0 * (-1.0)
    assert routes["surface_rel_error"] < LEDGER_TOLERANCE
    assert routes["far_field_rel_error"] < LEDGER_TOLERANCE
    # Both routes are built on the same density field, so their agreement is a
    # check of the exterior representation formula and the Gauss-law
    # bookkeeping (not of the boundary quadrature, which the analytic anchor
    # and the convergence test above cover).
    assert routes["far_field"]["far_field_vs_surface_rel_error"] < 1e-5
    assert routes["far_field"]["extrapolated_current_A"] == pytest.approx(
        exact, rel=LEDGER_TOLERANCE
    )
    # the enclosing spheres are far enough: monopole law holds to <1e-3
    assert max(routes["far_field"]["monopole_max_rel_error_by_radius"]) < 1e-3
    assert routes["pass"] is True


def test_far_field_rejects_enclosing_radius_inside_body():
    from tensorbem.cathodic_protection import solve_cathodic_protection
    from tensorbem.geometry3d import icosphere

    mesh = icosphere(1, radius=1.0)
    panels = np.arange(mesh.panel_count, dtype=np.int64)
    empty = np.array([], dtype=np.int64)
    result = solve_cathodic_protection(
        mesh, empty, empty, panels,
        protection_potential=-1.0, anode_potential=-1.0,
        conductivity=SEAWATER_CONDUCTIVITY,
    )
    with pytest.raises(ValueError):
        far_field_current(mesh, result, radii=(0.9, 2.0, 4.0))
    with pytest.raises(ValueError):
        far_field_current(mesh, result, radii=(3.0, 2.0, 4.0))
    with pytest.raises(ValueError):
        fibonacci_sphere_directions(3)


def test_benchmark_cli_handles_single_level(tmp_path):
    """A single refinement level has no observed-order pair; it must not crash."""
    out = tmp_path / "single.json"
    exit_code = sphere_bench.main(["--subdivisions", "1", "--output", str(out)])
    assert exit_code == 0
    report = json.loads(out.read_text())
    assert report["observed_order_finest_pair"] is None
    assert report["levels"][0]["current_rel_error"] < LEDGER_TOLERANCE


def test_benchmark_cli_routes_block(tmp_path):
    """``--routes-subdivisions`` appends the two-route cross-check."""
    out = tmp_path / "routes.json"
    exit_code = sphere_bench.main([
        "--subdivisions", "1", "--routes-subdivisions", "1", "--output", str(out),
    ])
    assert exit_code == 0
    report = json.loads(out.read_text())
    assert "routes" in report
    assert report["routes"]["far_field"]["far_field_vs_surface_rel_error"] < 1e-5