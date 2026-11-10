"""Gates for near-boundary near-singular quadrature and the heave radiation chain.

Two certified statements are pinned here (mission IN50-W4-N02):

**Quadrature.**  On a planar panel the single-layer ``int G dS`` and
double-layer ``int dG/dn dS`` kernels are integrated by two independent routes
-- adaptive midpoint subdivision and graded polar coordinates with the
``rho = eps*sinh(t)`` substitution -- and compared against a semi-analytic
arbiter whose radial integral is closed form.  The tests certify that

* both routes are ``eps``-uniform: their relative error does not degrade as the
  target approaches the panel (``eps/h`` from ``1`` down to ``1e-3``);
* both converge with the work invested (``e(N)`` slope is negative);
* the untransformed Duffy rule of a production constant-panel code is a genuine
  negative control: it *fails* in the same limit, so a green result above is
  not the gate being agreeable.

**Radiation.**  The Hulme (1982) heave potential of a floating hemisphere is
re-implemented on the wetted surface and pinned to
:func:`tensorbem.analytic.hulme_hemisphere_radiation` (independent code path,
same series), its far-field Kochin integral is cross-checked against the
near-field pressure integral (energy identity, residual ``< 1e-6``), and the
constant-panel BEM solve is anchored to Hulme over ``Ka = 0.1 .. 2``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.benchmarks import nearfield_quadrature as bench
from tensorbem.free_surface_solver3d import (
    far_field_damping_matrix_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh
from tensorbem.near_field_quadrature import (
    adaptive_triangle_quadrature,
    energy_identity_residual,
    gauss_legendre,
    heave_kochin_function,
    heave_near_field_damping,
    hulme_heave_coefficients,
    plain_duffy_quadrature,
    polar_triangle_quadrature,
    semi_analytic_reference,
)

V0 = np.array([0.0, 0.0, 0.0])
V1 = np.array([1.0, 0.0, 0.0])
V2 = np.array([0.0, 1.0, 0.0])
SIZE = math.sqrt(2.0)
FOOTPRINTS = {
    "centroid": (1.0 / 3.0, 1.0 / 3.0),
    "interior": (0.5, 0.25),
    "near_vertex": (0.1, 0.1),
}
CLEARANCES = (1.0, 1.0e-1, 1.0e-2, 1.0e-3)
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "nearfield_quadrature"


def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


def _target(footprint: str, ratio: float) -> np.ndarray:
    fx, fy = FOOTPRINTS[footprint]
    return np.array([fx, fy, ratio * SIZE])


def _worst_error(values: dict, reference: dict) -> float:
    return max(_rel(values["single"], reference["single"]), _rel(values["double"], reference["double"]))


# ---------------------------------------------------------------------------
# quadrature primitives
# ---------------------------------------------------------------------------
def test_gauss_legendre_matches_numpy():
    from numpy.polynomial.legendre import leggauss

    for nodes in (5, 17, 64):
        x_ref, w_ref = leggauss(nodes)
        x, w = gauss_legendre(nodes)
        assert np.allclose(x, x_ref, atol=1e-14)
        assert np.allclose(w, w_ref, rtol=1e-13)
    # exactness on the highest representable polynomial degree
    x, w = gauss_legendre(48)
    for degree in range(0, 96):
        value = float(np.sum(w * x**degree))
        expected = 2.0 / (degree + 1) if degree % 2 == 0 else 0.0
        assert value == pytest.approx(expected, abs=1e-13)


def test_semi_analytic_reference_agrees_with_adaptive_route_far_field():
    """Two independent methods agree in the well-resolved limit (eps/h = 1)."""
    for footprint in FOOTPRINTS:
        target = _target(footprint, 1.0)
        reference = semi_analytic_reference(V0, V1, V2, target, angular_order=160)
        adaptive = adaptive_triangle_quadrature(V0, V1, V2, target, order=8, max_level=16)
        # In this regime the panel is accepted without refinement, so both routes
        # reduce to the order-8 Duffy rule; the agreement level is the rule's own
        # accuracy, ~3e-12 (single) and ~5e-11 (double).
        assert _rel(adaptive["single"], reference["single"]) < 1e-10
        assert _rel(adaptive["double"], reference["double"]) < 1e-9


# ---------------------------------------------------------------------------
# claim 1: eps-uniformity and convergence of the two routes
# ---------------------------------------------------------------------------
def test_adaptive_route_is_clearance_uniform():
    for footprint in FOOTPRINTS:
        for ratio in CLEARANCES:
            target = _target(footprint, ratio)
            reference = semi_analytic_reference(V0, V1, V2, target, angular_order=160)
            adaptive = adaptive_triangle_quadrature(
                V0, V1, V2, target, order=8, split_ratio=1.0, max_level=16
            )
            error = _worst_error(adaptive, reference)
            assert error < 1.0e-9, (footprint, ratio, error)


def test_polar_route_is_clearance_uniform():
    for footprint in FOOTPRINTS:
        for ratio in CLEARANCES:
            target = _target(footprint, ratio)
            reference = semi_analytic_reference(V0, V1, V2, target, angular_order=160)
            polar = polar_triangle_quadrature(
                V0, V1, V2, target, radial_order=32, angular_order=32
            )
            error = _worst_error(polar, reference)
            assert error < 1.0e-9, (footprint, ratio, error)


def test_error_law_is_flat_in_clearance():
    """e(eps) does not degrade: error at eps/h = 1e-3 is not >10x the eps/h = 1 value."""
    target_far = _target("centroid", 1.0)
    target_near = _target("centroid", 1.0e-3)
    ref_far = semi_analytic_reference(V0, V1, V2, target_far, angular_order=160)
    ref_near = semi_analytic_reference(V0, V1, V2, target_near, angular_order=160)
    for route in ("adaptive", "polar"):
        if route == "adaptive":
            far = adaptive_triangle_quadrature(V0, V1, V2, target_far, order=8, max_level=16)
            near = adaptive_triangle_quadrature(V0, V1, V2, target_near, order=8, max_level=16)
        else:
            far = polar_triangle_quadrature(V0, V1, V2, target_far, radial_order=12, angular_order=12)
            near = polar_triangle_quadrature(V0, V1, V2, target_near, radial_order=12, angular_order=12)
        ratio = _worst_error(near, ref_near) / max(_worst_error(far, ref_far), 1e-16)
        assert ratio < 10.0, (route, ratio)


def test_adaptive_route_converges_with_refinement_level():
    target = _target("centroid", 1.0e-3)
    reference = semi_analytic_reference(V0, V1, V2, target, angular_order=160)
    coarse = adaptive_triangle_quadrature(V0, V1, V2, target, order=8, max_level=2)
    fine = adaptive_triangle_quadrature(V0, V1, V2, target, order=8, max_level=12)
    assert _worst_error(coarse, reference) > 1.0e-3
    assert _worst_error(fine, reference) < 1.0e-10
    # error must fall with the work invested (negative log-log slope)
    slope = math.log(_worst_error(fine, reference) / _worst_error(coarse, reference)) / math.log(
        fine["nodes"] / coarse["nodes"]
    )
    assert slope < -1.0


def test_polar_route_converges_with_quadrature_order():
    target = _target("centroid", 1.0e-3)
    reference = semi_analytic_reference(V0, V1, V2, target, angular_order=160)
    coarse = polar_triangle_quadrature(V0, V1, V2, target, radial_order=4, angular_order=4)
    fine = polar_triangle_quadrature(V0, V1, V2, target, radial_order=24, angular_order=24)
    assert _worst_error(coarse, reference) > 1.0e-5
    assert _worst_error(fine, reference) < 1.0e-10
    slope = math.log(_worst_error(fine, reference) / _worst_error(coarse, reference)) / math.log(
        fine["nodes"] / coarse["nodes"]
    )
    assert slope < -1.0


def test_negative_control_plain_duffy_is_not_clearance_uniform():
    """The untransformed Duffy rule must fail: that is what makes the gates above meaningful."""
    far = _target("centroid", 1.0)
    near = _target("centroid", 1.0e-3)
    ref_far = semi_analytic_reference(V0, V1, V2, far, angular_order=160)
    ref_near = semi_analytic_reference(V0, V1, V2, near, angular_order=160)
    error_far = _worst_error(plain_duffy_quadrature(V0, V1, V2, far, order=8), ref_far)
    error_near = _worst_error(plain_duffy_quadrature(V0, V1, V2, near, order=8), ref_near)
    assert error_far < 1.0e-6
    assert error_near > 5.0e-2
    assert error_near / error_far > 100.0
    # raising the order does not rescue it: the failure is in the clearance, not in the order
    refined = _worst_error(plain_duffy_quadrature(V0, V1, V2, near, order=16), ref_near)
    assert refined > 1.0e-3


# ---------------------------------------------------------------------------
# claim 2: hemisphere heave radiation vs Hulme (1982)
# ---------------------------------------------------------------------------
def test_hulme_potential_reimplementation_matches_analytic_reference():
    for ka in (0.1, 1.0, 2.0):
        reference = hulme_hemisphere_radiation(ka, n_terms=96, quad_order=192)
        coefficients = hulme_heave_coefficients(ka, 96)
        near = heave_near_field_damping(ka, coefficients, mu_nodes=600)
        assert near["added_mass_dimless"] == pytest.approx(
            reference.heave_added_mass_dimless, rel=1e-11
        )
        assert near["damping_dimless"] == pytest.approx(reference.heave_damping_dimless, rel=1e-11)


def test_kochin_function_is_axisymmetric_and_carries_source_amplitude():
    ka = 1.0
    coefficients = hulme_heave_coefficients(ka, 256)
    azimuth = 2.0 * math.pi * np.arange(12) / 12.0
    kochin = heave_kochin_function(ka, coefficients, azimuth)
    spread = float(np.std(np.abs(kochin)) / np.mean(np.abs(kochin)))
    assert spread < 1.0e-10
    # for an axisymmetric heave field the Kochin amplitude is 4*pi*|c0| (source term)
    amplitude = 4.0 * math.pi * abs(coefficients[0])
    assert float(np.mean(np.abs(kochin))) == pytest.approx(amplitude, rel=1.0e-4)


def test_energy_identity_near_field_vs_far_field():
    """Near-field pressure integral and far-field Kochin integral of the same field."""
    for ka in (0.1, 1.0, 2.0):
        row = energy_identity_residual(ka, n_terms=1024)
        assert row["residual"] < 1.0e-6, (ka, row["residual"])
        assert row["kochin_relative_spread"] < 1.0e-10


def test_energy_identity_residual_shrinks_with_series_terms():
    coarse = energy_identity_residual(1.0, n_terms=64)["residual"]
    fine = energy_identity_residual(1.0, n_terms=512)["residual"]
    assert coarse > fine
    assert coarse / fine > 10.0


def test_hemisphere_heave_matches_hulme_256_panels():
    ka = 1.0
    mesh = hemisphere_hull_mesh(3, radius=1.0)
    radiation = solve_radiation_3d(mesh, ka, modes=[2], density=1025.0, gravity=9.80665)
    reference = hulme_hemisphere_radiation(ka, n_terms=96, quad_order=192)
    volume_ratio = 2.0 * math.pi / 3.0
    a33 = float(radiation.added_mass[0, 0]) / 1025.0
    b33 = float(radiation.damping[0, 0]) / (1025.0 * radiation.omega)
    assert abs(a33 / (reference.heave_added_mass_dimless * volume_ratio) - 1.0) < 0.03
    assert abs(b33 / (reference.heave_damping_dimless * volume_ratio) - 1.0) < 0.03
    # near-field and far-field readings of the same BEM solution must be close
    far = far_field_damping_matrix_3d(mesh, radiation, n_azimuth=72)[0, 0]
    assert abs(far - float(radiation.damping[0, 0])) / abs(float(radiation.damping[0, 0])) < 1.0e-3


# ---------------------------------------------------------------------------
# published artifacts
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not ARTIFACTS.exists(), reason="artifacts not generated in this checkout")
def test_published_gates_all_pass():
    payload = json.loads((ARTIFACTS / "gates_verified.json").read_text(encoding="utf-8"))
    failing = [gate["id"] for gate in payload["gates"] if not gate["pass"]]
    assert not failing, failing
    assert payload["all_gates_pass"] is True
    assert payload["hemisphere"]["ka_range"] == [pytest.approx(0.1), pytest.approx(2.0)]
    assert payload["energy_identity"]["worst_residual"] < 1.0e-6
    assert payload["quadrature"]["negative_control"]["error_at_clearance_1e-3"] > 0.05


@pytest.mark.skipif(not (ARTIFACTS / "a33_b33.csv").exists(), reason="artifacts not generated")
def test_published_a33_b33_csv_covers_the_sweep():
    with (ARTIFACTS / "a33_b33.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    ka_values = sorted({float(row["ka"]) for row in rows})
    assert ka_values[0] == pytest.approx(0.1)
    assert ka_values[-1] == pytest.approx(2.0)
    for row in rows:
        assert float(row["a33_rel_error"]) == pytest.approx(
            float(row["a33_over_rhoR3_bem"]) / float(row["a33_over_rhoR3_hulme1982"]) - 1.0, abs=1e-12
        )
        assert float(row["b33_over_rhowR3_bem"]) > 0.0
        assert float(row["b33_over_rhowR3_hulme1982"]) > 0.0

def test_published_gates_use_the_module_thresholds():
    """The artifact gates are produced by the driver; their thresholds must not drift."""
    assert bench._GATE_ENERGY_RESIDUAL == 1.0e-6
    assert bench._GATE_HULME_ERROR == 3.0e-2
    assert bench._GATE_ADAPTIVE_ERROR == 1.0e-9
    assert bench._GATE_NEGATIVE_CONTROL == 1.0e-1
    panel = bench._unit_panel()
    assert bench._panel_size(*panel) == pytest.approx(math.sqrt(2.0))
    payload = None
    if (ARTIFACTS / "gates_verified.json").exists():
        payload = json.loads((ARTIFACTS / "gates_verified.json").read_text(encoding="utf-8"))
    if payload is not None:
        thresholds = {gate["id"]: gate["threshold"] for gate in payload["gates"]}
        assert thresholds["E1-near-far-identity"] == bench._GATE_ENERGY_RESIDUAL
        assert thresholds["H1-heave-added-mass"] == bench._GATE_HULME_ERROR
        assert thresholds["Q4-negative-control"] == bench._GATE_NEGATIVE_CONTROL
