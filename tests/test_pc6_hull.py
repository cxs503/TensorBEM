"""Geometry gate for the parametric PC6-like hull.

The platform rule is that a hull geometry must be validated (``Cb``, ``L``,
``B``, ``T``) *before* any hydrodynamics is computed.  These tests pin that
gate: the mesh must agree with an independent quadrature of the same
parametric definition, and a deliberately under-resolved mesh or a
non-physical hull definition must be rejected.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.pc6_hull import (
    IceClassHull,
    hull_mass_properties,
    ice_class_hull_mesh,
    validate_hull_geometry,
)

HULL = IceClassHull(length=120.0, breadth=24.0, draft=9.0)


def test_mesh_reproduces_parametric_block_coefficient():
    """``Cb`` from the mesh must match the parametric quadrature to <1 %."""
    mesh = ice_class_hull_mesh(HULL, n_sections=21, n_side=5, n_bottom=3)
    check = validate_hull_geometry(HULL, mesh, cb_tolerance=0.01)
    assert check["block_coefficient"] == pytest.approx(
        check["block_coefficient_reference"], rel=0.01
    )
    assert 0.5 < check["block_coefficient"] < 0.85  # full-form ice-class hull
    assert check["length"] == pytest.approx(120.0, rel=1.0e-9)
    assert check["breadth"] == pytest.approx(24.0, rel=1.0e-9)
    assert check["draft"] == pytest.approx(9.0, rel=1.0e-9)
    assert check["waterplane_area_relative_error"] < 0.01


def test_waterplane_area_is_consistent_with_hydrostatics():
    """The waterline-derived waterplane area must match the hydrostatic module."""
    from tensorbem.hydrostatic import compute_hydrostatic_restoring_matrix

    mesh = ice_class_hull_mesh(HULL, n_sections=21, n_side=5, n_bottom=3)
    check = validate_hull_geometry(HULL, mesh)
    props = compute_hydrostatic_restoring_matrix(
        mesh, 1.0e6, np.array([0.0, 0.0, -4.5])
    )
    assert props.waterplane_area == pytest.approx(check["waterplane_area"], rel=1.0e-6)
    assert props.displaced_volume == pytest.approx(check["volume"], rel=1.0e-6)


def test_under_resolved_mesh_fails_the_geometry_gate():
    """A coarse mesh breaks the 1 % ``Cb`` band and must be rejected, not used."""
    coarse = ice_class_hull_mesh(HULL, n_sections=9, n_side=3, n_bottom=2)
    with pytest.raises(ValueError, match="block coefficient"):
        validate_hull_geometry(HULL, coarse, cb_tolerance=0.01)


def test_negative_and_zero_dimensions_rejected():
    with pytest.raises(ValueError, match="draft"):
        IceClassHull(draft=-9.0)
    with pytest.raises(ValueError, match="draft"):
        IceClassHull(draft=0.0)
    with pytest.raises(ValueError, match="breadth"):
        IceClassHull(breadth=-24.0)
    with pytest.raises(ValueError, match="length"):
        IceClassHull(length=0.0)
    with pytest.raises(ValueError, match="keel_ratio"):
        IceClassHull(keel_ratio=1.5)


def test_hull_mass_properties_are_consistent_with_displacement():
    """``mass = rho V`` at the design draft and the inertia tensor is positive."""
    mesh = ice_class_hull_mesh(HULL, n_sections=21, n_side=5, n_bottom=3)
    check = validate_hull_geometry(HULL, mesh)
    props = hull_mass_properties(HULL, check)
    assert props["mass"] == pytest.approx(1025.0 * check["volume"], rel=1.0e-12)
    inertia = np.asarray(props["inertia_about_origin"])
    assert np.all(np.linalg.eigvals(inertia) > 0.0)
    assert inertia[1, 1] > inertia[0, 0]  # pitch inertia > roll inertia for a ship
    assert props["center_of_gravity"][2] < 0.0

# ---------------------------------------------------------------------------
# Reference-quadrature integrity (mission m14775)
#
# The ``Cb`` gate compares the mesh volume against ``reference_volume()``.  Until
# m14775 that integral summed every node with unit weight, which is not the
# trapezoid rule: it overstates the reference volume, and the residual it leaves
# in ``validate_hull_geometry`` never vanishes however fine the mesh becomes.
# The tests below pin the reference to the analytic integral of the parametric
# definition instead of pinning the quadrature to itself.
# ---------------------------------------------------------------------------
_PARAMETER_SETS = {
    "default": {},
    "kcs-like": {"waterline_exponent": 2.0, "keel_ratio": 0.9295},
    "blunt": {
        "waterline_exponent": 1.2,
        "section_exponent": 1.0,
        "keel_ratio": 0.4,
        "end_ratio": 0.3,
    },
    "extreme": {
        "waterline_exponent": 40.0,
        "section_exponent": 6.0,
        "keel_ratio": 0.05,
        "end_ratio": 0.45,
    },
}


def _unweighted_node_sum_volume(hull, *, panels: int = 400) -> float:
    """The pre-m14775 quadrature: every node carries unit weight."""
    n = int(panels)
    u = np.linspace(-1.0, 1.0, n + 1)
    v = np.linspace(-1.0, 0.0, n + 1)
    wl = np.maximum(1.0 - np.abs(u) ** hull.waterline_exponent, hull.end_ratio)
    f = np.maximum(
        1.0 - (1.0 - hull.keel_ratio) * np.abs(v) ** hull.section_exponent, hull.keel_ratio
    )
    integrand = hull.breadth * np.outer(wl, f)
    dx = 0.5 * hull.length * (u[1] - u[0])
    dz = hull.draft * (v[1] - v[0])
    return float(np.sum(integrand) * dx * dz)


@pytest.mark.parametrize("name", sorted(_PARAMETER_SETS))
def test_reference_volume_matches_the_closed_form(name):
    """The reference integral must reproduce the analytic volume, not itself."""
    hull = IceClassHull(**_PARAMETER_SETS[name])
    exact = hull.closed_form_volume()
    quadrature = hull.reference_volume(panels=400)
    assert quadrature == pytest.approx(exact, rel=1.0e-4)


@pytest.mark.parametrize("name", sorted(_PARAMETER_SETS))
def test_reference_waterplane_area_matches_the_closed_form(name):
    hull = IceClassHull(**_PARAMETER_SETS[name])
    exact = hull.closed_form_waterplane_area()
    quadrature = hull.reference_waterplane_area(panels=400)
    assert quadrature == pytest.approx(exact, rel=1.0e-4)


@pytest.mark.parametrize("name", sorted(_PARAMETER_SETS))
def test_closed_form_integrals_match_dense_numerical_integration(name):
    """The closed forms must agree with a 10^-5-step trapezoid, i.e. be the integral."""
    hull = IceClassHull(**_PARAMETER_SETS[name])
    u = np.linspace(-1.0, 1.0, 200001)
    wl = np.maximum(1.0 - np.abs(u) ** hull.waterline_exponent, hull.end_ratio)
    assert hull.closed_form_waterline_integral() == pytest.approx(
        float(np.trapezoid(wl, u)), rel=1.0e-6
    )
    v = np.linspace(0.0, 1.0, 200001)
    f = np.maximum(1.0 - (1.0 - hull.keel_ratio) * v**hull.section_exponent, hull.keel_ratio)
    assert hull.closed_form_section_integral() == pytest.approx(
        float(np.trapezoid(f, v)), rel=1.0e-6
    )
    # Cwp = I_u / 2 for this family: the waterplane is locked to the section shape
    cwp = hull.closed_form_waterplane_area() / (hull.length * hull.breadth)
    assert cwp == pytest.approx(hull.closed_form_waterline_integral() / 2.0, rel=1.0e-12)
    # Cb = I_u I_v / 2, and the closed-form volume is what both should give
    assert hull.closed_form_block_coefficient() == pytest.approx(
        0.5 * hull.closed_form_waterline_integral() * hull.closed_form_section_integral(),
        rel=1.0e-12,
    )


def test_reference_volume_is_the_tensor_product_trapezoid_rule():
    """Independent implementation: two nested ``np.trapezoid`` calls, same value."""
    hull = IceClassHull(waterline_exponent=2.0, keel_ratio=0.9295)
    panels = 64
    u = np.linspace(-1.0, 1.0, panels + 1)
    v = np.linspace(-1.0, 0.0, panels + 1)
    wl = np.maximum(1.0 - np.abs(u) ** hull.waterline_exponent, hull.end_ratio)
    f = np.maximum(
        1.0 - (1.0 - hull.keel_ratio) * np.abs(v) ** hull.section_exponent, hull.keel_ratio
    )
    integrand = hull.breadth * np.outer(wl, f)
    reference = float(
        np.trapezoid(np.trapezoid(integrand, v * hull.draft, axis=1), 0.5 * hull.length * u)
    )
    assert hull.reference_volume(panels=panels) == pytest.approx(reference, rel=1.0e-12)


def test_unit_weight_quadrature_bias_is_measurably_removed():
    """Regression ratchet: the old rule biased ``Cb`` by ~1.7e-3, the new one by <1e-5."""
    hull = IceClassHull(waterline_exponent=2.0, keel_ratio=0.9295)
    denominator = hull.length * hull.breadth * hull.draft
    exact_cb = hull.closed_form_block_coefficient()
    old_bias = abs(_unweighted_node_sum_volume(hull) / denominator - exact_cb)
    new_bias = abs(hull.reference_block_coefficient() - exact_cb)
    assert old_bias > 1.0e-3  # the defect was real and is not a rounding artefact
    assert new_bias < 1.0e-5
    assert old_bias / new_bias > 100.0


def test_geometry_gate_uses_the_analytic_reference():
    """``validate_hull_geometry`` must compare against the closed form."""
    mesh = ice_class_hull_mesh(HULL, n_sections=21, n_side=5, n_bottom=3)
    check = validate_hull_geometry(HULL, mesh)
    assert check["block_coefficient_reference"] == pytest.approx(
        HULL.closed_form_block_coefficient(), rel=1.0e-4
    )
    assert check["waterplane_area_reference"] == pytest.approx(
        HULL.closed_form_waterplane_area(), rel=1.0e-4
    )


def test_gate_residual_converges_to_zero_not_to_the_reference_bias():
    """Mesh refinement drives the gate residual to 0; the old reference left a 2.5e-3 floor."""
    hull = IceClassHull(waterline_exponent=2.0, keel_ratio=0.9295)
    exact_cb = hull.closed_form_block_coefficient()
    denominator = hull.length * hull.breadth * hull.draft
    old_cb = _unweighted_node_sum_volume(hull) / denominator
    residuals_new = []
    residuals_old = []
    for n_sections, n_side, n_bottom in ((15, 3, 2), (21, 5, 3), (31, 5, 3), (41, 9, 5)):
        mesh = ice_class_hull_mesh(
            hull, n_sections=n_sections, n_side=n_side, n_bottom=n_bottom
        )
        check = validate_hull_geometry(hull, mesh)
        cb_mesh = check["block_coefficient"]
        residuals_new.append(abs(cb_mesh - exact_cb) / exact_cb)
        residuals_old.append(abs(cb_mesh - old_cb) / old_cb)
    # corrected reference: the residual keeps shrinking (monotone, >2x per step)
    assert residuals_new[-1] < 5.0e-4
    assert residuals_new[-1] < residuals_new[0] / 10.0
    assert all(later < earlier for earlier, later in zip(residuals_new, residuals_new[1:]))
    # old reference: a floor that refinement cannot lift
    assert residuals_old[-1] > 1.0e-3
    assert residuals_old[-1] > 5.0 * residuals_new[-1]


def test_reference_waterplane_area_rejects_degenerate_grids():
    with pytest.raises(ValueError, match="panels"):
        HULL.reference_waterplane_area(panels=4)
