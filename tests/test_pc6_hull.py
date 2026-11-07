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