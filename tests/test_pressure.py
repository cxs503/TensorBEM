"""Verification tests for :mod:`tensorbem.pressure`.

Every reference is an independent analytic identity, never a value produced by
the implementation under test:

* a uniform pressure on a closed surface produces neither force nor moment;
* a linear pressure field ``p = c . x`` integrates (divergence theorem) to
  ``F_out = c * V`` on a closed surface, so the signed mesh volume is the exact
  reference for the resultant;
* a hydrostatic field ``p = rho*g*(z_fs - z)`` gives Archimedes
  ``F_body = -rho*g*V*z_hat`` and zero moment about the centre of a sphere;
* the Bernoulli pressure coefficient of a sphere in uniform flow has the
  closed-form stagnation value ``1`` and equator value ``-5/4``.
"""

import json
import math

import numpy as np
import pytest

from tensorbem.geometry3d import ellipsoid_mesh, icosphere
from tensorbem.pressure import (
    PressureLoads,
    bernoulli_pressure,
    hydrostatic_pressure,
    integrate_pressure_loads,
    steady_pressure_coefficient,
    steady_pressure_loads,
)


@pytest.fixture(scope="module")
def sphere():
    return icosphere(2, radius=1.0)


@pytest.fixture(scope="module")
def ellipsoid():
    return ellipsoid_mesh(2, np.array([1.0, 0.7, 0.45]))


def test_uniform_pressure_has_no_resultant(sphere, ellipsoid):
    for mesh in (sphere, ellipsoid):
        loads = integrate_pressure_loads(mesh, np.full(mesh.panel_count, 7.5))
        scale = 7.5 * mesh.surface_area
        assert np.linalg.norm(loads.force) < 1.0e-12 * scale
        assert np.linalg.norm(loads.moment) < 1.0e-12 * scale * mesh.h_max
        assert loads.wetted_area == pytest.approx(mesh.surface_area)


def test_reference_pressure_translation_changes_nothing(sphere):
    pressure = 1000.0 + 250.0 * np.cos(sphere.centroids[:, 0])
    plain = integrate_pressure_loads(sphere, pressure)
    shifted = integrate_pressure_loads(sphere, pressure, reference_pressure=1000.0)
    assert np.allclose(plain.force, shifted.force, atol=1.0e-9)
    assert np.allclose(plain.moment, shifted.moment, atol=1.0e-9)
    assert plain.reference_pressure == 0.0
    assert shifted.reference_pressure == 1000.0


@pytest.mark.parametrize("mesh_name", ["sphere", "ellipsoid"])
def test_linear_pressure_matches_divergence_theorem(mesh_name, sphere, ellipsoid):
    mesh = sphere if mesh_name == "sphere" else ellipsoid
    gradient = np.array([3.0, -2.0, 5.0])
    pressure = mesh.centroids @ gradient
    loads = integrate_pressure_loads(mesh, pressure)
    expected = gradient * mesh.volume
    assert np.allclose(loads.force, expected, rtol=1.0e-13, atol=1.0e-13)
    # The moment of a linear pressure field about the origin is analytic too:
    # int (x x (c.x) n) dS = int (x x (c.x) n) dS = -int (u x) ... is not
    # needed here; only the force identity is asserted.


def test_hydrostatic_pressure_reproduces_archimedes(sphere, ellipsoid):
    density, gravity = 1025.0, 9.80665
    for mesh in (sphere, ellipsoid):
        pressure = hydrostatic_pressure(
            mesh.centroids, density=density, gravity=gravity, free_surface_z=3.0
        )
        loads = integrate_pressure_loads(mesh, pressure, density=density)
        buoyancy = loads.force_on_body
        expected = density * gravity * mesh.volume
        assert buoyancy[2] == pytest.approx(expected, rel=1.0e-13)
        assert abs(buoyancy[0]) < 1.0e-10 * expected
        assert abs(buoyancy[1]) < 1.0e-10 * expected


def test_hydrostatic_moment_is_zero_about_the_sphere_centre(sphere):
    pressure = hydrostatic_pressure(sphere.centroids, free_surface_z=2.0)
    loads = integrate_pressure_loads(sphere, pressure)
    scale = np.max(np.abs(pressure)) * sphere.surface_area * sphere.h_max
    assert np.linalg.norm(loads.moment) < 1.0e-12 * scale


def test_moment_about_a_shifted_reference_is_the_cross_product_identity(sphere):
    pressure = sphere.centroids @ np.array([1.0, 2.0, -3.0])
    origin = integrate_pressure_loads(sphere, pressure)
    center = np.array([0.3, -0.2, 0.5])
    shifted = integrate_pressure_loads(sphere, pressure, reference_center=center)
    expected = origin.moment - np.cross(center, origin.force)
    assert np.allclose(shifted.moment, expected, atol=1.0e-12)
    assert np.allclose(shifted.reference_center, center)
    # And the force is independent of the reference centre.
    assert np.allclose(shifted.force, origin.force)


def test_pressure_load_coefficients_are_normalised_by_dynamic_pressure(sphere):
    density, speed = 1000.0, 2.0
    pressure = np.full(sphere.panel_count, 0.5 * density * speed**2)
    loads = integrate_pressure_loads(sphere, pressure, density=density)
    # A uniform pressure has no resultant, so test the normalisation on a
    # hydrostatic case with a known closed-form resultant instead.
    hydrostatic = integrate_pressure_loads(
        sphere, hydrostatic_pressure(sphere.centroids, density=density),
        density=density,
    )
    area = math.pi * 1.0**2
    coefficients = hydrostatic.coefficients(
        density=density, speed=speed, area=area, length=1.0
    )
    expected = density * 9.80665 * sphere.volume / (0.5 * density * speed**2 * area)
    assert coefficients["cz"] == pytest.approx(expected, rel=1.0e-13)
    assert coefficients["magnitude"] == pytest.approx(expected, rel=1.0e-13)
    assert coefficients["cx"] == pytest.approx(0.0, abs=1.0e-10 * expected)
    assert np.allclose(loads.force, 0.0)


def test_steady_pressure_coefficient_matches_sphere_extrema():
    # On a sphere in a uniform stream of speed U the total surface speed is
    # (3/2) U sin(theta): zero at the stagnation points and (3/2) U on the
    # equator, so Cp = 1 - |v|**2/U**2 is 1 and -5/4 there.
    speed = 1.7
    stagnation = np.zeros((2, 3))
    equator = np.array([[0.0, 1.5 * speed, 0.0]])
    assert np.allclose(steady_pressure_coefficient(stagnation, speed), 1.0)
    assert steady_pressure_coefficient(equator, speed)[0] == pytest.approx(-1.25)


def test_bernoulli_pressure_steady_and_unsteady():
    density, speed = 1000.0, 3.0
    velocity = np.array([[speed, 0.0, 0.0], [0.0, 1.5 * speed, 0.0]])
    steady = bernoulli_pressure(velocity, density=density, free_stream_speed=speed)
    assert steady[0] == pytest.approx(0.0, abs=1.0e-12)
    assert steady[1] == pytest.approx(-0.5 * density * 1.25 * speed**2)
    unsteady = bernoulli_pressure(
        velocity,
        density=density,
        free_stream_speed=speed,
        free_stream_pressure=101325.0,
        potential_rate=np.array([1.0, -0.5]),
    )
    assert unsteady[0] == pytest.approx(101325.0 - density, rel=1.0e-15)
    assert unsteady[1] == pytest.approx(
        101325.0 - 0.5 * density * 1.25 * speed**2 + 0.5 * density
    )


def test_steady_pressure_loads_returns_the_cp_field(sphere):
    # A uniform stream with zero disturbance velocity has Cp = 0 everywhere,
    # so the gauge pressure vanishes identically and so does the resultant.
    speed = 4.0
    velocity = np.tile(np.array([speed, 0.0, 0.0]), (sphere.panel_count, 1))
    cp, loads = steady_pressure_loads(
        sphere, velocity, (speed, 0.0, 0.0), density=1025.0
    )
    assert np.allclose(cp, 0.0)
    assert np.linalg.norm(loads.force) < 1.0e-10 * 0.5 * 1025.0 * speed**2 * sphere.surface_area


def test_pressure_loads_to_dict_is_json_serialisable(sphere):
    loads = integrate_pressure_loads(
        sphere, hydrostatic_pressure(sphere.centroids), reference_center=(0.1, 0.0, 0.0)
    )
    assert isinstance(loads, PressureLoads)
    payload = json.loads(json.dumps(loads.to_dict()))
    assert set(payload) == {
        "force",
        "moment",
        "force_on_body",
        "moment_on_body",
        "reference_center",
        "wetted_area",
        "reference_pressure",
        "density",
    }
    assert payload["force"] == [-value for value in payload["force_on_body"]]


def test_integrate_pressure_loads_rejects_bad_input(sphere):
    with pytest.raises(TypeError):
        integrate_pressure_loads("mesh", np.zeros(sphere.panel_count))
    with pytest.raises(ValueError):
        integrate_pressure_loads(sphere, np.zeros(sphere.panel_count + 1))
    with pytest.raises(ValueError):
        bad = np.zeros(sphere.panel_count)
        bad[0] = np.nan
        integrate_pressure_loads(sphere, bad)
    with pytest.raises(ValueError):
        integrate_pressure_loads(sphere, np.zeros(sphere.panel_count), reference_center=(0.0, 0.0))
    with pytest.raises(ValueError):
        integrate_pressure_loads(sphere, np.zeros(sphere.panel_count), density=-1.0)
    with pytest.raises(ValueError):
        integrate_pressure_loads(
            sphere, np.zeros(sphere.panel_count), reference_pressure=float("inf")
        )


def test_field_helpers_reject_bad_input():
    with pytest.raises(ValueError):
        steady_pressure_coefficient(np.zeros((2, 2)), 1.0)
    with pytest.raises(ValueError):
        steady_pressure_coefficient(np.zeros((2, 3)), 0.0)
    with pytest.raises(TypeError):
        steady_pressure_coefficient(np.zeros((2, 3)), True)
    with pytest.raises(ValueError):
        bernoulli_pressure(np.zeros((3, 3)), potential_rate=np.zeros(2))
    with pytest.raises(ValueError):
        bernoulli_pressure(np.zeros((3, 3)), density=0.0)
    with pytest.raises(ValueError):
        hydrostatic_pressure(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        hydrostatic_pressure(np.zeros((3, 3)), gravity=float("nan"))


def test_steady_pressure_loads_rejects_bad_input(sphere):
    with pytest.raises(TypeError):
        steady_pressure_loads("mesh", np.zeros((sphere.panel_count, 3)), (1.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        steady_pressure_loads(sphere, np.zeros((sphere.panel_count, 2)), (1.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        steady_pressure_loads(sphere, np.zeros((sphere.panel_count, 3)), (1.0, 0.0))
    with pytest.raises(ValueError):
        steady_pressure_loads(sphere, np.zeros((sphere.panel_count, 3)), (0.0, 0.0, 0.0))