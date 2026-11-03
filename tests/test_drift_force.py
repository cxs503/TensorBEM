"""Regression tests for the mean second-order wave-drift force layer.

The tests are layered the way the module is: first the far-field algebra against
the momentum balance it comes from, then the tangential-velocity recovery against
a closed-form derivative, then the near-field functional against a quadrature
reference, and finally the two physics cases -- the zero-reflection cylinder where
the exact answer is known, and a strongly reflecting ellipse where the two routes
must agree with each other.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks.dean_diffraction import (
    submerged_circle_mesh,
    submerged_ellipse_mesh,
)
from tensorbem.drift import (
    MeanDriftForce,
    mean_drift_force,
    mean_drift_force_far_field,
    mean_drift_force_near_field,
    panel_arclength,
    surface_tangential_velocity,
)
from tensorbem.free_surface_solver import solve_diffraction_indirect

DENSITY = 1025.0
GRAVITY = 9.80665
KA = 0.6
ROOT = Path(__file__).resolve().parents[1]
STUDY_JSON = ROOT / "outputs" / "drift_force_2d" / "study.json"
DOC_PATH = ROOT / "docs" / "drift_force_2d.md"


def _incident_trace_and_derivative(mesh, nu, amplitude=1.0, gravity=GRAVITY):
    """Exact incident Airy potential and its surface tangential derivative."""
    x = mesh.centers[:, 0]
    z = mesh.centers[:, 1]
    omega = math.sqrt(gravity * nu)
    potential = -(1.0j * gravity * amplitude / omega) * np.exp(nu * z + 1.0j * nu * x)
    grad_x = 1.0j * nu * potential
    grad_z = nu * potential
    normals = mesh.normals
    tangent = np.column_stack((-normals[:, 1], normals[:, 0]))
    return potential, grad_x * tangent[:, 0] + grad_z * tangent[:, 1]


def test_far_field_equals_the_momentum_balance_it_comes_from():
    """F_x = (rho*g/4)(|A|^2 + |A_R|^2 - |A_T|^2) collapses to (rho*g/2)|A_R|^2."""
    amplitude = 0.8
    reflection = 0.35 * np.exp(0.4j)
    transmitted_modulus = math.sqrt(1.0 - abs(reflection) ** 2)
    momentum = 0.25 * DENSITY * GRAVITY * (
        amplitude**2 + abs(reflection * amplitude) ** 2 - (transmitted_modulus * amplitude) ** 2
    )
    computed = mean_drift_force_far_field(
        reflection, amplitude=amplitude, density=DENSITY, gravity=GRAVITY
    )
    assert computed == pytest.approx(momentum, rel=1.0e-12)


def test_far_field_scales_with_amplitude_squared_and_vanishes_without_reflection():
    unit = mean_drift_force_far_field(0.25, amplitude=1.0, density=DENSITY, gravity=GRAVITY)
    doubled = mean_drift_force_far_field(0.25, amplitude=2.0, density=DENSITY, gravity=GRAVITY)
    assert doubled == pytest.approx(4.0 * unit, rel=1.0e-13)
    assert mean_drift_force_far_field(0.0) == 0.0


def test_far_field_rejects_invalid_input():
    with pytest.raises(ValueError):
        mean_drift_force_far_field(0.1, amplitude=0.0)
    with pytest.raises(ValueError):
        mean_drift_force_far_field(0.1, density=-1.0)
    with pytest.raises(ValueError):
        mean_drift_force_far_field(complex(np.nan, 0.0))


def test_tangential_recovery_reproduces_the_exact_incident_derivative():
    """The recovered dphi/ds must converge to the closed-form plane-wave derivative."""
    nu = KA
    errors = []
    for panels in (32, 64, 128):
        mesh = submerged_circle_mesh(panels, 1.0, 2.0)
        potential, exact = _incident_trace_and_derivative(mesh, nu)
        recovered = surface_tangential_velocity(mesh, potential)
        errors.append(
            float(np.linalg.norm(recovered - exact) / np.linalg.norm(exact))
        )
    assert errors[-1] < 1.0e-3
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    for ratio in ratios:
        assert 3.4 < ratio < 4.6, f"observed order is not 2: ratios={ratios}"


def test_near_field_functional_matches_an_analytic_surface_integral():
    """Two-mode harmonic field with a known surface integral (exact truth).

    The wavenumbers differ and the coefficient is complex, so ``|grad phi|^2`` is
    not symmetric in x and the surface integral does not vanish identically.
    """
    nu, mu = 0.6, 0.9
    coefficient = 0.7 * np.exp(1.0j * math.pi / 3.0)

    def field(x, z):
        first = np.exp(nu * z + 1.0j * nu * x)
        second = np.exp(mu * z + 1.0j * mu * x)
        potential = first + coefficient * second
        grad_x = 1.0j * nu * first + 1.0j * mu * coefficient * second
        grad_z = nu * first + mu * coefficient * second
        return potential, grad_x, grad_z

    theta = np.linspace(0.0, 2.0 * math.pi, 65537)[:-1]
    _, grad_x, grad_z = field(np.cos(theta), np.sin(theta) - 2.0)
    tangential = grad_x * -np.sin(theta) + grad_z * np.cos(theta)
    step = 2.0 * math.pi / theta.size
    reference = 0.25 * DENSITY * np.sum(np.abs(tangential) ** 2 * np.cos(theta)) * step

    errors = []
    for panels in (32, 64, 128):
        mesh = submerged_circle_mesh(panels, 1.0, 2.0)
        potential, _, _ = field(mesh.centers[:, 0], mesh.centers[:, 1])
        force = mean_drift_force_near_field(mesh, potential, density=DENSITY)[0]
        errors.append(float(abs(force - reference) / abs(reference)))
    assert errors[0] < 2.0e-2
    assert errors[-1] < 2.0e-3
    assert errors[-1] < errors[0] / 10.0


def test_circular_cylinder_drift_force_is_zero_to_machine_precision():
    """Dean/Ursell zero-reflection theorem: the deep-water circular cylinder is force free."""
    mesh = submerged_circle_mesh(64, 1.0, 2.0)
    result = solve_diffraction_indirect(mesh, KA, 1.0, density=DENSITY, gravity=GRAVITY)
    force = mean_drift_force(mesh, result, amplitude=1.0, density=DENSITY, gravity=GRAVITY)
    assert abs(result.reflection_coefficient) < 1.0e-12
    assert force.far_field / force.energy_scale < 1.0e-20
    # The near-field route integrates |grad phi|^2, an O(rho*g*A^2) quantity, and
    # must cancel to a force that is negligible against that scale.
    assert abs(force.near_field[0]) / force.energy_scale < 1.0e-10


def test_ellipse_near_and_far_field_routes_agree():
    mesh = submerged_ellipse_mesh(96, 1.4, 0.7, 1.2)
    result = solve_diffraction_indirect(mesh, KA, 1.0, density=DENSITY, gravity=GRAVITY)
    force = mean_drift_force(mesh, result, amplitude=1.0, density=DENSITY, gravity=GRAVITY)
    assert abs(result.reflection_coefficient) > 0.2
    assert force.far_field / force.energy_scale > 0.02
    assert abs(force.relative_discrepancy) < 0.05


def test_force_is_quadratic_in_wave_amplitude():
    mesh = submerged_ellipse_mesh(32, 1.4, 0.7, 1.2)
    values = {}
    for amplitude in (0.5, 1.0, 2.0):
        result = solve_diffraction_indirect(
            mesh, KA, amplitude, density=DENSITY, gravity=GRAVITY
        )
        force = mean_drift_force(
            mesh, result, amplitude=amplitude, density=DENSITY, gravity=GRAVITY
        )
        values[amplitude] = (force.far_field, float(force.near_field[0]))
    far_field_ratios = {amp: pair[0] / amp**2 for amp, pair in values.items()}
    near_field_ratios = {amp: pair[1] / amp**2 for amp, pair in values.items()}
    for amp in (0.5, 2.0):
        assert far_field_ratios[amp] == pytest.approx(far_field_ratios[1.0], rel=1.0e-6)
        assert near_field_ratios[amp] == pytest.approx(near_field_ratios[1.0], rel=1.0e-6)
    assert values[2.0][1] > 0.0, "the strongly reflecting ellipse must feel a positive force"


def test_band_limit_of_the_recovery_is_not_load_bearing():
    mesh = submerged_ellipse_mesh(64, 1.4, 0.7, 1.2)
    result = solve_diffraction_indirect(mesh, KA, 1.0, density=DENSITY, gravity=GRAVITY)
    values = [
        float(
            mean_drift_force(
                mesh, result, density=DENSITY, gravity=GRAVITY, mode_fraction=fraction
            ).near_field[0]
        )
        for fraction in (0.15, 0.25, 0.35)
    ]
    spread = (max(values) - min(values)) / abs(sum(values) / len(values))
    assert spread < 0.02, f"band limit perturbs the near-field force: {values}"


def test_validation_of_recovery_inputs():
    mesh = submerged_circle_mesh(16, 1.0, 2.0)
    s, perimeter = panel_arclength(mesh)
    assert s.shape == (16,)
    assert np.all(np.diff(s) > 0.0)
    assert perimeter == pytest.approx(2.0 * math.pi, rel=5.0e-2)
    with pytest.raises(ValueError):
        surface_tangential_velocity(mesh, np.ones(15, dtype=complex))
    with pytest.raises(ValueError):
        surface_tangential_velocity(mesh, np.full(16, np.nan, dtype=complex))
    with pytest.raises(ValueError):
        surface_tangential_velocity(mesh, np.ones(16, dtype=complex), mode_fraction=0.0)
    with pytest.raises(ValueError):
        surface_tangential_velocity(mesh, np.ones(16, dtype=complex), mode_fraction=0.75)
    with pytest.raises(TypeError):
        panel_arclength("not a mesh")
    with pytest.raises(TypeError):
        mean_drift_force(mesh, "not a result")


def _load_doc_generator():
    path = ROOT / "docs" / "tools" / "gen_drift_force_doc.py"
    spec = importlib.util.spec_from_file_location("gen_drift_force_doc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not STUDY_JSON.exists(), reason="cached drift-force study not present")
def test_documentation_matches_the_cached_study():
    """No number in the docs may be retyped: re-rendering the study must reproduce it."""
    study = json.loads(STUDY_JSON.read_text(encoding="utf-8"))
    generator = _load_doc_generator()
    expected = generator.render(study)
    assert DOC_PATH.exists(), "docs/drift_force_2d.md has not been generated"
    assert DOC_PATH.read_text(encoding="utf-8") == expected


@pytest.mark.skipif(not STUDY_JSON.exists(), reason="cached drift-force study not present")
def test_documented_headline_numbers_are_the_measured_ones():
    study = json.loads(STUDY_JSON.read_text(encoding="utf-8"))
    rows = study["ellipse"]
    last = rows[-1]
    assert last["relative_discrepancy"] == pytest.approx(
        (last["near_field_force_x"] - last["far_field_force_x"]) / last["far_field_force_x"],
        rel=1.0e-12,
    )
    assert STUDY_JSON.stat().st_size > 0
    numbers = re.findall(r"-?\d+\.\d+e[-+]\d+", DOC_PATH.read_text(encoding="utf-8"))
    assert numbers, "the generated documentation should quote measured values"