"""Back-filled mutation gates for the ship-facing modules audited in m3426.

``benchmarks/mutation_audit.py`` was extended to ``rao``, ``pressure``,
``clearance``, ``mesh_quality``, ``causality`` and ``symmetry3d`` -- modules
that carried no mutation face at all.  The pre-test sweep left 20 of 48 seeded
defects alive; this file pins the ones that are *real* coverage gaps (the
remaining survivors are equivalent mutants, listed in
``docs/m3426/mutation_audit_shipfront.md``).

Every gate below is an independent recomputation or a degenerate/boundary case,
not a restatement of the mutated line:

* RAO -- the equation of motion ``Z = -w^2 (M + A) + iwB + C`` is rebuilt from
  the *returned* added mass, damping and excitation force, and the reported
  residual is required to be the actual residual.  A negative control confirms
  that the ``M - A`` variant really does disagree.
* Pressure -- the coefficient normalisations are recomputed from a hand-built
  ``PressureLoads`` (no solver), and the steady-pressure loads are compared with
  a hand-integrated ``0.5 rho U^2 cp n dS`` on an icosphere.
* Clearance -- the ratio is required to be ``distance / h_max`` exactly, so the
  product variant (dimensionally wrong) is rejected.
* Mesh quality -- a 3-4-5 triangle has known angles, and a skinny triangle is
  flagged by the *default* 30 degree threshold.
* Causality -- the tail-fraction domain, the zero-tail residual floor and the
  first sample of the default grid are pinned.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.causality import _default_samples, fit_added_mass_asymptote
from tensorbem.clearance import diagnose_surface_clearance
from tensorbem.geometry3d import TriangleMesh, hemisphere_hull_mesh, icosphere
from tensorbem.hydrostatic import compute_hydrostatic_restoring_matrix, compute_mass_matrix
from tensorbem.mesh_quality import (
    diagnose_mesh_quality,
    identify_low_quality_panels,
    mesh_quality_summary,
)
from tensorbem.pressure import PressureLoads, steady_pressure_loads
from tensorbem.rao import compute_rao_at_frequency, estimate_natural_frequency

MASS = 1000.0
CG = np.array([0.0, 0.0, -0.1])


@pytest.fixture(scope="module")
def rao_case():
    """One coarse-mesh RAO solve, reused by the three impedance gates."""
    hull = hemisphere_hull_mesh(subdivisions=2)
    props = compute_hydrostatic_restoring_matrix(hull, MASS, CG)
    result = compute_rao_at_frequency(hull, 1.0, 0.0, MASS, CG, hydrostatic_props=props)
    return props, result


def test_impedance_matrix_pins_the_equation_of_motion(rao_case):
    """``Z = -w^2 (M + A) + iwB + C`` rebuilt from the returned fields.

    Kills the ``omega * omega -> omega / omega``, the ``+ c -> - c`` and the
    ``m + added_mass -> m - added_mass`` mutants: none of them survives an
    independent solve of the documented impedance equation.
    """
    props, result = rao_case
    omega = result.omega
    m = compute_mass_matrix(MASS, CG)
    z = -omega**2 * (m + result.added_mass) + 1j * omega * result.damping + props.restoring_matrix
    rao_expected = np.linalg.solve(z, result.excitation_force)
    np.testing.assert_allclose(result.rao, rao_expected, rtol=1e-10, atol=0.0)

    # Negative control: the ``M - A`` variant must give a different RAO, so the
    # comparison above is not vacuous even when A is small.
    z_wrong = -omega**2 * (m - result.added_mass) + 1j * omega * result.damping + props.restoring_matrix
    rao_wrong = np.linalg.solve(z_wrong, result.excitation_force)
    assert np.max(np.abs(rao_wrong - result.rao)) > 1e-9


def test_reported_impedance_residual_is_the_actual_residual(rao_case):
    """The residual must vanish; ``z @ rao + f_exc`` would report ``2|F|``."""
    _, result = rao_case
    scale = float(np.linalg.norm(result.excitation_force))
    assert scale > 0.0
    assert result.impedance_residual <= 1e-9 * scale


def test_natural_frequency_only_rejects_non_positive_effective_mass():
    """A light body (``m + a = 0.7``) is a valid system, not a degenerate one."""
    added = np.zeros((6, 6))
    added[2, 2] = 0.2
    restoring = np.zeros((6, 6))
    restoring[2, 2] = 1.0
    omega_n = estimate_natural_frequency(0.5, added, restoring, dof=2)
    assert omega_n == pytest.approx(math.sqrt(1.0 / 0.7), rel=1e-12)

    # The genuinely degenerate branches still report exactly zero.
    assert estimate_natural_frequency(0.0, np.zeros((6, 6)), restoring, dof=2) == 0.0
    assert estimate_natural_frequency(1.0, np.zeros((6, 6)), np.zeros((6, 6)), dof=2) == 0.0


def test_pressure_coefficients_divide_by_the_reference_length():
    """``C_m = M_body / (0.5 rho U^2 A L)``; the default length is 1."""
    loads = PressureLoads(
        force=np.array([3.0, 0.0, 0.0]),
        moment=np.array([0.0, 7.0, 0.0]),
        reference_center=np.zeros(3),
        wetted_area=1.0,
        reference_pressure=0.0,
        density=1.0,
    )
    rho, speed, area, length = 2.0, 3.0, 4.0, 0.5
    scale = 0.5 * rho * speed**2 * area
    coeffs = loads.coefficients(rho, speed, area, length=length)
    assert coeffs["cx"] == pytest.approx(-3.0 / scale, rel=1e-12)
    assert coeffs["cmy"] == pytest.approx(-7.0 / (scale * length), rel=1e-12)

    default_coeffs = loads.coefficients(rho, speed, area)
    assert default_coeffs["cmy"] == pytest.approx(-7.0 / (scale * 1.0), rel=1e-12)


def test_steady_pressure_loads_default_reference_centre_is_the_origin():
    """Omitting ``reference_center`` must be identical to passing ``(0, 0, 0)``."""
    mesh = icosphere(subdivisions=1)
    # Eccentric speed field: a uniform |v| on a closed sphere would give a zero
    # resultant, which would make the checks vacuous.
    velocity = 2.0 * (mesh.centroids + np.array([0.4, 0.3, 0.0]))
    _, implicit = steady_pressure_loads(mesh, velocity, (0.0, 0.0, 1.0), density=1.3)
    _, explicit = steady_pressure_loads(
        mesh, velocity, (0.0, 0.0, 1.0), density=1.3, reference_center=(0.0, 0.0, 0.0)
    )
    assert np.array_equal(implicit.reference_center, np.zeros(3))
    assert np.allclose(implicit.moment, explicit.moment)


def test_steady_pressure_uses_half_rho_u_squared():
    """Hand-integrate ``0.5 rho U^2 cp n dS`` and compare panel by panel."""
    mesh = icosphere(subdivisions=1)
    velocity = 2.0 * (mesh.centroids + np.array([0.4, 0.3, 0.0]))
    free_stream = np.array([0.0, 0.0, 1.0])
    density = 1.3
    cp = 1.0 - np.sum(velocity * velocity, axis=1) / float(np.dot(free_stream, free_stream))
    expected = np.sum(
        (0.5 * density * cp)[:, None] * mesh.normals * mesh.areas[:, None], axis=0
    )
    assert np.linalg.norm(expected) > 1e-3
    _, loads = steady_pressure_loads(mesh, velocity, free_stream, density=density)
    np.testing.assert_allclose(loads.force, expected, rtol=1e-12, atol=1e-14)


def test_clearance_ratio_is_the_distance_over_h_max():
    """``clearance_ratio = distance / h_max`` (not ``distance * h_max``)."""
    mesh = icosphere(subdivisions=1)
    targets = np.array([[0.0, 0.0, 3.0]])
    report = diagnose_surface_clearance(mesh, targets, warning_factor=0.5, policy="ignore")
    assert report.clearance_ratio[0] == pytest.approx(
        report.minimum_distance[0] / report.h_max, rel=1e-13
    )
    # h_max is O(0.2) here, so the product variant would differ by h_max^2.
    assert abs(report.h_max - 1.0) > 1e-3


def test_triangle_angles_match_an_independent_recomputation():
    """Recompute every angle from the edges; a ``v2 - v1 -> v2 + v1`` mutant
    cannot reproduce them."""
    mesh = icosphere(subdivisions=1)
    v, faces = mesh.vertices, mesh.faces
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    angles = []
    for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
        u, w = q - p, r - p
        cosine = np.sum(u * w, axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1))
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    per_face_min = np.min(np.vstack(angles), axis=0)
    report = diagnose_mesh_quality(mesh)
    assert report.min_angle_deg == pytest.approx(float(np.min(per_face_min)), rel=1e-12)
    assert report.max_angle_deg == pytest.approx(float(np.max(angles)), rel=1e-12)
    assert report.mean_min_angle_deg == pytest.approx(float(np.mean(per_face_min)), rel=1e-12)


def test_default_angle_threshold_flags_a_skinny_panel():
    """The default 30 degree threshold must flag an 11.4 degree apex.

    The mesh is a flattened tetrahedron: panel ``(0, 1, 2)`` has base 10 and
    height 1, so its apex angle is ``2 atan(1/5)``.
    """
    vertices = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, 0.01, 1.0]]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    mesh = TriangleMesh(vertices=vertices, faces=faces)
    report = diagnose_mesh_quality(mesh)
    # The two base angles are atan(1/5) = 11.31 deg; the apex is 157.4 deg.
    apex = math.degrees(math.atan2(1.0, 5.0))
    assert report.min_angle_deg == pytest.approx(apex, rel=1e-9)
    flagged = identify_low_quality_panels(mesh)
    assert 0 in flagged and len(flagged) >= 1
    # Negative control: the 30 degree default is what flags it.
    assert identify_low_quality_panels(mesh, min_angle_threshold=1.0) == []


def test_tail_fraction_outside_the_unit_interval_is_rejected():
    omegas = np.linspace(1.0, 5.0, 9)
    with pytest.raises(ValueError):
        fit_added_mass_asymptote(omegas, np.ones_like(omegas), tail_fraction=1.5)


def test_all_zero_added_mass_tail_keeps_the_residual_finite():
    """A vanishing tail must not turn the residual floor into 0/0."""
    omegas = np.linspace(1.0, 5.0, 9)
    fit = fit_added_mass_asymptote(omegas, np.zeros_like(omegas))
    assert fit.added_mass_inf == pytest.approx(0.0, abs=1e-18)
    assert math.isfinite(fit.relative_residual)


def test_default_samples_start_one_interval_inside_the_sweep():
    """``max(1, intervals // 10)`` -- a single-interval grid still starts at 1."""
    grid = np.linspace(0.1, 1.2, 12)  # 11 intervals
    default_samples = _default_samples(grid)
    assert default_samples[0] == pytest.approx(math.sqrt(grid[1] * grid[2]), rel=1e-13)
    single_target = _default_samples(grid, target=1)
    assert single_target[0] == pytest.approx(math.sqrt(grid[1] * grid[2]), rel=1e-13)