"""Verification suite for the 3-D unbaffled exterior Helmholtz acoustic BEM.

Every test here is a machine-checkable verdict on a physical claim of
:mod:`tensorbem.acoustic3d`:

1. kernel continuity -- the ``k -> 0`` limit is the Laplace layer of
   :mod:`tensorbem.kernels3d`;
2. the exact single-layer self integral against the repository's polar
   quadrature;
3. the solid-angle row sums (``-1/2`` on the surface, ``-1`` inside) that fix
   the sign and the free term of the boundary-integral equation;
4. pulsating-sphere radiation against the closed-form monopole (criterion a,
   < 3 %);
5. rigid-sphere scattering cross-section against the Mie series (criterion b,
   < 5 %);
6. CHIEF suppression of the irregular-frequency peak (criterion c);
7. the ICES 209 one-third-octave machinery and a band-level comparison with the
   analytic reference spectrum (criterion d, < 2 dB);
8. energy balance of the BEM solution itself (surface intensity versus the
   far-field pattern), which needs no analytic reference;
9. the closed wetted-hull-plus-waterplane-lid body used for a real ship, with
   its block coefficient audited against the KCS anchor.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import acoustic3d as ac
from tensorbem.geometry3d import TriangleMesh, icosphere
from tensorbem.kcs_hull import kcs_class_mesh, kcs_geometry_report
from tensorbem.kernels3d import (
    double_layer_kernel_3d,
    integrate_single_layer_triangle_self,
)


# ---------------------------------------------------------------------------
# 1. kernels
# ---------------------------------------------------------------------------


def test_green_function_reduces_to_laplace_layer() -> None:
    """``G_k`` and ``dG_k/dn`` tend to the Rankine layer as ``k -> 0``."""

    r = np.array([0.3, 1.0, 2.5, 7.0])
    small_k = ac.helmholtz_green_3d(r, 1.0e-9)
    rankine = 1.0 / (4.0 * math.pi * r)
    assert np.allclose(np.real(small_k), rankine, rtol=0.0, atol=1e-9)
    assert np.max(np.abs(np.imag(small_k))) < 1e-7

    target = np.array([0.4, -0.2, 1.3])
    source = np.array([0.1, 0.5, 0.2])
    normal = np.array([0.0, 0.0, 1.0])
    helmholtz = ac.helmholtz_double_layer_kernel_3d(target, source, normal, 1.0e-9)
    laplace = double_layer_kernel_3d(target, source, normal)
    assert abs(helmholtz.real - laplace) < 1e-8
    assert abs(helmholtz.imag) < 1e-7


def test_green_function_is_outgoing() -> None:
    """The ``e^{-i*omega*t}`` convention gives an outward-travelling ``exp(+i*k*r)``."""
    r = 12.0
    k = 3.0
    value = ac.helmholtz_green_3d(r, k)
    assert abs(value - np.exp(1j * k * r) / (4.0 * math.pi * r)) < 1e-15
    # advancing time moves the phase outward: arg grows with r
    phase_r1 = np.angle(ac.helmholtz_green_3d(r, k))
    phase_r2 = np.angle(ac.helmholtz_green_3d(r + 0.1, k))
    assert phase_r2 != phase_r1
    with pytest.raises(ValueError):
        ac.helmholtz_green_3d(0.0, k)


# ---------------------------------------------------------------------------
# 2./3. operator assembly identities
# ---------------------------------------------------------------------------


def test_single_layer_self_integral_closed_form() -> None:
    """The analytic sector sum equals the repository's polar quadrature."""
    rng = np.random.default_rng(20261111)
    for _ in range(6):
        vertices = rng.normal(size=(3, 3)) * np.array([2.0, 1.0, 0.5])
        analytic = ac.single_layer_triangle_self_laplace(vertices)
        quadrature = integrate_single_layer_triangle_self(vertices, order=256)
        assert analytic == pytest.approx(quadrature, rel=1e-10)


def test_solid_angle_row_sums_fix_the_free_term() -> None:
    """Surface rows sum to ``-1/2`` and interior rows to ``-1`` (Laplace limit)."""
    mesh = icosphere(2, 1.0)
    surface = ac.assemble_helmholtz_layer_matrices(mesh, 0.0)
    assert np.allclose(surface.laplace_row_sums, -0.5, atol=1e-12)
    # the collocation panel itself carries the principal value, i.e. zero
    assert np.allclose(np.diag(surface.double_layer), 0.0, atol=1e-14)

    interior = ac.assemble_helmholtz_layer_matrices(mesh, 0.0, targets=np.zeros((1, 3)))
    assert interior.laplace_row_sums[0] == pytest.approx(-1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 4. pulsating sphere (criterion a)
# ---------------------------------------------------------------------------


def test_pulsating_sphere_radiation_matches_analytic() -> None:
    """Criterion (a): monopole radiation pressure within 3 % of the closed form."""
    radius = 1.0
    mesh = icosphere(3, radius)
    field_points = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, -5.0], [10.0, 3.0, 1.0]])
    errors = []
    for ka in (0.5, 1.0, 2.0, 3.0):
        wavenumber = ka / radius
        solution = ac.solve_exterior_neumann(mesh, wavenumber, 1.0)
        potential = ac.field_potential(
            mesh, wavenumber, solution.phi, solution.normal_velocity, field_points
        )
        analytic = ac.pulsating_sphere_potential(radius, 1.0, wavenumber, field_points)
        errors.append(np.max(np.abs(potential - analytic) / np.abs(analytic)))
        # the CHIEF-augmented system is solved in the least-squares sense, so the
        # surface rows are satisfied to the fit residual, not to machine zero
        assert solution.boundary_relative_residual < 1e-2
    assert max(errors) < 0.03, errors


def test_pressure_and_level_conventions() -> None:
    """``p = i*omega*rho*phi`` and ``SPL = 20*log10(p_rms/1 uPa)``."""
    k = 2.0
    phi = 3.0 + 4.0j
    pressure = ac.acoustic_pressure(phi, k)
    assert pressure == pytest.approx(1j * k * 1500.0 * 1025.0 * phi)
    # 1 Pa RMS is 120 dB re 1 uPa
    assert ac.sound_pressure_level(1.0) == pytest.approx(120.0, abs=1e-12)
    assert ac.rms_from_amplitude(2.0) == pytest.approx(math.sqrt(2.0), rel=1e-15)


# ---------------------------------------------------------------------------
# 5. rigid sphere scattering (criterion b)
# ---------------------------------------------------------------------------


def test_rigid_sphere_scattering_matches_mie() -> None:
    """Criterion (b): scattering cross-section within 5 % of the Mie series."""
    radius = 1.0
    mesh = icosphere(3, radius)
    ratios = []
    for ka in (0.5, 1.0, 2.0):
        wavenumber = ka / radius
        solution = ac.solve_rigid_scattering(mesh, wavenumber, np.array([1.0, 0.0, 0.0]))
        sigma = ac.scattering_cross_section(
            mesh, wavenumber, solution.phi, solution.normal_velocity, n_theta=64, n_phi=64
        )
        exact = ac.rigid_sphere_scattering_cross_section(wavenumber, radius)
        ratios.append(abs(sigma - exact) / exact)
    assert max(ratios) < 0.05, ratios


def test_rigid_sphere_mie_coefficients_are_finite_and_small_in_rayleigh_limit() -> None:
    """At ``ka << 1`` the rigid-sphere cross-section approaches the Rayleigh law."""
    radius = 1.0
    ka = 0.05
    sigma = ac.rigid_sphere_scattering_cross_section(ka, radius)
    # Rayleigh scattering of a rigid sphere: sigma -> (7/9) pi a^2 (ka)^4
    rayleigh = 7.0 / 9.0 * math.pi * radius**2 * (ka) ** 4
    assert sigma == pytest.approx(rayleigh, rel=2e-2)


# ---------------------------------------------------------------------------
# 6. CHIEF (criterion c)
# ---------------------------------------------------------------------------


def _radiation_error(mesh: TriangleMesh, radius: float, ka: float, chief: int, points):
    wavenumber = ka / radius
    if chief:
        solution = ac.solve_exterior_neumann(mesh, wavenumber, 1.0, n_chief=chief)
    else:
        solution = ac.solve_exterior_neumann(
            mesh, wavenumber, 1.0, chief_points=np.zeros((0, 3))
        )
    potential = ac.field_potential(mesh, wavenumber, solution.phi, solution.normal_velocity, points)
    analytic = ac.pulsating_sphere_potential(radius, 1.0, wavenumber, points)
    return float(np.max(np.abs(potential - analytic) / np.abs(analytic)))


def test_chief_suppresses_the_irregular_frequency_peak() -> None:
    """Criterion (c): the spurious interior-Dirichlet resonance disappears."""
    radius = 1.0
    mesh = icosphere(2, radius)
    points = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, -5.0]])
    window = np.linspace(3.0, 3.4, 5)
    bare = max(_radiation_error(mesh, radius, ka, 0, points) for ka in window)
    chief = max(_radiation_error(mesh, radius, ka, 5, points) for ka in window)
    assert bare > 0.2, f"no spurious peak detected: bare max error {bare:.3f}"
    assert chief < 0.06, f"CHIEF failed to remove the peak: {chief:.3f}"
    assert chief < 0.5 * bare


def test_chief_points_are_inside_the_body() -> None:
    """Auto-generated CHIEF points sit strictly inside the closed surface."""
    radius = 1.0
    mesh = icosphere(2, radius)
    points = ac.interior_chief_points(mesh, count=5)
    radii = np.linalg.norm(points, axis=1)
    # the rays stop at 40 % of the way to the *panel centroids*, which for a
    # faceted sphere sit slightly inside the exact radius
    assert np.all(radii > 0.35 * radius) and np.all(radii < 0.41 * radius)
    assert np.all(np.linalg.norm(points - points.mean(axis=0), axis=1) > 0.1 * radius)
    with pytest.raises(ValueError):
        ac.interior_chief_points(mesh, count=0)


# ---------------------------------------------------------------------------
# 7. ICES 209 bands (criterion d)
# ---------------------------------------------------------------------------


def test_ices209_nominal_band_grid() -> None:
    """The 31 nominal centres span 10 Hz - 10 kHz at one-third-octave spacing."""
    centres = ac.third_octave_centres()
    assert centres.shape == (31,)
    assert centres[0] == 10.0 and centres[-1] == 10000.0
    lower, upper = ac.third_octave_edges(centres)
    assert np.allclose(upper / lower, 2.0 ** (1.0 / 3.0), rtol=1e-12)
    assert np.allclose(centres[1:] / centres[:-1], 2.0 ** (1.0 / 3.0), rtol=0.06)
    assert ac.third_octave_centres(50.0, 200.0).tolist() == [50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0]
    with pytest.raises(ValueError):
        ac.third_octave_centres(0.0, 100.0)


def test_ices209_band_levels_match_the_analytic_monopole_spectrum() -> None:
    """Criterion (d): third-octave band SPL within 2 dB of the exact reference."""
    radius = 0.2
    field_point = np.array([[1.0, 0.0, 0.0]])
    levels_bem, levels_exact = [], []
    for frequency in (10.0, 1000.0, 10000.0):
        k = float(ac.wavenumber_from_frequency(frequency))
        mesh = ac.icosphere_for_acoustic(k, radius, panels_per_wavelength=4.0, max_subdivisions=3)
        solution = ac.solve_exterior_neumann(mesh, k, 1.0)
        potential = ac.field_potential(
            mesh, k, solution.phi, solution.normal_velocity, field_point
        )
        pressure = ac.acoustic_pressure(potential, k)
        analytic = ac.pulsating_sphere_pressure(radius, 1.0, k, field_point)
        levels_bem.append(ac.third_octave_band_levels([frequency], pressure)[0])
        levels_exact.append(ac.third_octave_band_levels([frequency], analytic)[0])
    difference = np.abs(np.asarray(levels_bem) - np.asarray(levels_exact))
    assert np.all(difference < 2.0), difference


# ---------------------------------------------------------------------------
# 8. energy balance of the BEM solution itself
# ---------------------------------------------------------------------------


def test_radiated_power_energy_balance() -> None:
    """Surface intensity and the far-field pattern give the same radiated power."""
    radius = 1.0
    mesh = icosphere(2, radius)
    for ka in (0.5, 2.0):
        wavenumber = ka / radius
        solution = ac.solve_exterior_neumann(mesh, wavenumber, 1.0)
        surface = ac.radiated_power_surface(
            mesh, wavenumber, solution.phi, solution.normal_velocity
        )
        far = ac.radiated_power_far_field(
            mesh, wavenumber, solution.phi, solution.normal_velocity, n_theta=48, n_phi=48
        )
        assert surface > 0.0
        assert abs(far - surface) / surface < 0.02


def test_monopole_superposition_is_the_no_scattering_surrogate() -> None:
    """The single-layer surrogate is right only where the body is compact.

    A uniformly pulsating sphere *is* a monopole, but the field it produces is
    represented by **both** layer potentials: the single layer carries the
    volume-velocity source (with the ``-G_k`` sign of a point source) and the
    double layer carries the surface-pressure (radiation loading) part.  In the
    compact limit ``ka -> 0`` the double layer vanishes for a far receiver and
    the surrogate must reproduce the BEM; at ``ka = 3`` it must *not*, which is
    exactly the effect this module exists to compute.
    """
    radius = 1.0
    mesh = icosphere(2, radius)
    field_point = np.array([[6.0, 0.0, 0.0]])

    def difference(ka: float) -> float:
        wavenumber = ka / radius
        solution = ac.solve_exterior_neumann(mesh, wavenumber, 1.0)
        bem = ac.acoustic_pressure(
            ac.field_potential(mesh, wavenumber, solution.phi, solution.normal_velocity, field_point),
            wavenumber,
        )
        surrogate = ac.monopole_superposition_pressure(
            mesh, wavenumber, solution.normal_velocity, field_point
        )
        return float(abs(surrogate[0] - bem[0]) / abs(bem[0]))

    assert difference(0.05) < 0.02      # compact: single layer alone is the source
    assert difference(3.0) > 0.5        # loading + phase averaging dominate


# ---------------------------------------------------------------------------
# 9. the ship body: closed wetted hull with a waterplane lid
# ---------------------------------------------------------------------------


def test_closed_hull_body_geometry_matches_the_kcs_anchor() -> None:
    """Closing the wetted hull keeps the block coefficient of the KCS anchor."""
    hull = kcs_class_mesh(n_sections=21)
    body = ac.close_surface_with_plane_lid(hull)
    assert body.faces.shape[0] == hull.faces.shape[0] + np.unique(hull.waterline_edges).shape[0]
    assert body.volume == pytest.approx(hull.volume, rel=1e-12)
    report = kcs_geometry_report(mesh=hull)
    block = body.volume / (report["kcs_length"] * report["kcs_breadth"] * report["kcs_draft"])
    assert block == pytest.approx(report["kcs_block_coefficient"], rel=1e-2)
    # the lid is the only part of the body at the waterplane, and it faces up
    lid_centroids = body.centroids[np.abs(body.centroids[:, 2]) < 1e-9]
    lid_normals = body.normals[np.abs(body.centroids[:, 2]) < 1e-9]
    assert lid_centroids.shape[0] >= 3
    assert np.allclose(lid_normals[:, 2], 1.0, atol=1e-12)


def test_hull_radiation_band_level_is_finite_and_resolved() -> None:
    """A KCS hull radiating at 25 Hz gives a finite, mesh-resolved band level."""
    hull = kcs_class_mesh(n_sections=41)
    body = ac.close_surface_with_plane_lid(hull)
    wetted = np.abs(body.centroids[:, 2]) > 1e-9
    source = np.array([-0.15 * 230.0, 0.0, -10.8])
    distance = np.linalg.norm(body.centroids - source[None, :], axis=1)
    velocity = np.zeros(body.panel_count, dtype=np.complex128)
    velocity[wetted] = 1.0e-3 * np.exp(-0.5 * (distance[wetted] / 13.8) ** 2)

    frequency = 25.0
    resolution = ac.acoustic_resolution(body, frequency)
    assert resolution.panels_per_wavelength < 1.0 / 6.0  # adequate at 25 Hz

    k = float(ac.wavenumber_from_frequency(frequency))
    solution = ac.solve_exterior_neumann(body, k, velocity)
    field_point = np.array([[0.0, 100.0, -25.0]])
    pressure = ac.acoustic_pressure(
        ac.field_potential(body, k, solution.phi, solution.normal_velocity, field_point), k
    )
    level = ac.sound_pressure_level(ac.rms_from_amplitude(pressure))[0]
    assert np.isfinite(level)
    assert 60.0 < level < 200.0
    # least-squares fit residual of the CHIEF-augmented system on a 548-panel hull
    assert solution.boundary_relative_residual < 1e-2

    power = ac.radiated_power_surface(body, k, solution.phi, solution.normal_velocity)
    assert power > 0.0

# ---------------------------------------------------------------------------
# 10. documentation pins (zero transcription)
# ---------------------------------------------------------------------------


def test_documented_pins_match_the_archived_json() -> None:
    """Every number quoted in the report doc is re-derived from ``criteria.json``.

    The pins block is *written* by ``scripts/render_acoustic3d_report.py`` and
    the doc reads it through, so this test is the anti-transcription guard: if
    anybody edits a number in the prose without regenerating the artifact (or
    regenerates the artifact without the doc), it goes red.
    """
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    out = repo / "docs" / "acoustic3d_ices209"
    criteria = json.loads((out / "criteria.json").read_text())
    hull = json.loads((out / "hull_band_table.json").read_text())
    doc = (repo / "docs" / "acoustic3d_helmholtz.md").read_text()
    begin = "<!-- pins:begin -->"
    end = "<!-- pins:end -->"
    assert begin in doc and end in doc
    block = doc[doc.index(begin) + len(begin) : doc.index(end)]
    pins = json.loads(block[block.index("{") : block.rindex("}") + 1])

    crit = criteria["criteria"]
    computed = [row for row in hull["rows"] if row["verdict"] == "COMPUTED"]
    expected = {
        "a_max_relative_error_pct": round(crit["a_pulsating_sphere"]["measured_max_relative_error"] * 100, 3),
        "b_max_relative_error_pct": round(crit["b_rigid_sphere_scattering"]["measured_max_relative_error"] * 100, 3),
        "c_bare_max_pct": round(crit["c_chief_irregular_frequency"]["measured_bare_max"] * 100, 2),
        "c_chief_max_pct": round(crit["c_chief_irregular_frequency"]["measured_chief_max"] * 100, 2),
        "d_max_abs_delta_db": round(crit["d_ices209_third_octave"]["measured_max_abs_delta_db"], 3),
        "energy_balance_max_pct": round(crit["energy_balance"]["measured_max_relative_difference"] * 100, 3),
        "hull_panels": int(hull["geometry"]["panel_count"]),
        "hull_block_coefficient": round(hull["geometry"]["block_coefficient"], 4),
        "hull_max_resolved_frequency_hz": max(row["centre_hz"] for row in computed),
        "hull_max_self_scattering_delta_db": round(
            max(row["self_scattering_delta_db"] for row in computed), 2
        ),
        "hull_bands_computed": len(computed),
        "hull_bands_withheld": sum(1 for row in hull["rows"] if row["verdict"] != "COMPUTED"),
    }
    assert pins == expected
    # the criteria themselves are archived as PASS/FAIL, and everything passed
    assert criteria["all_pass"] is True, criteria["verdicts"]
    for key, value in criteria["criteria"].items():
        assert value["verdict"] == "PASS", (key, value.get("measured_max_relative_error"))
