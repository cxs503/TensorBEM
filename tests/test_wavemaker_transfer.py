"""Tests for the wavemaker transfer-function module (CASE-WAVE-FLUME).

The analytic references are the Biesel-Havelock transfer functions; the tests
also pin their two limits from long-wave mass conservation (``H/S -> kh`` for a
piston, ``kh/2`` for a bed-hinged flap) so a transcription slip in the closed
form cannot hide behind an equally wrong BEM number.

Three structural checks are worth stating because they bound what the case
proves: the operator assembled for the open wall must be the *same* operator the
graded closed-body solver assembles (:func:`assemble_indirect_matrix`, applied to
a submerged circle), the vertical wall's own co-planar entries must vanish
exactly (so the flume case tests kernel quadrature, not the integral equation),
and the inclined face must be non-degenerate while returning to the piston value
as the inclination vanishes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.benchmarks.ursell_cylinder import submerged_circle_mesh
from tensorbem.benchmarks.wave_flume_wavemaker import inclined_wavemaker, run
from tensorbem.finite_depth_solver import assemble_indirect_matrix
from tensorbem.wavemaker import (
    GRAVITY,
    WavemakerSurface,
    _gauss_nodes,
    assemble_normal_derivative,
    evanescent_decay,
    exact_elevation,
    free_surface_potential,
    kochin_amplitude,
    solve_sources,
    solve_wavemaker,
    transfer_function,
    velocity_profile,
    vertical_wavemaker,
)


def _nu(kh: float, depth: float = 1.0) -> tuple[float, float]:
    k = kh / depth
    omega = math.sqrt(GRAVITY * k * math.tanh(kh))
    return omega, omega * omega / GRAVITY


# --------------------------------------------------------------------------
# Analytic reference
# --------------------------------------------------------------------------
def test_piston_and_flap_limits_follow_long_wave_mass_balance() -> None:
    # Long waves: the wall's volume flux must equal the wave's flux, which gives
    # H/S -> kh for a uniform (piston) profile and half of that for a profile
    # whose depth mean is U0/2 (flap hinged at the bed).
    for kh in (1.0e-3, 1.0e-4):
        assert transfer_function("piston", kh) == pytest.approx(kh, rel=1.0e-2)
        assert transfer_function("flap", kh) == pytest.approx(0.5 * kh, rel=1.0e-2)


def test_piston_deep_water_limit_is_two() -> None:
    assert transfer_function("piston", 40.0) == pytest.approx(2.0, rel=1.0e-10)
    # flap: the deep-water approach is 2 (1 - 1/kh), slower than the piston's
    assert transfer_function("flap", 40.0) == pytest.approx(2.0 * (1.0 - 1.0 / 40.0), rel=1.0e-3)


def test_transfer_function_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        transfer_function("wedge", 1.0)
    with pytest.raises(ValueError):
        transfer_function("piston", 0.0)
    with pytest.raises(ValueError):
        transfer_function("piston", float("nan"))


def test_velocity_profiles_are_unit_at_the_waterline() -> None:
    depth = 2.0
    z = np.array([-depth, -0.5 * depth, 0.0])
    assert velocity_profile("piston", z, depth) == pytest.approx(np.ones(3))
    assert velocity_profile("flap", z, depth) == pytest.approx([0.0, 0.5, 1.0])
    with pytest.raises(ValueError):
        velocity_profile("piston", np.array([0.5]), depth)
    with pytest.raises(ValueError):
        velocity_profile("flap", np.array([-3.0]), depth)


# --------------------------------------------------------------------------
# Geometry and operator
# --------------------------------------------------------------------------
def test_wavemaker_geometry_spans_bed_to_waterline_with_downstream_normals() -> None:
    surface = vertical_wavemaker(1.0, 4)
    assert surface.panel_count == 4
    assert surface.vertices[:, 1].min() == pytest.approx(-1.0)
    assert surface.ends[:, 1].max() == pytest.approx(0.0)
    assert surface.normals == pytest.approx(np.tile([1.0, 0.0], (4, 1)))
    assert surface.perimeter == pytest.approx(1.0)
    for bad in (0, 1):
        with pytest.raises(ValueError):
            vertical_wavemaker(1.0, bad)


def test_gauss_nodes_stay_strictly_inside_the_fluid_strip() -> None:
    surface = vertical_wavemaker(1.0, 5)
    _, source_z, _ = _gauss_nodes(surface, 8, 1.0)
    assert np.all(source_z < 0.0)
    assert np.all(source_z > -1.0)


def test_open_wall_operator_equals_the_closed_body_assembly() -> None:
    """Same operator as the graded solver: compare on a submerged closed body."""

    mesh = submerged_circle_mesh(16, radius=0.5, submergence=2.0)
    reference = assemble_indirect_matrix(mesh, nu=1.0, depth=5.0, panel_nodes=8)
    wrapped = WavemakerSurface(np.stack((mesh.vertices, mesh.ends), axis=1))
    local = assemble_normal_derivative(wrapped, nu=1.0, depth=5.0, panel_nodes=8)
    scale = float(np.max(np.abs(reference)))
    assert scale > 0.0
    assert np.max(np.abs(local - reference)) / scale < 1.0e-12


def test_vertical_wall_operator_is_exactly_diagonal() -> None:
    """Co-planar panels: every kernel depends on |dx|, so d/dx vanishes."""

    omega, nu = _nu(1.0)
    surface = vertical_wavemaker(1.0, 32)
    matrix = assemble_normal_derivative(surface, nu, 1.0)
    assert float(np.max(np.abs(matrix))) < 1.0e-12
    system = matrix + 0.5 * np.eye(surface.panel_count)
    assert float(np.linalg.cond(system)) == pytest.approx(1.0, rel=1.0e-12)
    assert omega > 0.0


def test_inclined_face_is_non_degenerate_and_returns_to_the_piston_limit() -> None:
    omega, nu = _nu(1.0)
    vertical = assemble_normal_derivative(vertical_wavemaker(1.0, 32), nu, 1.0)
    vertical_off = float(np.max(np.abs(vertical - np.diag(np.diag(vertical)))))

    small = inclined_wavemaker(1.0, 32, 1.0e-3)
    small_off = float(
        np.max(np.abs(
            assemble_normal_derivative(small, nu, 1.0)
            - np.diag(np.diag(assemble_normal_derivative(small, nu, 1.0)))
        ))
    )
    steep = inclined_wavemaker(1.0, 32, 30.0)
    steep_matrix = assemble_normal_derivative(steep, nu, 1.0)
    steep_off = float(
        np.max(np.abs(steep_matrix - np.diag(np.diag(steep_matrix))))
    )
    assert vertical_off < 1.0e-12
    assert small_off < steep_off

    piston = transfer_function("piston", 1.0)
    transfers = []
    for angle in (1.0e-3, 30.0):
        surface = inclined_wavemaker(1.0, 32, angle)
        data = -0.5j * omega * np.ones(surface.panel_count)
        sources, _ = solve_sources(surface, data, nu, 1.0)
        transfers.append(2.0 * abs(kochin_amplitude(surface, sources, nu, 1.0)))
    assert transfers[0] == pytest.approx(piston, rel=1.0e-3)
    assert abs(transfers[1] / piston - 1.0) > 0.05


# --------------------------------------------------------------------------
# BEM against the analytic reference
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["piston", "flap"])
def test_transfer_function_matches_biesel_havelock(kind: str) -> None:
    result = solve_wavemaker(kind, kh=1.0, depth=1.0, panels=32)
    assert result.relative_error < 1.0e-3
    assert result.relative_residual < 1.0e-10
    assert result.condition_number == pytest.approx(1.0, rel=1.0e-10)


@pytest.mark.parametrize("kind", ["piston", "flap"])
def test_two_far_field_routes_agree_in_amplitude_and_phase(kind: str) -> None:
    result = solve_wavemaker(kind, kh=1.0, depth=1.0, panels=32)
    assert result.route_disagreement < 1.0e-3
    phase = abs(np.angle(result.probe_amplitude / result.kochan_amplitude))
    assert phase < 1.0e-3


@pytest.mark.parametrize("kind", ["piston", "flap"])
def test_near_field_matches_the_exact_eigenfunction_series(kind: str) -> None:
    """Includes the evanescent modes, so this is a field-level check."""

    surface = vertical_wavemaker(1.0, 32)
    omega, nu = _nu(1.0)
    data = -0.5j * omega * velocity_profile(kind, surface.centers[:, 1], 1.0)
    sources, _ = solve_sources(surface, data, nu, 1.0)
    wavelength = 2.0 * math.pi / (1.0 / 1.0)
    x = np.array([0.25, 1.0, 4.0]) * wavelength
    eta = 1j * omega / GRAVITY * free_surface_potential(surface, sources, nu, 1.0, x)
    exact = exact_elevation(kind, x, kh=1.0, depth=1.0, modes=20)
    assert np.max(np.abs(eta - exact) / np.abs(exact)) < 1.0e-3


def test_exact_series_reproduces_the_far_field_transfer_function() -> None:
    for kind in ("piston", "flap"):
        far = exact_elevation(kind, [50.0], kh=1.5, depth=1.0, modes=4)[0]
        assert 2.0 * abs(far) == pytest.approx(transfer_function(kind, 1.5), rel=1.0e-8)


def test_panel_convergence_is_second_order_and_close_at_64_panels() -> None:
    coarse = solve_wavemaker("piston", kh=2.0, depth=1.0, panels=16).relative_error
    fine = solve_wavemaker("piston", kh=2.0, depth=1.0, panels=32).relative_error
    assert fine < coarse / 3.0
    assert solve_wavemaker("piston", kh=2.0, depth=1.0, panels=64).relative_error < 1.0e-4


def test_evanescent_modes_decay_to_a_flat_far_field() -> None:
    rows = evanescent_decay("piston", kh=1.0, depth=1.0, panels=32)
    deviations = [row["deviation_from_flat"] for row in rows]
    assert deviations[0] > deviations[1] >= deviations[2]
    assert max(deviations[-2:]) < 1.0e-9


# --------------------------------------------------------------------------
# Input validation and the study driver
# --------------------------------------------------------------------------
def test_solve_validates_inputs() -> None:
    with pytest.raises(ValueError):
        solve_wavemaker("wedge")
    with pytest.raises(ValueError):
        solve_wavemaker("piston", kh=0.0)
    with pytest.raises(ValueError):
        solve_wavemaker("piston", depth=-1.0)
    surface = vertical_wavemaker(1.0, 4)
    with pytest.raises(ValueError):
        solve_sources(surface, np.ones(3), 1.0, 1.0)


def test_study_driver_reports_the_anchor_as_passed() -> None:
    study = run(quick=True)
    summary = study["summary"]
    assert summary["transfer_within_10pct"] is True
    assert summary["transfer_within_3pct"] is True
    assert summary["field_within_1pct"] is True
    assert summary["vertical_operator_is_diagonal"] is True
    assert study["anchor"] == "CASE-WAVE-FLUME"