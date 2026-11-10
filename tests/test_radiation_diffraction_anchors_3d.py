"""Gates for the 3-D radiation/diffraction anchor surface.

These tests pin the *anchors* of
``tensorbem.benchmarks.radiation_diffraction_anchors_3d`` at a cost a
pull-request runner can afford:

* the closed hemisphere primitive is a valid closed mesh whose volume converges
  to ``2/3 pi a^3`` (the geometry every unbounded anchor is built on);
* the campaign's ``2/3 rho pi a^3`` closed form is reproduced by the *sphere*
  on a geodesic mesh (machinery anchor);
* the infinite-frequency (pressure-release) added mass of a floating hemisphere
  climbs towards the theorem-exact ``rho pi a^3 / 3`` as ``ka`` grows;
* the Froude-Krylov closed forms for a wall-sided barge match the panel
  integration of the incident-wave pressure well inside 3 %, with the right
  phase structure (heave real, surge imaginary);
* the long-wave / high-frequency RAO limits and the ``omega -> 0`` negative
  control behave as the campaign asks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import froude_krylov_force_3d, hemisphere_solid_mesh
from tensorbem.benchmarks.radiation_diffraction_anchors_3d import (
    barge_rao,
    froude_krylov_barge_closed_form,
    froude_krylov_barge_panels,
    hemisphere_high_frequency_limit,
    hemisphere_unbounded_added_mass,
    oblate_disk_limit,
    sphere_unbounded_added_mass,
)
from tensorbem.geometry3d import box_hull_mesh
from tensorbem.solver3d import compute_added_mass_matrix_3d

DENSITY = 1025.0
GRAVITY = 9.80665


def test_hemisphere_solid_mesh_is_closed_and_volume_converges() -> None:
    radii = (1.0, 0.4)
    for radius in radii:
        volumes = []
        for n_theta, n_phi in ((4, 8), (8, 16), (16, 32)):
            mesh = hemisphere_solid_mesh(radius, n_theta, n_phi)
            assert mesh.panel_count == 2 * n_phi * (n_theta - 1) + 2 * n_phi
            tri = mesh.vertices[mesh.faces]
            volumes.append(
                abs(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()) / 6.0
            )
            assert np.all(mesh.vertices[:, 2] <= 0.0)
        exact = 2.0 / 3.0 * math.pi * radius**3
        # inscribed polyhedron: monotone from below with second-order error
        assert volumes[0] < volumes[1] < volumes[2] < exact
        assert abs(volumes[2] - exact) / exact < 1.0e-2
        assert abs(volumes[1] - exact) / abs(volumes[2] - exact) > 3.0


def test_hemisphere_solid_mesh_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        hemisphere_solid_mesh(radius=-1.0)
    with pytest.raises(ValueError):
        hemisphere_solid_mesh(1.0, n_theta=0)
    with pytest.raises(ValueError):
        hemisphere_solid_mesh(1.0, n_phi=2)
    with pytest.raises(ValueError):
        hemisphere_solid_mesh(1.0, center=(0.0, 0.0))


def test_sphere_unbounded_added_mass_reproduces_campaign_closed_form() -> None:
    result = sphere_unbounded_added_mass(subdivisions=(1, 2), density=DENSITY)
    assert result["analytic_a33"] == pytest.approx(2.0 / 3.0 * DENSITY * math.pi, rel=1e-12)
    coarse, fine = result["rows"]
    assert fine["relative_residual"] < 1.0e-12
    # faceted geodesic sphere: error falls with refinement and the Richardson
    # limit recovers the classical closed form (the full study also carries the
    # 1280-panel level, 0.51 % raw error)
    assert fine["rel_error"] < coarse["rel_error"]
    assert fine["rel_error"] < 0.025
    assert result["richardson_rel_error"] < 5.0e-3


def test_closed_hemisphere_unbounded_added_mass_exceeds_the_disk_limit() -> None:
    """A closed hemisphere is not a sphere: its flat lid drives a piston flow.

    The campaign anchor ``2/3 rho pi a^3`` is the sphere value and the
    hemisphere's displaced mass; the measured axial added mass must instead sit
    just above the thin-disk limit ``8/3 rho a^3`` and converge monotonically.
    """
    result = hemisphere_unbounded_added_mass(((6, 12), (8, 16), (12, 24)), density=DENSITY)
    values = [row["a33"] for row in result["rows"]]
    assert values[0] < values[1] < values[2]
    disk = oblate_disk_limit(density=DENSITY)["closed_form"]
    sphere = 2.0 / 3.0 * DENSITY * math.pi
    assert result["richardson_a33"] > disk
    assert result["richardson_a33"] > 1.2 * sphere
    assert abs(oblate_disk_limit(density=DENSITY)["a33"] - disk) / disk < 1.0e-3


def test_high_frequency_added_mass_climbs_to_pressure_release_anchor() -> None:
    result = hemisphere_high_frequency_limit((1.0, 3.0, 8.0), subdivisions=2, density=DENSITY)
    limit = result["analytic_limit"]
    assert limit == pytest.approx(DENSITY * math.pi / 3.0, rel=1e-12)
    ratios = [row["a33"] / limit for row in result["rows"]]
    assert ratios[0] < ratios[-1]
    assert 0.85 < ratios[-1] < 1.0


def test_froude_krylov_closed_form_matches_panel_integration() -> None:
    payload = froude_krylov_barge_panels(((10, 5, 1), (20, 10, 2)), 0.5, density=DENSITY)
    finest = payload["rows"][-1]
    assert finest["f3_rel_error"] < 0.03
    assert finest["f5_rel_error"] < 0.03
    assert finest["f1_rel_error"] < 0.03
    # second order in the panel size
    assert payload["rows"][0]["f3_rel_error"] > finest["f3_rel_error"]


def test_froude_krylov_phase_structure() -> None:
    """Heave is in phase with the incident wave, surge is a pure added-mass load."""
    hull = box_hull_mesh(100.0, 20.0, 8.0, n_length=20, n_beam=10, n_draft=2)
    nu = 0.5**2 / GRAVITY
    force = froude_krylov_force_3d(hull, nu, density=DENSITY, gravity=GRAVITY)
    closed = froude_krylov_barge_closed_form(0.5, density=DENSITY)
    assert abs(force[2].real) > 1.0e3 * abs(force[2].imag)
    assert abs(force[0].imag) > 1.0e3 * abs(force[0].real)
    assert force[2].real > 0.0 and closed["f3"].real > 0.0
    # heave magnitude < incident-wave "piston" value because sin(kL/2)/(kL/2) < 1
    piston = DENSITY * GRAVITY * 100.0 * 20.0 * math.exp(-nu * 8.0)
    assert abs(force[2]) < piston


def test_long_wave_and_high_frequency_rao_limits() -> None:
    long_wave = barge_rao(0.05, n_length=10, n_beam=5, n_draft=1)
    assert 0.95 <= long_wave["heave_rao"] <= 1.05
    assert long_wave["pitch_rao"] < 0.1
    high = barge_rao(3.0, n_length=10, n_beam=5, n_draft=1)
    assert high["heave_rao"] < 0.2
    assert high["heave_rao"] < long_wave["heave_rao"]


def test_low_frequency_negative_control_added_mass_plateau_and_zero_damping() -> None:
    low = barge_rao(0.05, n_length=10, n_beam=5, n_draft=1)
    mid = barge_rao(0.1, n_length=10, n_beam=5, n_draft=1)
    assert abs(low["a33"] - mid["a33"]) / mid["a33"] < 0.05
    assert low["b33"] < mid["b33"]
    assert low["b33"] / (low["omega"] * low["a33"]) < 0.05


def test_report_document_is_generated_from_the_committed_payload() -> None:
    """The published report is a *render* of the committed JSON, not a copy.

    ``docs/radiation_diffraction_anchors_3d/build_report.py`` reads every number
    out of ``results.json``; re-rendering it here means an edit to the document
    (or a stale artifact) fails the suite instead of silently drifting.
    """
    import importlib.util
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    generator = repo / "docs" / "radiation_diffraction_anchors_3d" / "build_report.py"
    payload_path = repo / "docs" / "radiation_diffraction_anchors_3d" / "results.json"
    document = repo / "docs" / "radiation_diffraction_anchors_3d.md"
    spec = importlib.util.spec_from_file_location("_radiff_build_report", generator)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = json.loads(payload_path.read_text())
    rendered = module.render(payload)
    assert rendered.strip() == document.read_text().strip()

    # every criterion the payload reports must be visible in the document
    for item in payload["criteria"]:
        assert item["id"] in rendered
