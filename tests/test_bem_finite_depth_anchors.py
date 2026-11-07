"""Acceptance tests for the ``bem_finite_depth`` verification case.

The case is ``benchmarks/verified/bem_finite_depth``: a finite-depth
frequency-domain potential-flow loop with three anchors (hydrostatics, Kramers-
Kronig causality, diffraction energy), a finite-depth -> infinite-depth limit and
a section x depth x frequency matrix.  See
``docs/verification/bem_finite_depth.md`` for the physics.

Two kinds of test live here:

* **artifact gates** -- the committed ``result.json`` / ``gates.json`` must
  contain every measured number, every number must be inside its threshold, and
  the matrix must have the declared axis lengths.  These also check that the
  recorded numbers are *reproducible from the recorded inputs* (the analytic
  Ursell control is recomputed from the stored sweep), so a hand-edited artifact
  fails.
* **method gates** -- a handful of cheap recomputations of the machinery itself
  (section geometry, dispersion, one diffraction solve, the flat-bottom assembly
  that used to divide by zero).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.analytic import ursell_cylinder_added_mass
from tensorbem.benchmarks import bem_finite_depth_anchors as anchors
from tensorbem.finite_depth import finite_depth_wavenumber
from tensorbem.finite_depth_diffraction import solve_diffraction_indirect
from tensorbem.free_surface_solver import _rankine_normal_derivative_matrix
from tensorbem.geometry import PanelMesh

REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "benchmarks" / "verified" / "bem_finite_depth"


@pytest.fixture(scope="module")
def result() -> dict:
    path = CASE_DIR / "result.json"
    assert path.is_file(), f"missing case record {path}"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gates() -> dict:
    path = CASE_DIR / "gates.json"
    assert path.is_file(), f"missing gate record {path}"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. artifact gates
# ---------------------------------------------------------------------------
def test_every_gate_passes_and_is_reported(gates: dict) -> None:
    assert gates["case"] == "bem_finite_depth"
    assert gates["all_gates_passed"] is True
    assert gates["failed_gates"] == []
    ids = {row["id"] for row in gates["gates"]}
    assert ids == {
        "G1_geometry",
        "G2_kkr_added_mass",
        "G3_kkr_damping_shape",
        "G4_diffraction_energy",
        "G5_radiation_energy",
        "G6_deep_water_limit",
        "G7_matrix_completeness",
        "G8_panel_refinement",
    }


def test_measured_values_sit_inside_thresholds(gates: dict) -> None:
    for row in gates["gates"]:
        assert row["passed"] is True, row
        if row["id"] == "G7_matrix_completeness":
            assert row["measured"] == row["threshold"] == 45.0
        else:
            assert row["measured"] <= row["threshold"], row


def test_matrix_axes_are_complete(result: dict) -> None:
    matrix = result["legs"]["matrix"]
    cases = matrix["cases"]
    assert matrix["case_count"] == 45
    hulls = {case["hull"] for case in cases}
    depths = {case["depth_ratio"] for case in cases}
    frequencies = {case["nu_times_draught"] for case in cases}
    assert len(hulls) == 3
    assert depths == {1.5, 3.0, None}
    assert frequencies == set(anchors.MATRIX_NU_T)
    assert len({(c["hull"], c["depth_ratio"], c["nu_times_draught"]) for c in cases}) == 45
    for case in cases:
        assert case["heave"]["added_mass"] > 0.0
        assert case["heave"]["damping_pressure"] >= 0.0
        assert case["diffraction"]["transmission"] >= 0.0
        if case["depth"] is not None:
            # the section must sit strictly inside the strip
            assert case["depth"] > anchors.section_draught()


def test_matrix_energy_identities_are_recorded_per_case(result: dict) -> None:
    worst_diffraction = 0.0
    worst_radiation = 0.0
    for case in result["legs"]["matrix"]["cases"]:
        diffraction = case["diffraction"]
        closure = diffraction["reflection"] ** 2 + diffraction["transmission"] ** 2
        assert abs(closure - 1.0 - diffraction["energy_balance_error"]) < 1e-9
        worst_diffraction = max(worst_diffraction, diffraction["energy_balance_error"])
        for motion in ("heave", "sway"):
            worst_radiation = max(worst_radiation, case[motion]["energy_relative_error"])
    assert worst_diffraction <= anchors.GATE_DIFFRACTION_ENERGY
    assert worst_radiation <= anchors.GATE_RADIATION_ENERGY


def test_analytic_kkr_control_reproduces_from_the_recorded_sweep(result: dict) -> None:
    """The control curve must be recomputable, i.e. not hand-typed."""
    leg = result["legs"]["causality"]
    ka = np.asarray(leg["sweep_ka"], dtype=float)
    omega = np.sqrt(ka * anchors.GRAVITY / leg["geometry"]["radius"])
    recorded_mass = np.asarray(leg["analytic_control"]["added_mass"], dtype=float)
    recorded_damping = np.asarray(leg["analytic_control"]["damping"], dtype=float)
    assert recorded_mass.size == omega.size == anchors.KKR_NODES
    for index in (0, ka.size // 3, ka.size - 1):
        mass, damping = ursell_cylinder_added_mass(
            leg["geometry"]["radius"],
            leg["geometry"]["submergence"],
            float(omega[index]),
            density=anchors.DENSITY,
        )
        assert mass == pytest.approx(recorded_mass[index], rel=1e-12)
        assert damping == pytest.approx(recorded_damping[index], rel=1e-12)


def test_kkr_residuals_hit_the_gate_and_the_floor_is_reported(result: dict) -> None:
    leg = result["legs"]["causality"]
    main = leg["panel_sweeps"][-1]
    assert main["kkr_added_mass"] <= anchors.GATE_KKR
    assert main["kkr_damping_offset_removed"] <= anchors.GATE_KKR
    # the band-limited raw number is larger than the offset-removed one: the
    # constant part of the residual is a truncation artefact, and it is recorded.
    assert main["kkr_damping_raw"] > main["kkr_damping_offset_removed"]
    assert main["damping_constant_offset_relative"] > 0.0
    # refining the panel mesh must not change the causality verdict
    coarser = leg["panel_sweeps"][0]
    assert coarser["panels"] < main["panels"]
    assert abs(coarser["kkr_damping_offset_removed"] - main["kkr_damping_offset_removed"]) < 5e-3


def test_deep_water_limit_is_measured_and_ordered(result: dict) -> None:
    leg = result["legs"]["asymptotics"]
    assert leg["worst_deviation_at_deepest"] <= anchors.GATE_ASYMPTOTIC
    ratios = [row["depth_ratio"] for row in leg["rows"]]
    assert ratios == sorted(ratios)
    deviations = [row["heave_added_mass_relative_deviation"] for row in leg["rows"]]
    # the added-mass deficit must shrink as the depth grows, not just end small
    assert deviations[-1] < deviations[0] / 10.0
    assert leg["added_mass_convergence_slope"] < -1.0


def test_refinement_leg_shows_a_decreasing_residual(result: dict) -> None:
    leg = result["legs"]["refinement"]
    errors = [row["energy_balance_error"] for row in leg["rows"]]
    assert len(errors) == len(anchors.REFINEMENT_PANELS)
    assert errors[-1] < errors[0]
    assert leg["monotone"] is True
    assert max(errors) <= anchors.GATE_DIFFRACTION_ENERGY


def test_hydrostatic_leg_covers_three_hulls(result: dict) -> None:
    leg = result["legs"]["hydrostatics"]
    assert len(leg["hulls"]) == 3
    for row in leg["hulls"]:
        assert row["cb_relative_error"] <= anchors.GATE_GEOMETRY
        assert row["volume"] > 0.0
        assert 0.4 < row["block_coefficient"] < 0.9
    published = {row["hull"]: row["block_coefficient_published"] for row in leg["hulls"]}
    assert published["KVLCC2 (idealised)"] == pytest.approx(0.8098, abs=1e-12)
    assert published["Series 60 (idealised)"] == pytest.approx(0.600, abs=1e-12)


def test_limitations_are_recorded(result: dict) -> None:
    assert len(result["limitations"]) >= 5
    joined = " ".join(result["limitations"]).lower()
    assert "not ship raos" in joined
    assert "withheld" in joined


# ---------------------------------------------------------------------------
# 2. method gates (independent recomputation)
# ---------------------------------------------------------------------------
def test_superellipse_area_coefficient_matches_the_closed_form() -> None:
    assert anchors.section_area_coefficient(2.0) == pytest.approx(math.pi / 4.0, rel=1e-12)
    for cm in (2.0 / 3.0, 0.90, 0.96, 0.998):
        exponent = anchors.exponent_for_area_coefficient(cm)
        assert anchors.section_area_coefficient(exponent) == pytest.approx(cm, rel=1e-9)


def test_section_meshes_are_closed_and_submerged() -> None:
    for hull in anchors.HULLS:
        mesh = anchors.hull_section(hull, 64)
        a = 0.5 * float(hull["B"]) / float(hull["T"])
        area = anchors.section_mesh_area(mesh)
        cm = area / (4.0 * a)  # 2a * 2b with b = 1 m
        assert cm == pytest.approx(float(hull["Cm"]), rel=2e-2)
        assert mesh.vertices[:, 1].max() < 0.0
        # the contour must close: the closing panel is not part of `lengths`
        assert np.all(mesh.lengths > 0.0)
        assert mesh.panel_count == 64


def test_dispersion_relation_holds_at_the_recorded_depths(result: dict) -> None:
    for case in result["legs"]["matrix"]["cases"]:
        if case["depth"] is None:
            continue
        k0 = finite_depth_wavenumber(case["nu"], case["depth"])
        assert k0 * math.tanh(k0 * case["depth"]) == pytest.approx(case["nu"], rel=1e-12)


def test_one_diffraction_case_reproduces_the_recorded_energy_residual(result: dict) -> None:
    """Cheap end-to-end: solve one matrix case again and compare."""
    case = next(
        case
        for case in result["legs"]["matrix"]["cases"]
        if case["hull"] == "Wigley" and case["depth_ratio"] == 1.5 and case["nu_times_draught"] == 1.0
    )
    hull = next(hull for hull in anchors.HULLS if hull["name"] == case["hull"])
    mesh = anchors.hull_section(hull, 48)
    fresh = solve_diffraction_indirect(mesh, case["nu"], case["depth"])
    # different panel count, same physics: the residual is small in both cases
    assert fresh.energy_balance_error <= anchors.GATE_DIFFRACTION_ENERGY
    assert abs(fresh.energy_balance_error - case["diffraction"]["energy_balance_error"]) < 5e-3


def test_flat_bottom_panels_assemble_a_finite_rankine_matrix() -> None:
    """Regression: collinear panels on a flat bottom used to divide by zero.

    ``_rankine_normal_derivative_matrix`` divides by the perpendicular distance
    from each panel to the target panel's line.  Three collinear bottom panels
    make that distance vanish; the limit is zero (such a panel is edge-on to the
    target normal) and must be taken instead of ``0/0``.
    """
    points = np.array(
        [
            [-1.0, -1.0],
            [-0.3, -1.0],
            [0.3, -1.0],
            [1.0, -1.0],
            [1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=float,
    )
    mesh = PanelMesh(points)
    matrix = _rankine_normal_derivative_matrix(mesh)
    assert matrix.shape == (mesh.panel_count, mesh.panel_count)
    assert np.all(np.isfinite(matrix))

def test_collinear_guard_is_inactive_on_a_convex_mesh() -> None:
    """The flat-bottom guard must not change anything for a smooth contour.

    ``_rankine_normal_derivative_matrix`` drops the term of a panel that lies on
    the collocation panel's line.  On the circular section no such pair exists,
    so the branch is a no-op there -- this is the control that keeps the fix from
    silently altering the already-verified circle results.
    """
    panels = 64
    angles = 2.0 * math.pi * np.arange(panels) / panels
    mesh = PanelMesh(np.column_stack((np.cos(angles), np.sin(angles) - 2.0)))
    tolerance = 1.0e-9 * float(np.max(mesh.lengths))
    for row in range(mesh.panel_count):
        point = mesh.centers[row]
        tangent = mesh.vectors / mesh.lengths[:, None]
        delta = mesh.vertices - point
        projection = np.einsum("pj,pj->p", delta, tangent)
        offset = delta - projection[:, None] * tangent
        height = np.sqrt(np.einsum("pj,pj->p", offset, offset))
        height[row] = 1.0  # the self panel is handled separately
        assert height.min() > tolerance


def test_case_artifacts_and_documentation_exist() -> None:
    for name in ("result.json", "gates.json", "report.md"):
        assert (CASE_DIR / name).is_file(), name
    figures = sorted((CASE_DIR / "figures").glob("*.png"))
    assert len(figures) >= 4
    for figure in figures:
        assert figure.stat().st_size > 10_000, figure
    doc = REPO_ROOT / "docs" / "verification" / "bem_finite_depth.md"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    assert "bem_finite_depth" in text
    assert "Kramers" in text
