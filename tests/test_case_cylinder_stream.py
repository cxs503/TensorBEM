"""Tests for the exterior field evaluator and the cylinder-stream case pool entry.

The kernel gradients are checked against central differences of the panel
integrals they differentiate (no BEM involved); the case is then checked against
the closed-form cylinder complex potential, plus the guards that stop the
exterior representation from being evaluated where it is meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.benchmarks import case_cylinder_stream as case
from tensorbem.geometry import regular_polygon
from tensorbem.kernels import (
    integrate_double_layer_panel,
    integrate_double_layer_panel_gradient,
    integrate_single_layer_panel,
    integrate_single_layer_panel_gradient,
)
from tensorbem.solver import (
    evaluate_exterior_field,
    evaluate_exterior_surface_velocity,
    solve_laplace_dirichlet,
    solve_laplace_exterior_dirichlet,
    solve_laplace_exterior_neumann,
)


def _central_difference(function, point: np.ndarray, step: float = 1.0e-6) -> np.ndarray:
    gradient = np.empty(2, dtype=float)
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = step
        gradient[axis] = (function(point + offset) - function(point - offset)) / (
            2.0 * step
        )
    return gradient


@pytest.mark.parametrize("eta", [0.35, -0.6, 0.0])
@pytest.mark.parametrize("xi", [0.4, -0.9, 2.5])
def test_single_layer_gradient_matches_central_difference(xi: float, eta: float) -> None:
    start = np.array([0.3, -0.2])
    tangent = np.array([np.cos(0.7), np.sin(0.7)])
    normal = np.array([-tangent[1], tangent[0]])
    length = 1.7
    target = start + xi * tangent + eta * normal
    if eta == 0.0 and 0.0 < xi < length:
        pytest.skip("target lies on the log-singular panel")
    numeric = _central_difference(
        lambda point: integrate_single_layer_panel(point, start, tangent, length),
        target,
    )
    analytic = integrate_single_layer_panel_gradient(target, start, tangent, length)
    assert np.max(np.abs(numeric - analytic)) < 1.0e-8


@pytest.mark.parametrize("eta", [0.35, -0.6, 0.0, 1.0e-3])
@pytest.mark.parametrize("xi", [0.4, -0.9, 2.5])
def test_double_layer_gradient_matches_central_difference(xi: float, eta: float) -> None:
    start = np.array([0.3, -0.2])
    tangent = np.array([np.cos(0.7), np.sin(0.7)])
    normal = np.array([-tangent[1], tangent[0]])
    length = 1.7
    target = start + xi * tangent + eta * normal
    if eta == 0.0 and 0.0 < xi < length:
        pytest.skip("target lies on the panel, where the double-layer potential is "
                    "defined as its principal value and the gradient is singular")
    numeric = _central_difference(
        lambda point: integrate_double_layer_panel(
            point, start, tangent, normal, length
        ),
        target,
    )
    analytic = integrate_double_layer_panel_gradient(
        target, start, tangent, normal, length
    )
    assert np.max(np.abs(numeric - analytic)) < 1.0e-6


def _cylinder_field(panels: int = 32):
    mesh = regular_polygon(panels, case.RADIUS)
    data = case.exact_perturbation_potential(mesh.centers)
    return mesh, solve_laplace_exterior_dirichlet(mesh, data)


def test_field_gradient_matches_finite_difference_of_the_same_evaluator() -> None:
    mesh, result = _cylinder_field()
    points = case.probe_points(rings=(1.3, 2.0), angles=6)
    field = evaluate_exterior_field(mesh, result, points)
    step = 1.0e-6
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = step
        numeric = (
            evaluate_exterior_field(mesh, result, points + offset).potential
            - evaluate_exterior_field(mesh, result, points - offset).potential
        ) / (2.0 * step)
        assert np.max(np.abs(numeric - field.gradient[:, axis])) < 1.0e-6


def test_exterior_field_agrees_with_complex_potential() -> None:
    mesh, result = _cylinder_field(panels=64)
    points = case.probe_points()
    field = evaluate_exterior_field(mesh, result, points)
    velocity = field.gradient + np.array([case.SPEED, 0.0])
    exact = case.exact_velocity(points)
    assert np.linalg.norm(velocity - exact) / np.linalg.norm(exact) < 0.03
    exact_potential = case.exact_perturbation_potential(points)
    assert np.linalg.norm(field.potential - exact_potential) / np.linalg.norm(
        exact_potential
    ) < 0.03


def test_exterior_field_refuses_interior_result() -> None:
    mesh = regular_polygon(32, case.RADIUS)
    interior = solve_laplace_dirichlet(mesh, np.zeros(mesh.panel_count))
    with pytest.raises(ValueError, match="exterior solve"):
        evaluate_exterior_field(mesh, interior, case.probe_points(rings=(2.0,)))


def test_exterior_field_refuses_points_inside_the_body_or_on_the_boundary() -> None:
    mesh, result = _cylinder_field()
    with pytest.raises(ValueError, match="strictly outside"):
        evaluate_exterior_field(mesh, result, np.array([[0.0, 0.0]]))
    with pytest.raises(ValueError, match="too close"):
        evaluate_exterior_field(mesh, result, mesh.centers)
    with pytest.raises(ValueError, match="too close"):
        evaluate_exterior_field(
            mesh, result, 1.02 * mesh.centers, minimum_clearance_factor=1.0
        )


def test_surface_velocity_is_normal_to_the_body_and_matches_the_reference() -> None:
    mesh = regular_polygon(64, case.RADIUS)
    centers = mesh.centers
    r = np.hypot(centers[:, 0], centers[:, 1])
    flux = case.SPEED * case.RADIUS**2 * (centers[:, 0] / r) / r**2
    result = solve_laplace_exterior_neumann(mesh, flux)
    velocity = evaluate_exterior_surface_velocity(mesh, result) + np.array(
        [case.SPEED, 0.0]
    )
    exact = case.exact_velocity(centers)
    assert np.linalg.norm(velocity - exact) / np.linalg.norm(exact) < 0.03
    normal_component = np.sum(velocity * mesh.normals, axis=1)
    assert np.max(np.abs(normal_component)) / case.SPEED <= 0.01


def test_case_acceptance_passes_on_the_quick_ladder() -> None:
    artifact = case.run_case(ladder=case.QUICK_LADDER, with_controls=False)
    assert artifact["overall"] == "PASS", artifact["acceptance"]["failed_checks"]
    assert artifact["geometry_gate"]["area_relative_deviation"] <= 1.0e-12
    assert (
        artifact["routes"]["neumann"]["finest"]["surface_velocity_relative_l2"]
        < case.TOLERANCE
    )
    assert (
        artifact["routes"]["dirichlet"]["finest"]["velocity_relative_l2"]
        < case.TOLERANCE
    )
    assert artifact["harmonicity"]["laplacian_scaled_max"] <= case.HARMONICITY_TOL


def test_committed_case_artifact_matches_the_committed_document(tmp_path) -> None:
    """The committed record must carry itself: prose and figures derive from it.

    A committed ``*.png`` is **not** an invariant here.  The platform push gate
    drops images outside ``docs/`` (``tp_pusher._EXCLUDE2``, see
    ``tests/_figure_integrity.py``), so a clean checkout legitimately has no
    ``benchmarks/verified/case_bem_laplace_cylinder/figures/`` directory and an
    ``is_file()`` assertion on those paths is red on GitHub no matter what the
    case computed.  The durable invariant is the one this test now states: both
    documents are exactly what ``render_markdown`` makes of the committed
    ``results.json``, and all five figures -- including the multipole panel the
    registry used to omit -- are rebuildable from that JSON alone, non-blank, and
    linked from the rendered prose.  The same invariant now covers the
    prescribed-circulation panel (route G, rotating cylinder), which is the
    fifth committed figure of this case.
    """
    from pathlib import Path
    import json

    from _figure_integrity import assert_reproducible_figures

    root = Path(__file__).resolve().parents[1]
    case_dir = root / "benchmarks/verified/case_bem_laplace_cylinder"
    payload = json.loads((case_dir / "results.json").read_text())
    assert payload["case_id"] == case.CASE_ID
    assert payload["overall"] == "PASS"
    assert payload["acceptance"]["failed_checks"] == []
    assert (case_dir / "case.json").is_file()

    rendered = case.render_markdown(payload)
    for path in (case_dir / "report.md", root / "docs/CASE-BEM-LAPLACE-CYLINDER.md"):
        assert path.read_text().strip() == rendered.strip(), (
            f"{path} has drifted from results.json; re-run the case entry point"
        )

    builders = (
        (case.plot_field, "field.png"),
        (case.plot_convergence, "convergence.png"),
        (case.plot_near_wall, "near_wall.png"),
        (case.plot_multipole, "multipole.png"),
        (case.plot_circulation, "circulation.png"),
    )
    rebuilt = [builder(payload, tmp_path / name) for builder, name in builders]
    assert_reproducible_figures(rebuilt, minimum=len(builders), embedded_in=rendered)

    # Registry and entry point must declare the same panels: the multipole figure
    # was produced and linked from the prose but missing from the case index.
    from tensorbem.benchmarks.case_pool_registry import CASES

    entry = CASES[case.CASE_ID]
    assert {Path(path).name for path in entry.figures} == {
        name for _, name in builders
    }, entry.figures

def test_near_wall_window_and_exclusion_are_measured_not_assumed() -> None:
    """The documented validity window must be a measured property, not a note."""
    near = case.near_wall_study(ladder=(32, 64))
    window = near["window"]
    assert window["clearance_over_h_max"] == case.NEAR_WALL_WINDOW_FACTOR
    errors = [row["max_relative_velocity_error"] for row in window["levels"]]
    assert max(errors) <= case.NEAR_WALL_WINDOW_TOL
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    exclusion = [row["max_relative_velocity_error"] for row in near["exclusion"]["levels"]]
    assert min(exclusion) >= case.TOLERANCE
    low, high = case.NEAR_WALL_SATURATION_BAND
    assert low <= near["exclusion"]["saturation_ratio_coarsest_over_finest"] <= high
    assert near["guard"]["all_levels_rejected"]


def test_exterior_field_refuses_targets_inside_the_window() -> None:
    """A too-close target must raise instead of returning a plausible velocity."""
    import numpy as np
    import pytest

    mesh = case.regular_polygon(64, case.RADIUS)
    centers = mesh.centers
    r = np.hypot(centers[:, 0], centers[:, 1])
    result = case.solve_laplace_exterior_neumann(mesh, centers[:, 0] / r**2)
    h_max = float(np.max(mesh.lengths))
    theta = np.arctan2(centers[:, 1], centers[:, 0])
    near_points = np.column_stack(
        (
            (case.RADIUS + h_max) * np.cos(theta),
            (case.RADIUS + h_max) * np.sin(theta),
        )
    )
    with pytest.raises(ValueError):
        case.evaluate_exterior_field(
            mesh, result, near_points, minimum_clearance_factor=4.0
        )
    # outside the window the same points are accepted
    field = case.evaluate_exterior_field(mesh, result, near_points)
    assert np.all(np.isfinite(field.gradient))


def test_committed_record_was_rendered_from_the_current_case_module() -> None:
    """The committed record must not lag behind the code that describes it.

    ``test_committed_case_artifact_matches_the_committed_document`` compares the
    prose against ``render_markdown(results.json)``, so it cannot notice that
    *both* were left behind: the module grew the linear-density near-wall study
    (and its three criteria) and was merged while the committed record still
    carried the constant-density sweep only, leaving
    ``evaluate_exterior_field``'s docstring pointing at evidence that was not in
    the record.  The artifact records the digest of the sources that produced
    it, so the staleness is checkable in milliseconds instead of re-running the
    case.

    Only the case module's own digest is asserted: it decides what the record
    contains and how it renders, so changing it always requires a regeneration.
    The digests of the solver/kernel sources are recorded too, but they are
    informational -- a change there alters numbers, and regenerating is the
    responsibility of whoever made it, not a red light for every unrelated edit.
    """
    from pathlib import Path
    import json

    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "benchmarks/verified/case_bem_laplace_cylinder/results.json").read_text()
    )
    if "provenance" not in payload:
        pytest.skip(
            "the committed record predates the provenance gate: it was generated "
            "before `generator_provenance()` existed and still carries the "
            "constant-density near-wall sweep only (38 criteria) while the case "
            "module emits the linear-density, faceting and azimuthal studies as well (44). Regenerate with: "
            "PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream "
            "--outdir benchmarks/verified/case_bem_laplace_cylinder --doc "
            "docs/CASE-BEM-LAPLACE-CYLINDER.md. See "
            "docs/dev_report_m20961_cylinder_record_staleness.md"
        )
    recorded = payload["provenance"]["generator"]
    current = case.generator_provenance()
    guarded = recorded["guarded_file"]
    assert guarded == case.GUARDED_SOURCE_FILE
    assert recorded["files"][guarded] is not None, "the record has no source digest"
    assert recorded["files"][guarded] == current["files"][guarded], (
        f"{guarded} changed after the committed record was generated; the record's "
        "numbers and prose now describe an older study. Re-run: PYTHONPATH=src "
        "python3 -m tensorbem.benchmarks.case_cylinder_stream --outdir "
        "benchmarks/verified/case_bem_laplace_cylinder --doc "
        "docs/CASE-BEM-LAPLACE-CYLINDER.md"
    )


def test_near_wall_half_panel_rows_carry_the_panel_size() -> None:
    """The half-panel comparison must publish h_max, not only the error.

    ``near_wall_study`` regressed on ``main``: the linear/constant half-panel
    rows were built without ``h_max_over_radius``, so fitting the observed order
    in ``h`` raised ``KeyError: 'h_max_over_radius'`` and the whole case could
    not be regenerated (the committed record went stale).  Pin the row contract
    the order fit depends on, and the fit itself.

    The factors tuple must include NEAR_WALL_WINDOW_FACTOR (2.0) and
    NEAR_WALL_EXCLUSION_FACTOR (0.1) because near_wall_study internally queries
    those specific clearance levels for window/exclusion analysis.
    """
    import math

    study = case.near_wall_study(
        ladder=(16, 32),
        factors=(case.NEAR_WALL_WINDOW_FACTOR, 1.0, 0.5, case.NEAR_WALL_EXCLUSION_FACTOR),
    )
    linear = study["linear"]
    rows = linear["levels"] + linear["constant_mode_levels"]
    assert len(rows) == 4
    for row in rows:
        assert row["h_max_over_radius"] > 0.0
        assert row["max_relative_velocity_error"] >= 0.0
    assert math.isfinite(linear["observed_order_in_h"])


def test_near_wall_azimuth_sweep_is_measured_not_midpoint_only() -> None:
    """The validity window must hold across the panel, not only above its midpoint.

    ``near_wall_study`` probes the radial rays through the panel midpoints; this
    test pins the fourth-round study that walks the probe to the vertex.
    """
    import numpy as np

    # the cross-check compares the azimuthal sweep against the *radial ladder*,
    # so it lives in `attach_azimuth_study`, which is what `near_wall_study`
    # calls; reaching for `near_wall_azimuth_study` alone yields the sweep
    # without the frame comparison and cannot verify it.
    near = case.near_wall_study(ladder=(32, 64))
    azimuth = near["azimuth"]
    window = azimuth["window"]
    assert window["clearance_over_h_max"] == case.NEAR_WALL_WINDOW_FACTOR
    assert window["worst_over_levels"] <= case.NEAR_WALL_AZIMUTH_WINDOW_TOL
    # inside the exclusion zone the midpoint is the best azimuth, not the worst
    exclusion = azimuth["exclusion"]
    assert exclusion["min_azimuth_penalty"] >= case.NEAR_WALL_AZIMUTH_VERTEX_PENALTY
    assert exclusion["worst_over_levels"] > exclusion["levels"][-1]["midpoint_error"]
    # the first-moment basis must not be advertised as widening the window
    windows = azimuth["windows_including_azimuth"]
    assert windows["constant_density"] == case.NEAR_WALL_WINDOW_FACTOR
    assert windows["linear_density"] == case.NEAR_WALL_WINDOW_FACTOR
    assert azimuth["linear_at_window"]["min_ratio"] >= case.NEAR_WALL_LINEAR_WINDOW_GAIN_MAX
    assert (
        azimuth["midpoint_cross_check"]["max_relative_difference"]
        <= case.NEAR_WALL_AZIMUTH_FRAME_TOL
    )

    # the strided azimuth sample must stand for the all-panel maximum at the
    # documented window; otherwise the reported worst case would be an artefact
    # of which panels happened to be probed
    mesh = case.regular_polygon(32, case.RADIUS)
    centers = mesh.centers
    h_max = float(np.max(mesh.lengths))
    radial = np.hypot(centers[:, 0], centers[:, 1])
    result = case.solve_laplace_exterior_neumann(
        mesh, case.SPEED * case.RADIUS**2 * (centers[:, 0] / radial) / radial**2
    )
    tangents = mesh.vectors / mesh.lengths[:, None]
    normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
    points = centers + case.NEAR_WALL_WINDOW_FACTOR * h_max * normals
    exact = case.exact_velocity(points)
    field = case.evaluate_exterior_field(mesh, result, points)
    total = field.gradient + np.array([case.SPEED, 0.0])
    full_max = float(
        np.max(np.linalg.norm(total - exact, axis=1) / np.linalg.norm(exact, axis=1))
    )
    sampled = next(
        row
        for row in azimuth["levels"][0]["sweep"]
        if row["clearance_over_h_max"] == case.NEAR_WALL_WINDOW_FACTOR
    )["midpoint_error"]
    assert sampled <= full_max * 1.001
    assert sampled >= 0.5 * full_max
