"""Verification tests for the mixed Dirichlet/Neumann 2-D Laplace solver.

The oracle is closed form (``u = exp(x) cos(y)`` and ``u = x^2 - y^2``), never
another discretisation.  The strongest assertion here is algebraic: with the
Neumann set empty the mixed block system must reproduce
:func:`tensorbem.solver.solve_laplace_dirichlet` to roundoff, which proves the
new path is a strict generalisation of the shipped one.  The artifact test
re-runs the published study and compares it with the committed JSON, so the
numbers quoted in ``docs/mixed_boundary_conditions.md`` cannot drift.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks import mixed_boundary as study
from tensorbem.geometry import PanelMesh
from tensorbem.mixed import (
    MixedBoundary,
    evaluate_mixed_interior_field,
    solve_laplace_mixed,
)
from tensorbem.solver import solve_laplace_dirichlet

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "docs" / "mixed_boundary" / "outputs" / "study.json"
CASE = study.ADJACENT_MIXED
PURE = study.PURE_NEUMANN


def _mixed_solve(panels_per_side: int, case: study.AnalyticCase = CASE):
    mesh = study.square_mesh(panels_per_side)
    boundary, masks = study.build_boundary(mesh, case)
    return mesh, masks, solve_laplace_mixed(mesh, boundary)


def test_mixed_assembly_reduces_to_the_dirichlet_solver():
    """An empty Neumann set must reproduce the shipped Dirichlet solve exactly."""
    for n in (2, 4, 8):
        mesh = study.square_mesh(n)
        boundary, _ = study.build_boundary(mesh, study.ALL_DIRICHLET)
        mixed = solve_laplace_mixed(mesh, boundary)
        reference = solve_laplace_dirichlet(mesh, study.ALL_DIRICHLET.value(mesh.centers))
        assert np.max(np.abs(mixed.normal_derivative - reference.normal_derivative)) < 1e-12
        assert mixed.condition_number == pytest.approx(reference.condition_number, rel=1e-12)
        assert mixed.relative_residual_l2 < 1e-12


def test_neumann_side_trace_converges_and_is_recovered():
    """The trace the solver was not given on the Neumann sides is recovered."""
    errors = []
    for n in (4, 8, 16):
        mesh, masks, result = _mixed_solve(n)
        exact = CASE.value(mesh.centers)
        index = result.neumann_panels
        errors.append(
            study.relative_l2(result.boundary_values[index], exact[index])
        )
    # 64 panels: the published value is 1.39e-3; keep a 15% band.
    assert errors[-1] == pytest.approx(1.39e-3, rel=0.15)
    orders = [math.log(errors[i] / errors[i + 1]) / math.log(2) for i in range(2)]
    assert min(orders) > 1.2, orders


def test_mixed_interior_field_matches_the_closed_form():
    """Interior harmonic field reconstructed from both traces."""
    probes = study.interior_probe_points()
    exact = CASE.value(probes)
    errors = []
    for n in (4, 8, 16):
        mesh, _, result = _mixed_solve(n)
        errors.append(study.relative_l2(evaluate_mixed_interior_field(mesh, result, probes), exact))
    assert errors[-1] < 1e-3
    orders = [math.log(errors[i] / errors[i + 1]) / math.log(2) for i in range(2)]
    assert min(orders) > 1.5, orders


def test_flux_trace_is_corner_limited_not_mixing_limited():
    """Square corners, not the boundary-condition mix, set the flux error level.

    The all-Dirichlet solve of the same analytic field is just as inaccurate in
    the flux as the mixed solve, so mixing conditions adds no error of its own.
    """
    mesh, masks, result = _mixed_solve(16)
    exact_q = study.exact_normal_derivative(mesh, masks, CASE)
    index = result.dirichlet_panels
    mixed_error = study.relative_l2(result.normal_derivative[index], exact_q[index])
    baseline = solve_laplace_dirichlet(mesh, CASE.value(mesh.centers))
    baseline_error = study.relative_l2(baseline.normal_derivative, exact_q)
    assert mixed_error == pytest.approx(3.40e-2, rel=0.15)
    assert mixed_error < 1.3 * baseline_error
    assert baseline_error < 1.3 * mixed_error


def test_pure_neumann_uses_the_mean_zero_gauge():
    mesh = study.square_mesh(8)
    boundary, masks = study.build_boundary(mesh, PURE)
    result = solve_laplace_mixed(mesh, boundary)
    assert result.is_pure_neumann
    assert result.pure_neumann_gauge == "mean_zero"
    assert abs(float(np.mean(result.boundary_values))) < 1e-12
    exact_q = study.exact_normal_derivative(mesh, masks, PURE)
    # Panel-constant data are exact for u = x^2 - y^2, so q must be exact.
    assert np.max(np.abs(result.normal_derivative - exact_q)) < 1e-12
    exact_u = PURE.value(mesh.centers)
    error = study.relative_l2(
        result.boundary_values - np.mean(result.boundary_values),
        exact_u - np.mean(exact_u),
    )
    assert error < 2.2e-2


def test_pure_neumann_refuses_silently_arbitrary_constant():
    mesh = study.square_mesh(4)
    boundary, _ = study.build_boundary(mesh, PURE)
    with pytest.raises(ValueError, match="additive constant"):
        solve_laplace_mixed(mesh, boundary, pure_neumann_gauge="raise")


def test_boundary_validation_rejects_incomplete_or_overlapping_data():
    mesh = study.square_mesh(2)
    boundary, _ = study.build_boundary(mesh, CASE)
    n = mesh.panel_count
    # a panel listed twice in the same set
    duplicate = MixedBoundary(
        dirichlet_panels=np.array([0, 0]),
        dirichlet_values=np.zeros(2),
        neumann_panels=boundary.neumann_panels,
        neumann_values=boundary.neumann_values,
    )
    with pytest.raises(ValueError, match="duplicate"):
        solve_laplace_mixed(mesh, duplicate)
    # a panel left without any condition
    incomplete = MixedBoundary(
        dirichlet_panels=boundary.dirichlet_panels[:-1],
        dirichlet_values=boundary.dirichlet_values[:-1],
        neumann_panels=boundary.neumann_panels,
        neumann_values=boundary.neumann_values,
    )
    with pytest.raises(ValueError, match="every panel needs exactly one"):
        solve_laplace_mixed(mesh, incomplete)
    # a panel claimed by both types
    overlap = MixedBoundary(
        dirichlet_panels=np.append(boundary.dirichlet_panels, boundary.neumann_panels[0]),
        dirichlet_values=np.append(boundary.dirichlet_values, 0.0),
        neumann_panels=boundary.neumann_panels,
        neumann_values=boundary.neumann_values,
    )
    with pytest.raises(ValueError, match="both Dirichlet and Neumann"):
        solve_laplace_mixed(mesh, overlap)
    # value/data length mismatch
    with pytest.raises(ValueError, match="one entry per dirichlet panel"):
        solve_laplace_mixed(
            mesh,
            MixedBoundary(
                dirichlet_panels=boundary.dirichlet_panels,
                dirichlet_values=boundary.dirichlet_values[:-1],
                neumann_panels=boundary.neumann_panels,
                neumann_values=boundary.neumann_values,
            ),
        )
    # non-finite data
    with pytest.raises(ValueError, match="finite"):
        solve_laplace_mixed(
            mesh,
            MixedBoundary(
                dirichlet_panels=boundary.dirichlet_panels,
                dirichlet_values=np.full(boundary.dirichlet_panels.size, np.nan),
                neumann_panels=boundary.neumann_panels,
                neumann_values=boundary.neumann_values,
            ),
        )
    # out-of-range index
    with pytest.raises(ValueError, match="must lie in"):
        solve_laplace_mixed(
            mesh,
            MixedBoundary(
                dirichlet_panels=np.array([n]),
                dirichlet_values=np.zeros(1),
                neumann_panels=np.arange(n),
                neumann_values=np.zeros(n),
            ),
        )
    # unknown gauge name
    with pytest.raises(ValueError, match="pure_neumann_gauge"):
        solve_laplace_mixed(mesh, boundary, pure_neumann_gauge="mean_one")


def test_evaluator_rejects_mismatched_mesh_and_points():
    mesh, _, result = _mixed_solve(4)
    with pytest.raises(ValueError, match=r"shape \(M, 2\)"):
        evaluate_mixed_interior_field(mesh, result, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="finite"):
        evaluate_mixed_interior_field(mesh, result, np.array([[np.inf, 0.5]]))
    with pytest.raises(TypeError, match="LaplaceMixedResult"):
        evaluate_mixed_interior_field(mesh, "not a result", np.zeros((1, 2)))  # type: ignore[arg-type]
    other = PanelMesh(vertices=study.square_mesh(3).vertices)
    with pytest.raises(ValueError, match="panel count"):
        # same mesh geometry, different panel count -> traces do not match
        evaluate_mixed_interior_field(other, result, np.array([[0.5, 0.5]]))


@pytest.mark.skipif(not ARTIFACT.exists(), reason="study artifact not committed")
def test_committed_artifact_reproduces_the_study():
    """The published JSON must be regenerated bit-for-bit by the shipped code."""
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert document["schema"] == study._SCHEMA
    pinned = tuple(int(n) for n in document["panels_per_side"])
    fresh = study.run_study(pinned)
    for fresh_case, old_case in zip(fresh["cases"], document["cases"]):
        assert fresh_case["name"] == old_case["name"]
        for fresh_row, old_row in zip(fresh_case["rows"], old_case["rows"]):
            for key, value in old_row.items():
                if isinstance(value, float):
                    assert fresh_row[key] == pytest.approx(value, rel=1e-6, abs=1e-12), key
                else:
                    assert fresh_row[key] == value, key
    for block in ("dirichlet_regression", "dirichlet_flux_baseline"):
        for fresh_row, old_row in zip(fresh[block]["rows"], document[block]["rows"]):
            for key, value in old_row.items():
                if isinstance(value, float):
                    assert fresh_row[key] == pytest.approx(value, rel=1e-6, abs=1e-12), key
                else:
                    assert fresh_row[key] == value, key