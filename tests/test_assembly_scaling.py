"""Evidence tests for the batched panel self-integrals (GAP-BEM-ASSEMBLY-SCALING).

The claim under test is narrow and checkable: the batched polar-sector rule must
return the *same numbers* as the per-panel loop (the quadrature is not being
replaced), it must actually be the path the solvers use, and the published
artifacts must agree with the verdict printed in the document.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorbem import finite_depth_solver3d, free_surface_solver3d, solver3d
from tensorbem.benchmarks.assembly_scaling import ARTIFACT_ROOT, RESULT_FILE
from tensorbem.geometry3d import hemisphere_hull_mesh, icosphere
from tensorbem.kernels3d import (
    _polar_sectors,
    _polar_sectors_batch,
    gradient_single_layer_triangle_self,
    gradient_single_layer_triangle_self_batch,
    integrate_single_layer_triangle_self,
    integrate_single_layer_triangle_self_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "assembly_scaling.md"


def _sliver_triangles() -> np.ndarray:
    """A deterministic mix of well-shaped and nearly degenerate triangles."""
    base = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0e-3, 0.0]],  # sliver
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0e-6, 0.0]],  # thinner sliver
            [[1.0, 2.0, 3.0], [1.5, 2.2, 3.1], [1.1, 2.9, 2.4]],
        ],
        dtype=float,
    )
    return base


@pytest.mark.parametrize("order", [2, 8, 16, 64])
def test_self_integral_batch_matches_scalar(order: int) -> None:
    triangles = _sliver_triangles()
    scalar = np.array(
        [integrate_single_layer_triangle_self(tri, order=order) for tri in triangles]
    )
    batched = integrate_single_layer_triangle_self_batch(triangles, order=order)
    assert batched.shape == scalar.shape
    assert np.all(np.isfinite(batched))
    assert np.allclose(batched, scalar, rtol=1e-12, atol=0.0)


def test_self_integral_batch_chunking_is_invisible() -> None:
    triangles = icosphere(1).vertices[icosphere(1).faces]
    whole = integrate_single_layer_triangle_self_batch(triangles, panel_chunk=10_000)
    chunked = integrate_single_layer_triangle_self_batch(triangles, panel_chunk=7)
    assert np.array_equal(whole, chunked)


def test_gradient_self_batch_matches_scalar() -> None:
    triangles = _sliver_triangles()
    scalar = np.array(
        [gradient_single_layer_triangle_self(tri) for tri in triangles]
    )
    batched = gradient_single_layer_triangle_self_batch(triangles)
    assert batched.shape == scalar.shape
    norm = float(np.max(np.linalg.norm(scalar, axis=1)))
    assert np.max(np.linalg.norm(batched - scalar, axis=1)) <= 1e-12 * norm


def test_polar_sector_batch_returns_elementwise_identical_data() -> None:
    triangle = _sliver_triangles()[3]
    theta, weights, extent, direction = _polar_sectors(triangle, 16)
    b_weights, b_extent, b_direction = _polar_sectors_batch(
        triangle[None, :, :], 16, with_direction=True
    )
    assert np.allclose(b_weights[0], weights, rtol=0.0, atol=1e-15)
    assert np.allclose(b_extent[0], extent, rtol=1e-14, atol=1e-15)
    assert np.allclose(b_direction[0], direction, rtol=0.0, atol=1e-15)


def test_batch_validation_rejects_bad_input() -> None:
    good = _sliver_triangles()[:1]
    with pytest.raises(ValueError):
        integrate_single_layer_triangle_self_batch(np.zeros((3, 3)))  # 2-D
    degenerate = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    with pytest.raises(ValueError):
        integrate_single_layer_triangle_self_batch(degenerate)
    not_finite = good.copy()
    not_finite[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        integrate_single_layer_triangle_self_batch(not_finite)
    empty = np.zeros((0, 3, 3))
    with pytest.raises(ValueError):
        integrate_single_layer_triangle_self_batch(empty)
    for bad_chunk in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            integrate_single_layer_triangle_self_batch(good, panel_chunk=bad_chunk)


def test_assemble_surface_matrices_uses_the_batched_path(monkeypatch) -> None:
    """Structural proof: the scalar per-panel routine must not be called at all."""
    def _boom(*args, **kwargs):  # pragma: no cover - only on regression
        raise AssertionError("per-panel self integral used by assemble_surface_matrices")

    monkeypatch.setattr(solver3d, "integrate_single_layer_triangle_self", _boom)
    mesh = icosphere(1)
    single, double = solver3d.assemble_surface_matrices(mesh, quadrature_order=3)
    assert np.all(np.diag(single) > 0.0)
    assert np.allclose(np.sum(double, axis=1), -0.5, atol=5e-2)


def test_free_surface_and_finite_depth_use_the_batched_path(monkeypatch) -> None:
    def _boom(*args, **kwargs):  # pragma: no cover - only on regression
        raise AssertionError("per-panel self integral used by the Kelvin assembly")

    mesh = hemisphere_hull_mesh(1)
    monkeypatch.setattr(
        free_surface_solver3d, "integrate_single_layer_triangle_self", _boom
    )
    ops = free_surface_solver3d.assemble_free_surface_operators_3d(
        mesh, 1.0, chief_points=None
    )
    assert np.all(np.diag(ops.single_layer).real > 0.0)

    monkeypatch.setattr(
        finite_depth_solver3d, "integrate_single_layer_triangle_self", _boom
    )
    single, _ = finite_depth_solver3d._rankine_operators(
        mesh.centroids, mesh, self_interaction=True, order=3
    )
    assert np.all(np.diag(single).real > 0.0)


def test_solver_gradient_self_term_is_batched(monkeypatch) -> None:
    def _boom(*args, **kwargs):  # pragma: no cover - only on regression
        raise AssertionError("per-panel self gradient used by the surface velocity")

    mesh = icosphere(1)
    result = solver3d.solve_laplace_surface(
        mesh, np.ones(mesh.panel_count), domain="exterior", quadrature_order=3
    )
    monkeypatch.setattr(
        solver3d, "gradient_single_layer_triangle_self", _boom
    )
    values = solver3d.evaluate_representation_surface_velocity_at_panels(
        mesh, result, quadrature_order=3
    )
    assert values.shape == (mesh.panel_count, 3)
    assert np.all(np.isfinite(values))


def test_published_artifacts_match_the_document() -> None:
    result_path = ARTIFACT_ROOT / RESULT_FILE
    assert result_path.is_file(), "run python -m tensorbem.benchmarks.assembly_scaling --write"
    assert DOC_PATH.is_file()
    study = json.loads(result_path.read_text())
    doc = DOC_PATH.read_text()
    verdict = study["verdict"]

    assert verdict["speedup_target"] == 3.0
    assert verdict["deviation_target"] == 1e-10
    assert verdict["memory_target"] == 0.10
    # The verdict word in the document is read back from the numbers, not typed.
    assert f"**{'PASS' if verdict['pass'] else 'FAIL'}**" in doc
    for row in study["assembly"]:
        assert f"| {row['panels']} |" in doc
        assert f"{row['assembly_speedup']:.2f}x" in doc
        assert f"{row['max_abs_operator_deviation']:.2e}" in doc
    # A PASS must be earned by all three criteria at once.
    if verdict["pass"]:
        assert verdict["speedup_at_finest"] >= verdict["speedup_target"]
        assert verdict["deviation_worst"] < verdict["deviation_target"]
        assert verdict["peak_growth_worst"] <= verdict["memory_target"]
    assert "zero hand-copied digits" in doc