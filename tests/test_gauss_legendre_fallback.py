"""The Gauss-Legendre rule must survive a LAPACK ``eigvalsh`` failure.

Motivation (field failure): the round-2 capytaine cross-check
(``tensorbem.benchmarks.capytaine_crosscheck_r2``) died on an HPC node with

    numpy.linalg.LinAlgError: Eigenvalues did not converge

raised inside ``numpy.polynomial.legendre.leggauss(64)``, i.e. in the
Jacobi-matrix eigenvalue route, not in TensorBEM's own arithmetic.  The order
(64) is the ordinary default of the self-panel polar quadrature, so the whole
3-D BEM solve became hostage to the linear-algebra backend.

``tensorbem.kernels3d.leggauss`` keeps the NumPy rule when it works (numbers do
not move) and falls back to Newton iteration on ``P_n`` when it does not.  These
tests pin both halves of that statement:

1. the Newton rule agrees with NumPy's rule to round-off over the orders the
   code actually uses;
2. the wrapper really does fall back when NumPy raises;
3. the *quantities built on the rule* -- the Duffy weights, the polar sector
   weights and the single-layer self-panel integral -- are unchanged by the
   fallback, so a backend failure cannot silently change a BEM result.
"""

from __future__ import annotations

import numpy as np
import pytest

import tensorbem.kernels3d as kernels3d


ORDERS = (2, 3, 5, 8, 16, 32, 64, 128)


@pytest.mark.parametrize("order", ORDERS)
def test_newton_rule_matches_numpy(order: int) -> None:
    nodes_np, weights_np = np.polynomial.legendre.leggauss(order)
    nodes, weights = kernels3d._legendre_newton(order)
    assert np.max(np.abs(nodes - nodes_np)) < 1.0e-12
    assert np.max(np.abs(weights - weights_np)) < 1.0e-12
    assert weights.sum() == pytest.approx(2.0, abs=1.0e-13)
    assert np.all(np.diff(nodes) > 0.0), "nodes must be increasing"


def test_wrapper_falls_back_when_numpy_raises(monkeypatch) -> None:
    def boom(order):  # pragma: no cover - exercised through the wrapper
        raise np.linalg.LinAlgError("Eigenvalues did not converge")

    monkeypatch.setattr(kernels3d, "_numpy_leggauss", boom)
    nodes, weights = kernels3d.leggauss(64)
    expected_nodes, expected_weights = kernels3d._legendre_newton(64)
    assert np.array_equal(nodes, expected_nodes)
    assert np.array_equal(weights, expected_weights)


def _triangle() -> np.ndarray:
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.3, 0.9, 0.0]], dtype=float
    )


def test_quadrature_quantities_are_unchanged_by_the_fallback(monkeypatch) -> None:
    triangle = _triangle()
    batch = triangle[None, :, :]

    reference = {
        "duffy": kernels3d.duffy_quadrature(triangle, order=8),
        "duffy_batch": kernels3d.duffy_quadrature_batch(
            batch[:, :1, :], batch[:, 1:2, :], batch[:, 2:, :], order=8
        ),
        "self": kernels3d.integrate_single_layer_triangle_self(triangle, order=16),
        "polar": kernels3d._polar_sectors(triangle, order=16)[:2],
        "polar_batch": kernels3d._polar_sectors_batch(batch, order=16)[:2],
    }

    def boom(order):
        raise np.linalg.LinAlgError("Eigenvalues did not converge")

    monkeypatch.setattr(kernels3d, "_numpy_leggauss", boom)

    fallback = {
        "duffy": kernels3d.duffy_quadrature(triangle, order=8),
        "duffy_batch": kernels3d.duffy_quadrature_batch(
            batch[:, :1, :], batch[:, 1:2, :], batch[:, 2:, :], order=8
        ),
        "self": kernels3d.integrate_single_layer_triangle_self(triangle, order=16),
        "polar": kernels3d._polar_sectors(triangle, order=16)[:2],
        "polar_batch": kernels3d._polar_sectors_batch(batch, order=16)[:2],
    }

    for key in reference:
        if isinstance(reference[key], tuple):
            for a, b in zip(reference[key], fallback[key]):
                assert np.allclose(a, b, rtol=0.0, atol=1.0e-13), key
        else:
            assert reference[key] == pytest.approx(fallback[key], rel=0.0, abs=1.0e-13), key

    # the self-panel integral of 1/r over a flat triangle must stay positive and
    # finite -- the "did not converge" failure used to abort the solve instead
    assert np.isfinite(fallback["self"]) and fallback["self"] > 0.0