"""Guards for the import-time Gauss rules in ``free_surface3d`` (m4789).

``free_surface3d`` builds three quadrature rules at import time.  ``numpy``'s
``leggauss``/``laggauss`` reach LAPACK through ``eigvalsh``, and some LAPACK
builds raise ``LinAlgError: Eigenvalues did not converge`` for the 40-point
Jacobi matrix (seen with numpy 2.0.1 on loongarch64).  Because the rules are
module-level constants, such a failure makes ``import tensorbem`` fail, which
turns every test in the suite red -- a whole-repo failure with a
quadrature-library cause.  The module now falls back to an explicit Jacobi
matrix fed to ``eigh``; these tests pin both halves of that claim: the fallback
is numerically identical to numpy's rule, and the module still imports when the
primary driver is broken.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss

from tensorbem import free_surface3d as fs


def test_legendre_fallback_reproduces_numpys_rule() -> None:
    nodes, weights = fs._legendre_gauss_golub_welsch(40)
    ref_nodes, ref_weights = leggauss(40)
    assert np.allclose(nodes, ref_nodes, rtol=0.0, atol=1e-13)
    assert np.allclose(weights, ref_weights, rtol=0.0, atol=1e-13)
    # Symmetric rule on [-1, 1]: total weight is exactly 2 and it integrates
    # every polynomial of degree < 2n without error.
    assert abs(weights.sum() - 2.0) < 1e-13
    for degree in (0, 1, 2, 39, 79):
        exact = 0.0 if degree % 2 else 2.0 / (degree + 1)
        assert abs(float(np.dot(weights, nodes**degree)) - exact) < 1e-12


def test_laguerre_fallback_is_an_exact_laguerre_rule() -> None:
    """The rule the import path needs: exact for every polynomial of degree < 2n."""
    nodes, weights = fs._laguerre_gauss_golub_welsch(40)
    assert np.all(np.diff(nodes) > 0.0)  # ascending, like the Legendre rule
    assert abs(weights.sum() - 1.0) < 1e-12  # int_0^inf exp(-x) dx = Gamma(1)
    for degree in range(0, 80):
        exact = float(math.factorial(degree))
        got = float(np.dot(weights, nodes**degree))
        assert abs(got - exact) <= 1e-10 * max(1.0, exact), degree
    # numpy's ``laggauss`` returns the nodes in descending order; the fallback
    # returns them ascending like the Legendre rule, so compare the sorted sets.
    ref_nodes, ref_weights = laggauss(40)
    order = np.argsort(ref_nodes)
    assert np.allclose(nodes, ref_nodes[order], rtol=0.0, atol=1e-9)
    assert np.allclose(weights, ref_weights[order], rtol=0.0, atol=1e-12)


def test_module_import_survives_a_broken_eigvalsh(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure mode is ``eigvalsh``; the module must survive it."""

    reference_nodes = leggauss(40)[0]  # measure before the driver is broken

    def refuse(matrix, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise np.linalg.LinAlgError("Eigenvalues did not converge")

    with monkeypatch.context() as patch:
        patch.setattr(np.linalg, "eigvalsh", refuse)
        reloaded = importlib.reload(fs)
        assert reloaded._LEG40_X.shape == (40,)
        assert reloaded._LEG40_W.shape == (40,)
        assert reloaded._LAG40_U.shape == (40,)
        assert np.allclose(reloaded._LEG40_X, reference_nodes, rtol=0.0, atol=1e-13)
    importlib.reload(fs)  # restore the module for the rest of the session


def test_rules_used_by_the_green_function_are_unchanged() -> None:
    """The constants the solver actually reads keep numpy's values bit-for-bit."""

    for name, ref in (
        ("_LEG20_X", leggauss(20)[0]),
        ("_LEG20_W", leggauss(20)[1]),
        ("_LEG40_X", leggauss(40)[0]),
        ("_LEG40_W", leggauss(40)[1]),
        ("_LAG40_U", laggauss(40)[0]),
        ("_LAG40_W", laggauss(40)[1]),
    ):
        assert np.array_equal(getattr(fs, name), ref), name
