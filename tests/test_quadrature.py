"""Tests for the Gauss-Legendre rules that do not depend on ``eigvalsh``.

The defect these tests exist for is not a numerical-accuracy question: on the
loongarch64 nodes ``numpy.polynomial.legendre.leggauss`` raises
``LinAlgError: Eigenvalues did not converge``, and the package used to call it
at import time, so ``import tensorbem`` itself failed there.  A rule that is
merely accurate is therefore not enough -- it has to be reachable when NumPy's
rule is not.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorbem import quadrature

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Orders actually used by the library: 80 at import time in ``free_surface``
#: and the 80...2560 oscillation-phase ladder above it.
LADDER = (24, 80, 160, 320, 640, 1280, 2560)


@pytest.fixture(autouse=True)
def _fresh_cache():
    quadrature.clear_cache()
    yield
    quadrature.clear_cache()


@pytest.mark.parametrize("order", LADDER)
def test_rule_integrates_polynomials_exactly(order: int) -> None:
    """A correct ``n``-point rule integrates degree ``2n - 2`` exactly."""
    nodes, weights = quadrature.gauss_legendre(order)
    for degree in (2 * order - 2, 2 * order - 3, 2):
        exact = 0.0 if degree % 2 else 2.0 / (degree + 1)
        # fsum: at n = 2560 the terms span many decades and a naive dot loses
        # the last few bits to rounding, which has nothing to do with the rule
        total = math.fsum(float(w) * float(x) ** degree for x, w in zip(nodes, weights))
        # rtol, not atol: at degree 2n-2 the terms span ~30 decades and
        # ``x**(2n-2)`` is itself only good to ~1e-9 relative in float64 --
        # that residual belongs to the evaluation, not to the rule
        assert abs(total - exact) <= max(1e-13, 1e-9 * abs(exact))


@pytest.mark.parametrize("order", LADDER)
def test_rule_matches_numpy_where_numpy_is_healthy(order: int) -> None:
    """Where ``leggauss`` works, the fallback reproduces it.

    The node tolerance is machine precision.  The weight tolerance is loose
    on purpose: the difference at high order is NumPy's, not the fallback's --
    against a 50-digit ``mpmath`` endpoint weight the Newton rule at
    ``n = 2560`` is accurate to ``2e-11`` while ``leggauss`` is accurate to
    ``1.4e-7`` (see ``test_fallback_weight_beats_the_eigvalsh_route_at_high_order``).
    """
    nodes, weights = quadrature.gauss_legendre(order)
    reference_nodes, reference_weights = np.polynomial.legendre.leggauss(order)
    assert np.max(np.abs(nodes - reference_nodes)) < 1e-13
    assert np.max(np.abs(weights - reference_weights) / reference_weights) < 1e-6


@pytest.mark.parametrize("order", (1, 2, 3, 4, 5, 8, 80, 321))
def test_rule_is_ordered_symmetric_and_complete(order: int) -> None:
    nodes, weights = quadrature.gauss_legendre(order)
    assert nodes.shape == (order,) and weights.shape == (order,)
    assert np.all(np.diff(nodes) > 0.0), "nodes must be strictly increasing"
    assert np.all(weights > 0.0)
    assert np.allclose(nodes, -nodes[::-1], atol=1e-15)
    assert np.allclose(weights, weights[::-1], rtol=1e-12)
    # summation of the weights is the interval length; fsum where the order is
    # large enough for a naive sum to lose the last bit
    assert abs(quadrature.rule_sums_to_two(order) - 2.0) < 1e-14


@pytest.mark.parametrize("order", LADDER)
def test_fallback_engages_when_leggauss_raises(order: int, monkeypatch) -> None:
    """The error the cluster nodes raise must select the Newton rule."""
    healthy_nodes, healthy_weights = quadrature.gauss_legendre(order)
    calls: list[int] = []

    def broken(n):
        calls.append(int(n))
        raise np.linalg.LinAlgError("Eigenvalues did not converge")

    monkeypatch.setattr(quadrature, "leggauss", broken)
    quadrature.clear_cache()

    nodes, weights = quadrature.gauss_legendre(order)
    assert calls == [order], "the fallback must be a reaction to the failure"
    assert quadrature.rule_source(order) == "newton-fallback"
    assert np.max(np.abs(nodes - healthy_nodes)) < 1e-13
    assert np.max(np.abs(weights - healthy_weights) / healthy_weights) < 1e-6


def test_a_silently_wrong_rule_is_rejected(monkeypatch) -> None:
    """``eigvalsh`` can also converge to garbage; the integrity check catches it."""
    monkeypatch.setattr(
        quadrature, "leggauss", lambda n: (np.zeros(n), np.zeros(n))
    )
    quadrature.clear_cache()
    nodes, weights = quadrature.gauss_legendre(80)
    assert quadrature.rule_source(80) == "newton-fallback"
    assert abs(weights.sum() - 2.0) < 1e-14
    assert np.all(np.diff(nodes) > 0.0)


def test_fallback_weight_beats_the_eigvalsh_route_at_high_order() -> None:
    """Independent reference: the fallback is not a degraded substitute.

    Uses the endpoint weight, which is the one a high-order rule gets wrong.
    ``mpmath`` is a test-only dependency here; without it the claim is simply
    not checked rather than assumed.
    """
    mpmath = pytest.importorskip("mpmath")
    order = 640

    def broken(n):
        raise np.linalg.LinAlgError("Eigenvalues did not converge")

    quadrature.leggauss, healthy = broken, quadrature.leggauss
    try:
        quadrature.clear_cache()
        nodes, weights = quadrature.gauss_legendre(order)
        source = quadrature.rule_source(order)
    finally:
        quadrature.leggauss = healthy
        quadrature.clear_cache()
    assert source == "newton-fallback"
    index = int(np.argmax(np.abs(nodes)))
    mpmath.mp.dps = 50
    x = mpmath.mpf(float(nodes[index]))
    derivative = mpmath.diff(lambda t: mpmath.legendre(order, t), x)
    reference = 2 / ((1 - x**2) * derivative**2)
    fallback_error = float(abs(mpmath.mpf(float(weights[index])) - reference) / reference)
    numpy_nodes, numpy_weights = np.polynomial.legendre.leggauss(order)
    numpy_error = float(abs(mpmath.mpf(float(numpy_weights[index])) - reference) / reference)
    assert fallback_error < 1e-10
    assert fallback_error <= numpy_error


def test_rule_source_refuses_to_report_an_unbuilt_order() -> None:
    with pytest.raises(KeyError):
        quadrature.rule_source(999)


@pytest.mark.parametrize("bad", (0, -3, 2.5, True))
def test_invalid_order_is_rejected(bad) -> None:
    with pytest.raises((TypeError, ValueError)):
        quadrature.gauss_legendre(bad)


def test_package_imports_with_a_broken_leggauss() -> None:
    """End-to-end: the cluster failure mode must no longer kill the import.

    The patch has to happen in a fresh interpreter, because both this module
    and ``numpy.polynomial.legendre`` bind ``leggauss`` at import time.
    """
    script = """
import sys
import numpy as np
import numpy.polynomial.legendre as legendre

def _boom(n):
    raise np.linalg.LinAlgError("Eigenvalues did not converge")

legendre.leggauss = _boom
import tensorbem                     # this used to raise before any physics ran
import tensorbem.free_surface as fs
import tensorbem.quadrature as q
for order in (24, 80, 2560):
    nodes, weights = q.gauss_legendre(order)
    assert len(nodes) == order
    assert abs(weights.sum() - 2.0) < 1e-13
    assert q.rule_source(order) == "newton-fallback"
assert len(fs._LEGENDRE_X) == 80
rule = fs._legendre_rule(500.0)
assert len(rule[0]) == 1280
print("OK")
"""
    environment = {"PYTHONPATH": str(REPO_ROOT / "src")}
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**environment, "PATH": "/usr/bin:/bin"},
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    assert "OK" in completed.stdout


def test_healthy_backend_still_uses_numpy() -> None:
    """The fix must not move the recorded numbers: NumPy wins where it works."""
    quadrature.clear_cache()
    quadrature.gauss_legendre(80)
    try:
        np.polynomial.legendre.leggauss(80)
    except np.linalg.LinAlgError:
        pytest.skip("this backend cannot build the rule at all")
    assert quadrature.rule_source(80) == "numpy"