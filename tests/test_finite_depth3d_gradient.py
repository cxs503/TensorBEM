"""Gates for the 3-D finite-depth kernel gradient and its order-1 Bessel layer.

Companion of :mod:`tensorbem.benchmarks.finite_depth3d_gradient`.  The gradient
``nabla G`` is what a finite-depth panel layer needs; ``K_1``/``H_1`` are new and
``K_0`` was re-branched because its ascending series carried a ``2.1e-8``
relative-accuracy cliff at ``x ~ 8.9`` (see the ledger).

Gates fall in four groups:

* **Bessel layer** -- ``K_1`` against SciPy references in every branch, seam
  continuity of the quadrature branch, and the ``K_0`` cliff regression;
* **analytic derivatives** -- against central/Richardson differences of ``G``,
  and the closed symmetry identity ``nabla_x G(x, xi) == nabla_xi G(xi, x)``;
* **physical closures** -- free-surface and bottom conditions carried by the
  horizontal gradient, and the irrotationality of the horizontal field;
* **limits** -- ``1/(4 pi r)`` singularity strength and deep-water collapse onto
  the independent Kelvin kernel of :mod:`tensorbem.free_surface3d`.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.benchmarks.finite_depth3d_gradient import (
    K_REFERENCE,
    THRESHOLDS,
    run,
)
from tensorbem.finite_depth3d import (
    _K0_SERIES_LIMIT,
    _K_ASYMPTOTIC_LIMIT,
    _bessel_k0_k1_quad,
    bessel_k0,
    bessel_k1,
    finite_depth_wavenumber,
    green_function_3d,
    green_gradient_3d,
    green_gradient_horizontal_3d,
    green_gradient_source_3d,
    green_gradient_z_3d,
    hankel_h0,
    hankel_h1,
)
from tensorbem.free_surface3d import green_gradient_field_3d as deep_water_gradient

NU = 0.6
DEPTH = 4.0
TARGETS = np.array([[0.7, 0.3, -0.4], [1.5, 1.0, -2.0], [2.0, -1.0, -3.0]])
SOURCE = np.array([0.2, -0.1, -1.0])


# ---------------------------------------------------------------------------
# 1. Bessel layer
# ---------------------------------------------------------------------------
def test_bessel_k1_matches_scipy_in_every_branch():
    special = pytest.importorskip("scipy.special")
    x = np.concatenate(
        [
            np.logspace(-3.0, 0.6, 25),  # ascending series
            np.linspace(4.0, 16.0, 61),  # Laplace-integral quadrature
            np.logspace(np.log10(16.0), 2.0, 25),  # asymptotic
        ]
    )
    k1 = bessel_k1(x)
    assert np.max(np.abs(k1 / special.kv(1, x) - 1.0)) < 5e-13


def test_bessel_k0_accuracy_cliff_is_gone():
    """The pre-fix series carried 2.1e-8 relative error near ``x = 8.9``."""
    special = pytest.importorskip("scipy.special")
    x = np.linspace(_K0_SERIES_LIMIT, _K_ASYMPTOTIC_LIMIT, 121)
    k0 = bessel_k0(x)
    worst = np.max(np.abs(k0 / special.kv(0, x) - 1.0))
    assert worst < 1e-12, worst


def test_quadrature_branch_matches_both_neighbouring_branches():
    """Seam continuity: the quadrature must agree with series *and* asymptotic."""
    for value in (_K0_SERIES_LIMIT + 1e-3, _K_ASYMPTOTIC_LIMIT - 1e-3):
        single = np.array([value])
        k0_quad, k1_quad = _bessel_k0_k1_quad(single)
        assert abs(float(k0_quad[0]) / float(bessel_k0(single)[0]) - 1.0) < 1e-11
        assert abs(float(k1_quad[0]) / float(bessel_k1(single)[0]) - 1.0) < 1e-11


def test_bessel_reference_pins():
    """Pinned mpmath values (40 significant digits) at every branch seam."""
    x = np.array([row[0] for row in K_REFERENCE])
    k0_ref = np.array([row[1] for row in K_REFERENCE])
    k1_ref = np.array([row[2] for row in K_REFERENCE])
    assert np.max(np.abs(bessel_k0(x) / k0_ref - 1.0)) < 1e-12
    assert np.max(np.abs(bessel_k1(x) / k1_ref - 1.0)) < 1e-12


def test_bessel_k1_is_minus_derivative_of_k0():
    """``K_1 = -dK_0/dx`` on a step ladder (the best step wins)."""
    x = np.array([0.3, 1.0, 3.0, 6.0, 12.0])
    best = np.inf
    for step in (1e-2, 1e-3, 1e-4, 1e-5):
        coarse = (bessel_k0(x + step) - bessel_k0(x - step)) / (2.0 * step)
        fine = (bessel_k0(x + 0.5 * step) - bessel_k0(x - 0.5 * step)) / step
        derivative = (4.0 * fine - coarse) / 3.0
        best = min(best, float(np.max(np.abs(derivative + bessel_k1(x)) / bessel_k1(x))))
    assert best < 1e-7, best


def test_hankel_h1_matches_h0_derivative_and_scipy():
    special = pytest.importorskip("scipy.special")
    x = np.array([0.5, 2.0, 8.0, 20.0])
    expected = special.jv(1, x) + 1j * special.yv(1, x)
    assert np.max(np.abs(hankel_h1(x) - expected) / np.abs(expected)) < 1e-12
    assert np.max(np.abs(hankel_h1(x, outgoing=False) - np.conjugate(expected))) < 1e-12
    step = 1e-5
    coarse = (hankel_h0(x + step) - hankel_h0(x - step)) / (2.0 * step)
    fine = (hankel_h0(x + 0.5 * step) - hankel_h0(x - 0.5 * step)) / step
    derivative = (4.0 * fine - coarse) / 3.0
    assert np.max(np.abs(derivative + hankel_h1(x)) / np.abs(hankel_h1(x))) < 1e-8


def test_bessel_rejects_non_positive_arguments():
    with pytest.raises(ValueError):
        bessel_k0(np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        bessel_k1(np.array([-1.0, 1.0]))


# ---------------------------------------------------------------------------
# 2. Analytic derivatives
# ---------------------------------------------------------------------------
def test_gradient_matches_richardson_differences():
    analytic = green_gradient_3d(TARGETS, SOURCE, NU, DEPTH)
    step = 1e-4
    for axis in range(3):
        def shifted(delta: float) -> np.ndarray:
            points = TARGETS.copy()
            points[:, axis] += delta
            return green_function_3d(points, SOURCE, NU, DEPTH)

        coarse = (shifted(step) - shifted(-step)) / (2.0 * step)
        fine = (shifted(0.5 * step) - shifted(-0.5 * step)) / step
        derivative = (4.0 * fine - coarse) / 3.0
        scale = np.max(np.abs(analytic[:, axis]))
        assert np.max(np.abs(derivative - analytic[:, axis])) / scale < 1e-9


def test_horizontal_gradient_is_antisymmetric_under_swap():
    field = green_gradient_horizontal_3d(TARGETS, SOURCE, NU, DEPTH)
    swapped = green_gradient_horizontal_3d(SOURCE, TARGETS, NU, DEPTH)
    assert np.max(np.abs(field + swapped)) == 0.0


def test_field_and_source_gradients_satisfy_the_symmetry_identity():
    field = green_gradient_3d(TARGETS, SOURCE, NU, DEPTH)
    source = green_gradient_source_3d(SOURCE, TARGETS, NU, DEPTH)
    assert field.shape == source.shape == (TARGETS.shape[0], 3)
    assert np.max(np.abs(field - source)) / np.max(np.abs(field)) < 1e-12


def test_incoming_branch_gradient_matches_its_own_finite_difference():
    """``outgoing=False`` swaps ``H^{(1)} -> H^{(2)}``: check it against ``G`` directly."""
    incoming = green_gradient_3d(TARGETS, SOURCE, NU, DEPTH, outgoing=False)
    outgoing = green_gradient_3d(TARGETS, SOURCE, NU, DEPTH)
    assert np.max(np.abs(incoming - outgoing)) > 0.0
    step = 1e-4
    for axis in range(3):
        def shifted(delta: float) -> np.ndarray:
            points = TARGETS.copy()
            points[:, axis] += delta
            return green_function_3d(points, SOURCE, NU, DEPTH, outgoing=False)

        coarse = (shifted(step) - shifted(-step)) / (2.0 * step)
        fine = (shifted(0.5 * step) - shifted(-0.5 * step)) / step
        derivative = (4.0 * fine - coarse) / 3.0
        scale = np.max(np.abs(incoming[:, axis]))
        assert np.max(np.abs(derivative - incoming[:, axis])) / scale < 1e-9


def test_gradient_z_matches_the_shipped_vertical_routine():
    full = green_gradient_3d(TARGETS, SOURCE, NU, DEPTH)
    vertical = green_gradient_z_3d(TARGETS, SOURCE, NU, DEPTH)
    assert np.max(np.abs(full[:, 2] - vertical)) == 0.0


# ---------------------------------------------------------------------------
# 3. Physical closures
# ---------------------------------------------------------------------------
def test_horizontal_gradient_carries_the_free_surface_and_bottom_conditions():
    step = 1e-5
    surface = np.array([[0.7, 0.3, 0.0], [1.5, 1.0, 0.0]])
    bottom = np.array([[0.7, 0.3, -DEPTH], [1.5, 1.0, -DEPTH]])

    def d_dx_of_dz(points: np.ndarray) -> np.ndarray:
        plus = points.copy()
        minus = points.copy()
        plus[:, 0] += step
        minus[:, 0] -= step
        return (
            green_gradient_z_3d(plus, SOURCE, NU, DEPTH)
            - green_gradient_z_3d(minus, SOURCE, NU, DEPTH)
        ) / (2.0 * step)

    gx_surface = green_gradient_3d(surface, SOURCE, NU, DEPTH)[:, 0]
    assert np.max(np.abs(d_dx_of_dz(surface) - NU * gx_surface)) / np.max(
        np.abs(NU * gx_surface)
    ) < 1e-8
    assert np.max(np.abs(d_dx_of_dz(bottom))) / np.max(np.abs(NU * gx_surface)) < 1e-8


def test_horizontal_gradient_is_irrotational():
    step = 1e-5

    def d(field: np.ndarray, axis: int) -> np.ndarray:
        plus = TARGETS.copy()
        minus = TARGETS.copy()
        plus[:, axis] += step
        minus[:, axis] -= step
        grad_plus = green_gradient_3d(plus, SOURCE, NU, DEPTH)
        grad_minus = green_gradient_3d(minus, SOURCE, NU, DEPTH)
        return (grad_plus - grad_minus) / (2.0 * step)

    curl = d(TARGETS, 1)[:, 0] - d(TARGETS, 0)[:, 1]
    scale = np.max(np.abs(green_gradient_3d(TARGETS, SOURCE, NU, DEPTH)))
    assert np.max(np.abs(curl)) / scale < 1e-7


# ---------------------------------------------------------------------------
# 4. Limits
# ---------------------------------------------------------------------------
def test_gradient_has_the_expected_singularity_strength():
    for radius in (2.5e-2, 2.5e-3):
        target = np.array([[radius, 0.0, -1.0]])
        source = np.array([0.0, 0.0, -1.0 - 1.0e-9])
        gradient = green_gradient_3d(target, source, NU, DEPTH)[0]
        strength = 4.0 * np.pi * radius * radius * np.linalg.norm(gradient)
        assert abs(strength - 1.0) < 1e-4
        unit = gradient / np.linalg.norm(gradient)
        assert unit[0].real < -1.0 + 1e-6


def test_gradient_collapses_onto_the_deep_water_kernel():
    depth = 128.0
    k0 = finite_depth_wavenumber(NU, depth)
    assert k0 * depth > 70.0  # deep-water regime
    finite = green_gradient_3d(TARGETS, SOURCE, NU, depth)
    deep = deep_water_gradient(TARGETS, SOURCE, NU)
    assert np.max(np.abs(finite - deep)) / np.max(np.abs(deep)) < 1e-6


# ---------------------------------------------------------------------------
# 5. Input validation and shape
# ---------------------------------------------------------------------------
def test_horizontal_gradient_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        green_gradient_horizontal_3d(np.array([[0.0, 0.0, 0.1]]), SOURCE, NU, DEPTH)
    with pytest.raises(ValueError):
        green_gradient_horizontal_3d(np.array([[0.0, 0.0, -DEPTH - 1.0]]), SOURCE, NU, DEPTH)
    with pytest.raises(ValueError):
        green_gradient_horizontal_3d(TARGETS, SOURCE, 0.0, DEPTH)
    with pytest.raises(ValueError):
        green_gradient_horizontal_3d(TARGETS, SOURCE, NU, 0.0)
    with pytest.raises(ValueError):
        green_gradient_horizontal_3d(np.array([[0.2, -0.1, -1.0]]), SOURCE, NU, DEPTH)


def test_gradient_shapes_match_the_kernel_broadcasting():
    point = np.array([0.4, 0.1, -0.8])
    single = green_gradient_3d(point, SOURCE, NU, DEPTH)
    assert single.shape == (3,)
    batch = green_gradient_3d(TARGETS, SOURCE, NU, DEPTH)
    assert batch.shape == (TARGETS.shape[0], 3)
    assert np.allclose(batch[0], green_gradient_3d(TARGETS[0], SOURCE, NU, DEPTH))


# ---------------------------------------------------------------------------
# 6. The ledger itself
# ---------------------------------------------------------------------------
def test_ledger_passes_every_gate():
    ledger = run()
    assert ledger["verdict"]["pass"] is True, ledger["verdict"]["failures"]
    for check in ledger["checks"]:
        assert check["pass"] is True
        assert check["measured"] <= THRESHOLDS[check["name"]]
    # The Bessel fix is only interesting if the legacy branches really were worse.
    assert ledger["bessel"]["scan"]["legacy_k0_max_relative_error"] > 1e-8
    assert ledger["bessel"]["scan"]["current_k0_max_relative_error"] < 1e-12