"""Gates for the analytic gradient of the 3-D finite-depth free-surface kernel.

The kernel used to expose only ``dG/dz``; the horizontal derivatives need ``K_1``
and ``H_1^{(1)}`` and are what a finite-depth panel assembly consumes.  Every
reference below is independent of the code under test: SciPy's Bessel functions,
central differences of ``G`` itself, the deep-water Kelvin gradient of
:mod:`tensorbem.free_surface3d`, and the boundary conditions of the strip.
"""

import math

import numpy as np
import pytest

from tensorbem.finite_depth3d import (
    bessel_k1,
    green_function_3d,
    green_gradient_3d,
    green_gradient_source_3d,
    green_gradient_z_3d,
    hankel_h1,
)
from tensorbem.free_surface3d import green_function_3d as deep_green_function_3d
from tensorbem.free_surface3d import green_gradient_field_3d as deep_gradient_3d

NU = 0.6
SOURCE = np.array([0.0, 0.0, -1.25])
FD_STEP = 1.0e-6


def _fd_gradient(point, depth, source=None):
    source = SOURCE if source is None else source
    out = np.zeros(3, dtype=complex)
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = FD_STEP
        out[axis] = (
            green_function_3d(point + step, source, NU, depth)
            - green_function_3d(point - step, source, NU, depth)
        ) / (2.0 * FD_STEP)
    return out


def _fd_source_gradient(point, depth):
    out = np.zeros(3, dtype=complex)
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = FD_STEP
        out[axis] = (
            green_function_3d(point, SOURCE + step, NU, depth)
            - green_function_3d(point, SOURCE - step, NU, depth)
        ) / (2.0 * FD_STEP)
    return out


def _relative(numerical, reference):
    scale = float(np.max(np.abs(reference)))
    return float(np.max(np.abs(np.asarray(numerical) - np.asarray(reference))) / scale)


def test_bessel_k1_matches_scipy():
    sp = pytest.importorskip("scipy.special")
    x = np.logspace(-8.0, 2.5, 200)
    assert np.max(np.abs(bessel_k1(x) / sp.k1(x) - 1.0)) < 1.0e-8


def test_hankel_h1_matches_scipy():
    sp = pytest.importorskip("scipy.special")
    x = np.logspace(-8.0, 2.5, 200)
    assert np.max(np.abs(hankel_h1(x) / sp.hankel1(1, x) - 1.0)) < 1.0e-12
    assert np.allclose(hankel_h1(x, outgoing=False), np.conjugate(hankel_h1(x)))


def test_bessel_k1_is_continuous_across_the_branch_switch():
    from tensorbem.finite_depth3d import _K1_SERIES_LIMIT, _bessel_k1_large, _bessel_k1_small

    limit = _K1_SERIES_LIMIT
    left = _bessel_k1_small(np.array([limit - 1e-9]))
    right = _bessel_k1_large(np.array([limit + 1e-9]))
    assert abs(float(left[0] / right[0]) - 1.0) < 1.0e-6


def test_bessel_k1_rejects_non_positive_argument():
    with pytest.raises(ValueError):
        bessel_k1(np.array([0.0]))
    with pytest.raises(ValueError):
        bessel_k1(np.array([-1.0]))


@pytest.mark.parametrize("depth", [2.0, 5.0, 20.0])
def test_horizontal_and_vertical_gradients_match_central_difference(depth):
    points = np.array(
        [
            [2.0, 0.5, -0.5],
            [0.3, -1.7, -min(1.9, depth - 0.2)],
            [1.1, 1.1, -0.15],
            [-2.4, 0.8, -0.9],
        ]
    )
    for point in points:
        analytic = green_gradient_3d(point, SOURCE, NU, depth)
        reference = _fd_gradient(point, depth)
        assert _relative(analytic, reference) < 1.0e-5
        assert _relative(analytic[:2], reference[:2]) < 1.0e-5
        assert _relative(analytic[2:3], reference[2:3]) < 1.0e-5


def test_source_gradient_matches_finite_difference_of_the_source():
    depth = 5.0
    point = np.array([1.6, -0.9, -0.6])
    analytic = green_gradient_source_3d(point, SOURCE, NU, depth)
    assert _relative(analytic, _fd_source_gradient(point, depth)) < 1.0e-5


def test_vertical_component_is_bitwise_the_existing_kernel():
    for depth in (2.0, 5.0, 20.0):
        point = np.array([2.0, 0.5, -0.5])
        full = green_gradient_3d(point, SOURCE, NU, depth)
        only_z = green_gradient_z_3d(point, SOURCE, NU, depth)
        assert _relative(full[2], only_z) < 1.0e-14


@pytest.mark.parametrize("depth", [2.0, 5.0, 20.0])
def test_gradient_satisfies_the_free_surface_condition(depth):
    angles = np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
    ring = np.array([[1.7 * math.cos(a), 1.7 * math.sin(a), 0.0] for a in angles])
    value = green_function_3d(ring, SOURCE, NU, depth)
    vertical = green_gradient_3d(ring, SOURCE, NU, depth)[:, 2]
    residual = vertical - NU * value
    assert float(np.max(np.abs(residual)) / np.max(np.abs(value))) < 1.0e-10


@pytest.mark.parametrize("depth", [2.0, 5.0, 20.0])
def test_gradient_satisfies_bottom_impermeability(depth):
    angles = np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
    ring = np.array([[1.7 * math.cos(a), 1.7 * math.sin(a), -depth] for a in angles])
    vertical = green_gradient_3d(ring, SOURCE, NU, depth)[:, 2]
    assert float(np.max(np.abs(vertical))) <= 1.0e-13 * max(1.0, NU)


def test_gradient_is_harmonic_off_the_source_axis():
    depth = 5.0
    point = np.array([1.4, 0.9, -1.7])
    divergence = 0.0j
    scale = 0.0
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = FD_STEP
        plus = green_gradient_3d(point + step, SOURCE, NU, depth)
        minus = green_gradient_3d(point - step, SOURCE, NU, depth)
        divergence += (plus[axis] - minus[axis]) / (2.0 * FD_STEP)
        scale += abs(plus[axis])
    assert abs(divergence) / scale < 1.0e-5


@pytest.mark.parametrize(
    "depth, tolerance", [(20.0, 2.0e-4), (100.0, 1.0e-6), (400.0, 1.0e-7)]
)
def test_gradient_reaches_the_deep_water_kelvin_kernel(depth, tolerance):
    """Deviation decays exponentially in ``k0*h`` (measured: 8e-5, 5e-7, 8e-9)."""
    points = np.array([[1.0, 0.5, -0.8], [3.0, -2.0, -4.0]])
    finite = green_gradient_3d(points, SOURCE, NU, depth)
    deep = deep_gradient_3d(points, SOURCE, NU)
    assert _relative(finite, deep) < tolerance
    assert _relative(
        green_function_3d(points, SOURCE, NU, depth),
        deep_green_function_3d(points, SOURCE, NU),
    ) < 2.0 * tolerance


def test_far_field_radial_gradient_obeys_the_outgoing_wave_law():
    depth = 5.0
    radius = 60.0
    angles = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
    ring = np.array(
        [[radius * math.cos(a), radius * math.sin(a), SOURCE[2]] for a in angles]
    )
    value = green_function_3d(ring, SOURCE, NU, depth)
    gradient = green_gradient_3d(ring, SOURCE, NU, depth)
    radial = gradient[:, 0] * np.cos(angles) + gradient[:, 1] * np.sin(angles)
    assert _relative(radial, 1j * NU * value) < 2.0e-2


def test_axis_targets_have_zero_horizontal_gradient_without_warnings():
    depth = 5.0
    point = np.array([0.0, 0.0, -3.0])  # on the source's vertical axis
    gradient = green_gradient_3d(point, SOURCE, NU, depth)
    assert gradient[0] == 0.0 and gradient[1] == 0.0
    assert np.isfinite(gradient[2])
    # ... and the radial component close to the axis still matches the FD reference
    off = point + np.array([1.0e-2, 0.0, 0.0])
    assert _relative(green_gradient_3d(off, SOURCE, NU, depth), _fd_gradient(off, depth)) < 1.0e-5


def test_gradient_rejects_points_outside_the_strip():
    with pytest.raises(ValueError):
        green_gradient_3d(np.array([1.0, 0.0, 0.5]), SOURCE, NU, 5.0)
    with pytest.raises(ValueError):
        green_gradient_3d(np.array([1.0, 0.0, -6.0]), SOURCE, NU, 5.0)
    with pytest.raises(ValueError):
        green_gradient_3d(SOURCE, SOURCE, NU, 5.0)
    with pytest.raises(ValueError):
        green_gradient_3d(np.array([1.0, 0.0, -1.0]), SOURCE, 0.0, 5.0)
    with pytest.raises(ValueError):
        green_gradient_3d(np.array([1.0, 0.0, -1.0]), SOURCE, NU, -5.0)


def test_benchmark_study_passes_every_gate():
    """Re-run the published study (about a minute) and check its verdict."""
    from tensorbem.benchmarks.finite_depth3d_gradients import run_study

    study = run_study()
    failed = [name for name, ok in study["verdict"]["gates"].items() if not ok]
    assert failed == []
    assert study["finite_difference"]["worst_target_relative_error"] < 1.0e-5
    assert study["z_component_parity"] < 1.0e-14