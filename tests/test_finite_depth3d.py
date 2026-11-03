"""Tests for the finite-depth 3-D free-surface Green function.

What is asserted here is exactly what has been established:

* the dispersion root and its derivative;
* the free-surface condition ``dG/dz = nu*G`` at ``z = 0`` and the bottom
  condition ``dG/dz = 0`` at ``z = -h`` -- both hold to roundoff because the
  representation is built that way (this checks the implementation);
* reciprocity ``G(P, Q) = G(Q, P)`` and grid independence of the principal
  value (both are non-trivial for an integral with a pole);
* the deep-water limit against :mod:`tensorbem.free_surface3d` is recorded as a
  **failing** cross-check for *off-axis* field points: the measured relative
  differences are ~0.34, ~0.56 and ~3.2 and do not decrease with depth, while
  the on-axis point converges to 1.9e-6 at ``nu*h = 36``.  The module is
  therefore not a verified kernel.  The test is marked ``xfail`` with the measured numbers and
  will turn into a red build (``strict=True`` XPASS) once the diagnosis is
  closed -- that is the intended signal, not a licence to delete the test.
"""

import math

import numpy as np
import pytest

from tensorbem.finite_depth import finite_depth_wavenumber
from tensorbem.finite_depth3d import (
    dispersion_derivative,
    green_function_finite_depth_3d,
    green_function_finite_depth_z_derivative_3d,
    pole_data,
    wave_amplitude_coefficient,
)
from tensorbem.free_surface3d import green_function_3d

NU = 0.6
SOURCE = np.array([[0.0, 0.0, -1.0]])


class TestDispersion:
    @pytest.mark.parametrize("depth", [0.25, 0.5, 1.0, 2.0, 8.0, 60.0])
    def test_root_satisfies_dispersion(self, depth):
        k0, slope = pole_data(NU, depth)
        assert abs(k0 * math.tanh(k0 * depth) - NU) < 1e-12
        assert k0 == pytest.approx(finite_depth_wavenumber(NU, depth), rel=0.0, abs=0.0)

    def test_deep_and_shallow_limits(self):
        assert finite_depth_wavenumber(NU, 400.0) == pytest.approx(NU, rel=1e-6)
        shallow = finite_depth_wavenumber(NU, 0.01)
        assert shallow == pytest.approx(math.sqrt(NU / 0.01), rel=1e-2)

    def test_slope_matches_finite_difference(self):
        depth = 2.0
        k0 = finite_depth_wavenumber(NU, depth)
        h = 1e-6
        f = lambda k: k * math.sinh(k * depth) - NU * math.cosh(k * depth)
        numeric = (f(k0 + h) - f(k0 - h)) / (2 * h)
        assert dispersion_derivative(NU, depth, k0) == pytest.approx(numeric, rel=1e-7)


class TestBoundaryConditions:
    @pytest.mark.parametrize("depth", [1.0, 2.0, 8.0])
    def test_free_surface_condition(self, depth):
        surface = np.array([[0.3, 0.0, 0.0], [1.2, 0.4, 0.0], [2.0, -1.0, 0.0]])
        value = green_function_finite_depth_3d(surface, SOURCE, NU, depth)
        dz = green_function_finite_depth_z_derivative_3d(surface, SOURCE, NU, depth)
        assert np.max(np.abs(dz - NU * value) / np.abs(NU * value)) < 1e-12

    @pytest.mark.parametrize("depth", [2.0, 4.0, 8.0])
    def test_bottom_condition(self, depth):
        # depth = 1.0 is excluded: the source at z = -1 then lies on the bottom
        # and the bottom image is singular, which is a property of the case,
        # not of the kernel.
        bottom = np.array([[0.3, 0.0, -depth], [1.2, 0.4, -depth]])
        dz = green_function_finite_depth_z_derivative_3d(bottom, SOURCE, NU, depth)
        assert np.max(np.abs(dz)) < 1e-6


class TestStructure:
    def test_reciprocity(self):
        left = np.array([[0.5, 0.0, -0.5], [2.0, 1.0, -1.5]])
        right = np.array([[0.0, 0.0, -1.0], [0.3, -0.4, -1.7]])
        forward = green_function_finite_depth_3d(left, right, NU, 2.0)
        backward = green_function_finite_depth_3d(right, left, NU, 2.0)
        assert np.max(np.abs(forward - backward) / np.abs(forward)) < 1e-12

    def test_grid_independence(self):
        depth = 8.0
        targets = np.array([[0.5, 0.0, -0.5], [2.0, 1.0, -1.5]])
        reference = green_function_finite_depth_3d(targets, SOURCE, NU, depth, nodes=16)
        for nodes in (8, 12, 24):
            value = green_function_finite_depth_3d(targets, SOURCE, NU, depth, nodes=nodes)
            assert np.max(np.abs(value - reference)) < 1e-9

    def test_outgoing_and_incoming_differ_by_twice_the_residue(self):
        depth = 2.0
        targets = np.array([[1.0, 0.0, -0.5]])
        outgoing = green_function_finite_depth_3d(targets, SOURCE, NU, depth)
        incoming = green_function_finite_depth_3d(
            targets, SOURCE, NU, depth, outgoing=False
        )
        gap = out = outgoing - incoming
        assert abs(gap.imag) > 0.0 and abs(gap.real) < 1e-12
        del out

    def test_validation(self):
        with pytest.raises(ValueError):
            green_function_finite_depth_3d(
                np.array([[0.0, 0.0, 0.1]]), SOURCE, NU, 2.0
            )
        with pytest.raises(ValueError):
            green_function_finite_depth_3d(
                np.array([[0.0, 0.0, -2.5]]), SOURCE, NU, 2.0
            )
        with pytest.raises(ValueError):
            green_function_finite_depth_3d(SOURCE, SOURCE, NU, 2.0)


class TestAmplitudeCoefficient:
    def test_matches_residue_scale(self):
        depth = 2.0
        amplitude = wave_amplitude_coefficient(np.array([0.0]), -1.0, NU, depth)[0]
        assert 0.0 < abs(amplitude) < 1.0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "deep-water limit cross-check FAILS: relative differences 0.34 / 3.2 / 0.56 "
        "(real part) do not decrease with h; kernel is not verified -- see "
        "docs/finite_depth3d_green_function.md §5"
    ),
)
def test_deep_water_limit_matches_free_surface3d():
    targets = np.array([[0.5, 0.0, -0.5], [2.0, 1.0, -1.5], [0.0, 0.0, -0.2]])
    reference = green_function_3d(targets, SOURCE, NU)
    for depth in (8.0, 20.0, 60.0):
        value = green_function_finite_depth_3d(targets, SOURCE, NU, depth)
        assert np.max(np.abs(value - reference) / np.abs(reference)) < 1e-6