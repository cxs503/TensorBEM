import numpy as np
import pytest

from tensorbem.kernels import fundamental_solution, double_layer_kernel


def test_fundamental_solution_reference_length_convention():
    assert np.isclose(float(fundamental_solution(2.0)), 0.0)
    assert np.isclose(float(fundamental_solution(1.0)), np.log(2.0) / (2.0 * np.pi))


def test_fundamental_solution_rejects_singularity():
    with pytest.raises(ValueError):
        fundamental_solution(0.0)


def test_double_layer_kernel():
    value = double_layer_kernel(
        np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0])
    )
    assert np.isclose(value, 1.0 / (2.0 * np.pi))
