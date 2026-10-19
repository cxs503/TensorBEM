import numpy as np
import pytest

from tensorbem.kernels import double_layer_kernel, fundamental_solution


def test_fundamental_solution_known_value():
    assert np.isclose(float(fundamental_solution(1.0)), 0.0)
    assert np.isclose(float(fundamental_solution(np.e)), -1.0 / (2.0 * np.pi))


def test_fundamental_solution_rejects_singularity():
    with pytest.raises(ValueError):
        fundamental_solution(0.0)


def test_double_layer_kernel():
    value = double_layer_kernel(
        np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0])
    )
    assert np.isclose(value, 1.0 / (2.0 * np.pi))
