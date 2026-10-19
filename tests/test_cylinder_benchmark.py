import numpy as np

from tensorbem.benchmarks.cylinder_added_mass import run


def test_circular_cylinder_added_mass_converges_to_analytic_value():
    coarse = run(32)
    fine = run(64)

    assert fine["added_mass_relative_error"] < coarse["added_mass_relative_error"]
    assert fine["added_mass_relative_error"] < 3.0e-2
    assert fine["normal_derivative_rmse"] < coarse["normal_derivative_rmse"]


def test_circular_cylinder_added_mass_scales_with_density_and_radius_squared():
    base = run(64, radius=1.0, density=1000.0)
    scaled = run(64, radius=2.0, density=500.0)

    assert np.isclose(
        scaled["exact_added_mass_per_unit_span"],
        2.0 * base["exact_added_mass_per_unit_span"],
    )
    assert np.isclose(
        scaled["added_mass_per_unit_span"],
        2.0 * base["added_mass_per_unit_span"],
    )
