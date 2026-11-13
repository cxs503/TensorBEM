"""Integration test: translating-sphere added-mass benchmark (CASE-BEM-SPHERE-ADDED-MASS).

Calls the production entry ``tensorbem.benchmarks.sphere_added_mass.run``
(exterior Dirichlet BEM via ``tensorbem.solver3d.solve_laplace_surface``)
and the refinement-ladder benchmark ``run_refinement`` against the closed
form ``m_A = 0.5*rho*V`` (Lamb 1932 sec. 6.10; Newman 1977 eq. 4.132 sphere
limit).
"""

import math

from tensorbem.benchmarks.sphere_added_mass import run, run_refinement


def test_sphere_added_mass_passes_three_percent_at_320_panels():
    row = run(2)  # icosphere subdivision 2 -> 320 panels
    assert row["panels"] == 320
    assert row["added_mass_relative_error"] < 3.0e-2
    assert math.isclose(
        row["exact_added_mass"], 0.5 * 1025.0 * 4.0 / 3.0 * math.pi, rel_tol=1e-15
    )


def test_refinement_ladder_converges_and_passes():
    bench = run_refinement((1, 2), tolerance=0.03)
    rows = bench["levels"]
    assert [r["panels"] for r in rows] == [80, 320]
    assert rows[1]["rel_error"] < rows[0]["rel_error"]
    assert rows[1]["pass"]
    assert bench["first_passing_level"] == 320
    # closed form cross-checked against the repo analytic module
    assert bench["reference"]["analytic_module_rel_diff"] < 1e-9
    # base-2 convergence order (icosphere halves the edge per subdivision)
    assert 1.0 < bench["convergence_orders_log2"][0] < 3.0


def test_negative_control_radius_scaling_is_not_hardcoded():
    bench = run_refinement((1,), tolerance=0.03)
    neg = bench["negative_control_radius_2x"]
    assert math.isclose(neg["scale_measured"], 8.0, rel_tol=1e-9)
    assert neg["rel_error_unchanged"]
