"""Regression guard: Hulme hemisphere ka=3.5 must NOT show sign-flip or >5% error.

This test guards against the numpy 2.0.1 solver bug that caused catastrophic
errors at ka=3.5 (A11 sign flip, -104%; B33 sign flip, -102%) as documented
in dev report m20034 and m20910.  After the numpy 2.4.6 environment fix,
all six quantities at ka=3.5 are within 3% of the Hulme (1982) reference.

Reproduce:
    PYTHONPATH=src python -m pytest tests/test_hulme_numpy2_regression.py -v
"""

import math

import numpy as np
import pytest

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh

# --- constants (match hulme_hemisphere_case.py) ---
RADIUS = 1.0
DENSITY = 1025.0
GRAVITY = 9.80665
AMPLITUDE = 1.0
SUBDIVISIONS = 4  # 1024 panels
KA = 3.5
GATE = 0.05  # 5% tolerance


@pytest.fixture(scope="module")
def ka35_cell():
    """Single (ka=3.5, 1024-panel) radiation+diffraction cell."""
    nu = KA / RADIUS
    mesh = hemisphere_hull_mesh(SUBDIVISIONS, radius=RADIUS)
    ops = assemble_free_surface_operators_3d(mesh, nu, chief_points="auto")
    rad = solve_radiation_3d(
        mesh, nu, modes=(0, 2), density=DENSITY, gravity=GRAVITY, operators=ops
    )
    diff = solve_diffraction_3d(
        mesh, nu, amplitude=AMPLITUDE, wave_angle=0.0,
        density=DENSITY, gravity=GRAVITY, operators=ops,
    )
    vol = (2.0 / 3.0) * math.pi * RADIUS**3
    norm_a = DENSITY * vol
    norm_b = DENSITY * math.sqrt(GRAVITY * nu) * vol
    scale_f = DENSITY * GRAVITY * AMPLITUDE * RADIUS**2

    ref = hulme_hemisphere_radiation(
        KA, radius=RADIUS, amplitude=AMPLITUDE, density=DENSITY,
        gravity=GRAVITY, n_terms=144, quad_order=288,
    )

    fx = complex(diff.excitation_force[0]) / scale_f
    fz = complex(diff.excitation_force[2]) / scale_f

    return {
        "ka": KA,
        "panels": mesh.panel_count,
        "condition_number": ops.condition_number,
        "radiation_residual": rad.relative_residual,
        "diffraction_residual": diff.relative_residual,
        "A11": float(rad.added_mass[0, 0]) / norm_a,
        "A33": float(rad.added_mass[1, 1]) / norm_a,
        "B11": float(rad.damping[0, 0]) / norm_b,
        "B33": float(rad.damping[1, 1]) / norm_b,
        "Fx": abs(fx),
        "Fz": abs(fz),
        "ref_A11": ref.surge_added_mass_dimless,
        "ref_A33": ref.heave_added_mass_dimless,
        "ref_B11": ref.surge_damping_dimless,
        "ref_B33": ref.heave_damping_dimless,
        "ref_Fx": abs(complex(ref.excitation_force_x) / scale_f),
        "ref_Fz": abs(complex(ref.excitation_force_z) / scale_f),
    }


def test_ka35_no_sign_flip(ka35_cell):
    """A11 and B33 must have the same sign as the reference (no sign flip)."""
    c = ka35_cell
    assert c["A11"] * c["ref_A11"] > 0, (
        f"A11 sign flip: got {c['A11']:.6f}, ref {c['ref_A11']:.6f}"
    )
    assert c["B33"] * c["ref_B33"] > 0, (
        f"B33 sign flip: got {c['B33']:.6f}, ref {c['ref_B33']:.6f}"
    )


def test_ka35_all_errors_below_5pct(ka35_cell):
    """All six quantities must be within 5% of the Hulme reference."""
    c = ka35_cell
    errors = {
        "A11": abs(c["A11"] - c["ref_A11"]) / abs(c["ref_A11"]),
        "A33": abs(c["A33"] - c["ref_A33"]) / abs(c["ref_A33"]),
        "B11": abs(c["B11"] - c["ref_B11"]) / abs(c["ref_B11"]),
        "B33": abs(c["B33"] - c["ref_B33"]) / abs(c["ref_B33"]),
        "Fx": abs(c["Fx"] - c["ref_Fx"]) / abs(c["ref_Fx"]),
        "Fz": abs(c["Fz"] - c["ref_Fz"]) / abs(c["ref_Fz"]),
    }
    worst = max(errors.values())
    worst_key = max(errors, key=errors.get)
    assert worst < GATE, (
        f"ka={KA}: worst error {worst:.2%} ({worst_key}) exceeds {GATE:.0%} gate. "
        f"All errors: {errors}"
    )


def test_ka35_residual_is_healthy(ka35_cell):
    """Solver residual must be well below 1e-2 (E0 admissibility gate)."""
    c = ka35_cell
    assert c["radiation_residual"] < 1e-2, (
        f"Radiation residual {c['radiation_residual']:.2e} >= 1e-2"
    )
    assert c["diffraction_residual"] < 1e-2, (
        f"Diffraction residual {c['diffraction_residual']:.2e} >= 1e-2"
    )


def test_ka35_condition_number_is_reasonable(ka35_cell):
    """Condition number should be low (not an ill-conditioning issue)."""
    c = ka35_cell
    assert c["condition_number"] < 100, (
        f"Condition number {c['condition_number']:.2f} is suspiciously high"
    )


def test_ka35_values_match_documented_numpy246_results(ka35_cell):
    """Values should match the numpy 2.4.6 results documented in m21563.

    Tolerance: 0.1% (to catch any regression from the documented values).
    """
    c = ka35_cell
    # Documented numpy 2.4.6 values (from m21563 dev report)
    expected = {
        "A11": 0.16184,
        "A33": 0.42068,
        "B11": 0.18236,
        "B33": 0.03193,
    }
    for key, exp in expected.items():
        got = c[key]
        rel_err = abs(got - exp) / abs(exp) if exp != 0 else abs(got - exp)
        assert rel_err < 0.001, (
            f"{key}: got {got:.6f}, expected ~{exp:.5f}, rel_err={rel_err:.2%}"
        )
