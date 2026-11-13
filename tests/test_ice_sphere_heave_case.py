"""Case ``BEM-ICE-SPHERE-HEAVE``: Lamb limit of a submerged sphere under ice.

The case tolerance is ``< 5 %`` against Lamb's infinite-fluid heave added mass
``(2/3) pi rho R**3``.  These tests pin the three legs that make that number
trustworthy:

* the reference itself -- the closed form equals half the displaced mass and the
  *independent* multipole solution (:func:`submerged_sphere_wave_multipole`)
  reaches it as ``d/R -> infinity``;
* the physics -- the finite-submergence offset decays monotonically with
  ``d/R``, so "Lamb limit" is a statement about submergence, not about
  frequency (at ``omega -> 0`` the free surface becomes a zero-potential surface
  and the added mass stays ~1.4 % above Lamb at ``d = 3R``);
* the solver -- the 3-D ice-covered BEM at the predeclared primary point is
  within the case tolerance of both Lamb and the exact multipole value.

The BEM check costs one 352-panel solve (~10-20 s); everything else is
closed form or the multipole series.
"""

from __future__ import annotations

import math

import pytest

from tensorbem.benchmarks.ice_sphere_heave import (
    CASE_TOLERANCE,
    MESH_LADDER,
    exact_heave_added_mass,
    lamb_infinite_fluid_heave_added_mass,
    panel_count,
    run_case,
)
from tensorbem.ice import DEFAULT_WATER_DENSITY

RADIUS = 5.0
RHO = DEFAULT_WATER_DENSITY


def test_panel_ladder_counts() -> None:
    """The documented 168 / 352 / 816 / 1472 ladder must be the real mesh size."""
    assert [panel_count(*mesh) for mesh in MESH_LADDER] == [168, 352, 816, 1472]


def test_lamb_reference_is_half_the_displaced_mass() -> None:
    """``(2/3) pi rho R**3`` is exactly half the displaced water mass."""
    lamb = lamb_infinite_fluid_heave_added_mass(RADIUS, RHO)
    displaced = RHO * 4.0 / 3.0 * math.pi * RADIUS ** 3
    assert lamb == pytest.approx(0.5 * displaced, rel=1.0e-14)
    assert lamb == pytest.approx(268344.3724941281, rel=1.0e-12)


def test_multipole_reaches_lamb_limit_with_submergence() -> None:
    """Independent route: exact multipole -> Lamb as ``d/R`` grows at low ``kR``."""
    lamb = lamb_infinite_fluid_heave_added_mass(RADIUS, RHO)
    offsets = []
    for ratio in (3.0, 5.0, 8.0, 12.0, 20.0):
        exact = exact_heave_added_mass(RADIUS, ratio * RADIUS, 1.0e-3, density=RHO)
        offsets.append((exact["added_mass"] - lamb) / lamb)
    assert all(later < earlier for earlier, later in zip(offsets, offsets[1:])), offsets
    assert abs(offsets[-1]) < 5.0e-4  # 0.05 % at d/R = 20
    # the classic d = 3R free-surface offset, quoted in the case report
    assert offsets[0] == pytest.approx(0.01404, abs=2.0e-3)


def test_finite_submergence_offset_decays_under_ice_too() -> None:
    """Same decay at the ice wavenumber ``k_i R = 0.1301`` (h_i = 0.9 m, omega = 0.5)."""
    lamb = lamb_infinite_fluid_heave_added_mass(RADIUS, RHO)
    kR = 0.1301357105974778
    offsets = [
        (exact_heave_added_mass(RADIUS, ratio * RADIUS, kR / RADIUS, density=RHO)["added_mass"] - lamb)
        / lamb
        for ratio in (3.0, 5.0, 8.0, 12.0)
    ]
    assert all(later < earlier for earlier, later in zip(offsets, offsets[1:])), offsets
    assert offsets[0] == pytest.approx(0.0296, abs=2.0e-3)
    assert abs(offsets[-1]) < 1.0e-3


def test_primary_point_meets_case_tolerance() -> None:
    """Predeclared P1/P2 gates at (d/R = 3, h_i = 0.9 m, omega = 0.5, 352 panels)."""
    payload = run_case(quick=True)
    verification = payload["verification"]
    assert verification["gate_primary_lt_5pct (P1)"] is True
    assert verification["gate_mesh_vs_exact_lt_5pct (P2)"] is True
    assert verification["primary_error_vs_lamb"] < CASE_TOLERANCE
    row = payload["rows"][0]
    assert row["kernel_at_ki_identity_relative_difference"] == pytest.approx(0.0, abs=1.0e-12)
    assert row["relative_residual"] < 1.0e-3
    assert len(row["surface_potential"]["phi_heave_real"]) == row["panels"]