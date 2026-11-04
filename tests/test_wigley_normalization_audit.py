"""Guards for the Wigley normalization audit (m3125).

Each test pins one *measured* property of the released code so that neither the
code nor the documentation can drift silently:

* the Kochin path is **not** dimensionally homogeneous (output scales as
  ``kg s^2 m^-1``); fixing it must break ``test_kochin_path_is_not_homogeneous``
  on purpose, which is the signal to update the report;
* the pressure path **is** homogeneous but returns a **negative** resistance at
  every Froude number and does not converge under mesh refinement;
* both paths pass geometric similarity, which is why the repository's existing
  convergence studies could not detect either defect;
* the documented ``Cw`` values lie below the smallest dimensionally admissible
  corrected value, so they must not be quoted as a physical wave-resistance
  coefficient;
* the audit verdict stays ``WITHHELD`` for the absolute magnitude.
"""

from __future__ import annotations

import math

import pytest

from tensorbem import wigley
from tensorbem.benchmarks import wigley_normalization_audit as audit

G = audit.DEFAULT_GRAVITY
RHO = audit.DEFAULT_DENSITY
L, B, T = 2.0, 0.2, 0.125


def test_kochin_path_is_not_homogeneous():
    """Kochin output responds as kg s^2 m^-1, not as a force."""
    dims = audit.implied_dimensions("kochin")
    assert dims["mass"] == pytest.approx(1.0, abs=1e-3)
    assert dims["length"] == pytest.approx(-1.0, abs=1e-3)
    assert dims["time"] == pytest.approx(2.0, abs=1e-3)

    res = audit.unit_rescaling_test("kochin")
    assert res.homogeneous is False
    # ratio of ratios is exactly sl^-2 st^4 = g^2 in the chosen unit change
    gap = res.ratio / res.required_ratio
    expected = audit.UNIT_CHANGE["time"] ** 4 / audit.UNIT_CHANGE["length"] ** 2
    assert gap == pytest.approx(expected, rel=1e-10)


def test_pressure_path_is_homogeneous():
    """The pressure path responds exactly as a force under a unit change."""
    dims = audit.implied_dimensions("pressure")
    assert dims["mass"] == pytest.approx(1.0, abs=1e-3)
    assert dims["length"] == pytest.approx(1.0, abs=1e-3)
    assert dims["time"] == pytest.approx(-2.0, abs=1e-3)
    assert audit.unit_rescaling_test("pressure").homogeneous is True


def test_both_paths_pass_geometric_similarity():
    """Similarity test is blind: both give exactly s^3 (this is the trap)."""
    rows = audit.scale_invariance(s=2.0)
    for path in ("kochin", "pressure"):
        assert rows[path]["rel_error"] < 1e-9, path


@pytest.mark.parametrize("froude", [0.20, 0.25, 0.30, 0.35, 0.40])
def test_pressure_path_resistance_is_negative(froude):
    """Physically impossible: a hull in steady forward motion does positive work."""
    U = froude * math.sqrt(G * L)
    mesh = wigley.wigley_hull_mesh(L, B, T, nx=16, nz=8)
    rw = wigley.wave_resistance_pressure(mesh, L, B, T, U, rho=RHO, g=G)
    assert rw < 0.0, f"Fr={froude}: Rw_pressure={rw} - if this now passes, update docs/m3125"


def test_pressure_path_is_not_mesh_convergent():
    """|Rw| shrinks with no limit, sign never changes -> no converged value."""
    lad = audit.pressure_mesh_ladder(subdivisions=((8, 4), (16, 8), (24, 12), (32, 16)))
    vals = [r["Rw_pressure"] for r in lad["rows"]]
    assert all(v < 0.0 for v in vals)
    mags = [abs(v) for v in vals]
    # monotone decrease, and still far from settled at the finest mesh
    assert all(mags[i] > mags[i + 1] for i in range(len(mags) - 1))
    assert (mags[-2] - mags[-1]) / mags[-2] > 0.05


def test_documented_cw_is_below_the_admissible_band():
    """The documented Cw cannot be a physical coefficient: it is off by g^2/C."""
    lad = audit.froude_ladder(froudes=(0.30,), nx=16, nz=8)
    row = lad["rows"][0]
    documented = row["Cw_kochin_S_wet"]
    smallest_admissible = row["Cw_corrected_band"]["C=4pi"]  # most conservative C
    assert documented < smallest_admissible / 5.0, (
        f"documented Cw={documented:.3e} vs smallest admissible "
        f"{smallest_admissible:.3e}"
    )


def test_kochin_geometric_family_reproduces_documented_numbers():
    """The documented Cw values are reproducible from the release (they are real
    outputs of the code - they are just not homogeneous)."""
    lad = audit.froude_ladder(froudes=(0.30,), nx=16, nz=8)
    row = lad["rows"][0]
    assert row["Rw_kochin"] == pytest.approx(9.8341e-02, rel=1e-4)
    assert row["Cw_kochin_S_wet"] == pytest.approx(1.827e-04, rel=1e-3)


def test_verdict_is_withheld():
    """No absolute magnitude may be claimed until C is pinned by a baseline."""
    assert audit.HONEST_VERDICT["absolute_cw_magnitude"] == "WITHHELD"
    assert audit.HONEST_VERDICT["kochin_path_dimensionally_homogeneous"] is False
    assert audit.HONEST_VERDICT["pressure_path_dimensionally_homogeneous"] is True
    assert audit.DIMENSIONAL_GAP["prefactor_ratio_dims"].startswith("g^2")