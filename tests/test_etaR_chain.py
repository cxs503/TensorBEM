"""Gates for the W31 ``eta_R`` chain (mission 22286).

The tests fall in three groups.

1. **Module invariants** (:mod:`tensorbem.wake_field`): the wake profile's area
   mean equals the requested wake fraction, the inflow field conserves the mean
   advance ratio, and the ``eta_R`` helper refuses the tautology it replaces.
2. **The four predeclared gates C1-C4**, recomputed from the archived artefact
   ``benchmarks/artifacts/w31_openwater_etaR_chain.json``.  The gates are
   recomputed here from the stored numbers instead of being trusted: a verdict
   string that disagreed with its own table would fail these tests.  C4 is
   carried as ``xfail(strict=True)`` because it is an **honest FAIL** (worst
   ``|dK_T| = 29.13 %`` at ``J = 0.8``, gate 15 %): the strict xfail makes the
   honest FAIL part of the test suite and turns red the day the number starts
   passing without the document being updated.
3. **Anti-degeneracy**: the null control (uniform wake) must return ``eta_R =
   1`` exactly, and the artefact must be reproducible from the archive (a
   subset of states is re-solved and compared bit for bit).

Run::

    PYTHONPATH=src python -m pytest tests/test_etaR_chain.py -q
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "benchmarks" / "artifacts" / "w31_openwater_etaR_chain.json"

C1_THRESHOLD = 1e-3
C2_MIN_STATES = 9
C2_WINDOW = (0.90, 1.10)
C3_THRESHOLD = 1e-6
C4_WORST_DKT_PCT = 15.0
C4_STRICT_PCT = 3.0
C4_REGION_MAX_J = 0.8


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT.exists(), "the W31 artefact is missing: run benchmarks/anchors/build_w31_etaR_chain.py"
    return json.loads(ARTIFACT.read_text())


# ---------------------------------------------------------------------------
# 1. module invariants
# ---------------------------------------------------------------------------


def test_wake_profile_mean_equals_requested_fraction():
    """The area-weighted mean of ``w(r)`` must be the wake fraction that was asked for.

    Tolerance 1e-5, not machine precision: the check integrates ``(1-xi)**q``
    with a trapezoid rule and that integrand has an infinite slope at the tip,
    so the *test's* quadrature -- not the module, which uses the closed-form
    annulus mean -- limits the agreement to a few 1e-6.
    """

    from tensorbem.wake_field import nominal_wake_fraction

    hub = 0.18
    xi = np.linspace(0.0, 1.0, 400001)
    radius_ratio = hub + xi * (1.0 - hub)
    weight = radius_ratio
    for exponent in (0.0, 0.25, 0.5, 1.0):
        w = nominal_wake_fraction(radius_ratio, 0.3038, exponent=exponent, hub_ratio=hub)
        mean = float(np.sum(w * weight) / np.sum(weight))
        assert mean == pytest.approx(0.3038, rel=1e-5), f"q={exponent} mean {mean}"
        assert np.all(w >= -1e-15)


def test_wake_inflow_conserves_mean_advance_ratio():
    """The axial component of the wake inflow must average to ``J`` (mass flux)."""

    from tensorbem.wake_field import wake_inflow

    hub = 0.18
    advance = 0.5372
    xi = np.linspace(0.0, 1.0, 400001)
    radius_ratio = hub + xi * (1.0 - hub)
    # blade radii of the D = 1 lattice: r = (D/2) * r/R
    radius = 0.5 * radius_ratio
    points = np.stack([np.zeros_like(radius), radius, np.zeros_like(radius)], axis=-1)
    for exponent in (0.0, 0.25, 0.5, 1.0):
        field = wake_inflow(advance, 0.3038, exponent=exponent, hub_ratio=hub, diameter=1.0)
        axial = field(points)[:, 0]
        mean = float(np.sum(axial * radius) / np.sum(radius))
        assert mean == pytest.approx(advance, rel=1e-5), f"q={exponent} mean J_x {mean}"


def test_relative_rotative_efficiency_refuses_the_tautology():
    """Equal numerator and denominator is the platform failure mode: it must not return 1."""

    from tensorbem.wake_field import relative_rotative_efficiency

    with pytest.warns(RuntimeWarning, match="carries no information"):
        assert math.isnan(relative_rotative_efficiency(0.55, 0.55))
    assert relative_rotative_efficiency(0.55, 0.60) == pytest.approx(0.55 / 0.60)


def test_interpolation_reproduces_ladder_nodes():
    from tensorbem.wake_field import interpolate_open_water_curve

    js = [0.10, 0.15, 0.20, 0.25]
    vs = [0.42, 0.40, 0.37, 0.34]
    for j, v in zip(js, vs):
        assert interpolate_open_water_curve(js, vs, j) == pytest.approx(v, abs=1e-15)
    assert interpolate_open_water_curve(js, vs, 0.175) == pytest.approx(0.385, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. the predeclared gates
# ---------------------------------------------------------------------------


def test_c1_etar_is_not_a_constant(artifact):
    """C1: std of the computed eta_R over the 11 states must exceed 1e-3."""

    values = np.array([s["etaR"]["openwater_interp"] for s in artifact["states"]])
    assert values.size == 11
    assert np.std(values) > C1_THRESHOLD, f"std={np.std(values):.3e} is at or below 1e-3"

    # the same gate on the platform's own series has to fail; if it ever passes,
    # the platform stopped returning a constant and this whole de-constantisation
    # exercise has to be re-scoped
    platform = np.array([s["etaR"]["platform_returned"] for s in artifact["states"]])
    assert np.std(platform) == 0.0
    assert not np.std(platform) > C1_THRESHOLD


def test_c2_physical_window(artifact):
    """C2: at least 9 of 11 computed eta_R inside [0.90, 1.10]."""

    values = np.array([s["etaR"]["openwater_interp"] for s in artifact["states"]])
    inside = int(np.sum((values >= C2_WINDOW[0]) & (values <= C2_WINDOW[1])))
    assert inside >= C2_MIN_STATES, f"{inside}/11 inside {C2_WINDOW}: {values}"


def test_c3_identity_closes_on_every_state(artifact):
    """C3: eta_D = eta_0 eta_R eta_H, relative residual <= 1e-6 per state."""

    worst = 0.0
    for row in artifact["identity_table"]:
        product = row["eta0"] * row["etaR"] * row["etaH"]
        residual = abs(product - row["etaD_chain"]) / abs(product)
        worst = max(worst, residual)
        assert residual <= C3_THRESHOLD
    assert worst <= C3_THRESHOLD
    # and the stored residual column must agree with the recomputation
    assert max(abs(r["relative_residual"] - abs(r["eta0"] * r["etaR"] * r["etaH"] - r["etaD_chain"])
                   / abs(r["eta0"] * r["etaR"] * r["etaH"])) for r in artifact["identity_table"]) < 1e-15


def test_c3_is_not_evidence_of_correctness(artifact):
    """The identity is *definitional* for eta_R, so it must not be sold as a check.

    The artefact has to carry the number that shows what the identity cannot see:
    the difference between the chain with a computed eta_R and the platform's own
    eta_D (which uses the constant 1.0).
    """

    diffs = [abs(r["chain_vs_platform_pct"]) for r in artifact["identity_table"]]
    assert max(diffs) > 1.0, "if chain and platform agreed to <1 % the eta_R change would be invisible"
    assert artifact["verdicts"]["C3"]["note"].strip()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "C4 is an honest FAIL: worst |dK_T| vs the KP505 measured curve over J<=0.8 is "
        "29.13 % at J=0.8 (gate 15 %). Remove this xfail only together with a regenerated "
        "artefact and an updated doc."
    ),
)
def test_c4_point_by_point_against_kp505(artifact):
    """C4: over J <= 0.8 the worst |dK_T| must be below 15 %."""

    rows = [r for r in artifact["kp505_comparison"]["rows"] if r["J"] <= C4_REGION_MAX_J + 1e-12]
    assert rows, "no KP505 point inside the gate region"
    worst = max(abs(r["dKT_pct_VLM"]) for r in rows)
    assert worst < C4_WORST_DKT_PCT, f"worst |dK_T| = {worst:.2f} %"


def test_c4_every_point_is_answered(artifact):
    """C4's second half: each point answered for < 3 %, no point dropped."""

    rows = artifact["kp505_comparison"]["rows"]
    assert len(rows) == 10
    for row in rows:
        for key in ("lt3pct_VLM_KT", "lt3pct_VLM_10KQ", "lt3pct_B_series_KT"):
            assert isinstance(row[key], bool)
    in_region = [r for r in rows if r["J"] <= C4_REGION_MAX_J + 1e-12]
    decided = [r for r in in_region if abs(r["dKT_pct_VLM"]) < C4_STRICT_PCT]
    assert len(decided) == artifact["kp505_comparison"]["n_points_lt3pct_VLM_KT"]
    # the verdict recorded in the artefact must be recomputable from its own table
    worst = max(abs(r["dKT_pct_VLM"]) for r in in_region)
    expected = "PASS" if worst < C4_WORST_DKT_PCT else "honest FAIL"
    assert artifact["verdicts"]["C4"]["verdict"] == expected


# ---------------------------------------------------------------------------
# 3. anti-degeneracy and reproducibility
# ---------------------------------------------------------------------------


def test_null_control_is_exactly_unity(artifact):
    """A uniform wake has to reproduce the open-water point: eta_R = 1 to machine precision."""

    controls = artifact["control_null"]
    assert len(controls) >= 2
    for control in controls:
        assert control["etaR_openwater_direct"] == pytest.approx(1.0, abs=1e-12)
        assert control["KT_open_water"] == pytest.approx(control["KT_behind_hull"], rel=1e-15)
        # the interpolation bias is a separate, quantified effect -- not a hidden one
        assert abs(control["interpolation_bias_pct"]) < 0.5
    assert any("no information" in msg for c in controls for msg in c["tautology_guard_messages"])


def test_wake_shape_sensitivity_is_monotone(artifact):
    """A stronger inboard deficit must not make eta_R wander outside the window."""

    rows = artifact["wake_shape_sensitivity"]
    for state in {r["state"] for r in rows}:
        series = sorted((r["wake_exponent"], r["etaR"]) for r in rows if r["state"] == state)
        values = [v for _, v in series]
        assert values == sorted(values, reverse=True), f"{state}: eta_R is not monotone in q: {values}"
        assert all(C2_WINDOW[0] <= v <= C2_WINDOW[1] for v in values)


def test_panel_sensitivity_is_bounded(artifact):
    """eta_R must be stable against panel density, otherwise it is a mesh artefact."""

    rows = artifact["panel_sensitivity"]
    for state in {r["state"] for r in rows}:
        series = {r["n_panels_per_blade"]: r["etaR"] for r in rows if r["state"] == state}
        assert set(series) == {8, 32, 48}
        spread = max(series.values()) - min(series.values())
        assert spread < 0.02, f"{state}: panel spread {spread:.4f}"
        assert abs(series[32] - series[48]) < 2e-3, f"{state}: 16x2 vs 24x2 drift {abs(series[32]-series[48]):.2e}"


def test_artifact_reproduces_from_source(artifact):
    """Re-solve three states (both pitches) and compare with the archive bit for bit."""

    import importlib.util
    import warnings

    spec = importlib.util.spec_from_file_location(
        "build_w31_etaR_chain", REPO / "benchmarks" / "anchors" / "build_w31_etaR_chain.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    receipts = json.loads(Path(module.RECEIPTS).read_text())
    cache: dict = {}
    archived = {s["id"]: s for s in artifact["states"]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for state in receipts["states"]:
            if state["id"] not in ("S1", "S6", "S11"):
                continue
            fresh = module.state_chain(state, cache, exponent=0.5)
            old = archived[state["id"]]
            assert fresh["etaR"]["openwater_interp"] == pytest.approx(
                old["etaR"]["openwater_interp"], abs=1e-12
            )
            assert fresh["behind_hull"]["KT"] == pytest.approx(old["behind_hull"]["KT"], abs=1e-12)


def test_platform_audit_shows_the_constant_is_structural(artifact):
    """The platform's K_T/K_Q must still be its own open-water curve."""

    audit = artifact["platform_block_audit"]
    assert audit["worst_abs_dKT"] < 1e-3
    assert audit["worst_abs_dKQ"] < 1e-4
    assert artifact["etar_summary"]["std_platform_returned"] == 0.0
    assert artifact["etar_summary"]["n_in_window_power_route_reverse"] <= 3