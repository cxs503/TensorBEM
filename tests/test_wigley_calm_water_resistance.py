"""Pins for case ``CASE-BEM-WIGLEY-CALM-WATER-RESISTANCE``.

The case driver (:mod:`tensorbem.benchmarks.wigley_calm_water_resistance`)
publishes a Wigley III calm-water resistance study: a geometry gate, four
independent routes to the Michell wave resistance, the ITTC-1957/Holtrop
friction protocol, and the comparison of the assembled total resistance with the
Holtrop (1984) regression.

Each item is a separate test because each fails for its own reason:

* the geometry gate -- if the panelised hull does not reproduce ``Cb = 4/9``
  every downstream resistance number describes a different ship;
* the ITTC-1957 line against hand-computable values;
* the *coefficient transcription* of the Holtrop chain, pinned against a
  published intermediate-coefficient table for the KCS hullform (public
  particulars ``Lwl = 232.5 m``, ``B = 32.2 m``, ``T = 10.8 m``,
  ``vol = 52030 m^3``, ``Cb = 0.6509``, ``Cm = 0.9849``, ``Cwp = 0.7630``,
  ``lcb = -1.48 %L``, ``ABT = 10 m^2``, ``hB = 5 m``).  The reference values
  come from the ship-performance-platform registration
  ``docs/prereg/EX1_KCS_prereg_v1.md`` (sections 2 and 3) and from its module
  ``docs/prereg/ex1_kcs/holtrop_mennen.py``: ``L_R = 70.54 m``,
  ``iE = 13.90 deg``, ``c7 = 0.13849``, ``c1 = 0.9909``, ``c3 = 7.510331e-3``,
  ``c2 = 0.848919``, ``c5 = 1``, ``c15 = -1.69385``, ``c16 = 1.29910``,
  ``lambda = 0.73902``, ``m1 = -1.94207``, ``C_A = 0.000320``.  They are an
  independent implementation of the same equations, so this is a
  cross-implementation check of the transcription, not a self-consistency test;
* the form factor of the two published Holtrop forms, including the open item
  that the 1984 expression evaluated here gives ``1.1656`` for KCS where the
  cited table gives ``1.1362`` (2.6 %, recorded rather than papered over);
* the homogeneous scaling of ``R_W`` with ``rho g vol`` at fixed Froude number,
  an exact property that a coefficient typo would break;
* the wave resistance from the closed-form Michell amplitude against the
  independent ``theta``-form quadrature and against the BEM centreplane sheet,
  i.e. the verification of the chain the kernel contributes;
* the size of the legacy ``tensorbem.wigley.michell_wave_resistance``
  discrepancy, so the known open item cannot be quietly "fixed" or forgotten;
* the published artefacts in ``artifacts/bem_wigley_calm_water_resistance``:
  the JSON verdicts must follow from the numbers stored beside them, and the
  report figures must be real PNG files.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import struct
import subprocess
import sys

import numpy as np
import pytest

from tensorbem.benchmarks import wigley_calm_water_resistance as case
from tensorbem.resistance import (
    DEFAULT_GRAVITY,
    holtrop_correlation_allowance,
    holtrop_form_factor,
    holtrop_half_angle_of_entrance,
    holtrop_length_of_run,
    holtrop_wave_resistance,
    ittc57_friction_coefficient,
    reynolds_number,
)
from tensorbem.thin_ship_michell import (
    WigleyHull,
    wave_resistance_from_amplitude,
    wave_resistance_theta_form,
    wigley_amplitude_function,
    wigley_panels,
)
from tensorbem.wigley import michell_wave_resistance as legacy_wave_resistance

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts" / "bem_wigley_calm_water_resistance"
HULL = WigleyHull(L=10.0, B=1.0, T=0.625)

#: KCS particulars (public KRISO/ITTC benchmark hull) used to pin the regression.
KCS = {
    "L": 232.5,
    "B": 32.2,
    "T": 10.8,
    "volume": 52030.0,
    "Cb": 0.6509,
    "Cm": 0.9849,
    "Cwp": 0.7630,
    "lcb": -1.48,
    "ABT": 10.0,
    "hB": 5.0,
}
KCS_CP = KCS["Cb"] / KCS["Cm"]


def test_geometry_gate_reproduces_wigley_closed_forms() -> None:
    """C1: the panelised hull must reproduce Cb, volume, Awp and S_w."""
    report = case.geometry_validation(n_sections=32, n_levels=16)
    assert report["closed_form"]["Cb"] == pytest.approx(4.0 / 9.0, rel=1e-12)
    assert report["closed_form"]["volume"] == pytest.approx(4.0 * 10.0 * 1.0 * 0.625 / 9.0)
    assert report["parameters"]["L_over_B"] == pytest.approx(10.0, rel=1e-12)
    assert report["parameters"]["B_over_T"] == pytest.approx(1.6, rel=1e-12)
    assert report["L_over_B_exact"] and report["B_over_T_exact"]
    assert all(report["passed"].values()), report["relative_error"]
    assert report["relative_error"]["cb"] < 5.0e-3
    assert report["relative_error"]["wetted_area"] < 2.0e-2
    mesh = report["mesh_values"]
    assert mesh["Cb"] == pytest.approx(
        mesh["volume"]
        / (
            report["parameters"]["L"]
            * report["parameters"]["B"]
            * report["parameters"]["T"]
        ),
        rel=1e-12,
    )
    # Fore-aft symmetry of the Wigley form: LCB and waterplane centroid at x = 0.
    assert abs(mesh["waterplane_centroid_x"]) < 1e-9
    assert mesh["center_of_buoyancy_z"] == pytest.approx(-3.0 * 0.625 / 8.0, rel=1e-3)


def test_ittc57_friction_line_matches_closed_form() -> None:
    """C4: C_F against the hand-computable values of 0.075/(log10 Re - 2)^2."""
    assert ittc57_friction_coefficient(1.0e6) == pytest.approx(0.075 / 16.0, rel=1e-14)
    assert ittc57_friction_coefficient(1.0e9) == pytest.approx(0.075 / 49.0, rel=1e-14)
    re = np.logspace(5, 10, 40)
    cf = np.array([ittc57_friction_coefficient(value) for value in re])
    assert np.all(np.diff(cf) < 0.0)
    assert reynolds_number(2.9708559, 10.0, 1.19e-6) == pytest.approx(2.4965e7, rel=1e-4)
    with pytest.raises(ValueError):
        ittc57_friction_coefficient(100.0)
    with pytest.raises(ValueError):
        reynolds_number(1.0, 0.0)


def test_holtrop_coefficients_match_published_kcs_table() -> None:
    """Cross-implementation pin of the regression transcription (KCS hullform)."""
    reference = {
        "length_of_run": 70.54,
        "iE_deg": 13.90,
        "c7": 0.13849,
        "c1": 0.9909,
        "c3": 7.510331e-03,
        "c2": 0.848919,
        "c5": 1.0,
        "c15": -1.69385,
        "c16": 1.29910,
        "lam": 0.73902,
        "m1": -1.94207,
    }
    speed = 0.26 * math.sqrt(DEFAULT_GRAVITY * 230.0)  # Fn = 0.26 on Lpp
    wave = holtrop_wave_resistance(
        KCS["L"],
        KCS["B"],
        KCS["T"],
        KCS["volume"],
        KCS_CP,
        KCS["Cwp"],
        KCS["Cm"],
        speed,
        lcb_percent=KCS["lcb"],
        bulb_area=KCS["ABT"],
        bulb_centre_height=KCS["hB"],
        include_c17=False,
    )
    assert wave["length_of_run"] == pytest.approx(
        holtrop_length_of_run(KCS["L"], KCS_CP, KCS["lcb"]), rel=1e-12
    )
    for key, expected in reference.items():
        assert wave[key] == pytest.approx(expected, rel=2.0e-3), key
    assert holtrop_half_angle_of_entrance(
        KCS["L"], KCS["B"], KCS["T"], KCS["volume"], KCS_CP, KCS["Cwp"], KCS["lcb"]
    ) == pytest.approx(reference["iE_deg"], rel=2.0e-3)
    assert holtrop_correlation_allowance(
        KCS["L"], KCS["Cb"], KCS["T"], bulb_coefficient=wave["c2"], convention="holtrop84"
    ) == pytest.approx(0.000320, rel=2.0e-2)
    # A bulbous bow lowers c2 below one, i.e. lowers the wave resistance.
    bare = holtrop_wave_resistance(
        KCS["L"],
        KCS["B"],
        KCS["T"],
        KCS["volume"],
        KCS_CP,
        KCS["Cwp"],
        KCS["Cm"],
        speed,
        lcb_percent=KCS["lcb"],
        include_c17=False,
    )
    assert bare["c2"] == pytest.approx(1.0)
    assert wave["Rw"] < bare["Rw"]


def test_holtrop_form_factor_forms_and_open_item() -> None:
    """The two published 1+k expressions, and the recorded 2.6 % open item."""
    kcs_1984 = holtrop_form_factor(
        KCS["L"], KCS["B"], KCS["T"], KCS["volume"], KCS_CP, KCS["lcb"], form="1984"
    )
    kcs_1982 = holtrop_form_factor(
        KCS["L"], KCS["B"], KCS["T"], KCS["volume"], KCS_CP, KCS["lcb"], form="1982"
    )
    # Our 1984 evaluation follows the printed exponents ...
    assert kcs_1984 == pytest.approx(1.1656, rel=1e-3)
    # ... and differs from the 1.1362 of the cited registration by 2.6 %, which is
    # an open item (both transcriptions claim the same equation); it does not
    # affect the case comparison, where the form factor is common to both routes.
    assert abs(kcs_1984 - 1.1362) / 1.1362 < 0.03
    assert 0.95 < kcs_1982 < 1.05  # the 1982 bracket has a 0.93 floor
    assert kcs_1982 < kcs_1984
    wigley_volume = 4.0 * 10.0 * 1.0 * 0.625 / 9.0
    wigley = holtrop_form_factor(10.0, 1.0, 0.625, wigley_volume, 2.0 / 3.0, form="1984")
    assert wigley == pytest.approx(1.1503, rel=1e-3)
    # Stern correction raises the form factor (c14 = 1 + 0.011 C_stern).
    assert (
        holtrop_form_factor(
            10.0, 1.0, 0.625, wigley_volume, 2.0 / 3.0, c_stern=10.0, form="1984"
        )
        > wigley
    )
    # Correlation-allowance conventions: zero by default, positive otherwise, and
    # the ITTC-78 allowance dominates the friction line at model scale.
    assert holtrop_correlation_allowance(10.0, 4.0 / 9.0, 0.625) == 0.0
    assert holtrop_correlation_allowance(10.0, 4.0 / 9.0, 0.625, convention="holtrop84") > 0.0
    assert holtrop_correlation_allowance(10.0, 4.0 / 9.0, 0.625, convention="ittc78") > 1.0e-3
    assert ittc57_friction_coefficient(reynolds_number(2.97, 10.0)) < 3.0e-3


def test_holtrop_wave_resistance_scales_with_displacement() -> None:
    """``R_W`` is homogeneous of degree one in ``rho g vol`` at fixed Fn."""
    speed = 0.30 * math.sqrt(DEFAULT_GRAVITY * 10.0)
    arguments = dict(lcb_percent=0.0, include_c17=True, regime="m4")
    base = holtrop_wave_resistance(
        10.0, 1.0, 0.625, 4.0 * 10.0 * 1.0 * 0.625 / 9.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, speed,
        **arguments,
    )
    doubled = holtrop_wave_resistance(
        20.0, 2.0, 1.25, 8.0 * 4.0 * 10.0 * 1.0 * 0.625 / 9.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0,
        0.30 * math.sqrt(DEFAULT_GRAVITY * 20.0), **arguments,
    )
    assert doubled["Rw"] / base["Rw"] == pytest.approx(8.0, rel=1e-9)
    # The c17 convention is a plain multiplier of the wave term.
    without_c17 = holtrop_wave_resistance(
        10.0, 1.0, 0.625, 4.0 * 10.0 * 1.0 * 0.625 / 9.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, speed,
        lcb_percent=0.0, include_c17=False,
    )
    assert base["Rw"] / without_c17["Rw"] == pytest.approx(base["c17"], rel=1e-12)
    assert 50.0 < base["Rw"] < 400.0  # same order as the kernel at Fn = 0.30


def test_wave_resistance_routes_agree_at_froude_0p30() -> None:
    """C2/C3: closed form vs theta form vs BEM centreplane sheet."""
    speed = 0.30 * math.sqrt(DEFAULT_GRAVITY * HULL.L)
    amplitude = wigley_amplitude_function(HULL)
    k0 = DEFAULT_GRAVITY / speed**2
    lam_max = 20.0
    closed = wave_resistance_from_amplitude(amplitude, speed, lam_max=lam_max, n_lam=240)
    theta = wave_resistance_theta_form(
        lambda angle: amplitude(k0 / np.cos(angle) ** 2, k0 / np.cos(angle)),
        speed,
        theta_max=math.acos(1.0 / lam_max),
        n_theta=400,
        order=8,
    )
    assert closed == pytest.approx(144.14, rel=1e-3)  # published anchor, 144.1 N
    assert abs(theta / closed - 1.0) < 1.0e-2
    coarse = wave_resistance_from_amplitude(
        wigley_panels(HULL, 16, 8).amplitude, speed, lam_max=lam_max, n_lam=240
    )
    fine = wave_resistance_from_amplitude(
        wigley_panels(HULL, 32, 16).amplitude, speed, lam_max=lam_max, n_lam=240
    )
    assert abs(fine / closed - 1.0) < abs(coarse / closed - 1.0)
    assert abs(fine / closed - 1.0) < 2.0e-2


def test_legacy_wigley_path_stays_an_order_of_magnitude_low() -> None:
    """Known open item: the legacy path returns 1/12 of the standard Michell value.

    ``tensorbem.wigley.michell_wave_resistance`` carries ``lambda^-2`` instead of
    ``lambda^2`` in the weight and drops the ``g^2`` prefactor, so it returns
    ``12.29 N`` where the standard form gives ``144.14 N`` at ``Fn = 0.30``.
    This test does not bless that value -- it pins the size of the discrepancy so
    the report, the ledger and the code keep referring to the same fact.
    """
    speed = 0.30 * math.sqrt(DEFAULT_GRAVITY * HULL.L)
    legacy = legacy_wave_resistance(HULL.L, HULL.B, HULL.T, speed)
    standard = wave_resistance_from_amplitude(
        wigley_amplitude_function(HULL), speed, lam_max=20.0, n_lam=240
    )
    assert legacy == pytest.approx(12.29, rel=5e-3)
    assert standard == pytest.approx(144.14, rel=1e-3)
    assert 0.05 < legacy / standard < 0.15


def test_study_criteria_are_self_consistent() -> None:
    """A reduced scan: every JSON verdict must follow from the stored numbers."""
    study = case.resistance_scan(froude_numbers=(0.20, 0.30))
    criteria = case.criteria_summary(study)
    assert criteria["C1_geometry"]["passed"] is True
    assert criteria["C2_wave_resistance_routes"]["passed"] is True
    assert criteria["C3_panel_convergence"]["passed"] is True
    assert criteria["C4_ittc57_identity"]["passed"] is True
    assert criteria["C4_ittc57_identity"]["max_rel_deviation"] <= 1.0e-12
    empirical = criteria["C5_empirical_total_resistance"]
    for row in study["rows"]:
        key = f"Fn={row['froude_number']:.2f}"
        error = row["rel_error_total_vs_holtrop"]
        assert empirical["measured"][key] == pytest.approx(error, rel=1e-12)
        assert empirical["passed_map"][key] is bool(abs(error) <= case.LIMITS["empirical"])
        # Components must add up to the totals that were compared.
        assembled = row["R_F_times_form_factor"] + row["R_A"] + row["Rw_michell"]
        assert assembled == pytest.approx(row["R_T_case"], rel=1e-12)
        holtrop_assembled = row["R_F_times_form_factor"] + row["R_A"] + row["Rw_holtrop"]
        assert holtrop_assembled == pytest.approx(row["R_T_holtrop"], rel=1e-12)
    grid = case.variant_grid(study)
    assert len(grid["variants"]) == 2 * 2 * 3
    for variant in grid["variants"]:
        assert variant["passed_count"] == sum(
            1
            for value in variant["rel_error_total_vs_holtrop"].values()
            if abs(value) <= case.LIMITS["empirical"]
        )


def test_published_artifacts_match_their_verdicts() -> None:
    """The committed results.json must agree with its own numbers."""
    results = ARTIFACTS / "results.json"
    if not results.exists():  # pragma: no cover - artefacts are committed
        pytest.skip("results.json not generated")
    study = json.loads(results.read_text(encoding="utf-8"))
    criteria = study["criteria"]
    assert sorted(criteria) == [
        "C1_geometry",
        "C2_wave_resistance_routes",
        "C3_panel_convergence",
        "C4_ittc57_identity",
        "C5_empirical_total_resistance",
    ]
    for key in (
        "C1_geometry",
        "C2_wave_resistance_routes",
        "C3_panel_convergence",
        "C4_ittc57_identity",
    ):
        assert criteria[key]["passed"] is True, key
    band = [row for row in study["rows"] if row["in_criterion_band"]]
    assert [row["froude_number"] for row in band] == list(case.CRITERION_FROUDE)
    assert criteria["C5_empirical_total_resistance"]["convention"] == case.PRIMARY_CONVENTION
    c5 = criteria["C5_empirical_total_resistance"]
    # C5 is an honest FAIL in the published study: the empirical baseline is not
    # a <=10 % yardstick for this hullform.  If the physics ever changes, the
    # report and the artefacts have to be reconciled together.
    assert c5["passed"] is False
    assert c5["passed_count"] == sum(c5["passed_map"].values())
    assert c5["passed_count"] < c5["evaluated_points"]
    assert study["variant_grid"]["passed_count_range"][0] < c5["evaluated_points"]
    assert study["case"]["evidence_level"]["total_resistance_vs_experiment"].startswith("WITHHELD")
    for figure in study["figures"]:
        path = pathlib.Path(figure["path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        assert path.exists(), path
        raw = path.read_bytes()
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = struct.unpack(">II", raw[16:24])
        assert width > 400 and height > 300


def test_benchmark_driver_help_exits_zero() -> None:
    """The driver is the reproduction path of the report tables."""
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    completed = subprocess.run(
        [sys.executable, "-m", "tensorbem.benchmarks.wigley_calm_water_resistance", "--help"],
        capture_output=True,
        env=env,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    assert b"--output" in completed.stdout