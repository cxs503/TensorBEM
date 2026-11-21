"""Regression checks for the ship resistance and free-edge plate benchmarks."""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from tensorbem.added_resistance import bow_reflection, radiated_resistance
from tensorbem.floe_scattering import scatter_floe

ROOT = Path(__file__).resolve().parents[1]


def table(case, name):
    with (ROOT / "benchmarks" / case / (name + ".csv")).open() as stream:
        return list(csv.DictReader(stream))


def test_sectional_work_integral_and_encounter_convention():
    x = np.array([0.0, 1.0, 2.0])
    added_mass = np.array([0.0, 1.0, 0.0])
    damping = np.ones(3)
    args = dict(k=2.0, omega_e=4.0, speed=1.0)
    # |relative velocity|² = [1,4,9]. The piecewise integral is 9 and
    # the convection term is .5 * (3+5) = 4, prior to k/(2 omega_e).
    assert radiated_resistance(
        x, added_mass, damping, -np.array([1.0, 2.0, 3.0]), [0, 0], **args
    ) == pytest.approx(13 / 4)
    heave, pitch = 0.7 + 1j, 0.2 - 0.1j
    following = -4j * (heave - x * pitch) + pitch
    assert radiated_resistance(
        x, added_mass, damping, following, [heave, pitch], **args
    ) == pytest.approx(0)
    with pytest.raises(ValueError):
        radiated_resistance(x, np.ones(3), damping, following, [heave, pitch], **args)


def test_short_wave_reflection_and_projected_bow():
    bow = np.array([[1.0, 0.0], [0.0, 1.0]])
    kwargs = dict(draft=1.0, froude=0.04, density=1000.0, gravity=10.0, amplitude=0.5)
    # 45-degree half-bow twice integrated gives a projected breadth of one.
    assert bow_reflection(bow, k=1000, **kwargs) == pytest.approx(2500.0)
    assert bow_reflection(bow, k=0.001, **kwargs) < 1e-7
    with pytest.raises(ValueError):
        bow_reflection(bow[::-1], k=1.0, **kwargs)


def test_transparent_plate_matches_open_strip_phase_under_refinement():
    # Choose k and derive omega independently of the solver's dispersion inversion.
    k, depth, length = 4.0, 0.9, 1.0
    omega = np.sqrt(9.81 * k * np.tanh(k * depth))
    errors = []
    for n in (12, 24):
        v = scatter_floe(
            omega, length=length, depth=depth, rigidity=0, mass_per_area=0, panels=n
        )
        errors.append(
            [abs(v.reflection), abs(v.transmission - np.exp(3j * length * k))]
        )
        assert v.energy_defect < 1e-9
    assert np.all(np.array(errors[1]) < 0.4 * np.array(errors[0]))
    assert errors[1][1] < 0.03


def test_measured_floe_row_and_joint_energy_gates():
    case = "floe_transmission"
    obs = table(case, "observations")
    assert Counter(r["configuration"] for r in obs) == {
        "free": 30,
        "moored": 30,
        "barriers": 30,
    }
    assert max(abs(float(r["ka"]) - float(r["ka_R"])) for r in obs) <= 0.001
    rows = table(case, "results")
    r = next(
        r
        for r in rows
        if r["material"] == "PVC" and r["period"] == "0.9" and r["panels"] == "12"
    )
    v = scatter_floe(
        2 * np.pi / 0.9,
        length=1.0,
        depth=0.9,
        rigidity=5e8 * 0.01**3 / (12 * (1 - 0.4**2)),
        mass_per_area=5.69,
        panels=12,
    )
    assert float(r["predicted_energy_R"]) == pytest.approx(
        abs(v.reflection) ** 2, abs=1e-7
    )
    assert float(r["predicted_energy_T"]) == pytest.approx(
        abs(v.transmission) ** 2, abs=1e-7
    )
    assert v.energy_defect < 1e-7 and v.relative_residual < 1e-7
    scores = table(case, "scores")
    for s in scores:
        group = [
            r
            for r in rows
            if (r["configuration"], r["panels"]) == (s["configuration"], s["panels"])
        ]
        err = np.array(
            [
                float(r["predicted_energy_" + s["quantity"]])
                - float(r["observed_energy_" + s["quantity"]])
                for r in group
            ]
        )
        assert len(err) == 30
        rms = np.sqrt(np.mean(err**2))
        assert rms == pytest.approx(float(s["rms"]), abs=1e-9)
        assert s["verdict"] == (
            "PASS" if rms <= 0.10 and max(abs(err)) <= 0.20 else "FAIL"
        )
    assert {
        (s["configuration"], s["quantity"])
        for s in scores
        if s["panels"] == "72" and s["verdict"] == "PASS"
    } == {("free", "T"), ("barriers", "T"), ("barriers", "R")}


def test_all_original_wigley_trials_and_live_baseline(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.journee_added_resistance import run

    all_trials = table("journee_added_resistance", "trials")
    trials = [r for r in all_trials if r["source_table"] == "8"]
    zero = [r for r in all_trials if r["source_table"] == "10"]
    assert Counter(r["hull"] for r in zero) == {"III": 13, "IV": 7}
    assert [
        float(r["observed"])
        for r in zero
        if r["hull"] == "III" and r["lambda_L"] == "1.25"
    ] == [2.92, 1.12, 0.80]
    assert Counter(r["hull"] for r in trials) == {
        "I": 34,
        "II": 22,
        "III": 64,
        "IV": 32,
    }
    assert (
        len(
            {
                tuple(r[k] for k in ("hull", "Fn", "source_line", "source_group"))
                for r in trials
            }
        )
        == 152
    )
    assert sum(float(r["observed"]) < 0 for r in trials) == 19
    assert sum(not (r["pitch_deg"] and r["heave_deg"]) for r in trials) == 3
    records = table("journee_added_resistance", "results")
    t = next(t for t in trials if not (t["pitch_deg"] and t["heave_deg"]))
    result = run.calculate(
        (t["hull"], float(t["Fn"]), float(t["lambda_L"]), [t], run.GRIDS[0])
    )[0]
    saved = next(
        r
        for r in records
        if r["panels"] == str(result["panels"]) and all(r[k] == t[k] for k in t)
    )
    assert result["with_observed_motion"] == ""
    assert result["predicted"] == pytest.approx(float(saved["predicted"]), rel=1e-8)
    for key in ("native_heave", "native_pitch", "restrained", "with_reflection"):
        assert result[key] == pytest.approx(float(saved[key]), rel=1e-8)
    assert len(records) == 344
    scores = table("journee_added_resistance", "scores")
    assert [
        (r["hull"], r["Fn"])
        for r in scores
        if float(r["Fn"]) > 0
        and r["panels"] == "480"
        and r["route"] == "predicted"
        and r["verdict"] == "PASS"
    ] == [("II", "0.2")]


def test_nmri_mesh_volume_and_live_kvlcc2_anchor(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.kvlcc2_head_seas import geometry, run

    with np.load(geometry.ROOT / "geometry.npz") as a:
        meshes = [
            geometry.mesh_from_patch(a[f"patch{n}"]) for n, _, _ in geometry.GRIDS
        ]
    assert [len(m.faces) for m in meshes] == [430, 1118]
    assert all(len(np.unique(m.faces)) == len(m.vertices) for m in meshes)
    vol = 312622.0 / 80**3
    assert abs(meshes[1].volume / vol - 1) < 0.01 < abs(meshes[0].volume / vol - 1)
    result = {r["quantity"]: r for r in run.calculate((0.3, geometry.GRIDS[0]))}
    rows = table("kvlcc2_head_seas", "results")
    for r in rows:
        if r["panels"] == "430" and float(r["lambda_L"]) == 0.3:
            assert result[r["quantity"]]["predicted"] == pytest.approx(
                float(r["predicted"]), rel=1e-7
            )
    assert len(table("kvlcc2_head_seas", "observations")) == 25
