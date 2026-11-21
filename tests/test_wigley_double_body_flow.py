"""Manufactured steady-flow limits, impedance invariances and tank-data isolation."""

import csv
from pathlib import Path

import numpy as np
import pytest

from tensorbem.double_body_flow import prepare_double_body_flow
from tensorbem.forward_speed import solve_forward_speed_heave_pitch
from tensorbem.geometry3d import hemisphere_hull_mesh
from tensorbem.wigley3_hull import wigley3_wetted_hull_mesh


def test_surge_sphere_recovers_velocity_and_translational_and_rotational_m_terms():
    errors = []
    for subdivision in (2, 3):
        mesh = hemisphere_hull_mesh(subdivisions=subdivision)
        flow = prepare_double_body_flow(mesh)
        n = mesh.centroids / np.linalg.norm(mesh.centroids, axis=1)[:, None]
        velocity = -1.5 * (np.eye(3)[0] - n[:, 0, None] * n)
        heave_m = 4.5 * n[:, 0] * n[:, 2]

        def rms(x):
            return np.sqrt(np.average(x * x, weights=mesh.areas))

        errors.append(
            [
                rms(np.linalg.norm(flow.velocity - velocity, axis=1)),
                rms(flow.translation_m[:, 2] - heave_m),
                rms(flow.rotation_m[:, 1]),
            ]
        )
        assert np.max(abs(np.sum(flow.velocity * mesh.normals, axis=1))) < 1e-12
        assert flow.relative_residual < 1e-12
    assert np.all(np.array(errors[1]) < np.array(errors[0]))
    assert np.all(np.array(errors[1]) < [0.03, 0.12, 0.08])


def test_double_body_zero_speed_reference_shift_and_mesh_identity():
    mesh = wigley3_wetted_hull_mesh(nx=8, nz=4)
    flow = prepare_double_body_flow(mesh)

    def run(speed, center=(0.0, 0.0, 0.0), base=flow):
        return solve_forward_speed_heave_pitch(
            mesh, 4.0, speed, center=center, base_flow=base
        )

    native, zero = run(0.0, base=None), run(0.0)
    for field in ("added_mass", "damping", "excitation"):
        np.testing.assert_allclose(
            getattr(native, field), getattr(zero, field), atol=1e-10
        )
    dx = 0.25
    centered = run(0.9)
    shifted = run(0.9, (dx, 0.0, 0.0))
    T = np.array([[1.0, dx], [0.0, 1.0]])
    for field in ("added_mass", "damping"):
        np.testing.assert_allclose(
            getattr(shifted, field), T.T @ getattr(centered, field) @ T, atol=1e-9
        )
    np.testing.assert_allclose(shifted.excitation, T.T @ centered.excitation, atol=1e-9)
    with pytest.raises(ValueError, match="supplied mesh"):
        solve_forward_speed_heave_pitch(
            wigley3_wetted_hull_mesh(nx=8, nz=4), 4.0, 0.9, base_flow=flow
        )


def test_double_body_live_anchor_and_attribution_sum(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.journee_wigley3_forward_speed import (
        base_flow_study as study,
        double_body,
    )

    records = study.load_coefficients(study.ROOT / "double_body_coefficients.csv")
    assert len(records) == 312
    r = next(
        r
        for r in records
        if r["panels"] == "264" and r["kind"] == "motion" and r["Fn"] == "0.3"
    )
    live = double_body.evaluate((12, 6, "motion", 0.3, float(r["lambda_L"])))
    fields = list(r)[5:]
    np.testing.assert_allclose(
        [float(v) for v in live[5:]],
        [float(r[k]) for k in fields],
        atol=1e-9,
        rtol=1e-9,
    )
    curves = study.predictions(study.ROOT, records)
    effects = study.component_effects(study.ROOT, curves)
    assert len({(r[1], r[2]) for r in effects}) == 34
    for r in effects:
        assert r[7] - r[8] == pytest.approx(sum(r[9:]), abs=1e-12)
    _, forced = study.forced_errors(study.ROOT, records)
    assert len(forced) == 416


def test_speed_parity_and_held_speed_independence(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.journee_wigley3_forward_speed import speed_parity as study

    for fn in (0.2, 0.3, 0.4):
        for cross in (False, True):
            p = np.array(study.speed_weights(fn, 0.2, 0.4, cross, parity=True))
            m = np.array(study.speed_weights(-fn, 0.2, 0.4, cross, parity=True))
            np.testing.assert_allclose(m, (-1 if cross else 1) * p)
    with pytest.raises(ValueError):
        study.speed_weights(0.5, 0.2, 0.4, True, parity=True)
    (tmp_path / "tensorbem.csv").symlink_to(study.ROOT / "tensorbem.csv")
    for file in ("radiation.csv", "loads.csv"):
        with (study.ROOT / file).open() as f, (tmp_path / file).open("w") as dest:
            input_rows = csv.DictReader(f)
            w = csv.DictWriter(dest, fieldnames=input_rows.fieldnames)
            w.writeheader()
            w.writerows(r for r in input_rows if float(r["Fn"]) != 0.30)
    clean, _ = study.parity_motions(tmp_path)
    standard, _ = study.parity_motions()
    for k in ("heave", "pitch", "heave_deg", "pitch_deg"):
        np.testing.assert_array_equal(clean[k], standard[k])
    scores, _ = study.compare(predictions_by_solver={"parity": clean})
    assert [r[-1] for r in scores if r[2] == 0.30 and r[3] in ("heave", "pitch")] == [
        "PASS",
        "PASS",
    ]
    rows = study.cross_hull()
    assert {h: sum(r[0] == h and r[7] for r in rows) for h in ("I", "III", "IV")} == {
        "I": 112,
        "III": 100,
        "IV": 112,
    }
