"""Stable low-frequency Green functions, oblique-wave modes, and S-175 experimental coverage."""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from tensorbem.free_surface3d import wave_function_3d
from tensorbem.forward_speed import (
    solve_forward_speed_heave_pitch,
    solve_forward_speed_modes,
)
from tensorbem.wigley3_hull import wigley3_wetted_hull_mesh


def test_small_wavenumber_cone_limit_near_synchronism():
    # With X=rY and Y->0 the Bessel-transform correction to -gamma-log(Y)
    # has the analytic value log(2/(1+sqrt(1+r^2))).
    r = np.array([0.0, 0.1, 0.34])
    y = 1e-12
    root = np.sqrt(1 + r * r)
    with np.errstate(over="raise", invalid="raise"):
        f, fx, fy = wave_function_3d(r * y, np.full(len(r), y))
    np.testing.assert_allclose(
        f + np.log(y) + np.euler_gamma, np.log(2 / (1 + root)), atol=2e-9
    )
    np.testing.assert_allclose(fx * y, -r / (root * (1 + root)), atol=2e-9)
    np.testing.assert_allclose(fy * y, -1 / root, atol=2e-9)


def test_heading_convention_mode_order_and_following_branch():
    mesh = wigley3_wetted_hull_mesh(nx=8, nz=4)
    w, u, g = 3.0, 0.55, 9.81
    head = solve_forward_speed_heave_pitch(mesh, w, u, gravity=g)
    all_head = solve_forward_speed_modes(mesh, w, u, wave_angle=np.pi, gravity=g)
    ix = [2, 4]
    np.testing.assert_allclose(
        head.added_mass, all_head.added_mass[np.ix_(ix, ix)], atol=1e-10
    )
    np.testing.assert_allclose(head.excitation, all_head.excitation[ix], atol=1e-10)

    from tensorbem.geometry3d import WettedHullMesh

    reflected = WettedHullMesh(mesh.vertices * [1, -1, 1], mesh.faces[:, [0, 2, 1]])

    def solve(body, b):
        return solve_forward_speed_modes(
            body, w, u, wave_angle=b, gravity=g, dofs=[4, 3, 2]
        )

    port, starboard = solve(mesh, 0.9), solve(reflected, -0.9)
    np.testing.assert_allclose(
        port.excitation, starboard.excitation * [1, -1, 1], atol=1e-7
    )
    assert abs(port.excitation[1]) > 1e-3
    follow = solve_forward_speed_modes(mesh, w, 1.1 * g / w, wave_angle=0.0, gravity=g)
    assert (
        follow.encounter_omega < 0
        and follow.relative_residual < 1e-10
        and np.isfinite(follow.excitation).all()
    )
    with pytest.raises(ValueError, match="zero encounter"):
        solve_forward_speed_modes(mesh, w, g / w, wave_angle=0.0, gravity=g)
    for dofs in ([1, 1], [6], [], [0.5]):
        with pytest.raises(ValueError):
            solve_forward_speed_modes(mesh, w, u, wave_angle=1.0, dofs=dofs)


def test_s175_source_coverage_and_mesh(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    from benchmarks.s175_oblique.run import ROOT, MASS, RHO, model, calculate

    with (ROOT / "observations.csv").open() as f:
        obs = list(csv.DictReader(f))
    assert len(obs) == 87 and {r["laboratory"] for r in obs} == {"SRI", "SHI", "IHI"}
    assert Counter(r["quantity"] for r in obs) == {"roll": 25, "heave": 31, "pitch": 31}
    for g, n in ((6, 596), (8, 1372)):
        mesh, m, c = model(g)
        assert mesh.panel_count == n and c[3, 3] > 0
        assert abs(mesh.volume / (MASS / RHO) - 1) < (0.012 if g == 6 else 0.01)
    with (ROOT / "results.csv").open() as f:
        results = list(csv.DictReader(f))
    assert len(results) == 2 * len(obs)
    with (ROOT / "capytaine.csv").open() as f:
        external = list(csv.DictReader(f))
    assert len(external) == 9
    anchor = next(
        r for r in external if r["heading_deg"] == "60" and r["quantity"] == "heave"
    )
    native = next(
        r
        for r in results
        if r["heading_deg"] == "60"
        and r["panels"] == "596"
        and r["quantity"] == "heave"
        and r["omega_sqrt_L_g"] == anchor["omega_sqrt_L_g"]
    )
    assert abs(float(native["predicted"]) - float(anchor["capytaine"])) < 0.005
    near = min(results, key=lambda r: abs(float(r["encounter_ratio"])))
    assert abs(float(near["encounter_ratio"])) < 0.001 and np.isfinite(
        float(near["predicted"])
    )

    t = next(
        r
        for r in results
        if r["quantity"] == "roll"
        and int(r["heading_deg"]) == 150
        and r["panels"] == "596"
    )
    v = calculate((6, 150, float(t["omega_sqrt_L_g"])))
    assert v["roll"]["predicted"] == pytest.approx(float(t["predicted"]), rel=1e-8)
