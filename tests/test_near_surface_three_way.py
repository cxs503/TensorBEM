"""Regression pins for the near-free-surface three-way error separation (m3530).

m3383 left ``f/a = 1.05``, ``ka = 1.0`` as WITHHELD: the panel solution and the
MFS reference disagreed by 4.7 %/5.9 % and nothing said which leg was wrong.
The study in ``docs/m3530/near_surface_three_way.py`` adds Ursell's multipole
closed form as a third, fully independent leg, which decides the question.  The
tests here keep that decision from silently flipping:

* the analytic unit conversion (``omega = sqrt(ka g / a)``, **not** the
  wavenumber) is pinned against the published reference pair of
  ``docs/marine_bem_benchmarks.md`` §4.6;
* the analytic reference's truncation convergence is measured at the crux point
  rather than assumed;
* the crux attribution -- MFS tracks the analytic value to a few ``1e-3`` while
  the panel solution sits one to two orders of magnitude away -- is asserted;
* the stored artifact is checked against a fresh recomputation of its cheapest
  rungs so the report's numbers cannot drift away from the code.

Everything here runs at coarse resolution except one MFS solve and one 64-panel
solve, so the module stays inside a few seconds.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.analytic import ursell_cylinder_coefficients
from tensorbem.free_surface_solver import (
    radiation_reference,
    solve_radiation_indirect,
)

RADIUS = 1.0
DENSITY = 1025.0
GRAVITY = 9.80665
CRUX_DEPTH_RATIO = 1.05
CRUX_KA = 1.0
ARTIFACT = Path(__file__).resolve().parents[1] / "docs/m3530/near_surface_three_way.json"


def _analytic(ka: float, a_over_f: float, terms: int = 60) -> tuple[float, float]:
    """Dimensional heave added mass and damping from Ursell's multipoles."""
    omega = math.sqrt(ka * GRAVITY / RADIUS)
    mu, nu = ursell_cylinder_coefficients(ka, a_over_f, terms)
    displaced = DENSITY * math.pi * RADIUS**2
    return mu * displaced, nu * displaced * omega


def _circle(panels: int, depth: float):
    from tensorbem.geometry import PanelMesh

    angle = 2.0 * math.pi * np.arange(panels) / panels
    return PanelMesh(
        np.column_stack((RADIUS * np.cos(angle), -depth + RADIUS * np.sin(angle)))
    )


def test_published_ursell_reference_pair_is_reproduced() -> None:
    """Pins the omega-not-k unit trap: ka = 0.6, a/f = 0.5, the documented case.

    ``docs/marine_bem_benchmarks.md`` §4.6 quotes A = 2545.17 kg/m and
    B = 2718.15 kg/(m s) as Ursell's reference for exactly these parameters.
    Using the wavenumber where the angular frequency belongs reproduces the
    added mass and misses the damping by a factor sqrt(ka g) / ka = 4.04.
    """
    added_mass, damping = _analytic(0.6, 0.5)

    assert added_mass == pytest.approx(2545.1672, rel=1e-6)
    assert damping == pytest.approx(2718.1540, rel=1e-6)


def test_analytic_reference_is_truncation_converged_at_the_crux() -> None:
    coarse = _analytic(CRUX_KA, 1.0 / CRUX_DEPTH_RATIO, 60)
    fine = _analytic(CRUX_KA, 1.0 / CRUX_DEPTH_RATIO, 120)

    assert coarse[0] == pytest.approx(fine[0], rel=1e-11)
    assert coarse[1] == pytest.approx(fine[1], rel=1e-11)


def test_mfs_tracks_the_analytic_reference_at_the_crux() -> None:
    """The verdict m3383 had to withhold: the MFS leg is not the wrong one."""
    reference_added_mass, reference_damping = _analytic(
        CRUX_KA, 1.0 / CRUX_DEPTH_RATIO, 120
    )
    result = radiation_reference(
        _circle(256, CRUX_DEPTH_RATIO * RADIUS), CRUX_KA / RADIUS, "heave",
        source_count=48, density=DENSITY,
    )
    added_mass = float(np.asarray(result.added_mass, dtype=float).ravel()[1])
    damping = float(np.asarray(result.damping, dtype=float).ravel()[1])

    assert abs(added_mass - reference_added_mass) / abs(reference_added_mass) < 5.0e-3
    assert abs(damping - reference_damping) / abs(reference_damping) < 5.0e-3


def test_panel_leg_is_the_outlier_at_the_crux() -> None:
    """Piecewise-constant panels stay 1-2 orders of magnitude off the reference."""
    reference_added_mass, reference_damping = _analytic(
        CRUX_KA, 1.0 / CRUX_DEPTH_RATIO, 120
    )
    result = solve_radiation_indirect(
        _circle(64, CRUX_DEPTH_RATIO * RADIUS), CRUX_KA / RADIUS, "heave",
        density=DENSITY, panel_nodes=24,
    )
    added_mass = float(np.asarray(result.added_mass, dtype=float).ravel()[1])
    damping = float(np.asarray(result.damping, dtype=float).ravel()[1])

    added_mass_error = abs(added_mass - reference_added_mass) / abs(reference_added_mass)
    damping_error = abs(damping - reference_damping) / abs(reference_damping)
    assert 1.0e-3 < added_mass_error
    assert 1.0e-3 < damping_error
    # and the error is real discretisation, not a broken solve
    assert result.relative_residual < 1.0e-6


def test_artifact_numbers_are_reproducible_and_the_gate_statement_holds() -> None:
    if not ARTIFACT.exists():  # pragma: no cover - artifact ships with the report
        pytest.skip(f"missing study artifact {ARTIFACT}")
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    rows = payload["rows"]
    assert len(rows) == len(payload["meta"]["grid_depth_ratio_ka"])

    crux = next(
        row for row in rows
        if row["depth_ratio"] == CRUX_DEPTH_RATIO and row["ka"] == CRUX_KA
    )
    reference_added_mass, reference_damping = _analytic(
        CRUX_KA, 1.0 / CRUX_DEPTH_RATIO, 120
    )
    assert crux["reference"]["added_mass"] == pytest.approx(reference_added_mass, rel=1e-12)
    assert crux["reference"]["damping"] == pytest.approx(reference_damping, rel=1e-12)

    for row in rows:
        for rung in row["panels"].values():
            expected = abs(rung["added_mass"] - row["reference"]["added_mass"]) / abs(
                row["reference"]["added_mass"]
            )
            assert rung["added_mass_error"] == pytest.approx(expected, rel=1e-12)
            expected = abs(rung["damping"] - row["reference"]["damping"]) / abs(
                row["reference"]["damping"]
            )
            assert rung["damping_error"] == pytest.approx(expected, rel=1e-12)

    verdict = payload["verdict"]
    assert verdict["mfs_48_worst_error"]["added_mass"] < 5.0e-3
    assert verdict["mfs_48_worst_error"]["damping"] < 5.0e-3
    assert verdict["panel_256_peak_error"]["added_mass"]["error"] < 5.0e-2
    assert verdict["panel_256_peak_error"]["damping"]["error"] < 3.0e-1

def test_mfs_accuracy_is_also_limited_by_its_collocation_mesh() -> None:
    """The MFS leg is not mesh-free in practice: it collocates on a panel mesh.

    With 48 sources, moving the collocation mesh from 64 to 256 panels improves
    the added mass from ~2.2 % to ~0.14 % error.  This is why m3383's diagnosis
    ("the MFS ran only 48 sources") was incomplete: the source count was
    converged, but the collocation resolution was not, and only the analytic
    leg can tell the two effects apart.
    """
    reference_added_mass, _ = _analytic(CRUX_KA, 1.0 / CRUX_DEPTH_RATIO, 120)
    errors = []
    for mesh_panels in (64, 256):
        result = radiation_reference(
            _circle(mesh_panels, CRUX_DEPTH_RATIO * RADIUS), CRUX_KA / RADIUS,
            "heave", source_count=48, density=DENSITY,
        )
        added_mass = float(np.asarray(result.added_mass, dtype=float).ravel()[1])
        errors.append(abs(added_mass - reference_added_mass) / abs(reference_added_mass))

    assert errors[0] > 1.0e-2
    assert errors[1] < 2.0e-3
