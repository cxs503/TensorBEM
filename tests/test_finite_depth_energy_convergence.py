"""Regression tests for the finite-depth energy-identity attribution.

The debt being closed (``docs/finite_depth_radiation.md`` §5, limitation 1) is
that the wave-amplitude damping and the pressure damping differ by about one
percent at 64 panels, ``+1.06 %`` in finite depth and ``-1.04 %`` in deep water.
A one-percent offset with an unexplained sign is exactly what a wrong
normalisation looks like, so these tests pin *how it behaves*:

* it is first order in the panel size (so it is discretisation, not a constant),
* it does not move when the per-panel Gauss order changes (not quadrature),
* replacing the midpoint reading of the far-field integral by a proper panel
  quadrature changes it by ~1e-4 (not the amplitude integrator),
* the committed artifact is reproducible from the code in this repository.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks.finite_depth_energy_convergence import (
    _finite_depth_case,
    far_field_damping_panel_quadrature,
    observed_order,
)
from tensorbem.benchmarks.finite_depth_radiation import submerged_circle_mesh
from tensorbem.finite_depth_solver import _solve_sources

ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "finite_depth_energy_identity"
    / "convergence.json"
)

SHALLOW_DEPTH = 3.2
KA = 0.6
RADIUS = 1.0
SUBMERGENCE = 2.0
DENSITY = 1025.0


def _case(panels: int, panel_nodes: int = 8, depth: float = SHALLOW_DEPTH,
          motion: str = "heave") -> dict:
    return _finite_depth_case(
        depth, motion, panels, KA, RADIUS, SUBMERGENCE, DENSITY,
        panel_nodes, 24, 6,
    )


def test_offset_converges_at_first_order():
    """The energy-identity offset is a discretisation error: it halves per refinement."""
    mesh = (32, 64, 128)
    offsets = []
    for panels in mesh:
        row = _case(panels)
        # the residual proves the linear solve is not the culprit
        assert row["relative_residual"] < 1e-10
        offsets.append(row["energy_offset"])
    assert all(offset > 0.0 for offset in offsets), offsets
    orders = observed_order(mesh, tuple(offsets))
    # constant panels on a circle give O(h**1); the fit is allowed to wobble
    assert all(0.7 < order < 1.4 for order in orders), orders
    assert abs(offsets[-1]) < 2.0e-2


def test_offset_is_not_quadrature_limited():
    """Doubling the per-panel Gauss order must not move the offset."""
    coarse = _case(64, panel_nodes=4)
    fine = _case(64, panel_nodes=16)
    assert abs(fine["energy_offset"] - coarse["energy_offset"]) < 1.0e-4
    assert abs(fine["damping_near"] / coarse["damping_near"] - 1.0) < 1.0e-4


def test_far_field_midpoint_reading_is_not_the_source_of_the_offset():
    """Midpoint vs panel quadrature on the *same* densities differ by ~1e-4."""
    row = _case(64)
    assert abs(row["far_reading_difference"]) < 1.0e-3
    # and the offset survives the better integrator
    assert abs(row["energy_offset_quadrature"] - row["energy_offset"]) < 1.0e-3
    # the improved reading must not change the order of magnitude of the offset
    assert abs(row["energy_offset_quadrature"]) > 1.0e-3


def test_panel_quadrature_reading_matches_the_solver_helper_in_deep_limit():
    """In the deep limit both far-field readings collapse onto each other.

    The finite-depth kernel is tested separately against the deep-water kernel;
    here only the damping reading is under test, so a large depth with the same
    machinery is used.
    """
    mesh = submerged_circle_mesh(64, RADIUS, SUBMERGENCE)
    nu = KA / RADIUS
    sources, _system, _data, _potential = _solve_sources(
        mesh, nu, 60.0, "heave", 8, None
    )
    from tensorbem.finite_depth_solver import far_field_damping

    direct = far_field_damping(mesh, sources, nu, 60.0, DENSITY)
    quadrature = far_field_damping_panel_quadrature(
        mesh, sources, nu, 60.0, DENSITY, nodes=6
    )
    assert abs(quadrature / direct - 1.0) < 5.0e-4


def test_mfs_reference_is_independent_but_converged():
    """The MFS reference is converged in its own parameter, not in the mesh.

    It is *not* used as truth: at the shallowest depth the converged MFS value
    differs from the 64-panel solution by a few percent, which is the panel
    discretisation error this study measures, not a disagreement about physics.
    """
    from tensorbem.finite_depth_solver import radiation_reference

    mesh = submerged_circle_mesh(64, RADIUS, SUBMERGENCE)
    nu = KA / RADIUS
    coarse = radiation_reference(mesh, nu, SHALLOW_DEPTH, "heave",
                                source_count=16, density=DENSITY)
    fine = radiation_reference(mesh, nu, SHALLOW_DEPTH, "heave",
                              source_count=96, density=DENSITY)
    assert abs(fine.damping[1] / coarse.damping[1] - 1.0) < 1.0e-3
    panel = _case(64)
    gap = abs(fine.damping[1] / panel["damping_near"] - 1.0)
    assert 1.0e-3 < gap < 1.0e-1


@pytest.mark.skipif(not ARTIFACT.exists(), reason="benchmark artifact not generated")
def test_report_tables_match_the_artifact_verbatim():
    """The tables in the report are generated from the JSON, not transcribed."""
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "finite_depth_energy_identity"
        / "build_tables.py"
    )
    spec = importlib.util.spec_from_file_location("_fd_energy_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = json.loads(ARTIFACT.read_text())
    report = (path.parent / "report.md").read_text()
    for name, block in module.blocks(payload).items():
        assert f"<!-- BEGIN:{name} -->\n{block}\n<!-- END:{name} -->" in report, name


@pytest.mark.skipif(not ARTIFACT.exists(), reason="benchmark artifact not generated")
def test_committed_artifact_is_reproducible():
    """The numbers quoted in the documentation come from this file."""
    payload = json.loads(ARTIFACT.read_text())
    rows = [
        row for row in payload["rows"]
        if row["depth"] == pytest.approx(SHALLOW_DEPTH)
        and row["motion"] == "heave"
        and row["panels"] == 64
    ]
    assert len(rows) == 1
    committed = rows[0]
    fresh = _case(64)
    for key in (
        "added_mass",
        "damping_near",
        "damping_far",
        "damping_far_quadrature",
        "mfs_damping",
    ):
        assert fresh[key] == pytest.approx(committed[key], rel=1e-9), key
    for key in ("energy_offset", "energy_offset_quadrature"):
        assert fresh[key] == pytest.approx(committed[key], abs=1e-12), key
    summary = payload["summaries"]["shallow:heave"]
    assert summary["median_order"] == pytest.approx(1.0, abs=0.3)
    assert abs(summary["offset_limit"]) < 5.0e-3
    # the damping limit is shared by both readings
    assert summary["near_vs_far_limit_rel"] < 5.0e-3


@pytest.mark.skipif(not ARTIFACT.exists(), reason="benchmark artifact not generated")
def test_deep_water_control_shows_the_same_offset():
    """The already-verified deep-water solver shows the same 1 % at 64 panels.

    That is the reason the offset is attributed to the panel discretisation:
    it is not a property of the finite-depth kernel.
    """
    payload = json.loads(ARTIFACT.read_text())
    rows = [
        row for row in payload["deep_water_control"]
        if row["motion"] == "heave" and row["panels"] == 64
    ]
    assert len(rows) == 1
    assert abs(rows[0]["energy_offset"]) == pytest.approx(1.04e-2, rel=0.15)
    summary = payload["deep_water_summaries"]["heave"]
    assert summary["median_order"] == pytest.approx(1.0, abs=0.35)