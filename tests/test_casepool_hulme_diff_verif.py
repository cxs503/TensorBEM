"""Regression gate for CASE-BEM-HULME-DIFF-VERIF (case-pool id 177).

The case grades the floating-hemisphere **total exciting force** of three
routes that do not share a discretisation: the Hulme (1982) wave-free multipole
series, the TensorBEM Kelvin BIE, and ``capytaine`` diffraction completed by an
exact Froude-Krylov quadrature (frozen artefacts, capytaine never imported
here).  These tests pin the committed verdicts, re-derive the cheap anchors from
the library, and freeze the one convention that a future reader is most likely
to get wrong: capytaine's ``DiffractionProblem.forces`` excludes Froude-Krylov.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    froude_krylov_force_3d,
    solve_diffraction_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh

CASE = Path("benchmarks/verified/casepool_bem_hulme_diff_verif")
RESULTS = CASE / "results.json"
REFERENCE = CASE / "reference.json"
CAPY = CASE / "capytaine_scattered.json"

DENSITY = 1025.0
GRAVITY = 9.80665


def _results() -> dict:
    return json.loads(RESULTS.read_text())


def test_artifacts_present():
    for path in (RESULTS, REFERENCE, CAPY):
        assert path.exists(), f"missing case artefact {path}"
        assert path.stat().st_size > 1000


def test_all_gates_pass():
    data = _results()
    assert data["anchor"] == "CASE-BEM-HULME-DIFF-VERIF"
    for name, gate in data["gates"].items():
        assert gate["verdict"] == "PASS", f"{name} verdict {gate['verdict']}: {gate}"
    assert data["verdict"] == "PASS"


def test_primary_gate_is_the_ledger_tolerance():
    gate = _results()["gates"]["G1_complex_relative_case_mesh"]
    assert gate["target"] == pytest.approx(0.05)  # ledger tolerance <5 %
    assert gate["worst"] < 0.01  # measured, with margin


def test_reference_truncation_is_converged():
    for row in _results()["reference"].values():
        assert row["truncation_relative"]["surge"] < 1.0e-5
        assert row["truncation_relative"]["heave"] < 1.0e-3


def test_reference_long_wave_limit_matches_waterplane_force():
    """X3(ka->0) -> rho g A A_wp; the raw ka = 0.02 value approaches it as O(ka)."""
    data = _results()
    long_wave = data["long_wave"]
    assert long_wave["limit"] == pytest.approx(math.pi, rel=1e-12)
    assert long_wave["extrapolated_relative"] < 0.01
    assert long_wave["relative"] < 0.03


def test_reference_row_recomputed_from_library():
    """No hand-copied digits: the committed reference re-derives from analytic.py."""
    stored = _results()["reference"]["1.000"]
    fresh = hulme_hemisphere_radiation(
        1.0, radius=1.0, density=DENSITY, gravity=GRAVITY, n_terms=144,
        quad_order=288,
    )
    for dof, value in (("surge", fresh.excitation_force_x), ("heave", fresh.excitation_force_z)):
        expected = complex(stored[dof]["re"], stored[dof]["im"])
        assert abs(complex(value) - expected) / abs(expected) < 1e-6


def test_exact_froude_krylov_matches_panel_rule():
    """The exact quadrature and the solver's panel rule agree to discretisation."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "hulme_diff_case", CASE / "run_case.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    mesh = hemisphere_hull_mesh(3, radius=1.0)
    fk = froude_krylov_force_3d(mesh, 1.0, density=DENSITY, gravity=GRAVITY)
    exact_x, exact_z = module.exact_froude_krylov(1.0)
    assert abs(fk[0] - exact_x) / abs(exact_x) < 0.03
    assert abs(fk[2] - exact_z) / abs(exact_z) < 0.01


def test_bem_total_force_at_coarse_mesh():
    """Cheap end-to-end re-derivation: 64 panels must already be within 10 %."""
    mesh = hemisphere_hull_mesh(2, radius=1.0)
    ops = assemble_free_surface_operators_3d(mesh, 1.0, chief_points="auto")
    diff = solve_diffraction_3d(mesh, 1.0, wave_angle=0.0, operators=ops,
                               density=DENSITY, gravity=GRAVITY)
    ref = hulme_hemisphere_radiation(1.0, density=DENSITY, gravity=GRAVITY, n_terms=144,
                                     quad_order=288)
    for got, want in ((diff.excitation_force[0], ref.excitation_force_x),
                      (diff.excitation_force[2], ref.excitation_force_z)):
        assert abs(complex(got) - complex(want)) / abs(complex(want)) < 0.10


def test_haskind_reciprocity_frozen():
    worst = _results()["gates"]["G6_haskind_reciprocity"]["worst"]
    assert worst is not None and worst < 0.02


def test_capytaine_convention_is_pinned():
    """capytaine's diffraction force is scattered-only: probe at ka = 0.05."""
    probe = json.loads(CAPY.read_text())["convention_probe"]
    assert probe["verdict"] == "scattered-only"
    heave = complex(probe["capytaine_scattered"]["heave"]["re"],
                    probe["capytaine_scattered"]["heave"]["im"])
    assert abs(heave) < 0.1 * math.pi  # the total would tend to pi, not to zero
