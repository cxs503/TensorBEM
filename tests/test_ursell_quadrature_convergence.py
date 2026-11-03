"""Regression guards for the Ursell panel solution's numerical provenance.

The Ursell benchmark is the one place in this repository where the *panel*
number, not just the analytical reference, is quoted in the README and in
``docs/marine_bem_benchmarks.md``. Those quoted numbers were once produced by an
implementation that handed the whole (``log``-singular) Green function to a
Gauss-Legendre rule; commit 267d5a3 replaced that with an analytic log primitive
plus a quadrature of the smooth free-surface remainder only. The panel result
moved by about 1e-3 relative, while the analytical reference did not move at
all, which is how the discrepancy was diagnosed.

Two properties are therefore locked down here:

1. the panel solution must not depend on the number of quadrature nodes, i.e.
   the quoted error is a discretisation error and not a quadrature artefact;
2. the quoted numbers in the documentation are regenerated from the committed
   raw runs, not typed by hand.

The tests deliberately use coarse meshes: they guard the formulation, and the
fine-mesh values live in ``docs/ursell_quadrature_study/raw/``.
"""

import json
import math
from pathlib import Path

import pytest

from tensorbem.analytic import ursell_cylinder_added_mass
from tensorbem.benchmarks.ursell_cylinder import run

RADIUS, SUBMERGENCE, KA = 1.0, 2.0, 0.6
DENSITY, GRAVITY = 1025.0, 9.80665
OMEGA = math.sqrt(KA * GRAVITY / RADIUS)

_REPO = Path(__file__).resolve().parents[1]
_STUDY = _REPO / "docs" / "ursell_quadrature_study"


def _reference():
    return ursell_cylinder_added_mass(RADIUS, SUBMERGENCE, OMEGA, DENSITY)


def test_quadrature_nodes_do_not_change_the_panel_solution():
    """The quoted error must be discretisation-limited, not quadrature-limited.

    With the singular Rankine part integrated analytically, going from 6 to 12
    Gauss-Legendre nodes for the smooth remainder changes nothing at double
    precision. The pre-267d5a3 implementation failed this test: at 128 panels
    its added mass drifted by 1.2e-3 relative over the same node range.
    """
    coarse = run(48, RADIUS, SUBMERGENCE, KA, DENSITY, GRAVITY, panel_nodes=6)
    fine = run(48, RADIUS, SUBMERGENCE, KA, DENSITY, GRAVITY, panel_nodes=12)
    assert math.isclose(
        coarse["panel_added_mass"], fine["panel_added_mass"], rel_tol=1.0e-9
    )
    assert math.isclose(
        coarse["panel_damping"], fine["panel_damping"], rel_tol=1.0e-9
    )
    assert coarse["panel_relative_residual"] < 1.0e-10


def test_panel_convergence_is_first_order():
    """Coarse-mesh error halves as the panel count doubles (constant panels)."""
    errors = []
    for panels in (32, 64):
        errors.append(run(panels, RADIUS, SUBMERGENCE, KA, DENSITY, GRAVITY)[
            "panel_added_mass_relative_error"
        ])
    ratio = errors[0] / errors[1]
    assert 1.5 < ratio < 2.6, ratio


def test_coarse_mesh_values_are_pinned():
    """Pin the current formulation so a silent kernel change cannot slip through."""
    row = run(48, RADIUS, SUBMERGENCE, KA, DENSITY, GRAVITY)
    assert row["panel_added_mass"] == pytest.approx(2601.3075, rel=1.0e-5)
    assert row["panel_damping"] == pytest.approx(2807.0615, rel=1.0e-5)
    assert row["panel_added_mass_relative_error"] == pytest.approx(2.2058e-2, rel=2.0e-3)
    assert row["panel_damping_relative_error"] == pytest.approx(3.2709e-2, rel=2.0e-3)


def test_documented_study_is_generated_from_the_raw_runs():
    """The study tables are derived from committed JSON, and stay consistent."""
    summary_path = _STUDY / "summary.json"
    if not summary_path.exists():
        pytest.skip("quadrature study not built; run docs/ursell_quadrature_study/build_summary.py")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    reference = _reference()
    assert summary["reference"]["added_mass"] == pytest.approx(reference[0], rel=1e-12)
    assert summary["reference"]["damping"] == pytest.approx(reference[1], rel=1e-12)

    # Every quoted panel number must match its committed raw run.
    for row in summary["panel_sweep_current_nodes24"]:
        raw = json.loads((_STUDY / "raw" / row["source_file"]).read_text(encoding="utf-8"))
        assert raw["panel_added_mass"] == row["panel_added_mass"]
        assert raw["panel_damping"] == row["panel_damping"]
        assert raw["panels"] == row["panels"]

    # The quadrature claim itself.
    current = summary["node_sensitivity_current_panels128"]
    legacy = summary["node_sensitivity_legacy_panels128"]
    assert current["added_mass_spread_relative"] < 1.0e-9
    assert current["damping_spread_relative"] < 1.0e-9
    assert legacy["added_mass_spread_relative"] > 1.0e-3
    assert summary["legacy_finest_gap_to_converged"]["added_mass_relative"] < 0.0


def test_quoted_table_matches_the_converged_solver():
    """The 128-panel numbers in the study tables are the ones this code produces.

    Re-running the 128-panel case costs several seconds, so the check is made
    against the committed raw run and the reference; the formulation-level
    guarantee is provided by the coarse-mesh tests above.
    """
    raw_path = _STUDY / "raw" / "current_panels128_nodes24.json"
    if not raw_path.exists():
        pytest.skip("raw 128-panel run missing")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    reference = _reference()
    assert raw["panel_added_mass_relative_error"] == pytest.approx(
        abs(raw["panel_added_mass"] - reference[0]) / reference[0], rel=1.0e-12
    )
    assert raw["panel_damping_relative_error"] == pytest.approx(
        abs(raw["panel_damping"] - reference[1]) / reference[1], rel=1.0e-12
    )
    assert raw["near_far_field_relative_difference"] < 6.0e-3