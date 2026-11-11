"""Regression gates for the m14931 ``tensorbem.fmm`` cost/routing audit.

These tests pin the *facts* that ``tests/test_fmm.py`` does not decide:

1. the multipole path is actually exercised on a surface distribution;
2. the shipped "physics invariance" configuration routes nothing to it
   (``max_leaf = panel_count`` ⇒ one leaf ⇒ direct sum on both sides);
3. the multipole expansion is accurate on the geometry class the operator would
   be used on, at the 15 % acceptance threshold, and converges with ``p`` there
   -- before m14931 it did *not* (``0.688`` at ``p=2`` and ``3.917`` at ``p=4``
   because a leaf was expanded about its centre while the target sat inside the
   leaf's own source radius);
4. the operator is not faster than the vectorised dense matvec at any measured
   ``N``, so no acceleration claim may be made from wall time.

The committed audit artefacts live in ``docs/fmm_cost_audit/`` and are produced by
``python -m tensorbem.benchmarks.fmm_cost_audit``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorbem.benchmarks.fmm_cost_audit import (
    cost_point,
    routing_counts,
    separated_ladder,
    surface_ladder,
    tautology_probe,
)
from tensorbem.fmm import FMMOperator
from tensorbem.geometry3d import icosphere

ARTIFACT = Path(__file__).resolve().parents[1] / "docs" / "fmm_cost_audit" / "audit.json"


@pytest.fixture(scope="module")
def audit() -> dict:
    return json.loads(ARTIFACT.read_text())


def test_artifact_is_committed_and_complete(audit: dict) -> None:
    for key in ("cost", "accuracy_separated", "accuracy_surface",
                "shipped_physics_test_routing"):
        assert key in audit, f"audit artifact missing {key!r}"
    assert audit["cost"] and audit["accuracy_separated"] and audit["accuracy_surface"]


def test_far_field_path_is_exercised_on_a_surface() -> None:
    """At least a quarter of the target-leaf pairs must use the multipole."""

    mesh = icosphere(subdivisions=2, radius=1.0)
    centroids = mesh.centroids
    op = FMMOperator(p=4, max_leaf=32)
    op.setup(centroids, mesh.areas)
    route = routing_counts(op, centroids)
    assert route["far_pairs"] > 0.25 * route["total_pairs"], route


def test_shipped_physics_configuration_uses_no_multipole() -> None:
    """``max_leaf = panel_count`` degenerates to direct summation (documented)."""

    probe = tautology_probe()
    assert probe["leaves"] == 1
    assert probe["far_pairs"] == 0
    assert probe["near_pairs"] == probe["total_pairs"]
    assert probe["rel_error"] == pytest.approx(0.0, abs=1e-12)


def test_surface_multipole_accuracy_meets_the_fifteen_percent_gate(audit: dict) -> None:
    """The regression pin for the admissibility fix (pre-fix: 0.688 / 3.917)."""

    for row in audit["accuracy_surface"]:
        assert row["far_pairs"] > 0
        assert row["max_rel_error"] < 0.15, row


def test_surface_accuracy_converges_with_order() -> None:
    rows = surface_ladder(p_values=(2, 4))
    assert rows[1]["max_rel_error"] < rows[0]["max_rel_error"], rows
    assert rows[1]["max_rel_error"] < 1e-3, rows


def test_separated_multipole_ladder_converges() -> None:
    rows = separated_ladder(p_values=(2, 8))
    assert rows[0]["max_rel_error"] < 1e-2, rows
    assert rows[1]["max_rel_error"] < 1e-5, rows
    assert rows[1]["max_rel_error"] < rows[0]["max_rel_error"]


def test_operator_is_not_faster_than_dense_on_a_surface(audit: dict) -> None:
    """No measured N is accelerated; the artifact must keep saying so."""

    for row in audit["cost"]:
        assert row["time_ratio"] > 10.0, row
        assert row["per_unit_cost_ratio"] > 10.0, row


def test_cost_measurement_reproduces_within_an_order_of_magnitude(audit: dict) -> None:
    """Fresh timing must not drift by more than 10x from the committed one."""

    committed = next(row for row in audit["cost"]
                     if row["level"] == 2 and row["p"] == 2)
    fresh = cost_point(level=2, p=2, max_leaf=32)
    ratio = fresh["time_ratio"] / committed["time_ratio"]
    assert 0.1 < ratio < 10.0, (fresh["time_ratio"], committed["time_ratio"])
    assert fresh["leaves"] == committed["leaves"]
    assert fresh["far_pairs"] == committed["far_pairs"]