"""Gates for route "T" -- the external (trimesh) referee of the rigid mass matrix.

Route T is the case-pool anchor's only check whose numbers do not come out of
TensorBEM: it replaces both the surface discretisation (uv_sphere instead of the
icosphere) and the volume integration (Mirtich projection instead of the
signed-tetrahedron quadrature).  These tests keep that independence honest:

* the referee reproduces a number nobody had to integrate (a cube) to round-off,
  so it is calibrated before it judges anything;
* on shared triangles it agrees with TensorBEM's integrator to round-off, which
  is what makes the accepted-level error attributable to panelisation;
* on its own discretisation it lands inside the anchor tolerance, and its
  negative control (the pre-fix coupling convention) does not;
* it does *not* replace the topology guard: `trimesh.Trimesh.is_watertight`
  accepts a single flipped face, which is the load-bearing evidence for keeping
  the stricter guard;
* a skipped route is reported as ``not_run`` and never as a pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks import rigid_body_mass_external_route as route_t

trimesh = pytest.importorskip("trimesh", reason="route T needs the optional referee")

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "docs" / "case_bem_rigid_body_mass" / "results.json"
VERDICT = REPO_ROOT / "docs" / "case_bem_rigid_body_mass" / "verdict.json"


# ---------------------------------------------------------------------------
# the referee's own calibration
# ---------------------------------------------------------------------------
def test_referee_reproduces_the_analytic_cube_mass_properties() -> None:
    """``m = rho L^3``, ``I_cg = m L^2/6`` -- exact for a polyhedron."""
    calibration = route_t.self_calibration(density=1000.0, side=2.0)
    assert calibration["watertight"] is True
    assert calibration["volume_rel_error"] < 1.0e-12
    assert calibration["mass_rel_error"] < 1.0e-12
    assert calibration["centroid_abs_error_m"] < 1.0e-12
    assert calibration["inertia_rel_error"] < 1.0e-12


def test_external_integration_rejects_a_non_watertight_surface() -> None:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    faces = np.asarray(mesh.faces, dtype=int)[:-1]  # remove one face -> open
    with pytest.raises(ValueError, match="watertight"):
        route_t.external_mass_properties(
            np.asarray(mesh.vertices, dtype=float), faces, 1025.0
        )
    # ... and the caller can opt out when the open surface itself is the subject.
    loose = route_t.external_mass_properties(
        np.asarray(mesh.vertices, dtype=float), faces, 1025.0, require_watertight=False
    )
    assert loose["watertight"] is False


# ---------------------------------------------------------------------------
# layer attribution
# ---------------------------------------------------------------------------
def test_both_integrators_agree_to_roundoff_on_shared_triangles() -> None:
    """If they did not, the anchor's error budget could not be attributed."""
    attribution = route_t.integrator_cross_check(subdivisions=(1, 2, 3))
    assert attribution["checks"]["external_integrator_roundoff"] is True
    assert attribution["worst_matrix_difference"] < route_t.ROUNDOFF_LIMIT
    for row in attribution["levels"]:
        assert row["external_watertight"] is True
        # both routes must sit at the *same* distance from the closed form: the
        # error is the mesh's, not the integrator's.
        assert row["external_vs_closed_form"] == pytest.approx(
            row["tensorbem_vs_closed_form"], rel=1.0e-9, abs=1.0e-12
        )


# ---------------------------------------------------------------------------
# the independent route
# ---------------------------------------------------------------------------
def test_external_route_reaches_the_tolerance_on_its_own_discretisation() -> None:
    ladder = route_t.independent_mesh_ladder(counts=(16, 32), accepted_counts=(32,))
    accepted = [row for row in ladder["levels"] if row["accepted"]]
    assert accepted, "the accepted set must not be empty"
    assert all(row["within_tolerance"] for row in accepted)
    assert ladder["checks"]["external_mesh_route_within_tolerance"] is True
    # The ladder itself must converge: refinement may not increase the error.
    errors = [row["max_entry_rel_error"] for row in ladder["levels"]]
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    # Honesty about the margin: the acceptance set is *predeclared*, and the
    # coarsest level that already passes is reported next to it (the external
    # discretisation reaches 3 % well before the declared level).
    assert ladder["coarsest_within_tolerance_count"] is not None
    assert ladder["coarsest_within_tolerance_count"] <= min(ladder["accepted_counts"])


def test_external_route_has_discriminating_power() -> None:
    ladder = route_t.independent_mesh_ladder(counts=(16, 32), accepted_counts=(32,))
    assert ladder["checks"]["external_route_rejects_legacy_coupling"] is True
    assert ladder["legacy_coupling_worst_rel_error"] > route_t.TOLERANCE


def test_external_route_certifies_the_ship_shaped_body() -> None:
    ship = route_t.ship_shape_check(n_sections=24, n_levels=12)
    assert ship["mesh"]["external_watertight"] is True
    assert ship["checks"]["external_ship_shape_within_tolerance"] is True
    assert ship["matrix_max_entry_rel_error"] < route_t.TOLERANCE
    # every block of M_RB is exercised: a ship CG out of the midship plane makes
    # the coupling block non-trivial.
    assert ship["block_max_entry_rel_error"]["translation-rotation"] > 0.0
    assert ship["hull"]["block_coefficient"] == pytest.approx(4.0 / 9.0)


# ---------------------------------------------------------------------------
# the external referee of the topology guard
# ---------------------------------------------------------------------------
def test_external_predicate_never_certifies_what_the_guard_rejects() -> None:
    referee = route_t.topology_referee(subdivisions=1)
    assert referee["guard_never_laxer"] is True
    assert referee["guard_rejects_every_corruption"] is True
    assert referee["checks"]["external_topology_guard_agrees"] is True
    for row in referee["variants"][1:]:
        assert row["guard_accepts"] is False, f"guard accepted {row['variant']}"


def test_external_predicate_is_not_a_substitute_for_the_guard() -> None:
    """A single flipped face stays 'watertight' for trimesh: keep the guard."""
    referee = route_t.topology_referee(subdivisions=1)
    missed = referee["corruptions_the_external_predicate_misses"]
    assert "single flipped face" in missed, (
        "if trimesh ever starts rejecting flipped faces, this test should be "
        "re-argued rather than deleted: the guard's extra strictness is what "
        "the anchor relies on"
    )
    assert referee["checks"]["external_guard_is_load_bearing"] is True
    flipped = next(row for row in referee["variants"] if row["variant"] == "single flipped face")
    assert flipped["external_accepts"] is True
    assert flipped["guard_orientation_mismatches"] > 0


def test_welding_policy_is_the_only_difference_on_a_split_vertex_mesh() -> None:
    referee = route_t.topology_referee(subdivisions=1)
    weld = referee["weld_sensitivity"]
    assert weld["guard_is_closed"] is True
    assert weld["external_watertight_as_given"] is False
    assert weld["external_watertight_after_merge"] is True
    assert weld["policies_agree_after_alignment"] is True


# ---------------------------------------------------------------------------
# a skipped route is not a pass
# ---------------------------------------------------------------------------
def test_missing_dependency_is_reported_as_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(route_t, "available", lambda: False)
    section = route_t.run()
    assert section["status"] == "not_run"
    assert section["checks"] == {}
    assert "not passed" in section["reason"] or "skipped" in section["reason"]
    assert section["required_checks"] == list(route_t.CHECKS)


def test_driver_without_route_t_cannot_certify_the_anchor() -> None:
    from tensorbem.benchmarks import casepool_bem_rigid_body_mass as driver

    payload = driver.run(subdivisions=(3, 4), with_rao=False, with_external=False)
    verdict = payload["verdict"]
    assert verdict["overall"] == "INCOMPLETE"
    assert verdict["all_pass"] is False
    for name in route_t.CHECKS:
        assert name in verdict["not_run"]
        assert name not in verdict["checks"]
    assert payload["external_route"]["status"] == "not_run"


# ---------------------------------------------------------------------------
# the committed bundle
# ---------------------------------------------------------------------------
def test_committed_bundle_carries_every_route_t_check() -> None:
    assert RESULTS.is_file() and VERDICT.is_file()
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    verdict = json.loads(VERDICT.read_text(encoding="utf-8"))
    section = results["external_route"]
    assert section["status"] == "run"
    assert verdict["overall"] == "PASS"
    assert verdict["not_run"] == []
    for name in route_t.CHECKS:
        assert verdict["checks"][name] is True, f"{name} not certified"
        assert section["checks"][name] is True, f"{name} disagrees with the section"
    ship = section["ship_shape"]
    assert ship["matrix_max_entry_rel_error"] < route_t.TOLERANCE
    ladder = section["independent_mesh_route"]
    assert ladder["worst_accepted_entry_rel_error"] < route_t.TOLERANCE
    assert section["layer_attribution"]["worst_matrix_difference"] < route_t.ROUNDOFF_LIMIT