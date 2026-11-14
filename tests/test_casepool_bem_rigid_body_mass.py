"""Gates for the driver of anchor ``CASE-BEM-RIGID-BODY-MASS``.

The anchor's *machine* verdict (``docs/case_bem_rigid_body_mass/verdict.json``,
read by ``scripts/case_pool_run.py --check``) used to check the closed-form
comparison, the momentum route and the two controls, but not the end-to-end
probe that shows the seakeeping solver actually consumes ``M_RB``: those
numbers were in ``results.json`` with no invariant attached, so a caller
regression would have left the anchor ``PASS``.

These tests bind that probe to the verdict:

* the derivation helper (``rao_caller_section``) must turn the raw replay into
  two checked invariants -- reproduction to round-off and discriminating power
  above the anchor tolerance -- and must report ``not_run`` (never a pass) when
  the probe was skipped;
* the committed artefact must declare and satisfy those invariants;
* the driver must mark the anchor ``INCOMPLETE`` when invoked with ``--no-rao``
  while still passing the checks it did execute.

The end-to-end numbers themselves belong to
``tests/test_case_rigid_body_mass.py::test_rao_solver_consumes_the_reference_matrix``
(producer side); these tests gate the anchor's acceptance ledger.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks import casepool_bem_rigid_body_mass as module

ARTIFACTS = Path(__file__).resolve().parents[1] / "docs" / "case_bem_rigid_body_mass"
RAO_CHECKS = (
    "rao_caller_verified",
    "rao_caller_replay_roundoff",
    "rao_caller_legacy_coupling_material",
)


# --------------------------------------------------------------------------
# derivation helper: three-state, never a silent pass
# --------------------------------------------------------------------------
def test_missing_probe_is_reported_as_not_run_not_as_pass() -> None:
    section = module.rao_caller_section(None)
    assert section["status"] == "not_run"
    assert section["checks"] == {}
    assert "not passed" in section["reason"]
    # the criteria travel with the not_run record, so the artefact stays readable
    assert section["replay_roundoff_limit"] == pytest.approx(module.RAO_REPLAY_ROUNDOFF)
    assert section["legacy_change_floor"] == pytest.approx(module.TOLERANCE)


def _raw(replay: float, legacy_change: float) -> dict:
    return {
        "caller": "tensorbem.rao.compute_rao_at_frequency",
        "caller_line": "src/tensorbem/rao.py: m = compute_mass_matrix(...)",
        "omega": 0.6,
        "panels": 108,
        "modes": list(module.MODES),
        "rao_shipped": [0.2, 0.03, 0.42, 3.6e-3, 0.025, 5.9e-4],
        "rao_with_legacy_coupling": [0.3, 0.02, 0.42, 7.0e-3, 0.025, 5.6e-4],
        "rao_change_fraction": [0.5, 0.3, 1e-3, legacy_change, 4e-3, 0.05],
        "max_rao_change": legacy_change,
        "max_rao_change_mode": "roll",
        "replay_with_injected_ref_matrix": replay,
    }


def test_helper_accepts_a_good_replay() -> None:
    section = module.rao_caller_section(_raw(0.0, 0.94))
    assert section["status"] == "run"
    assert all(section["checks"][name] for name in RAO_CHECKS)


@pytest.mark.parametrize(
    "replay, legacy_change, failing",
    [
        (1e-6, 0.94, "rao_caller_replay_roundoff"),
        (0.0, 0.001, "rao_caller_legacy_coupling_material"),
    ],
)
def test_helper_rejects_a_bad_replay(replay: float, legacy_change: float, failing: str) -> None:
    section = module.rao_caller_section(_raw(replay, legacy_change))
    assert section["checks"][failing] is False
    assert section["checks"]["rao_caller_verified"] is True, (
        "verified only means 'the probe ran'; the two invariants carry the verdict"
    )


# --------------------------------------------------------------------------
# committed artefact: the invariants are declared and satisfied
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def results() -> dict:
    path = ARTIFACTS / "results.json"
    assert path.is_file(), f"missing artefact: {path} (run the driver)"
    return json.loads(path.read_text(encoding="utf-8"))


def test_committed_verdict_declares_the_caller_checks(results: dict) -> None:
    verdict = json.loads((ARTIFACTS / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["not_run"] == [], "a certified anchor may not have unrun checks"
    for name in RAO_CHECKS:
        assert name in verdict["checks"], f"{name} missing from the anchor verdict"
        assert verdict["checks"][name] is True
    assert verdict["overall"] == "PASS"
    assert verdict["all_pass"] is True
    # the pre-existing checks must survive the extension
    assert verdict["checks"]["worst_accepted_entry_within_tolerance"] is True
    assert verdict["worst_accepted_entry_rel_error"] < module.TOLERANCE


def test_committed_rao_numbers_obey_the_declared_limits(results: dict) -> None:
    rao = results["rao_caller"]
    assert rao["status"] == "run"
    assert rao["replay_rel_residual"] < module.RAO_REPLAY_ROUNDOFF
    change = np.asarray(rao["rao_change_fraction"], dtype=float)
    assert change.shape == (len(module.MODES),)
    assert len(rao["rao_shipped"]) == change.size
    # the summary scalars must be the summary of the arrays, not a second number
    assert rao["legacy_coupling_max_rao_change"] == pytest.approx(float(change.max()))
    assert rao["legacy_coupling_max_rao_change_mode"] == module.MODES[int(change.argmax())]
    assert float(change.max()) > module.RAO_LEGACY_CHANGE_FLOOR
    assert np.all(np.isfinite(change))
    # the probe has discriminating power in more than one mode (roll and surge here)
    assert int((change > module.TOLERANCE).sum()) >= 2


def test_call_site_target_of_the_injected_matrix_is_the_shipped_assembly() -> None:
    """Premise of the zero replay residual: the shipped matrix *is* the block form.

    If the shipped assembly ever diverges from the closed form the document
    injects, the "residual is exactly zero" statement would silently become a
    different claim, so the premise is pinned here rather than in prose.
    """
    from tensorbem import rao as rao_module
    from tensorbem.benchmarks.case_rigid_body_mass import closed_form_mass_matrix

    mass, r_g, inertia = 4.2e6, np.array([2.0, 0.0, -1.5]), np.diag([1.0e8, 4.0e8, 4.5e8])
    shipped = np.asarray(rao_module.compute_mass_matrix(mass, r_g, inertia), dtype=float)
    reference = np.asarray(closed_form_mass_matrix(mass, r_g, inertia), dtype=float)
    scale = np.abs(reference).max()
    assert np.abs(shipped - reference).max() / scale < 1e-14


# --------------------------------------------------------------------------
# driver semantics: INCOMPLETE is not PASS
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def without_probe() -> dict:
    return module.run([3], with_rao=False)


def test_driver_without_the_probe_cannot_certify(without_probe: dict) -> None:
    verdict = without_probe["verdict"]
    assert verdict["overall"] == "INCOMPLETE"
    assert verdict["all_pass"] is False, "a skipped end-to-end check may not pass the anchor"
    assert verdict["not_run"] == list(RAO_CHECKS)
    assert without_probe["rao_caller"]["status"] == "not_run"
    # the checks that *did* run are still reported honestly
    executed = {k: v for k, v in verdict["checks"].items() if k not in RAO_CHECKS}
    assert executed and all(executed.values())


def test_driver_with_the_probe_certifies(without_probe: dict) -> None:
    payload = module.run([3], with_rao=True)
    verdict = payload["verdict"]
    assert verdict["overall"] == "PASS"
    assert verdict["not_run"] == []
    assert all(verdict["checks"][name] for name in RAO_CHECKS)
    rao = payload["rao_caller"]
    assert rao["status"] == "run"
    assert math.isfinite(rao["legacy_coupling_max_rao_change"])
    # the end-to-end probe must not move the geometric verdict it accompanies
    assert verdict["worst_accepted_entry_rel_error"] == pytest.approx(
        without_probe["verdict"]["worst_accepted_entry_rel_error"]
    )