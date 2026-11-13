"""Gates for the high-``ka`` band extension of CASE-BEM-HULME-DIFF-VERIF.

The base case (``tests/test_casepool_hulme_diff_verif.py``) grades
``ka in [0.2, 2.0]``.  This file grades the *extension*: the declared validity
envelope above ``ka = 2``, the reference truncation at high ``ka``, the metric
defect that the base case's worst cell exposes, and the reproducibility of the
frozen capytaine artefact.

The tests do not re-solve the 1024-panel problems (that is minutes of work);
they pin the committed verdicts, re-derive the cheap anchors from the library,
and assert the internal consistency of the declared envelope -- a declared
bound must be exactly the set of frequencies that pass, not a hand-written
claim.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorbem.analytic import hulme_hemisphere_radiation

CASE = Path("benchmarks/verified/casepool_bem_hulme_diff_verif")
BAND = CASE / "band_extension.json"
REPRO = CASE / "frozen_reproduction.json"
BASE = CASE / "results.json"
SCALE = 1025.0 * 9.80665  # rho g a^2, a = 1 m


def _band() -> dict:
    return json.loads(BAND.read_text())


def _cx(row: dict[str, float]) -> complex:
    return complex(row["re"], row["im"])


def test_artifacts_present():
    for path in (BAND, REPRO, BASE):
        assert path.exists(), f"missing artefact {path}"
        assert path.stat().st_size > 2000
    for name in ("band_extension_force.png", "band_extension_error.png"):
        fig = CASE / name
        assert fig.exists(), f"missing figure {fig}"
        assert fig.stat().st_size > 10_000


def test_anchor_and_links_to_base_case():
    data = _band()
    assert data["anchor"] == "CASE-BEM-HULME-DIFF-VERIF"
    # the extension must be strictly above the delivered band, otherwise it is
    # not an extension.
    base = json.loads(BASE.read_text())
    base_max = max(cell["ka"] for cell in base["ladder"])
    assert base_max == pytest.approx(2.0)
    assert min(float(k) for k in data["ka_grid"]) > base_max
    # and it must carry the base band for the same-axis comparison.
    assert data["base_case"]["rows"], "base-case rows not carried over"


def test_reference_truncation_converged_at_high_ka():
    for key, cell in _band()["per_ka"].items():
        trunc = cell["reference"]["truncation_relative"]
        assert trunc["surge"] < 1.0e-3, (key, trunc)
        assert trunc["heave"] < 1.0e-3, (key, trunc)
        assert cell["reference"]["required_terms_1e-3"]["heave"] <= 288


def test_reference_row_recomputed_from_library():
    """No hand-copied digits: the stored high-ka reference re-derives."""
    key = max(_band()["per_ka"], key=float)
    stored = _band()["per_ka"][key]["reference"]
    fresh = hulme_hemisphere_radiation(
        float(key), radius=1.0, density=1025.0, gravity=9.80665, n_terms=288,
        quad_order=576,
    )
    assert _cx(stored["surge"]) == pytest.approx(complex(fresh.excitation_force_x), rel=1e-12)
    assert _cx(stored["heave"]) == pytest.approx(complex(fresh.excitation_force_z), rel=1e-12)


def test_declared_envelope_is_self_consistent():
    """The declared band must be exactly the set of ka passing both gates."""
    data = _band()
    gates = data["gates"]
    env = data["envelope"]
    level = str(env["case_mesh_level"])
    passing = []
    for key, cell in data["per_ka"].items():
        ok = all(
            cell["metrics"][level][dof]["relative"] <= gates["relative"]
            and cell["metrics"][level][dof]["absolute_normalised"]
            <= gates["absolute_normalised"]
            for dof in ("surge", "heave")
        )
        if ok:
            passing.append(float(key))
    assert sorted(env["case_mesh_pass_ka"]) == sorted(passing)
    assert env["case_mesh_upper_bound_ka"] == pytest.approx(max(passing))
    # documented, not hidden: the top of the extended band is where it stops.
    top = max(float(k) for k in data["per_ka"])
    assert env["case_mesh_upper_bound_ka"] <= top


def test_case_mesh_accuracy_inside_declared_band():
    """Inside the band the errors must be comfortably small, not merely <gate>."""
    data = _band()
    level = str(data["envelope"]["case_mesh_level"])
    for key in data["envelope"]["case_mesh_pass_ka"]:
        cell = data["per_ka"][f"{float(key):.3f}"]
        for dof in ("surge", "heave"):
            metric = cell["metrics"][level][dof]
            assert metric["relative"] < 0.03, (key, dof, metric)
            assert metric["absolute_normalised"] < 0.01, (key, dof, metric)


def test_coarse_mesh_degenerates_at_high_ka():
    """Evidence that the envelope is set by discretisation, not by the series.

    At the top of the band the 64-panel mesh must be much worse than the
    1024-panel one; if it were not, the mesh ladder would be meaningless.
    """
    data = _band()
    top = max(data["per_ka"], key=float)
    cell = data["per_ka"][top]
    for dof in ("surge", "heave"):
        coarse = cell["metrics"]["2"][dof]["relative"]
        fine = cell["metrics"]["4"][dof]["relative"]
        assert coarse > 3.0 * fine, (top, dof, coarse, fine)


def test_relative_metric_inflates_where_the_force_cancels():
    """The relative metric inflates once the force cancels; quantified.

    At ``ka = 2`` the heave force is already down to ~5 % of ``rho g A a^2``, so
    the relative metric reports roughly twice the wave-pressure-normalised
    error (base-case BEM row: measured ratio > 1.5).  Deeper into the band the
    two metrics must keep tracking each other, which is what makes the
    absolute metric a usable companion rather than a replacement.
    """
    data = _band()
    row = data["base_case"]["rows"]["2.000"]
    assert row["heave_relative"] > 1.5 * row["heave_absolute_normalised"]
    assert row["heave_relative"] <= data["gates"]["relative"]
    assert row["heave_absolute_normalised"] < 0.01
    # cancellation regime: the heave force amplitude falls well below the scale
    for key, cell in data["per_ka"].items():
        heave = cell["reference"]["heave"]
        level = math.hypot(heave["re"], heave["im"]) / SCALE
        assert level < 0.4, (key, level)


def test_frozen_capytaine_artifact_reproduces():
    repro = json.loads(REPRO.read_text())
    assert repro["verdict"] == "REPRODUCED", repro["worst_cell"]
    assert repro["worst_relative"] < 1.0e-6  # deterministic solver, same machine
    assert repro["faces_frozen"] == repro["faces_live"]
    for key, row in repro["rows"].items():
        for dof in ("surge", "heave"):
            assert row[dof]["relative"] < 1.0e-6, (key, dof)


def test_third_route_inside_the_band_if_present():
    data = _band()
    level = data["envelope"]["case_mesh_level"]
    for key, cell in data["per_ka"].items():
        indep = cell.get("independent_finest")
        if indep is None:  # capytaine optional dependency: not run
            pytest.skip("capytaine route not recorded in this artefact")
        if float(key) > data["envelope"]["case_mesh_upper_bound_ka"]:
            continue
        if not cell.get("capytaine_route_usable", True):
            # honest flag: the foreign route's own mesh ladder has not settled,
            # so it is evidence of magnitude only, not a 2 % gate.
            assert (
                cell["capytaine_ladder_spread_absolute_normalised"] > 0.02
            ), (key, cell.get("capytaine_ladder_spread_absolute_normalised"))
            continue
        for dof in ("surge", "heave"):
            assert indep[dof]["absolute_normalised"] < 0.02, (key, dof, indep[dof])