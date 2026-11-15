"""The frozen capytaine reference must cover the whole pre-registered grid.

Anchor: ``XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN``.

The cross-validation is a two-sided study: a TensorBEM panel ladder on one side
and a capytaine panel ladder on the other.  Until this tick the frozen artifact
held **one** reference cell out of twelve, so no amount of TensorBEM laddering
could ever produce a study-level verdict -- every gate stayed unjudgeable.  The
reference is now complete; these assertions pin that down so that a partial
shard merge (which the producer script refuses, but a hand-edited artifact would
not) cannot silently shrink the study back to a subset.
"""

from __future__ import annotations

import json
from pathlib import Path

from tensorbem.benchmarks import finite_depth_crossval_plan as plan

REFERENCE = Path(
    "benchmarks/verified/capytaine_finite_depth/capytaine_ref.json"
)


def _reference() -> dict:
    return json.loads(REFERENCE.read_text())


def test_reference_covers_every_planned_cell_exactly_once():
    reference = _reference()
    planned = [plan.cell_id(cell["h_over_a"], cell["ka"]) for cell in plan.cells()]
    covered = [cell["cell"] for cell in reference["cells"]]
    assert covered == planned, (
        "the frozen reference does not cover the planned grid: "
        f"missing {sorted(set(planned) - set(covered))}, "
        f"extra {sorted(set(covered) - set(planned))}"
    )


def test_reference_carries_the_full_resolution_ladder_per_cell():
    reference = _reference()
    wanted = [list(res) for res in plan.CAPYTAINE_RESOLUTIONS]
    legacy = list(plan.CAPYTAINE_LEGACY_RESOLUTION)
    for cell in reference["cells"]:
        got = [list(level["resolution"]) for level in cell["levels"]]
        assert got == wanted, f"{cell['cell']}: resolution ladder {got} != {wanted}"
        for level in cell["levels"]:
            values = level["values"]
            assert set(values) == {"surge", "heave"}
            for dof in values.values():
                assert dof["added_mass"] > 0.0 and dof["damping"] >= 0.0
        # the legacy Green-function formula is the in-code second opinion; it is
        # carried per cell so a cell cannot be merged without it.
        assert list(cell["legacy"]["resolution"]) == legacy
        assert cell["legacy"]["values"]["heave"]["added_mass"] > 0.0


def test_reference_spans_the_planned_depth_range_in_the_solved_cells():
    reference = _reference()
    k0h = sorted(cell["k0h"] for cell in reference["cells"])
    assert len(k0h) == len(list(plan.cells()))
    assert k0h[0] == min(k0h) and k0h[0] <= 1.5 + 1e-12  # deepest water is probed
    assert k0h[-1] >= 39.9  # and effectively-deep water is probed too
    # the per-cell dispersion relation k0 tanh(k0 h) = nu must be inverted, not
    # guessed: each cell states the nu it was solved at.
    for cell in reference["cells"]:
        import math

        resid = cell["k0"] * math.tanh(cell["k0"] * cell["depth"]) - cell["nu"]
        assert abs(resid) <= 1e-9 * max(1.0, abs(cell["nu"]))


def test_reference_provenance_names_the_producer_and_the_formula():
    provenance = _reference()["provenance"]
    assert provenance["producer"] == "scripts/capytaine_finite_depth_ref.py"
    assert provenance["plan_module"].endswith("finite_depth_crossval_plan")
    assert provenance["tool"] == "capytaine"
    assert provenance["water_depth"].startswith("finite")
    assert provenance["density"] == plan.DENSITY
    assert provenance["gravity"] == plan.GRAVITY