"""Artifact tests for the hemisphere rotational-null extension.

The study itself solves BEM systems (seconds to minutes per cell), so these
tests do not re-solve: they read the frozen artefacts and assert that the
measured evidence still supports the claim, and that the pre-registered gates
have not been quietly relaxed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorbem.benchmarks import hemisphere_rotational_null as mod

CASE_DIR = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "verified"
    / "casepool_hulme_rotational_null"
)
RESULTS = CASE_DIR / "results.json"
CAPYTAINE = CASE_DIR / "capytaine_rotational_null.json"


@pytest.fixture(scope="module")
def results() -> dict:
    if not RESULTS.exists():
        pytest.skip(f"missing artefact {RESULTS}")
    return json.loads(RESULTS.read_text())


@pytest.fixture(scope="module")
def capytaine() -> dict:
    if not CAPYTAINE.exists():
        pytest.skip(f"missing artefact {CAPYTAINE}")
    return json.loads(CAPYTAINE.read_text())


def _finest_cells(results: dict) -> list[dict]:
    finest = max(c["subdivisions"] for c in results["cells"])
    cells = [c for c in results["cells"] if c["subdivisions"] == finest]
    return cells


def test_pre_registered_gates_are_pinned() -> None:
    """Gates are part of the claim: changing them must be a deliberate edit."""
    assert mod.SELF_GATE == pytest.approx(1.0e-4)
    assert mod.BLOCK_GATE == pytest.approx(5.0e-3)
    assert mod.ORDER_MIN == pytest.approx(1.0)
    assert mod.SUPERPOSITION_GATE == pytest.approx(1.0e-3)
    assert mod.PIN_GATE == pytest.approx(1.0e-9)
    assert mod.SYMMETRY_GATE == pytest.approx(1.0e-6)
    assert mod.BAND_KA == (0.2, 0.5, 0.8, 1.0, 1.5, 2.0)
    assert mod.LADDER_SUBDIVISIONS == (2, 3, 4)


def test_band_and_ladder_coverage(results: dict) -> None:
    covered = {(c["subdivisions"], c["ka"]) for c in results["cells"]}
    levels = sorted({s for s, _ in covered})
    assert levels[0] == 2 and len(levels) >= 2
    for ka in mod.BAND_KA:
        assert (levels[0], ka) in covered
        assert (levels[-1], ka) in covered
    for cell in results["cells"]:
        assert cell["panels"] == 4 * 4 ** cell["subdivisions"]
        assert cell["relative_residual"] < 1.0e-8


def test_rotational_self_terms_are_null_on_finest_mesh(results: dict) -> None:
    """A44 = A55 = A66 = B44 = B55 = B66 = 0 analytically."""
    worst = max(cell["rotational_self_max"] for cell in _finest_cells(results))
    assert worst < mod.SELF_GATE, worst


def test_whole_rotational_block_is_null_on_finest_mesh(results: dict) -> None:
    """Reciprocity removes rows and columns 4..6, including A42 and A51."""
    worst = max(cell["rotational_null_max"] for cell in _finest_cells(results))
    assert worst < mod.BLOCK_GATE, worst


def test_translational_block_is_not_null(results: dict) -> None:
    """Guard against a silent all-zero solve: the null must be selective."""
    for cell in _finest_cells(results):
        a = cell["added_mass_dimless"]
        assert a[0][0] > 0.3 and a[2][2] > 0.3
        assert cell["damping_dimless"][2][2] > 0.01


def test_null_residual_shrinks_under_refinement(results: dict) -> None:
    by_level: dict[float, dict[int, float]] = {}
    for cell in results["cells"]:
        by_level.setdefault(cell["ka"], {})[cell["subdivisions"]] = cell[
            "rotational_null_max"
        ]
    orders = []
    for ka, per_level in by_level.items():
        levels = sorted(per_level)
        assert per_level[levels[-1]] < per_level[levels[0]], ka
        orders.append(
            math.log(per_level[levels[0]] / per_level[levels[-1]])
            / math.log(2.0 ** (levels[-1] - levels[0]))
        )
    assert min(orders) >= mod.ORDER_MIN


def test_offset_axis_superposition_identity(results: dict) -> None:
    probe = results["offset_probe"]
    assert probe["rows"], "offset probe missing"
    assert probe["rows"][0]["offset_d"] == 0.0
    dev = max(
        max(row["added_mass_rel_dev"], row["damping_rel_dev"]) for row in probe["rows"]
    )
    assert dev < mod.SUPERPOSITION_GATE, dev
    # the null really becomes the d^2 A22 term once the axis is moved
    for row in probe["rows"]:
        if row["offset_d"] > 0.0:
            assert isinstance(row["added_mass"], float)
            assert abs(row["added_mass"]) > 10.0


def test_single_dof_path_untouched(results: dict) -> None:
    """The 6-DOF solve reproduces the graded surge/sway/heave values."""
    for cell in results["cells"]:
        for key, value in cell["pin_single_dof"].items():
            assert value < mod.PIN_GATE, (cell["ka"], key, value)


def test_foreign_code_residual_falls(capytaine: dict) -> None:
    ladder = capytaine["mesh_ladder"]
    residuals = []
    for row in ladder:
        worst = 0.0
        for key, per_dof in row["values"].items():
            omega = math.sqrt(mod.GRAVITY * float(key))
            for values in per_dof.values():
                worst = max(
                    worst,
                    abs(values["added_mass"]) / (mod.DENSITY * mod.RADIUS**5),
                    abs(values["damping"]) / (mod.DENSITY * omega * mod.RADIUS**5),
                )
        residuals.append(worst)
    assert residuals[-1] < residuals[0]
    assert residuals[-1] < mod.BLOCK_GATE
    for probe in capytaine["rotation_centre_probe"].values():
        assert probe["relative_dev_full_superposition"] < mod.SUPERPOSITION_GATE


def test_evaluation_and_report_agree(results: dict) -> None:
    evaluation = results["evaluation"]
    assert set(evaluation["criteria"]) >= {
        "G1_rotational_block_null_below_gate",
        "G2_rotational_self_null_below_gate",
        "G5_offset_axis_superposition",
        "G6_graded_single_dof_untouched",
    }
    report = CASE_DIR / "report.md"
    assert report.exists()
    text = report.read_text()
    assert f"**Verdict: {evaluation['verdict']}**" in text
    for name in evaluation["criteria"]:
        assert name in text