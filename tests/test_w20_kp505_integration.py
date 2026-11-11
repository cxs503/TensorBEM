"""Integration caller for the W20 item-5 KP505 GCI driver (m14569 rework).

The maturity gate's L3 criterion requires a *real* call from the test
suite / platform side into the benchmark implementation, not a pointer
at a file.  These tests import the driver through the public package
path (``tensorbem.benchmarks.kp505_w20_gci``), execute its entry
functions against the main solver (``tensorbem.propeller.
open_water_point``), and cross-check the live results against the
committed artefact ``benchmarks/verified/bem_kp505_openwater_w20/
gci.json`` so any silent drift between code and artefact turns the
suite red.

Pytest node ids (stable, for the maturity gate's caller grep re-check)::

    tests/test_w20_kp505_integration.py::test_driver_is_exported_and_wired_to_the_main_solver
    tests/test_w20_kp505_integration.py::test_live_l1_evaluation_matches_the_artefact
    tests/test_w20_kp505_integration.py::test_decompose_closes_and_matches_the_artefact
    tests/test_w20_kp505_integration.py::test_anchor_values_match_the_transcription
    tests/test_w20_kp505_integration.py::test_entry_command_is_declared_and_reproducible

The live evaluation is the coarsest ladder level (16 panels/blade,
~0.5 s on loongarch64), so the suite stays fast; the full three-density
ladder is exercised by the driver itself (``python -m
tensorbem.benchmarks.kp505_w20_gci``) and by the cached raw outputs in
``runs/w20_kp505_gci_cache``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GCI_REL = "benchmarks/verified/bem_kp505_openwater_w20/gci.json"
TRANSCRIPTION_REL = "papers/data/w20-anchors/kp505_transcription.json"


@pytest.fixture(scope="module")
def gci() -> dict:
    return json.loads((REPO_ROOT / GCI_REL).read_text())


@pytest.fixture(scope="module")
def transcription() -> dict:
    return json.loads((REPO_ROOT / TRANSCRIPTION_REL).read_text())


def _close(a: float, b: float, rel: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-12)


def test_driver_is_exported_and_wired_to_the_main_solver() -> None:
    """The driver is importable via the package and calls the solver."""
    import tensorbem.benchmarks as bench
    from tensorbem.benchmarks import kp505_w20_gci as driver
    from tensorbem.propeller import open_water_point

    assert bench.kp505_w20_gci is driver
    # the driver's lattice evaluation is built on the main solver entry
    assert driver.open_water_point is open_water_point
    assert driver.REPO_ROOT == REPO_ROOT


def test_live_l1_evaluation_matches_the_artefact(gci: dict) -> None:
    """A live solver call at L1 reproduces the committed L1 values."""
    from tensorbem.benchmarks.kp505_w20_gci import PANEL_LEVELS, evaluate

    live = evaluate(0.7, "fixed", PANEL_LEVELS[0])
    cell = next(
        c for c in gci["cells"] if c["J"] == 0.7 and c["core_policy"] == "fixed"
    )
    ref = cell["levels"][0]
    for key in ("KT", "10KQ", "eta0"):
        assert _close(live[key], ref[key]), (key, live[key], ref[key])
    assert live["n_panels_per_blade"] == ref["n_panels_per_blade"] == 16


def test_decompose_closes_and_matches_the_artefact(gci: dict) -> None:
    """The decomposition identity holds live and matches the artefact."""
    from tensorbem.benchmarks.kp505_w20_gci import decompose

    cell = next(
        c for c in gci["cells"] if c["J"] == 0.5 and c["core_policy"] == "fixed"
    )
    ref = cell["quantities"]["KT"]["decomposition"]
    live = decompose(ref["f_fine"], ref["f_extrapolated"], ref["f_anchor"])
    assert live["closure_residual_pct_points"] < 1e-9
    for key in ("d_total_pct", "d_discretisation_pct", "d_model_pct"):
        assert _close(live[key], ref[key]), (key, live[key], ref[key])


def test_anchor_values_match_the_transcription(gci: dict, transcription: dict) -> None:
    """anchor_values() reads the same numbers the transcription registers."""
    from tensorbem.benchmarks.kp505_w20_gci import anchor_values

    for j in (0.5, 0.7, 0.9):
        live = anchor_values(j)
        point = next(p for p in transcription["anchor_points"] if p["J"] == j)
        assert _close(live["KT"], point["KT"])
        assert _close(live["10KQ"], point["10KQ"])
    # and the artefact's anchor block agrees with both
    for j in (0.5, 0.7, 0.9):
        assert _close(gci["anchor"]["values"][str(j)]["KT"], anchor_values(j)["KT"])


def test_entry_command_is_declared_and_reproducible(gci: dict) -> None:
    """The declared regenerate command names the real entry module."""
    assert gci["regenerate"] == "python -m tensorbem.benchmarks.kp505_w20_gci"
    assert gci["generated_by"] == "tensorbem.benchmarks.kp505_w20_gci"
    assert (REPO_ROOT / "src/tensorbem/benchmarks/kp505_w20_gci.py").is_file()
